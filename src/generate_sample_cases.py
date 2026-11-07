"""Generate sample-specific train/development evidence without reference labels."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache/numba'))
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from threadpoolctl import threadpool_limits

DEFAULT_H5AD = Path(os.environ.get('SCPCA_H5AD', str(ROOT / 'data/source/SCPCP000004_ann-data_2026-09-13/SCPCP000004_single-cell_merged/SCPCP000004_merged_rna.h5ad')))
PANEL = 'CHGA CHGB PHOX2B TH DBH CD3D CD3E TRAC LYZ AIF1 CD68 COL1A1 DCN ACTA2 PDGFRB CDH19 SOX10 PLP1 PECAM1 VWF HBB HBA1 S100A8 S100A9 S100B MPZ NGFR RGS5 MCAM PDGFRA LUM CD163 MRC1 CSF1R C1QA C1QB C1QC F13A1 MS4A1 CD79A NKG7 GNLY FCGR3B CSF3R'.split()
CONFIG = {
    'version': 'sample-cases-v1', 'expression_source': 'raw/X',
    'normalization': 'Per-cell total-count normalization to 10000, followed by natural log1p; freshly recomputed',
    'min_cells_per_gene': 3, 'hvg_flavor': 'seurat', 'n_hvg': 2000,
    'n_pcs_max': 50, 'n_neighbors_max': 20, 'leiden_resolution': .5,
    'leiden_flavor': 'igraph', 'leiden_iterations': 2, 'random_state': 0,
    'marker_method': 'wilcoxon', 'marker_reference': 'rest_of_same_sample',
    'marker_candidates': 50, 'markers_per_packet': 30, 'panel': PANEL,
    'small_cluster_flag_below': 25, 'small_sample_flag_below': 100,
    'scaling': 'No per-gene z-scoring; centered PCA on log-expression HVGs',
    'technical_libraries': 'Combine only libraries sharing the frozen biological sample ID; no across-sample integration',
    'test_access': 'Disallowed in this generator',
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def extract_counts(path, source_rows, block_genes=1024):
    """Select cell rows from CSC storage without constructing all-patient matrices."""
    source_rows = np.asarray(source_rows, dtype=np.int64)
    if len(np.unique(source_rows)) != len(source_rows):
        raise ValueError('Duplicate source rows')
    row_chunks, column_chunks, value_chunks = [], [], []
    with h5py.File(path, 'r') as f:
        g = f['raw/X']
        shape = tuple(int(x) for x in g.attrs['shape'])
        if np.any(source_rows < 0) or np.any(source_rows >= shape[0]):
            raise ValueError('Source row outside matrix')
        row_map = np.full(shape[0], -1, dtype=np.int32)
        row_map[source_rows] = np.arange(len(source_rows), dtype=np.int32)
        indptr = g['indptr'][()]
        for first in range(0, shape[1], block_genes):
            last = min(first + block_genes, shape[1])
            start, end = int(indptr[first]), int(indptr[last])
            rows = row_map[g['indices'][start:end]]
            values = g['data'][start:end]
            keep = (rows >= 0) & (values != 0)
            selected_values = values[keep]
            if not np.isfinite(selected_values).all() or (selected_values < 0).any():
                raise ValueError('Counts must be finite and nonnegative')
            if not np.allclose(selected_values, np.rint(selected_values), atol=1e-6, rtol=0):
                raise ValueError('Selected raw/X contains noninteger values; verify source semantics')
            columns = np.repeat(np.arange(first, last, dtype=np.int32), np.diff(indptr[first:last + 1]))
            row_chunks.append(rows[keep])
            column_chunks.append(columns[keep])
            value_chunks.append(selected_values.astype(np.float32))
    result = sparse.coo_matrix((np.concatenate(value_chunks),
                                (np.concatenate(row_chunks), np.concatenate(column_chunks))),
                               shape=(len(source_rows), shape[1])).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    return result


def validate_selected_cells(frame):
    if frame.empty:
        raise ValueError('No selected cells')
    if not frame.partition.isin(['train', 'development']).all():
        raise ValueError('Only train/development cells allowed; test access is prohibited')
    if not frame.analysis_eligible.eq(True).all():
        raise ValueError('Ineligible cells selected')


def panel_rows(logcounts, symbols, panel=PANEL):
    rows, missing = [], []
    for gene in panel:
        indices = np.flatnonzero(symbols == gene)
        if len(indices) != 1:
            missing.append({'gene': gene, 'reason': 'not_in_source_gene_metadata' if len(indices) == 0 else 'ambiguous_duplicate_gene_symbol'})
            continue
        vector = logcounts[:, int(indices[0])]
        rows.append({'gene': gene, 'mean_log_expression': round(float(vector.mean()), 4),
                     'fraction_expressing': round(float((vector > 0).mean()), 4)})
    return rows, missing


def read_gene_metadata(path):
    with h5py.File(path, 'r') as f:
        g = f['raw/var']
        key = g.attrs.get('_index', '_index')
        if isinstance(key, bytes):
            key = key.decode()
        ids = g[key].asstr()[()]
        node = g['gene_symbol']
        if isinstance(node, h5py.Dataset):
            symbols = node.asstr()[()]
        else:
            cats = node['categories'].asstr()[()]
            symbols = np.asarray([cats[c] if c >= 0 else '' for c in node['codes'][()]], dtype=object)
    return pd.DataFrame({'gene_symbol': symbols}, index=pd.Index(ids, name='gene_id'))


def generate_sample(counts, cells, genes, sample, partition, out, source_fingerprint):
    start = time.perf_counter()
    destination = out / sample
    destination.mkdir(exist_ok=False)
    counts = counts.tocsr()
    # Validate count-layer meaning against independently provided UMI totals.
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    if not np.allclose(row_sums, cells['sum'].to_numpy(), atol=.01, rtol=0):
        raise ValueError(f'{sample}: raw/X row sums do not match source UMI totals')
    full = ad.AnnData(X=counts.copy(), var=genes.copy())
    full.obs_names = cells.obs_name.to_numpy()
    sc.pp.normalize_total(full, target_sum=10000)
    sc.pp.log1p(full)
    symbols = full.var.gene_symbol.fillna('').astype(str).to_numpy()
    present = np.asarray((counts > 0).sum(axis=0)).ravel() >= CONFIG['min_cells_per_gene']
    a = full[:, present].copy()
    sc.pp.highly_variable_genes(a, flavor=CONFIG['hvg_flavor'], n_top_genes=min(CONFIG['n_hvg'], a.n_vars))
    n_hvg = int(a.var.highly_variable.sum())
    n_pcs = min(CONFIG['n_pcs_max'], a.n_obs - 1, n_hvg - 1)
    if n_pcs < 2:
        raise ValueError(f'{sample}: insufficient cells/variable genes for PCA')
    sc.pp.pca(a, n_comps=n_pcs, mask_var='highly_variable', svd_solver='arpack', random_state=0)
    sc.pp.neighbors(a, n_neighbors=min(CONFIG['n_neighbors_max'], a.n_obs - 1), use_rep='X_pca', random_state=0)
    sc.tl.leiden(a, resolution=CONFIG['leiden_resolution'], key_added='cluster',
                 flavor='igraph', directed=False, n_iterations=2, random_state=0)
    sc.tl.umap(a, random_state=0)
    if a.obs.cluster.nunique() < 2:
        raise ValueError(f'{sample}: only one cluster; within-sample contrast is unavailable')
    sc.tl.rank_genes_groups(a, groupby='cluster', method='wilcoxon', reference='rest',
                           use_raw=False, pts=True, n_genes=min(50, a.n_vars), key_added='markers')
    markers = sc.get.rank_genes_groups_df(a, group=None, key='markers', gene_symbols='gene_symbol')
    markers['gene'] = markers['gene_symbol'].astype('string').fillna(markers['names'].astype('string'))
    markers = markers.loc[markers.scores.gt(0)].copy()
    markers.to_csv(destination / 'ranked_markers.csv', index=False)
    membership = cells[['source_row', 'obs_name', 'library_id', 'sample_id', 'participant_id', 'partition']].copy()
    membership['cluster'] = a.obs.cluster.astype(str).to_numpy()
    membership['UMAP1'], membership['UMAP2'] = a.obsm['X_umap'].T
    membership.to_csv(destination / 'cell_membership.csv.gz', index=False)
    a.var.loc[a.var.highly_variable].to_csv(destination / 'highly_variable_genes.csv')
    summaries, packets = [], []
    for cluster in sorted(a.obs.cluster.unique(), key=int):
        mask = a.obs.cluster.eq(cluster).to_numpy()
        case_id = 'NB1-' + hashlib.sha256(f'{sample}|{cluster}'.encode()).hexdigest()[:12]
        ranked = markers.loc[markers.group.astype(str).eq(str(cluster))].head(30)
        panel, missing = panel_rows(full.X[mask], symbols)
        marker_list = []
        for _, row in ranked.iterrows():
            marker_list.append({'gene_id': row['names'], 'gene': row['gene'],
                                'score': round(float(row['scores']), 4),
                                'fraction_expressing': round(float(row['pct_nz_group']), 4),
                                'fraction_expressing_rest': round(float(row['pct_nz_reference']), 4)})
        case = {
            'case_id': case_id, 'tissue_context': 'Neuroblastoma cohort',
            'cell_count': int(mask.sum()), 'capture_unit': source_fingerprint['capture_unit'],
            'ranked_markers': marker_list, 'marker_panel': panel,
            'missing_panel_evidence': missing,
            'technical_summary': {
                'median_umi': float(np.median(cells.loc[mask, 'sum'])),
                'median_detected_genes': float(np.median(cells.loc[mask, 'detected'])),
            },
            'evidence_notes': [
                'Markers compare this cluster with the rest of the same biological sample.',
                'Panel means use natural-log1p expression after per-cell scaling to 10000 counts.',
                'Fractions range from 0 to 1. Zero expression differs from unavailable evidence.',
                'Cluster averages cannot establish single-cell coexpression, doublets, or malignant status.',
                'Wilcoxon scores rank markers within this analysis; they are not patient-level biological significance tests.',
            ],
        }
        json.dumps(case, allow_nan=False)
        (destination / f'{case_id}.json').write_text(json.dumps(case, indent=2, allow_nan=False))
        summaries.append({'case_id': case_id, 'sample_id': sample, 'partition': partition,
                          'cluster': str(cluster), 'cells': int(mask.sum()),
                          'small_cluster_flag': int(mask.sum()) < 25,
                          'small_sample_flag': len(cells) < 100,
                          'n_positive_markers': len(marker_list), 'n_panel_genes': len(panel)})
        packets.append(case)
    pd.DataFrame(summaries).to_csv(destination / 'case_manifest.csv', index=False)
    # Plot only measured unique panel genes; panel table itself retains measured zeros.
    available = [gene for gene in PANEL if (symbols == gene).sum() == 1]
    plot = full[:, [int(np.flatnonzero(symbols == gene)[0]) for gene in available]].copy()
    plot.var_names = available
    plot.obs['cluster'] = a.obs.cluster.to_numpy()
    dp = sc.pl.dotplot(plot, var_names=available, groupby='cluster', standard_scale='var',
                       figsize=(19, max(4, len(summaries) * .45)), return_fig=True, show=False)
    dp.savefig(destination / 'marker_panel.pdf')
    dp.savefig(destination / 'marker_panel.png', dpi=150)
    plt.close('all')
    sc.pl.umap(a, color='cluster', legend_loc='on data', show=False,
               title=f'{sample}: sample-specific clusters')
    plt.savefig(destination / 'cluster_umap.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    report = {'sample_id': sample, 'partition': partition, 'cells': len(cells),
              'clusters': len(summaries), 'n_hvg': n_hvg, 'n_pcs': n_pcs,
              'generation_seconds': time.perf_counter() - start,
              'count_layer_verified_against_umi_totals': True,
              'source_fingerprint': source_fingerprint,
              'config': CONFIG, 'case_ids': [p['case_id'] for p in packets]}
    (destination / 'sample_run.json').write_text(json.dumps(report, indent=2))
    print(f'{sample}: {len(cells):,} cells, {len(summaries)} clusters, {report["generation_seconds"]:.1f}s', flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad', type=Path, default=DEFAULT_H5AD)
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'outputs/patient-benchmark-v1')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/sample-cases-v1')
    parser.add_argument('--sample', help='Optional single training/development sample for a technical pilot')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.split_dir / 'split_protocol.json').read_text())
    assert digest(args.h5ad) == protocol['source_h5ad']['sha256'], 'Source matrix differs from frozen dataset'
    cell_path = args.split_dir / 'cell_manifest.csv.gz'
    assert digest(cell_path) == protocol['output_sha256']['cell_manifest.csv.gz']
    cells = pd.read_csv(cell_path, keep_default_na=False, low_memory=False)
    selected = cells.loc[cells.analysis_eligible.eq(True) & cells.partition.isin(['train', 'development'])].copy()
    if args.sample:
        selected = selected.loc[selected.sample_id.eq(args.sample)].copy()
    validate_selected_cells(selected)
    rows = selected.source_row.to_numpy()
    metadata = pd.read_csv(args.split_dir / 'sample_manifest.csv').set_index('sample_id')
    genes = read_gene_metadata(args.h5ad)
    print(f'Reading counts for {len(selected):,} train/development cells only; no test cases will be generated.', flush=True)
    counts = extract_counts(args.h5ad, rows)
    with h5py.File(args.h5ad, 'r') as f:
        source_names = f['obs/_index'].asstr()[()]
    assert np.array_equal(source_names[rows], selected.obs_name.to_numpy())
    assert counts.shape == (len(selected), len(genes))
    environment = {name: importlib.metadata.version(name) for name in ['scanpy', 'anndata', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'igraph', 'leidenalg']}
    all_summaries = []
    with threadpool_limits(limits=4):
        for sample in sorted(selected.sample_id.unique()):
            mask = selected.sample_id.eq(sample).to_numpy()
            sample_cells = selected.loc[mask].reset_index(drop=True)
            assert sample_cells.participant_id.nunique() == sample_cells.partition.nunique() == 1
            fingerprint = {'h5ad_sha256_from_frozen_split': protocol['source_h5ad']['sha256'],
                           'generator_sha256': digest(__file__),
                           'cell_manifest_sha256': digest(cell_path),
                           'capture_unit': metadata.loc[sample, 'capture_units']}
            all_summaries.extend(generate_sample(counts[mask], sample_cells, genes, sample,
                                                 sample_cells.partition.iloc[0], args.out, fingerprint))
            gc.collect()
    pd.DataFrame(all_summaries).to_csv(args.out / 'case_manifest.csv', index=False)
    (args.out / 'run_manifest.json').write_text(json.dumps({
        'config': CONFIG, 'environment': environment, 'python': platform.python_version(),
        'generator_sha256': digest(__file__), 'split_protocol_sha256': digest(args.split_dir / 'split_protocol.json'),
        'samples': sorted(selected.sample_id.unique()), 'cells': len(selected),
        'cases': len(all_summaries), 'test_cases_generated': 0,
        'recipe_status': 'Initial development recipe; freeze after technical QC, before test case generation',
    }, indent=2))
    print(f'Finished {len(all_summaries)} cases from {selected.sample_id.nunique()} samples.', flush=True)


if __name__ == '__main__':
    main()
