"""Generate all reserved-donor evidence using the unchanged validated sample recipe.

Explicitly authorized test-data access; no annotation columns or model inference.
The training/development generator and its test-access guard remain unchanged.
"""
import gc
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import generate_sample_cases as recipe

pd, np, h5py = recipe.pd, recipe.np, recipe.h5py
ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / 'outputs/patient-benchmark-v1'
OUT = ROOT / 'outputs/reserved-test-cases-v1'


def main():
    protocol = json.loads((SPLIT / 'split_protocol.json').read_text())
    frozen = json.loads((ROOT / 'outputs/sample-cases-v1/case_recipe_v1.json').read_text())
    assert recipe.digest(recipe.__file__) == frozen['generator_sha256']
    assert recipe.CONFIG == frozen['config'], 'Do not change the validated recipe'
    environment = {k: importlib.metadata.version(k) for k in frozen['environment']}
    assert environment == frozen['environment'], 'Environment differs from original case generation'
    cell_path = SPLIT / 'cell_manifest.csv.gz'
    assert recipe.digest(cell_path) == protocol['output_sha256']['cell_manifest.csv.gz']
    assert recipe.digest(recipe.DEFAULT_H5AD) == protocol['source_h5ad']['sha256']
    cells = pd.read_csv(cell_path, keep_default_na=False, low_memory=False)
    selected = cells.loc[cells.analysis_eligible.eq(True) & cells.partition.eq('test')].copy()
    donors = {k for k, v in protocol['assignment'].items() if v == 'test'}
    assert set(selected.participant_id) == donors and len(donors) == 4
    assert len(selected) == 42927 and selected.sample_id.nunique() == 6
    assert selected.source_row.is_unique and selected.obs_name.is_unique
    assert set(selected.source_row).isdisjoint(cells.loc[cells.partition.isin(['train', 'development']), 'source_row'])
    metadata = pd.read_csv(SPLIT / 'sample_manifest.csv').set_index('sample_id')
    genes = recipe.read_gene_metadata(recipe.DEFAULT_H5AD)
    print(f'Extracting {len(selected):,} cells from 4 reserved donors / 6 samples.', flush=True)
    rows = selected.source_row.to_numpy()
    counts = recipe.extract_counts(recipe.DEFAULT_H5AD, rows)
    with h5py.File(recipe.DEFAULT_H5AD, 'r') as handle:
        assert np.array_equal(handle['obs/_index'].asstr()[()][rows], selected.obs_name.to_numpy())
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'generation-start.json').write_text(json.dumps({
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User requested all cases from all four reserved donors, for later reviewer selection.',
        'recipe_sha256': recipe.digest(recipe.__file__), 'wrapper_sha256': recipe.digest(__file__),
        'config': recipe.CONFIG, 'test_access_override': 'Authorized in this wrapper only',
        'reference_annotations_read': False, 'model_inference_run': False,
    }, indent=2))
    summaries = []
    with recipe.threadpool_limits(limits=4):
        for sample in sorted(selected.sample_id.unique()):
            mask = selected.sample_id.eq(sample).to_numpy()
            frame = selected.loc[mask].reset_index(drop=True)
            assert frame.participant_id.nunique() == 1 and frame.partition.eq('test').all()
            assert len(frame) == metadata.loc[sample, 'eligible_cells']
            fingerprint = {
                'h5ad_sha256_from_frozen_split': protocol['source_h5ad']['sha256'],
                'generator_sha256': recipe.digest(recipe.__file__),
                'cell_manifest_sha256': recipe.digest(cell_path),
                'capture_unit': metadata.loc[sample, 'capture_units'],
                'test_generation_authorized': True,
            }
            result = recipe.generate_sample(counts[mask], frame, genes, sample, 'test', OUT, fingerprint)
            for item in result:
                item['participant_id'] = frame.participant_id.iloc[0]
            summaries.extend(result)
            gc.collect()
    manifest = pd.DataFrame(summaries)
    assert manifest.case_id.is_unique and manifest.cells.sum() == len(selected)
    old_ids = set(pd.read_csv(ROOT / 'outputs/sample-cases-v1/case_manifest.csv').case_id)
    assert old_ids.isdisjoint(manifest.case_id)
    memberships = pd.concat([pd.read_csv(OUT / sample / 'cell_membership.csv.gz')
                             for sample in sorted(selected.sample_id.unique())], ignore_index=True)
    assert memberships.source_row.is_unique
    assert set(memberships.source_row) == set(selected.source_row)
    assert memberships.partition.eq('test').all()
    packets = [json.loads((OUT / r.sample_id / f'{r.case_id}.json').read_text())
               for r in manifest.itertuples()]
    for p, r in zip(packets, manifest.itertuples()):
        assert p['case_id'] == r.case_id and p['cell_count'] == r.cells
        assert not {'participant_id', 'sample_id', 'partition'}.intersection(p)
        json.dumps(p, allow_nan=False)
    manifest.to_csv(OUT / 'case_manifest.csv', index=False)
    (OUT / 'case-evidence.jsonl').write_text(''.join(json.dumps(p, allow_nan=False) + '\n' for p in packets))
    donor_summary = manifest.groupby('participant_id').agg(samples=('sample_id', 'nunique'),
        cases=('case_id', 'size'), cells=('cells', 'sum'), small_clusters=('small_cluster_flag', 'sum'))
    donor_summary.to_csv(OUT / 'donor_summary.csv')
    manifest.groupby(['participant_id', 'sample_id']).agg(cases=('case_id', 'size'),
        cells=('cells', 'sum'), small_clusters=('small_cluster_flag', 'sum')).to_csv(OUT / 'sample_summary.csv')
    record = {
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'donors': sorted(donors),
        'samples': sorted(selected.sample_id.unique()), 'cells': len(selected), 'cases': len(packets),
        'environment': environment, 'python': platform.python_version(), 'config': recipe.CONFIG,
        'test_access_override': 'Explicit user authorization; original guard remains unchanged',
        'generator_sha256': recipe.digest(recipe.__file__), 'wrapper_sha256': recipe.digest(__file__),
        'split_protocol_sha256': recipe.digest(SPLIT / 'split_protocol.json'),
        'reference_annotations_read': False, 'model_inference_run': False,
        'all_eligible_test_cells_reconciled': True, 'overlap_with_train_development': 0,
        'recipe_and_environment_match_original': True,
    }
    (OUT / 'verification.json').write_text(json.dumps(record, indent=2))
    table = '\n'.join(f'| {donor} | {int(r.samples)} | {int(r.cases)} | {int(r.cells):,} | {int(r.small_clusters)} |'
                      for donor, r in donor_summary.iterrows())
    (OUT / 'README.md').write_text(f'''# Reserved-donor test evidence cases

Generated {len(packets)} cases from all 42,927 eligible cells across six samples and four reserved donors.
This is evidence generation only: no model inference or expert labeling has been run.

| Donor | Samples | Cases | Cells | Clusters below 25 cells |
|---|---:|---:|---:|---:|
{table}

The original training/development generator and numerical settings were reused unchanged.
Within each biological sample, counts were normalized, clustered, and analyzed for marker genes independently.
Same-sample technical libraries were combined. All cells reconcile to the frozen test partition; none overlap training/development.
No public reference annotations were read. Coordinator donor/sample identifiers are excluded from model evidence packets.

## Contents

- `case_manifest.csv`: case inventory with coordinator identifiers, cell counts, and small-cluster flags.
- `donor_summary.csv` and `sample_summary.csv`: workload totals.
- `case-evidence.jsonl`: all model-input evidence packets, without answers.
- Each sample folder: individual evidence JSONs, ranked markers, marker-panel PDF/PNG, UMAP, exact cell membership, and provenance.
- `verification.json`: source, recipe, environment, and membership integrity checks.

Small clusters remain included and flagged. Marker-panel colors are scaled separately per gene; dot size is the fraction expressing the gene.
Zero expression is distinguished from unavailable evidence. These checks do not establish biological correctness.

Reviewer selection is pending. Before model evaluation, lock the evaluation procedure and obtain blinded expert annotations.
Earlier investigator exposure to the whole-cohort pilot remains a limitation; these donors are held out from subsequent adaptation, not investigator-unseen.
The earlier frozen development results remain unchanged.
''')
    print(donor_summary.to_string(), flush=True)
    print(f'COMPLETE: {len(packets)} cases; all {len(selected):,} cells reconciled.', flush=True)


if __name__ == '__main__':
    main()
