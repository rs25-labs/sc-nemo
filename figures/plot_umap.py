"""Redraw original sample UMAP coordinates with source-preserving expert labels."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/figures'
OUT.mkdir(parents=True,exist_ok=True)
SAMPLE='SCPCS000105'
COORDS=ROOT/f'outputs/reserved-test-cases-v1/{SAMPLE}/cell_membership.csv.gz'
MANIFEST=ROOT/f'outputs/reserved-test-cases-v1/{SAMPLE}/case_manifest.csv'
LEDGER=ROOT/'outputs/test-review-67-cases/independent-expert-ledger/test-independent-annotations.jsonl'

# Display only expert broad identities; avoid unvalidated subtype/malignancy claims.
# Label positions are presentation offsets. They do not change UMAP coordinates.
DISPLAY={
 0:('Neuroblastic /\nsympathoadrenal',(-3.2,25.7),'neuronal'),
 1:('Endothelial',(-7.2,-7.1),'endothelial'),
 2:('Endothelial',(1.0,7.2),'endothelial'),
 3:('Endothelial',(-7.5,6.4),'endothelial'),
 4:('Perivascular\nmural cells',(8.7,1.8),'mural'),
 5:('T cells',(1.1,14.3),'t'),
 6:('B cells',(8.6,15.2),'b'),
 7:('Neuroblastic /\nsympathoadrenal',(10.2,25.7),'neuronal'),
 8:('Myeloid',(11.0,11.7),'myeloid'),
 9:('Endothelial',(-3.0,.2),'endothelial'),
 10:('Fibroblasts',(11.0,-8.0),'fibroblast'),
 11:('Schwann-lineage /\nSchwannian stroma',(16.5,8.3),'schwann'),
 12:('Erythroid-like',(-6.2,10.5),'uncertain'),
 13:('T cells',(-4.5,14.8),'t'),
 14:('Neuroblastic-like\n(low RNA content)',(5.2,18.1),'uncertain'),
}
COLORS={'neuronal':'#7300BA','endothelial':'#0058D4','mural':'#00836A',
        't':'#138000','b':'#D1005D','myeloid':'#D56A00',
        'fibroblast':'#008FAE','schwann':'#963D00','uncertain':'#555555'}
SUPPRESSED_LABELS={2,9}  # Redundant labels in the contiguous endothelial region.


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cells=pd.read_csv(COORDS)
    cases=list(csv.DictReader(MANIFEST.open()))
    refs={int(r['cluster']):r for r in map(json.loads,LEDGER.read_text().splitlines()) if r['sample_id']==SAMPLE}
    assert len(cells)==3115 and cells.source_row.is_unique
    assert set(cells.cluster)==set(DISPLAY)==set(refs)==set(range(15))
    assert cells[['UMAP1','UMAP2']].apply(np.isfinite).all().all()
    assert {int(r['cluster']):int(r['cells']) for r in cases}==cells.groupby('cluster').size().to_dict()
    assert {int(r['cluster']):r['case_id'] for r in cases}=={k:r['case_id']for k,r in refs.items()}
    assert {k for k,r in refs.items()if r['requires_further_review']}=={12,14}
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(12,11),facecolor='white')
    ax=fig.add_axes([.045,.16,.91,.72])
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(-11,22);ax.set_ylim(-10.0,28.0)
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_xlabel('UMAP 1',fontsize=14,color='black',labelpad=12)
    ax.set_ylabel('UMAP 2',fontsize=14,color='black',labelpad=12)
    ax.spines[['top','right']].set_visible(False)
    for side in ['left','bottom']:
        ax.spines[side].set_color('black');ax.spines[side].set_linewidth(.8)
    fig.text(.5,.953,'Cell populations in a neuroblastoma sample',ha='center',fontsize=22,
             color='black',fontweight='bold')
    fig.text(.5,.917,'Expert annotations  •  3,115 cells  •  15 expression-based clusters',
             ha='center',fontsize=14,color='black')
    labels=[]
    for cluster in sorted(DISPLAY):
        label,position,group=DISPLAY[cluster]
        c=cells.loc[cells.cluster.eq(cluster)]
        # Real saved cell coordinates only: no jitter, smoothing, or reclustering.
        ax.scatter(c.UMAP1,c.UMAP2,s=5.0,alpha=1.0,c=COLORS[group],linewidths=0,
                   edgecolors='none',rasterized=False,zorder=2)
        if cluster in SUPPRESSED_LABELS:
            continue
        anchor=c[['UMAP1','UMAP2']].median().to_numpy()
        a=ax.annotate(label,xy=anchor,xytext=position,textcoords='data',
            ha='center',va='center',fontsize=14,fontweight='bold',
            color='black',
            linespacing=1.2,
            arrowprops={'arrowstyle':'-','color':'black','lw':.9,'shrinkA':6,'shrinkB':4,
                        'connectionstyle':'arc3,rad=0'},
            zorder=4)
        a.set_path_effects([pe.withStroke(linewidth=3.2,foreground='white')])
        labels.append(a)
    fig.text(.5,.098,'Each dot is a cell. The same cell identity can occur in several clusters.',
             ha='center',fontsize=12,color='black')
    fig.text(.5,.067,'Gray clusters have tentative identities and require further expert review.',
             ha='center',fontsize=12,color='black')
    fig.text(.5,.035,'Sample SCPCS000105  •  Original UMAP positions preserved  •  Expert labels are not model predictions',
             ha='center',fontsize=10.5,color='black')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # Check label text separately from leader lines to detect overlap and clipping.
    boxes=[]
    from matplotlib.text import Text
    for label in labels:
        b=Text.get_window_extent(label,renderer=renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height
        boxes.append(b)
    overlaps=[(i,j)for i,a in enumerate(boxes)for j,b in enumerate(boxes)if i<j and a.overlaps(b)]
    assert not overlaps,overlaps
    stem=OUT/'umap-SCPCS000105-expert-labels'
    for ext in ['pdf','svg']:
        fig.savefig(stem.with_suffix('.'+ext),facecolor='white')
    fig.savefig(stem.with_suffix('.png'),dpi=600,facecolor='white')
    fig.savefig(OUT/'umap-SCPCS000105-expert-labels-preview.png',dpi=180,facecolor='white')
    plt.close(fig)
    provenance=[]
    for cluster,(label,position,group)in DISPLAY.items():
        r=refs[cluster]
        provenance.append({'cluster':cluster,'case_id':r['case_id'],'cells':r['cell_count'],
            'expert_label_verbatim':r['broad_identity_verbatim'],'display_label':label.replace('\n',' '),
            'review_decision_verbatim':r['review_decision_verbatim'],'further_review_required':r['requires_further_review'],
            'display_group':group,'color':COLORS[group], 'source_workbook':r['source_workbook'],
            'label_displayed':cluster not in SUPPRESSED_LABELS,
            'source_sheet':r['source_sheet'],'source_cell':r['field_cells']['Broad cell identity']})
    with (OUT/'umap-SCPCS000105-label-provenance.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=list(provenance[0]));writer.writeheader();writer.writerows(provenance)
    (OUT/'umap-SCPCS000105-verification.json').write_text(json.dumps({
        'sample':SAMPLE,'cells':3115,'clusters':15,'expert_labels':15,'displayed_labels':13,
        'suppressed_redundant_labels':sorted(SUPPRESSED_LABELS),'review_required_clusters':[12,14],
        'coordinates_sha256':sha(COORDS),'case_manifest_sha256':sha(MANIFEST),'expert_ledger_sha256':sha(LEDGER),
        'coordinates_modified':False,'reclustering_performed':False,'label_text_overlap_count':len(overlaps),
        'colors_encode':'Expert broad identities; review-required cases in gray',
        'label_source':'Independent Expert reviewer A annotations, not model predictions',
        'subtype_policy':'Resolved T cytotoxic annotation shortened to broad T cells; no subtype or malignancy claim.',
        'png_dpi':600,'png_pixels':[7200,6600],'vector_formats':['pdf','svg'],
    },indent=2))
    print('Exported all 3,115 original cells / 15 clusters, with 13 displayed labels and two redundant labels suppressed.')


if __name__=='__main__':main()
