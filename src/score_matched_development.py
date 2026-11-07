"""Audit returned paired development predictions and score explicit broad-lineage agreement.

No training or model calls. The post-run mapping is an analyst-defined interpretation,
not a preregistered ontology or an independent expert validation of model rationales.
"""
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from jsonschema import validate
from sklearn.metrics import f1_score
from matched_evaluation import OUTPUT_SCHEMA, canonical_hash, case_schedule

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/matched-development-analysis-v1'
GROUPS = ['neuronal_neuroendocrine', 'myeloid', 'schwann_glial',
          't_lymphoid', 'stromal_mesenchymal', 'endothelial']
LABEL_MAP = {
    'neuronal': GROUPS[0], 'neuroblastoma': GROUPS[0],
    'neuroblastic (sympathoadrenal-lineage) tumor-like cells': GROUPS[0],
    'tentative: malignant neuroblastic (sympathoadrenal-lineage) tumor-like cells': GROUPS[0],
    'myeloid cells (macrophages)': GROUPS[1],
    'myeloid cells (macrophage/monocyte)': GROUPS[1],
    'tentative: macrophages (myeloid lineage, non-specific evidence)': GROUPS[1],
    'tentative: macrophages (myeloid lineage, phagocytic state, likely tumor-associated)': GROUPS[1],
    'schwann cells': GROUPS[2],
    'schwann-lineage glial cells (schwannian stroma)': GROUPS[2],
    't cells': GROUPS[3], 't lymphocytes': GROUPS[3],
    'mesenchymal': GROUPS[4], 'stromal / mesenchymal cells': GROUPS[4],
    'endothelial cells': GROUPS[5],
    'tentative: t/nk cells': 'mixed_t_nk',
    'immune cell': 'overbroad_immune',
    'neural crest-derived': 'overbroad_neural_crest',
}


def normalize_label(label):
    return LABEL_MAP[' '.join(label.casefold().split())]


def arm_metrics(rows):
    resolved = [r for r in rows if r['resolved']]
    accepted = [r for r in rows if r['complete'] and not r['review']]
    accepted_resolved = [r for r in accepted if r['resolved']]
    positive = [r for r in rows if not r['resolved']]
    requested = [r for r in rows if r['complete'] and r['review']]
    tp = sum(not r['resolved'] for r in requested)
    yt = [r['reference_group'] for r in resolved]
    yp = [r['predicted_group'] if r['complete'] else 'invalid' for r in resolved]
    ya = [r['predicted_group'] if r['complete'] and not r['review'] else 'review' for r in resolved]
    return dict(
        cases=len(rows), completed=sum(r['complete'] for r in rows), resolved_cases=len(resolved),
        proposed_correct=sum(r['correct'] for r in resolved),
        proposed_accuracy=sum(r['correct'] for r in resolved)/len(resolved),
        proposed_macro_f1=float(f1_score(yt, yp, labels=GROUPS, average='macro', zero_division=0)),
        accepted_macro_f1=float(f1_score(yt, ya, labels=GROUPS, average='macro', zero_division=0)),
        explicit_review_requests=len(requested), format_failure_review_routes=sum(not r['complete'] for r in rows),
        review_true_positive=tp, review_false_positive=len(requested)-tp,
        review_false_negative=len(positive)-tp,
        review_precision=tp/len(requested) if requested else None,
        review_recall=tp/len(positive) if positive else None,
        accepted=len(accepted), automatic_coverage=len(accepted)/len(rows),
        accepted_resolved=len(accepted_resolved),
        accepted_resolved_accuracy=(sum(r['correct'] for r in accepted_resolved)/len(accepted_resolved)
                                    if accepted_resolved else None),
        accepted_wrong_resolved=sum(not r['correct'] for r in accepted_resolved),
        accepted_expert_uncertain=sum(not r['resolved'] for r in accepted),
        safe_accepted_yield=sum(r['correct'] for r in accepted_resolved)/len(rows),
        generation_seconds=sum(r['seconds'] for r in rows),
        median_seconds=float(np.median([r['seconds'] for r in rows])),
        p90_seconds=float(np.percentile([r['seconds'] for r in rows], 90)),
        total_output_tokens=sum(r['tokens'] for r in rows),
        peak_allocated_gib=max(r['peak_gib'] for r in rows))


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = dict(version='broad-lineage-v1', mapping=LABEL_MAP,
                   provenance='Post-run analyst mapping, applied identically to both arms; not preregistered or independently blinded. Saved before aggregate scoring.',
                   rules=['No malignancy or subtype validation from lineage agreement.',
                          'Generic immune/neural crest and mixed T/NK do not receive specific-lineage credit.',
                          'Six classes equally weighted for macro-F1; failures remain missed predictions.'])
    (OUT/'label-mapping-v1.json').write_text(json.dumps(mapping, indent=2))
    archive = ROOT/'outputs/nemotron_4b_matched_development_v1.zip'
    ledger = ROOT/'outputs/sample-baseline-review-20261101/independent-expert-ledger/development-independent-annotations.jsonl'
    refs = {r['case_id']: r for r in map(json.loads, ledger.read_text().splitlines())}
    baseline = {r['case_id']: r for r in map(json.loads, (ledger.parent.parent/'baseline-records-preserved.jsonl').read_text().splitlines()) if r['partition']=='development'}
    cases = {r['case_id']: r for r in csv.DictReader((ROOT/'outputs/sample-cases-v1/case_manifest.csv').open())}
    samples = {r['sample_id']: r for r in csv.DictReader((ROOT/'outputs/patient-benchmark-v1/sample_manifest.csv').open())}
    rows, originals = [], []
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names)==len(set(names))
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        manifest = json.loads(z.read('run_manifest.json'))
        assert manifest['schema']==OUTPUT_SCHEMA
        assert manifest['runner_source_sha256']==hashlib.sha256((ROOT/'src/matched_evaluation.py').read_bytes()).hexdigest()
        assert manifest['case_order']==[list(p) for p in case_schedule(refs)]
        assert set(manifest['case_input_hashes'])==set(refs)==set(baseline)
        tm = json.loads(z.read('training_metadata/run-manifest.json'))
        identity = {k:tm[k] for k in ['config','package_manifest','adapter_modules','software']}
        assert canonical_hash(identity)==tm['identity_hash']==manifest['training_identity']
        for filename, digest in tm['package_manifest']['files'].items():
            assert hashlib.sha256((ROOT/'outputs/expert-training-v1'/filename).read_bytes()).hexdigest()==digest
        logs = [json.loads(l) for l in z.read('training_metadata/training-log.jsonl').decode().splitlines()]
        assert [l['update'] for l in logs]==list(range(1,40))
        status = json.loads(z.read('training_metadata/training-status.json'))
        assert status==manifest['training_status'] and status['optimizer_updates']==39
        for arm in ['untuned','expert_lora']:
            arm_names = [n for n in names if n.startswith(arm+'/') and n.endswith('.json')]
            assert len(arm_names)==21
            assert {Path(n).stem for n in arm_names}==set(refs)
            for cid in sorted(refs):
                r=json.loads(z.read(f'{arm}/{cid}.json'));ref=refs[cid];old=baseline[cid]
                assert r['case_id']==cid and r['partition']=='development' and r['arm']==arm
                assert r['run_hash']==canonical_hash(manifest)
                assert canonical_hash(r['messages'])==r['input_sha256']==manifest['case_input_hashes'][cid]==old['input_sha256']
                assert r['messages']==old['messages'] and r['input_tokens']==old['input_tokens']
                assert r['output_tokens']==len(r['output_token_ids'])
                assert r['stopped_at_eos']==(r['output_token_ids'][-1]==11)
                assert r['annotation']==json.loads(r['raw_text'])
                validate(r['annotation'], manifest['schema'])
                a=r['annotation']; resolved=not ref['requires_further_review']
                predicted=normalize_label(a['broad_cell_type'])
                group=normalize_label(ref['broad_identity_verbatim']) if resolved else None
                complete=r['status']=='complete' and r['stopped_at_eos']
                packet=ref['evidence']; genes=a['supporting_genes']
                evidence={g['gene'] for g in packet['ranked_markers']+packet['marker_panel'] if g.get('gene')}
                zero={g['gene'] for g in packet['marker_panel'] if g['fraction_expressing']==0 and g['mean_log_expression']==0}
                sample=cases[cid]['sample_id']
                rows.append(dict(case_id=cid,arm=arm,sample_id=sample,donor=samples[sample]['participant_id'],
                    cells=int(cases[cid]['cells']),expert_label=ref['broad_identity_verbatim'],resolved=resolved,
                    reference_group=group,model_label=a['broad_cell_type'],predicted_group=predicted,
                    complete=complete,status=r['status'],correct=bool(resolved and complete and group==predicted),
                    review=a['needs_expert_review'],expert_review=not resolved,
                    seconds=r['generation_seconds'],tokens=r['output_tokens'],peak_gib=r['peak_gpu_allocated_gib'],
                    support_entries=len(genes),supplied_support_entries=sum(g in evidence for g in genes),
                    absent_support_genes=';'.join(g for g in genes if g not in evidence),
                    explicit_zero_support_genes=';'.join(g for g in genes if g in zero),
                    duplicate_support_entries=len(genes)-len(set(genes)),
                    explanation_words=len(a['explanation'].split())))
                originals.append(r)
        for n in names:
            if n.startswith('training_metadata/') or n.startswith('warmup_history/'):
                dest=OUT/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
        audit=dict(zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            canonical_records=len(rows),attempt_history_records=sum(n.startswith('attempt_history/') for n in names),
            exact_source_prompts_and_hashes_verified=True,runner_source_hash_verified=True,
            eos_token_lengths_and_schema_verified=True,training_package_hashes_and_39_updates_verified=True,
            adapter_weights_independently_verified=False,
            note='Adapter hashes are recorded by the runner; weights are not in this ZIP. Full model tokens can be decoded independently using verify_frozen_test_results.py for frozen test outputs.')
    metrics={arm:arm_metrics([r for r in rows if r['arm']==arm]) for arm in ['untuned','expert_lora']}
    for arm,m in metrics.items():
        rr=[r for r in rows if r['arm']==arm]
        m['supplied_support_entries']=sum(r['supplied_support_entries'] for r in rr)
        m['support_entries']=sum(r['support_entries'] for r in rr)
        m['cases_with_explicit_zero_support']=sum(bool(r['explicit_zero_support_genes']) for r in rr)
        m['duplicate_support_entries']=sum(r['duplicate_support_entries'] for r in rr)
        m['word_limit_violations']=sum(r['explanation_words']>100 for r in rr)
    donor_rows=[]
    for donor in sorted({r['donor'] for r in rows}):
        for arm in metrics:
            rr=[r for r in rows if r['donor']==donor and r['arm']==arm]
            resolved=[r for r in rr if r['resolved']]
            donor_rows.append(dict(donor=donor,arm=arm,cases=len(rr),resolved=len(resolved),
                correct=sum(r['correct'] for r in resolved),accuracy=sum(r['correct'] for r in resolved)/len(resolved) if resolved else None,
                uncertain=sum(not r['resolved'] for r in rr),review_requests=sum(r['review'] for r in rr)))
    class_rows=[]
    for arm in metrics:
        for g in GROUPS:
            rr=[r for r in rows if r['arm']==arm and r['resolved'] and r['reference_group']==g]
            class_rows.append(dict(arm=arm,group=g,support=len(rr),correct=sum(r['correct'] for r in rr)))
    write_csv(OUT/'case-comparison.csv',rows);write_csv(OUT/'donor-comparison.csv',donor_rows)
    write_csv(OUT/'class-comparison.csv',class_rows)
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    (OUT/'predictions-preserved.jsonl').write_text('\n'.join(json.dumps(r) for r in originals)+'\n')
    make_figure(rows,metrics,class_rows)
    report=['# Matched development results — November 1, 2026','',
        '21 clusters from four development donors, 16 expert-resolved identities and five cases requiring further review. Both arms use identical structured decoding. This is one checkpoint/seed and one anonymous expert reference, not final-test performance.','',
        '| Metric | Untuned | Expert-adapted |','|---|---:|---:|']
    for title,key,fmt in [('Completed answers','completed','count'),('Correct proposed broad identity (16 resolved)','proposed_correct','resolved'),
         ('Proposed-label macro-F1 (six classes)','proposed_macro_f1','float'),('Accepted-label macro-F1','accepted_macro_f1','float'),
         ('Requests for review (21 cases)','explicit_review_requests','count'),('Expert-uncertain cases accepted (five)','accepted_expert_uncertain','count'),
         ('Wrong resolved identities accepted','accepted_wrong_resolved','count'),('Median generation seconds','median_seconds','float')]:
        def show(m):
            v=m[key]
            return f'{v}/16 ({v/16:.1%})' if fmt=='resolved' else f'{v:.3f}' if fmt=='float' else str(v)
        report.append(f'| {title} | {show(metrics["untuned"])} | {show(metrics["expert_lora"])} |')
    report += ['', 'The adapted model improves proposed lineage agreement but loses review sensitivity: all five expert-uncertain cases are automatically accepted. Four accepted resolved cases have incorrect broad identities. The untuned model accepts no answers, so its accepted-answer accuracy is undefined.', '',
        '## Interpretation and limitations','',
        '- Normalization is explicitly analyst-defined after the run, not preregistered or independently blinded. The versioned map treats generic immune/neural-crest and mixed T/NK labels as insufficient for specific-lineage credit. Lineage agreement does not confirm malignancy or subtype.',
        '- One development donor has no resolved identities. See donor-comparison.csv rather than treating clusters as independent patient replicates. No significance or final-test claim is made.',
        '- Training logs confirm 101 training examples, three epochs, 39 updates, 225.945 seconds compute and 16.686 GiB peak allocation. These exclude setup, warmup and idle time. Billing and reviewer time are unavailable.',
        '- Supplied-gene and explicit-zero checks in case-comparison.csv are deterministic evidence flags, not validated reasoning scores or proof of biological truth. Subtypes remain tentative.',
        '- The grammar constrained both arms; completion improvements relative to older unconstrained runs cannot be attributed solely to fine-tuning.',
        '- Automatic-target adaptation, retrieval controls and the donor-separated final test remain necessary for the original study. Do not rewrite the main claim as established expert benefit from this comparison alone.', '',
        '## Files','', 'metrics.json: full denominators, review decisions, latency and evidence checks. case-comparison.csv: all 42 predictions with expert references. donor-comparison.csv and class-comparison.csv: stratified counts. development-results.png/pdf: exploratory scientific figure. audit.json: integrity checks. Source ZIP is preserved unchanged.']
    (OUT/'RESULTS.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(metrics,indent=2));print('Saved',OUT)


def make_figure(rows,metrics,class_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    colors=['#627d98','#1b8a70'];names=['Untuned','Expert-adapted']
    x=np.arange(6)
    for i,arm in enumerate(metrics):
        rr=[r for r in class_rows if r['arm']==arm]
        vals=[r['correct']/r['support'] for r in rr]
        bars=axes[0].bar(x+(i-.5)*.36,vals,.36,label=names[i],color=colors[i])
        for b,r in zip(bars,rr):
            axes[0].text(b.get_x()+b.get_width()/2,b.get_height()+.025,f'{r["correct"]}/{r["support"]}',ha='center',fontsize=9)
    axes[0].set_xticks(x,['Neuronal /\nneuroendocrine','Myeloid','Schwann /\nglial','T lymphoid','Stromal','Endothelial'],fontsize=8)
    axes[0].set_ylim(0,1.2);axes[0].set_ylabel('Proposed broad-label agreement');axes[0].legend(frameon=False)
    axes[0].set_title('Expert-resolved cases (n=16)')
    categories=[('Correct accepted',lambda r:r['resolved'] and r['correct'] and not r['review']),
        ('Wrong accepted',lambda r:r['resolved'] and not r['correct'] and not r['review']),
        ('Expert-uncertain accepted',lambda r:not r['resolved'] and not r['review']),
        ('Requested review',lambda r:r['review'])]
    cs=['#1b8a70','#be4250','#e4a343','#a9b5c2'];bottom=np.zeros(2)
    for (label,fn),c in zip(categories,cs):
        counts=[sum(fn(r) for r in rows if r['arm']==arm) for arm in metrics]
        axes[1].bar(names,counts,bottom=bottom,label=label,color=c,width=.6)
        for i,n in enumerate(counts):
            if n:axes[1].text(i,bottom[i]+n/2,str(n),ha='center',va='center',fontweight='bold')
        bottom+=counts
    axes[1].set_ylabel('Cases');axes[1].set_ylim(0,23);axes[1].set_title('Acceptance and review (n=21)')
    axes[1].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2)
    fig.suptitle('Development: better lineage agreement, lost review sensitivity',fontsize=14)
    fig.savefig(OUT/'development-results.png',dpi=180);fig.savefig(OUT/'development-results.pdf');plt.close(fig)


if __name__=='__main__':
    main()
