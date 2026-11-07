"""Review policy, evidence traceability and latency metrics."""
import csv
import hashlib
import io
import json
import statistics
import time
import zipfile
from pathlib import Path

REVIEW_SYSTEM='''Decide whether the supplied single-cell RNA-seq cluster needs expert review of its BROAD cell identity.
Use only the supplied expression evidence and established biological knowledge.
Request review if a single broad lineage is not defensible because of weak, conflicting, mixed or poor-quality evidence.
An unsupported subtype or unconfirmed malignancy alone does not require broad-identity review when lineage is otherwise clear.
Missing evidence is not zero expression. Cluster averages cannot establish single-cell coexpression or doublets.
Make this review decision BEFORE assigning any cell identity. Do not assign an identity here.
Return only JSON, in this order: needs_expert_review (Boolean), supporting_genes (up to four supplied genes informing the decision), explanation (brief, at most 40 words).'''
REVIEW_SCHEMA={'type':'object','additionalProperties':False,'properties':{
    'needs_expert_review':{'type':'boolean'},
    'supporting_genes':{'type':'array','maxItems':4,'items':{'type':'string','minLength':1,'maxLength':32}},
    'explanation':{'type':'string','maxLength':240}},
    'required':['needs_expert_review','supporting_genes','explanation']}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_policy(predictions,refs,mapping,gates=None):
    rows=[]
    for cid,ref in refs.items():
        r=predictions.get(cid,{})
        a=r.get('annotation') or {}
        identity_valid=r.get('status')=='complete' and bool(a.get('broad_cell_type'))
        decision=gates.get(cid,{}) if gates is not None else r
        d=decision.get('annotation') or {}
        valid=decision.get('status')=='complete' and type(d.get('needs_expert_review')) is bool
        explicit=d['needs_expert_review'] if valid else None
        group=mapping.get(' '.join(a.get('broad_cell_type','').casefold().split()),'unmapped') if identity_valid else 'invalid'
        routed=explicit is not False or not identity_valid
        resolved=not ref['review']
        rows.append(dict(case_id=cid,expert_review=ref['review'],resolved=resolved,reference_group=ref['group'],
            proposed_group=group,model_label=a.get('broad_cell_type'),identity_valid=identity_valid,
            explicit_review=explicit,decision_valid=valid,operational_review=routed,
            correct=bool(resolved and identity_valid and group==ref['group']),
            decision_status=decision.get('status','not_run')))
    resolved=[r for r in rows if r['resolved']];accepted=[r for r in rows if not r['operational_review']]
    accepted_resolved=[r for r in accepted if r['resolved']]
    tp=sum(r['expert_review'] and r['explicit_review'] is True for r in rows)
    requests=sum(r['explicit_review'] is True for r in rows)
    positives=sum(r['expert_review'] for r in rows)
    def macro(accepted_only=False):
        values=[]
        for g in DEVELOPMENT_GROUPS:
            def prediction(r):return r['proposed_group'] if not accepted_only or not r['operational_review'] else 'review'
            t=sum(r['reference_group']==g and prediction(r)==g for r in resolved)
            f=sum(r['reference_group']!=g and prediction(r)==g for r in resolved)
            n=sum(r['reference_group']==g and prediction(r)!=g for r in resolved)
            values.append(2*t/(2*t+f+n) if 2*t+f+n else 0)
        return sum(values)/len(values)
    return dict(cases=len(rows),identity_complete=sum(r['identity_valid'] for r in rows),
        decision_complete=sum(r['decision_valid'] for r in rows),resolved_cases=len(resolved),
        proposed_correct=sum(r['correct'] for r in resolved),
        proposed_accuracy=sum(r['correct'] for r in resolved)/len(resolved) if resolved else None,
        proposed_macro_f1=macro(),accepted_macro_f1=macro(True),
        explicit_review_requests=requests,explicit_review_true_positive=tp,
        explicit_review_false_positive=sum(r['resolved'] and r['explicit_review'] is True for r in rows),
        explicit_review_precision=tp/requests if requests else None,
        explicit_review_recall=tp/positives if positives else None,
        operational_review_true_positive=sum(r['expert_review'] and r['operational_review'] for r in rows),
        format_failure_routes=sum(not r['decision_valid'] or not r['identity_valid'] for r in rows),
        accepted=len(accepted),automatic_coverage=len(accepted)/len(rows),
        accepted_resolved=len(accepted_resolved),
        accepted_resolved_accuracy=sum(r['correct'] for r in accepted_resolved)/len(accepted_resolved) if accepted_resolved else None,
        wrong_resolved_accepted=sum(not r['correct'] for r in accepted_resolved),
        expert_uncertain_accepted=sum(r['expert_review'] for r in accepted),
        safe_accepted_count=sum(r['correct'] for r in accepted_resolved),
        safe_accepted_yield=sum(r['correct'] for r in accepted_resolved)/len(rows)),rows


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evidence_checks(records,packets,word_limit=100):
    rows=[]
    for cid,r in records.items():
        a=r.get('annotation') or {};genes=a.get('supporting_genes',[])
        p=packets[cid];given={g['gene'] for g in p['ranked_markers']+p['marker_panel'] if g.get('gene')}
        zero={g['gene'] for g in p['marker_panel'] if g['fraction_expressing']==0 and g['mean_log_expression']==0}
        rows.append(dict(case_id=cid,citations=len(genes),citations_in_packet=sum(g in given for g in genes),
            unverifiable_genes=';'.join(g for g in genes if g not in given),
            explicit_zero_support_genes=';'.join(g for g in genes if g in zero),
            duplicate_entries=len(genes)-len(set(genes)),explanation_words=len(a.get('explanation','').split())))
    total=sum(r['citations'] for r in rows);provided=sum(r['citations_in_packet'] for r in rows)
    return dict(cited_entries=total,cited_entries_in_packet=provided,citation_traceability=provided/total if total else None,
        cases_with_zero_support_flag=sum(bool(r['explicit_zero_support_genes']) for r in rows),
        duplicate_entries=sum(r['duplicate_entries'] for r in rows),
        explanation_word_limit=word_limit,explanation_word_limit_violations=sum(r['explanation_words']>word_limit for r in rows),
        caveat='Gene presence/zero/duplicate checks do not validate biological interpretation or reasoning.'),rows


def latency(records):
    if not records:return {'attempted':0}
    times=sorted(r['generation_seconds'] for r in records.values());p=(len(times)-1)*.9;i=int(p)
    return dict(attempted=len(records),complete=sum(r['status']=='complete' for r in records.values()),
        status_counts={status:sum(r['status']==status for r in records.values()) for status in sorted({r['status'] for r in records.values()})},
        eos_completed=sum(r.get('stopped_at_eos',False) for r in records.values()),
        generation_seconds=sum(times),median_seconds=statistics.median(times),
        p90_seconds=times[i]+(times[min(i+1,len(times)-1)]-times[i])*(p-i),
        total_output_tokens=sum(r['output_tokens'] for r in records.values()),
        peak_allocated_gib=max(r['peak_gpu_allocated_gib'] for r in records.values()),
        note='Includes timed-out attempted generations; excludes warmup, setup and idle. Prior conditions are historical reused measurements.')


DEVELOPMENT_GROUPS = ['neuronal_neuroendocrine', 'myeloid', 'schwann_glial', 't_lymphoid', 'stromal_mesenchymal', 'endothelial']
DEVELOPMENT_LABEL_MAP = {'neuronal': 'neuronal_neuroendocrine', 'neuroblastoma': 'neuronal_neuroendocrine', 'neuroblastic (sympathoadrenal-lineage) tumor-like cells': 'neuronal_neuroendocrine', 'tentative: malignant neuroblastic (sympathoadrenal-lineage) tumor-like cells': 'neuronal_neuroendocrine', 'myeloid cells (macrophages)': 'myeloid', 'myeloid cells (macrophage/monocyte)': 'myeloid', 'tentative: macrophages (myeloid lineage, non-specific evidence)': 'myeloid', 'tentative: macrophages (myeloid lineage, phagocytic state, likely tumor-associated)': 'myeloid', 'schwann cells': 'schwann_glial', 'schwann-lineage glial cells (schwannian stroma)': 'schwann_glial', 't cells': 't_lymphoid', 't lymphocytes': 't_lymphoid', 'mesenchymal': 'stromal_mesenchymal', 'stromal / mesenchymal cells': 'stromal_mesenchymal', 'endothelial cells': 'endothelial', 'tentative: t/nk cells': 'mixed_t_nk', 'immune cell': 'overbroad_immune', 'neural crest-derived': 'overbroad_neural_crest', 'macrophages': 'myeloid', 'tentative: macrophages (myeloid lineage, phagocytic state, with some dendritic cell features)': 'myeloid', 'enteric (ileal) neuroendocrine cells': 'neuronal_neuroendocrine', 'enteric neurons': 'neuronal_neuroendocrine', 'neuroblast': 'neuronal_neuroendocrine', 'neuroblastic': 'neuronal_neuroendocrine', 'neuroblastic (chga+), neural crest lineage': 'neuronal_neuroendocrine', 'neuroblastic (sympathoadrenal) lineage': 'neuronal_neuroendocrine', 'neuroblastic cells': 'neuronal_neuroendocrine', 't cell': 't_lymphoid'}
