"""Audit and score the returned test run using only the pre-inference scoring lock."""
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from jsonschema import validate
from matched_evaluation import canonical_hash,case_schedule
from evaluation_metrics import evidence_checks,latency

ROOT = Path(__file__).resolve().parents[1]
HANDOFF=ROOT/'outputs/frozen-test-evaluation-v1'
OUT=ROOT/'outputs/final-test-results'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_policy(predictions,refs,mapping,groups,gates=None):
    rows=[]
    for cid,ref in refs.items():
        identity=predictions.get(cid,{})
        a=identity.get('annotation') or {}
        identity_valid=identity.get('status')=='complete' and bool(a.get('broad_cell_type'))
        decision=(gates if gates is not None else predictions).get(cid,{})
        d=decision.get('annotation') or {}
        decision_valid=decision.get('status')=='complete' and type(d.get('needs_expert_review')) is bool
        explicit=d['needs_expert_review'] if decision_valid else None
        group=mapping.get(' '.join(a.get('broad_cell_type','').casefold().split()),'unmapped') if identity_valid else 'invalid'
        resolved=not ref['review']
        rows.append({
            'case_id':cid,'resolved':resolved,'expert_review':ref['review'],'reference_group':ref['group'],
            'model_label':a.get('broad_cell_type'),'proposed_group':group,
            'identity_valid':identity_valid,'identity_status':identity.get('status','not_run'),
            'decision_valid':decision_valid,'decision_status':decision.get('status','not_run'),
            'explicit_review':explicit,'operational_review':explicit is not False or not identity_valid,
            'correct':bool(resolved and identity_valid and group==ref['group']),
        })
    resolved=[r for r in rows if r['resolved']]
    accepted=[r for r in rows if not r['operational_review']]
    accepted_resolved=[r for r in accepted if r['resolved']]
    def macro(accepted_only=False):
        scores=[]
        for g in groups:
            def predicted(r):return r['proposed_group'] if not accepted_only or not r['operational_review'] else 'review'
            tp=sum(r['reference_group']==g and predicted(r)==g for r in resolved)
            fp=sum(r['reference_group']!=g and predicted(r)==g for r in resolved)
            fn=sum(r['reference_group']==g and predicted(r)!=g for r in resolved)
            scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        return sum(scores)/len(scores) if scores else None
    tp=sum(r['expert_review'] and r['explicit_review'] is True for r in rows)
    requests=sum(r['explicit_review'] is True for r in rows)
    positives=sum(r['expert_review'] for r in rows)
    wrong_flagged=[r['case_id'] for r in resolved if not r['correct'] and r['explicit_review'] is True]
    m={
        'cases':len(rows),'identity_complete':sum(r['identity_valid'] for r in rows),
        'decision_complete':sum(r['decision_valid'] for r in rows),'resolved_cases':len(resolved),
        'proposed_correct':sum(r['correct'] for r in resolved),
        'proposed_accuracy':sum(r['correct'] for r in resolved)/len(resolved) if resolved else None,
        'proposed_macro_f1':macro(),'accepted_macro_f1':macro(True),
        'explicit_review_requests':requests,'explicit_review_true_positive':tp,
        'explicit_review_false_positive':sum(r['resolved'] and r['explicit_review'] is True for r in rows),
        'explicit_review_false_negative':positives-tp,
        'explicit_review_precision':tp/requests if requests else None,
        'explicit_review_recall':tp/positives if positives else None,
        'operational_review_true_positive':sum(r['expert_review'] and r['operational_review'] for r in rows),
        'operational_review_routes':sum(r['operational_review'] for r in rows),
        'format_failure_routes':sum(not r['decision_valid'] or not r['identity_valid'] for r in rows),
        'accepted':len(accepted),'automatic_coverage':len(accepted)/len(rows) if rows else None,
        'accepted_resolved':len(accepted_resolved),
        'accepted_resolved_accuracy':sum(r['correct'] for r in accepted_resolved)/len(accepted_resolved) if accepted_resolved else None,
        'wrong_resolved_accepted':sum(not r['correct'] for r in accepted_resolved),
        'expert_uncertain_accepted':sum(r['expert_review'] for r in accepted),
        'safe_accepted_count':sum(r['correct'] for r in accepted_resolved),
        'safe_accepted_yield':sum(r['correct'] for r in accepted_resolved)/len(rows) if rows else None,
        'accepted_supported_fraction':sum(r['correct'] for r in accepted_resolved)/len(accepted) if accepted else None,
        'unmapped_resolved_labels':sum(r['proposed_group']=='unmapped' for r in resolved),
        'unmapped_all_cases':sum(r['proposed_group']=='unmapped' for r in rows),
        'wrong_identity_intercepted':len(wrong_flagged),'wrong_identity_intercepted_case_ids':wrong_flagged,
    }
    return m,rows


def write_csv(path,rows):
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit_archive(archive,config,lock):
    packets={p['case_id']:p for p in map(json.loads,(HANDOFF/'inference-package/case-evidence.jsonl').read_text().splitlines())}
    records={};canonical=[];manifests={}
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert len(names)==len(set(names)) and z.testzip() is None
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        for name in ['config.json','case-evidence.jsonl','matched_evaluation.py','frozen_test_evaluation.py']:
            assert z.read('protocol_sources/'+name)==(HANDOFF/'inference-package'/name).read_bytes()
        returned_summary=json.loads(z.read('summary.json'))
        assert returned_summary['expected_attempts']==268 and returned_summary['all_attempts_saved'] is True
        for stage in ['identity','review_first']:
            manifest=json.loads(z.read(stage+'/run_manifest.json'));manifests[stage]=manifest
            assert manifest['protocol']==config['protocol'] and manifest['stage']==stage and manifest['partition']=='test'
            for key in ['model_id','revision','adapter_hashes','software','gpu','scoring_lock_sha256','scoring_reference_sha256']:
                assert manifest[key]==config[key],key
            assert manifest['package_config_sha256']==digest(HANDOFF/'inference-package/config.json')
            assert manifest['scoring_lock_sha256']==digest(HANDOFF/'scoring-protocol.json')
            assert manifest['scoring_reference_sha256']==lock['reference_ledger_sha256']
            assert manifest['schema']==config['stages'][stage]['schema']
            assert manifest['decoding']==config['stages'][stage]['decoding']
            assert manifest['case_order']==[list(p) for p in case_schedule(lock['references'])]
            assert set(manifest['case_input_hashes'])==set(packets)==set(lock['references'])
            expected_system=config['identity_system' if stage=='identity' else 'review_system']
            records[stage]={}
            for arm in ['untuned','expert_lora']:
                selected=[n for n in names if n.startswith(stage+'/'+arm+'/') and n.endswith('.json')]
                assert len(selected)==67 and {Path(n).stem for n in selected}==set(packets)
                arm_records={}
                for n in selected:
                    r=json.loads(z.read(n));cid=r['case_id']
                    assert Path(n).stem==cid and r['stage']==stage and r['partition']=='test' and r['arm']==arm
                    assert r['run_hash']==canonical_hash(manifest)
                    expected_messages=[{'role':'system','content':expected_system}, {'role':'user','content':json.dumps(packets[cid])}]
                    assert r['messages']==expected_messages
                    assert r['input_sha256']==manifest['case_input_hashes'][cid]==canonical_hash(expected_messages)
                    assert r['output_tokens']==len(r['output_token_ids'])
                    assert r['stopped_at_eos']==bool(r['output_token_ids'] and r['output_token_ids'][-1]==11)
                    assert r['terminal'] is True and r['runtime_error'] is None
                    if r['status']=='complete':
                        assert r['stopped_at_eos'] and r['annotation']==json.loads(r['raw_text'])
                        validate(r['annotation'],manifest['schema'])
                    else:
                        assert r['status']=='time_limit' and not r['stopped_at_eos']
                        assert r['generation_seconds']>=manifest['decoding']['max_time_seconds']
                    assert datetime.fromisoformat(r['created_utc'])>datetime.fromisoformat(lock['locked_utc'])
                    arm_records[cid]=r;canonical.append(r)
                records[stage][arm]=arm_records
                counts=dict(Counter(r['status'] for r in arm_records.values()))
                assert returned_summary['stages'][stage]['counts'][arm]==counts
            history=[json.loads(z.read(n)) for n in names if n.startswith(stage+'/attempt_history/') and n.endswith('.json')]
            for arm,arm_records in records[stage].items():
                for r in arm_records.values():assert r in history
        assert len(canonical)==268
    audit={'source_zip_sha256':digest(archive),'zip_integrity_verified':True,'canonical_attempts':len(canonical),
           'complete_attempts':sum(r['status']=='complete' for r in canonical),
           'retained_failures':[{'case_id':r['case_id'],'stage':r['stage'],'arm':r['arm'],'status':r['status']} for r in canonical if r['status']!='complete'],
           'exact_expression_packets_and_prompts_verified':True,'schema_and_decoding_verified':True,
           'frozen_scoring_lock_and_reference_hashes_verified':True,'source_scripts_byte_identical':True,
           'v1_adapter_hashes_match_development':True,'actual_adapter_binary_independently_checked':False,
           'gpu_and_software_match_development':True,'all_canonical_outputs_present_in_attempt_history':True,
           'no_training_or_additional_model_calls_performed_by_scorer':True,
           'scoring_lock_sha256':digest(HANDOFF/'scoring-protocol.json')}
    return records,packets,audit,canonical


def main():
    config=json.loads((HANDOFF/'inference-package/config.json').read_text())
    lock=json.loads((HANDOFF/'scoring-protocol.json').read_text())
    ledger_path=ROOT/'outputs/test-review-67-cases/independent-expert-ledger/test-independent-annotations.jsonl'
    assert digest(ledger_path)==lock['reference_ledger_sha256']==config['scoring_reference_sha256']
    assert digest(HANDOFF/'scoring-protocol.json')==config['scoring_lock_sha256']
    ledger={r['case_id']:r for r in map(json.loads,ledger_path.read_text().splitlines())}
    assert set(ledger)==set(lock['references'])
    for cid,r in ledger.items():
        assert r['requires_further_review']==lock['references'][cid]['review']
        if not r['requires_further_review']:
            assert lock['label_map'][' '.join(r['broad_identity_verbatim'].casefold().split())]==lock['references'][cid]['group']
    archive=ROOT/'outputs/nemotron_4b_frozen_test_v1.zip'
    records,packets,audit,canonical=audit_archive(archive,config,lock)
    OUT.mkdir(exist_ok=True)
    metrics={};rows=[];evidence={};evidence_rows=[];resources={}
    for stage,arms in records.items():
        for arm,rr in arms.items():
            label=('untuned' if arm=='untuned' else 'v1')+('' if stage=='identity' else '_review_first')
            m,ev=evidence_checks(rr,packets,100 if stage=='identity' else 40)
            evidence[label]=m;evidence_rows.extend(dict(condition=label,stage=stage,**r)for r in ev)
            resources[label]=latency(rr)
            resources[label]['note']='Fresh test attempts, including the timeout; excludes setup, warmup, idle and restart overhead.'
    for name,arm,gated in [('untuned','untuned',False),('v1','expert_lora',False),('untuned_review_first','untuned',True),('v1_review_first','expert_lora',True)]:
        metrics[name],rr=score_policy(records['identity'][arm],lock['references'],lock['label_map'],lock['groups'],
                                     records['review_first'][arm] if gated else None)
        for r in rr:
            reference=ledger[r['case_id']]
            rows.append(dict(condition=name,donor=reference['participant_id'],sample_id=reference['sample_id'],
                cluster=reference['cluster'],cells=reference['cell_count'],
                expert_label=reference['broad_identity_verbatim'],expert_review_category=reference['review_decision_verbatim'],**r))
    donors=[];classes=[]
    for name in metrics:
        for donor in sorted(set(lock['donors'].values())):
            rr=[r for r in rows if r['condition']==name and r['donor']==donor]
            n=sum(r['resolved'] for r in rr)
            donors.append(dict(condition=name,donor=donor,cases=len(rr),resolved=n,
                expert_uncertain=sum(r['expert_review'] for r in rr),proposed_correct=sum(r['correct'] for r in rr),
                proposed_accuracy=sum(r['correct'] for r in rr)/n if n else None,
                explicit_review_requests=sum(r['explicit_review'] is True for r in rr),
                uncertain_cases_flagged=sum(r['expert_review'] and r['explicit_review'] is True for r in rr),
                supported_correct_accepted=sum(r['correct'] and not r['operational_review'] for r in rr)))
        for g in lock['groups']:
            rr=[r for r in rows if r['condition']==name and r['resolved'] and r['reference_group']==g]
            classes.append(dict(condition=name,reference_group=g,support=len(rr),proposed_correct=sum(r['correct'] for r in rr)))
    paired=[]
    for cid,reference in lock['references'].items():
        if reference['review']:continue
        a=next(r for r in rows if r['condition']=='untuned' and r['case_id']==cid)
        b=next(r for r in rows if r['condition']=='v1' and r['case_id']==cid)
        paired.append(dict(case_id=cid,donor=lock['donors'][cid],untuned_correct=a['correct'],v1_correct=b['correct'],
                           change='gain' if not a['correct'] and b['correct'] else 'loss' if a['correct'] and not b['correct'] else 'unchanged'))
    report={'scope':'Frozen reserved-donor test evaluation: 67 clusters, six biological samples, four donors; 57 references not requesting further review and ten review-positive references.',
        'metrics':metrics,'evidence':evidence,'generation_resources':resources,
        'paired_changes':dict(Counter(r['change'] for r in paired)),
        'reference_groups':lock['identity_reference_group_counts'],
        'expert_time_savings':{'measured':False,'value':None},'cost':{'measured':False,'usd':None},
        'source_protocol_sha256':digest(HANDOFF/'scoring-protocol.json'),
        'interpretation':'Agreement uses the locked exact alias map. Unmapped labels count as non-agreement; not every such label is an independently confirmed biological error. Subtype and malignancy are not validated.'}
    write_csv(OUT/'case-comparison.csv',rows);write_csv(OUT/'donor-comparison.csv',donors)
    write_csv(OUT/'class-comparison.csv',classes);write_csv(OUT/'paired-identity-changes.csv',paired)
    write_csv(OUT/'evidence-checks.csv',evidence_rows)
    write_csv(OUT/'unmapped-labels.csv',[r for r in rows if r['condition'] in ['untuned','v1'] and r['proposed_group']=='unmapped'])
    (OUT/'final-test-metrics.json').write_text(json.dumps(report,indent=2))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    (OUT/'predictions-preserved.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in canonical))
    for source,name in [(HANDOFF/'scoring-protocol.json','locked-scoring-protocol.json'),(ledger_path,'test-independent-annotations.jsonl'),(archive,'raw-test-results.zip')]:
        shutil.copyfile(source,OUT/name)
    print(json.dumps({'metrics':metrics,'paired_changes':report['paired_changes'],'audit':audit},indent=2))


if __name__=='__main__':main()
