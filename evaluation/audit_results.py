"""Study-specific independent verification; all private inputs stay outside version control."""
import csv,hashlib,io,json,statistics,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description="Audit the frozen 402-judgment DeepEval study return against its private source package.")
parser.add_argument('--archive',type=Path,required=True)
parser.add_argument('--package',type=Path,required=True)
parser.add_argument('--predictions',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
OUT=args.output;PACKAGE=args.package;ARCHIVE=args.archive
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jl(p):return [json.loads(s)for s in p.read_text().splitlines()if s.strip()]
with zipfile.ZipFile(ARCHIVE)as z:
 assert z.testzip()is None
 for n in z.namelist():
  dest=OUT/'raw'/n
  assert dest.resolve().is_relative_to((OUT/'raw').resolve())
  if n.endswith('/'):continue
  dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.read_bytes()==z.read(n)
  else:dest.write_bytes(z.read(n))
raw=OUT/'raw/nemotron120b_deepeval_v1'
manifest=json.loads((raw/'run-manifest.json').read_text());protocol=json.loads((raw/'protocol.json').read_text());supplied=json.loads((raw/'summary.json').read_text());inventory=json.loads((raw/'inventory.json').read_text())
assert manifest['package_files']==inventory
for n,h in inventory.items():assert sha(PACKAGE/n)==h
for n in ['protocol.json','blind-map.jsonl']:assert sha(raw/n)==sha(PACKAGE/n)
assert sha(raw/'protocol.json')==manifest['protocol_sha256']
data={r['blind_id']:r for r in jl(PACKAGE/'dataset.jsonl')};mapping={r['blind_id']:r for r in jl(raw/'blind-map.jsonl')}
scores=jl(raw/'scores.jsonl');calls=jl(raw/'api-calls.jsonl');assert len(scores)==len(calls)==402
assert len({r['job_id']for r in scores})==len({r['job_id']for r in calls})==402
expected={bid+'__'+m['id']for bid in data for m in protocol['metrics']};assert {r['job_id']for r in scores}==expected
callmap={r['job_id']:r for r in calls};metricspec={m['id']:m for m in protocol['metrics']}
arm_metric=defaultdict(dict);bygroup=defaultdict(list);enriched=[];tokens=Counter();timings=[];models=Counter();fingerprints=Counter();request_checks=Counter();echoed_case_keys=[]
for r in scores:
 assert r['status']=='complete'and r['protocol_sha256']==manifest['protocol_sha256']
 c=callmap[r['job_id']];req=c['request'];resp=c['response'];row=data[r['blind_id']];ref=mapping[r['blind_id']];spec=metricspec[r['metric']]
 assert req['model']==protocol['judge']['model'];assert req['temperature']==0;assert req['seed']==20261106
 assert req['max_tokens']==768 and req['reasoning_effort']=='none'and req['stream']is False
 assert req['response_format']['type']=='json_schema'
 prompt=req['messages'][1]['content']
 assert row['input']in prompt and row['actual_output']in prompt
 assert (row['expected_output']in prompt)==spec['uses_expert_reference']
 assert ref['donor']not in prompt and ref['sample_id']not in prompt
 if ref['case_id']in prompt:
  assert ref['case_id']in row['actual_output'] and ref['case_id']not in row['input']
  echoed_case_keys.append({'job_id':r['job_id'],'case_id':ref['case_id'],'arm':ref['arm'],'source':'Identifier echoed in an unchanged original model answer, not added metadata'})
 choice=resp['choices'][0];assert choice['finish_reason']=='stop'
 answer=json.loads(choice['message']['content']);assert abs(answer['score']/10-r['score'])<1e-12
 assert answer['reason']==r['reason']and 0<=r['score']<=1
 assert resp['model']==r['returned_model']
 arm_metric[r['metric'],ref['arm']][ref['case_id']]=r['score'];bygroup[r['metric'],ref['arm'],ref['expert_requires_review']].append(r['score'])
 enriched.append({**ref,'metric':r['metric'],'score':r['score'],'reason':r['reason']})
 models[resp['model']]+=1;fingerprints[str(resp.get('system_fingerprint'))]+=1
 for k in ['prompt_tokens','completion_tokens','total_tokens']:tokens[k]+=(resp.get('usage')or{}).get(k,0)or 0
 timings.append(c['seconds']);request_checks[r['metric']]+=1
summary=[];donorrows=[]
for spec in protocol['metrics']:
 m=spec['id'];a=arm_metric[m,'untuned'];b=arm_metric[m,'expert_lora'];assert len(a)==len(b)==67 and a.keys()==b.keys()
 result={'metric':m,'name':spec['name'],'untuned_mean':statistics.mean(a.values()),'v1_mean':statistics.mean(b.values()),'paired_mean_delta':statistics.mean([b[k]-a[k]for k in a]),'v1_higher':sum(b[k]>a[k]for k in a),'ties':sum(b[k]==a[k]for k in a),'v1_lower':sum(b[k]<a[k]for k in a),'cases_per_arm':67}
 old=next(s for s in supplied['metrics']if s['metric']==m)
 for k in ['untuned_mean','v1_mean','paired_mean_delta']:assert abs(old[k]-result[k])<1e-12
 for k in ['v1_higher','ties','v1_lower']:assert old[k]==result[k]
 result['by_expert_review_status']={str(flag):{'untuned_mean':statistics.mean(bygroup[m,'untuned',flag]),'v1_mean':statistics.mean(bygroup[m,'expert_lora',flag]),'cases_per_arm':len(bygroup[m,'untuned',flag])}for flag in [False,True]}
 summary.append(result)
 for donor in sorted({r['donor']for r in mapping.values()}):
  cid={r['case_id']for r in mapping.values()if r['donor']==donor}
  donorrows.append({'donor':donor,'metric':m,'cases':len(cid),'untuned_mean':statistics.mean([a[k]for k in cid]),'v1_mean':statistics.mean([b[k]for k in cid]),'paired_delta':statistics.mean([b[k]-a[k]for k in cid])})
# Exact preservation: all original identity answers match the frozen dataset and both arms share input evidence.
original={(r['case_id'],r['arm']):r for r in jl(args.predictions)if r['stage']=='identity'}
for bid,row in data.items():
 ref=mapping[bid];assert row['actual_output']==original[ref['case_id'],ref['arm']]['raw_text']
for name,records in [('per-case-audited.csv',enriched),('per-donor.csv',donorrows)]:
 with (OUT/name).open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
audit={'archive_sha256':sha(ARCHIVE),'protocol_sha256':manifest['protocol_sha256'],'judgments':402,'unique_jobs':402,'complete':402,'failed':0,'missing':0,'answers':134,'cases_per_arm':67,'metrics':3,'all_402_prompt_inputs_and_outputs_verified':True,'expert_reference_only_used_for_fidelity':True,'arm_labels_donor_and_sample_identifiers_withheld':True,'input_case_identifiers_removed':True,'original_answer_case_key_echoes':echoed_case_keys,'all_response_scores_independently_recomputed':True,'all_original_answers_preserved':True,'all_source_package_hashes_match':True,'returned_models':dict(models),'system_fingerprints':dict(fingerprints),'api_usage':dict(tokens),'median_seconds_per_api_call':statistics.median(timings),'sum_api_seconds':sum(timings),'currency_cost':'Not supplied or inferred','score_semantics':'Normalized 0–10 rubric ratings, not biological accuracy percentages','metrics_summary':summary,'limitations':protocol['protocol_limits']+['Audit qualification: three original model answers echo their arbitrary case key. Those keys remained in 9 judge prompts because original answers were preserved; no donor/sample IDs or model-arm labels were added.']}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(OUT/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'checks':'passed','judgments':402,'models':dict(models),'tokens':dict(tokens),'metrics':summary,'donors':donorrows},indent=2))
