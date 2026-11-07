"""Inference-only test workflow. Never loads expert reference answers."""
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from matched_evaluation import (MatchedRunner, ResultStore, atomic_json,
    canonical_hash, case_schedule, export_results, file_hash)

PACKET_FIELDS={'case_id','tissue_context','cell_count','capture_unit','ranked_markers',
               'marker_panel','missing_panel_evidence','technical_summary','evidence_notes'}


def prepare_test_cases(package, config, tokenizer):
    package=Path(package)
    for name,expected in config['files'].items():
        if file_hash(package/name)!=expected:raise ValueError('Package file hash differs: '+name)
    packets=[json.loads(line) for line in (package/'case-evidence.jsonl').read_text().splitlines()]
    if any(set(p)-PACKET_FIELDS for p in packets):raise ValueError('Unexpected expression packet fields; references are forbidden')
    ids=[p['case_id'] for p in packets]
    if len(ids)!=len(set(ids)) or set(ids)!=set(config['case_ids']):raise ValueError('Case inventory differs')
    result={}
    for stage,key in [('identity','identity_system'),('review_first','review_system')]:
        cases=[]
        for p in sorted(packets,key=lambda p:p['case_id']):
            messages=[{'role':'system','content':config[key]}, {'role':'user','content':json.dumps(p)}]
            tokens=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            cases.append({'case_id':p['case_id'],'messages':messages,
                          'input_sha256':canonical_hash(messages),'input_tokens':len(tokens)})
        result[stage]=cases
    return result


class TestResultStore(ResultStore):
    def __init__(self,directory,manifest):
        self.stage=manifest['stage']
        super().__init__(directory,manifest)

    def save(self,record):
        # Reuse the validated runner without its historical development partition label.
        record['partition']='test'
        record['stage']=self.stage
        super().save(record)


def run_test_evaluation(env,total_minutes=70):
    import torch
    from contextlib import nullcontext
    from transformers import GenerationConfig
    model=env['test_model'];tokenizer=env['tokenizer']
    config=env['test_config'];package=Path(env['PACKAGE']);root=Path(env['ROOT'])
    model.set_adapter('expert');model.requires_grad_(False);model.eval()
    model.gradient_checkpointing_disable();model.config.use_cache=False
    assert tokenizer.eos_token_id==11 and next(model.parameters()).device.type=='cuda'
    cases_by_stage=prepare_test_cases(package,config,tokenizer)
    assert len(cases_by_stage['identity'])==67
    out=root/'nemotron_4b_frozen_test_v1';out.mkdir(exist_ok=True)
    # Capture the exact inference-only package and pre-inference scoring-lock hashes.
    for name in ['case-evidence.jsonl','config.json','matched_evaluation.py','frozen_test_evaluation.py']:
        dest=out/'protocol_sources'/name;dest.parent.mkdir(exist_ok=True)
        if dest.exists():assert file_hash(dest)==file_hash(package/name)
        else:shutil.copyfile(package/name,dest)
    stores={}
    for stage,cases in cases_by_stage.items():
        manifest={
            'protocol':'frozen-reserved-donor-test-v1','stage':stage,'partition':'test',
            'model_id':config['model_id'],'revision':config['revision'],
            'adapter_hashes':config['adapter_hashes'],'software':env['test_software'],
            'gpu':torch.cuda.get_device_name(0),'schema':config['stages'][stage]['schema'],
            'decoding':config['stages'][stage]['decoding'],
            'case_input_hashes':{c['case_id']:c['input_sha256'] for c in cases},
            'case_order':case_schedule([c['case_id'] for c in cases]),
            'package_config_sha256':file_hash(package/'config.json'),
            'scoring_lock_sha256':config['scoring_lock_sha256'],
            'scoring_reference_sha256':config['scoring_reference_sha256'],
            'scope':'Expression-only inference; expert answers remain local. No training or test-driven tuning.',
        }
        stores[stage]=TestResultStore(out/stage,manifest)
    archive=root/'nemotron_4b_frozen_test_v1.zip'
    start=time.perf_counter();runner=None
    try:
        for stage,cases in cases_by_stage.items():
            store=stores[stage];settings=config['stages'][stage]
            if all(store.is_done(c['case_id'],arm,c['input_sha256']) for c in cases for arm in ['untuned','expert_lora']):
                print(stage+': all 134 attempts already saved.',flush=True);continue
            if time.perf_counter()-start>=total_minutes*60:break
            if not (store.directory/'warmup.json').exists():
                warm=tokenizer.apply_chat_template(cases[0]['messages'],tokenize=True,add_generation_prompt=True,
                    enable_thinking=False,return_dict=True,return_tensors='pt').to('cuda')
                times={}
                for arm in ['untuned','expert_lora']:
                    torch.cuda.synchronize();t0=time.perf_counter()
                    with (model.disable_adapter() if arm=='untuned' else nullcontext()),torch.inference_mode():
                        one=model.generate(**warm,generation_config=GenerationConfig(max_new_tokens=1,
                            do_sample=False,use_cache=False,eos_token_id=11,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 11))
                    torch.cuda.synchronize();times[arm]=time.perf_counter()-t0;del one
                del warm
                atomic_json(store.directory/'warmup.json',{'scored':False,'seconds':times})
            if runner is None:
                runner=MatchedRunner(model,tokenizer,store,schema=settings['schema'],
                    max_new_tokens=settings['decoding']['max_new_tokens'],
                    max_seconds=settings['decoding']['max_time_seconds'])
            else:
                runner.store=store;runner.schema=settings['schema']
                runner.max_new_tokens=settings['decoding']['max_new_tokens']
                runner.max_seconds=settings['decoding']['max_time_seconds']
            for i,(cid,arm) in enumerate(case_schedule([c['case_id'] for c in cases]),1):
                if time.perf_counter()-start>=total_minutes*60:break
                case=next(c for c in cases if c['case_id']==cid)
                print(f'\n{stage}: attempt {i}/134',flush=True)
                record=runner.run_one(case,arm)
                store.summary(cases)
                if record['status']=='interrupted':raise KeyboardInterrupt('Partial attempt saved; rerun this cell to resume.')
                if record['status']=='runtime_error':raise RuntimeError('Stopped on saved runtime error: '+str(record['runtime_error']))
            if time.perf_counter()-start>=total_minutes*60:break
    finally:
        status={stage:store.summary(cases_by_stage[stage]) for stage,store in stores.items()}
        complete=all(store.is_done(c['case_id'],arm,c['input_sha256'])
            for stage,store in stores.items() for c in cases_by_stage[stage]
            for arm in ['untuned','expert_lora']) if not any(
                'runtime_error' in item['counts'][arm]
                for item in status.values() for arm in ['untuned','expert_lora']) else False
        atomic_json(out/'summary.json',{'all_attempts_saved':complete,'expected_attempts':268,
            'stages':status,'expert_scoring_performed':False})
        export_results(SimpleNamespace(directory=out),archive)
        print('\nSaved ZIP:',archive,flush=True)
        print('All 268 attempts saved:',complete,flush=True)
        if not complete:print('Rerun this SAME inference cell to resume. Do not reload the model.',flush=True)
        env['test_archive']=archive
    return archive
