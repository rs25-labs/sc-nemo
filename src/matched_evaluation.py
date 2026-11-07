"""Self-contained matched inference runner, embedded into the clean Colab notebook."""
import csv
import gc
import hashlib
import io
import json
import time
import zipfile
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'broad_cell_type': {'type': 'string', 'minLength': 1, 'maxLength': 160},
        'subtype_or_state': {'anyOf': [{'type': 'null'}, {'type': 'string', 'maxLength': 150}]},
        'supporting_genes': {'type': 'array', 'maxItems': 8, 'items': {'type': 'string', 'minLength': 1, 'maxLength': 32}},
        'conflicting_or_missing_evidence': {'type': 'array', 'maxItems': 2, 'items': {'type': 'string', 'maxLength': 220}},
        'alternative_interpretations': {'type': 'array', 'maxItems': 2, 'items': {'type': 'string', 'maxLength': 160}},
        'needs_expert_review': {'type': 'boolean'},
        'explanation': {'type': 'string', 'maxLength': 400},
    },
}
OUTPUT_SCHEMA['required'] = list(OUTPUT_SCHEMA['properties'])


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def case_schedule(case_ids):
    return [(case_id, arm) for i, case_id in enumerate(sorted(case_ids))
            for arm in (['untuned', 'expert_lora'] if i % 2 == 0 else ['expert_lora', 'untuned'])]


def make_parser(schema=None):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    return JsonSchemaParser(schema or OUTPUT_SCHEMA, config=CharacterLevelParserConfig(
        force_json_field_order=True, max_consecutive_whitespaces=2))


def load_development_cases(root):
    """Read frozen case IDs and exact original prompts; never read expert targets."""
    root = Path(root)
    package = root / 'training-development-cases-v1.zip'
    assert file_hash(package) == '7e5f9b84bbab920076ee9516ead1b351ba9dbfcff97dc0a4c946249c77876d5f', 'Evidence ZIP differs.'
    with zipfile.ZipFile(package) as z:
        rows = list(csv.DictReader(io.StringIO(z.read('case_manifest.csv').decode())))
        packets = {p['case_id']: p for p in map(json.loads, z.read('case-evidence.jsonl').decode().splitlines())}
    development = {r['case_id'] for r in rows if r['partition'] == 'development'}
    training = {r['case_id'] for r in rows if r['partition'] == 'train'}
    assert len(development) == 21 and len(training) == 101 and not development & training
    source = root / 'nemotron_4b_sample_cases_v1'
    source_config = json.loads((source / 'run_manifest.json').read_text())
    cases = []
    for case_id in sorted(development):
        record = json.loads((source / 'development' / f'{case_id}.json').read_text())
        assert record['case_id'] == case_id and record['partition'] == 'development'
        assert record['run_hash'] == canonical_hash(source_config)
        assert canonical_hash(record['messages']) == record['input_sha256']
        assert record['messages'][0] == {'role': 'system', 'content': source_config['system_prompt']}
        assert [m['role'] for m in record['messages']] == ['system', 'user']
        assert json.loads(record['messages'][1]['content']) == packets[case_id]
        cases.append({key: record[key] for key in ['case_id', 'messages', 'input_sha256', 'input_tokens']})
    return cases, source_config


class ResultStore:
    def __init__(self, directory, manifest):
        # JSON round trips tuples in the case schedule into lists.
        manifest = json.loads(json.dumps(manifest))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_hash = canonical_hash(manifest)
        path = self.directory / 'run_manifest.json'
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError('Run configuration changed. Choose a new RUN_NAME; existing results were not overwritten.')
        else:
            atomic_json(path, manifest)

    def path(self, case_id, arm):
        return self.directory / arm / f'{case_id}.json'

    def is_done(self, case_id, arm, input_hash):
        path = self.path(case_id, arm)
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        if value['run_hash'] != self.run_hash or value['input_sha256'] != input_hash:
            raise ValueError('Saved result has a different run or input hash.')
        if value.get('status') == 'runtime_error':
            raise RuntimeError('A saved runtime error requires inspection; it will not be silently retried. See ' + str(path))
        return value.get('terminal', False)

    def save(self, record):
        assert record['run_hash'] == self.run_hash
        stamp = time.time_ns()
        atomic_json(self.directory / 'attempt_history' / f"{record['case_id']}-{record['arm']}-{stamp}.json", record)
        atomic_json(self.path(record['case_id'], record['arm']), record)

    def summary(self, cases):
        comparison, counts = [], {'untuned': Counter(), 'expert_lora': Counter()}
        for case in cases:
            row = {'case_id': case['case_id']}
            for arm in ['untuned', 'expert_lora']:
                path = self.path(case['case_id'], arm)
                result = json.loads(path.read_text()) if path.exists() else {'status': 'not_run'}
                counts[arm][result['status']] += 1
                annotation = result.get('annotation') or {}
                for key, value in {
                    'status': result['status'], 'broad_cell_type': annotation.get('broad_cell_type'),
                    'needs_expert_review': annotation.get('needs_expert_review'),
                    'output_tokens': result.get('output_tokens'), 'generation_seconds': result.get('generation_seconds'),
                }.items():
                    row[f'{arm}_{key}'] = value
            comparison.append(row)
        with (self.directory / 'comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
            writer.writeheader(); writer.writerows(comparison)
        summary = {'case_count': len(cases), 'expected_answers': len(cases) * 2,
                   'counts': {k: dict(v) for k, v in counts.items()},
                   'note': 'Completion counts only. Biological accuracy requires independent references and a frozen scoring mapping.'}
        atomic_json(self.directory / 'summary.json', summary)
        return summary


class MatchedRunner:
    def __init__(self, model, tokenizer, store, device='cuda', schema=None,
                 max_new_tokens=2048, max_seconds=90):
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        self.model, self.tokenizer, self.store = model, tokenizer, store
        self.device, self.schema = device, schema or OUTPUT_SCHEMA
        self.max_new_tokens, self.max_seconds = max_new_tokens, max_seconds
        print('Building the shared output-format tokenizer index once...', flush=True)
        self.tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

    def _sync(self):
        if self.device == 'cuda':
            import torch
            torch.cuda.synchronize()

    def run_one(self, case, arm):
        import torch
        import jsonschema
        from transformers import GenerationConfig
        from transformers.generation.streamers import BaseStreamer
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        assert arm in ['untuned', 'expert_lora']
        if self.store.is_done(case['case_id'], arm, case['input_sha256']):
            print(f"{case['case_id']} / {arm}: already saved", flush=True)
            return json.loads(self.store.path(case['case_id'], arm).read_text())
        inputs = self.tokenizer.apply_chat_template(case['messages'], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors='pt').to(self.device)
        n_input = inputs['input_ids'].shape[-1]
        assert n_input == case['input_tokens'], 'Input tokenization differs from baseline.'
        assert n_input + self.max_new_tokens <= self.model.config.max_position_embeddings
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, make_parser(self.schema))
        class ProgressRecorder(BaseStreamer):
            def __init__(self):
                self.first, self.tokens, self.started, self.last_print = True, [], time.perf_counter(), 0
            def put(self, value):
                if self.first:
                    self.first = False
                    return
                self.tokens.extend(value.reshape(-1).tolist())
                elapsed = time.perf_counter() - self.started
                if len(self.tokens) == 1 or len(self.tokens) % 32 == 0 or elapsed - self.last_print >= 10:
                    print(f'  {len(self.tokens)} tokens · {elapsed:.0f}s', flush=True)
                    self.last_print = elapsed
            def end(self):
                pass
        recorder = ProgressRecorder()
        self.model.eval()
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        self._sync()
        started = time.perf_counter()
        interrupted, error = False, None
        print(f"Starting {case['case_id']} / {arm} — {self.max_seconds}s limit", flush=True)
        try:
            context = self.model.disable_adapter() if arm == 'untuned' else nullcontext()
            with context, torch.inference_mode():
                generated = self.model.generate(**inputs,
                    generation_config=GenerationConfig(
                        max_new_tokens=self.max_new_tokens, max_time=self.max_seconds,
                        do_sample=False, num_beams=1, use_cache=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id),
                    prefix_allowed_tokens_fn=prefix_fn, streamer=recorder)
                assert generated[0, n_input:].tolist() == recorder.tokens
                del generated
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        self._sync()
        seconds = time.perf_counter() - started
        text = self.tokenizer.decode(recorder.tokens, skip_special_tokens=True)
        annotation, parse_error = None, None
        try:
            candidate = json.loads(text.strip())
            jsonschema.validate(candidate, self.schema)
            annotation = candidate
        except Exception as exc:
            parse_error = f'{type(exc).__name__}: {exc}'
        eos = bool(recorder.tokens) and recorder.tokens[-1] == self.tokenizer.eos_token_id
        status = ('interrupted' if interrupted else 'runtime_error' if error else
                  'complete' if eos and annotation is not None else
                  'valid_json_without_eos' if annotation is not None else
                  'time_limit' if not eos and seconds >= self.max_seconds else
                  'token_limit' if len(recorder.tokens) >= self.max_new_tokens else 'invalid_output')
        record = {'case_id': case['case_id'], 'partition': 'development', 'arm': arm,
                  'run_hash': self.store.run_hash, 'input_sha256': case['input_sha256'], 'messages': case['messages'],
                  'created_utc': datetime.now(timezone.utc).isoformat(), 'status': status,
                  'terminal': not interrupted, 'raw_text': text, 'annotation': annotation,
                  'parse_error': parse_error, 'runtime_error': error, 'stopped_at_eos': eos,
                  'output_token_ids': recorder.tokens, 'input_tokens': n_input, 'output_tokens': len(recorder.tokens),
                  'generation_seconds': seconds,
                  'explanation_over_100_words': len((annotation or {}).get('explanation','').split()) > 100,
                  'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated()/2**30 if self.device == 'cuda' else None}
        self.store.save(record)
        del inputs
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        print(f"Saved {case['case_id']} / {arm}: {status}, {len(recorder.tokens)} tokens, {seconds:.1f}s", flush=True)
        return record

    def run_all(self, cases, total_minutes=70):
        started = time.perf_counter()
        by_id = {c['case_id']: c for c in cases}
        schedule = case_schedule(by_id)
        for number, (case_id, arm) in enumerate(schedule, 1):
            if time.perf_counter() - started >= total_minutes * 60:
                print('Total session time limit reached. Saved results are safe; rerun this cell to resume.', flush=True)
                break
            print(f'\nAnswer {number}/{len(schedule)}', flush=True)
            result = self.run_one(by_id[case_id], arm)
            self.store.summary(cases)
            if result['status'] == 'interrupted':
                raise KeyboardInterrupt('Partial answer saved. Rerun this cell to resume.')
            if result['status'] == 'runtime_error':
                raise RuntimeError('Stopped to avoid wasting compute: ' + str(result['runtime_error']))
        summary = self.store.summary(cases)
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def export_results(store, destination):
    destination = Path(destination)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(store.directory.rglob('*')):
            if path.is_file() and path.suffix != '.tmp':
                z.write(path, path.relative_to(store.directory))
    return destination
