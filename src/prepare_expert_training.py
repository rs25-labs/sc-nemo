"""Create only training examples from independent expert judgments."""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/expert-training-v1'


def make_target(record):
    if record['partition'] != 'train':
        raise ValueError('Only training cases may enter the training package')
    fields = record['original_expert_fields']
    vocabulary = list(dict.fromkeys(g['gene'] for field in ['ranked_markers', 'marker_panel']
                                   for g in record['evidence'][field]
                                   if isinstance(g.get('gene'), str) and g['gene'].strip()))
    support = fields['Supporting genes / evidence']
    genes = [g for g in vocabulary if re.search(r'(?<![A-Za-z0-9_])' + re.escape(g) + r'(?![A-Za-z0-9_])', support)]
    subtype = record['subtype_or_state']
    if subtype and 'tentative' not in subtype.lower():
        subtype = 'Tentative: ' + subtype
    def prose_list(value):
        return [] if not value or value.strip().lower() in {'none', 'null'} else [value]
    target = {
        'broad_cell_type': fields['Broad cell identity'],
        'subtype_or_state': subtype,
        'supporting_genes': genes,
        'conflicting_or_missing_evidence': prose_list(fields['Conflicting / missing evidence (or none)']),
        'alternative_interpretations': prose_list(fields['Alternative interpretations (or none)']),
        'needs_expert_review': record['review_decision_verbatim'] != 'No',
        'explanation': fields['Explanation (brief)'],
    }
    assert len(target['explanation'].split()) <= 100
    return target


def assistant_labels(prompt_ids, full_ids, eos):
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('Chat template did not preserve the exact generation prefix')
    stop = full_ids.index(eos, len(prompt_ids)) + 1
    ids = full_ids[:stop]
    return ids, [-100] * len(prompt_ids) + ids[len(prompt_ids):]


def main():
    OUT.mkdir(exist_ok=True)
    ledger = ROOT / 'outputs/sample-baseline-review-20261101/independent-expert-ledger'
    records = [json.loads(line) for line in (ledger / 'train-independent-annotations.jsonl').read_text().splitlines()]
    dev = {json.loads(line)['case_id'] for line in (ledger / 'development-independent-annotations.jsonl').read_text().splitlines()}
    baseline = {r['case_id']: r for r in map(json.loads, (ROOT / 'outputs/sample-baseline-review-20261101/baseline-records-preserved.jsonl').read_text().splitlines())}
    examples, provenance = [], []
    for record in records:
        case_id = record['case_id']
        assert case_id not in dev
        original = baseline[case_id]
        assert original['partition'] == 'train'
        assert json.loads(original['messages'][1]['content']) == record['evidence']
        target = make_target(record)
        examples.append({'case_id': case_id, 'partition': 'train',
                         'messages': original['messages'] + [{'role': 'assistant', 'content': json.dumps(target)}]})
        provenance.append({'case_id': case_id, 'reviewer_alias': 'Expert reviewer A',
                           'source_sha256': record['source_sha256'], 'source_sheet': record['source_sheet'],
                           'field_cells': record['field_cells'], 'original_expert_fields': record['original_expert_fields'],
                           'input_sha256': original['input_sha256'],
                           'subtype_policy': 'Retain null or qualify every suggested subtype/state as tentative.',
                           'supporting_gene_policy': 'Literal gene-symbol matches in expert supporting prose, restricted to supplied marker/panel vocabulary; not a separately expert-approved gene list.'})
    assert len(examples) == len({r['case_id'] for r in examples}) == 101
    (OUT / 'expert-train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in examples))
    (OUT / 'target-provenance.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in provenance))
    shutil.copyfile(ROOT / 'src/expert_lora_training.py', OUT / 'expert_lora_training.py')
    (OUT / 'training-config.json').write_text(json.dumps({
        'model_id': 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16',
        'revision': 'dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f',
        'seed': 20261101, 'rank': 16, 'alpha': 32, 'dropout': 0.05,
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'],
        'epochs': 3, 'gradient_accumulation': 8, 'learning_rate': 0.0001,
        'max_sequence_length': 8192, 'training_wall_time_limit_minutes': 45,
        'eos_token_id': 11, 'training_examples': 101,
        'scope': 'Expert-annotation supervised LoRA arm. No development/test targets. Automatic-target matched arm remains required for the full study.',
    }, indent=2))
    (OUT / 'README.md').write_text('''# Expert-supervised LoRA package v1\n\n101 training cases only. No development or test examples or labels are included. Training inputs exactly match the saved baseline prompts. Baseline failures remain eligible for training on the expert answer. This is supervised training from independent expert annotations, not pairwise preference optimization or a completed second correction round.\n\nBroad labels and rationale are verbatim. Mixed/unresolved language remains intact. Every non-null subtype/state is qualified as tentative. Supporting genes are literal matches from the expert supporting-evidence prose against the provided gene vocabulary; see target-provenance.jsonl. Conflict and alternative prose remains intact as list entries. Yes and Insufficient evidence map to needs_expert_review=true; original categories remain in provenance. Reviewer alias: Expert reviewer A (user requested anonymity); name/date remain unrecorded.\n\nThe helper first performs a longest-example forward/backward check without changing weights, verifies finite nonzero adapter gradients and frozen base weights, and reports measured timing/memory. Full training is a separate explicit call. It trains attention/MLP adapters, not Mamba fused projections. No prompt tokens contribute to training loss; the assistant JSON and exact tokenizer EOS do. No evidence truncation or packing. The helper saves adapter/optimizer/RNG state after each optimizer update and resumes the same configuration after interruption. A 45-minute per-invocation wall-time cap saves partial progress; it is not a guarantee about total monetary cost.\n\nUse the existing A100 and loaded pinned base model. Install peft==0.17.1 with --no-deps to preserve the existing environment. GPU compatibility, training duration and peak memory must be verified in Colab; local checks only cover data integrity, token masking logic and code syntax. Do not claim training completed until its run status says complete.\n\nOriginal decoding baseline: 2048 max new tokens, greedy, cache disabled, reasoning off. The separate 8192 retry is diagnostic and not needed here. Subsequent matched model comparisons must use the same frozen input and decoding configuration. Subtype suggestions are not confirmed evaluation labels. Matched automatic-target training, retrieval/control comparisons and held-out test adjudication remain outstanding.\n\nSources checked November 1, 2026:\nhttps://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16\nhttps://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/modeling_nemotron_h.py\nhttps://huggingface.co/docs/peft/package_reference/lora\n''')
    names = ['expert-train.jsonl', 'target-provenance.jsonl', 'expert_lora_training.py', 'training-config.json', 'README.md']
    hashes = {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in names}
    (OUT / 'package-manifest.json').write_text(json.dumps({'files': hashes}, indent=2))
    with zipfile.ZipFile(OUT / 'expert-training-v1.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names + ['package-manifest.json']:
            z.write(OUT / name, name)
    print('Prepared 101 expert-supervised training examples. No development/test examples included.')


if __name__ == '__main__':
    main()
