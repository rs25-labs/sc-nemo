"""Independent CSV/scikit-learn scoring check and optional offline tokenizer audit."""
import argparse
import csv
import json
from pathlib import Path
from sklearn.metrics import f1_score
ROOT = Path(__file__).resolve().parents[1]


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError('Expected explicit boolean, got ' + repr(value))
    return value == 'True'


def verify_metrics(directory):
    directory = Path(directory)
    report = json.loads((directory / 'final-test-metrics.json').read_text())
    rows = list(csv.DictReader((directory / 'case-comparison.csv').open()))
    groups = list(report['reference_groups'])
    assert len(rows) == 268 and len(groups) == 9
    assert len({(r['condition'], r['case_id']) for r in rows}) == 268
    for condition, expected in report['metrics'].items():
        selected = [r for r in rows if r['condition'] == condition]
        resolved = [r for r in selected if boolean(r['resolved'])]
        accepted = [r for r in selected if not boolean(r['operational_review'])]
        accepted_resolved = [r for r in accepted if boolean(r['resolved'])]
        assert len(selected) == 67 and len(resolved) == 57
        truth = [r['reference_group'] for r in resolved]
        proposed = [r['proposed_group'] if boolean(r['identity_valid']) else 'invalid' for r in resolved]
        gated = [p if not boolean(r['operational_review']) else 'review' for p, r in zip(proposed, resolved)]
        for key, predictions in [('proposed_macro_f1', proposed), ('accepted_macro_f1', gated)]:
            assert abs(f1_score(truth, predictions, labels=groups, average='macro', zero_division=0) - expected[key]) < 1e-12
        for r in selected:
            assert boolean(r['correct']) == (boolean(r['resolved']) and boolean(r['identity_valid']) and r['proposed_group'] == r['reference_group'])
        counts = {
            'cases': len(selected), 'resolved_cases': len(resolved),
            'proposed_correct': sum(boolean(r['correct']) for r in resolved),
            'accepted': len(accepted), 'accepted_resolved': len(accepted_resolved),
            'safe_accepted_count': sum(boolean(r['correct']) for r in accepted),
            'wrong_resolved_accepted': sum(not boolean(r['correct']) for r in accepted_resolved),
            'expert_uncertain_accepted': sum(boolean(r['expert_review']) for r in accepted),
            'operational_review_routes': sum(boolean(r['operational_review']) for r in selected),
            'explicit_review_requests': sum(r['explicit_review'] == 'True' for r in selected),
            'explicit_review_true_positive': sum(boolean(r['expert_review']) and r['explicit_review'] == 'True' for r in selected),
            'explicit_review_false_positive': sum(not boolean(r['expert_review']) and r['explicit_review'] == 'True' for r in selected),
            'explicit_review_false_negative': sum(boolean(r['expert_review']) and r['explicit_review'] != 'True' for r in selected),
        }
        for key, value in counts.items():
            assert value == expected[key], (condition, key, value, expected[key])
        assert counts['safe_accepted_count'] + counts['wrong_resolved_accepted'] + counts['expert_uncertain_accepted'] == len(accepted)
        ratios = {
            'proposed_accuracy': (counts['proposed_correct'], len(resolved)),
            'automatic_coverage': (len(accepted), len(selected)),
            'safe_accepted_yield': (counts['safe_accepted_count'], len(selected)),
            'accepted_supported_fraction': (counts['safe_accepted_count'], len(accepted)),
            'accepted_resolved_accuracy': (counts['safe_accepted_count'], len(accepted_resolved)),
        }
        for key, (num, den) in ratios.items():
            actual = num / den if den else None
            assert actual == expected[key], (condition, key)
    result = {'conditions': len(report['metrics']), 'sklearn_macro_f1_reproduced': True,
              'accepted_macro_f1_reproduced': True, 'all_denominators_and_acceptance_categories_reconciled': True}
    (directory / 'independent-metrics-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def verify_tokens(directory, tokenizer_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=True)
    assert tokenizer.eos_token_id == 11
    directory = Path(directory)
    records = [json.loads(line) for line in (directory / 'predictions-preserved.jsonl').read_text().splitlines() if line.strip()]
    assert len(records) == 268
    assert len({(r['stage'], r['arm'], r['case_id']) for r in records}) == 268
    for r in records:
        ids = r['output_token_ids']
        assert tokenizer.decode(ids, skip_special_tokens=True).strip() == r['raw_text']
        prompt = tokenizer.apply_chat_template(r['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        assert len(prompt) == r['input_tokens'] and len(ids) == r['output_tokens']
        assert r['stopped_at_eos'] == bool(ids and ids[-1] == 11)
        if r['status'] == 'complete':
            assert ids[-1] == 11
    result = {'records_verified': len(records), 'includes_timeout': any(r['status'] == 'time_limit' for r in records),
              'all_raw_outputs_exactly_match_saved_tokens': True, 'all_input_token_counts_match_pinned_tokenizer': True,
              'gpu_used': False, 'model_weights_loaded': False}
    (directory / 'token-decode-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'outputs/final-test-results')
    parser.add_argument('--tokenizer-local', type=Path, help='Local snapshot for pinned revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f; no model weights loaded')
    args = parser.parse_args()
    print(json.dumps(verify_metrics(args.directory), indent=2))
    if args.tokenizer_local:
        print(json.dumps(verify_tokens(args.directory, args.tokenizer_local), indent=2))
