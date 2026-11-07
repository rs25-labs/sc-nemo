"""Read the returned expert workbook without modifying it or inferring new labels."""
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/sample-baseline-review-20261101'
SOURCE = BASE / 'reviewer-package/01-independent-bio-review-expert.xlsx'
OUT = BASE / 'independent-expert-ledger'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
FIELDS = [
    'Broad cell identity', 'Subtype / state (or null)', 'Supporting genes / evidence',
    'Conflicting / missing evidence (or none)', 'Alternative interpretations (or none)',
    'Broad identity needs further review?', 'Explanation (brief)',
    'Additional evidence requested (optional)', 'References / tools consulted (optional)',
    'Review time, minutes (optional)', 'Assessment complete?',
]


def main():
    OUT.mkdir(exist_ok=True)
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with (ROOT / 'outputs/sample-cases-v1/case_manifest.csv').open() as f:
        expected = {r['case_id']: r for r in csv.DictReader(f)}
    packets = {p['case_id']: p for p in map(json.loads, (ROOT / 'outputs/sample-cases-v1/case-evidence.jsonl').read_text().splitlines())}
    with zipfile.ZipFile(SOURCE) as z:
        assert z.testzip() is None
        shared = [''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        def read_sheet(number):
            result = {}
            for row in ET.fromstring(z.read(f'xl/worksheets/sheet{number}.xml')).findall('.//s:sheetData/s:row', NS):
                for cell in row:
                    value = cell.find('s:v', NS)
                    if value is not None:
                        result[cell.attrib['r']] = shared[int(value.text)] if cell.attrib.get('t') == 's' else value.text
                    elif cell.attrib.get('t') == 'inlineStr':
                        result[cell.attrib['r']] = ''.join(cell.find('s:is', NS).itertext())
            return result
        intro = read_sheet(1)
        reviewer = intro.get('B3') or None
        review_date = intro.get('B4') or None
        records = []
        for number, sheet_name, partition in [(2, 'Train cases', 'train'), (3, 'Development cases', 'development')]:
            cells = read_sheet(number)
            for address, case_id in cells.items():
                if not address.startswith('A') or not re.fullmatch('NB1-[a-f0-9]{12}', str(case_id)):
                    continue
                row = int(address[1:])
                assert case_id in expected and expected[case_id]['partition'] == partition
                fields, refs = {}, {}
                for offset, field in enumerate(FIELDS, 3):
                    assert cells[f'A{row+offset}'] == field
                    refs[field] = f'B{row+offset}'
                    fields[field] = cells.get(refs[field]) or None
                required = FIELDS[:7] + [FIELDS[-1]]
                assert all(fields[k] is not None and str(fields[k]).strip() for k in required), case_id
                assert fields['Assessment complete?'] == 'Yes', case_id
                review = fields['Broad identity needs further review?']
                assert review in ['Yes', 'No', 'Insufficient evidence']
                subtype_raw = fields['Subtype / state (or null)']
                subtype = None if str(subtype_raw).strip().lower() == 'null' else subtype_raw
                records.append({
                    'case_id': case_id, 'partition': partition,
                    'record_stage': 'independent_expert_annotation_before_model_correction',
                    'source_workbook': str(SOURCE.relative_to(ROOT)), 'source_sha256': source_hash,
                    'source_sheet': sheet_name, 'source_case_header': f'A{row}', 'field_cells': refs,
                    'reviewer_name': reviewer, 'review_date': review_date,
                    'reviewer_alias': 'Expert reviewer A',
                    'original_expert_fields': fields,
                    'broad_identity_verbatim': fields['Broad cell identity'],
                    'subtype_or_state_raw': subtype_raw, 'subtype_or_state': subtype,
                    'subtype_status': 'not_assigned' if subtype is None else 'tentative_under_reviewer_global_qualification',
                    'review_decision_verbatim': review,
                    'requires_further_review': review != 'No',
                    'requires_further_review_mapping': 'Yes and Insufficient evidence map to review; original categories preserved.',
                    'qualification_provenance': {
                        'source': 'User report accompanying returned workbook in this chat',
                        'statement': 'Subtype identification is tentative given the evidence.',
                        'scope': 'All non-null subtype/state suggestions; do not treat as confirmed hard labels.',
                    },
                    'evidence': packets[case_id],
                    'training_eligibility': 'candidate_pending_target_construction' if partition == 'train' else 'excluded_from_training_and_retrieval',
                    'limitations': ['Model correction feedback has not yet been collected for this case.',
                                    'Reviewer/date fields are blank.' if not reviewer or not review_date else 'Reviewer metadata recorded.',
                                    'Package withheld model outputs; independent-review completion is recorded, but a separate blinding attestation is absent.'],
                })
    assert len(records) == len({r['case_id'] for r in records}) == 122
    assert {r['case_id'] for r in records} == set(expected)
    records.sort(key=lambda r: (r['partition'] != 'train', r['case_id']))
    summary = {'source_workbook': str(SOURCE.relative_to(ROOT)), 'source_sha256': source_hash,
               'cases': len(records), 'all_required_fields_populated': True, 'all_marked_complete': True,
               'reviewer_name': reviewer, 'review_date': review_date,
               'subtype_qualification': 'All non-null subtypes/states remain tentative, per user report of reviewer qualification.',
               'partitions': {}}
    for partition in ['train', 'development']:
        subset = [r for r in records if r['partition'] == partition]
        (OUT / f'{partition}-independent-annotations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in subset))
        summary['partitions'][partition] = {
            'cases': len(subset),
            'broad_review_decisions': dict(Counter(r['review_decision_verbatim'] for r in subset)),
            'null_subtypes': sum(r['subtype_or_state'] is None for r in subset),
            'tentative_subtype_or_state_suggestions': sum(r['subtype_or_state'] is not None for r in subset),
        }
    (OUT / 'ingestion-audit.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'README.md').write_text('''# Independent expert annotations\n\nThe source workbook is unchanged and identified by SHA-256 in ingestion-audit.json and each record. All 122 cases match the frozen case manifest, have the required judgments populated and are marked complete. Original wording, field locations, and null subtype entries are preserved. Reviewer name/date and review times are blank in the returned workbook; no measured expert-time or named-reviewer provenance claim is supported.\n\nThe user relayed the reviewer's global qualification: subtype identification is tentative given the evidence. Non-null subtype/state suggestions are therefore retained as tentative; they are not confirmed hard labels and should not be used as definitive subtype accuracy targets. Null remains unassigned. Broad labels also retain their own unresolved/low-confidence wording and review decisions; do not silently coerce mixed or unresolved cases to one lineage.\n\n101 training records are candidates for target construction. The 21 development records are stored separately and excluded from fine-tuning and retrieval. No final test annotations are present. These files are independent expert judgments, not completed model-correction records or ready-to-run SFT examples. Supporting/conflicting evidence is prose and has not been automatically turned into a definitive gene list. A future broad-label evaluation mapping must preserve granularity and ambiguity, be versioned before scoring, and not be changed to improve model results.\n\nNext: obtain training-case correction feedback on the model answers, preserve the initial independent judgments, then construct uncertainty-aware supervised targets. Validation references remain separate. No adapter has been trained by this ingestion step.\n''')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
