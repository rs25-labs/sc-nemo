"""Preserve returned independent test annotations without editing or relabeling."""
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/test-review-67-cases'
SOURCE=BASE/'01-independent-test-review-expert-feedback.xlsx'
TEMPLATE=BASE/'reviewer-package/01-independent-test-review.xlsx'
OUT=BASE/'independent-expert-ledger'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
FIELDS={
 'F':'Broad cell identity','G':'Subtype / state (optional)',
 'H':'Supporting genes / evidence','I':'Conflicting / missing evidence (or none)',
 'J':'Alternative interpretations (or none)','K':'Broad identity needs further review?',
 'L':'Explanation / evidence needed','M':'Assessment complete?','N':'References / tools (optional)',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_workbook(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        strings=[''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else []
        sheets={}
        relations={e.attrib['Id']:e.attrib['Target'] for e in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',NS):
            rel=sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target=relations[rel]
            target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            cells={}
            for c in ET.fromstring(z.read(target)).findall('.//s:sheetData/s:row/s:c',NS):
                assert c.find('s:f',NS) is None, 'Formula detected; independent expert fields must be literal values'
                v=c.find('s:v',NS)
                if c.attrib.get('t')=='inlineStr':value=''.join(c.find('s:is',NS).itertext())
                elif v is not None:value=strings[int(v.text)] if c.attrib.get('t')=='s' else v.text
                else:value=None
                cells[c.attrib['r']]=value
            sheets[sheet.attrib['name']]=cells
        return sheets


def main():
    source_hash=digest(SOURCE)
    original=read_workbook(TEMPLATE)
    returned=read_workbook(SOURCE)
    assert set(returned)=={'Instructions','Review'}
    sheet=returned['Review']
    for col,label in FIELDS.items():assert sheet[col+'6']==label
    expected=json.loads((BASE/'review-cases.json').read_text())
    with (ROOT/'outputs/reserved-test-cases-v1/case_manifest.csv').open() as f:
        metadata={r['case_id']:r for r in csv.DictReader(f)}
    assert len(expected)==len(metadata)==67
    actual_ids=[v for a,v in sheet.items() if re.fullmatch(r'A\d+',a) and re.fullmatch(r'NB1-[a-f0-9]{12}',str(v))]
    assert len(actual_ids)==len(set(actual_ids))==67 and set(actual_ids)==set(metadata)
    records=[]
    for number,expected_case in enumerate(expected,7):
        case_id=expected_case['case_id'];m=metadata[case_id]
        assert sheet['A'+str(number)]==case_id
        for col in 'ABCDE':
            actual=sheet.get(f'{col}{number}')
            baseline=original['Review'].get(f'{col}{number}')
            if col in 'CD':
                # Excel resaved integer numeric cells as e.g. 77.0, with identical values.
                assert Decimal(actual)==Decimal(baseline),f'Numeric metadata changed: {col}{number}'
            else:
                assert actual==baseline,f'Metadata changed: {col}{number}'
        fields={label:sheet.get(f'{col}{number}') for col,label in FIELDS.items()}
        refs={label:f'{col}{number}' for col,label in FIELDS.items()}
        assert all(str(sheet.get(f'{col}{number}') or '').strip() for col in 'FHIJKLM'),case_id
        assert fields[FIELDS['M']]=='Yes'
        decision=fields[FIELDS['K']]
        assert decision in ['Yes','No','Insufficient evidence']
        subtype_raw=fields[FIELDS['G']]
        subtype=None if subtype_raw is None or str(subtype_raw).strip().lower() in ['', 'null', 'none', 'n/a'] else subtype_raw
        records.append({
            'case_id':case_id,'partition':'test','participant_id':m['participant_id'],
            'sample_id':m['sample_id'],'cluster':m['cluster'],'cell_count':int(m['cells']),
            'reviewer_alias':'Expert reviewer A',
            'record_stage':'independent_test_expert_annotation_before_model_prediction',
            'source_workbook':str(SOURCE.relative_to(ROOT)), 'source_sha256':source_hash,
            'source_sheet':'Review','source_row':number,'field_cells':refs,
            'review_date_raw':returned['Instructions'].get('B4'),
            'original_expert_fields':fields,
            'broad_identity_verbatim':fields[FIELDS['F']],
            'subtype_or_state_raw':subtype_raw,'subtype_or_state':subtype,
            'subtype_evaluation_status':'not_assigned' if subtype is None else 'retained_as_suggestion_not_validated_as_confirmed_subtype',
            'review_decision_verbatim':decision,'requires_further_review':decision!='No',
            'requires_further_review_mapping':'Yes and Insufficient evidence map to review; original three categories retained.',
            'assessment_complete':True,
            'training_eligibility':'excluded_from_training_and_retrieval',
            'limitations':['Package withheld model answers and public reference labels. No separate blinding attestation supplied.',
                          'Broad identity does not establish malignancy or validate subtype/state.',
                          'Review date not supplied.' if not returned['Instructions'].get('B4') else 'Review date supplied.'],
        })
    decisions=Counter(r['review_decision_verbatim'] for r in records)
    audit={
        'ingested_utc':datetime.now(timezone.utc).isoformat(),
        'source_workbook':str(SOURCE.relative_to(ROOT)),'source_sha256':source_hash,
        'template_sha256':digest(TEMPLATE),
        'case_manifest_sha256':digest(ROOT/'outputs/reserved-test-cases-v1/case_manifest.csv'),
        'case_evidence_sha256':digest(ROOT/'outputs/reserved-test-cases-v1/case-evidence.jsonl'),
        'cases':len(records),'donors':len({r['participant_id'] for r in records}),
        'samples':len({r['sample_id'] for r in records}),'cells':sum(r['cell_count'] for r in records),
        'all_required_fields_populated':True,'all_marked_complete':True,'coordinator_metadata_unchanged':True,
        'review_decisions':dict(decisions),'no_further_review_cases':decisions['No'],
        'further_review_or_insufficient_cases':decisions['Yes']+decisions['Insufficient evidence'],
        'review_date_supplied':bool(returned['Instructions'].get('B4')),
        'broad_identity_counts_verbatim':dict(Counter(r['broad_identity_verbatim'] for r in records)),
        'model_inference_run':False,'model_scoring_run':False,
        'subtype_claim':'Suggestions preserved verbatim; not independently validated subtype or malignancy targets.',
        'source_workbook_unchanged':digest(SOURCE)==source_hash,
    }
    assert audit['cells']==42927
    OUT.mkdir(exist_ok=False)
    ledger=OUT/'test-independent-annotations.jsonl'
    ledger.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    audit['ledger_sha256']=digest(ledger)
    (OUT/'ingestion-audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    with (OUT/'case-review-summary.csv').open('w') as f:
        keys=['case_id','participant_id','sample_id','cluster','cell_count','broad_identity_verbatim','review_decision_verbatim','requires_further_review']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in records)
    (OUT/'README.md').write_text('''# Independent test expert annotations

All 67 returned cases reconcile to the frozen reserved-donor case inventory: four donors, six samples, 42,927 cells. All required judgments are populated and all cases are marked complete. Source workbook and original coordinator metadata remain unchanged.

Expert decisions: 57 No, seven Yes, three Insufficient evidence. For binary review scoring, Yes and Insufficient evidence map to review while the original categories are preserved. This yields 57 references not requesting further broad-identity review and ten requiring review or having insufficient evidence. No identity label mapping or model score is created by this ingestion.

The JSONL preserves original expert wording, optional subtype entries, source sheet/row/cell references and source hashes. Reviewer attribution remains Expert reviewer A. Review date is blank. No measured review-time claim is supported. Subtype/state suggestions and malignancy are not validated by broad-identity labels.

Test annotations are excluded from model training and retrieval. Do not put them in model inputs. The review package withheld model predictions and public labels; a separate blinding attestation was not supplied. Earlier investigator exposure to whole-cohort exploratory analysis remains a limitation of the reserved donor split.

No test inference or model scoring has been run. Before inference, lock the model/checkpoint, prompts, generation rules, handling of failures, broad-label mapping and metrics using the frozen development procedure. Report test results once without tuning against these answers. The existing frozen development artifacts remain unchanged.
''')
    assert digest(SOURCE)==source_hash
    print(json.dumps({k:audit[k] for k in ['cases','donors','samples','cells','review_decisions','all_required_fields_populated','all_marked_complete','source_workbook_unchanged','review_date_supplied']},indent=2))


if __name__=='__main__':main()
