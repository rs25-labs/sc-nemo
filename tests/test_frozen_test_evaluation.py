import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class FrozenTestEvaluationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('frozen_test_evaluation'), 'Test-only evaluation helper must exist')
        import frozen_test_evaluation
        return frozen_test_evaluation

    def test_package_uses_expression_only_and_checks_input_hashes(self):
        api=self.api()
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return list(range(len(json.dumps(messages))))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            packet={'case_id':'NB1-test','marker_panel':[{'gene':'CHGA','fraction_expressing':1}]}
            (p/'case-evidence.jsonl').write_text(json.dumps(packet)+'\n')
            cfg={'case_ids':['NB1-test'],'identity_system':'Identity prompt','review_system':'Review prompt',
                 'files':{'case-evidence.jsonl':api.file_hash(p/'case-evidence.jsonl')}}
            groups=api.prepare_test_cases(p,cfg,Tokenizer())
            self.assertEqual(set(groups),{'identity','review_first'})
            for cases in groups.values():
                self.assertEqual([m['role'] for m in cases[0]['messages']],['system','user'])
                self.assertEqual(json.loads(cases[0]['messages'][1]['content']),packet)
            (p/'case-evidence.jsonl').write_text(json.dumps({**packet,'expert_answer':'label'})+'\n')
            with self.assertRaisesRegex(ValueError,'hash'):api.prepare_test_cases(p,cfg,Tokenizer())

    def test_package_rejects_reference_labels_even_if_hash_updated(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'case-evidence.jsonl').write_text(json.dumps({'case_id':'x','expert_answer':'T cells'})+'\n')
            cfg={'case_ids':['x'],'files':{'case-evidence.jsonl':api.file_hash(p/'case-evidence.jsonl')}}
            with self.assertRaisesRegex(ValueError,'fields'):api.prepare_test_cases(p,cfg,None)

    def test_stages_have_distinct_resume_identity_and_test_partition(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            a=api.TestResultStore(Path(d)/'identity',{'stage':'identity'})
            b=api.TestResultStore(Path(d)/'review_first',{'stage':'review_first'})
            record={'case_id':'x','arm':'untuned','partition':'development','input_sha256':'h',
                    'run_hash':a.run_hash,'status':'time_limit','terminal':True}
            a.save(record)
            saved=json.loads(a.path('x','untuned').read_text())
            self.assertEqual(saved['partition'],'test')
            self.assertEqual(saved['stage'],'identity')
            self.assertTrue(a.is_done('x','untuned','h'))
            self.assertFalse(b.is_done('x','untuned','h'))
            with self.assertRaisesRegex(ValueError,'input'):a.is_done('x','untuned','changed')


if __name__=='__main__':unittest.main()
