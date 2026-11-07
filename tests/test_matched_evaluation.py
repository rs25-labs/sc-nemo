import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class MatchedEvaluationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('matched_evaluation'), 'Matched runner must exist')
        import matched_evaluation
        return matched_evaluation

    def test_balanced_schedule_pairs_every_case_once(self):
        m = self.api()
        schedule = m.case_schedule(['b', 'a', 'c'])
        self.assertEqual(schedule, [('a','untuned'),('a','expert_lora'),('b','expert_lora'),('b','untuned'),('c','untuned'),('c','expert_lora')])

    def test_resume_rejects_changed_protocol_and_preserves_failures(self):
        m = self.api()
        with tempfile.TemporaryDirectory() as directory:
            store = m.ResultStore(directory, {'setting':1})
            record = {'case_id':'case1','arm':'untuned','input_sha256':'abc',
                      'run_hash':store.run_hash,'status':'time_limit','terminal':True}
            store.save(record)
            self.assertTrue(store.is_done('case1','untuned','abc'))
            with self.assertRaisesRegex(ValueError,'input'):
                store.is_done('case1','untuned','different')
            with self.assertRaisesRegex(ValueError,'configuration'):
                m.ResultStore(directory, {'setting':2})

    def test_same_schema_is_used_for_both_arms(self):
        m = self.api()
        parser = m.make_parser()
        value = {'broad_cell_type':'Unresolved','subtype_or_state':None,'supporting_genes':[],
                 'conflicting_or_missing_evidence':[],'alternative_interpretations':[],
                 'needs_expert_review':True,'explanation':'Insufficient evidence.'}
        for char in json.dumps(value):
            self.assertIn(char,parser.get_allowed_characters())
            parser=parser.add_character(char)
        self.assertTrue(parser.can_end())

    def test_resume_accepts_saved_pair_schedule(self):
        m = self.api()
        with tempfile.TemporaryDirectory() as directory:
            manifest = {'case_order': m.case_schedule(['a','b'])}
            first = m.ResultStore(directory, manifest)
            second = m.ResultStore(directory, manifest)
            self.assertEqual(first.run_hash, second.run_hash)

    def test_cpu_model_integration(self):
        m = self.api()
        import torch
        from tokenizers import Tokenizer, models, decoders, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from peft import LoraConfig, get_peft_model
        # Byte-level toy vocabulary permits every JSON character without network downloads.
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        vocabulary = {s:i for i,s in enumerate(['<pad>','<eos>'] + alphabet)}
        backend=Tokenizer(models.BPE(vocabulary,merges=[]))
        backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder=decoders.ByteLevel()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,eos_token='<eos>',pad_token='<pad>')
        tokenizer.chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}Answer:{% endif %}"
        torch.manual_seed(7)
        base=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_embd=32,n_layer=1,n_head=2,n_positions=512,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id))
        model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,target_modules=['c_attn']),adapter_name='expert')
        model.eval()
        ids=torch.tensor([[4,5,6]])
        with torch.no_grad():
            for name,p in model.named_parameters():
                if 'lora_B' in name:p.normal_(std=0.2)
            trained=model(ids).logits
            with model.disable_adapter():untuned=model(ids).logits
        self.assertFalse(torch.allclose(trained,untuned))
        self.assertEqual(model.active_adapters,['expert'])
        case={'case_id':'toy','messages':[{'role':'user','content':'JSON:'}], 'input_sha256':'toy',
              'input_tokens':len(tokenizer.apply_chat_template([{'role':'user','content':'JSON:'}],tokenize=True,add_generation_prompt=True))}
        schema={'type':'object','properties':{'x':{'type':'boolean'}},'required':['x'],'additionalProperties':False}
        with tempfile.TemporaryDirectory() as directory:
            store=m.ResultStore(directory,{'test':True})
            runner=m.MatchedRunner(model,tokenizer,store,device='cpu',schema=schema,
                                   max_new_tokens=100,max_seconds=30)
            for arm in ['untuned','expert_lora']:
                record=runner.run_one(case,arm)
                self.assertEqual(record['status'],'complete',record)
                self.assertIsInstance(record['annotation']['x'],bool)
                self.assertTrue(store.is_done('toy',arm,'toy'))
                self.assertTrue(record['output_token_ids'])
            timed = m.MatchedRunner(model,tokenizer,store,device='cpu',schema=schema,
                                    max_new_tokens=100,max_seconds=1e-12)
            timeout_case={**case,'case_id':'timeout'}
            result=timed.run_one(timeout_case,'untuned')
            self.assertEqual(result['status'],'time_limit')
            self.assertTrue(result['output_token_ids'])
            self.assertTrue(store.is_done('timeout','untuned','toy'))
            summary=store.summary([case,timeout_case])
            self.assertEqual(summary['expected_answers'],4)
            self.assertEqual(summary['counts']['untuned']['time_limit'],1)
            self.assertEqual(summary['counts']['expert_lora']['not_run'],1)
            import zipfile
            archive=m.export_results(store,Path(directory).parent/'matched-toy-test.zip')
            try:
                with zipfile.ZipFile(archive) as z:
                    self.assertIsNone(z.testzip())
                    self.assertIn('comparison.csv',z.namelist())
                    self.assertIn('untuned/timeout.json',z.namelist())
            finally:
                archive.unlink()


if __name__=='__main__':
    unittest.main()
