import unittest
from evaluation_metrics import evaluate_policy


class FinalPolicyTests(unittest.TestCase):
    def test_failure_routes_to_review_but_is_not_uncertainty_recognition(self):
        refs={'a':{'group':None,'review':True},'b':{'group':'myeloid','review':False}}
        predictions={'a':{'status':'complete','annotation':{'broad_cell_type':'Macrophages','supporting_genes':[]}},
                     'b':{'status':'complete','annotation':{'broad_cell_type':'Macrophages','supporting_genes':[]}}}
        gates={'a':{'status':'time_limit','annotation':None},
               'b':{'status':'complete','annotation':{'needs_expert_review':False,'supporting_genes':[]}}}
        m,rows=evaluate_policy(predictions,refs,{'macrophages':'myeloid'},gates)
        self.assertEqual(m['explicit_review_true_positive'],0)
        self.assertEqual(m['operational_review_true_positive'],1)
        self.assertEqual(m['format_failure_routes'],1)
        self.assertEqual(m['safe_accepted_count'],1)

    def test_zero_coverage_accuracy_is_undefined_and_no_time_savings_invented(self):
        refs={'a':{'group':'myeloid','review':False}}
        pred={'a':{'status':'complete','annotation':{'broad_cell_type':'Macrophages','needs_expert_review':True,'supporting_genes':[]}}}
        m,_=evaluate_policy(pred,refs,{'macrophages':'myeloid'})
        self.assertIsNone(m['accepted_resolved_accuracy'])
        self.assertEqual(m['automatic_coverage'],0)
        self.assertNotIn('expert_time_saved',m)

    def test_disk_adapter_load_reuses_base_and_restores_active_adapter_after_baseline(self):
        import tempfile
        import torch
        from transformers import GPT2Config,GPT2LMHeadModel
        from peft import LoraConfig,get_peft_model
        model=get_peft_model(GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=16,n_embd=8,n_layer=1,n_head=1)),
            LoraConfig(task_type='CAUSAL_LM',target_modules=['c_attn'],r=2))
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            pointer=next(model.get_base_model().parameters()).data_ptr()
            model.load_adapter(d,adapter_name='frozen_v1',is_trainable=False)
            self.assertEqual(next(model.get_base_model().parameters()).data_ptr(),pointer)
            model.set_adapter('frozen_v1');model.requires_grad_(False);model.eval()
            with model.disable_adapter(),torch.inference_mode():
                output=model(torch.tensor([[1,2,3]]))
                self.assertTrue(torch.isfinite(output.logits).all())
            self.assertEqual(model.active_adapters,['frozen_v1'])


if __name__=='__main__':unittest.main()
