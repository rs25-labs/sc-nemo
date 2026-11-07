import importlib.util
import unittest


class FrozenTestScoringTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('score_frozen_test'), 'Locked test scorer must exist')
        import score_frozen_test
        return score_frozen_test

    def test_gate_keeps_failures_and_uncertain_acceptance_separate(self):
        api=self.api()
        refs={'a':{'group':'t','review':False},'b':{'group':'t','review':False},'c':{'group':None,'review':True},'d':{'group':None,'review':True}}
        def pred(label,review=False):return {'status':'complete','annotation':{'broad_cell_type':label,'needs_expert_review':review}}
        ids={'a':pred('T cells'),'b':pred('B cells'),'c':pred('T cells'),'d':pred('T cells')}
        gate={'a':{'status':'time_limit'},'b':pred('ignored',True),'c':pred('ignored',False),'d':pred('ignored',True)}
        m,rows=api.score_policy(ids,refs,{'t cells':'t','b cells':'b'},['t'],gate)
        self.assertEqual(m['cases'],4)
        self.assertEqual(m['proposed_correct'],1)
        self.assertEqual(m['explicit_review_true_positive'],1)
        self.assertEqual(m['operational_review_routes'],3)
        self.assertEqual(m['format_failure_routes'],1)
        self.assertEqual(m['expert_uncertain_accepted'],1)
        self.assertEqual(m['safe_accepted_count'],0)
        self.assertEqual(m['wrong_identity_intercepted'],1)
        self.assertEqual(m['accepted_supported_fraction'],0)
        self.assertIsNone(m['accepted_resolved_accuracy'])

    def test_locked_map_does_not_strip_unlisted_qualifiers(self):
        api=self.api()
        refs={'a':{'group':'endothelial','review':False}}
        pred={'a':{'status':'complete','annotation':{'broad_cell_type':'Tentative: Endothelial cells','needs_expert_review':False}}}
        m,rows=api.score_policy(pred,refs,{'endothelial cells':'endothelial'},['endothelial'])
        self.assertEqual(m['proposed_correct'],0)
        self.assertEqual(m['unmapped_resolved_labels'],1)
        self.assertEqual(rows[0]['proposed_group'],'unmapped')

    def test_missing_attempts_stay_in_all_denominators(self):
        api=self.api()
        refs={'a':{'group':'t','review':False},'b':{'group':None,'review':True}}
        m,rows=api.score_policy({},refs,{},['t'])
        self.assertEqual(m['cases'],2)
        self.assertEqual(m['identity_complete'],0)
        self.assertEqual(m['proposed_accuracy'],0)
        self.assertEqual(m['explicit_review_true_positive'],0)
        self.assertEqual(m['operational_review_true_positive'],1)
        self.assertEqual(m['format_failure_routes'],2)
        self.assertEqual(m['proposed_macro_f1'],0)


if __name__=='__main__':unittest.main()
