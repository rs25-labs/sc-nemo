import unittest
from score_matched_development import normalize_label, arm_metrics


class ScoringTests(unittest.TestCase):
    def test_overbroad_and_mixed_labels_are_not_specific_lineages(self):
        self.assertNotEqual(normalize_label('immune cell'), 'myeloid')
        self.assertNotEqual(normalize_label('neural crest-derived'), 'schwann_glial')
        self.assertEqual(normalize_label('Tentative: T/NK cells'), 'mixed_t_nk')

    def test_no_accepted_answers_has_undefined_accuracy(self):
        rows = [dict(resolved=True, correct=True, review=True, complete=True,
                     reference_group='myeloid', predicted_group='myeloid',
                     seconds=1., tokens=20, peak_gib=1.)]
        m = arm_metrics(rows)
        self.assertIsNone(m['accepted_resolved_accuracy'])
        self.assertEqual(m['accepted_macro_f1'], 0)

    def test_failure_kept_in_denominator_and_uncertain_acceptance_separate(self):
        rows = [dict(resolved=True, correct=False, review=True, complete=False,
                     reference_group='myeloid', predicted_group='invalid',
                     seconds=90., tokens=20, peak_gib=1.),
                dict(resolved=False, correct=False, review=False, complete=True,
                     reference_group=None, predicted_group='myeloid',
                     seconds=1., tokens=20, peak_gib=1.)]
        m = arm_metrics(rows)
        self.assertEqual(m['proposed_accuracy'], 0)
        self.assertEqual(m['accepted_expert_uncertain'], 1)
        self.assertEqual(m['accepted_wrong_resolved'], 0)
        self.assertEqual(m['completed'], 1)


if __name__ == '__main__':
    unittest.main()
