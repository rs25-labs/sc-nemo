import importlib.util
import unittest
from pathlib import Path


class TrainingPreparationTests(unittest.TestCase):
    def api(self):
        spec = importlib.util.find_spec('prepare_expert_training')
        self.assertIsNotNone(spec, 'Training preparation module must exist')
        import prepare_expert_training
        return prepare_expert_training

    def fixture(self):
        return {
            'case_id': 'NB1-test', 'partition': 'train',
            'subtype_or_state': 'Cycling',
            'review_decision_verbatim': 'Insufficient evidence',
            'original_expert_fields': {
                'Broad cell identity': 'Mixed / unresolved',
                'Supporting genes / evidence': 'PECAM1 90%, VWF 80%; absent SOX10 is not discussed here.',
                'Conflicting / missing evidence (or none)': 'Additional cells needed',
                'Alternative interpretations (or none)': 'Endothelial and mural mixture',
                'Explanation (brief)': 'The population remains mixed.',
            },
            'evidence': {'marker_panel': [{'gene': 'PECAM1'}, {'gene': 'VWF'}], 'ranked_markers': [{'gene': ''}]},
        }

    def test_development_cannot_become_training(self):
        module = self.api()
        record = self.fixture(); record['partition'] = 'development'
        with self.assertRaisesRegex(ValueError, 'training'):
            module.make_target(record)

    def test_preserves_uncertainty_and_does_not_invent_genes(self):
        target = self.api().make_target(self.fixture())
        self.assertEqual(target['broad_cell_type'], 'Mixed / unresolved')
        self.assertEqual(target['subtype_or_state'], 'Tentative: Cycling')
        self.assertEqual(target['supporting_genes'], ['PECAM1', 'VWF'])
        self.assertTrue(target['needs_expert_review'])

    def test_masks_entire_prompt_and_keeps_eos(self):
        module = self.api()
        ids, labels = module.assistant_labels([4, 5, 6], [4, 5, 6, 9, 11, 8], 11)
        self.assertEqual(ids, [4, 5, 6, 9, 11])
        self.assertEqual(labels, [-100, -100, -100, 9, 11])
        with self.assertRaisesRegex(ValueError, 'prefix'):
            module.assistant_labels([4, 5], [4, 7, 9, 11], 11)


if __name__ == '__main__':
    unittest.main()
