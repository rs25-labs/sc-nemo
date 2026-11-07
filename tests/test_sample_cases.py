import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


class SampleCaseSafetyTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec('generate_sample_cases')
        self.assertIsNotNone(spec, 'Sample-specific case generator has not been implemented')
        import generate_sample_cases
        self.builder = generate_sample_cases

    def test_raw_extraction_excludes_other_patients_and_preserves_row_order(self):
        raw = sparse.csc_matrix(np.array([[1, 0, 2], [900, 800, 700], [0, 3, 0], [600, 500, 400]]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'tiny.h5ad'
            with h5py.File(path, 'w') as f:
                group = f.create_group('raw/X')
                group.attrs['shape'] = raw.shape
                for name in ['data', 'indices', 'indptr']:
                    group.create_dataset(name, data=getattr(raw, name))
            extracted = self.builder.extract_counts(path, np.array([2, 0]), block_genes=2)
            np.testing.assert_array_equal(extracted.toarray(), [[0, 3, 0], [1, 0, 2]])

    def test_test_cells_are_rejected_before_case_generation(self):
        frame = pd.DataFrame({'partition': ['train', 'test'], 'analysis_eligible': [True, True]})
        with self.assertRaisesRegex(ValueError, 'test'):
            self.builder.validate_selected_cells(frame)

    def test_panel_distinguishes_zero_expression_from_unavailable_gene(self):
        logcounts = sparse.csr_matrix([[0, np.log1p(2)], [0, 0]], dtype=np.float32)
        rows, missing = self.builder.panel_rows(logcounts, np.array(['ZERO', 'B']), ['ZERO', 'B', 'MISSING'])
        lookup = {r['gene']: r for r in rows}
        self.assertEqual(lookup['ZERO']['fraction_expressing'], 0)
        self.assertEqual(lookup['B']['fraction_expressing'], .5)
        self.assertIn({'gene': 'MISSING', 'reason': 'not_in_source_gene_metadata'}, missing)


if __name__ == '__main__':
    unittest.main()
