#!/usr/bin/env python3
"""
Lightweight unit tests that run without GPU or training data.

Run with:
    python -m pytest deepmirt/tests/test_unit.py -v
"""

import unittest


class TestPreprocessing(unittest.TestCase):
    """Test data preprocessing utilities."""

    def test_dna_to_rna(self):
        from deepmirt.data_module.preprocessing import dna_to_rna

        self.assertEqual(dna_to_rna("ATCGATCG"), "AUCGAUCG")
        self.assertEqual(dna_to_rna("AUCGAUCG"), "AUCGAUCG")
        self.assertEqual(dna_to_rna("atcg"), "AUCG")

    def test_validate_rna_sequence(self):
        from deepmirt.data_module.preprocessing import validate_rna_sequence

        self.assertTrue(validate_rna_sequence("AUCGAUCG", 5, 30))
        self.assertFalse(validate_rna_sequence("ATCG", 5, 30))  # contains T
        self.assertFalse(validate_rna_sequence("AU", 5, 30))  # too short


class TestPredictValidation(unittest.TestCase):
    """Test input validation in the public predict API."""

    def test_mismatched_lengths(self):
        from deepmirt.predict import predict

        with self.assertRaises(ValueError):
            predict(mirna_seqs=["AUGC"], target_seqs=["AUGC", "AUGC"])

    def test_empty_input(self):
        from deepmirt.predict import predict

        result = predict(mirna_seqs=[], target_seqs=[])
        self.assertEqual(len(result), 0)

    def test_validate_sequences_invalid_chars(self):
        from deepmirt.predict import _validate_sequences

        with self.assertRaises(ValueError):
            _validate_sequences(["AUGCXYZ"], ["AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCA"])

    def test_validate_sequences_empty(self):
        from deepmirt.predict import _validate_sequences

        with self.assertRaises(ValueError):
            _validate_sequences([""], ["AUGC"])

    def test_validate_sequences_cleaning(self):
        from deepmirt.predict import _validate_sequences

        mirna, target = _validate_sequences(
            ["  atcgatcg  "],
            ["AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCA"],
        )
        self.assertEqual(mirna[0], "ATCGATCG")
        self.assertEqual(target[0], "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCA")


class TestImports(unittest.TestCase):
    """Test that key modules can be imported without GPU."""

    def test_import_package(self):
        import deepmirt

        self.assertTrue(hasattr(deepmirt, "__version__"))
        self.assertEqual(deepmirt.__version__, "1.0.0")

    def test_import_predict_module(self):
        from deepmirt.predict import cli_main, predict, predict_from_csv

        self.assertTrue(callable(predict))
        self.assertTrue(callable(predict_from_csv))
        self.assertTrue(callable(cli_main))


if __name__ == "__main__":
    unittest.main()
