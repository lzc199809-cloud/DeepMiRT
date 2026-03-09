#!/usr/bin/env python3
"""
Data deduplication and split leakage verification tests.

Verifies that train/val/test splits have no overlapping records,
no label conflicts, no intra-split duplicates, and balanced class distribution.

Run with:
    python -m pytest deepmirt/tests/test_split_leakage.py -v
    python -m unittest deepmirt.tests.test_split_leakage -v
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

DATA_DIR = Path(_PROJECT_ROOT) / "deepmirt" / "data" / "training"
CURRICULUM_DIR = DATA_DIR / "curriculum"
DEDUP_KEY = ["mirna_seq", "target_fragment_40nt"]


class TestSplitLeakage(unittest.TestCase):
    """Test suite for data deduplication and split integrity."""

    def test_no_cross_split_leakage(self):
        """Verify no overlapping records between train/val/test splits."""
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        val_df = pd.read_csv(DATA_DIR / "val.csv")
        test_df = pd.read_csv(DATA_DIR / "test.csv")

        train_set = set(zip(train_df["mirna_seq"], train_df["target_fragment_40nt"]))
        val_set = set(zip(val_df["mirna_seq"], val_df["target_fragment_40nt"]))
        test_set = set(zip(test_df["mirna_seq"], test_df["target_fragment_40nt"]))

        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        self.assertEqual(len(train_val_overlap), 0, f"train-val overlap: {len(train_val_overlap)} records")
        self.assertEqual(len(train_test_overlap), 0, f"train-test overlap: {len(train_test_overlap)} records")
        self.assertEqual(len(val_test_overlap), 0, f"val-test overlap: {len(val_test_overlap)} records")

    def test_no_label_conflicts(self):
        """Verify no conflicting labels for the same (mirna_seq, target_fragment_40nt) pair."""
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        val_df = pd.read_csv(DATA_DIR / "val.csv")
        test_df = pd.read_csv(DATA_DIR / "test.csv")

        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        grouped = combined_df.groupby(DEDUP_KEY)["label"].nunique()

        max_labels_per_pair = grouped.max()
        self.assertEqual(max_labels_per_pair, 1, f"Found pairs with {max_labels_per_pair} different labels")

    def test_no_intra_split_duplicates(self):
        """Verify no duplicate records within each split."""
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        val_df = pd.read_csv(DATA_DIR / "val.csv")
        test_df = pd.read_csv(DATA_DIR / "test.csv")

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            duplicates = split_df.duplicated(subset=DEDUP_KEY + ["label"], keep=False).sum()
            self.assertEqual(duplicates, 0, f"{split_name} has {duplicates} duplicate records")

    def test_balance_ratio(self):
        """Verify balanced class distribution across all splits."""
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        val_df = pd.read_csv(DATA_DIR / "val.csv")
        test_df = pd.read_csv(DATA_DIR / "test.csv")

        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        label_counts = combined_df["label"].value_counts()

        pos_count = (combined_df["label"] == 1).sum()
        neg_count = (combined_df["label"] == 0).sum()

        self.assertGreater(pos_count, 0, "No positive samples found")
        self.assertGreater(neg_count, 0, "No negative samples found")

        ratio = neg_count / pos_count
        self.assertGreaterEqual(ratio, 0.96, f"Negative/positive ratio {ratio:.4f} below 0.96")
        self.assertLessEqual(ratio, 1.04, f"Negative/positive ratio {ratio:.4f} above 1.04")

    def test_curriculum_no_leakage(self):
        """Verify no overlapping records between splits in curriculum tiers."""
        for tier in ["gold", "silver"]:
            tier_dir = CURRICULUM_DIR / tier
            train_df = pd.read_csv(tier_dir / "train.csv")
            val_df = pd.read_csv(tier_dir / "val.csv")
            test_df = pd.read_csv(tier_dir / "test.csv")

            train_set = set(zip(train_df["mirna_seq"], train_df["target_fragment_40nt"]))
            val_set = set(zip(val_df["mirna_seq"], val_df["target_fragment_40nt"]))
            test_set = set(zip(test_df["mirna_seq"], test_df["target_fragment_40nt"]))

            train_val_overlap = train_set & val_set
            train_test_overlap = train_set & test_set
            val_test_overlap = val_set & test_set

            self.assertEqual(len(train_val_overlap), 0,
                           f"{tier}: train-val overlap: {len(train_val_overlap)} records")
            self.assertEqual(len(train_test_overlap), 0,
                           f"{tier}: train-test overlap: {len(train_test_overlap)} records")
            self.assertEqual(len(val_test_overlap), 0,
                           f"{tier}: val-test overlap: {len(val_test_overlap)} records")

    def test_final_dataset_matches_splits(self):
        """Verify final_dataset.csv row counts match individual split files."""
        final_df = pd.read_csv(DATA_DIR / "final_dataset.csv")
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        val_df = pd.read_csv(DATA_DIR / "val.csv")
        test_df = pd.read_csv(DATA_DIR / "test.csv")

        final_train_count = (final_df["split"] == "train").sum()
        final_val_count = (final_df["split"] == "val").sum()
        final_test_count = (final_df["split"] == "test").sum()

        self.assertEqual(final_train_count, len(train_df),
                        f"final_dataset train rows {final_train_count} != train.csv {len(train_df)}")
        self.assertEqual(final_val_count, len(val_df),
                        f"final_dataset val rows {final_val_count} != val.csv {len(val_df)}")
        self.assertEqual(final_test_count, len(test_df),
                        f"final_dataset test rows {final_test_count} != test.csv {len(test_df)}")


if __name__ == '__main__':
    unittest.main()
