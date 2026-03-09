#!/usr/bin/env python3
"""
Error analysis: FP/FN characterization, seed match analysis.

6a. Systematic FP/FN analysis:
- Distribution by evidence type
- Distribution by miRNA length
- By GC content / sequence complexity
- Confidence distribution: high-confidence errors (FP with prob>0.9, FN with prob<0.1)

6b. Seed match analysis (following Mimosa):
- canonical seed types: 8mer, 7mer-m8, 7mer-A1, 6mer
- vs non-canonical
- Are FNs concentrated at non-canonical sites?
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


# ── Seed match detection ──


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of an RNA sequence."""
    complement = {"A": "U", "U": "A", "C": "G", "G": "C"}
    return "".join(complement.get(c, "N") for c in reversed(seq.upper()))


def _to_rna(seq: str) -> str:
    """DNA -> RNA conversion."""
    return seq.upper().replace("T", "U")


def find_seed_match(mirna_seq: str, target_seq: str) -> str:
    """
    Detect the seed match type between miRNA seed region and target sequence.

    miRNA seed region = positions 2-8 (counting from the 5' end, 1-indexed).
    Matching on the target requires detecting the reverse complement.

    Seed match hierarchy (strongest to weakest):
    - 8mer:     positions 2-8 fully matched + position 1 opposite is A
    - 7mer-m8:  positions 2-8 fully matched
    - 7mer-A1:  positions 2-7 matched + position 1 opposite is A
    - 6mer:     positions 2-7 matched
    - non-canonical: none of the above

    Note: The target sequence is a 3'UTR fragment; seed match requires finding the reverse complement.

    Args:
        mirna_seq: miRNA sequence (DNA or RNA notation)
        target_seq: Target 3'UTR sequence (DNA or RNA notation)

    Returns:
        Seed match type string
    """
    mirna_rna = _to_rna(mirna_seq)
    target_rna = _to_rna(target_seq)

    if len(mirna_rna) < 8:
        return "too_short"

    # miRNA seed: positions 2-8 (0-indexed: [1:8])
    seed_8 = mirna_rna[1:8]  # positions 2-8, 7nt
    seed_7 = mirna_rna[1:7]  # positions 2-7, 6nt

    # Find the reverse complement of the seed on the target
    seed_8_rc = _reverse_complement(seed_8)  # 7nt
    seed_7_rc = _reverse_complement(seed_7)  # 6nt

    # Search in the target sequence
    has_8_match = seed_8_rc in target_rna
    has_7_match = seed_7_rc in target_rna

    if has_8_match:
        # Check 8mer: positions 2-8 matched + A immediately downstream of the match site at 3' end
        # Find match position
        pos = target_rna.find(seed_8_rc)
        # 8mer requires the 3' end of the match region (i.e., the next position on the target) to be A
        if pos + len(seed_8_rc) < len(target_rna):
            downstream = target_rna[pos + len(seed_8_rc)]
            if downstream == "A":
                return "8mer"
        return "7mer-m8"

    if has_7_match:
        # Check 7mer-A1: positions 2-7 matched + A immediately downstream of the match site at 3' end
        pos = target_rna.find(seed_7_rc)
        if pos + len(seed_7_rc) < len(target_rna):
            downstream = target_rna[pos + len(seed_7_rc)]
            if downstream == "A":
                return "7mer-A1"
        return "6mer"

    return "non-canonical"


def classify_seed_matches(df: pd.DataFrame) -> pd.Series:
    """
    Classify seed match types for all positive samples in the DataFrame.

    Args:
        df: Must contain mirna_seq, target_fragment_40nt columns

    Returns:
        pd.Series: Seed match types
    """
    return df.apply(
        lambda row: find_seed_match(row["mirna_seq"], row["target_fragment_40nt"]),
        axis=1,
    )


# ── Error classification ──


def classify_errors(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Classify prediction results as TP/TN/FP/FN.

    Args:
        df: Prediction DataFrame (containing label, prob columns)
        threshold: Classification threshold

    Returns:
        A copy of df with a new error_type column
    """
    df = df.copy()
    preds = (df["prob"] >= threshold).astype(int)

    conditions = [
        (df["label"] == 1) & (preds == 1),  # TP
        (df["label"] == 0) & (preds == 0),  # TN
        (df["label"] == 0) & (preds == 1),  # FP
        (df["label"] == 1) & (preds == 0),  # FN
    ]
    labels = ["TP", "TN", "FP", "FN"]

    df["error_type"] = np.select(conditions, labels, default="Unknown")
    return df


def compute_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sequence features: miRNA length, GC content.
    """
    df = df.copy()
    df["mirna_length"] = df["mirna_seq"].str.len()
    df["gc_content"] = df["mirna_seq"].apply(
        lambda s: (s.upper().count("G") + s.upper().count("C")) / len(s) if len(s) > 0 else 0
    )
    df["target_gc_content"] = df["target_fragment_40nt"].apply(
        lambda s: (s.upper().count("G") + s.upper().count("C")) / len(s) if len(s) > 0 else 0
    )
    return df


# ── Main analysis pipeline ──


def run_error_analysis(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict:
    """
    Full error analysis pipeline.

    Args:
        df: Prediction DataFrame

    Returns:
        dict: Error analysis results
    """
    results = {}

    # 1. Error classification
    df = classify_errors(df, threshold)
    df = compute_sequence_features(df)

    error_counts = df["error_type"].value_counts().to_dict()
    results["error_counts"] = error_counts
    logger.info(f"Error distribution: {error_counts}")

    # 2. High-confidence errors
    high_conf_fp = df[(df["error_type"] == "FP") & (df["prob"] > 0.9)]
    high_conf_fn = df[(df["error_type"] == "FN") & (df["prob"] < 0.1)]
    results["high_confidence_errors"] = {
        "FP_prob_gt_0.9": len(high_conf_fp),
        "FN_prob_lt_0.1": len(high_conf_fn),
        "FP_prob_gt_0.9_pct": len(high_conf_fp) / max(error_counts.get("FP", 1), 1) * 100,
        "FN_prob_lt_0.1_pct": len(high_conf_fn) / max(error_counts.get("FN", 1), 1) * 100,
    }

    # 3. FP/FN by evidence type
    for et in ["FP", "FN"]:
        sub = df[df["error_type"] == et]
        if len(sub) > 0:
            results[f"{et}_by_evidence"] = sub["evidence_type"].value_counts().to_dict()

    # 4. FP/FN by miRNA length
    for et in ["FP", "FN"]:
        sub = df[df["error_type"] == et]
        if len(sub) > 0:
            results[f"{et}_mirna_length_stats"] = {
                "mean": float(sub["mirna_length"].mean()),
                "std": float(sub["mirna_length"].std()),
                "median": float(sub["mirna_length"].median()),
            }

    # 5. FP/FN by GC content
    for et in ["FP", "FN"]:
        sub = df[df["error_type"] == et]
        if len(sub) > 0:
            results[f"{et}_gc_content_stats"] = {
                "mean": float(sub["gc_content"].mean()),
                "std": float(sub["gc_content"].std()),
            }

    # 6. Error analysis DataFrame (for visualization)
    results["error_df"] = df

    return results


def run_seed_match_analysis(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict:
    """
    Seed match analysis (following Mimosa).

    Group positive samples by seed match type, analyze whether FNs are concentrated at non-canonical sites.

    Returns:
        dict: Seed match analysis results
    """
    results = {}

    # Only analyze positive samples (and FNs)
    positives = df[df["label"] == 1].copy()
    logger.info(f"Analyzing seed match for {len(positives)} positive samples...")

    positives["seed_type"] = classify_seed_matches(positives)
    seed_dist = positives["seed_type"].value_counts().to_dict()
    results["seed_type_distribution"] = seed_dist
    logger.info(f"Seed type distribution: {seed_dist}")

    # canonical vs non-canonical
    canonical_types = {"8mer", "7mer-m8", "7mer-A1", "6mer"}
    positives["is_canonical"] = positives["seed_type"].isin(canonical_types)

    n_canonical = positives["is_canonical"].sum()
    n_noncanonical = (~positives["is_canonical"]).sum()
    results["canonical_count"] = int(n_canonical)
    results["noncanonical_count"] = int(n_noncanonical)

    # Compute prediction performance by seed type
    preds = (positives["prob"] >= threshold).astype(int)
    positives["pred"] = preds

    for seed_type in ["canonical", "non-canonical"] + list(canonical_types):
        if seed_type == "canonical":
            mask = positives["is_canonical"]
        elif seed_type == "non-canonical":
            mask = ~positives["is_canonical"]
        else:
            mask = positives["seed_type"] == seed_type

        sub = positives[mask]
        if len(sub) == 0:
            continue

        recall = sub["pred"].mean()
        results[f"recall_{seed_type}"] = float(recall)
        results[f"n_{seed_type}"] = len(sub)
        logger.info(
            f"  {seed_type}: n={len(sub)}, recall={recall:.4f}"
        )

    # Proportion of non-canonical among FNs
    fn_mask = (positives["label"] == 1) & (preds == 0)
    fn_samples = positives[fn_mask]
    if len(fn_samples) > 0:
        fn_noncanonical_pct = (~fn_samples["is_canonical"]).mean() * 100
        results["FN_noncanonical_pct"] = float(fn_noncanonical_pct)
        logger.info(
            f"FN non-canonical percentage: {fn_noncanonical_pct:.1f}%"
        )

    # Seed match analysis table (for CSV output)
    rows = []
    for seed_type in sorted(seed_dist.keys()):
        sub = positives[positives["seed_type"] == seed_type]
        row = {
            "seed_type": seed_type,
            "count": len(sub),
            "pct_of_positives": len(sub) / len(positives) * 100,
            "recall": float(sub["pred"].mean()) if len(sub) > 0 else 0,
            "mean_prob": float(sub["prob"].mean()) if len(sub) > 0 else 0,
        }
        rows.append(row)

    results["seed_match_table"] = pd.DataFrame(rows)

    return results
