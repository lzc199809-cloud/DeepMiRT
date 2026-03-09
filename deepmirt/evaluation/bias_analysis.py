#!/usr/bin/env python3
"""
miRNA frequency bias analysis (miRBench method).

Reference: miRBench (Bioinformatics 2025):
1. Count the occurrence frequency of each miRNA in the training set
2. Divide into 5 quintiles by frequency (Q1=most frequent, Q5=least frequent)
3. Compute AUROC/AUPRC on the test set grouped by quintile

Core question: Is the performance on low-frequency miRNAs (Q5) significantly lower than on high-frequency miRNAs (Q1)?
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics, compute_metrics_with_ci

logger = logging.getLogger(__name__)


def compute_mirna_frequency(
    train_csv_path: str,
) -> pd.Series:
    """
    Count the occurrence frequency of each miRNA in the training set.

    Args:
        train_csv_path: Path to the training set CSV file

    Returns:
        pd.Series: mirna_name -> count
    """
    train_df = pd.read_csv(train_csv_path, usecols=["mirna_name"])
    freq = train_df["mirna_name"].value_counts()
    logger.info(
        f"miRNA frequency stats: {len(freq)} unique miRNAs, "
        f"median freq={freq.median():.0f}, max={freq.max()}, min={freq.min()}"
    )
    return freq


def assign_frequency_quintile(
    mirna_names: pd.Series,
    mirna_freq: pd.Series,
    n_quintiles: int = 5,
) -> pd.Series:
    """
    Assign miRNAs to quintiles based on training set frequency.

    Q1 = most frequent (highest occurrence count), Q5 = least frequent.
    miRNAs not seen in the training set are labeled as "Unseen".

    Args:
        mirna_names: miRNA names in the test set
        mirna_freq: Training set frequency statistics
        n_quintiles: Number of groups

    Returns:
        pd.Series: Quintile label for each sample
    """
    # Map frequencies to the test set
    freqs = mirna_names.map(mirna_freq).fillna(0)

    # Assign quintiles to non-zero frequencies
    quintiles = pd.Series(index=mirna_names.index, dtype=str)

    nonzero_mask = freqs > 0
    if nonzero_mask.sum() > 0:
        # Use unique miRNA frequencies to determine quintile boundaries
        unique_mirnas = mirna_names[nonzero_mask].unique()
        unique_freqs = pd.Series(
            {m: mirna_freq.get(m, 0) for m in unique_mirnas}
        )

        # Group by frequency: Q1=most frequent (top 20%), Q5=least frequent (bottom 20%)
        try:
            quintile_labels = pd.qcut(
                unique_freqs, n_quintiles,
                labels=[f"Q{i+1}" for i in range(n_quintiles)],
                duplicates="drop",
            )
        except ValueError:
            # If too few unique values to form n_quintiles groups
            quintile_labels = pd.cut(
                unique_freqs, n_quintiles,
                labels=[f"Q{i+1}" for i in range(n_quintiles)],
                duplicates="drop",
            )

        # Reverse: qcut defaults to Q1=lowest, but we want Q1=highest
        label_map = {
            f"Q{i+1}": f"Q{n_quintiles - i}"
            for i in range(n_quintiles)
        }
        quintile_labels = quintile_labels.map(label_map)

        # Build miRNA -> quintile mapping
        mirna_to_quintile = dict(zip(unique_mirnas, quintile_labels[unique_mirnas].values))

        for idx in mirna_names[nonzero_mask].index:
            mirna = mirna_names[idx]
            quintiles[idx] = mirna_to_quintile.get(mirna, "Unknown")

    # Unseen miRNAs
    quintiles[~nonzero_mask] = "Unseen"

    return quintiles


def evaluate_by_frequency_quintile(
    pred_df: pd.DataFrame,
    train_csv_path: str,
    n_bootstrap: int = 1000,
    n_quintiles: int = 5,
) -> dict[str, dict]:
    """
    Stratified evaluation by miRNA frequency quintile.

    Args:
        pred_df: Prediction DataFrame (containing mirna_name, label, prob)
        train_csv_path: Path to the training set CSV file
        n_bootstrap: Number of bootstrap iterations
        n_quintiles: Number of quintiles

    Returns:
        {quintile_label: {metric_name: (point, ci_low, ci_high)}}
    """
    # Compute frequencies
    mirna_freq = compute_mirna_frequency(train_csv_path)

    # Assign quintiles
    df = pred_df.copy()
    df["frequency_quintile"] = assign_frequency_quintile(
        df["mirna_name"], mirna_freq, n_quintiles
    )

    # Report statistics for each quintile
    for q in sorted(df["frequency_quintile"].unique()):
        sub = df[df["frequency_quintile"] == q]
        unique_mirnas = sub["mirna_name"].nunique()
        logger.info(
            f"  {q}: {len(sub)} samples, {unique_mirnas} unique miRNAs, "
            f"pos={sub['label'].sum()}, neg={(sub['label'] == 0).sum()}"
        )

    # Evaluate by quintile
    results = {}
    for q in sorted(df["frequency_quintile"].unique()):
        sub = df[df["frequency_quintile"] == q]
        labels = sub["label"].values
        probs = sub["prob"].values

        if len(sub) < 50 or labels.sum() == 0 or labels.sum() == len(labels):
            logger.warning(f"Skipping quintile {q}: insufficient data")
            continue

        metrics_ci = compute_metrics_with_ci(labels, probs, n_bootstrap=n_bootstrap)
        point_metrics = compute_all_metrics(labels, probs)
        results[q] = {**metrics_ci}
        for key in ["n_samples", "n_positive", "n_negative", "prevalence"]:
            results[q][key] = point_metrics[key]

    return results


def compute_frequency_summary_table(
    pred_df: pd.DataFrame,
    train_csv_path: str,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """
    Generate a miRNA frequency analysis summary table.

    Returns:
        DataFrame with columns:
            Quintile, n_samples, n_unique_mirna, mean_freq,
            min_freq, max_freq, AUROC, AUPRC
    """
    mirna_freq = compute_mirna_frequency(train_csv_path)
    df = pred_df.copy()
    df["frequency_quintile"] = assign_frequency_quintile(
        df["mirna_name"], mirna_freq, n_quintiles
    )
    df["mirna_freq"] = df["mirna_name"].map(mirna_freq).fillna(0)

    rows = []
    for q in sorted(df["frequency_quintile"].unique()):
        sub = df[df["frequency_quintile"] == q]
        labels = sub["label"].values
        probs = sub["prob"].values

        row = {
            "Quintile": q,
            "n_samples": len(sub),
            "n_unique_mirna": sub["mirna_name"].nunique(),
            "mean_freq": sub["mirna_freq"].mean(),
            "min_freq": sub["mirna_freq"].min(),
            "max_freq": sub["mirna_freq"].max(),
        }

        if labels.sum() > 0 and labels.sum() < len(labels):
            from sklearn.metrics import average_precision_score, roc_auc_score

            row["AUROC"] = roc_auc_score(labels, probs)
            row["AUPRC"] = average_precision_score(labels, probs)
        else:
            row["AUROC"] = np.nan
            row["AUPRC"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)
