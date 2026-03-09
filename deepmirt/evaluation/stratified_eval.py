#!/usr/bin/env python3
"""
Stratified evaluation: compute metrics grouped by evidence type / data source / negative tier.

Dimension 2: Stratified by evidence_type
- experimental (general experimental validation)
- experimental_eCLIP (high-throughput eCLIP)
- experimental_CLASH (direct chimera capture, highest quality)
- synthetic_shuffled (synthetic negative samples)

Dimension 3: Negative tier analysis
- pos vs tier1 (experimental negatives)
- pos vs tier4 (shuffled negatives)
"""

from __future__ import annotations

import logging

import pandas as pd

from .metrics import compute_all_metrics, compute_metrics_with_ci

logger = logging.getLogger(__name__)


def evaluate_by_evidence_type(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> dict[str, dict]:
    """
    Compute all metrics + bootstrap CI stratified by evidence_type.

    Args:
        df: Prediction DataFrame, must contain label, prob, evidence_type columns
        n_bootstrap: Number of bootstrap iterations

    Returns:
        {evidence_type: {metric_name: (point, ci_low, ci_high)}}
    """
    results = {}

    # evidence_type may contain multiple types separated by semicolons (e.g., "experimental;experimental_CLASH")
    # Assign the primary evidence type for each sample
    df = df.copy()
    df["primary_evidence"] = df["evidence_type"].apply(_extract_primary_evidence)

    for etype in sorted(df["primary_evidence"].unique()):
        mask = df["primary_evidence"] == etype
        sub = df[mask]

        if len(sub) < 10:
            logger.warning(f"Skipping evidence_type={etype}: only {len(sub)} samples")
            continue

        labels = sub["label"].values
        probs = sub["prob"].values

        # Both positive and negative samples are required to compute AUROC
        if labels.sum() == 0 or labels.sum() == len(labels):
            logger.warning(
                f"Skipping evidence_type={etype}: only one class "
                f"(pos={labels.sum()}, neg={(labels == 0).sum()})"
            )
            continue

        logger.info(
            f"Evidence type '{etype}': {len(sub)} samples "
            f"(pos={labels.sum()}, neg={(labels == 0).sum()})"
        )

        metrics_ci = compute_metrics_with_ci(labels, probs, n_bootstrap=n_bootstrap)
        point_metrics = compute_all_metrics(labels, probs)

        # Merge point estimate metrics that do not have CI
        results[etype] = {**metrics_ci}
        for key in ["n_samples", "n_positive", "n_negative", "prevalence",
                     "Optimal_Threshold", "F1_at_Optimal", "Accuracy_at_Optimal"]:
            if key in point_metrics:
                results[etype][key] = point_metrics[key]

    return results


def _extract_primary_evidence(evidence_str: str) -> str:
    """
    Extract the primary type from an evidence_type string that may contain semicolons.

    Priority: experimental_CLASH > experimental_eCLIP > experimental > synthetic_shuffled
    """
    if not isinstance(evidence_str, str) or not evidence_str:
        return "unknown"

    parts = [p.strip() for p in evidence_str.split(";")]

    # Sort by priority
    priority = {
        "experimental_CLASH": 4,
        "experimental_eCLIP": 3,
        "experimental": 2,
        "synthetic_shuffled": 1,
    }

    best = "unknown"
    best_priority = 0
    for p in parts:
        if p in priority and priority[p] > best_priority:
            best = p
            best_priority = priority[p]

    return best if best != "unknown" else parts[0]


def evaluate_by_negative_tier(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> dict[str, dict]:
    """
    Negative tier analysis: compute performance for pos vs different types of negatives.

    tier1: Experimentally validated negatives (evidence_type contains experimental and label=0)
    tier4: Synthetic shuffled negatives (evidence_type is synthetic_shuffled and label=0)

    If tier4 AUROC is much higher than tier1, the model may have learned statistical
    features of "real vs shuffled" rather than true binding signals.
    """
    df = df.copy()
    df["primary_evidence"] = df["evidence_type"].apply(_extract_primary_evidence)

    positives = df[df["label"] == 1]
    results = {}

    # Tier 1: pos vs experimentally validated negatives
    exp_negatives = df[
        (df["label"] == 0)
        & df["primary_evidence"].isin(["experimental", "experimental_eCLIP", "experimental_CLASH"])
    ]
    if len(exp_negatives) > 0 and len(positives) > 0:
        tier1_df = pd.concat([positives, exp_negatives])
        labels = tier1_df["label"].values
        probs = tier1_df["prob"].values
        logger.info(
            f"Tier1 (pos vs experimental neg): {len(tier1_df)} samples "
            f"(pos={labels.sum()}, neg={(labels == 0).sum()})"
        )
        metrics_ci = compute_metrics_with_ci(labels, probs, n_bootstrap=n_bootstrap)
        point_metrics = compute_all_metrics(labels, probs)
        results["Tier1: pos vs exp_neg"] = {**metrics_ci}
        for key in ["n_samples", "n_positive", "n_negative"]:
            results["Tier1: pos vs exp_neg"][key] = point_metrics[key]

    # Tier 4: pos vs shuffled negatives
    shuffled_negatives = df[
        (df["label"] == 0)
        & (df["primary_evidence"] == "synthetic_shuffled")
    ]
    if len(shuffled_negatives) > 0 and len(positives) > 0:
        tier4_df = pd.concat([positives, shuffled_negatives])
        labels = tier4_df["label"].values
        probs = tier4_df["prob"].values
        logger.info(
            f"Tier4 (pos vs shuffled neg): {len(tier4_df)} samples "
            f"(pos={labels.sum()}, neg={(labels == 0).sum()})"
        )
        metrics_ci = compute_metrics_with_ci(labels, probs, n_bootstrap=n_bootstrap)
        point_metrics = compute_all_metrics(labels, probs)
        results["Tier4: pos vs shuffled_neg"] = {**metrics_ci}
        for key in ["n_samples", "n_positive", "n_negative"]:
            results["Tier4: pos vs shuffled_neg"][key] = point_metrics[key]

    # Overall: all samples
    labels = df["label"].values
    probs = df["prob"].values
    metrics_ci = compute_metrics_with_ci(labels, probs, n_bootstrap=n_bootstrap)
    point_metrics = compute_all_metrics(labels, probs)
    results["Overall"] = {**metrics_ci}
    for key in ["n_samples", "n_positive", "n_negative"]:
        results["Overall"][key] = point_metrics[key]

    return results


def evaluate_by_source_database(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> dict[str, dict]:
    """Compute metrics stratified by source_database."""
    results = {}

    for db in sorted(df["source_database"].dropna().unique()):
        mask = df["source_database"] == db
        sub = df[mask]

        if len(sub) < 50:
            continue

        labels = sub["label"].values
        probs = sub["prob"].values

        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        metrics_ci = compute_metrics_with_ci(labels, probs, n_bootstrap=n_bootstrap)
        point_metrics = compute_all_metrics(labels, probs)
        results[db] = {**metrics_ci}
        results[db]["n_samples"] = point_metrics["n_samples"]

    return results


def stratified_results_to_dataframe(
    results: dict[str, dict],
    group_name: str = "Group",
) -> pd.DataFrame:
    """Convert stratified evaluation results to a DataFrame for easy CSV export."""
    rows = []
    for group, metrics in results.items():
        row = {group_name: group}
        for key, val in metrics.items():
            if isinstance(val, tuple) and len(val) == 3:
                row[key] = val[0]
                row[f"{key}_CI_low"] = val[1]
                row[f"{key}_CI_high"] = val[2]
            else:
                row[key] = val
        rows.append(row)
    return pd.DataFrame(rows)
