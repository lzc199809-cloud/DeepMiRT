#!/usr/bin/env python3
"""
All visualization functions.

Chart list:
- ROC curve (with AUROC annotation)
- PR curve (with AUPRC annotation)
- Confusion matrix (normalized heatmap)
- Evidence type comparison bar chart (AUROC/AUPRC per evidence_type + CI error bars)
- Negative sample tier comparison
- Calibration reliability diagram
- Frequency bias bar chart
- Prediction score distribution (positive/negative overlaid histogram)
- Threshold sensitivity curve
- Error analysis dashboard
- Multi-model ROC overlay
- Multi-model AUROC bar chart comparison
- Multi-model radar chart

All charts saved as PDF + PNG, DPI=300.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .metrics import compute_specificity

# Global style settings
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)


def _save_fig(fig: plt.Figure, output_dir: str, name: str) -> None:
    """Save figure as PDF + PNG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
    model_name: str = "Ours (RNA-FM)",
) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"{model_name} (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUROC = 0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir, "roc_curve")


def plot_pr_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
    model_name: str = "Ours (RNA-FM)",
) -> None:
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    auprc = average_precision_score(labels, probs)
    baseline = labels.mean()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, color="#2563EB", lw=2, label=f"{model_name} (AUPRC = {auprc:.4f})")
    ax.axhline(y=baseline, color="k", ls="--", lw=1, alpha=0.5, label=f"Baseline (prevalence = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir, "pr_curve")


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    output_dir: str,
    normalize: bool = True,
) -> None:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=1 if normalize else None)

    for i in range(2):
        for j in range(2):
            text = f"{cm_display[i, j]:.3f}\n({cm[i, j]:,})"
            ax.text(j, i, text, ha="center", va="center", fontsize=11,
                    color="white" if cm_display[i, j] > 0.5 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Normalized)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save_fig(fig, output_dir, "confusion_matrix")


def plot_evidence_type_comparison(
    stratified_metrics: dict[str, dict],
    output_dir: str,
    metric_names: list[str] | None = None,
) -> None:
    """Plot stratified metrics comparison bar chart by evidence type (with CI error bars)."""
    if metric_names is None:
        metric_names = ["AUROC", "AUPRC"]

    groups = list(stratified_metrics.keys())
    n_metrics = len(metric_names)
    x = np.arange(len(groups))
    width = 0.8 / n_metrics
    colors = ["#2563EB", "#F97316", "#10B981", "#EF4444"]

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 2), 6))
    for i, metric in enumerate(metric_names):
        vals = []
        errs_low = []
        errs_high = []
        for g in groups:
            m = stratified_metrics[g]
            if metric in m and isinstance(m[metric], tuple):
                point, ci_low, ci_high = m[metric]
                vals.append(point)
                errs_low.append(point - ci_low)
                errs_high.append(ci_high - point)
            elif metric in m:
                vals.append(m[metric])
                errs_low.append(0)
                errs_high.append(0)
            else:
                vals.append(0)
                errs_low.append(0)
                errs_high.append(0)

        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width, label=metric,
            color=colors[i % len(colors)], alpha=0.85,
            yerr=[errs_low, errs_high], capsize=3,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Metrics by Evidence Type")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, output_dir, "evidence_type_comparison")


def plot_negative_tier_comparison(
    tier_metrics: dict[str, dict],
    output_dir: str,
) -> None:
    """Plot negative sample tier (tier1 vs tier4) comparison chart."""
    plot_evidence_type_comparison(tier_metrics, output_dir, ["AUROC", "AUPRC", "MCC"])
    # Rename output files
    src = Path(output_dir)
    for ext in ["pdf", "png"]:
        old = src / f"evidence_type_comparison.{ext}"
        new = src / f"negative_tier_comparison.{ext}"
        if old.exists():
            old.rename(new)


def plot_calibration_reliability(
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
    n_bins: int = 15,
) -> None:
    """Plot calibration reliability diagram."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_accs.append(labels[mask].mean())
        bin_confs.append(probs[mask].mean())
        bin_counts.append(mask.sum())

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Upper panel: reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax1.bar(bin_confs, bin_accs, width=0.05, alpha=0.6, color="#2563EB", edgecolor="navy", label="Model")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Reliability Diagram")
    ax1.legend(loc="upper left")
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.grid(True, alpha=0.3)

    # Lower panel: sample count per bin
    ax2.bar(bin_confs, bin_counts, width=0.05, alpha=0.6, color="#6B7280")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Samples per Bin")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, output_dir, "calibration_reliability")


def plot_score_distribution(
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
) -> None:
    """Plot overlaid histogram of prediction score distributions for positive/negative samples."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        probs[labels == 0], bins=100, alpha=0.6, label="Negative",
        color="#EF4444", density=True,
    )
    ax.hist(
        probs[labels == 1], bins=100, alpha=0.6, label="Positive",
        color="#2563EB", density=True,
    )
    ax.axvline(x=0.5, color="k", ls="--", lw=1, alpha=0.5, label="Threshold = 0.5")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution (Positive vs Negative)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir, "score_distribution")


def plot_threshold_sensitivity(
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
) -> None:
    """Plot F1/Precision/Recall/Specificity vs threshold curves."""
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s, precs, recs, specs = [], [], [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1s.append(f1_score(labels, preds, zero_division=0))
        precs.append(
            preds.sum() and (labels[preds == 1].sum() / preds.sum()) or 0
        )
        recs.append(recall_score(labels, preds, zero_division=0))
        specs.append(compute_specificity(labels, preds))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s, label="F1", lw=2)
    ax.plot(thresholds, precs, label="Precision", lw=2)
    ax.plot(thresholds, recs, label="Recall (Sensitivity)", lw=2)
    ax.plot(thresholds, specs, label="Specificity", lw=2)
    ax.axvline(x=0.5, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sensitivity Analysis")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir, "threshold_sensitivity")


def plot_frequency_bias(
    quintile_metrics: dict[str, dict],
    output_dir: str,
) -> None:
    """Plot miRNA frequency bias bar chart (AUROC per quintile)."""
    quintiles = list(quintile_metrics.keys())
    aurocs = []
    auprcs = []
    auroc_errs = [[], []]
    auprc_errs = [[], []]

    for q in quintiles:
        m = quintile_metrics[q]
        if isinstance(m.get("AUROC"), tuple):
            pt, lo, hi = m["AUROC"]
            aurocs.append(pt)
            auroc_errs[0].append(pt - lo)
            auroc_errs[1].append(hi - pt)
        else:
            aurocs.append(m.get("AUROC", 0))
            auroc_errs[0].append(0)
            auroc_errs[1].append(0)

        if isinstance(m.get("AUPRC"), tuple):
            pt, lo, hi = m["AUPRC"]
            auprcs.append(pt)
            auprc_errs[0].append(pt - lo)
            auprc_errs[1].append(hi - pt)
        else:
            auprcs.append(m.get("AUPRC", 0))
            auprc_errs[0].append(0)
            auprc_errs[1].append(0)

    x = np.arange(len(quintiles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, aurocs, width, label="AUROC", color="#2563EB",
           yerr=auroc_errs, capsize=4, alpha=0.85)
    ax.bar(x + width / 2, auprcs, width, label="AUPRC", color="#F97316",
           yerr=auprc_errs, capsize=4, alpha=0.85)

    for i, (a, p) in enumerate(zip(aurocs, auprcs)):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, p + 0.01, f"{p:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(quintiles)
    ax.set_ylabel("Score")
    ax.set_title("Performance by miRNA Frequency Quintile (Q1=highest freq)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, output_dir, "frequency_bias")


def plot_error_analysis_dashboard(
    error_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Plot error analysis dashboard (multi-dimensional feature distribution of FP/FN).

    error_df must contain columns: error_type (FP/FN/TP/TN), prob, evidence_type,
                                   mirna_length, gc_content
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. FP/FN by evidence type
    ax = axes[0, 0]
    error_types = ["FP", "FN"]
    for i, et in enumerate(error_types):
        sub = error_df[error_df["error_type"] == et]
        if len(sub) > 0:
            counts = sub["evidence_type"].value_counts()
            ax.barh(
                [f"{et}: {k}" for k in counts.index],
                counts.values,
                color=["#EF4444", "#F97316"][i],
                alpha=0.7,
            )
    ax.set_xlabel("Count")
    ax.set_title("FP/FN Distribution by Evidence Type")
    ax.grid(axis="x", alpha=0.3)

    # 2. Confidence distribution
    ax = axes[0, 1]
    for et, color in [("FP", "#EF4444"), ("FN", "#F97316"), ("TP", "#10B981"), ("TN", "#6B7280")]:
        sub = error_df[error_df["error_type"] == et]
        if len(sub) > 0:
            ax.hist(sub["prob"], bins=50, alpha=0.5, label=et, color=color, density=True)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution by Error Type")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. miRNA length distribution
    ax = axes[1, 0]
    for et, color in [("FP", "#EF4444"), ("FN", "#F97316")]:
        sub = error_df[error_df["error_type"] == et]
        if len(sub) > 0 and "mirna_length" in sub.columns:
            ax.hist(sub["mirna_length"], bins=15, alpha=0.5, label=et, color=color, density=True)
    ax.set_xlabel("miRNA Length (nt)")
    ax.set_ylabel("Density")
    ax.set_title("miRNA Length Distribution (FP vs FN)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. GC content distribution
    ax = axes[1, 1]
    for et, color in [("FP", "#EF4444"), ("FN", "#F97316")]:
        sub = error_df[error_df["error_type"] == et]
        if len(sub) > 0 and "gc_content" in sub.columns:
            ax.hist(sub["gc_content"], bins=20, alpha=0.5, label=et, color=color, density=True)
    ax.set_xlabel("GC Content")
    ax.set_ylabel("Density")
    ax.set_title("GC Content Distribution (FP vs FN)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Error Analysis Dashboard", fontsize=16, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "error_analysis_dashboard")


def plot_multi_model_roc(
    model_results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str,
) -> None:
    """
    Multi-model ROC curve overlay.

    Args:
        model_results: {model_name: (labels, probs)}
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))

    for (name, (labels, probs)), color in zip(model_results.items(), colors):
        fpr, tpr, _ = roc_curve(labels, probs)
        auroc_val = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} ({auroc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-Model ROC Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir, "multi_model_roc")


def plot_multi_model_auroc_bar(
    comparison_df: pd.DataFrame,
    output_dir: str,
    metrics: list[str] | None = None,
) -> None:
    """Multi-model AUROC/AUPRC bar chart comparison."""
    if metrics is None:
        metrics = ["AUROC", "AUPRC"]

    models = comparison_df["Method"].tolist()
    n_metrics = len(metrics)
    x = np.arange(len(models))
    width = 0.8 / n_metrics
    colors = ["#2563EB", "#F97316", "#10B981", "#EF4444"]

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 6))
    for i, metric in enumerate(metrics):
        vals = comparison_df[metric].values
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=metric,
                      color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Multi-Model Performance Comparison")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_dir, "multi_model_auroc_bar")


def plot_multi_model_radar(
    comparison_df: pd.DataFrame,
    output_dir: str,
    metrics: list[str] | None = None,
    max_models: int = 8,
) -> None:
    """
    Multi-model multi-dimensional metrics radar chart comparison.

    Args:
        comparison_df: DataFrame containing Method and metric columns
        output_dir: output directory
        metrics: radar chart dimensions
        max_models: maximum number of models to display (top by AUROC)
    """
    if metrics is None:
        metrics = ["AUROC", "AUPRC", "F1", "MCC", "Sensitivity", "Specificity"]

    # Filter valid columns
    available = [m for m in metrics if m in comparison_df.columns]
    if len(available) < 3:
        return

    # Sort by AUROC, take top N
    df = comparison_df.dropna(subset=available).copy()
    if len(df) > max_models:
        df = df.nlargest(max_models, "AUROC")

    num_vars = len(available)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[m] if not np.isnan(row[m]) else 0 for m in available]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=row["Method"],
                color=colors[idx], markersize=4)
        ax.fill(angles, values, alpha=0.05, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Model Radar Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    _save_fig(fig, output_dir, "multi_model_radar")
