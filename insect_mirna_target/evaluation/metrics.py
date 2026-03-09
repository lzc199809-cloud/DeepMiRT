#!/usr/bin/env python3
"""
Extended metrics computation library with bootstrap confidence intervals.

In addition to existing AUROC/AUPRC/Accuracy/F1, adds:
- Sensitivity (Recall), Specificity, Precision (PPV)
- MCC (Matthews Correlation Coefficient)
- Brier Score, ECE (Expected Calibration Error)
- Log Loss
- Optimal threshold (Youden's J)
All metrics include 95% bootstrap confidence intervals (n=1000).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_specificity(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute Specificity = TN / (TN + FP)."""
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_ece(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE).

    Divides predicted probabilities into n_bins equal-width bins and computes
    the weighted absolute difference between mean predicted probability and
    actual positive fraction in each bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
    return ece


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    Find the optimal classification threshold using Youden's J statistic.

    J = Sensitivity + Specificity - 1 = TPR - FPR

    Returns:
        dict with keys: threshold, f1, accuracy, sensitivity, specificity
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])

    preds = (probs >= best_threshold).astype(int)
    return {
        "threshold": best_threshold,
        "f1": float(f1_score(labels, preds)),
        "accuracy": float(accuracy_score(labels, preds)),
        "sensitivity": float(recall_score(labels, preds)),
        "specificity": float(compute_specificity(labels, preds)),
    }


def compute_all_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all classification metrics.

    Args:
        labels: ground truth labels (0/1)
        probs: predicted probabilities [0, 1]
        threshold: classification threshold

    Returns:
        dict: name-value mapping for all metrics
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    optimal = find_optimal_threshold(labels, probs)

    metrics = {
        "AUROC": float(roc_auc_score(labels, probs)),
        "AUPRC": float(average_precision_score(labels, probs)),
        "Accuracy": float(accuracy_score(labels, preds)),
        "F1": float(f1_score(labels, preds)),
        "Sensitivity": float(recall_score(labels, preds)),
        "Specificity": float(compute_specificity(labels, preds)),
        "Precision": float(precision_score(labels, preds, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, preds)),
        "Brier_Score": float(brier_score_loss(labels, probs)),
        "ECE": float(compute_ece(labels, probs)),
        "Log_Loss": float(log_loss(labels, probs)),
        "Optimal_Threshold": optimal["threshold"],
        "F1_at_Optimal": optimal["f1"],
        "Accuracy_at_Optimal": optimal["accuracy"],
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "prevalence": float(labels.mean()),
    }
    return metrics


def compute_metrics_with_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, tuple[float, float, float]]:
    """
    Compute metrics with bootstrap confidence intervals.

    Args:
        labels: ground truth labels (0/1)
        probs: predicted probabilities [0, 1]
        n_bootstrap: number of bootstrap resampling iterations
        confidence: confidence level
        seed: random seed
        threshold: classification threshold

    Returns:
        dict: metric_name -> (point_estimate, ci_lower, ci_upper)
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    n = len(labels)

    # List of metrics to bootstrap
    metric_names = [
        "AUROC",
        "AUPRC",
        "Accuracy",
        "F1",
        "Sensitivity",
        "Specificity",
        "Precision",
        "MCC",
        "Brier_Score",
        "ECE",
        "Log_Loss",
    ]

    # Point estimates
    point_estimates = compute_all_metrics(labels, probs, threshold)

    # Bootstrap
    rng = np.random.RandomState(seed)
    boot_results = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        b_labels = labels[idx]
        b_probs = probs[idx]

        # Ensure bootstrap sample contains both positive and negative classes
        if b_labels.sum() == 0 or b_labels.sum() == n:
            continue

        b_preds = (b_probs >= threshold).astype(int)

        try:
            boot_results["AUROC"].append(roc_auc_score(b_labels, b_probs))
            boot_results["AUPRC"].append(average_precision_score(b_labels, b_probs))
            boot_results["Accuracy"].append(accuracy_score(b_labels, b_preds))
            boot_results["F1"].append(f1_score(b_labels, b_preds))
            boot_results["Sensitivity"].append(recall_score(b_labels, b_preds))
            boot_results["Specificity"].append(compute_specificity(b_labels, b_preds))
            boot_results["Precision"].append(
                precision_score(b_labels, b_preds, zero_division=0)
            )
            boot_results["MCC"].append(matthews_corrcoef(b_labels, b_preds))
            boot_results["Brier_Score"].append(brier_score_loss(b_labels, b_probs))
            boot_results["ECE"].append(compute_ece(b_labels, b_probs))
            boot_results["Log_Loss"].append(log_loss(b_labels, b_probs))
        except Exception:
            continue

    # Compute confidence intervals
    alpha = (1 - confidence) / 2
    result = {}
    for name in metric_names:
        point = point_estimates[name]
        if boot_results[name]:
            arr = np.array(boot_results[name])
            ci_low = float(np.percentile(arr, 100 * alpha))
            ci_high = float(np.percentile(arr, 100 * (1 - alpha)))
        else:
            ci_low = ci_high = point
        result[name] = (point, ci_low, ci_high)

    return result
