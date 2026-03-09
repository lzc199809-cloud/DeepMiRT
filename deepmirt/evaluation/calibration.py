#!/usr/bin/env python3
"""
Calibration analysis: Brier Score, ECE, reliability diagram data, temperature scaling.

If the model is poorly calibrated, apply temperature scaling:
Fit parameter T on the validation set, calibrated probability = sigmoid(logit / T).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid
from sklearn.metrics import brier_score_loss, log_loss

from .metrics import compute_ece

logger = logging.getLogger(__name__)


def compute_calibration_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    logits: np.ndarray | None = None,
    n_bins: int = 15,
) -> dict:
    """
    Compute calibration-related metrics.

    Returns:
        dict with keys:
            brier_score, ece, log_loss,
            bin_accs, bin_confs, bin_counts, bin_edges
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if mask.sum() == 0:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_counts.append(0)
            continue
        bin_accs.append(float(labels[mask].mean()))
        bin_confs.append(float(probs[mask].mean()))
        bin_counts.append(int(mask.sum()))

    return {
        "brier_score": float(brier_score_loss(labels, probs)),
        "ece": float(compute_ece(labels, probs, n_bins)),
        "log_loss": float(log_loss(labels, probs)),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges.tolist(),
        "n_bins": n_bins,
    }


def fit_temperature_scaling(
    labels: np.ndarray,
    logits: np.ndarray,
) -> float:
    """
    Fit the temperature scaling parameter T on the validation set.

    Calibrated probability = sigmoid(logit / T)
    Find the optimal T by minimizing NLL (negative log likelihood).

    Args:
        labels: Validation set labels
        logits: Validation set raw logits (before sigmoid)

    Returns:
        Optimal temperature parameter T
    """
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)

    def nll(T):
        scaled_probs = expit(logits / T)
        # Clip to avoid log(0)
        scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
        return -np.mean(
            labels * np.log(scaled_probs) + (1 - labels) * np.log(1 - scaled_probs)
        )

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    optimal_T = result.x

    logger.info(f"Temperature scaling: T = {optimal_T:.4f}")
    return float(optimal_T)


def apply_temperature_scaling(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply temperature scaling and return calibrated probabilities."""
    return expit(logits / temperature)


def run_calibration_analysis(
    labels: np.ndarray,
    probs: np.ndarray,
    logits: np.ndarray,
    val_labels: np.ndarray | None = None,
    val_logits: np.ndarray | None = None,
    n_bins: int = 15,
) -> dict:
    """
    Full calibration analysis pipeline.

    Args:
        labels: Test set labels
        probs: Test set predicted probabilities
        logits: Test set raw logits
        val_labels: Validation set labels (used for fitting temperature scaling)
        val_logits: Validation set logits
        n_bins: Number of bins for the reliability diagram

    Returns:
        dict: Calibration analysis results
    """
    results = {}

    # Original calibration metrics
    cal_metrics = compute_calibration_metrics(labels, probs, logits, n_bins)
    results["original"] = cal_metrics

    logger.info(
        f"Original calibration: Brier={cal_metrics['brier_score']:.4f}, "
        f"ECE={cal_metrics['ece']:.4f}"
    )

    # Temperature scaling
    if val_labels is not None and val_logits is not None:
        temperature = fit_temperature_scaling(val_labels, val_logits)
        calibrated_probs = apply_temperature_scaling(logits, temperature)
        cal_metrics_after = compute_calibration_metrics(
            labels, calibrated_probs, logits, n_bins
        )
        results["temperature"] = temperature
        results["calibrated"] = cal_metrics_after
        results["calibrated_probs"] = calibrated_probs

        logger.info(
            f"After temperature scaling (T={temperature:.4f}): "
            f"Brier={cal_metrics_after['brier_score']:.4f}, "
            f"ECE={cal_metrics_after['ece']:.4f}"
        )
    else:
        logger.info("No validation data provided for temperature scaling")

    return results
