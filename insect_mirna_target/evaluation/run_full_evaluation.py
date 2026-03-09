#!/usr/bin/env python3
"""
Comprehensive evaluation main pipeline orchestration script.

Execution flow:
1. Load model -> inference on test set -> cache predictions
2. Compute extended metrics + bootstrap CI
3. Stratified evaluation (evidence type, negative sample tier)
4. Calibration analysis
5. Frequency bias analysis
6. Error analysis + seed match
7. Baseline comparison
8. Generate all visualizations
9. Output summary report (Markdown + CSV tables + JSON)

Usage:
    python -m insect_mirna_target.evaluation.run_full_evaluation \
        --ckpt checkpoints/epoch=27-val_auroc=0.9612.ckpt \
        --config insect_mirna_target/configs/default.yaml \
        --test-csv insect_mirna_target/data/training/test.csv \
        --output-dir evaluation_outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str) -> None:
    """Configure logging (output to both console and file)."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "evaluation.log", mode="w"),
        ],
    )


def load_eval_config(eval_config_path: str | None = None) -> dict:
    """Load evaluation configuration."""
    default_path = (
        Path(__file__).parent / "configs" / "eval_default.yaml"
    )
    config_path = eval_config_path or str(default_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def run_step_1_inference(
    ckpt_path: str,
    config_path: str,
    test_csv_path: str,
    output_dir: str,
    eval_cfg: dict,
) -> pd.DataFrame:
    """Step 1: Model inference."""
    from .predict import run_inference

    # Prefer parquet; auto fallback to csv if pyarrow is unavailable
    try:
        import pyarrow  # noqa: F401
        cache_path = str(Path(output_dir) / "predictions_test.parquet")
    except ImportError:
        cache_path = str(Path(output_dir) / "predictions_test.csv")
    inf_cfg = eval_cfg.get("inference", {})

    pred_df = run_inference(
        ckpt_path=ckpt_path,
        config_path=config_path,
        test_csv_path=test_csv_path,
        batch_size=inf_cfg.get("batch_size", 256),
        num_workers=inf_cfg.get("num_workers", 8),
        device=inf_cfg.get("device", "cuda"),
        cache_path=cache_path,
    )

    return pred_df


def run_step_2_metrics(
    pred_df: pd.DataFrame,
    output_dir: str,
    eval_cfg: dict,
) -> tuple[dict, dict]:
    """Step 2: Extended metrics + bootstrap CI."""
    from .metrics import compute_all_metrics, compute_metrics_with_ci

    met_cfg = eval_cfg.get("metrics", {})
    labels = pred_df["label"].values
    probs = pred_df["prob"].values
    threshold = met_cfg.get("threshold", 0.5)

    logger.info("Computing extended metrics...")
    point_metrics = compute_all_metrics(labels, probs, threshold)

    logger.info("Computing bootstrap confidence intervals...")
    metrics_ci = compute_metrics_with_ci(
        labels,
        probs,
        n_bootstrap=met_cfg.get("n_bootstrap", 1000),
        confidence=met_cfg.get("confidence", 0.95),
        threshold=threshold,
    )

    # Save
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Point estimate CSV
    metrics_df = pd.DataFrame([point_metrics])
    metrics_df.to_csv(tables_dir / "overall_metrics.csv", index=False)

    # CI results
    ci_rows = []
    for name, (pt, lo, hi) in metrics_ci.items():
        ci_rows.append({
            "Metric": name,
            "Point_Estimate": pt,
            "CI_Lower": lo,
            "CI_Upper": hi,
            "CI_Width": hi - lo,
        })
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(tables_dir / "metrics_with_ci.csv", index=False)

    logger.info("Overall metrics computed:")
    for name, (pt, lo, hi) in metrics_ci.items():
        logger.info(f"  {name}: {pt:.4f} [{lo:.4f}, {hi:.4f}]")

    return point_metrics, metrics_ci


def run_step_3_stratified(
    pred_df: pd.DataFrame,
    output_dir: str,
    eval_cfg: dict,
) -> tuple[dict, dict]:
    """Step 3: Stratified evaluation."""
    from .stratified_eval import (
        evaluate_by_evidence_type,
        evaluate_by_negative_tier,
        stratified_results_to_dataframe,
    )

    strat_cfg = eval_cfg.get("stratified", {})
    n_bootstrap = strat_cfg.get("n_bootstrap", 1000)
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating by evidence type...")
    evidence_results = evaluate_by_evidence_type(pred_df, n_bootstrap=n_bootstrap)
    evidence_df = stratified_results_to_dataframe(evidence_results, "Evidence_Type")
    evidence_df.to_csv(tables_dir / "per_evidence_type_metrics.csv", index=False)

    logger.info("Evaluating by negative sample tier...")
    tier_results = evaluate_by_negative_tier(pred_df, n_bootstrap=n_bootstrap)
    tier_df = stratified_results_to_dataframe(tier_results, "Tier")
    tier_df.to_csv(tables_dir / "per_negative_tier_metrics.csv", index=False)

    return evidence_results, tier_results


def run_step_4_calibration(
    pred_df: pd.DataFrame,
    output_dir: str,
    eval_cfg: dict,
) -> dict:
    """Step 4: Calibration analysis."""
    from .calibration import run_calibration_analysis

    cal_cfg = eval_cfg.get("calibration", {})

    labels = pred_df["label"].values
    probs = pred_df["prob"].values
    logits = pred_df["logit"].values

    logger.info("Running calibration analysis...")
    cal_results = run_calibration_analysis(
        labels=labels,
        probs=probs,
        logits=logits,
        n_bins=cal_cfg.get("n_bins", 15),
    )

    return cal_results


def run_step_5_frequency_bias(
    pred_df: pd.DataFrame,
    output_dir: str,
    eval_cfg: dict,
) -> dict:
    """Step 5: Frequency bias analysis."""
    from .bias_analysis import compute_frequency_summary_table, evaluate_by_frequency_quintile

    freq_cfg = eval_cfg.get("frequency_bias", {})
    train_csv = freq_cfg.get("train_csv", "insect_mirna_target/data/training/train.csv")
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating by miRNA frequency quintile...")
    quintile_results = evaluate_by_frequency_quintile(
        pred_df,
        train_csv_path=train_csv,
        n_bootstrap=freq_cfg.get("n_bootstrap", 1000),
        n_quintiles=freq_cfg.get("n_quintiles", 5),
    )

    summary = compute_frequency_summary_table(
        pred_df, train_csv, n_quintiles=freq_cfg.get("n_quintiles", 5)
    )
    summary.to_csv(tables_dir / "frequency_bias_metrics.csv", index=False)

    return quintile_results


def run_step_6_error_analysis(
    pred_df: pd.DataFrame,
    output_dir: str,
    eval_cfg: dict,
) -> tuple[dict, dict]:
    """Step 6: Error analysis + seed match."""
    from .error_analysis import run_error_analysis, run_seed_match_analysis

    err_cfg = eval_cfg.get("error_analysis", {})
    threshold = err_cfg.get("threshold", 0.5)
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running error analysis...")
    error_results = run_error_analysis(pred_df, threshold=threshold)

    # Save error summary
    summary_data = {
        k: v for k, v in error_results.items()
        if k != "error_df" and not isinstance(v, pd.DataFrame)
    }
    with open(tables_dir / "error_analysis_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    logger.info("Running seed match analysis...")
    seed_results = run_seed_match_analysis(pred_df, threshold=threshold)

    if "seed_match_table" in seed_results:
        seed_results["seed_match_table"].to_csv(
            tables_dir / "seed_match_analysis.csv", index=False
        )

    return error_results, seed_results


def run_step_7_comparison(
    pred_df: pd.DataFrame,
    our_metrics: dict,
    output_dir: str,
    eval_cfg: dict,
) -> pd.DataFrame:
    """Step 7: Baseline and tool comparison."""
    from .comparison import run_all_comparisons

    comp_cfg = eval_cfg.get("comparison", {})
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running comparison with baselines and other tools...")
    comparison_df = run_all_comparisons(
        df=pred_df,
        our_metrics=our_metrics,
        run_mirbench=comp_cfg.get("run_mirbench", True),
        run_external_tools=comp_cfg.get("run_external_tools", True),
        mirbench_sample_size=comp_cfg.get("mirbench_sample_size", 50000),
        external_sample_size=comp_cfg.get("external_sample_size", 10000),
        seed=comp_cfg.get("seed", 42),
    )

    comparison_df.to_csv(tables_dir / "comparison_results.csv", index=False)
    logger.info("Comparison results:")
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    return comparison_df


def run_step_8_visualization(
    pred_df: pd.DataFrame,
    evidence_results: dict,
    tier_results: dict,
    quintile_results: dict,
    error_results: dict,
    comparison_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Step 8: Generate all visualizations."""
    from . import visualization as viz

    fig_dir = str(Path(output_dir) / "figures")

    labels = pred_df["label"].values
    probs = pred_df["prob"].values
    preds = pred_df["pred"].values

    logger.info("Generating visualizations...")

    # Basic charts
    viz.plot_roc_curve(labels, probs, fig_dir)
    viz.plot_pr_curve(labels, probs, fig_dir)
    viz.plot_confusion_matrix(labels, preds, fig_dir)
    viz.plot_score_distribution(labels, probs, fig_dir)
    viz.plot_threshold_sensitivity(labels, probs, fig_dir)
    viz.plot_calibration_reliability(labels, probs, fig_dir)

    # Stratified charts
    if evidence_results:
        viz.plot_evidence_type_comparison(evidence_results, fig_dir)
    if tier_results:
        viz.plot_negative_tier_comparison(tier_results, fig_dir)

    # Frequency bias
    if quintile_results:
        viz.plot_frequency_bias(quintile_results, fig_dir)

    # Error analysis
    if "error_df" in error_results:
        viz.plot_error_analysis_dashboard(error_results["error_df"], fig_dir)

    # Multi-model comparison
    if comparison_df is not None and len(comparison_df) > 1:
        viz.plot_multi_model_auroc_bar(comparison_df, fig_dir)
        viz.plot_multi_model_radar(comparison_df, fig_dir)

    logger.info(f"All figures saved to {fig_dir}")


def run_step_9_report(
    pred_df: pd.DataFrame,
    point_metrics: dict,
    metrics_ci: dict,
    evidence_results: dict,
    tier_results: dict,
    cal_results: dict,
    quintile_results: dict,
    error_results: dict,
    seed_results: dict,
    comparison_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Step 9: Generate summary report."""
    out = Path(output_dir)

    # JSON summary
    json_summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(pred_df),
        "overall_metrics": {
            k: v for k, v in point_metrics.items()
            if isinstance(v, (int, float))
        },
        "metrics_with_ci": {
            k: {"point": v[0], "ci_low": v[1], "ci_high": v[2]}
            for k, v in metrics_ci.items()
        },
        "calibration": {
            "brier_score": cal_results.get("original", {}).get("brier_score"),
            "ece": cal_results.get("original", {}).get("ece"),
            "temperature": cal_results.get("temperature"),
        },
    }
    with open(out / "metrics_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2, default=str)

    # Markdown report
    report_lines = [
        "# miRNA Target Prediction — Comprehensive Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Test samples:** {len(pred_df):,}",
        f"**Positive:** {int(pred_df['label'].sum()):,} "
        f"({pred_df['label'].mean():.1%})",
        "",
        "---",
        "",
        "## 1. Overall Metrics (with 95% Bootstrap CI)",
        "",
        "| Metric | Value | 95% CI |",
        "|--------|-------|--------|",
    ]

    for name, (pt, lo, hi) in metrics_ci.items():
        report_lines.append(f"| {name} | {pt:.4f} | [{lo:.4f}, {hi:.4f}] |")

    report_lines.extend([
        "",
        f"- Optimal Threshold (Youden's J): {point_metrics.get('Optimal_Threshold', 'N/A'):.4f}",
        f"- F1 at Optimal Threshold: {point_metrics.get('F1_at_Optimal', 'N/A'):.4f}",
        "",
        "---",
        "",
        "## 2. Stratified Evaluation by Evidence Type",
        "",
        "| Evidence Type | AUROC | AUPRC | n_samples |",
        "|--------------|-------|-------|-----------|",
    ])

    for etype, metrics in evidence_results.items():
        auroc = metrics["AUROC"][0] if isinstance(metrics.get("AUROC"), tuple) else metrics.get("AUROC", "N/A")
        auprc = metrics["AUPRC"][0] if isinstance(metrics.get("AUPRC"), tuple) else metrics.get("AUPRC", "N/A")
        n = metrics.get("n_samples", "N/A")
        report_lines.append(
            f"| {etype} | {auroc:.4f} | {auprc:.4f} | {n:,} |"
            if isinstance(auroc, float) else
            f"| {etype} | {auroc} | {auprc} | {n} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Negative Sample Tier Analysis",
        "",
        "| Tier | AUROC | AUPRC | n_samples |",
        "|------|-------|-------|-----------|",
    ])

    for tier, metrics in tier_results.items():
        auroc = metrics["AUROC"][0] if isinstance(metrics.get("AUROC"), tuple) else metrics.get("AUROC", "N/A")
        auprc = metrics["AUPRC"][0] if isinstance(metrics.get("AUPRC"), tuple) else metrics.get("AUPRC", "N/A")
        n = metrics.get("n_samples", "N/A")
        report_lines.append(
            f"| {tier} | {auroc:.4f} | {auprc:.4f} | {n:,} |"
            if isinstance(auroc, float) else
            f"| {tier} | {auroc} | {auprc} | {n} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Calibration Analysis",
        "",
        f"- **Brier Score:** {cal_results.get('original', {}).get('brier_score', 'N/A'):.4f}"
        if isinstance(cal_results.get('original', {}).get('brier_score'), float) else
        f"- **Brier Score:** N/A",
        f"- **ECE:** {cal_results.get('original', {}).get('ece', 'N/A'):.4f}"
        if isinstance(cal_results.get('original', {}).get('ece'), float) else
        f"- **ECE:** N/A",
    ])

    if "temperature" in cal_results:
        report_lines.append(
            f"- **Temperature (post-hoc):** {cal_results['temperature']:.4f}"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 5. miRNA Frequency Bias Analysis",
        "",
        "| Quintile | AUROC | AUPRC | n_samples |",
        "|----------|-------|-------|-----------|",
    ])

    for q, metrics in sorted(quintile_results.items()):
        auroc = metrics["AUROC"][0] if isinstance(metrics.get("AUROC"), tuple) else metrics.get("AUROC", "N/A")
        auprc = metrics["AUPRC"][0] if isinstance(metrics.get("AUPRC"), tuple) else metrics.get("AUPRC", "N/A")
        n = metrics.get("n_samples", "N/A")
        report_lines.append(
            f"| {q} | {auroc:.4f} | {auprc:.4f} | {n:,} |"
            if isinstance(auroc, float) else
            f"| {q} | {auroc} | {auprc} | {n} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 6. Error Analysis",
        "",
        f"- **Total FP:** {error_results.get('error_counts', {}).get('FP', 0):,}",
        f"- **Total FN:** {error_results.get('error_counts', {}).get('FN', 0):,}",
        f"- **High-confidence FP (prob>0.9):** "
        f"{error_results.get('high_confidence_errors', {}).get('FP_prob_gt_0.9', 0):,}",
        f"- **High-confidence FN (prob<0.1):** "
        f"{error_results.get('high_confidence_errors', {}).get('FN_prob_lt_0.1', 0):,}",
    ])

    # Seed match
    if seed_results:
        report_lines.extend([
            "",
            "### Seed Match Analysis",
            "",
            f"- Canonical seed matches: {seed_results.get('canonical_count', 'N/A'):,}",
            f"- Non-canonical: {seed_results.get('noncanonical_count', 'N/A'):,}",
            f"- Recall (canonical): {seed_results.get('recall_canonical', 'N/A'):.4f}"
            if isinstance(seed_results.get('recall_canonical'), float) else
            f"- Recall (canonical): N/A",
            f"- Recall (non-canonical): {seed_results.get('recall_non-canonical', 'N/A'):.4f}"
            if isinstance(seed_results.get('recall_non-canonical'), float) else
            f"- Recall (non-canonical): N/A",
            f"- FN non-canonical %: {seed_results.get('FN_noncanonical_pct', 'N/A'):.1f}%"
            if isinstance(seed_results.get('FN_noncanonical_pct'), float) else
            f"- FN non-canonical %: N/A",
        ])

    report_lines.extend([
        "",
        "---",
        "",
        "## 7. Comparison with Other Methods",
        "",
    ])

    if comparison_df is not None and len(comparison_df) > 0:
        try:
            report_lines.append(comparison_df.to_markdown(index=False))
        except ImportError:
            # tabulate not installed, fallback to manual formatting
            cols = comparison_df.columns.tolist()
            report_lines.append("| " + " | ".join(cols) + " |")
            report_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, row in comparison_df.iterrows():
                vals = []
                for c in cols:
                    v = row[c]
                    vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
                report_lines.append("| " + " | ".join(vals) + " |")
    else:
        report_lines.append("No comparison results available.")

    report_lines.extend([
        "",
        "---",
        "",
        "## 8. Figures",
        "",
        "All figures saved in `figures/` directory:",
        "- `roc_curve.pdf/.png`",
        "- `pr_curve.pdf/.png`",
        "- `confusion_matrix.pdf/.png`",
        "- `evidence_type_comparison.pdf/.png`",
        "- `negative_tier_comparison.pdf/.png`",
        "- `calibration_reliability.pdf/.png`",
        "- `frequency_bias.pdf/.png`",
        "- `score_distribution.pdf/.png`",
        "- `threshold_sensitivity.pdf/.png`",
        "- `error_analysis_dashboard.pdf/.png`",
        "- `multi_model_auroc_bar.pdf/.png`",
        "- `multi_model_radar.pdf/.png`",
    ])

    report_text = "\n".join(report_lines)
    with open(out / "report.md", "w") as f:
        f.write(report_text)

    logger.info(f"Report saved to {out / 'report.md'}")


# ── Main function ──


def main():
    parser = argparse.ArgumentParser(
        description="miRNA Target Prediction — Comprehensive Evaluation"
    )
    parser.add_argument(
        "--ckpt", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        default="insect_mirna_target/configs/default.yaml",
        help="Training config YAML path",
    )
    parser.add_argument(
        "--test-csv",
        default="insect_mirna_target/data/training/test.csv",
        help="Test set CSV path",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Evaluation config YAML path (default: eval_default.yaml)",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip external tool comparisons (faster)",
    )
    parser.add_argument(
        "--skip-mirbench",
        action="store_true",
        help="Skip miRBench comparisons",
    )
    parser.add_argument(
        "--skip-external-tools",
        action="store_true",
        help="Skip RNAhybrid/miRanda",
    )
    args = parser.parse_args()

    setup_logging(args.output_dir)
    eval_cfg = load_eval_config(args.eval_config)

    # Override config based on command-line arguments
    if args.skip_comparison:
        eval_cfg.setdefault("comparison", {})["run_mirbench"] = False
        eval_cfg.setdefault("comparison", {})["run_external_tools"] = False
    if args.skip_mirbench:
        eval_cfg.setdefault("comparison", {})["run_mirbench"] = False
    if args.skip_external_tools:
        eval_cfg.setdefault("comparison", {})["run_external_tools"] = False

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("miRNA Target Prediction — Comprehensive Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Test CSV: {args.test_csv}")
    logger.info(f"Output: {args.output_dir}")

    # Step 1: Inference
    logger.info("\n[Step 1/9] Running inference...")
    pred_df = run_step_1_inference(
        args.ckpt, args.config, args.test_csv, args.output_dir, eval_cfg
    )

    # Step 2: Extended metrics
    logger.info("\n[Step 2/9] Computing extended metrics...")
    point_metrics, metrics_ci = run_step_2_metrics(pred_df, args.output_dir, eval_cfg)

    # Step 3: Stratified evaluation
    logger.info("\n[Step 3/9] Stratified evaluation...")
    evidence_results, tier_results = run_step_3_stratified(
        pred_df, args.output_dir, eval_cfg
    )

    # Step 4: Calibration analysis
    logger.info("\n[Step 4/9] Calibration analysis...")
    cal_results = run_step_4_calibration(pred_df, args.output_dir, eval_cfg)

    # Step 5: Frequency bias
    logger.info("\n[Step 5/9] Frequency bias analysis...")
    quintile_results = run_step_5_frequency_bias(pred_df, args.output_dir, eval_cfg)

    # Step 6: Error analysis
    logger.info("\n[Step 6/9] Error analysis...")
    error_results, seed_results = run_step_6_error_analysis(
        pred_df, args.output_dir, eval_cfg
    )

    # Step 7: Comparison
    logger.info("\n[Step 7/9] Comparison with other methods...")
    comparison_df = run_step_7_comparison(
        pred_df, point_metrics, args.output_dir, eval_cfg
    )

    # Step 8: Visualization
    logger.info("\n[Step 8/9] Generating visualizations...")
    run_step_8_visualization(
        pred_df,
        evidence_results,
        tier_results,
        quintile_results,
        error_results,
        comparison_df,
        args.output_dir,
    )

    # Step 9: Report
    logger.info("\n[Step 9/9] Generating report...")
    run_step_9_report(
        pred_df,
        point_metrics,
        metrics_ci,
        evidence_results,
        tier_results,
        cal_results,
        quintile_results,
        error_results,
        seed_results,
        comparison_df,
        args.output_dir,
    )

    elapsed = time.time() - start_time
    logger.info(f"\nEvaluation complete in {elapsed / 60:.1f} minutes.")
    logger.info(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
