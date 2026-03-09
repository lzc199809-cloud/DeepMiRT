#!/usr/bin/env python3
"""
Comparison with baseline methods and mainstream tools.

7a. Simple baselines (internally implemented):
    - Random baseline
    - Seed match baseline

7b. miRBench framework (11 predictors)

7c. Mimosa (NAR 2024)

7d. Traditional tools: RNAhybrid, miRanda

7e. Aggregated comparison table + visualization
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .error_analysis import find_seed_match
from .metrics import compute_all_metrics, compute_specificity

logger = logging.getLogger(__name__)


# ── Unified metrics computation ──


def compute_comparison_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Compute a unified subset of metrics for the comparison table.

    Args:
        labels: ground truth labels
        probs: prediction scores (no need to normalize to [0,1]; AUROC/AUPRC only require correct ranking)
        threshold: threshold. None uses Youden's J to automatically find the optimal threshold.

    Returns:
        dict containing AUROC, AUPRC, F1, MCC, Sensitivity, Specificity, Threshold
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    # Threshold-free metrics (always valid)
    metrics = {
        "AUROC": float(roc_auc_score(labels, probs)),
        "AUPRC": float(average_precision_score(labels, probs)),
    }

    # Automatically determine the optimal threshold (Youden's J)
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(labels, probs)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        threshold = float(thresholds[best_idx])

    preds = (probs >= threshold).astype(int)
    metrics.update({
        "F1": float(f1_score(labels, preds, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, preds)),
        "Sensitivity": float(recall_score(labels, preds, zero_division=0)),
        "Specificity": float(compute_specificity(labels, preds)),
        "Threshold": float(threshold),
    })
    return metrics


# ── 7a. Simple baselines ──


def random_baseline(labels: np.ndarray, seed: int = 42) -> dict:
    """Random baseline: AUROC ~ 0.5."""
    rng = np.random.RandomState(seed)
    probs = rng.rand(len(labels))
    return compute_comparison_metrics(labels, probs, threshold=None)


def seed_match_baseline(
    df: pd.DataFrame,
    seed_types: set[str] | None = None,
) -> dict:
    """
    Seed match baseline: predict positive if canonical seed match is found.

    Args:
        df: prediction DataFrame (containing mirna_seq, target_fragment_40nt, label)
        seed_types: which seed types count as positive prediction, default all canonical

    Returns:
        comparison metrics
    """
    if seed_types is None:
        seed_types = {"8mer", "7mer-m8", "7mer-A1", "6mer"}

    logger.info("Computing seed match baseline...")
    seed_matches = df.apply(
        lambda row: find_seed_match(row["mirna_seq"], row["target_fragment_40nt"]),
        axis=1,
    )
    preds = seed_matches.isin(seed_types).astype(int)
    # Use 0/1 as "probability" (rule-based methods have no continuous scores)
    probs = preds.astype(float).values
    labels = df["label"].values

    return compute_comparison_metrics(labels, probs, threshold=None)


# ── 7b. miRBench framework ──


def run_mirbench_predictors(
    df: pd.DataFrame,
    predictors: list[str] | None = None,
    sample_size: int | None = 50000,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Run multiple predictors through the miRBench framework.

    Args:
        df: prediction DataFrame (containing mirna_seq, target_fragment_40nt, label)
        predictors: list of predictor names to run, None to run all
        sample_size: sample size (to reduce runtime for slow models), None to skip sampling
        seed: random seed

    Returns:
        {predictor_name: comparison_metrics}
    """
    try:
        from miRBench.encoder import get_encoder
        from miRBench.predictor import get_predictor
    except ImportError:
        logger.warning(
            "miRBench not installed. Install with: pip install miRBench\n"
            "Skipping miRBench comparison."
        )
        return {}

    if predictors is None:
        predictors = [
            "TargetScanCnn_McGeary2019",
            "TargetNet_Min2021",
            "miRBind_Klimentova2022",
            "CnnMirTarget_Zheng2020",
            "miRNA_CNN_Hejret2023",
            "InteractionAwareModel_Yang2024",
            "RNACofold",
            "Seed8mer",
            "Seed7mer",
            "Seed6mer",
            "Seed6merBulgeOrMismatch",
        ]

    # Prepare DataFrame in miRBench format (requires DNA format, U->T)
    bench_df = pd.DataFrame(
        {
            "noncodingRNA": df["mirna_seq"].str.replace("U", "T").str.replace("u", "t").values,
            "gene": df["target_fragment_40nt"].str.replace("U", "T").str.replace("u", "t").values,
            "label": df["label"].values,
        }
    )

    # Filter out sequences containing non-standard bases like N (miRBench encoder does not support them)
    valid_bases = set("ACGTacgt")
    valid_mask = bench_df["noncodingRNA"].apply(
        lambda s: all(c in valid_bases for c in str(s))
    ) & bench_df["gene"].apply(
        lambda s: all(c in valid_bases for c in str(s))
    )
    n_filtered = (~valid_mask).sum()
    if n_filtered > 0:
        logger.info(f"Filtered {n_filtered} samples with non-standard bases (N, etc.)")
        bench_df = bench_df[valid_mask].reset_index(drop=True)

    # Optional sampling
    if sample_size and len(bench_df) > sample_size:
        bench_df = bench_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        logger.info(f"Sampled {sample_size} samples for miRBench evaluation")

    labels = bench_df["label"].values
    results = {}

    for pred_name in predictors:
        logger.info(f"Running miRBench predictor: {pred_name}")
        try:
            encoder = get_encoder(pred_name)
            predictor = get_predictor(pred_name)
            encoded = encoder(bench_df[["noncodingRNA", "gene"]].copy())
            scores = predictor(encoded)
            scores = np.asarray(scores, dtype=float).flatten()

            # Some predictors may return NaN
            valid_mask = ~np.isnan(scores)
            if valid_mask.sum() < len(scores) * 0.5:
                logger.warning(
                    f"  {pred_name}: too many NaN ({(~valid_mask).sum()}/{len(scores)}), skipping"
                )
                continue

            valid_labels = labels[valid_mask]
            valid_scores = scores[valid_mask]

            # No min-max normalization, keep original scores
            # AUROC/AUPRC only require correct ranking, Youden's J finds the threshold automatically
            metrics = compute_comparison_metrics(valid_labels, valid_scores, threshold=None)
            results[pred_name] = metrics
            logger.info(
                f"  {pred_name}: AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}"
            )

        except Exception as e:
            logger.error(f"  {pred_name} failed: {e}")
            continue

    return results


# ── 7b+. miRBench standard benchmark evaluation ──

_METHOD_TYPE_MAP = {
    "TargetScanCnn_McGeary2019": "CNN",
    "TargetNet_Min2021": "DL",
    "miRBind_Klimentova2022": "DL",
    "CnnMirTarget_Zheng2020": "CNN",
    "miRNA_CNN_Hejret2023": "CNN",
    "InteractionAwareModel_Yang2024": "DL+Attn",
    "RNACofold": "Thermodynamic",
    "Seed8mer": "Rule",
    "Seed7mer": "Rule",
    "Seed6mer": "Rule",
    "Seed6merBulgeOrMismatch": "Rule",
}


def run_mirbench_standard_benchmark(
    our_ckpt_path: str,
    our_config_path: str,
    datasets: list[str] | None = None,
    predictors: list[str] | None = None,
    device: str = "cuda",
    batch_size: int = 256,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Evaluate all methods on miRBench standard benchmark datasets (gold standard for fair comparison).

    Each method is evaluated on exactly the same data, using the same metrics.

    Args:
        our_ckpt_path: path to our model's checkpoint
        our_config_path: path to our model's config YAML
        datasets: list of miRBench dataset names, default all 3
        predictors: list of miRBench predictor names, default all 11
        device: inference device
        batch_size: inference batch size for our model
        max_samples: max samples per dataset (for sampling large datasets), None to skip sampling
        seed: random seed

    Returns:
        {dataset_name: comparison_df}
    """
    try:
        from miRBench.dataset import get_dataset_df
        from miRBench.encoder import get_encoder
        from miRBench.predictor import get_predictor
    except ImportError:
        logger.warning("miRBench not installed. Skipping standard benchmark evaluation.")
        return {}

    if datasets is None:
        datasets = [
            "AGO2_CLASH_Hejret2023",
            "AGO2_eCLIP_Klimentova2022",
            "AGO2_eCLIP_Manakov2022",
        ]

    if predictors is None:
        predictors = [
            "TargetScanCnn_McGeary2019",
            "TargetNet_Min2021",
            "miRBind_Klimentova2022",
            "CnnMirTarget_Zheng2020",
            "miRNA_CNN_Hejret2023",
            "InteractionAwareModel_Yang2024",
            "RNACofold",
            "Seed8mer",
            "Seed7mer",
            "Seed6mer",
            "Seed6merBulgeOrMismatch",
        ]

    all_benchmark_results = {}

    for ds_name in datasets:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Benchmark dataset: {ds_name}")
        logger.info(f"{'=' * 60}")

        # Load standard test split
        bench_df = get_dataset_df(ds_name, split="test")
        labels = bench_df["label"].values

        # Optional sampling (for large Manakov dataset)
        if max_samples and len(bench_df) > max_samples:
            bench_df = bench_df.sample(n=max_samples, random_state=seed).reset_index(
                drop=True
            )
            labels = bench_df["label"].values
            logger.info(f"Sampled {max_samples} from {ds_name}")

        n_pos = int(labels.sum())
        n_neg = int((labels == 0).sum())
        logger.info(f"Loaded {len(bench_df)} test samples (pos={n_pos}, neg={n_neg})")

        results = []

        # --- Run miRBench predictors ---
        for pred_name in predictors:
            logger.info(f"  Running miRBench predictor: {pred_name}")
            try:
                encoder = get_encoder(pred_name)
                predictor = get_predictor(pred_name)
                encoded = encoder(bench_df[["noncodingRNA", "gene"]].copy())
                scores = predictor(encoded)
                scores = np.asarray(scores, dtype=float).flatten()

                valid_mask = ~np.isnan(scores)
                if valid_mask.sum() < len(scores) * 0.5:
                    logger.warning(f"    {pred_name}: too many NaN, skipping")
                    continue

                valid_labels = labels[valid_mask]
                valid_scores = scores[valid_mask]

                metrics = compute_comparison_metrics(
                    valid_labels, valid_scores, threshold=None
                )
                results.append({
                    "Method": pred_name,
                    "Type": _METHOD_TYPE_MAP.get(pred_name, "DL"),
                    **metrics,
                })
                logger.info(
                    f"    AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}"
                )

            except Exception as e:
                logger.error(f"    {pred_name} failed: {e}")

        # --- Run our model ---
        logger.info(f"  Running our model on {ds_name}...")
        try:
            from .predict import predict_on_sequences

            # miRBench gene column is 50nt, our model was trained with 40nt
            # Center-crop to 40nt
            mirna_seqs = bench_df["noncodingRNA"].tolist()
            target_seqs_50nt = bench_df["gene"].tolist()
            target_seqs_40nt = []
            for seq in target_seqs_50nt:
                seq = str(seq)
                if len(seq) > 40:
                    start = (len(seq) - 40) // 2
                    seq = seq[start : start + 40]
                elif len(seq) < 40:
                    seq = seq + "N" * (40 - len(seq))
                target_seqs_40nt.append(seq)

            our_probs = predict_on_sequences(
                our_ckpt_path, our_config_path,
                mirna_seqs, target_seqs_40nt,
                batch_size=batch_size, device=device,
            )
            our_metrics = compute_comparison_metrics(labels, our_probs, threshold=None)
            results.append({
                "Method": "Ours (RNA-FM)",
                "Type": "DL+LM",
                **our_metrics,
            })
            logger.info(
                f"    Ours: AUROC={our_metrics['AUROC']:.4f}, AUPRC={our_metrics['AUPRC']:.4f}"
            )
        except Exception as e:
            logger.error(f"    Our model failed on {ds_name}: {e}")
            import traceback
            traceback.print_exc()

        # Build comparison table
        comp_df = pd.DataFrame(results)
        comp_df = comp_df.sort_values("AUROC", ascending=False).reset_index(drop=True)
        all_benchmark_results[ds_name] = comp_df

    return all_benchmark_results


# ── 7c. Mimosa ──


def run_mimosa(
    df: pd.DataFrame,
    mimosa_dir: str = "Mimosa",
    sample_size: int | None = 50000,
    seed: int = 42,
) -> dict | None:
    """
    Run the Mimosa predictor (requires cloning the repository first).

    Args:
        df: prediction DataFrame
        mimosa_dir: path to the Mimosa repository
        sample_size: sample size
        seed: random seed

    Returns:
        comparison_metrics or None
    """
    mimosa_path = Path(mimosa_dir)
    if not mimosa_path.exists():
        logger.warning(
            f"Mimosa directory not found at {mimosa_dir}. "
            "Clone with: git clone https://github.com/biyueeee/Mimosa"
        )
        return None

    logger.info("Running Mimosa predictor...")
    try:
        import sys
        sys.path.insert(0, str(mimosa_path))

        # Prepare input data
        bench_df = df.copy()
        if sample_size and len(bench_df) > sample_size:
            bench_df = bench_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

        # Convert to Mimosa required format and write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            mimosa_input = pd.DataFrame(
                {
                    "miRNA": bench_df["mirna_seq"].values,
                    "target": bench_df["target_fragment_40nt"].values,
                    "label": bench_df["label"].values,
                }
            )
            mimosa_input.to_csv(tmp.name, index=False)
            input_path = tmp.name

        logger.info(
            f"Mimosa input prepared: {len(mimosa_input)} samples at {input_path}"
        )

        # Note: actually running Mimosa requires its specific environment and model weights.
        # This provides the framework code; the actual inference logic needs to be adapted to Mimosa's API.
        logger.warning(
            "Mimosa integration requires manual setup. "
            "See Mimosa README for model weights and dependencies."
        )
        return None

    except Exception as e:
        logger.error(f"Mimosa failed: {e}")
        return None


# ── 7d. Traditional command-line tools ──


def run_rnahybrid(
    df: pd.DataFrame,
    sample_size: int | None = 10000,
    seed: int = 42,
) -> dict | None:
    """
    Run RNAhybrid (minimum free energy method).

    Negated MFE scores are used as binding affinity scores.
    """
    # Check installation (prefer path in conda environment)
    import shutil
    rnahybrid_bin = shutil.which("RNAhybrid")
    if rnahybrid_bin is None:
        # Try conda environment path
        import sys
        conda_prefix = Path(sys.executable).parent
        candidate = conda_prefix / "RNAhybrid"
        if candidate.exists():
            rnahybrid_bin = str(candidate)
        else:
            logger.warning(
                "RNAhybrid not found. Install with: conda install -c bioconda rnahybrid"
            )
            return None

    bench_df = df.copy()
    if sample_size and len(bench_df) > sample_size:
        bench_df = bench_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    logger.info(f"Running RNAhybrid on {len(bench_df)} samples...")
    mfe_scores = []
    total = len(bench_df)

    for idx, (_, row) in enumerate(bench_df.iterrows()):
        if idx % 1000 == 0 and idx > 0:
            logger.info(f"  RNAhybrid progress: {idx}/{total}")
        mirna = row["mirna_seq"].upper().replace("T", "U")
        target = row["target_fragment_40nt"].upper().replace("T", "U")

        # RNAhybrid -t/-q requires file paths
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fa", delete=False
        ) as mirna_fa, tempfile.NamedTemporaryFile(
            mode="w", suffix=".fa", delete=False
        ) as target_fa:
            mirna_fa.write(f">mirna\n{mirna}\n")
            target_fa.write(f">target\n{target}\n")
            mirna_path = mirna_fa.name
            target_path = target_fa.name

        try:
            result = subprocess.run(
                [rnahybrid_bin, "-s", "3utr_human", "-c",
                 "-t", target_path, "-q", mirna_path],
                capture_output=True, text=True, timeout=10,
            )
            # Parse MFE (colon-separated, 5th field)
            output = result.stdout.strip()
            if output:
                parts = output.split(":")
                mfe = float(parts[4]) if len(parts) > 4 else 0.0
            else:
                mfe = 0.0
        except (subprocess.TimeoutExpired, ValueError, Exception):
            mfe = 0.0

        # Clean up temp files
        Path(mirna_path).unlink(missing_ok=True)
        Path(target_path).unlink(missing_ok=True)

        mfe_scores.append(mfe)

    mfe_arr = np.array(mfe_scores)
    # More negative MFE = more likely to bind; use -MFE as score (higher = more likely to bind)
    # AUROC/AUPRC only require correct ranking, Youden's J finds the optimal threshold automatically
    scores = -mfe_arr

    labels = bench_df["label"].values
    return compute_comparison_metrics(labels, scores, threshold=None)


def run_miranda(
    df: pd.DataFrame,
    sample_size: int | None = 10000,
    seed: int = 42,
) -> dict | None:
    """
    Run miRanda (sequence complementarity + thermodynamic scoring).

    Score is used directly as binding affinity.
    """
    import shutil
    miranda_bin = shutil.which("miranda")
    if miranda_bin is None:
        import sys
        conda_prefix = Path(sys.executable).parent
        candidate = conda_prefix / "miranda"
        if candidate.exists():
            miranda_bin = str(candidate)
        else:
            logger.warning(
                "miRanda not found. Install with: conda install -c bioconda miranda"
            )
            return None

    bench_df = df.copy()
    if sample_size and len(bench_df) > sample_size:
        bench_df = bench_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    logger.info(f"Running miRanda on {len(bench_df)} samples...")
    scores = []
    total = len(bench_df)

    for idx, (_, row) in enumerate(bench_df.iterrows()):
        if idx % 1000 == 0 and idx > 0:
            logger.info(f"  miRanda progress: {idx}/{total}")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fa", delete=False
        ) as mirna_fa, tempfile.NamedTemporaryFile(
            mode="w", suffix=".fa", delete=False
        ) as target_fa:
            mirna_fa.write(f">mirna\n{row['mirna_seq']}\n")
            target_fa.write(f">target\n{row['target_fragment_40nt']}\n")
            mirna_path = mirna_fa.name
            target_path = target_fa.name

        try:
            result = subprocess.run(
                [miranda_bin, mirna_path, target_path, "-sc", "80", "-en", "-10"],
                capture_output=True, text=True, timeout=5,
            )
            # Parse score (take the highest)
            max_score = 0.0
            for line in result.stdout.split("\n"):
                if line.startswith(">>"):
                    parts = line.split("\t")
                    if len(parts) >= 4:
                        try:
                            s = float(parts[2])
                            max_score = max(max_score, s)
                        except ValueError:
                            pass
            scores.append(max_score)
        except (subprocess.TimeoutExpired, Exception):
            scores.append(0.0)

        # Clean up temp files
        Path(mirna_path).unlink(missing_ok=True)
        Path(target_path).unlink(missing_ok=True)

    score_arr = np.array(scores)
    # Use miRanda alignment score directly (higher = more likely to bind)
    # AUROC/AUPRC only require correct ranking, Youden's J finds the optimal threshold automatically
    labels = bench_df["label"].values
    return compute_comparison_metrics(labels, score_arr, threshold=None)


# ── 7e. Aggregated comparison ──


def run_all_comparisons(
    df: pd.DataFrame,
    our_metrics: dict,
    run_mirbench: bool = True,
    run_external_tools: bool = True,
    mirbench_sample_size: int = 50000,
    external_sample_size: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run all comparison methods and aggregate into a DataFrame.

    Args:
        df: prediction DataFrame
        our_metrics: metrics from our model
        run_mirbench: whether to run miRBench predictors
        run_external_tools: whether to run external command-line tools
        mirbench_sample_size: sample size for miRBench
        external_sample_size: sample size for external tools
        seed: random seed

    Returns:
        comparison results DataFrame
    """
    all_results = []

    # Our model
    our_row = {"Method": "Ours (RNA-FM)", "Type": "DL+LM"}
    for k in ["AUROC", "AUPRC", "F1", "MCC", "Sensitivity", "Specificity"]:
        our_row[k] = our_metrics.get(k, np.nan)
    all_results.append(our_row)

    labels = df["label"].values

    # Random baseline
    logger.info("Computing random baseline...")
    rand_metrics = random_baseline(labels, seed)
    all_results.append({"Method": "Random", "Type": "Random", **rand_metrics})

    # Seed match baseline
    logger.info("Computing seed match baseline...")
    try:
        seed_metrics = seed_match_baseline(df)
        all_results.append({"Method": "Seed Match", "Type": "Rule", **seed_metrics})
    except Exception as e:
        logger.error(f"Seed match baseline failed: {e}")

    # miRBench predictors
    if run_mirbench:
        mirbench_results = run_mirbench_predictors(
            df, sample_size=mirbench_sample_size, seed=seed
        )
        type_map = {
            "TargetScanCnn_McGeary2019": "CNN",
            "TargetNet_Min2021": "DL",
            "miRBind_Klimentova2022": "DL",
            "CnnMirTarget_Zheng2020": "CNN",
            "miRNA_CNN_Hejret2023": "CNN",
            "InteractionAwareModel_Yang2024": "DL+Attn",
            "RNACofold": "Thermodynamic",
            "Seed8mer": "Rule",
            "Seed7mer": "Rule",
            "Seed6mer": "Rule",
            "Seed6merBulgeOrMismatch": "Rule",
        }
        for name, metrics in mirbench_results.items():
            all_results.append({
                "Method": name,
                "Type": type_map.get(name, "DL"),
                **metrics,
            })

    # Mimosa
    mimosa_result = run_mimosa(df, seed=seed)
    if mimosa_result:
        all_results.append({"Method": "Mimosa", "Type": "DL+Attn", **mimosa_result})

    # External tools
    if run_external_tools:
        rnahybrid_result = run_rnahybrid(df, sample_size=external_sample_size, seed=seed)
        if rnahybrid_result:
            all_results.append({
                "Method": "RNAhybrid", "Type": "MFE", **rnahybrid_result
            })

        miranda_result = run_miranda(df, sample_size=external_sample_size, seed=seed)
        if miranda_result:
            all_results.append({
                "Method": "miRanda", "Type": "Complement+MFE", **miranda_result
            })

    # Aggregate
    comparison_df = pd.DataFrame(all_results)

    # Sort by AUROC in descending order
    comparison_df = comparison_df.sort_values("AUROC", ascending=False).reset_index(
        drop=True
    )

    return comparison_df
