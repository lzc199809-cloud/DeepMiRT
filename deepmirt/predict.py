#!/usr/bin/env python3
"""
Public prediction API for DeepMiRT.

Provides simple interfaces for miRNA-target interaction prediction:
- predict(): Python API for sequence pairs
- predict_from_csv(): Batch prediction from CSV files
- cli_main(): Command-line entry point

Model weights are automatically downloaded from Hugging Face Hub on first use.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Hugging Face Hub model repository
HF_REPO_ID = "liuliu2333/deepmirt"
HF_CKPT_FILENAME = "epoch=27-val_auroc=0.9612.ckpt"
HF_CONFIG_FILENAME = "config.yaml"

# Valid nucleotide characters (before T→U conversion)
_VALID_BASES = re.compile(r"^[AUGCTaugct]+$")

# Module-level model cache (avoids reloading 495 MB on every call)
_model_cache: dict = {}


def _get_model_files() -> tuple[str, str]:
    """Download model checkpoint and config from Hugging Face Hub (cached locally)."""
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CKPT_FILENAME)
    config_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CONFIG_FILENAME)
    return ckpt_path, config_path


def _get_cached_model(device: str):
    """Load model and alphabet, caching for subsequent calls."""
    if device not in _model_cache:
        import fm

        from deepmirt.evaluation.predict import load_model_from_checkpoint

        ckpt_path, config_path = _get_model_files()
        logger.info("Loading DeepMiRT model (first call, will be cached)...")
        lit_model, config = load_model_from_checkpoint(ckpt_path, config_path, device)
        _, alphabet = fm.pretrained.rna_fm_t12()
        _model_cache[device] = (lit_model, alphabet, ckpt_path, config_path)
        logger.info("Model loaded and cached.")

    return _model_cache[device]


def _validate_sequences(
    mirna_seqs: list[str], target_seqs: list[str]
) -> tuple[list[str], list[str]]:
    """Validate and clean input sequences."""
    cleaned_mirna = []
    cleaned_target = []

    for i, (m, t) in enumerate(zip(mirna_seqs, target_seqs)):
        m = str(m).strip().upper()
        t = str(t).strip().upper()

        if not m:
            raise ValueError(f"Empty miRNA sequence at index {i}")
        if not t:
            raise ValueError(f"Empty target sequence at index {i}")

        if not _VALID_BASES.match(m):
            invalid = set(m) - set("AUGCT")
            raise ValueError(
                f"miRNA at index {i} contains invalid characters: {invalid}. "
                f"Only A/U/G/C/T are allowed."
            )
        if not _VALID_BASES.match(t):
            invalid = set(t) - set("AUGCT")
            raise ValueError(
                f"Target at index {i} contains invalid characters: {invalid}. "
                f"Only A/U/G/C/T are allowed."
            )

        cleaned_mirna.append(m)
        cleaned_target.append(t)

    # Warn about unusual lengths (non-blocking)
    mirna_lens = [len(s) for s in cleaned_mirna]
    target_lens = [len(s) for s in cleaned_target]
    if any(n < 15 or n > 30 for n in mirna_lens):
        warnings.warn(
            "Some miRNA sequences have unusual length (expected 18-25 nt). "
            "Results may be less reliable.",
            stacklevel=3,
        )
    if any(n != 40 for n in target_lens):
        warnings.warn(
            "Some target sequences are not 40 nt. The model was trained on 40-nt "
            "target fragments. Results may be less reliable for other lengths.",
            stacklevel=3,
        )

    return cleaned_mirna, cleaned_target


def predict(
    mirna_seqs: list[str],
    target_seqs: list[str],
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Predict miRNA-target interaction probabilities.

    Automatically downloads model weights from Hugging Face Hub on first call.
    The model is cached in memory for subsequent calls.
    Sequences can be in DNA (T) or RNA (U) format -- conversion is handled internally.

    Args:
        mirna_seqs: List of miRNA sequences (typically 18-25 nt).
        target_seqs: List of target site sequences (40 nt recommended).
        device: Inference device ("cpu" or "cuda").
        batch_size: Batch size for inference.

    Returns:
        Numpy array of interaction probabilities, shape (n_samples,).
        Values range from 0 (no interaction) to 1 (strong interaction).

    Example:
        >>> from deepmirt import predict
        >>> probs = predict(
        ...     mirna_seqs=["UGAGGUAGUAGGUUGUAUAGUU"],
        ...     target_seqs=["ACUGCAGCAUAUCUACUAUUUGCUACUGUAACCAUUGAUCU"],
        ... )
        >>> print(f"Interaction probability: {probs[0]:.4f}")
    """
    if len(mirna_seqs) != len(target_seqs):
        raise ValueError(
            f"mirna_seqs and target_seqs must have the same length, "
            f"got {len(mirna_seqs)} and {len(target_seqs)}"
        )
    if len(mirna_seqs) == 0:
        return np.array([])

    mirna_seqs, target_seqs = _validate_sequences(mirna_seqs, target_seqs)

    from deepmirt.evaluation.predict import predict_on_sequences

    lit_model, alphabet, ckpt_path, config_path = _get_cached_model(device)

    return predict_on_sequences(
        ckpt_path=ckpt_path,
        config_path=config_path,
        mirna_seqs=mirna_seqs,
        target_seqs=target_seqs,
        batch_size=batch_size,
        device=device,
        _lit_model=lit_model,
        _alphabet=alphabet,
    )


def predict_from_csv(
    csv_path: str,
    output_path: str | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    mirna_col: str = "mirna_seq",
    target_col: str = "target_seq",
) -> pd.DataFrame:
    """
    Batch prediction from a CSV file.

    The CSV must contain columns for miRNA and target sequences.

    Args:
        csv_path: Path to input CSV file.
        output_path: Path to save results CSV. If None, results are only returned.
        device: Inference device ("cpu" or "cuda").
        batch_size: Batch size for inference.
        mirna_col: Column name for miRNA sequences.
        target_col: Column name for target sequences.

    Returns:
        DataFrame with original columns plus 'probability' and 'prediction'.
    """
    df = pd.read_csv(csv_path)

    if mirna_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{mirna_col}' and '{target_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    mirna_seqs = df[mirna_col].astype(str).tolist()
    target_seqs = df[target_col].astype(str).tolist()

    probs = predict(mirna_seqs, target_seqs, device=device, batch_size=batch_size)

    df["probability"] = probs
    df["prediction"] = (probs >= 0.5).astype(int)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    return df


def cli_main() -> None:
    """Command-line entry point for deepmirt-predict."""
    parser = argparse.ArgumentParser(
        prog="deepmirt-predict",
        description="DeepMiRT: Predict miRNA-target interactions",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single prediction
    single = subparsers.add_parser("single", help="Predict a single miRNA-target pair")
    single.add_argument("--mirna", required=True, help="miRNA sequence")
    single.add_argument("--target", required=True, help="Target sequence (40 nt)")
    single.add_argument("--device", default="cpu", help="Device (cpu or cuda)")

    # Batch prediction
    batch = subparsers.add_parser("batch", help="Batch prediction from CSV")
    batch.add_argument("--input", required=True, help="Input CSV path")
    batch.add_argument("--output", required=True, help="Output CSV path")
    batch.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    batch.add_argument("--batch-size", type=int, default=256, help="Batch size")
    batch.add_argument("--mirna-col", default="mirna_seq", help="miRNA column name")
    batch.add_argument("--target-col", default="target_seq", help="Target column name")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "single":
        probs = predict([args.mirna], [args.target], device=args.device)
        prob = probs[0]
        label = "INTERACTION" if prob >= 0.5 else "NO INTERACTION"
        print(f"Probability: {prob:.4f}")
        print(f"Prediction:  {label}")
    elif args.command == "batch":
        df = predict_from_csv(
            csv_path=args.input,
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size,
            mirna_col=args.mirna_col,
            target_col=args.target_col,
        )
        print(f"Processed {len(df)} samples. Results saved to {args.output}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
