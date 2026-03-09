#!/usr/bin/env python3
"""
Inference engine: load checkpoint and generate prediction DataFrame on the test set.

Independent of Lightning trainer.test(), performs batch inference directly and
retains all metadata. Prediction results are cached as parquet to avoid repeated inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    ckpt_path: str,
    config_path: str,
    device: str = "cuda",
):
    """
    Load a trained model from checkpoint.

    Args:
        ckpt_path: path to the checkpoint file
        config_path: path to the training config YAML
        device: inference device

    Returns:
        (model, config) tuple
    """
    from insect_mirna_target.training.lightning_module import MiRNATargetLitModule

    with open(config_path) as f:
        config = yaml.safe_load(f)

    lit_model = MiRNATargetLitModule.load_from_checkpoint(
        ckpt_path, config=config, map_location=device
    )
    lit_model.eval()
    lit_model.to(device)
    return lit_model, config


def run_inference(
    ckpt_path: str,
    config_path: str,
    test_csv_path: str,
    batch_size: int = 256,
    num_workers: int = 8,
    device: str = "cuda",
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Run model inference on the test set, returning a DataFrame with predictions and metadata.

    If cache_path exists and is non-empty, loads cached results directly.

    Args:
        ckpt_path: path to the checkpoint
        config_path: path to the config YAML
        test_csv_path: path to test.csv
        batch_size: inference batch size
        num_workers: number of DataLoader worker threads
        device: inference device
        cache_path: cache file path (parquet), None to disable caching

    Returns:
        DataFrame with columns:
            mirna_seq, target_fragment_40nt, label, prob, pred, logit,
            species, mirna_name, target_gene_name, evidence_type, source_database
    """
    # Check cache (supports both parquet and csv formats)
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached predictions from {cache_path}")
        if cache_path.endswith(".parquet"):
            return pd.read_parquet(cache_path)
        else:
            return pd.read_csv(cache_path)

    logger.info(f"Loading model from {ckpt_path}")
    lit_model, config = load_model_from_checkpoint(ckpt_path, config_path, device)

    # Load data (using DataModule approach for consistency)
    import fm

    from insect_mirna_target.data_module.datamodule import MiRNATargetDataModule
    from insect_mirna_target.data_module.dataset import MiRNATargetDataset

    _, alphabet = fm.pretrained.rna_fm_t12()
    del _
    padding_idx = alphabet.padding_idx

    dataset = MiRNATargetDataset(test_csv_path, alphabet)

    # Use the DataModule's collate_fn logic
    dm = MiRNATargetDataModule.__new__(MiRNATargetDataModule)
    dm._padding_idx = padding_idx

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dm._collate_fn,
    )

    # Inference
    all_logits = []
    all_labels = []
    all_metadata = {
        "species": [],
        "mirna_name": [],
        "target_gene_name": [],
        "evidence_type": [],
        "source_database": [],
    }

    logger.info(f"Running inference on {len(dataset)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            mirna_tokens = batch["mirna_tokens"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            labels = batch["labels"]
            attn_mask_mirna = batch["attention_mask_mirna"].to(device)
            attn_mask_target = batch["attention_mask_target"].to(device)

            logits = lit_model.model(
                mirna_tokens, target_tokens, attn_mask_mirna, attn_mask_target
            )
            logits = logits.squeeze(-1).cpu()

            all_logits.append(logits)
            all_labels.append(labels)

            metadata = batch.get("metadata", {})
            for key in all_metadata:
                if key in metadata:
                    all_metadata[key].extend(metadata[key])
                else:
                    all_metadata[key].extend([""] * len(labels))

            if (batch_idx + 1) % 500 == 0:
                logger.info(
                    f"  Processed {(batch_idx + 1) * batch_size} / {len(dataset)}"
                )

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
    all_preds = (all_probs >= 0.5).astype(int)

    # Build raw sequence columns (read directly from CSV)
    raw_df = pd.read_csv(
        test_csv_path,
        usecols=["mirna_seq", "target_fragment_40nt"],
        dtype=str,
    )

    result_df = pd.DataFrame(
        {
            "mirna_seq": raw_df["mirna_seq"].values,
            "target_fragment_40nt": raw_df["target_fragment_40nt"].values,
            "label": all_labels.astype(int),
            "prob": all_probs,
            "pred": all_preds,
            "logit": all_logits,
            "species": all_metadata["species"],
            "mirna_name": all_metadata["mirna_name"],
            "target_gene_name": all_metadata["target_gene_name"],
            "evidence_type": all_metadata["evidence_type"],
            "source_database": all_metadata["source_database"],
        }
    )

    # Cache results (prefer parquet, fallback to csv)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            if cache_path.endswith(".parquet"):
                result_df.to_parquet(cache_path, index=False)
            else:
                result_df.to_csv(cache_path, index=False)
        except ImportError:
            # pyarrow not installed, fallback to csv
            csv_path = cache_path.replace(".parquet", ".csv")
            result_df.to_csv(csv_path, index=False)
            logger.info(f"pyarrow not available, saved as CSV: {csv_path}")
            cache_path = csv_path
        logger.info(f"Predictions cached to {cache_path}")

    logger.info(
        f"Inference complete: {len(result_df)} samples, "
        f"pos={result_df['label'].sum()}, neg={(result_df['label'] == 0).sum()}"
    )
    return result_df


def predict_on_sequences(
    ckpt_path: str,
    config_path: str,
    mirna_seqs: list[str],
    target_seqs: list[str],
    batch_size: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run inference on arbitrary miRNA + target sequence pairs.

    Used to run our model on external data such as miRBench standard benchmark datasets.
    Sequences are automatically converted to RNA format (T->U).

    Args:
        ckpt_path: path to the checkpoint
        config_path: path to the config YAML
        mirna_seqs: list of miRNA sequences (DNA or RNA format accepted)
        target_seqs: list of target sequences (DNA or RNA format, should be 40nt)
        batch_size: inference batch size
        device: inference device

    Returns:
        numpy array of predicted probabilities, shape (n_samples,)
    """
    import fm
    from torch.nn.utils.rnn import pad_sequence

    logger.info(f"Loading model from {ckpt_path}")
    lit_model, config = load_model_from_checkpoint(ckpt_path, config_path, device)

    _, alphabet = fm.pretrained.rna_fm_t12()
    del _
    batch_converter = alphabet.get_batch_converter()
    padding_idx = alphabet.padding_idx

    def _to_rna(seq: str) -> str:
        return seq.upper().replace("T", "U")

    all_probs = []
    n_samples = len(mirna_seqs)
    logger.info(f"Running inference on {n_samples} sequences...")

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_mirna = mirna_seqs[i : i + batch_size]
            batch_target = target_seqs[i : i + batch_size]

            mirna_tokens_list = []
            target_tokens_list = []
            for m_seq, t_seq in zip(batch_mirna, batch_target):
                m_rna = _to_rna(str(m_seq))
                t_rna = _to_rna(str(t_seq))
                _, _, m_tok = batch_converter([("m", m_rna)])
                _, _, t_tok = batch_converter([("t", t_rna)])
                mirna_tokens_list.append(m_tok[0])
                target_tokens_list.append(t_tok[0])

            mirna_padded = pad_sequence(
                mirna_tokens_list, batch_first=True, padding_value=padding_idx
            )
            target_stacked = torch.stack(target_tokens_list)

            attn_mask_mirna = (mirna_padded != padding_idx).long()
            attn_mask_target = torch.ones_like(target_stacked, dtype=torch.long)

            mirna_padded = mirna_padded.to(device)
            target_stacked = target_stacked.to(device)
            attn_mask_mirna = attn_mask_mirna.to(device)
            attn_mask_target = attn_mask_target.to(device)

            logits = lit_model.model(
                mirna_padded, target_stacked, attn_mask_mirna, attn_mask_target
            )
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.append(probs)

            if (i // batch_size + 1) % 100 == 0:
                logger.info(f"  Processed {min(i + batch_size, n_samples)} / {n_samples}")

    return np.concatenate(all_probs)
