#!/usr/bin/env python3
"""
DeepMiRT Web Demo — Gradio interface for miRNA-target interaction prediction.

Run locally:
    python app.py

Deploy on Hugging Face Spaces:
    Set sdk: gradio in the Space README.md metadata.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model (loaded once at startup)
# ---------------------------------------------------------------------------
_model = None
_alphabet = None
_config = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    """Load model from Hugging Face Hub (cached after first download)."""
    global _model, _alphabet, _config

    if _model is not None:
        return

    import fm
    import torch
    from huggingface_hub import hf_hub_download

    from deepmirt.evaluation.predict import load_model_from_checkpoint

    repo_id = "liuliu2333/deepmirt"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="epoch=27-val_auroc=0.9612.ckpt")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")

    logger.info("Loading model...")
    _model, _config = load_model_from_checkpoint(ckpt_path, config_path, device=_device)
    _, _alphabet = fm.pretrained.rna_fm_t12()
    logger.info("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
_VALID_BASES = set("AUGC")


def _validate_seq(seq: str, name: str, min_len: int = 1, max_len: int = 200) -> str:
    """Validate and clean an RNA/DNA sequence."""
    seq = seq.strip().upper().replace("T", "U")
    if not seq:
        raise gr.Error(f"{name} sequence is empty.")
    if len(seq) < min_len or len(seq) > max_len:
        raise gr.Error(f"{name} must be {min_len}-{max_len} nt, got {len(seq)} nt.")
    invalid = set(seq) - _VALID_BASES
    if invalid:
        raise gr.Error(f"{name} contains invalid characters: {invalid}. Only A/U/G/C/T allowed.")
    return seq


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
def _predict_pair(mirna_seq: str, target_seq: str) -> np.ndarray:
    """Run model inference on a single pair."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    _load_model()

    batch_converter = _alphabet.get_batch_converter()
    padding_idx = _alphabet.padding_idx

    _, _, m_tok = batch_converter([("m", mirna_seq)])
    _, _, t_tok = batch_converter([("t", target_seq)])

    mirna_padded = pad_sequence([m_tok[0]], batch_first=True, padding_value=padding_idx)
    target_stacked = torch.stack([t_tok[0]])

    attn_mask_mirna = (mirna_padded != padding_idx).long().to(_device)
    attn_mask_target = torch.ones_like(target_stacked, dtype=torch.long).to(_device)
    mirna_padded = mirna_padded.to(_device)
    target_stacked = target_stacked.to(_device)

    with torch.no_grad():
        logits = _model.model(mirna_padded, target_stacked, attn_mask_mirna, attn_mask_target)
        prob = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    return prob


def predict_single(mirna_seq: str, target_seq: str):
    """Gradio callback for single prediction."""
    mirna_rna = _validate_seq(mirna_seq, "miRNA", min_len=15, max_len=30)
    target_rna = _validate_seq(target_seq, "Target", min_len=20, max_len=50)

    prob = _predict_pair(mirna_rna, target_rna)
    p = float(prob[0])
    label = "INTERACTION" if p >= 0.5 else "NO INTERACTION"
    color = "#2ecc71" if p >= 0.5 else "#e74c3c"
    details = {
        "probability": round(p, 6),
        "prediction": label,
        "threshold": 0.5,
        "mirna_length": len(mirna_rna),
        "target_length": len(target_rna),
    }
    return (
        f"<div style='text-align:center;padding:20px;'>"
        f"<span style='font-size:48px;font-weight:bold;color:{color};'>{p:.4f}</span><br>"
        f"<span style='font-size:20px;color:{color};'>{label}</span></div>"
    ), details


def predict_batch(file):
    """Gradio callback for batch prediction."""
    if file is None:
        raise gr.Error("Please upload a CSV file.")

    _load_model()

    df = pd.read_csv(file.name)

    mirna_col = None
    target_col = None
    for col in df.columns:
        cl = col.lower().strip()
        if "mirna" in cl:
            mirna_col = col
        elif "target" in cl:
            target_col = col

    if mirna_col is None or target_col is None:
        raise gr.Error(
            "CSV must contain a column with 'mirna' and a column with 'target' in the name. "
            f"Found columns: {list(df.columns)}"
        )

    mirna_seqs = df[mirna_col].astype(str).tolist()
    target_seqs = df[target_col].astype(str).tolist()

    # Validate and convert
    cleaned_mirna = []
    cleaned_target = []
    for i, (m, t) in enumerate(zip(mirna_seqs, target_seqs)):
        m = m.strip().upper().replace("T", "U")
        t = t.strip().upper().replace("T", "U")
        invalid_m = set(m) - _VALID_BASES
        invalid_t = set(t) - _VALID_BASES
        if invalid_m or invalid_t:
            raise gr.Error(f"Row {i}: invalid characters in sequences.")
        cleaned_mirna.append(m)
        cleaned_target.append(t)

    # Batch inference
    import torch
    from torch.nn.utils.rnn import pad_sequence

    batch_converter = _alphabet.get_batch_converter()
    padding_idx = _alphabet.padding_idx
    all_probs = []
    batch_size = 128

    with torch.no_grad():
        for start in range(0, len(cleaned_mirna), batch_size):
            batch_m = cleaned_mirna[start : start + batch_size]
            batch_t = cleaned_target[start : start + batch_size]

            m_toks = []
            t_toks = []
            for ms, ts in zip(batch_m, batch_t):
                _, _, mt = batch_converter([("m", ms)])
                _, _, tt = batch_converter([("t", ts)])
                m_toks.append(mt[0])
                t_toks.append(tt[0])

            mirna_padded = pad_sequence(m_toks, batch_first=True, padding_value=padding_idx)
            target_stacked = torch.stack(t_toks)
            attn_mask_mirna = (mirna_padded != padding_idx).long().to(_device)
            attn_mask_target = torch.ones_like(target_stacked, dtype=torch.long).to(_device)

            logits = _model.model(
                mirna_padded.to(_device),
                target_stacked.to(_device),
                attn_mask_mirna,
                attn_mask_target,
            )
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs)
    df["probability"] = all_probs
    df["prediction"] = (all_probs >= 0.5).astype(int)

    # Save to temp file for download
    out_path = Path(tempfile.mkdtemp()) / "deepmirt_predictions.csv"
    df.to_csv(str(out_path), index=False)
    return str(out_path), df.head(20)


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
EXAMPLES = [
    # [miRNA, target_40nt] - real miRNA-target pairs
    ["UGAGGUAGUAGGUUGUAUAGUU", "ACUGCAGCAUAUCUACUAUUUGCUACUGUAACCAUUGAUCU"],   # let-7a / lin-41
    ["UAAAGUGCUUAUAGUGCAGGUAG", "GCAGCAUUGUACAGGGCUAUCAGAAACUAUUGACACUAAAA"],  # miR-20a / E2F1
    ["UAGCAGCACGUAAAUAUUGGCG", "GCAAUGUUUUCCACAGUGCUUACACAGAAAUAGCAACUUUA"],   # miR-16 / BCL2
    ["CAUCAAAGUGGAGGCCCUCUCU", "AAUGCUUCUAAAUUGAAUCCAAACUGCAGUUUAUUAGUGGU"],   # miR-198 (negative)
    ["UGGAAUGUAAAGAAGUAUGUAU", "UCGAAUCCAUGCAAAACAGCUUGAUUUGUUAGUACACGAAU"],   # miR-1 / HAND2
]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_demo():
    with gr.Blocks(
        title="DeepMiRT: miRNA Target Prediction",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # DeepMiRT: miRNA Target Prediction with RNA Foundation Models

            Predict miRNA-target interactions using RNA-FM embeddings and cross-attention.
            Ranked **#1** on eCLIP benchmarks (AUROC 0.75) and achieves **AUROC 0.96** on our comprehensive test set.

            **Paper:** *coming soon* | **GitHub:** [DeepMiRT](https://github.com/lzc199809-cloud/DeepMiRT) | **Model:** [Hugging Face](https://huggingface.co/liuliu2333/deepmirt)
            """
        )

        with gr.Tab("Single Prediction"):
            with gr.Row():
                with gr.Column():
                    mirna_input = gr.Textbox(
                        label="miRNA Sequence",
                        placeholder="e.g., UGAGGUAGUAGGUUGUAUAGUU",
                        info="18-25 nt. DNA (T) or RNA (U) format accepted.",
                    )
                    target_input = gr.Textbox(
                        label="Target Sequence",
                        placeholder="e.g., ACUGCAGCAUAUCUACUAUUUGCUACUGUAACCAUUGAUCU",
                        info="40 nt recommended. DNA (T) or RNA (U) format accepted.",
                    )
                    predict_btn = gr.Button("Predict", variant="primary")

                with gr.Column():
                    result_html = gr.HTML(label="Prediction Result")
                    result_json = gr.JSON(label="Details")

            predict_btn.click(
                predict_single,
                inputs=[mirna_input, target_input],
                outputs=[result_html, result_json],
            )

            gr.Examples(
                examples=EXAMPLES,
                inputs=[mirna_input, target_input],
                outputs=[result_html, result_json],
                fn=predict_single,
                cache_examples=False,
            )

        with gr.Tab("Batch Prediction"):
            gr.Markdown(
                """
                Upload a CSV file with columns containing **mirna** and **target** in the column names.

                Example format:
                | mirna_seq | target_seq |
                |-----------|------------|
                | UGAGGUAGUAGGUUGUAUAGUU | ACUGCAGCAUAUCUACUAUUUGCUACUGUAACCAUUGAUCU |
                """
            )
            csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
            batch_btn = gr.Button("Run Batch Prediction", variant="primary")
            csv_output = gr.File(label="Download Results")
            preview = gr.Dataframe(label="Preview (first 20 rows)")

            batch_btn.click(
                predict_batch,
                inputs=[csv_input],
                outputs=[csv_output, preview],
            )

        with gr.Tab("About"):
            gr.Markdown(
                """
                ## Model Architecture

                DeepMiRT uses a **shared RNA-FM encoder** (12-layer Transformer, pre-trained on 23M non-coding RNAs)
                to embed both miRNA and target sequences into the same representation space.
                A **cross-attention module** (2 layers, 8 heads) allows the target to attend to the miRNA,
                capturing interaction patterns. The attended representations are pooled and classified
                by an **MLP head** (640 → 256 → 64 → 1).

                ```
                miRNA  → [RNA-FM Encoder] → miRNA embedding  ─────────┐
                                                                       ↓
                Target → [RNA-FM Encoder] → target embedding → [Cross-Attention] → Pool → [MLP] → probability
                ```

                ## Training

                - **Data:** miRNA-target interactions from multiple databases and literature mining
                - **Two-phase training:** Phase 1 (frozen backbone) → Phase 2 (unfreeze top 3 RNA-FM layers)
                - **Hardware:** 2× NVIDIA L20 GPUs, mixed-precision (fp16)
                - **Best checkpoint:** epoch 27, val AUROC = 0.9612

                ## Performance

                | Benchmark | AUROC | Rank |
                |-----------|-------|------|
                | miRBench eCLIP (Klimentova 2022) | 0.7511 | #1/12 |
                | miRBench eCLIP (Manakov 2022) | 0.7543 | #1/12 |
                | miRBench CLASH (Hejret 2023) | 0.6952 | #5/12 |
                | Our test set (813K samples, 16 methods) | 0.9606 | #1/16 |

                ## Citation

                If you use DeepMiRT in your research, please cite:
                ```
                @software{liu2026deepmirt,
                  title={DeepMiRT: miRNA Target Prediction with RNA Foundation Models},
                  author={Liu, Zicheng},
                  year={2026},
                  url={https://github.com/lzc199809-cloud/DeepMiRT}
                }
                ```

                ## License

                MIT License. See [LICENSE](https://github.com/lzc199809-cloud/DeepMiRT/blob/main/LICENSE).
                """
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
