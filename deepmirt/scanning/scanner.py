"""Core genome-wide scanning engine for DeepMiRT."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from deepmirt.scanning._utils import (
    _iter_fasta_records,
    _normalize_dna,
    extract_window,
)
from deepmirt.scanning.site_finder import find_all_seed_sites

logger = logging.getLogger(__name__)


@dataclass
class ScanHit:
    """A single predicted binding site."""

    mirna_id: str
    target_id: str
    position: int  # binding site center position on target (0-based)
    probability: float
    seed_type: str  # "8mer"/"7mer-m8"/"7mer-A1"/"6mer"/"window"
    window_seq: str  # 40nt window fed to model
    mirna_seq: str
    target_length: int


@dataclass
class TargetScanResult:
    """All hits for one miRNA-target pair."""

    mirna_id: str
    target_id: str
    hits: list[ScanHit] = field(default_factory=list)
    target_length: int = 0
    mirna_length: int = 0


class TargetScanner:
    """Genome-wide miRNA target site scanner.

    Loads the DeepMiRT model once, then scans targets using one of three modes:
    - seed: Only score positions with seed matches (fastest)
    - hybrid: Seed matches + sliding window to fill gaps (default)
    - exhaustive: Stride-1 sliding window over entire target (slowest)

    Args:
        device: Inference device ("cpu" or "cuda").
        batch_size: GPU batch size for inference.
        prob_threshold: Minimum probability to report a hit.
        scan_mode: One of "seed", "hybrid", "exhaustive".
        stride: Window stride for hybrid/exhaustive modes.
        window: Window size (default 40, matching training).
        top_k: If set, keep only top-K hits per miRNA-target pair.
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 512,
        prob_threshold: float = 0.5,
        scan_mode: str = "hybrid",
        stride: int = 20,
        window: int = 40,
        top_k: int | None = None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.prob_threshold = prob_threshold
        self.scan_mode = scan_mode
        self.stride = stride
        self.window = window
        self.top_k = top_k

        self._model = None
        self._config = None

    def _ensure_model(self) -> None:
        """Load model on first use (reuses predict.py's cached model)."""
        if self._model is not None:
            return

        from deepmirt.predict import _get_cached_model

        lit_model, alphabet, _, _ = _get_cached_model(self.device)
        self._model = lit_model
        self._alphabet = alphabet
        logger.info("Model loaded for scanning")

    def _get_candidate_positions(
        self, mirna_seq: str, target_seq: str
    ) -> list[tuple[int, str]]:
        """Get candidate positions to score based on scan mode.

        Returns list of (position, seed_type) tuples.
        """
        target = _normalize_dna(target_seq)
        target_len = len(target)

        if self.scan_mode == "seed":
            # Only seed match positions
            sites = find_all_seed_sites(mirna_seq, target)
            return [(s.position, s.seed_type) for s in sites]

        elif self.scan_mode == "exhaustive":
            # Every position with stride=stride (default 1 for exhaustive in CLI)
            positions = []
            for pos in range(0, max(1, target_len - self.window + 1), self.stride):
                positions.append((pos + self.window // 2, "window"))
            return positions

        else:  # hybrid
            # Seed sites + sliding window to fill gaps
            sites = find_all_seed_sites(mirna_seq, target)
            seed_positions = {s.position for s in sites}
            candidates = [(s.position, s.seed_type) for s in sites]

            # Add sliding window positions that aren't near a seed site
            for pos in range(0, max(1, target_len - self.window + 1), self.stride):
                center = pos + self.window // 2
                # Skip if close to an existing seed site
                if not any(abs(center - sp) < self.stride // 2 for sp in seed_positions):
                    candidates.append((center, "window"))

            candidates.sort(key=lambda x: x[0])
            return candidates

    def _batch_predict(
        self, mirna_seqs: list[str], target_seqs: list[str]
    ) -> np.ndarray:
        """Run batch inference on sequence pairs."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        alphabet = self._alphabet
        batch_converter = alphabet.get_batch_converter()
        padding_idx = alphabet.padding_idx

        def _to_rna(seq: str) -> str:
            return seq.upper().replace("T", "U")

        all_probs = []
        n_samples = len(mirna_seqs)

        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                batch_mirna = mirna_seqs[i : i + self.batch_size]
                batch_target = target_seqs[i : i + self.batch_size]

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

                mirna_padded = mirna_padded.to(self.device)
                target_stacked = target_stacked.to(self.device)
                attn_mask_mirna = attn_mask_mirna.to(self.device)
                attn_mask_target = attn_mask_target.to(self.device)

                logits = self._model.model(
                    mirna_padded, target_stacked, attn_mask_mirna, attn_mask_target
                )
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs)

    def scan(
        self,
        mirna_fasta: str | Path | dict[str, str],
        target_fasta: str | Path,
        output_prefix: str | Path | None = None,
    ) -> list[TargetScanResult]:
        """Scan targets for miRNA binding sites.

        Args:
            mirna_fasta: Path to miRNA FASTA, or dict of {id: seq}.
            target_fasta: Path to target FASTA.
            output_prefix: If given, write _details.txt, _summary.tsv, _hits.tsv.

        Returns:
            List of TargetScanResult, one per miRNA-target pair with hits.
        """
        self._ensure_model()

        # Load miRNAs
        if isinstance(mirna_fasta, dict):
            mirna_dict = {k: _normalize_dna(v) for k, v in mirna_fasta.items()}
        else:
            mirna_dict = {}
            for header, seq in _iter_fasta_records(Path(mirna_fasta)):
                mid = header.split()[0]
                mirna_dict[mid] = _normalize_dna(seq)

        logger.info(f"Loaded {len(mirna_dict)} miRNA(s)")

        # Process targets in streaming fashion
        all_results: list[TargetScanResult] = []

        # Accumulate pairs for batch inference
        pending_mirna_seqs: list[str] = []
        pending_target_seqs: list[str] = []
        pending_meta: list[tuple[str, str, int, str, str, int]] = []
        # Each meta entry: (mirna_id, target_id, position, seed_type, mirna_seq, target_length)

        flush_size = 50_000
        target_count = 0

        for header, raw_seq in _iter_fasta_records(Path(target_fasta)):
            target_id = header.split()[0]
            target_seq = _normalize_dna(raw_seq)
            target_len = len(target_seq)
            target_count += 1

            if target_len < 10:
                continue

            for mirna_id, mirna_seq in mirna_dict.items():
                candidates = self._get_candidate_positions(mirna_seq, target_seq)

                for pos, seed_type in candidates:
                    window_seq = extract_window(target_seq, pos, self.window)
                    pending_mirna_seqs.append(mirna_seq)
                    pending_target_seqs.append(window_seq)
                    pending_meta.append(
                        (mirna_id, target_id, pos, seed_type, mirna_seq, target_len)
                    )

                # Flush when accumulated enough
                if len(pending_mirna_seqs) >= flush_size:
                    self._flush_predictions(
                        pending_mirna_seqs, pending_target_seqs,
                        pending_meta, all_results,
                    )
                    pending_mirna_seqs.clear()
                    pending_target_seqs.clear()
                    pending_meta.clear()

            if target_count % 100 == 0:
                logger.info(f"  Processed {target_count} targets...")

        # Final flush
        if pending_mirna_seqs:
            self._flush_predictions(
                pending_mirna_seqs, pending_target_seqs,
                pending_meta, all_results,
            )

        logger.info(
            f"Scanning complete: {target_count} targets, "
            f"{len(all_results)} pairs with hits"
        )

        # Write output files
        if output_prefix is not None:
            from deepmirt.scanning.output_formatter import (
                write_details_txt,
                write_hits_tsv,
                write_summary_tsv,
            )

            prefix = str(output_prefix)
            write_details_txt(
                all_results, f"{prefix}_details.txt",
                scan_mode=self.scan_mode,
                threshold=self.prob_threshold,
                stride=self.stride,
            )
            write_hits_tsv(all_results, f"{prefix}_hits.tsv")
            write_summary_tsv(all_results, f"{prefix}_summary.tsv")
            logger.info(f"Results written to {prefix}_details.txt, _hits.tsv, _summary.tsv")

        return all_results

    def _flush_predictions(
        self,
        mirna_seqs: list[str],
        target_seqs: list[str],
        meta: list[tuple[str, str, int, str, str, int]],
        results: list[TargetScanResult],
    ) -> None:
        """Run inference on accumulated pairs and collect hits."""
        logger.info(f"  Running inference on {len(mirna_seqs)} windows...")
        probs = self._batch_predict(mirna_seqs, target_seqs)

        # Group hits by (mirna_id, target_id)
        pair_hits: dict[tuple[str, str], list[ScanHit]] = {}
        pair_info: dict[tuple[str, str], tuple[int, int]] = {}

        for idx, (mirna_id, target_id, pos, seed_type, mirna_seq, target_len) in enumerate(meta):
            prob = float(probs[idx])
            if prob < self.prob_threshold:
                continue

            key = (mirna_id, target_id)
            if key not in pair_hits:
                pair_hits[key] = []
                pair_info[key] = (len(mirna_seq), target_len)

            pair_hits[key].append(ScanHit(
                mirna_id=mirna_id,
                target_id=target_id,
                position=pos,
                probability=prob,
                seed_type=seed_type,
                window_seq=target_seqs[idx],
                mirna_seq=mirna_seq,
                target_length=target_len,
            ))

        for key, hits in pair_hits.items():
            # Sort by probability descending
            hits.sort(key=lambda h: h.probability, reverse=True)

            # Apply top_k filter
            if self.top_k is not None:
                hits = hits[: self.top_k]

            mirna_len, target_len = pair_info[key]
            results.append(TargetScanResult(
                mirna_id=key[0],
                target_id=key[1],
                hits=hits,
                target_length=target_len,
                mirna_length=mirna_len,
            ))
