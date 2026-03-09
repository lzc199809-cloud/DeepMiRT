"""Output formatters for genome-wide scanning results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from deepmirt.scanning._utils import _normalize_dna

if TYPE_CHECKING:
    from deepmirt.scanning.scanner import TargetScanResult


def _generate_alignment(mirna_seq: str, window_seq: str, seed_type: str) -> str:
    """Generate miRanda-style ASCII alignment between miRNA and target window.

    The miRNA is displayed 3'->5' (reversed), aligned against the target 5'->3'.
    Watson-Crick pairs are marked with '|', G:U wobble with ':', mismatches with ' '.
    The seed region (positions 2-8 from 5' end) is shown in UPPERCASE.

    The alignment is anchored so the miRNA seed region matches the seed site
    in the target window (which is centered on the seed position).
    """
    mirna = _normalize_dna(mirna_seq)
    target = _normalize_dna(window_seq)

    # Reverse miRNA for 3'->5' display
    mirna_rev = mirna[::-1]
    mirna_len = len(mirna)

    # In the reversed miRNA, the seed region (original 5'-end positions 1-7)
    # maps to reversed positions (mirna_len-8) to (mirna_len-2).
    seed_start_rev = mirna_len - 8  # first seed position in reversed miRNA

    # The window was extracted centered on the seed match, so the seed match
    # is near the center of the window. Find the core seed complement (7mer-m8
    # = RC of miRNA positions 2-8) to anchor the alignment correctly.
    from deepmirt.scanning._utils import get_mirna_seed
    _, seed_7mer_m8, _, seed_6mer = get_mirna_seed(mirna_seq)

    # Find the 7mer-m8 (RC of positions 2-8) in the window for anchoring
    seed_pos_in_window = -1
    if seed_7mer_m8:
        seed_pos_in_window = target.find(seed_7mer_m8)
    if seed_pos_in_window == -1 and seed_6mer:
        seed_pos_in_window = target.find(seed_6mer)
    if seed_pos_in_window == -1:
        # No seed found — center alignment on window midpoint
        seed_pos_in_window = max(0, len(target) // 2 - 3)

    # The 7mer-m8 on the target aligns with reversed miRNA positions seed_start_rev
    # to seed_start_rev+6. Anchor accordingly.
    target_offset = seed_pos_in_window - seed_start_rev

    mirna_display = []
    target_display = []
    match_display = []

    for i in range(mirna_len):
        t_idx = target_offset + i
        m_nt = mirna_rev[i]

        # Original 0-based position in the miRNA
        orig_pos = mirna_len - 1 - i
        in_seed = 1 <= orig_pos <= 7

        if 0 <= t_idx < len(target):
            t_nt = target[t_idx]
        else:
            t_nt = " "

        if in_seed:
            mirna_display.append(m_nt.upper())
            target_display.append(t_nt.upper() if t_nt != " " else " ")
        else:
            mirna_display.append(m_nt.lower())
            target_display.append(t_nt.lower() if t_nt != " " else " ")

        # Check complementarity
        if t_nt == " ":
            match_display.append(" ")
        else:
            m_complement = {"A": "T", "T": "A", "C": "G", "G": "C"}.get(m_nt, "N")
            if t_nt == m_complement:
                match_display.append("|")
            elif (m_nt == "G" and t_nt == "T") or (m_nt == "T" and t_nt == "G"):
                match_display.append(":")
            else:
                match_display.append(" ")

    mirna_str = "".join(mirna_display)
    match_str = "".join(match_display)
    target_str = "".join(target_display)

    lines = [
        f"    miRNA  3' ...{mirna_str}... 5'",
        f"              {' ' * 3}{match_str}",
        f"    Target 5' ...{target_str}... 3'",
    ]
    return "\n".join(lines)


def write_details_txt(
    results: list[TargetScanResult],
    output_path: str | Path,
    scan_mode: str = "hybrid",
    threshold: float = 0.5,
    stride: int = 20,
) -> None:
    """Write detailed TXT output with ASCII alignments.

    Format: {prefix}_details.txt
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DeepMiRT Genome-Wide Scanning Results\n")
        f.write(f"Mode: {scan_mode} | Threshold: {threshold} | Stride: {stride}\n")
        f.write("=" * 80 + "\n\n")

        total_hits = sum(len(r.hits) for r in results)
        total_pairs = len(results)
        f.write(f"Total miRNA-target pairs scanned: {total_pairs}\n")
        f.write(f"Total hits above threshold: {total_hits}\n\n")

        for result in results:
            if not result.hits:
                continue

            f.write("-" * 80 + "\n")
            f.write(
                f"Scanning: {result.mirna_id} vs {result.target_id} "
                f"({result.target_length} nt)\n"
            )
            f.write(f"  miRNA length: {result.mirna_length} nt\n")
            f.write(f"  Hits found: {len(result.hits)}\n\n")

            for hit in result.hits:
                f.write(
                    f"  Hit at position {hit.position}, "
                    f"Prob: {hit.probability:.4f}, "
                    f"Seed: {hit.seed_type}\n\n"
                )
                alignment = _generate_alignment(
                    hit.mirna_seq, hit.window_seq, hit.seed_type
                )
                f.write(alignment + "\n\n")

            f.write("\n")


def write_hits_tsv(
    results: list[TargetScanResult],
    output_path: str | Path,
) -> None:
    """Write per-hit TSV output.

    Columns: mirna_id, target_id, position, probability, seed_type,
             window_start, window_end, window_seq
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "mirna_id", "target_id", "position", "probability",
            "seed_type", "window_start", "window_end", "window_seq",
        ])

        for result in results:
            for hit in result.hits:
                window_start = max(0, hit.position - 20)
                window_end = window_start + 40
                writer.writerow([
                    hit.mirna_id,
                    hit.target_id,
                    hit.position,
                    f"{hit.probability:.6f}",
                    hit.seed_type,
                    window_start,
                    window_end,
                    hit.window_seq,
                ])


def write_summary_tsv(
    results: list[TargetScanResult],
    output_path: str | Path,
) -> None:
    """Write per-target summary TSV output.

    Columns: mirna_id, target_id, num_hits, max_prob, mean_prob,
             mirna_len, target_len, hit_positions, seed_types
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "mirna_id", "target_id", "num_hits", "max_prob", "mean_prob",
            "mirna_len", "target_len", "hit_positions", "seed_types",
        ])

        for result in results:
            if not result.hits:
                continue

            probs = [h.probability for h in result.hits]
            positions = ",".join(str(h.position) for h in result.hits)
            seed_types = ",".join(h.seed_type for h in result.hits)

            writer.writerow([
                result.mirna_id,
                result.target_id,
                len(result.hits),
                f"{max(probs):.6f}",
                f"{sum(probs) / len(probs):.6f}",
                result.mirna_length,
                result.target_length,
                positions,
                seed_types,
            ])
