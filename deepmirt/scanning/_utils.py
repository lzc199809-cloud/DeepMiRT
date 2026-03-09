"""Utility functions for the scanning module.

These are copied from deepmirt.pipeline.site_utils so that the scanning
package can be used stand-alone without the pipeline package installed.
"""

from __future__ import annotations

from pathlib import Path


def _normalize_dna(seq: str) -> str:
    seq = str(seq).upper().replace("U", "T")
    out = []
    for ch in seq:
        if ch in {"A", "C", "G", "T", "N"}:
            out.append(ch)
        elif ch.isspace():
            continue
        else:
            out.append("N")
    return "".join(out)


def _reverse_complement(seq: str) -> str:
    trans = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"})
    return seq.translate(trans)[::-1]


def _iter_fasta_records(fasta_path: Path):
    header = None
    seq_parts: list[str] = []
    with open(fasta_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)


def get_mirna_seed(mirna_seq: str) -> tuple[str, str, str, str]:
    """Get seed match patterns from miRNA sequence (positions 2-8 from 5' end).
    Returns (seed_8mer, seed_7mer_m8, seed_7mer_A1, seed_6mer) as target-side complements.
    Complement: A<->T, G<->C. These are the patterns to search in the 3'UTR.
    """
    seq = _normalize_dna(mirna_seq)
    seed7 = seq[1:8]
    seed6 = seq[1:7]

    if len(seed7) < 7:
        return "", "", "", ""

    rc7 = _reverse_complement(seed7)
    rc6 = _reverse_complement(seed6)

    seed_8mer = f"A{rc7}"
    seed_7mer_m8 = rc7
    seed_7mer_A1 = f"A{rc6}"
    seed_6mer = rc6
    return seed_8mer, seed_7mer_m8, seed_7mer_A1, seed_6mer


def extract_window(utr_seq: str, site_pos: int, window: int = 40, clamp: bool = False) -> str:
    """Extract window centered on site_pos.
    Pad with 'N' if window exceeds UTR boundaries.
    When clamp=True and seq length is at least `window`, shift to stay in bounds.
    Returns exactly `window` nucleotides.
    """
    seq = _normalize_dna(utr_seq)
    if window <= 0:
        return ""
    if not seq:
        return "N" * window

    site_pos = max(0, int(site_pos))
    half = window // 2
    start = site_pos - half
    end = start + window

    if clamp and len(seq) >= window:
        if start < 0:
            start = 0
            end = window
        elif end > len(seq):
            end = len(seq)
            start = end - window

    pad_left = max(0, -start)
    pad_right = max(0, end - len(seq))

    clip_start = max(0, start)
    clip_end = min(len(seq), end)
    core = seq[clip_start:clip_end]

    out = ("N" * pad_left) + core + ("N" * pad_right)
    if len(out) < window:
        out += "N" * (window - len(out))
    elif len(out) > window:
        out = out[:window]

    return out
