"""Find all seed match sites on a target sequence for a given miRNA."""

from __future__ import annotations

from dataclasses import dataclass

from deepmirt.pipeline.site_utils import _normalize_dna, get_mirna_seed


@dataclass
class SeedSite:
    """A seed match position on a target sequence."""

    position: int  # 0-based start on target
    seed_type: str  # "8mer", "7mer-m8", "7mer-A1", "6mer"


def find_all_seed_sites(mirna_seq: str, target_seq: str) -> list[SeedSite]:
    """Find ALL seed match positions on the target (not just the best one).

    Searches for each seed type in priority order (8mer > 7mer-m8 > 7mer-A1 > 6mer)
    and returns every occurrence. Positions may overlap across seed types.

    Args:
        mirna_seq: miRNA sequence (DNA or RNA format).
        target_seq: Target/UTR sequence (DNA or RNA format).

    Returns:
        List of SeedSite objects sorted by position, then seed_type priority.
    """
    target = _normalize_dna(target_seq)
    seed_8mer, seed_7mer_m8, seed_7mer_A1, seed_6mer = get_mirna_seed(mirna_seq)

    search_patterns = [
        ("8mer", seed_8mer),
        ("7mer-m8", seed_7mer_m8),
        ("7mer-A1", seed_7mer_A1),
        ("6mer", seed_6mer),
    ]

    sites: list[SeedSite] = []
    seen_positions: set[int] = set()

    for seed_type, pattern in search_patterns:
        if not pattern:
            continue
        start = 0
        while True:
            pos = target.find(pattern, start)
            if pos == -1:
                break
            # Deduplicate: skip if a higher-priority seed already covers this position
            if pos not in seen_positions:
                sites.append(SeedSite(position=pos, seed_type=seed_type))
                seen_positions.add(pos)
            start = pos + 1

    sites.sort(key=lambda s: s.position)
    return sites
