#!/usr/bin/env python3
"""
miRNA-Target Pair Dataset — PyTorch Dataset Implementation

[Data Flow ASCII Diagram]
┌─────────────────────────────────────────────────────────────────────┐
│                  MiRNATargetDataset Data Flow                       │
│                                                                     │
│  CSV file (train.csv / val.csv / test.csv)                          │
│     │                                                               │
│     ▼                                                               │
│  pd.read_csv() ─→ DataFrame (loaded entirely into memory)           │
│     │                                                               │
│     ▼                                                               │
│  __getitem__(idx) ─→ retrieve row idx                               │
│     │                                                               │
│     ├─→ mirna_seq: "ATCGATCG"                                      │
│     │       │                                                       │
│     │       ▼                                                       │
│     │   dna_to_rna() ─→ "AUCGAUCG"  (T→U conversion)               │
│     │       │                                                       │
│     │       ▼                                                       │
│     │   batch_converter([("mirna", "AUCGAUCG")])                    │
│     │       │                                                       │
│     │       ▼                                                       │
│     │   tokens: tensor([0, 4, 7, 5, 6, ...., 2])                   │
│     │           ^^BOS                   ^^EOS                       │
│     │                                                               │
│     ├─→ target_fragment_40nt: "TAGCTAGC..."                         │
│     │       │  (same dna_to_rna + batch_converter pipeline)         │
│     │       ▼                                                       │
│     │   tokens: tensor([0, ..., 2])  (fixed 42 tokens: BOS+40nt+EOS)│
│     │                                                               │
│     └─→ return dict:                                                │
│           {                                                         │
│             'mirna_tokens':  1D LongTensor (variable 17-32)         │
│             'target_tokens': 1D LongTensor (fixed 42)               │
│             'label':         float32 scalar (0.0 or 1.0)            │
│             'metadata':      dict (species, mirna_name, ...)        │
│           }                                                         │
└─────────────────────────────────────────────────────────────────────┘

[RNA-FM batch_converter Input/Output Format]
- Input: List[Tuple[str, str]] = [("label_name", "RNA_sequence")]
  e.g.: [("mirna", "AUCGAUCG")]

- Output: Tuple[List[str], List[str], Tensor]
  - labels: ["mirna"]           — label list (not used by us)
  - strs:   ["AUCGAUCG"]       — raw sequences (not used by us)
  - tokens: tensor([[0, 4, 7, 5, 6, 4, 7, 5, 6, 2]])
             shape = (batch=1, seq_len)
             where 0=BOS(<cls>), 2=EOS(<eos>), 1=PAD(<pad>)
             A=4, C=5, G=6, U=7

- Important: batch_converter already adds BOS and EOS for us!
  So 22nt miRNA → 24 tokens (BOS + 22nt + EOS)
  40nt target → 42 tokens (BOS + 40nt + EOS)
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from insect_mirna_target.data_module.preprocessing import dna_to_rna


class MiRNATargetDataset(Dataset):
    """
    PyTorch Dataset for miRNA-target pairs.

    [Overview]
    Loads miRNA-target sequence pairs from a CSV file, tokenizes them using
    the RNA-FM alphabet, and returns token tensors and labels for training.

    [Usage]
    >>> import fm
    >>> _, alphabet = fm.pretrained.rna_fm_t12()
    >>> ds = MiRNATargetDataset('path/to/train.csv', alphabet)
    >>> sample = ds[0]
    >>> sample['mirna_tokens']   # tensor([0, 4, 7, 5, ..., 2])
    >>> sample['label']          # tensor(1.)

    [Why inherit from torch.utils.data.Dataset?]
    - It is the standard PyTorch interface for data loading
    - After defining __len__ and __getitem__, it can be used with DataLoader
    - DataLoader automatically handles batching, multi-process loading, shuffling, etc.
    """

    def __init__(
        self,
        csv_path: str,
        alphabet,
        max_mirna_len: int = 30,
        max_target_len: int = 40,
    ):
        """
        Initialize the dataset.

        Args:
            csv_path (str): Path to the CSV file, which must contain the following columns:
                - mirna_seq: miRNA sequence (DNA notation)
                - target_fragment_40nt: target fragment sequence (DNA notation)
                - label: binary label (0 or 1)
                - species, mirna_name, target_gene_name: metadata columns
            alphabet: RNA-FM alphabet object that provides tokenization capability
            max_mirna_len (int): maximum nucleotide length for miRNA, default 30
                (actual token count = max_mirna_len + 2, due to BOS and EOS)
            max_target_len (int): maximum nucleotide length for target, default 40
                (actual token count = max_target_len + 2 = 42)

        [Design Decision: Memory Strategy]
        We use pd.read_csv() to load the entire CSV into a DataFrame at once.
        This is the simplest approach — for our data scale (~5.4 million training rows),
        the DataFrame occupies approximately 2-3 GB of memory.

        The current system has 1TB RAM, so this is not an issue at all.

        # Design decision: if memory is limited (e.g., 8GB), consider these alternatives:
        # 1. Byte-offset indexing: first pass records byte positions of each row in the file,
        #    __getitem__ uses file.seek(offset) to jump to and read that row
        # 2. Memory mapping (mmap): open the file with mmap, read on demand
        # 3. Chunked reading: load in chunks, combined with LRU cache
        # These methods sacrifice code simplicity for lower memory usage
        """
        super().__init__()

        # Save configuration parameters
        self.csv_path = csv_path
        self.alphabet = alphabet
        self.max_mirna_len = max_mirna_len
        self.max_target_len = max_target_len

        # Get batch_converter for tokenization
        # batch_converter is the tokenization tool provided by RNA-FM, converting RNA strings to token IDs
        self.batch_converter = alphabet.get_batch_converter()

        # Design decision: load entire CSV into memory (see docstring above for details)
        # On a 1TB RAM system, 5.4 million rows ≈ 2-3 GB, easily affordable
        self.df = pd.read_csv(
            csv_path,
            dtype={"target_gene_name": str, "target_gene_id": str},
        )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        DataLoader calls this method to determine how many steps per epoch.
        e.g.: len(dataset)=557521, batch_size=128 → ~4356 steps per epoch
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve the idx-th sample, returning a dict of tokenized tensors.

        [Processing Pipeline]
        1. Extract row idx from the DataFrame
        2. Get mirna_seq and target_fragment_40nt
        3. Apply dna_to_rna() for T→U conversion
        4. Tokenize with RNA-FM batch_converter
        5. Assemble and return the dict

        Args:
            idx (int): sample index, range [0, len(self)-1]

        Returns:
            dict: containing the following key-value pairs:
                - 'mirna_tokens': 1D LongTensor, miRNA token sequence
                    shape = (mirna_len+2,), including BOS and EOS
                - 'target_tokens': 1D LongTensor, target token sequence
                    shape = (42,), fixed length (BOS + 40nt + EOS)
                - 'label': float32 scalar tensor (0.0 or 1.0)
                - 'metadata': dict, containing species, mirna_name, target_gene_name
        """
        # ── Step 1: Extract one row from the DataFrame ──
        row = self.df.iloc[idx]

        # ── Step 2: Extract sequences and label ──
        mirna_seq_raw = row["mirna_seq"]
        target_seq_raw = row["target_fragment_40nt"]
        label = row["label"]

        # ── Step 3: DNA-to-RNA conversion (T → U) ──
        # Sequences in the dataset use DNA notation (T for thymine),
        # but the RNA-FM model expects RNA notation (U for uridine), so conversion is needed
        mirna_rna = dna_to_rna(mirna_seq_raw)
        target_rna = dna_to_rna(target_seq_raw)

        # ── Step 4: Tokenize using RNA-FM batch_converter ──
        # batch_converter input format: List[Tuple[label, sequence]]
        # It automatically adds BOS(<cls>=0) and EOS(<eos>=2) tokens around the sequence
        #
        # e.g.: [("mirna", "AUCG")]
        # output tokens: tensor([[0, 4, 7, 5, 6, 2]])
        #                 BOS=0  A  U  C  G  EOS=2
        #
        # Here we process only 1 sequence at a time (batch_size=1),
        # so we use tokens[0] to extract the first one, yielding a 1D tensor

        # Tokenize miRNA
        _, _, mirna_tokens = self.batch_converter([("mirna", mirna_rna)])
        mirna_tokens = mirna_tokens[0]  # (1, seq_len) → (seq_len,)

        # Tokenize target
        _, _, target_tokens = self.batch_converter([("target", target_rna)])
        target_tokens = target_tokens[0]  # (1, 42) → (42,)

        # ── Step 5: Assemble the return dict ──
        # Why use float32 for label?
        # Because training uses BCEWithLogitsLoss (binary cross-entropy),
        # which requires both target and prediction to be float type.
        # If label is int/long, PyTorch will raise a type mismatch error.
        return {
            "mirna_tokens": mirna_tokens,       # 1D LongTensor, variable (17-32)
            "target_tokens": target_tokens,     # 1D LongTensor, fixed 42
            "label": torch.tensor(label, dtype=torch.float32),  # scalar float32
            "metadata": {
                "species": row["species"],
                "mirna_name": row["mirna_name"],
                "target_gene_name": row["target_gene_name"],
                "evidence_type": row.get("evidence_type", ""),
                "source_database": row.get("source_database", ""),
            },
        }
