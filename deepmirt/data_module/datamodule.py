#!/usr/bin/env python3
"""
miRNA-Target PyTorch Lightning DataModule

[Lightning DataModule Lifecycle]
Lightning DataModule encapsulates data loading logic into a reusable module.
Its lifecycle is as follows:

  1. prepare_data()    — download data (runs only on main process; not needed in this project)
  2. setup(stage)      — create Dataset instances (runs on every process)
     - stage='fit'     → create train_dataset + val_dataset
     - stage='test'    → create test_dataset
     - stage='predict' → create predict_dataset
  3. train_dataloader() — return training DataLoader
  4. val_dataloader()   — return validation DataLoader
  5. test_dataloader()  — return test DataLoader

[Why use DataModule instead of manually creating DataLoaders?]
- Centralizes all data-related logic (paths, batch size, tokenizer, data splits)
- Lightning Trainer automatically calls the correct methods, reducing boilerplate
- Makes it easy to reuse the same data configuration across different experiments

[collate_fn Explained — The Core Difficulty of This Module]
Since miRNA sequence lengths are variable (15-30nt → 17-32 tokens),
samples in the same batch may have mirna_tokens of different lengths.
PyTorch's default collate cannot stack variable-length tensors,
so we need a custom collate_fn to:
  1. Find the longest miRNA sequence in the batch
  2. Pad all miRNA sequences to the same length
  3. Generate an attention mask indicating which positions are real tokens vs. padding

Target sequences are fixed at 40nt (→ 42 tokens) and do not require additional padding.
"""

from __future__ import annotations

import os

import fm
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from deepmirt.data_module.dataset import MiRNATargetDataset


class MiRNATargetDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for miRNA-target pairs.

    [Responsibilities]
    - Manage creation and DataLoader configuration for train / val / test datasets
    - Provide a custom collate_fn to handle variable-length miRNA sequence padding
    - Encapsulate RNA-FM alphabet loading to avoid redundant initialization in multiple places
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        """
        Initialize the DataModule.

        Args:
            data_dir (str): path to the directory containing train.csv / val.csv / test.csv
            batch_size (int): number of samples per batch, default 128
            num_workers (int): number of DataLoader worker processes, default 8
                # Design decision: num_workers controls data prefetching parallelism
                # - 0 = load in main process (for debugging, slow but easy to troubleshoot)
                # - 8 = 8 subprocesses load in parallel (for training, fully utilize multi-core CPU)
                # - Rule of thumb: set to half of CPU cores or GPU count x 4
                # - Too many will cause memory overhead and process switching overhead
            pin_memory (bool): whether to pin data to page-locked memory, default True
                # Design decision: pin_memory accelerates CPU→GPU data transfer
                # - True: data is first copied to pinned memory, then transferred to GPU via DMA
                #   Eliminates one memory copy, improving throughput by ~2x
                # - False: data is in pageable memory and must be copied to pinned memory before transfer
                # - Only meaningful when using GPU; set to False for CPU training
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Dataset instances, created in setup()
        self.train_dataset: MiRNATargetDataset | None = None
        self.val_dataset: MiRNATargetDataset | None = None
        self.test_dataset: MiRNATargetDataset | None = None

        # Load RNA-FM alphabet in the main process (before DDP fork)
        # This way the alphabet is loaded only once, avoiding redundant full model loading on each DDP rank
        _model, alphabet = fm.pretrained.rna_fm_t12()
        del _model  # Free model weights, keep only the alphabet (tokenizer)
        self._alphabet = alphabet
        self._padding_idx = alphabet.padding_idx  # padding_idx = 1

    def setup(self, stage: str | None = None) -> None:
        """
        Create Dataset instances.

        Lightning automatically calls this method before training/validation/testing begins.
        Each process (including multi-GPU DDP scenarios) calls setup() independently.

        Args:
            stage: 'fit' (train+val), 'test', 'predict', or None (all)
        """
        # alphabet was already loaded in __init__() (before DDP fork, loaded only once)
        alphabet = self._alphabet

        if stage == "fit" or stage is None:
            self.train_dataset = MiRNATargetDataset(
                os.path.join(self.data_dir, "train.csv"), alphabet
            )
            self.val_dataset = MiRNATargetDataset(
                os.path.join(self.data_dir, "val.csv"), alphabet
            )

        if stage == "test" or stage is None:
            self.test_dataset = MiRNATargetDataset(
                os.path.join(self.data_dir, "test.csv"), alphabet
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader (shuffle=True to randomize data order)."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader (shuffle=False to preserve order for reproducible evaluation)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: list[dict]) -> dict:
        """
        Custom batch collation function — handles padding of variable-length miRNA sequences.

        [Why is a custom collate_fn needed?]
        PyTorch's default collate_fn attempts to stack all sample tensors.
        But miRNA sequence lengths are variable (15-30nt → 17-32 tokens), and direct stacking fails:
            RuntimeError: stack expects each tensor to be equal size

        [Why does miRNA need padding but target does not?]
        - miRNA has variable length: 15-30 nucleotides → 17-32 tokens after adding BOS+EOS
          A single batch may contain lengths of both 17 and 32, which must be aligned
        - Target has fixed length: all samples are 40 nucleotides → 42 tokens
          Naturally aligned, no padding needed

        [Role of attention_mask]
        - Tells the model which positions are real tokens (1) and which are padding (0)
        - The Transformer's self-attention uses the mask to block padding positions
        - Prevents padding tokens from participating in attention computation, avoiding noise

        # Design decision: use pad_sequence instead of manual loop padding
        # pad_sequence is a PyTorch built-in utility, optimized in C++, faster than Python loops
        # It automatically finds the maximum length and pads shorter sequences with the specified value

        Args:
            batch: list of dicts, each dict from MiRNATargetDataset.__getitem__

        Returns:
            dict: containing the following key-value pairs:
                - 'mirna_tokens':          (batch_size, max_mirna_len) LongTensor
                - 'target_tokens':         (batch_size, 42) LongTensor
                - 'labels':                (batch_size,) float32 Tensor
                - 'attention_mask_mirna':  (batch_size, max_mirna_len) LongTensor
                - 'attention_mask_target': (batch_size, 42) LongTensor
        """
        # ── 1. Collect individual fields ──
        mirna_list = [sample["mirna_tokens"] for sample in batch]
        target_list = [sample["target_tokens"] for sample in batch]
        label_list = [sample["label"] for sample in batch]

        # ── 2. Pad miRNA sequences ──
        # pad_sequence converts list of 1D tensors → 2D tensor (batch, max_len)
        # batch_first=True ensures the batch dimension comes first
        # padding_value=1 is RNA-FM's <pad> token ID
        mirna_padded = pad_sequence(
            mirna_list, batch_first=True, padding_value=self._padding_idx
        )

        # ── 3. Stack target sequences (fixed 42 tokens, no padding needed) ──
        target_stacked = torch.stack(target_list)

        # ── 4. Stack labels ──
        labels = torch.stack(label_list)

        # ── 5. Generate attention masks ──
        # miRNA mask: non-padding positions = 1, padding positions = 0
        attention_mask_mirna = (mirna_padded != self._padding_idx).long()

        # target mask: all positions are real tokens, so all 1s
        # Because target is fixed at 40nt with no padding, every position is valid
        attention_mask_target = torch.ones_like(target_stacked, dtype=torch.long)

        # ── 6. Collect metadata (for stratified analysis during evaluation) ──
        # Each metadata field is collected as list[str], kept on CPU
        metadata_keys = batch[0].get("metadata", {}).keys()
        metadata = {
            key: [sample["metadata"][key] for sample in batch]
            for key in metadata_keys
        } if metadata_keys else {}

        return {
            "mirna_tokens": mirna_padded,             # (B, max_mirna_len)
            "target_tokens": target_stacked,          # (B, 42)
            "labels": labels,                         # (B,)
            "attention_mask_mirna": attention_mask_mirna,    # (B, max_mirna_len)
            "attention_mask_target": attention_mask_target,  # (B, 42)
            "metadata": metadata,                     # dict[str, list[str]]
        }
