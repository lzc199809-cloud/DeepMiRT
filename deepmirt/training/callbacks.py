#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false
"""
Custom Lightning Callbacks -- staged unfreezing + species-level metrics.

[What is a Callback?]
A Callback is a "hook" function that Lightning calls at specific points in the training loop.
It lets you inject additional logic without modifying the LightningModule:
    - on_train_epoch_start() -> before each training epoch starts
    - on_validation_epoch_end() -> after each validation epoch ends
    - on_train_batch_end() -> after each training batch ends
    - ... and dozens more hooks

Advantage: Separation of Concerns --
    training logic lives in the LightningModule, auxiliary strategies live in Callbacks.

[This file contains two Callbacks]

    1. StagedUnfreezeCallback
       ─────────────────────
       Implements a staged unfreezing strategy:

       Epoch 0 ~ N-1:  [RNA-FM fully frozen] -> only train Cross-Attention + classifier head
                        Benefit: new layers converge quickly to a reasonable parameter range
       From Epoch N:    [RNA-FM top K layers unfrozen] -> start fine-tuning the backbone
                        Benefit: backbone fine-tunes under stabilized downstream gradient signals

       Why unfreeze from the top?
       - Lower Transformer layers capture general RNA sequence features (base pairing, secondary structure)
       - Upper layers are closer to task-specific semantic representations
       - Unfreezing only the top few layers: preserve general features + adapt to downstream task = optimal transfer

       Analogy: similar to fine-tuning BERT where the bottom embedding + first few Transformer layers
                are typically frozen, and only the top layers + classifier head are fine-tuned.

    2. SpeciesMetricsCallback
       ────────────────────
       (Extension goal) Computes validation metrics grouped by species.
       If the batch contains metadata (species information), computes separate
       AUROC/AUPRC per species.

       Current implementation: defensively checks if metadata exists; silently skips if not.
"""

from __future__ import annotations

import pytorch_lightning as pl


class StagedUnfreezeCallback(pl.Callback):
    """
    Callback for progressive unfreezing of the RNA-FM backbone network.

    Unlike unfreezing N layers all at once, this Callback unfreezes 1 layer every
    `unfreeze_interval` epochs, starting from the topmost layer and working downward.
    After each unfreezing event, the backbone parameter group's learning rate is reduced
    to 1/10 of the target value, then linearly warmed up back to the target LR over
    `warmup_epochs` epochs.

    Training strategy timeline (default config: start=5, interval=3, num_layers=3):

        Epoch:  0  1  2  3  4  5  6  7  8  9  10  11  12  ...
                |              |        |          |
                | Phase 1:     | +L12   | +L11     | +L10
                | backbone     | (1 lyr)| (2 lyrs) | (3 lyrs)
                | fully frozen | warmup | warmup   | warmup
                | train new    |        |          |
                | layers only  |        |          |
                +--------------+--------+----------+------

    Args:
        unfreeze_at_epoch: Epoch at which the first layer is unfrozen (default 5)
        num_layers_to_unfreeze: Total number of layers to unfreeze (default 3)
        unfreeze_interval: Number of epochs between each layer unfreezing (default 3)
        warmup_epochs: Number of epochs for LR warmup after each unfreezing event (default 1)
    """

    def __init__(
        self,
        unfreeze_at_epoch: int = 5,
        num_layers_to_unfreeze: int = 3,
        unfreeze_interval: int = 3,
        warmup_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.num_layers_to_unfreeze = num_layers_to_unfreeze
        self.unfreeze_interval = unfreeze_interval
        self.warmup_epochs = warmup_epochs

        # Runtime state: current number of unfrozen layers, warmup tracking
        self._current_unfrozen: int = 0
        self._warmup_target_lr: float | None = None
        self._warmup_start_lr: float | None = None
        self._warmup_start_epoch: int = -1

    def _get_backbone_param_group(self, pl_module: pl.LightningModule) -> dict | None:
        """Find the parameter group with name='backbone' in the optimizer."""
        optimizers = pl_module.optimizers()
        if optimizers is None:
            return None
        opt = optimizers if not isinstance(optimizers, list) else optimizers[0]
        for pg in opt.param_groups:
            if pg.get("name") == "backbone":
                return pg
        return None

    def _start_warmup(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """After unfreezing, reduce backbone LR to 1/10 and record the warmup target."""
        pg = self._get_backbone_param_group(pl_module)
        if pg is None:
            return
        self._warmup_target_lr = pg["lr"]
        self._warmup_start_lr = pg["lr"] / 10.0
        self._warmup_start_epoch = trainer.current_epoch
        pg["lr"] = self._warmup_start_lr

    def _step_warmup(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Linearly interpolate backbone LR during the warmup period."""
        if self._warmup_target_lr is None:
            return
        elapsed = trainer.current_epoch - self._warmup_start_epoch
        if elapsed >= self.warmup_epochs:
            # Warmup complete, restore target LR
            pg = self._get_backbone_param_group(pl_module)
            if pg is not None:
                pg["lr"] = self._warmup_target_lr
            self._warmup_target_lr = None
            return
        # Linear interpolation
        frac = elapsed / self.warmup_epochs
        new_lr = self._warmup_start_lr + frac * (self._warmup_target_lr - self._warmup_start_lr)
        pg = self._get_backbone_param_group(pl_module)
        if pg is not None:
            pg["lr"] = new_lr

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Check at the start of each epoch whether to unfreeze a new layer or advance warmup."""
        epoch = trainer.current_epoch

        # Check if it is time to unfreeze
        if (
            self._current_unfrozen < self.num_layers_to_unfreeze
            and epoch >= self.unfreeze_at_epoch
        ):
            trigger_epoch = self.unfreeze_at_epoch + self._current_unfrozen * self.unfreeze_interval
            if epoch == trigger_epoch:
                self._current_unfrozen += 1
                pl_module.model.encoder.unfreeze(num_layers=self._current_unfrozen)

                trainable = sum(
                    p.numel() for p in pl_module.model.parameters() if p.requires_grad
                )
                total = sum(p.numel() for p in pl_module.model.parameters())

                print(
                    f"[StagedUnfreeze] Epoch {epoch}: "
                    f"Unfreezing top {self._current_unfrozen} RNA-FM layer(s). "
                    f"Trainable params: {trainable:,} / {total:,} "
                    f"({trainable / total * 100:.1f}%)"
                )

                # Start LR warmup
                self._start_warmup(trainer, pl_module)
                return  # Warmup just started, no step needed

        # Advance ongoing warmup
        self._step_warmup(trainer, pl_module)


class SpeciesMetricsCallback(pl.Callback):
    """
    (Extension goal) Callback that computes validation metrics grouped by species.

    Design intent:
    - miRNA-target prediction performance may vary across species
    - Per-species evaluation helps identify whether the model generalizes poorly to specific species

    Current limitations:
    - The DataModule's collate_fn currently only outputs tokens + labels + masks,
      without passing metadata (species, mirna_name, etc.)
    - Therefore, the current implementation only performs a defensive check: if the batch
      contains metadata, compute metrics; otherwise silently skip
    - Once the DataModule adds metadata pass-through, this Callback will work without modification

    Future extension plan:
    1. Add a 'metadata': {'species': list[str]} field in the collate_fn
    2. This Callback automatically collects predictions and labels from each validation batch
    3. At epoch end, compute AUROC / AUPRC grouped by species
    """

    def __init__(self, target_species: str = "Homo sapiens") -> None:
        super().__init__()
        self.target_species = target_species
        # Collect prediction results from each validation step
        self._val_outputs: list[dict] = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        After each validation batch, collect predictions and metadata (if available).

        Defensive design: the batch may not contain a 'metadata' key; use .get() for safe access.
        """
        # Check if the batch contains metadata
        metadata = batch.get("metadata", None)
        if metadata is None:
            return

        # If metadata is present, recompute probs (to avoid depending on validation_step's return value)
        with __import__("torch").no_grad():
            logits = pl_module.model(
                batch["mirna_tokens"],
                batch["target_tokens"],
                batch["attention_mask_mirna"],
                batch["attention_mask_target"],
            )
            probs = __import__("torch").sigmoid(logits.squeeze(-1))

        self._val_outputs.append(
            {
                "probs": probs.detach().cpu(),
                "labels": batch["labels"].detach().cpu(),
                "species": metadata.get("species", []),
            }
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        After the validation epoch ends, compute metrics grouped by species.

        If no batches with metadata were collected, return silently.
        """
        if not self._val_outputs:
            return

        import torch
        import torchmetrics

        # Merge results from all batches
        all_probs = torch.cat([o["probs"] for o in self._val_outputs])
        all_labels = torch.cat([o["labels"] for o in self._val_outputs])
        all_species: list[str] = []
        for o in self._val_outputs:
            all_species.extend(o["species"])

        if not all_species or len(all_species) != len(all_probs):
            self._val_outputs.clear()
            return

        # Compute metrics for the configured target species
        target_species = self.target_species
        species_mask = [s == target_species for s in all_species]
        species_mask_tensor = torch.tensor(species_mask, dtype=torch.bool)

        if species_mask_tensor.any():
            sp_probs = all_probs[species_mask_tensor]
            sp_labels = all_labels[species_mask_tensor].long()

            auroc_fn = torchmetrics.AUROC(task="binary")
            auprc_fn = torchmetrics.AveragePrecision(task="binary")

            sp_auroc = auroc_fn(sp_probs, sp_labels)
            sp_auprc = auprc_fn(sp_probs, sp_labels)

            species_key = target_species.split()[-1].lower()[:4]
            pl_module.log(f"val_{species_key}_auroc", sp_auroc, on_epoch=True)
            pl_module.log(f"val_{species_key}_auprc", sp_auprc, on_epoch=True)

            print(
                f"[SpeciesMetrics] {target_species}: "
                f"AUROC={sp_auroc:.4f}, AUPRC={sp_auprc:.4f} "
                f"(n={species_mask_tensor.sum().item()})"
            )

        # Clear cache in preparation for the next epoch
        self._val_outputs.clear()
