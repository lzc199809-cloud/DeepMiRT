#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false
"""
PyTorch Lightning training module for miRNA-target prediction.

[Lightning Training Loop Overview -- Full Lifecycle of One Epoch]

    ┌─────────────────────────────────────────────────────────────────────┐
    │                     Lifecycle of One Epoch                          │
    │                                                                     │
    │  on_train_epoch_start()                                             │
    │       │                                                             │
    │       v                                                             │
    │  ┌──────────────────────────────────────────┐                       │
    │  │  for batch in train_dataloader:          │                       │
    │  │      training_step(batch)                │  ← forward + loss    │
    │  │      backward()          [automatic]     │  ← backpropagation   │
    │  │      optimizer.step()    [automatic]     │  ← update params     │
    │  └──────────────────────────────────────────┘                       │
    │       │                                                             │
    │       v                                                             │
    │  on_train_epoch_end()                                               │
    │       │                                                             │
    │       v                                                             │
    │  ┌──────────────────────────────────────────┐                       │
    │  │  for batch in val_dataloader:            │                       │
    │  │      validation_step(batch)              │  ← forward only, no  │
    │  │                                          │     param updates     │
    │  └──────────────────────────────────────────┘                       │
    │       │                                                             │
    │       v                                                             │
    │  on_validation_epoch_end()                                          │
    │       │                                                             │
    │       v                                                             │
    │  lr_scheduler.step()                [automatic]                     │
    └─────────────────────────────────────────────────────────────────────┘

    Things Lightning handles automatically (no manual code needed):
    - loss.backward()
    - optimizer.zero_grad()
    - optimizer.step()
    - Switching to model.eval() and torch.no_grad() during validation
    - Gradient accumulation (if accumulate_grad_batches is configured)
    - Multi-GPU distributed synchronization (if using DDP)

    You only need to focus on:
    - training_step(): return the loss
    - validation_step(): compute validation metrics
    - configure_optimizers(): define the optimizer and learning rate scheduler

[Key Design Decisions]

    1. BCEWithLogitsLoss vs BCELoss:
       - BCEWithLogitsLoss = Sigmoid + BCELoss, using the log-sum-exp trick internally
       - Numerical stability: directly computing log(sigmoid(x)) can produce log(0) at
         extreme values. BCEWithLogitsLoss uses the equivalent formula
         max(x,0) - x*y + log(1+exp(-|x|)) to avoid overflow
       - Therefore the model outputs raw logits (no sigmoid); the loss function handles it

    2. Differential Learning Rate:
       - Backbone (RNA-FM): base_lr x 0.01 -- pretrained weights encode rich RNA knowledge;
         a large learning rate would cause catastrophic forgetting of this knowledge
       - Cross-attention layers: base_lr x 0.1 -- new module but needs stable attention
         pattern learning
       - Classifier head: base_lr x 1.0 -- learning from scratch, needs the highest
         learning rate for fast convergence

    3. Evaluation Metric Selection:
       - AUROC (Area Under ROC Curve): measures the model's ranking ability, i.e., the
         probability of ranking a positive sample above a negative one. Threshold-independent.
       - AUPRC (Average Precision / PR-AUC): measures the precision-recall tradeoff;
         more sensitive than AUROC on class-imbalanced data (biological data often has
         positive:negative ratios of 1:10+)
       - Accuracy: intuitive but can be misleading on imbalanced data (predicting all
         negatives still yields 90% accuracy)
       - F1: harmonic mean of precision and recall, balancing both

    4. Logging Strategy -- on_step=False, on_epoch=True:
       - Training loss: fluctuates heavily per step; step-level logging aids debugging
       - Evaluation metrics: require full epoch data to be statistically meaningful,
         hence on_epoch=True
       - prog_bar=True: displays key metrics in the training progress bar for real-time
         monitoring
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from insect_mirna_target.model.mirna_target_model import MiRNATargetModel


class MiRNATargetLitModule(pl.LightningModule):
    """
    Lightning training module for miRNA-target binary classification prediction.

    Responsibilities:
    - Wraps MiRNATargetModel, managing forward pass / loss / metric computation
    - Configures optimizer with differential learning rates and LR scheduler
    - Provides training_step / validation_step / test_step

    Args:
        config: Nested dictionary with the following structure:
            {
                'model': {
                    'freeze_backbone': bool,
                    'cross_attn_heads': int,
                    'cross_attn_layers': int,
                    'classifier_hidden': list[int],
                    'dropout': float,
                },
                'training': {
                    'lr': float,             # base learning rate (used by classifier head)
                    'weight_decay': float,   # L2 regularization coefficient
                    'scheduler': str,        # 'cosine' or 'onecycle'
                    'max_epochs': int,       # total training epochs (needed by scheduler)
                }
            }
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        # Save hyperparameters to the checkpoint for restoring the full config on reload
        # Design decision: save_hyperparameters ensures reproducibility -- checkpoint carries the full config
        self.save_hyperparameters(config)
        self.config = config

        # ── Extract model parameters from config and instantiate ──
        model_cfg = config["model"]
        self.model = MiRNATargetModel(
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            cross_attn_heads=model_cfg.get("cross_attn_heads", 8),
            cross_attn_layers=model_cfg.get("cross_attn_layers", 2),
            classifier_hidden=model_cfg.get("classifier_hidden", [256, 64]),
            dropout=model_cfg.get("dropout", 0.3),
        )

        # ── Loss function ──
        # Design decision: BCEWithLogitsLoss is more numerically stable than sigmoid + BCELoss.
        # Internal formula: loss = max(logit, 0) - logit * label + log(1 + exp(-|logit|))
        # This formula avoids numerical overflow from log(sigmoid(x)) at extreme values of x.
        self.loss_fn = nn.BCEWithLogitsLoss()

        # ── Training metrics ──
        # torchmetrics automatically handles metric aggregation in distributed settings (DDP sync)
        self.train_auroc = torchmetrics.AUROC(task="binary")

        # ── Validation metrics ──
        self.val_auroc = torchmetrics.AUROC(task="binary")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

        # ── Test metrics (same as validation, but separate instances to avoid state contamination) ──
        self.test_auroc = torchmetrics.AUROC(task="binary")
        self.test_auprc = torchmetrics.AveragePrecision(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")

    # ─────────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Single training step: forward pass -> compute loss -> update metrics.

        Lightning automatically calls backward() and optimizer.step() on the returned loss.
        There is no need to manually call loss.backward() or optimizer.zero_grad().

        Args:
            batch: Dictionary output from the DataModule collate_fn, containing:
                - mirna_tokens:          (B, max_mirna_len)
                - target_tokens:         (B, 42)
                - labels:                (B,)  float32
                - attention_mask_mirna:  (B, max_mirna_len)
                - attention_mask_target: (B, 42)
            batch_idx: Index of the current batch (automatically passed by Lightning)

        Returns:
            loss: Scalar tensor; Lightning automatically backpropagates through it
        """
        # Step 1: Extract inputs from the batch dictionary
        mirna_tokens = batch["mirna_tokens"]
        target_tokens = batch["target_tokens"]
        labels = batch["labels"]
        attention_mask_mirna = batch["attention_mask_mirna"]
        attention_mask_target = batch["attention_mask_target"]

        # Step 2: Forward pass -> logits shape (B, 1)
        logits = self.model(
            mirna_tokens, target_tokens, attention_mask_mirna, attention_mask_target
        )

        # Step 3: Compute loss
        # squeeze(-1) reduces logits from (B, 1) to (B,), aligning with labels (B,)
        loss = self.loss_fn(logits.squeeze(-1), labels)

        # Step 4: Compute prediction probabilities and update metrics
        # Note: sigmoid is only used for metric computation, not for the loss (BCEWithLogitsLoss includes sigmoid internally)
        probs = torch.sigmoid(logits.squeeze(-1))
        self.train_auroc(probs, labels.long())

        # Step 5: Logging
        # Design decision: train_loss uses on_step=True to monitor convergence trends,
        # train_auroc uses on_epoch=True because per-step AUROC has little statistical significance.
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_auroc",
            self.train_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    # ─────────────────────────────────────────────────────────────
    # Validation step
    # ─────────────────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """
        Single validation step: forward pass -> compute loss and full metric suite.

        Lightning automatically handles the following during validation:
        - Switches to model.eval() mode (disables Dropout, uses running mean for BatchNorm)
        - Wraps in torch.no_grad(), skipping gradient computation to save memory

        Args:
            batch: Same as training_step
            batch_idx: Index of the current batch
        """
        mirna_tokens = batch["mirna_tokens"]
        target_tokens = batch["target_tokens"]
        labels = batch["labels"]
        attention_mask_mirna = batch["attention_mask_mirna"]
        attention_mask_target = batch["attention_mask_target"]

        logits = self.model(
            mirna_tokens, target_tokens, attention_mask_mirna, attention_mask_target
        )

        loss = self.loss_fn(logits.squeeze(-1), labels)
        probs = torch.sigmoid(logits.squeeze(-1))

        # Update all validation metrics
        self.val_auroc(probs, labels.long())
        self.val_auprc(probs, labels.long())
        self.val_acc(probs, labels.long())
        self.val_f1(probs, labels.long())

        # Design decision: all validation metrics use on_epoch=True, as they need full data to be statistically meaningful
        # sync_dist=True automatically aggregates metrics across GPUs in multi-GPU settings
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_epoch=True, prog_bar=True)
        self.log("val_auprc", self.val_auprc, on_epoch=True)
        self.log("val_acc", self.val_acc, on_epoch=True)
        self.log("val_f1", self.val_f1, on_epoch=True)

    # ─────────────────────────────────────────────────────────────
    # Test step
    # ─────────────────────────────────────────────────────────────

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """
        Single test step: same logic as validation_step, using separate test metric instances.

        Test metrics are instantiated separately from validation metrics to avoid state
        contamination. For example, val_auroc resets at the end of each validation epoch,
        while test_auroc is only used when trainer.test() is called.
        """
        mirna_tokens = batch["mirna_tokens"]
        target_tokens = batch["target_tokens"]
        labels = batch["labels"]
        attention_mask_mirna = batch["attention_mask_mirna"]
        attention_mask_target = batch["attention_mask_target"]

        logits = self.model(
            mirna_tokens, target_tokens, attention_mask_mirna, attention_mask_target
        )

        loss = self.loss_fn(logits.squeeze(-1), labels)
        probs = torch.sigmoid(logits.squeeze(-1))

        # Update test metrics
        self.test_auroc(probs, labels.long())
        self.test_auprc(probs, labels.long())
        self.test_acc(probs, labels.long())
        self.test_f1(probs, labels.long())

        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        self.log("test_auroc", self.test_auroc, on_epoch=True)
        self.log("test_auprc", self.test_auprc, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)

    # ─────────────────────────────────────────────────────────────
    # Optimizer and learning rate scheduling
    # ─────────────────────────────────────────────────────────────

    def configure_optimizers(self) -> dict:
        """
        Configure AdamW optimizer with differential learning rates and cosine annealing scheduler.

        [Differential Learning Rates -- Why use different learning rates for different modules?]

            Module          Learning Rate   Reason
            ─────────────  ─────────────  ──────────────────────────────────
            RNA-FM backbone base_lr×0.01   Pretrained weights contain rich RNA structure/sequence
                                           knowledge; a large LR would destroy this knowledge
                                           (catastrophic forgetting)
            Cross-attention base_lr×0.1    Newly initialized module, but needs to stably learn
                                           miRNA-target attention patterns
            Classifier head base_lr×1.0    Learns the binary classification decision boundary
                                           from scratch; needs the highest LR for fast convergence

        Design decision: The LR ratios [0.01, 0.1, 1.0] follow common transfer learning practice;
        the paper "Universal Language Model Fine-tuning" (Howard & Ruder, 2018)
        calls this "discriminative fine-tuning".

        [CosineAnnealingLR Scheduler]
        The learning rate decays from its initial value toward 0 following a cosine curve:
            lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))
        Advantage: fast learning early on, fine-grained adjustment later, avoiding instability
        from sudden LR drops.

        Returns:
            Dictionary containing the optimizer and lr_scheduler
        """
        training_cfg = self.config["training"]
        base_lr = training_cfg["lr"]
        weight_decay = training_cfg.get("weight_decay", 1e-5)
        scheduler_type = training_cfg.get("scheduler", "cosine")
        max_epochs = training_cfg.get("max_epochs", 30)

        # Design decision: 3 parameter groups correspond to the model's 3 semantic modules;
        # learning rates decrease from downstream to upstream (farther from the task = smaller LR).
        param_groups = [
            {
                "params": list(self.model.encoder.parameters()),
                "lr": base_lr * 0.01,
                "name": "backbone",
            },
            {
                "params": list(self.model.cross_attention.parameters()),
                "lr": base_lr * 0.1,
                "name": "cross_attention",
            },
            {
                "params": list(self.model.classifier.parameters()),
                "lr": base_lr,
                "name": "classifier",
            },
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        # Design decision: CosineAnnealingLR is a safe default choice --
        # it does not require knowing total steps (unlike OneCycleLR), and provides smooth decay.
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        elif scheduler_type == "onecycle":
            # OneCycleLR requires total_steps = steps_per_epoch * max_epochs,
            # but at the configure_optimizers stage the DataLoader has not been created yet,
            # so steps_per_epoch is unavailable. Therefore, fall back to CosineAnnealingLR.
            # If OneCycleLR is needed, it should be configured in train.py via the Trainer's
            # estimated_stepping_batches.
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # Design decision: interval='epoch' adjusts the learning rate once per epoch,
                # which is more stable than 'step' (adjusting after every batch), suitable for small to medium datasets.
                "interval": "epoch",
            },
        }
