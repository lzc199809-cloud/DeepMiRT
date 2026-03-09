#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false
"""
Full miRNA-target model: shared RNA-FM encoder + Cross-Attention + MLP classifier head.

Complete data flow (with tensor shapes):

    miRNA tokens (B, M_tok)  ---> [RNA-FM Encoder] ---> miRNA_emb  (B, M, D) ---┐
                                                                                  |
                                                                                  v
    target tokens (B, T_tok) ---> [RNA-FM Encoder] ---> target_emb (B, T, D) --> [Cross-Attention]
                                                                                  |
                                                                                  v
                                                                        cross_out (B, T, D)
                                                                                  |
                                                                                  v
                                                                      masked mean pool
                                                                                  |
                                                                                  v
                                                                              (B, D)
                                                                                  |
                                                                                  v
                                                                            [MLP Head]
                                                                                  |
                                                                                  v
                                                                              logits
                                                                              (B, 1)

Where D is automatically inferred from RNA-FM (typically 640) to avoid hard-coding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .classifier import MLPClassifier
from .cross_attention import CrossAttentionBlock
from .rnafm_encoder import RNAFMEncoder


class MiRNATargetModel(nn.Module):
    """End-to-end model for miRNA-target binary classification."""

    def __init__(
        self,
        freeze_backbone: bool = True,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 2,
        classifier_hidden: Sequence[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = list(classifier_hidden) if classifier_hidden is not None else [256, 64]

        self.encoder = RNAFMEncoder(freeze_backbone=freeze_backbone)
        embed_dim = self.encoder.embed_dim

        # Design decision: the interaction layer uses a smaller dropout (~1/3 of main dropout)
        # to preserve attention signals while still providing basic regularization.
        self.cross_attention = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=cross_attn_heads,
            dropout=dropout * 0.33,
            num_layers=cross_attn_layers,
        )
        self.classifier = MLPClassifier(
            input_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(
        self,
        mirna_tokens: Tensor,
        target_tokens: Tensor,
        attention_mask_mirna: Tensor | None = None,
        attention_mask_target: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass (step by step):
        1) miRNA encoding: `(B, M_tok)` -> `(B, M, D)`
        2) target encoding: `(B, T_tok)` -> `(B, T, D)`
        3) Build key_padding_mask: attention_mask(1=real, 0=padding) -> (==0)
        4) Cross-Attention: target(Q) queries miRNA(K/V) -> `(B, T, D)`
        5) Masked mean pooling over target sequence -> `(B, D)`
        6) Classifier head outputs logits -> `(B, 1)`
        """
        # Step 1: Shared encoder processes miRNA (shared weights)
        mirna_emb = self.encoder(mirna_tokens)

        # Step 2: Same encoder processes target to ensure consistent representation space
        target_emb = self.encoder(target_tokens)

        # Step 3: PyTorch MHA key_padding_mask convention: True=ignore.
        key_padding_mask = None
        if attention_mask_mirna is not None:
            key_padding_mask = attention_mask_mirna == 0

        # Step 4: target as Query, miRNA as Key/Value.
        cross_out = self.cross_attention(
            query=target_emb,
            key_value=mirna_emb,
            key_padding_mask=key_padding_mask,
        )

        # Step 5: Masked mean pooling over target sequence to obtain a fixed-length representation.
        if attention_mask_target is None:
            pooling_mask = torch.ones(
                cross_out.size(0),
                cross_out.size(1),
                1,
                device=cross_out.device,
                dtype=cross_out.dtype,
            )
        else:
            pooling_mask = attention_mask_target.to(dtype=cross_out.dtype).unsqueeze(-1)

        summed = (cross_out * pooling_mask).sum(dim=1)
        denom = pooling_mask.sum(dim=1).clamp_min(1e-6)
        pooled = summed / denom

        # Step 6: Output raw logits without applying sigmoid.
        logits = self.classifier(pooled)
        return logits
