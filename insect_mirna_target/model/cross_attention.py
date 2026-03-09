#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false
“””
Cross-Attention interaction module.

Data flow diagram (target as Query, miRNA as Key/Value):

    target_emb (B, T, D) -------------------------------> Q
                                                          |
                                                          | Multi-Head Cross Attention
                                                          | (batch_first=True)
                                                          |
    miRNA_emb  (B, M, D) ---> K, V -------------------->

    Output: context_target (B, T, D)

Why target=Q and miRNA=K/V:
- Our task is to determine whether a target is regulated by a given miRNA.
- Having each target position “query” miRNA information aligns with the semantics
  of locating potential binding sites on the target.

Mask convention:
- `key_padding_mask=True` indicates a padding position that should be ignored.
“””

from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossAttentionBlock(nn.Module):
    """Interaction module composed of stacked Cross-Attention + FFN layers."""

    def __init__(
        self,
        embed_dim: int = 640,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = nn.ModuleDict(
                {
                    "cross_attn": nn.MultiheadAttention(
                        embed_dim=self.embed_dim,
                        num_heads=self.num_heads,
                        dropout=dropout,
                        batch_first=True,
                    ),
                    "dropout_attn": nn.Dropout(dropout),
                    "norm1": nn.LayerNorm(self.embed_dim),
                    "ffn": nn.Sequential(
                        nn.Linear(self.embed_dim, self.embed_dim * 4),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.embed_dim * 4, self.embed_dim),
                    ),
                    "norm2": nn.LayerNorm(self.embed_dim),
                }
            )
            self.layers.append(layer)

        # Design decision: 2 layers by default is a lightweight yet effective trade-off;
        # establish a trainable baseline first, then deepen based on data scale.
        # Design decision: 8 attention heads by default improves interaction modeling across
        # different subspaces while keeping GPU memory overhead manageable.

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            query: Target representation, shape `(batch, target_len, embed_dim)`.
            key_value: miRNA representation, shape `(batch, mirna_len, embed_dim)`.
            key_padding_mask: miRNA padding mask, shape `(batch, mirna_len)`,
                where True indicates positions to ignore.

        Returns:
            Updated target representation, shape `(batch, target_len, embed_dim)`.
        """
        hidden = query
        attn_mask = key_padding_mask
        if attn_mask is not None and attn_mask.dtype is not torch.bool:
            attn_mask = attn_mask.to(dtype=torch.bool)

        for layer in self.layers:
            # Step 1: Cross-Attention (target queries miRNA)
            attn_out, _ = layer["cross_attn"](
                query=hidden,
                key=key_value,
                value=key_value,
                key_padding_mask=attn_mask,
                need_weights=False,
            )

            # Step 2: Residual + LayerNorm to stabilize deep training and mitigate vanishing gradients
            hidden = layer["norm1"](hidden + layer["dropout_attn"](attn_out))

            # Step 3: Feed-forward network refines channel-wise features
            ffn_out = layer["ffn"](hidden)

            # Step 4: Residual + LayerNorm
            hidden = layer["norm2"](hidden + ffn_out)

        return hidden
