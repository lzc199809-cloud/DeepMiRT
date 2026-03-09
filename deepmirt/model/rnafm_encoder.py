#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false
"""
RNA-FM encoder wrapper (Shared Encoder).

Architecture diagram (single-path encoding):

    Input tokens (B, L)
          |
          v
    [RNA-FM: 12-layer Transformer]
          |
          v
    representations[12] (B, L, D)
                     D is typically 640

Training strategy diagram (freeze / staged unfreezing):

    Frozen phase:    [L1][L2][L3]...[L12]   all requires_grad=False
    Unfrozen phase:  [L1]...[L9][L10][L11][L12]
                                    ^^^^^^^^
                                    only unfreeze top N layers (e.g., N=3)

Notes:
- Both miRNA and target are RNA sequences, so sharing a single RNA-FM encoder is the most natural approach.
- `repr_layers=[12]` extracts the 12th (final) layer output as the contextualized representation.
"""

from __future__ import annotations

from collections.abc import Sequence

import fm
from torch import Tensor, nn


class RNAFMEncoder(nn.Module):
    """Lightweight wrapper around RNA-FM providing forward encoding, freezing, and staged unfreezing."""

    def __init__(self, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.num_layers = len(self.model.layers)
        self.embed_dim = self._infer_embed_dim(default=640)

        # Design decision: freeze backbone by default to first stabilize training of the
        # upper interaction module and classifier head, avoiding catastrophic forgetting
        # from full fine-tuning on small datasets.
        if freeze_backbone:
            self.freeze()

    def _infer_embed_dim(self, default: int = 640) -> int:
        """Try to infer the embedding dimension from the RNA-FM model; fall back to default on failure."""
        model_embed_dim = getattr(self.model, "embed_dim", None)
        if model_embed_dim is not None:
            return int(model_embed_dim)

        model_args = getattr(self.model, "args", None)
        if model_args is not None and hasattr(model_args, "embed_dim"):
            return int(model_args.embed_dim)

        embed_tokens = getattr(self.model, "embed_tokens", None)
        if embed_tokens is not None and hasattr(embed_tokens, "embedding_dim"):
            return int(embed_tokens.embedding_dim)

        return int(default)

    def forward(self, tokens: Tensor, repr_layers: Sequence[int] | None = None) -> Tensor:
        """
        Encode an RNA token sequence.

        Args:
            tokens: Token tensor of shape `(batch, seq_len)`.
            repr_layers: List of layer indices to extract. Defaults to `[12]` (final layer).

        Returns:
            Contextualized representations of shape `(batch, seq_len, embed_dim)`.
        """
        if repr_layers is None:
            # Design decision: use the final layer representation by default (most semantically
            # complete), consistent with common pre-trained model usage.
            repr_layers = [self.num_layers]

        layer_ids = list(repr_layers)
        if not layer_ids:
            raise ValueError("repr_layers must not be empty; provide at least one layer index.")

        outputs = self.model(tokens, repr_layers=layer_ids)
        # Note: typically repr_layers=[12] is passed, so this retrieves representations[12].
        final_layer_id = max(layer_ids)
        return outputs["representations"][final_layer_id]

    def freeze(self) -> None:
        """Freeze all RNA-FM backbone parameters (requires_grad=False)."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self, num_layers: int = 3) -> None:
        """
        Unfreeze only the per-layer parameters of the top N Transformer layers.

        Example: when `num_layers=3`, unfreezes layer[9], layer[10], layer[11].

        Note: global LayerNorm (e.g., emb_layer_norm_after) is NOT unfrozen,
        because unfreezing it would shift the output distribution of all layers at once,
        leading to training instability.
        """
        # Design decision: always freeze all first, then selectively unfreeze, ensuring the
        # set of trainable parameters is controllable and reproducible.
        self.freeze()

        n = max(0, min(int(num_layers), self.num_layers))
        if n > 0:
            start = self.num_layers - n
            for layer in self.model.layers[start:]:
                for param in layer.parameters():
                    param.requires_grad = True
