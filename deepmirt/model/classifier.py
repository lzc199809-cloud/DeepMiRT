#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false
"""
MLP classifier head (maps sequence representations to binary classification logits).

Architecture diagram:

    pooled_feature (B, 640)
            |
            v
    Linear(640 -> 256)
            |
            v
    BatchNorm + ReLU + Dropout(0.3)
            |
            v
    Linear(256 -> 64) + ReLU + Dropout(0.2)
            |
            v
    Linear(64 -> 1)
            |
            v
    logits (B, 1)

Note:
- The output is logits (raw scores); do not apply sigmoid inside the model.
- During training, use BCEWithLogitsLoss which applies sigmoid internally for numerical stability.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class MLPClassifier(nn.Module):
    """MLP head for binary classification, outputting a single logit."""

    def __init__(
        self,
        input_dim: int = 640,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        dims = list(hidden_dims) if hidden_dims is not None else [256, 64]
        if len(dims) != 2:
            raise ValueError("hidden_dims must contain exactly two elements, e.g. [256, 64].")

        hidden1, hidden2 = int(dims[0]), int(dims[1])
        in_dim = int(input_dim)

        # Design decision: [256, 64] balances expressiveness and overfitting risk,
        # suitable for small-to-medium scale biological data.
        # Design decision: first layer uses BatchNorm + Dropout; second layer retains
        # a smaller Dropout for lightweight regularization.
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Pooled sequence representation, shape `(batch, input_dim)`.

        Returns:
            Logits, shape `(batch, 1)`.
        """
        return self.layers(x)
