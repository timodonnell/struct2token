"""Finite Scalar Quantization (FSQ).

Quantizes continuous vectors to a finite set of codes using a straight-through
estimator. Each dimension is independently quantized to one of `levels[d]` values.

With levels=(8,5,5,5), the codebook size is 8*5*5*5 = 1000 tokens.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class FSQ(nn.Module):
    """Finite Scalar Quantization module.

    Maps continuous vectors to discrete codes by rounding each dimension
    to one of a fixed number of levels. Uses straight-through estimator
    for gradient computation.
    """

    def __init__(self, levels: Tuple[int, ...] = (8, 5, 5, 5)):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.codebook_size = math.prod(levels)

        # Register levels as buffer for device placement
        _levels = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("_levels", _levels, persistent=False)

        # Basis for converting multi-dim indices to flat index
        basis = torch.ones(self.dim, dtype=torch.long)
        for i in range(self.dim - 2, -1, -1):
            basis[i] = basis[i + 1] * levels[i + 1]
        self.register_buffer("_basis", basis, persistent=False)

    @property
    def num_codes(self) -> int:
        return self.codebook_size

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Map from (-inf, inf) to [0, level-1] range per dimension using tanh."""
        # tanh maps to (-1, 1), then scale to (0, level-1)
        half_levels = (self._levels - 1) / 2
        return torch.tanh(x) * half_levels + half_levels

    def _quantize(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """Round to nearest integer with straight-through estimator."""
        x_rounded = x_scaled.round()
        # Clamp to valid range
        for d in range(self.dim):
            x_rounded[..., d] = x_rounded[..., d].clamp(0, self.levels[d] - 1)
        # Straight-through: gradient flows through as if no rounding
        return x_scaled + (x_rounded - x_scaled).detach()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize input vectors.

        Args:
            x: (..., dim) continuous vectors.

        Returns:
            quantized: (..., dim) quantized vectors (with straight-through gradient).
            indices: (...,) flat codebook indices.
        """
        assert x.shape[-1] == self.dim, f"Expected last dim {self.dim}, got {x.shape[-1]}"
        x_scaled = self._scale(x)
        quantized = self._quantize(x_scaled)
        indices = self._codes_to_indices(quantized.detach().round().long())
        return quantized, indices

    def _codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert multi-dimensional codes to flat indices."""
        return (codes * self._basis).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert flat indices back to multi-dimensional codes.

        Args:
            indices: (...,) flat codebook indices.

        Returns:
            codes: (..., dim) quantized code vectors.
        """
        codes = torch.zeros(*indices.shape, self.dim, device=indices.device, dtype=torch.float32)
        remainder = indices.clone()
        for d in range(self.dim):
            codes[..., d] = remainder // self._basis[d]
            remainder = remainder % self._basis[d]
        return codes
