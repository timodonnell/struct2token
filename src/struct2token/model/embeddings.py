"""Atom embedding layer: sum of coordinate, atom-type, residue-type, and metastructure embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.tokens import NUM_ATOM_TYPES, NUM_META_CLASSES, NUM_RESIDUE_TYPES


class AtomEmbedding(nn.Module):
    """Produce per-atom embeddings by summing four learned components.

    Components:
        1. Coordinate projection: Linear(3, d_model) with SiLU
        2. Atom type embedding: nn.Embedding(NUM_ATOM_TYPES, d_model)
        3. Residue type embedding: nn.Embedding(NUM_RESIDUE_TYPES, d_model)
        4. Metastructure class embedding: nn.Embedding(NUM_META_CLASSES, d_model)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.coord_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.SiLU(),
        )
        self.atom_type_embed = nn.Embedding(NUM_ATOM_TYPES, d_model, padding_idx=0)
        self.residue_type_embed = nn.Embedding(NUM_RESIDUE_TYPES, d_model, padding_idx=0)
        self.meta_class_embed = nn.Embedding(NUM_META_CLASSES, d_model)

    def forward(
        self,
        coords: torch.Tensor,        # (B, L, 3)
        atom_types: torch.Tensor,     # (B, L)
        residue_types: torch.Tensor,  # (B, L)
        meta_classes: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """Returns (B, L, d_model) atom embeddings."""
        h = (
            self.coord_proj(coords)
            + self.atom_type_embed(atom_types)
            + self.residue_type_embed(residue_types)
            + self.meta_class_embed(meta_classes)
        )
        return h
