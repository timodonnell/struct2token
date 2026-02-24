"""Variable-length collation for batching structures with different atom counts."""

from __future__ import annotations

from typing import Dict, List

import torch


def _pad_to(length: int, multiple: int = 8) -> int:
    """Round up to the next multiple (for Flash Attention efficiency)."""
    return ((length + multiple - 1) // multiple) * multiple


def collate_structures(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length atom sequences and stack into a batch.

    Returns a dict with:
        coords: (B, L, 3) float32
        atom_types: (B, L) int64
        residue_types: (B, L) int64
        residue_ids: (B, L) int64
        meta_classes: (B, L) int64
        known_mask: (B, L) bool
        padding_mask: (B, L) bool — True for real atoms, False for padding
        lengths: (B,) int64 — original sequence lengths
        entity_types: (B,) int64 — entity type per sample (0=protein, 1=rna, 2=small_molecule)
    """
    lengths = [s["coords"].shape[0] for s in batch]
    max_len = max(lengths)
    padded_len = _pad_to(max_len)

    B = len(batch)
    out: Dict[str, torch.Tensor] = {}

    out["coords"] = torch.zeros(B, padded_len, 3, dtype=torch.float32)
    out["atom_types"] = torch.zeros(B, padded_len, dtype=torch.long)
    out["residue_types"] = torch.zeros(B, padded_len, dtype=torch.long)
    out["residue_ids"] = torch.zeros(B, padded_len, dtype=torch.long)
    out["meta_classes"] = torch.full((B, padded_len), 3, dtype=torch.long)  # 3 = pad
    out["known_mask"] = torch.zeros(B, padded_len, dtype=torch.bool)
    out["padding_mask"] = torch.zeros(B, padded_len, dtype=torch.bool)
    out["lengths"] = torch.tensor(lengths, dtype=torch.long)
    out["entity_types"] = torch.tensor(
        [s.get("entity_type", 0) for s in batch], dtype=torch.long
    )

    for i, sample in enumerate(batch):
        n = lengths[i]
        out["coords"][i, :n] = sample["coords"]
        out["atom_types"][i, :n] = sample["atom_types"].long()
        out["residue_types"][i, :n] = sample["residue_types"].long()
        out["residue_ids"][i, :n] = sample["residue_ids"].long()
        out["meta_classes"][i, :n] = sample["meta_classes"].long()
        out["known_mask"][i, :n] = sample["known_mask"].bool()
        out["padding_mask"][i, :n] = True

    return out
