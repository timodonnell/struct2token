"""Encode macromolecular structures to discrete token indices."""

from __future__ import annotations

from pathlib import Path

import torch

from ..data.collate import collate_structures
from ..data.dataset import StructureDataset
from ..model.dae import AllAtomDAE


@torch.no_grad()
def encode_structure(
    model: AllAtomDAE,
    sample: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single structure sample to token indices.

    Args:
        model: trained AllAtomDAE (in eval mode)
        sample: dict from StructureDataset.__getitem__
        device: torch device

    Returns:
        codes: (1, K, d_dec) quantized representations
        indices: (1, K) FSQ codebook indices
    """
    model.eval()
    batch = collate_structures([sample])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    codes, indices = model.encode(
        batch["coords"],
        batch["atom_types"],
        batch["residue_types"],
        batch["meta_classes"],
        batch["padding_mask"],
    )
    return codes, indices


@torch.no_grad()
def encode_batch(
    model: AllAtomDAE,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a pre-collated batch to token indices.

    Args:
        model: trained AllAtomDAE (in eval mode)
        batch: collated batch dict (already on device)

    Returns:
        codes: (B, K, d_dec) quantized representations
        indices: (B, K) FSQ codebook indices
    """
    model.eval()
    codes, indices = model.encode(
        batch["coords"],
        batch["atom_types"],
        batch["residue_types"],
        batch["meta_classes"],
        batch["padding_mask"],
    )
    return codes, indices
