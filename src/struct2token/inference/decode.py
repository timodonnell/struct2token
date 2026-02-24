"""Decode discrete token indices to 3D coordinates via diffusion sampling."""

from __future__ import annotations

import torch

from ..config import InferenceConfig
from ..model.dae import AllAtomDAE


@torch.no_grad()
def decode_tokens(
    model: AllAtomDAE,
    indices: torch.Tensor,
    n_atoms: int | torch.Tensor,
    config: InferenceConfig | None = None,
) -> torch.Tensor:
    """Decode FSQ token indices to 3D coordinates.

    Uses Euler-Maruyama diffusion sampling with classifier-free guidance.

    Args:
        model: trained AllAtomDAE (in eval mode)
        indices: (B, K) FSQ codebook indices
        n_atoms: number of atoms to generate (int or (B,) tensor)
        config: inference configuration

    Returns:
        coords: (B, N, 3) predicted coordinates in nm
    """
    if config is None:
        config = InferenceConfig()

    model.eval()
    return model.decode(
        indices=indices,
        n_atoms=n_atoms,
        n_steps=config.n_steps,
        cfg_weight=config.cfg_weight,
        noise_weight=config.noise_weight,
    )


@torch.no_grad()
def roundtrip(
    model: AllAtomDAE,
    batch: dict[str, torch.Tensor],
    config: InferenceConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch then decode it back to coordinates.

    Useful for evaluation: encode ground truth → decode → compare.

    Args:
        model: trained AllAtomDAE
        batch: collated batch dict (on device)
        config: inference config

    Returns:
        pred_coords: (B, N, 3) reconstructed coordinates
        indices: (B, K) token indices used
    """
    if config is None:
        config = InferenceConfig()

    model.eval()

    # Encode
    codes, indices = model.encode(
        batch["coords"],
        batch["atom_types"],
        batch["residue_types"],
        batch["meta_classes"],
        batch["padding_mask"],
    )

    # Decode
    n_atoms = batch["lengths"]
    pred_coords = model.decode(
        indices=indices,
        n_atoms=n_atoms,
        n_steps=config.n_steps,
        cfg_weight=config.cfg_weight,
        noise_weight=config.noise_weight,
    )

    return pred_coords, indices


@torch.no_grad()
def roundtrip_with_n_tokens(
    model: AllAtomDAE,
    batch: dict[str, torch.Tensor],
    n_tokens: int,
    config: InferenceConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch then decode using only the first n_tokens tokens.

    Tests adaptive tokenization: since the model is trained with nested dropout,
    any prefix of the token sequence should produce a valid reconstruction.
    Fewer tokens = more compression = higher RMSD.

    Args:
        model: trained AllAtomDAE
        batch: collated batch dict (on device)
        n_tokens: number of tokens to use (prefix of full K=128 sequence)
        config: inference config

    Returns:
        pred_coords: (B, N, 3) reconstructed coordinates
        indices: (B, K) full token indices (before truncation)
    """
    if config is None:
        config = InferenceConfig()

    model.eval()

    # Encode to full token sequence
    codes, indices = model.encode(
        batch["coords"],
        batch["atom_types"],
        batch["residue_types"],
        batch["meta_classes"],
        batch["padding_mask"],
    )

    B, K = indices.shape
    device = indices.device

    # Create conditioning mask: True for first n_tokens, False for rest
    cond_mask = torch.arange(K, device=device).unsqueeze(0).expand(B, -1) < n_tokens

    # Decode with truncated conditioning
    n_atoms = batch["lengths"]
    pred_coords = model.decode(
        indices=indices,
        n_atoms=n_atoms,
        n_steps=config.n_steps,
        cfg_weight=config.cfg_weight,
        noise_weight=config.noise_weight,
        cond_mask=cond_mask,
    )

    return pred_coords, indices
