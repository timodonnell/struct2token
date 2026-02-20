"""Data augmentation for structure training: random SO(3) rotations and centering."""

from __future__ import annotations

import torch


def random_rotation_matrix(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample uniform random SO(3) rotation matrices.

    Uses the QR decomposition method for uniform sampling.

    Args:
        batch_size: number of rotations to sample
        device: torch device

    Returns:
        R: (B, 3, 3) rotation matrices
    """
    # Random matrix from standard normal
    M = torch.randn(batch_size, 3, 3, device=device)
    # QR decomposition gives orthogonal Q
    Q, R_tri = torch.linalg.qr(M)
    # Ensure proper rotation (det = +1)
    signs = torch.diagonal(R_tri, dim1=-2, dim2=-1).sign()
    Q = Q * signs.unsqueeze(-2)
    # Fix determinant
    dets = torch.det(Q)
    Q[:, :, -1] *= dets.sign().unsqueeze(-1)
    return Q


def apply_random_rotation(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply independent random SO(3) rotation to each sample's coordinates.

    Modifies batch["coords"] in-place and returns the batch.
    """
    coords = batch["coords"]  # (B, L, 3)
    B = coords.shape[0]
    R = random_rotation_matrix(B, coords.device)  # (B, 3, 3)
    # (B, L, 3) @ (B, 3, 3)^T = (B, L, 3)
    batch["coords"] = torch.bmm(coords, R.transpose(1, 2))
    return batch


def center_coords(
    coords: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Center coordinates by subtracting the centroid of valid atoms.

    Args:
        coords: (B, L, 3) or (L, 3)
        mask: (B, L) or (L,) bool

    Returns:
        centered: same shape as coords
    """
    if mask is not None:
        mask_f = mask.float().unsqueeze(-1)  # (..., L, 1)
        centroid = (coords * mask_f).sum(dim=-2, keepdim=True) / mask_f.sum(dim=-2, keepdim=True).clamp(min=1)
    else:
        centroid = coords.mean(dim=-2, keepdim=True)
    return coords - centroid
