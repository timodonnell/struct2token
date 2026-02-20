"""Kabsch-aligned all-atom RMSD computation.

Implements optimal rigid-body alignment via SVD (Kabsch algorithm)
and RMSD calculation for structure comparison.
"""

from __future__ import annotations

import torch
from scipy.spatial.transform import Rotation


def kabsch_align(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align pred to target using Kabsch algorithm (SVD-based).

    Args:
        pred: (N, 3) or (B, N, 3) predicted coordinates
        target: (N, 3) or (B, N, 3) target coordinates
        mask: (N,) or (B, N) bool, atoms to include in alignment

    Returns:
        aligned_pred: same shape as pred, optimally rotated+translated
    """
    batched = pred.dim() == 3
    if not batched:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, N, _ = pred.shape
    aligned = torch.zeros_like(pred)

    for b in range(B):
        p = pred[b]    # (N, 3)
        t = target[b]  # (N, 3)

        if mask is not None:
            m = mask[b].bool()
            p_sel = p[m]
            t_sel = t[m]
        else:
            p_sel = p
            t_sel = t

        if p_sel.shape[0] < 3:
            aligned[b] = p
            continue

        # Center
        p_mean = p_sel.mean(dim=0, keepdim=True)
        t_mean = t_sel.mean(dim=0, keepdim=True)
        p_centered = p_sel - p_mean
        t_centered = t_sel - t_mean

        # SVD
        H = p_centered.T @ t_centered  # (3, 3)
        U, S, Vt = torch.linalg.svd(H)

        # Handle reflection
        d = torch.det(Vt.T @ U.T)
        sign_matrix = torch.eye(3, device=p.device)
        sign_matrix[2, 2] = d.sign()

        R = Vt.T @ sign_matrix @ U.T

        # Apply to all atoms
        aligned[b] = (p - p_mean) @ R.T + t_mean

    if not batched:
        aligned = aligned.squeeze(0)

    return aligned


def compute_rmsd(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    align: bool = True,
) -> torch.Tensor:
    """Compute RMSD between predicted and target coordinates.

    Args:
        pred: (N, 3) or (B, N, 3)
        target: (N, 3) or (B, N, 3)
        mask: (N,) or (B, N) bool, atoms to include
        align: if True, Kabsch-align first

    Returns:
        rmsd: scalar or (B,) RMSD values in same units as input
    """
    batched = pred.dim() == 3
    if not batched:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    if align:
        pred = kabsch_align(pred, target, mask)

    diff = (pred - target) ** 2  # (B, N, 3)
    sq_dist = diff.sum(dim=-1)   # (B, N)

    if mask is not None:
        mask_f = mask.float()
        msd = (sq_dist * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1)
    else:
        msd = sq_dist.mean(dim=-1)

    rmsd = msd.sqrt()

    if not batched:
        rmsd = rmsd.squeeze(0)

    return rmsd


def backbone_rmsd(
    pred: torch.Tensor,
    target: torch.Tensor,
    meta_classes: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSD computed only on backbone atoms (meta_class == 0 or 1)."""
    bb_mask = (meta_classes == 0) | (meta_classes == 1)  # backbone + cref
    if padding_mask is not None:
        bb_mask = bb_mask & padding_mask
    return compute_rmsd(pred, target, bb_mask, align=True)


def sidechain_rmsd(
    pred: torch.Tensor,
    target: torch.Tensor,
    meta_classes: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSD computed only on sidechain atoms (meta_class == 2).

    Alignment is still done on backbone atoms.
    """
    bb_mask = (meta_classes == 0) | (meta_classes == 1)
    if padding_mask is not None:
        bb_mask = bb_mask & padding_mask

    # Align on backbone
    pred_aligned = kabsch_align(pred, target, bb_mask)

    # Evaluate on sidechain
    sc_mask = meta_classes == 2
    if padding_mask is not None:
        sc_mask = sc_mask & padding_mask

    diff = (pred_aligned - target) ** 2
    sq_dist = diff.sum(dim=-1)

    mask_f = sc_mask.float()
    msd = (sq_dist * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1)
    return msd.sqrt()
