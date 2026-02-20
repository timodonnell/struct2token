"""Inter-atomic distance loss from Bio2Token.

Computes RMSE of intra-residue pairwise distances between prediction and
ground truth. This is rotation-invariant and does not require alignment.
"""

from __future__ import annotations

import torch


def intra_residue_distance_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    residue_ids: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute RMSE of intra-residue pairwise distances.

    For each residue, compute all pairwise distances between atoms in that
    residue, then compute the RMSE between predicted and target distances.

    Args:
        pred: (B, N, 3) predicted coordinates
        target: (B, N, 3) target coordinates
        residue_ids: (B, N) residue index per atom
        padding_mask: (B, N) bool

    Returns:
        rmse: (B,) per-sample RMSE of pairwise distances
    """
    B, N, _ = pred.shape
    results = []

    for b in range(B):
        mask = padding_mask[b] if padding_mask is not None else torch.ones(N, dtype=torch.bool, device=pred.device)
        rids = residue_ids[b][mask]
        p = pred[b][mask]
        t = target[b][mask]

        unique_rids = rids.unique()
        all_diffs = []

        for rid in unique_rids:
            sel = rids == rid
            n_atoms = sel.sum().item()
            if n_atoms < 2:
                continue

            p_res = p[sel]  # (n, 3)
            t_res = t[sel]  # (n, 3)

            # Pairwise distances
            p_dist = torch.cdist(p_res, p_res)  # (n, n)
            t_dist = torch.cdist(t_res, t_res)  # (n, n)

            # Upper triangle (avoid double-counting and self-distances)
            idx = torch.triu_indices(n_atoms, n_atoms, offset=1, device=pred.device)
            diff = (p_dist[idx[0], idx[1]] - t_dist[idx[0], idx[1]]) ** 2
            all_diffs.append(diff)

        if all_diffs:
            all_diffs = torch.cat(all_diffs)
            rmse = all_diffs.mean().sqrt()
        else:
            rmse = torch.tensor(0.0, device=pred.device)

        results.append(rmse)

    return torch.stack(results)
