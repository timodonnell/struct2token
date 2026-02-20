"""TM-score computation for structural similarity evaluation.

Non-differentiable metric used for monitoring only. Computed on Cα atoms
(proteins) or C3' atoms (RNA).
"""

from __future__ import annotations

import torch

from .rmsd import kabsch_align


def compute_tm_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute TM-score between predicted and target structures.

    TM-score ranges from 0 to 1, where 1 indicates a perfect match.
    Scores > 0.5 generally indicate the same fold.

    Uses the standard TM-score formula:
        TM = max_rotation [ 1/L * sum_i 1/(1 + (d_i/d_0)^2) ]
    where d_0 = 1.24 * (L - 15)^(1/3) - 1.8

    Args:
        pred: (N, 3) or (B, N, 3) predicted coordinates (Cα or C3')
        target: (N, 3) or (B, N, 3) target coordinates
        mask: (N,) or (B, N) bool mask for valid positions

    Returns:
        tm: scalar or (B,) TM-scores
    """
    batched = pred.dim() == 3
    if not batched:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, N, _ = pred.shape

    # Align
    pred_aligned = kabsch_align(pred, target, mask)

    scores = []
    for b in range(B):
        if mask is not None:
            m = mask[b].bool()
            p = pred_aligned[b][m]
            t = target[b][m]
        else:
            p = pred_aligned[b]
            t = target[b]

        L = p.shape[0]
        if L < 16:
            scores.append(torch.tensor(0.0, device=pred.device))
            continue

        # d_0 normalization factor
        d_0 = 1.24 * ((L - 15) ** (1.0 / 3.0)) - 1.8
        d_0 = max(d_0, 0.5)  # Avoid division issues

        # Per-atom distances
        d_i = (p - t).norm(dim=-1)  # (L,)

        # TM-score
        tm = (1.0 / (1.0 + (d_i / d_0) ** 2)).sum() / L
        scores.append(tm)

    result = torch.stack(scores)
    if not batched:
        result = result.squeeze(0)
    return result
