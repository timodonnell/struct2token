"""Evaluation metrics: aggregate RMSD, TM-score, and inter-atomic distance metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from ..losses.inter_atom_distance import intra_residue_distance_rmse
from ..losses.rmsd import backbone_rmsd, compute_rmsd, sidechain_rmsd
from ..losses.tm import compute_tm_score


@dataclass
class MetricsAccumulator:
    """Accumulate per-sample metrics and compute aggregate statistics."""

    all_atom_rmsd: List[float] = field(default_factory=list)
    bb_rmsd: List[float] = field(default_factory=list)
    sc_rmsd: List[float] = field(default_factory=list)
    tm_scores: List[float] = field(default_factory=list)
    dist_rmse: List[float] = field(default_factory=list)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        meta_classes: torch.Tensor,
        residue_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        cref_mask: torch.Tensor | None = None,
    ):
        """Compute and accumulate metrics for a batch.

        Args:
            pred: (B, N, 3) predicted coordinates
            target: (B, N, 3) ground truth coordinates
            meta_classes: (B, N) metastructure classes
            residue_ids: (B, N) residue indices
            padding_mask: (B, N) bool
            cref_mask: (B, N) bool — Cα/C3' positions for TM-score
        """
        B = pred.shape[0]

        # All-atom RMSD
        aa_rmsd = compute_rmsd(pred, target, padding_mask, align=True)
        for b in range(B):
            self.all_atom_rmsd.append(aa_rmsd[b].item())

        # Backbone RMSD
        bb = backbone_rmsd(pred, target, meta_classes, padding_mask)
        for b in range(B):
            self.bb_rmsd.append(bb[b].item())

        # Sidechain RMSD
        sc = sidechain_rmsd(pred, target, meta_classes, padding_mask)
        for b in range(B):
            self.sc_rmsd.append(sc[b].item())

        # TM-score (on Cα / C3' atoms)
        if cref_mask is None:
            cref_mask = meta_classes == 1  # META_CREF
        for b in range(B):
            m = cref_mask[b] & padding_mask[b]
            if m.sum() >= 16:
                tm = compute_tm_score(pred[b], target[b], m)
                self.tm_scores.append(tm.item())

        # Inter-atomic distance RMSE
        dist = intra_residue_distance_rmse(pred, target, residue_ids, padding_mask)
        for b in range(B):
            self.dist_rmse.append(dist[b].item())

    def summary(self) -> Dict[str, float]:
        """Compute aggregate statistics."""
        import numpy as np

        def _stats(values: List[float], name: str) -> Dict[str, float]:
            if not values:
                return {}
            arr = np.array(values)
            return {
                f"{name}/mean": float(arr.mean()),
                f"{name}/median": float(np.median(arr)),
                f"{name}/std": float(arr.std()),
            }

        result = {}
        result.update(_stats(self.all_atom_rmsd, "all_atom_rmsd"))
        result.update(_stats(self.bb_rmsd, "backbone_rmsd"))
        result.update(_stats(self.sc_rmsd, "sidechain_rmsd"))
        result.update(_stats(self.tm_scores, "tm_score"))
        result.update(_stats(self.dist_rmse, "dist_rmse"))
        result["n_samples"] = len(self.all_atom_rmsd)
        return result
