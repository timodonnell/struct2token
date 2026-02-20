"""Tests for loss functions: RMSD, Kabsch alignment, TM-score."""

import torch
import pytest

from struct2token.losses.rmsd import compute_rmsd, kabsch_align, backbone_rmsd
from struct2token.losses.tm import compute_tm_score
from struct2token.losses.inter_atom_distance import intra_residue_distance_rmse


class TestKabschAlign:
    def test_identity(self):
        """Identical structures should have zero RMSD."""
        coords = torch.randn(50, 3)
        rmsd = compute_rmsd(coords, coords, align=True)
        assert rmsd.item() < 1e-5

    def test_pure_rotation(self):
        """After alignment, a rotated structure should have zero RMSD."""
        coords = torch.randn(50, 3)
        # Apply a known rotation (90 degrees around z-axis)
        R = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        rotated = coords @ R.T
        rmsd = compute_rmsd(rotated, coords, align=True)
        assert rmsd.item() < 1e-4

    def test_pure_translation(self):
        """After alignment, a translated structure should have zero RMSD."""
        coords = torch.randn(50, 3)
        translated = coords + torch.tensor([10.0, -5.0, 3.0])
        rmsd = compute_rmsd(translated, coords, align=True)
        assert rmsd.item() < 1e-4

    def test_with_mask(self):
        """RMSD with mask should only consider selected atoms."""
        coords = torch.randn(50, 3)
        perturbed = coords.clone()
        perturbed[40:] += 100.0  # large perturbation to last 10

        mask = torch.ones(50, dtype=torch.bool)
        mask[40:] = False

        rmsd_masked = compute_rmsd(perturbed, coords, mask=mask, align=True)
        rmsd_full = compute_rmsd(perturbed, coords, align=True)
        assert rmsd_masked < rmsd_full

    def test_batched(self):
        pred = torch.randn(3, 30, 3)
        target = torch.randn(3, 30, 3)
        rmsd = compute_rmsd(pred, target, align=True)
        assert rmsd.shape == (3,)


class TestTMScore:
    def test_identical(self):
        """Identical structures should have TM-score ≈ 1."""
        coords = torch.randn(100, 3) * 10  # spread out
        tm = compute_tm_score(coords, coords)
        assert tm.item() > 0.99

    def test_range(self):
        """TM-score should be in [0, 1]."""
        pred = torch.randn(100, 3) * 10
        target = torch.randn(100, 3) * 10
        tm = compute_tm_score(pred, target)
        assert 0.0 <= tm.item() <= 1.0

    def test_short_sequence(self):
        """Short sequences (< 16 residues) should return 0."""
        coords = torch.randn(10, 3)
        tm = compute_tm_score(coords, coords)
        assert tm.item() == 0.0


class TestInterAtomDistance:
    def test_identical(self):
        """Identical coords should have zero distance RMSE."""
        coords = torch.randn(1, 20, 3)
        rids = torch.tensor([[0]*5 + [1]*5 + [2]*5 + [3]*5])
        mask = torch.ones(1, 20, dtype=torch.bool)
        rmse = intra_residue_distance_rmse(coords, coords, rids, mask)
        assert rmse[0].item() < 1e-5

    def test_nonzero(self):
        """Different coords should have non-zero distance RMSE."""
        pred = torch.randn(1, 20, 3)
        target = torch.randn(1, 20, 3)
        rids = torch.tensor([[0]*5 + [1]*5 + [2]*5 + [3]*5])
        mask = torch.ones(1, 20, dtype=torch.bool)
        rmse = intra_residue_distance_rmse(pred, target, rids, mask)
        assert rmse[0].item() > 0
