"""Symmetric sidechain permutation resolution.

Before computing RMSD, resolve symmetric sidechain atoms by trying both
possible assignments and selecting the one with lower local RMSD.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

# Symmetric atom pairs: (resname, [(atom1, atom2), ...])
# For each pair, swapping atom1 and atom2 gives an equivalent structure.
SYMMETRIC_ATOMS: Dict[str, List[Tuple[str, str]]] = {
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "ARG": [("NH1", "NH2")],
    "LEU": [("CD1", "CD2")],
    "VAL": [("CG1", "CG2")],
}

# Precompute per-residue sidechain atom orderings for lookup
from ..data.molecule_conventions import PROTEIN_BACKBONE_ATOMS, PROTEIN_SIDECHAIN_ATOMS

def _build_swap_indices() -> Dict[str, List[Tuple[int, int]]]:
    """Build swap index pairs relative to full atom ordering per residue.

    Returns dict mapping resname → list of (idx_a, idx_b) pairs for swapping.
    """
    n_bb = len(PROTEIN_BACKBONE_ATOMS)
    result = {}

    for resname, pairs in SYMMETRIC_ATOMS.items():
        sc_atoms = PROTEIN_SIDECHAIN_ATOMS.get(resname, [])
        swap_pairs = []
        for a1, a2 in pairs:
            if a1 in sc_atoms and a2 in sc_atoms:
                idx1 = n_bb + sc_atoms.index(a1)
                idx2 = n_bb + sc_atoms.index(a2)
                swap_pairs.append((idx1, idx2))
        if swap_pairs:
            result[resname] = swap_pairs

    return result


SWAP_INDICES = _build_swap_indices()


def resolve_permutation_symmetry(
    pred: torch.Tensor,
    target: torch.Tensor,
    residue_ids: torch.Tensor,
    atom_names_per_residue: dict[int, tuple[str, list[str]]],
) -> torch.Tensor:
    """Resolve symmetric sidechain atoms to minimize RMSD.

    For each residue with symmetric atoms, try both assignments and keep
    the one with lower local RMSD.

    Args:
        pred: (N, 3) predicted coordinates
        target: (N, 3) target coordinates
        residue_ids: (N,) residue index per atom
        atom_names_per_residue: dict mapping residue_id → (resname, [atom_names])

    Returns:
        pred_resolved: (N, 3) with symmetric atoms potentially swapped
    """
    pred_resolved = pred.clone()

    for rid, (resname, _atom_names) in atom_names_per_residue.items():
        if resname not in SWAP_INDICES:
            continue

        # Get atoms for this residue
        mask = residue_ids == rid
        indices = mask.nonzero(as_tuple=True)[0]
        n_atoms = indices.shape[0]

        swap_pairs = SWAP_INDICES[resname]

        # Try all combinations of swaps (2^n_pairs possibilities)
        # For small number of pairs (<= 3), enumerate all
        n_pairs = len(swap_pairs)
        best_rmsd = float("inf")
        best_pred = pred_resolved[indices].clone()

        for combo in range(1 << n_pairs):
            trial = pred_resolved[indices].clone()
            for p_idx in range(n_pairs):
                if combo & (1 << p_idx):
                    i, j = swap_pairs[p_idx]
                    if i < n_atoms and j < n_atoms:
                        trial[i], trial[j] = trial[j].clone(), trial[i].clone()

            # Local RMSD for this residue
            diff = (trial - target[indices]) ** 2
            rmsd = diff.sum().sqrt()
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_pred = trial

        pred_resolved[indices] = best_pred

    return pred_resolved
