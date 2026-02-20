"""Tests for data pipeline: tokens, molecule conventions, collation."""

import torch
import pytest

from struct2token.data.tokens import (
    ATOM_TYPE_VOCAB,
    NUM_ATOM_TYPES,
    NUM_RESIDUE_TYPES,
    RESIDUE_TYPE_VOCAB,
    element_to_idx,
    residue_to_idx,
)
from struct2token.data.molecule_conventions import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_SIDECHAIN_ATOMS,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS,
    get_protein_atom_order,
    get_rna_atom_order,
)
from struct2token.data.collate import collate_structures


class TestTokens:
    def test_atom_type_count(self):
        assert NUM_ATOM_TYPES == 20

    def test_residue_type_count(self):
        assert NUM_RESIDUE_TYPES == 33

    def test_element_lookup(self):
        assert element_to_idx("C") == ATOM_TYPE_VOCAB["C"]
        assert element_to_idx("N") == ATOM_TYPE_VOCAB["N"]
        assert element_to_idx("O") == ATOM_TYPE_VOCAB["O"]

    def test_unknown_element(self):
        idx = element_to_idx("Zr")  # not in vocab
        assert idx == 4  # UNK_IDX

    def test_residue_lookup(self):
        assert residue_to_idx("ALA") == RESIDUE_TYPE_VOCAB["ALA"]
        assert residue_to_idx("GLY") == RESIDUE_TYPE_VOCAB["GLY"]

    def test_unknown_residue(self):
        idx = residue_to_idx("XYZ")
        assert idx == 4  # UNK_IDX


class TestMoleculeConventions:
    def test_protein_backbone(self):
        assert len(PROTEIN_BACKBONE_ATOMS) == 4
        assert "N" in PROTEIN_BACKBONE_ATOMS
        assert "CA" in PROTEIN_BACKBONE_ATOMS

    def test_protein_sidechain_gly(self):
        assert len(PROTEIN_SIDECHAIN_ATOMS["GLY"]) == 0

    def test_protein_sidechain_arg(self):
        assert len(PROTEIN_SIDECHAIN_ATOMS["ARG"]) == 7

    def test_rna_backbone(self):
        assert len(RNA_BACKBONE_ATOMS) == 12

    def test_get_protein_atom_order(self):
        bb, sc = get_protein_atom_order("ALA")
        assert len(bb) == 4
        assert len(sc) == 1
        assert sc[0] == "CB"

    def test_get_rna_atom_order(self):
        bb, base = get_rna_atom_order("A")
        assert len(bb) == 12
        assert len(base) == 10


class TestCollation:
    def _make_sample(self, n_atoms: int) -> dict:
        return {
            "coords": torch.randn(n_atoms, 3),
            "atom_types": torch.randint(0, 20, (n_atoms,)),
            "residue_types": torch.randint(0, 33, (n_atoms,)),
            "residue_ids": torch.zeros(n_atoms, dtype=torch.long),
            "meta_classes": torch.randint(0, 4, (n_atoms,)),
            "known_mask": torch.ones(n_atoms, dtype=torch.long),
        }

    def test_same_length(self):
        batch = [self._make_sample(10), self._make_sample(10)]
        out = collate_structures(batch)
        assert out["coords"].shape[0] == 2
        # Padded to multiple of 8
        assert out["coords"].shape[1] % 8 == 0
        assert out["coords"].shape[1] >= 10

    def test_variable_length(self):
        batch = [self._make_sample(7), self._make_sample(15)]
        out = collate_structures(batch)
        assert out["coords"].shape[0] == 2
        assert out["coords"].shape[1] >= 15
        # Check padding mask
        assert out["padding_mask"][0, :7].all()
        assert not out["padding_mask"][0, 7:].any()
        assert out["padding_mask"][1, :15].all()

    def test_lengths(self):
        batch = [self._make_sample(5), self._make_sample(12), self._make_sample(3)]
        out = collate_structures(batch)
        assert out["lengths"].tolist() == [5, 12, 3]
