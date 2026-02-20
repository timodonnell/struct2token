"""Per-residue canonical atom ordering for proteins and RNA.

Based on Bio2Token conventions. Each residue type maps to an ordered list of
atom names, split into backbone and sidechain atoms.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Protein backbone atoms (same for all amino acids)
# ---------------------------------------------------------------------------
PROTEIN_BACKBONE_ATOMS: List[str] = ["N", "CA", "C", "O"]
PROTEIN_CREF_ATOM: str = "CA"

# ---------------------------------------------------------------------------
# Protein sidechain atoms per residue type (canonical ordering)
# ---------------------------------------------------------------------------
PROTEIN_SIDECHAIN_ATOMS: Dict[str, List[str]] = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CB", "CG", "CD"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
    "SEC": ["CB", "SE"],  # selenocysteine
    "UNK": [],
}

# Maximum sidechain atoms across all residues (TRP has 10)
MAX_SIDECHAIN_ATOMS: int = max(len(v) for v in PROTEIN_SIDECHAIN_ATOMS.values())

# ---------------------------------------------------------------------------
# RNA backbone atoms
# ---------------------------------------------------------------------------
RNA_BACKBONE_ATOMS: List[str] = [
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
    "C2'", "O2'", "C1'",
]
RNA_CREF_ATOM: str = "C3'"

# ---------------------------------------------------------------------------
# RNA base atoms per nucleotide
# ---------------------------------------------------------------------------
RNA_BASE_ATOMS: Dict[str, List[str]] = {
    "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
    "DA": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_protein_atom_order(resname: str) -> Tuple[List[str], List[str]]:
    """Return (backbone_atoms, sidechain_atoms) for a protein residue."""
    sc = PROTEIN_SIDECHAIN_ATOMS.get(resname, PROTEIN_SIDECHAIN_ATOMS["UNK"])
    return PROTEIN_BACKBONE_ATOMS, sc


def get_rna_atom_order(resname: str) -> Tuple[List[str], List[str]]:
    """Return (backbone_atoms, base_atoms) for an RNA residue."""
    base = RNA_BASE_ATOMS.get(resname, [])
    return RNA_BACKBONE_ATOMS, base


def get_max_atoms_per_residue(entity_type: str) -> int:
    """Maximum atoms per residue for a given entity type."""
    if entity_type == "protein":
        return len(PROTEIN_BACKBONE_ATOMS) + MAX_SIDECHAIN_ATOMS
    elif entity_type == "rna":
        max_base = max(len(v) for v in RNA_BASE_ATOMS.values())
        return len(RNA_BACKBONE_ATOMS) + max_base
    else:
        return 50  # arbitrary cap for small molecules
