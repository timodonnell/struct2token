"""Atom-type and residue-type token definitions for all-atom tokenization.

Based on Bio2Token conventions.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Special tokens (shared across vocabularies)
# ---------------------------------------------------------------------------
PAD_IDX = 0
CLS_IDX = 1
EOS_IDX = 2
MASK_IDX = 3
UNK_IDX = 4
NA_IDX = 5

SPECIAL_TOKENS = ["<pad>", "<cls>", "<eos>", "<mask>", "<unk>", "<na>"]
NUM_SPECIAL = len(SPECIAL_TOKENS)

# ---------------------------------------------------------------------------
# Atom type vocabulary (element types)
# ---------------------------------------------------------------------------
ELEMENTS = ["H", "C", "N", "O", "F", "B", "Al", "Si", "P", "S", "Cl", "As", "Br", "I"]

ATOM_TYPE_VOCAB = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
for i, elem in enumerate(ELEMENTS):
    ATOM_TYPE_VOCAB[elem] = NUM_SPECIAL + i

NUM_ATOM_TYPES = len(ATOM_TYPE_VOCAB)  # 20

# ---------------------------------------------------------------------------
# Residue type vocabulary
# ---------------------------------------------------------------------------
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "SEC",  # selenocysteine
    "UNK",  # unknown amino acid
]

RNA_BASES = ["A", "C", "G", "U", "DA"]  # DA = deoxy-adenine / catch-all

RESIDUE_TYPE_VOCAB = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
offset = NUM_SPECIAL
for i, aa in enumerate(AMINO_ACIDS):
    RESIDUE_TYPE_VOCAB[aa] = offset + i
offset += len(AMINO_ACIDS)
for i, base in enumerate(RNA_BASES):
    RESIDUE_TYPE_VOCAB[base] = offset + i

NUM_RESIDUE_TYPES = NUM_SPECIAL + len(AMINO_ACIDS) + len(RNA_BASES)  # 33

# ---------------------------------------------------------------------------
# Metastructure classes
# ---------------------------------------------------------------------------
META_BACKBONE = 0
META_CREF = 1       # C-alpha (protein) or C3' (RNA)
META_SIDECHAIN = 2
META_PAD = 3

NUM_META_CLASSES = 4

# ---------------------------------------------------------------------------
# Entity types (for per-entity-type metric tracking)
# ---------------------------------------------------------------------------
ENTITY_PROTEIN = 0
ENTITY_RNA = 1
ENTITY_SMALL_MOLECULE = 2
ENTITY_TYPE_MAP = {"protein": 0, "rna": 1, "small_molecule": 2}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def element_to_idx(element: str) -> int:
    """Map element symbol to atom type index."""
    return ATOM_TYPE_VOCAB.get(element.strip(), UNK_IDX)


def residue_to_idx(resname: str) -> int:
    """Map residue name to residue type index."""
    resname = resname.strip().upper()
    # Standard 3-letter code lookup
    if resname in RESIDUE_TYPE_VOCAB:
        return RESIDUE_TYPE_VOCAB[resname]
    # Single-letter RNA codes
    if len(resname) == 1 and resname in RESIDUE_TYPE_VOCAB:
        return RESIDUE_TYPE_VOCAB[resname]
    return UNK_IDX
