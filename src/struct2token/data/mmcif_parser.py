"""Parse gzipped mmCIF files into all-atom feature dictionaries.

Uses BioPython's MMCIFParser. Filters hydrogens and organizes atoms
per-residue in canonical ordering following Bio2Token conventions.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .molecule_conventions import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_CREF_ATOM,
    PROTEIN_SIDECHAIN_ATOMS,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS,
    RNA_CREF_ATOM,
)
from .tokens import (
    META_BACKBONE,
    META_CREF,
    META_PAD,
    META_SIDECHAIN,
    PAD_IDX,
    element_to_idx,
    residue_to_idx,
)


def _detect_entity_type(residues: list) -> str:
    """Heuristic: if most residues have CA, it's protein; if P, it's RNA."""
    n_ca = sum(1 for r in residues if "CA" in {a.get_name() for a in r.get_atoms()})
    n_p = sum(1 for r in residues if "P" in {a.get_name() for a in r.get_atoms()})
    if n_ca > n_p:
        return "protein"
    elif n_p > 0:
        return "rna"
    return "small_molecule"


def _process_protein_residue(
    residue, resname: str
) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Extract atoms from a protein residue in canonical order.

    Returns (coords, atom_types, meta_classes, known_mask) arrays.
    """
    atom_dict = {a.get_name(): a for a in residue.get_atoms()
                 if a.element.strip() not in ("H", "D", "")}

    coords: list[np.ndarray] = []
    atypes: list[int] = []
    metas: list[int] = []
    known: list[int] = []

    # Backbone atoms
    for atom_name in PROTEIN_BACKBONE_ATOMS:
        if atom_name in atom_dict:
            a = atom_dict[atom_name]
            coords.append(a.get_vector().get_array())
            atypes.append(element_to_idx(a.element))
            meta = META_CREF if atom_name == PROTEIN_CREF_ATOM else META_BACKBONE
            metas.append(meta)
            known.append(1)
        else:
            coords.append(np.zeros(3, dtype=np.float32))
            atypes.append(PAD_IDX)
            metas.append(META_PAD)
            known.append(0)

    # Sidechain atoms
    sc_names = PROTEIN_SIDECHAIN_ATOMS.get(resname, [])
    for atom_name in sc_names:
        if atom_name in atom_dict:
            a = atom_dict[atom_name]
            coords.append(a.get_vector().get_array())
            atypes.append(element_to_idx(a.element))
            metas.append(META_SIDECHAIN)
            known.append(1)
        else:
            coords.append(np.zeros(3, dtype=np.float32))
            atypes.append(PAD_IDX)
            metas.append(META_PAD)
            known.append(0)

    return (
        np.array(coords, dtype=np.float32),
        atypes,
        metas,
        known,
    )


def _process_rna_residue(
    residue, resname: str
) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Extract atoms from an RNA residue in canonical order."""
    atom_dict = {a.get_name(): a for a in residue.get_atoms()
                 if a.element.strip() not in ("H", "D", "")}

    coords: list[np.ndarray] = []
    atypes: list[int] = []
    metas: list[int] = []
    known: list[int] = []

    # Backbone
    for atom_name in RNA_BACKBONE_ATOMS:
        if atom_name in atom_dict:
            a = atom_dict[atom_name]
            coords.append(a.get_vector().get_array())
            atypes.append(element_to_idx(a.element))
            meta = META_CREF if atom_name == RNA_CREF_ATOM else META_BACKBONE
            metas.append(meta)
            known.append(1)
        else:
            coords.append(np.zeros(3, dtype=np.float32))
            atypes.append(PAD_IDX)
            metas.append(META_PAD)
            known.append(0)

    # Base atoms
    base_names = RNA_BASE_ATOMS.get(resname, [])
    for atom_name in base_names:
        if atom_name in atom_dict:
            a = atom_dict[atom_name]
            coords.append(a.get_vector().get_array())
            atypes.append(element_to_idx(a.element))
            metas.append(META_SIDECHAIN)
            known.append(1)
        else:
            coords.append(np.zeros(3, dtype=np.float32))
            atypes.append(PAD_IDX)
            metas.append(META_PAD)
            known.append(0)

    return (
        np.array(coords, dtype=np.float32),
        atypes,
        metas,
        known,
    )


def _process_small_molecule_residue(
    residue,
) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Extract all non-hydrogen atoms from a small molecule residue."""
    coords: list[np.ndarray] = []
    atypes: list[int] = []
    metas: list[int] = []
    known: list[int] = []

    for a in residue.get_atoms():
        if a.element.strip() in ("H", "D", ""):
            continue
        coords.append(a.get_vector().get_array())
        atypes.append(element_to_idx(a.element))
        metas.append(META_BACKBONE)  # all atoms treated as backbone
        known.append(1)

    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.float32), [], [], []

    return np.array(coords, dtype=np.float32), atypes, metas, known


def parse_mmcif(
    path: str | Path,
    chain_id: Optional[str] = None,
) -> Dict[str, np.ndarray | str]:
    """Parse a gzipped mmCIF file into all-atom features.

    Args:
        path: Path to .cif.gz or .cif file.
        chain_id: If provided, only parse this chain. Otherwise parse first
            polymer chain.

    Returns:
        Dictionary with keys:
            coords: (N_atoms, 3) float32 — coordinates in Angstroms
            atom_types: (N_atoms,) int32 — element type indices
            residue_types: (N_atoms,) int32 — residue type indices
            residue_ids: (N_atoms,) int32 — residue index per atom
            meta_classes: (N_atoms,) int32 — backbone/cref/sidechain/pad
            known_mask: (N_atoms,) int32 — 1 if atom has known coords
            entity_type: str — "protein" | "rna" | "small_molecule"
    """
    from Bio.PDB import MMCIFParser as BioMMCIFParser

    parser = BioMMCIFParser(QUIET=True)
    path = Path(path)

    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            content = f.read()
        handle = io.StringIO(content)
        structure = parser.get_structure("s", handle)
    else:
        structure = parser.get_structure("s", str(path))

    model = structure[0]

    # Select chain
    if chain_id is not None:
        chain = model[chain_id]
    else:
        chains = list(model.get_chains())
        if not chains:
            raise ValueError(f"No chains found in {path}")
        chain = chains[0]

    # Gather residues (skip HOH, and hetero residues unless small_molecule)
    residues = []
    for res in chain.get_residues():
        het_flag = res.get_id()[0]
        if het_flag == "W":  # water
            continue
        residues.append(res)

    if not residues:
        raise ValueError(f"No residues found in chain {chain.get_id()} of {path}")

    entity_type = _detect_entity_type(residues)

    all_coords: List[np.ndarray] = []
    all_atypes: List[int] = []
    all_rtypes: List[int] = []
    all_rids: List[int] = []
    all_metas: List[int] = []
    all_known: List[int] = []

    for rid, residue in enumerate(residues):
        resname = residue.get_resname().strip()

        if entity_type == "protein":
            coords, atypes, metas, known = _process_protein_residue(residue, resname)
            rtype_idx = residue_to_idx(resname)
        elif entity_type == "rna":
            # Normalize RNA residue names
            rna_name = resname
            if len(resname) > 1:
                # e.g., "ADE" -> "A", "GUA" -> "G", etc.
                rna_map = {"ADE": "A", "GUA": "G", "CYT": "C", "URA": "U", "URI": "U"}
                rna_name = rna_map.get(resname, resname)
            coords, atypes, metas, known = _process_rna_residue(residue, rna_name)
            rtype_idx = residue_to_idx(rna_name)
        else:
            coords, atypes, metas, known = _process_small_molecule_residue(residue)
            rtype_idx = residue_to_idx(resname)

        n = len(atypes)
        if n == 0:
            continue

        all_coords.append(coords)
        all_atypes.extend(atypes)
        all_rtypes.extend([rtype_idx] * n)
        all_rids.extend([rid] * n)
        all_metas.extend(metas)
        all_known.extend(known)

    if len(all_atypes) == 0:
        raise ValueError(f"No atoms extracted from {path}")

    return {
        "coords": np.concatenate(all_coords, axis=0),
        "atom_types": np.array(all_atypes, dtype=np.int32),
        "residue_types": np.array(all_rtypes, dtype=np.int32),
        "residue_ids": np.array(all_rids, dtype=np.int32),
        "meta_classes": np.array(all_metas, dtype=np.int32),
        "known_mask": np.array(all_known, dtype=np.int32),
        "entity_type": entity_type,
    }
