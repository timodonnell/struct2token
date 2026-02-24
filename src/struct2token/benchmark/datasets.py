"""Benchmark dataset preparation for CASP14 and CASP15.

Provides hardcoded target lists (PDB ID + chain) and functions to download
structures from RCSB PDB and build index parquets for evaluation.

Target lists sourced from the CASP prediction center (predictioncenter.org).
"""

from __future__ import annotations

import gzip
import io
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd
from tqdm import tqdm

# CASP14 (2020) targets: (target_id, pdb_id, chain_id)
# Curated set of released experimental structures.
# Full list at: https://predictioncenter.org/casp14/targetlist.cgi
CASP14_TARGETS: List[Tuple[str, str, str]] = [
    ("T1024", "6VR4", "A"),
    ("T1025", "6XE1", "A"),
    ("T1027", "6T1Z", "A"),
    ("T1030", "7C3K", "A"),
    ("T1031", "7CDG", "A"),
    ("T1032", "7CDI", "A"),
    ("T1033", "7KDX", "A"),
    ("T1034", "6YA2", "A"),
    ("T1035", "7JTL", "A"),
    ("T1037", "6YJ1", "A"),
    ("T1038", "7KBR", "A"),
    ("T1039", "7KDR", "A"),
    ("T1042", "6N64", "A"),
    ("T1043", "6Y4F", "A"),
    ("T1049", "6VEY", "A"),
    ("T1050", "6T1Z", "B"),
    ("T1053", "7ABW", "A"),
    ("T1056", "7A4D", "A"),
    ("T1058", "7D2O", "A"),
    ("T1061", "7MEZ", "A"),
    ("T1064", "7O2E", "A"),
    ("T1065", "7O2K", "A"),
    ("T1067", "7JZC", "A"),
    ("T1070", "7KSG", "A"),
    ("T1073", "7LHQ", "A"),
    ("T1074", "7EIV", "A"),
    ("T1076", "7NWG", "A"),
    ("T1078", "7DQ8", "A"),
    ("T1079", "7NLI", "A"),
    ("T1080", "7N0U", "A"),
    ("T1082", "7NWI", "A"),
    ("T1083", "7NWH", "A"),
]

# CASP15 (2022) targets: (target_id, pdb_id, chain_id)
# Full list at: https://predictioncenter.org/casp15/targetlist.cgi
CASP15_TARGETS: List[Tuple[str, str, str]] = [
    ("T1104", "7XBX", "A"),
    ("T1106", "8BQR", "A"),
    ("T1109", "8B0X", "A"),
    ("T1110", "8D43", "A"),
    ("T1112", "7ZJJ", "A"),
    ("T1113", "8CEU", "A"),
    ("T1114", "8DWG", "A"),
    ("T1115", "7U3M", "A"),
    ("T1119", "8A31", "A"),
    ("T1120", "8DDQ", "A"),
    ("T1121", "8CF9", "A"),
    ("T1122", "8CME", "A"),
    ("T1123", "8GJL", "A"),
    ("T1124", "8F4I", "A"),
    ("T1125", "8GLP", "A"),
    ("T1127", "8FY0", "A"),
    ("T1129", "8ELJ", "A"),
    ("T1130", "8GHN", "A"),
    ("T1131", "8GHQ", "A"),
    ("T1132", "8GHR", "A"),
    ("T1134", "8HGG", "A"),
    ("T1137", "8HW6", "A"),
    ("T1146", "8IHF", "A"),
    ("T1152", "8JSV", "A"),
    ("T1154", "8OG7", "A"),
    ("T1157", "8SLV", "A"),
    ("T1158", "8SNA", "A"),
    ("T1170", "8QAN", "A"),
    ("T1172", "8S2M", "A"),
    ("T1187", "8UMF", "A"),
]

DATASET_TARGETS = {
    "casp14": CASP14_TARGETS,
    "casp15": CASP15_TARGETS,
}


def get_target_pdb_ids(dataset_name: str) -> List[Tuple[str, str, str]]:
    """Get (target_id, pdb_id, chain_id) tuples for a benchmark dataset."""
    targets = DATASET_TARGETS.get(dataset_name.lower())
    if targets is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_TARGETS.keys())}"
        )
    return targets


def download_mmcif(pdb_id: str, output_dir: Path) -> Optional[Path]:
    """Download a gzipped mmCIF file from RCSB PDB.

    Args:
        pdb_id: 4-character PDB ID (e.g., "6VR4")
        output_dir: Directory to save the file

    Returns:
        Path to downloaded .cif.gz file, or None on failure.
    """
    pdb_id_lower = pdb_id.lower()
    output_path = output_dir / f"{pdb_id_lower}.cif.gz"

    if output_path.exists():
        return output_path

    url = f"https://files.rcsb.org/download/{pdb_id_lower}.cif.gz"
    try:
        with urlopen(url, timeout=30) as response:
            with open(output_path, "wb") as f:
                shutil.copyfileobj(response, f)
        return output_path
    except (URLError, OSError) as e:
        print(f"  Failed to download {pdb_id}: {e}")
        return None


def _find_in_local_mirror(pdb_id: str, mmcif_dir: Path) -> Optional[Path]:
    """Search for a PDB structure in the local mmCIF mirror."""
    pdb_lower = pdb_id.lower()

    # Common directory layouts
    candidates = [
        mmcif_dir / f"{pdb_lower}.cif.gz",
        mmcif_dir / f"{pdb_lower}.cif",
        mmcif_dir / pdb_lower[1:3] / f"{pdb_lower}.cif.gz",
        mmcif_dir / pdb_lower[1:3] / f"{pdb_lower}.cif",
    ]

    for path in candidates:
        if path.exists():
            return path
    return None


def _count_atoms_for_chain(path: Path, chain_id: str) -> Optional[Dict]:
    """Count atoms for a specific chain in an mmCIF file."""
    try:
        from Bio.PDB import MMCIFParser

        parser = MMCIFParser(QUIET=True)

        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                content = f.read()
            handle = io.StringIO(content)
            structure = parser.get_structure("s", handle)
        else:
            structure = parser.get_structure("s", str(path))

        model = structure[0]

        # Find the chain
        for chain in model.get_chains():
            if chain.get_id() != chain_id:
                continue

            residues = [r for r in chain.get_residues() if r.get_id()[0] != "W"]
            if not residues:
                continue

            n_atoms = 0
            n_ca = 0
            n_p = 0
            for res in residues:
                for atom in res.get_atoms():
                    elem = atom.element.strip()
                    if elem in ("H", "D", ""):
                        continue
                    n_atoms += 1
                    if atom.get_name() == "CA":
                        n_ca += 1
                    elif atom.get_name() == "P":
                        n_p += 1

            if n_atoms == 0:
                continue

            if n_ca > n_p:
                entity_type = "protein"
            elif n_p > 0:
                entity_type = "rna"
            else:
                entity_type = "small_molecule"

            return {
                "path": str(path),
                "chain_id": chain_id,
                "entity_type": entity_type,
                "n_atoms": n_atoms,
                "n_residues": len(residues),
            }

        return None
    except Exception:
        traceback.print_exc()
        return None


def prepare_benchmark_index(
    dataset_name: str,
    mmcif_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    download_dir: Optional[str] = None,
) -> str:
    """Prepare a benchmark index parquet for a CASP dataset.

    1. Looks up target PDB IDs for the dataset
    2. Checks local mmCIF mirror for each structure
    3. Downloads missing ones from RCSB PDB
    4. Parses each and builds an index parquet
    5. Returns path to the parquet

    Args:
        dataset_name: "casp14" or "casp15"
        mmcif_dir: Path to local mmCIF mirror (optional)
        output_path: Where to save the index parquet
        download_dir: Directory for downloaded structures (default: data/benchmark/structures/)

    Returns:
        Path to the created index parquet.
    """
    targets = get_target_pdb_ids(dataset_name)
    print(f"Preparing {dataset_name.upper()}: {len(targets)} targets")

    if output_path is None:
        output_path = f"data/benchmark/{dataset_name}_index.parquet"
    if download_dir is None:
        download_dir = "data/benchmark/structures"

    output_path = Path(output_path)
    download_path = Path(download_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    download_path.mkdir(parents=True, exist_ok=True)

    mmcif_path = Path(mmcif_dir).expanduser() if mmcif_dir else None

    records = []
    found = 0
    downloaded = 0
    failed = 0

    for target_id, pdb_id, chain_id in tqdm(targets, desc=f"Processing {dataset_name}"):
        # Try local mirror first
        local_path = None
        if mmcif_path is not None:
            local_path = _find_in_local_mirror(pdb_id, mmcif_path)

        # Try download directory
        if local_path is None:
            local_path = _find_in_local_mirror(pdb_id, download_path)

        # Download if not found locally
        if local_path is None:
            local_path = download_mmcif(pdb_id, download_path)
            if local_path is not None:
                downloaded += 1

        if local_path is None:
            print(f"  Skipping {target_id} ({pdb_id}): not found")
            failed += 1
            continue

        found += 1

        # Count atoms for the specific chain
        record = _count_atoms_for_chain(local_path, chain_id)
        if record is None:
            # Try chain A as fallback
            if chain_id != "A":
                record = _count_atoms_for_chain(local_path, "A")
            # Try first available chain
            if record is None:
                try:
                    from Bio.PDB import MMCIFParser

                    parser = MMCIFParser(QUIET=True)
                    if local_path.suffix == ".gz":
                        with gzip.open(local_path, "rt") as f:
                            content = f.read()
                        handle = io.StringIO(content)
                        structure = parser.get_structure("s", handle)
                    else:
                        structure = parser.get_structure("s", str(local_path))
                    for chain in structure[0].get_chains():
                        record = _count_atoms_for_chain(local_path, chain.get_id())
                        if record is not None:
                            break
                except Exception:
                    pass

        if record is not None:
            record["target_id"] = target_id
            record["pdb_id"] = pdb_id
            records.append(record)
        else:
            print(f"  Skipping {target_id} ({pdb_id}): failed to parse chain {chain_id}")
            failed += 1

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)

    print(f"\n{dataset_name.upper()} summary:")
    print(f"  Targets: {len(targets)}")
    print(f"  Found locally: {found - downloaded}")
    print(f"  Downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Index written: {output_path} ({len(df)} chains)")

    if len(df) > 0:
        print(f"  Entity types: {dict(df['entity_type'].value_counts())}")
        print(f"  Atom count: mean={df['n_atoms'].mean():.0f}, "
              f"median={df['n_atoms'].median():.0f}, "
              f"range=[{df['n_atoms'].min()}, {df['n_atoms'].max()}]")

    return str(output_path)
