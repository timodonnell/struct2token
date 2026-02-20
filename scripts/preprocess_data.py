#!/usr/bin/env python3
"""Scan mmCIF files and build an index parquet for fast dataset loading.

Usage:
    python scripts/preprocess_data.py --mmcif_dir ~/tim1/helico-data/raw/mmCIF \
        --output data/index.parquet
"""

from __future__ import annotations

import argparse
import gzip
import io
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def count_atoms_quick(path: Path) -> dict | None:
    """Quick atom count from mmCIF without full parsing.

    Returns dict with (path, n_atoms, chain_id, entity_type) or None on failure.
    """
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
        results = []

        for chain in model.get_chains():
            residues = [r for r in chain.get_residues() if r.get_id()[0] != "W"]
            if not residues:
                continue

            # Count heavy atoms
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

            results.append({
                "path": str(path),
                "chain_id": chain.get_id(),
                "entity_type": entity_type,
                "n_atoms": n_atoms,
                "n_residues": len(residues),
            })

        return results
    except Exception:
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Build mmCIF index")
    parser.add_argument("--mmcif_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/index.parquet")
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    mmcif_dir = Path(args.mmcif_dir).expanduser()
    cif_files = sorted(mmcif_dir.glob("**/*.cif.gz")) + sorted(mmcif_dir.glob("**/*.cif"))

    if args.max_files:
        cif_files = cif_files[: args.max_files]

    print(f"Found {len(cif_files)} mmCIF files in {mmcif_dir}")

    all_records = []
    for path in tqdm(cif_files, desc="Indexing"):
        records = count_atoms_quick(path)
        if records:
            all_records.extend(records)

    df = pd.DataFrame(all_records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} chains to {output_path}")
    print(f"Entity type distribution:\n{df['entity_type'].value_counts()}")
    print(f"Atom count stats:\n{df['n_atoms'].describe()}")


if __name__ == "__main__":
    main()
