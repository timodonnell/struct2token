"""PyTorch Dataset for all-atom structure tokenization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mmcif_parser import parse_mmcif


class StructureDataset(Dataset):
    """Lazy-loading dataset for macromolecular structures.

    Reads an index parquet (path, chain_id, entity_type, n_atoms) and
    loads / caches structures on __getitem__.
    """

    def __init__(
        self,
        index_path: str | Path,
        cache_dir: str | Path | None = None,
        max_atoms: int = 8000,
        min_atoms: int = 10,
        training: bool = True,
    ):
        self.index = pd.read_parquet(index_path)
        # Filter by atom count
        self.index = self.index[
            (self.index["n_atoms"] >= min_atoms)
            & (self.index["n_atoms"] <= max_atoms)
        ].reset_index(drop=True)

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_atoms = max_atoms
        self.training = training

    def __len__(self) -> int:
        return len(self.index)

    def _cache_key(self, path: str, chain_id: str) -> Path:
        """Generate cache file path from mmcif path + chain."""
        h = hashlib.md5(f"{path}_{chain_id}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}.pt"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.index.iloc[idx]
        path = row["path"]
        chain_id = row.get("chain_id", None)

        # Try cache
        cached = None
        if self.cache_dir is not None:
            cache_path = self._cache_key(path, str(chain_id))
            if cache_path.exists():
                cached = torch.load(cache_path, weights_only=True)

        if cached is not None:
            sample = cached
        else:
            parsed = parse_mmcif(path, chain_id=chain_id)
            sample = {
                "coords": torch.from_numpy(parsed["coords"]),          # (N, 3)
                "atom_types": torch.from_numpy(parsed["atom_types"]),   # (N,)
                "residue_types": torch.from_numpy(parsed["residue_types"]),
                "residue_ids": torch.from_numpy(parsed["residue_ids"]),
                "meta_classes": torch.from_numpy(parsed["meta_classes"]),
                "known_mask": torch.from_numpy(parsed["known_mask"]),
            }
            if self.cache_dir is not None:
                torch.save(sample, cache_path)

        # Center coordinates on known atoms
        known = sample["known_mask"].bool()
        if known.any():
            centroid = sample["coords"][known].mean(dim=0, keepdim=True)
            sample["coords"] = sample["coords"] - centroid

        # Convert Angstroms to nm (÷10)
        sample["coords"] = sample["coords"] / 10.0

        # Truncate if needed
        n = sample["coords"].shape[0]
        if n > self.max_atoms:
            for k in sample:
                if isinstance(sample[k], torch.Tensor):
                    sample[k] = sample[k][: self.max_atoms]

        return sample
