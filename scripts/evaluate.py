#!/usr/bin/env python3
"""Evaluate a trained struct2token model on a test set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/final.pt \
        --index data/index.parquet --max_samples 500
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from struct2token.config import Config, InferenceConfig
from struct2token.data.collate import collate_structures
from struct2token.data.dataset import StructureDataset
from struct2token.inference.decode import roundtrip
from struct2token.inference.metrics import MetricsAccumulator
from struct2token.model.dae import AllAtomDAE
from struct2token.training.ema import EMA


def main():
    parser = argparse.ArgumentParser(description="Evaluate struct2token")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index", type=str, default=None, help="Override index path")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--cfg_weight", type=float, default=2.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

    # Load checkpoint
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config: Config = ckpt["config"]

    # Build model and load weights
    model = AllAtomDAE(config.model).to(device)
    model.load_state_dict(ckpt["model"])

    # Load EMA weights if available
    if "ema" in ckpt:
        ema = EMA(model, decay=config.training.ema_decay)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()
        print("Using EMA weights")

    model.eval()

    # Inference config
    inf_config = InferenceConfig(
        n_steps=args.n_steps,
        cfg_weight=args.cfg_weight,
    )

    # Dataset
    index_path = args.index or config.data.index_path
    dataset = StructureDataset(
        index_path=index_path,
        cache_dir=config.data.cache_dir,
        max_atoms=config.data.max_atoms,
        min_atoms=config.data.min_atoms,
        training=False,
    )

    # Limit samples
    if args.max_samples and len(dataset) > args.max_samples:
        indices = list(range(args.max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_structures,
        num_workers=2,
    )

    # Evaluate
    accumulator = MetricsAccumulator()

    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        pred_coords, indices = roundtrip(model, batch, inf_config)

        # Pad pred to match batch padding (collate pads to multiple of 8,
        # but decode produces max(lengths) atoms)
        N_pred = pred_coords.shape[1]
        N_batch = batch["coords"].shape[1]
        if N_pred < N_batch:
            pad = torch.zeros(
                pred_coords.shape[0], N_batch - N_pred, 3,
                device=pred_coords.device,
            )
            pred_coords = torch.cat([pred_coords, pad], dim=1)

        accumulator.update(
            pred=pred_coords,
            target=batch["coords"],
            meta_classes=batch["meta_classes"],
            residue_ids=batch["residue_ids"],
            padding_mask=batch["padding_mask"],
        )

    results = accumulator.summary()
    print("\n=== Evaluation Results ===")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
