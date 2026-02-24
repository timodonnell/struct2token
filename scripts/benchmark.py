#!/usr/bin/env python3
"""Benchmark struct2token against published baselines (APT, Bio2Token, Kanzi).

Evaluates on CASP14/CASP15 datasets at multiple token counts to measure
adaptive tokenization quality, then prints comparison tables.

Usage:
    uv run scripts/benchmark.py --checkpoint checkpoints/step_10000.pt \
        --datasets casp14 casp15 \
        --n_tokens 16 32 64 128 \
        --output results/benchmark.json

    # Quick single-dataset evaluation:
    uv run scripts/benchmark.py --checkpoint checkpoints/step_10000.pt \
        --datasets casp14 --n_tokens 128 --no-wandb
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from struct2token.benchmark.baselines import format_comparison_table
from struct2token.config import Config, InferenceConfig
from struct2token.data.collate import collate_structures
from struct2token.data.dataset import StructureDataset
from struct2token.inference.decode import roundtrip, roundtrip_with_n_tokens
from struct2token.inference.metrics import MetricsAccumulator
from struct2token.model.dae import AllAtomDAE
from struct2token.training.ema import EMA

NM_TO_ANGSTROM = 10.0


def evaluate_dataset(
    model: AllAtomDAE,
    loader: DataLoader,
    n_tokens: int,
    inf_config: InferenceConfig,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset at a specific token count.

    Returns dict with per-sample lists and aggregate stats, all in Angstroms.
    """
    accumulator = MetricsAccumulator()
    K = model.n_tokens  # full token count (128)

    for batch in tqdm(loader, desc=f"  n_tokens={n_tokens}", leave=False):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Use truncated roundtrip if n_tokens < K, else full roundtrip
        if n_tokens < K:
            pred_coords, indices = roundtrip_with_n_tokens(
                model, batch, n_tokens, inf_config
            )
        else:
            pred_coords, indices = roundtrip(model, batch, inf_config)

        accumulator.update(
            pred=pred_coords,
            target=batch["coords"],
            meta_classes=batch["meta_classes"],
            residue_ids=batch["residue_ids"],
            padding_mask=batch["padding_mask"],
        )

    # Get summary (in nm) and convert to Angstroms
    summary = accumulator.summary()
    result = {}
    for k, v in summary.items():
        if isinstance(v, float) and "rmsd" in k.lower() or "rmse" in k.lower():
            result[k] = v * NM_TO_ANGSTROM
        else:
            result[k] = v

    # Also store per-sample values (converted to Angstroms)
    result["per_sample"] = {
        "all_atom_rmsd": [v * NM_TO_ANGSTROM for v in accumulator.all_atom_rmsd],
        "ca_rmsd": [v * NM_TO_ANGSTROM for v in accumulator.ca_rmsd_values],
        "backbone_rmsd": [v * NM_TO_ANGSTROM for v in accumulator.bb_rmsd],
        "sidechain_rmsd": [v * NM_TO_ANGSTROM for v in accumulator.sc_rmsd],
        "tm_score": list(accumulator.tm_scores),
        "dist_rmse": [v * NM_TO_ANGSTROM for v in accumulator.dist_rmse],
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark struct2token against published baselines"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["casp14", "casp15"],
        help="Benchmark datasets to evaluate on",
    )
    parser.add_argument(
        "--n_tokens",
        nargs="+",
        type=int,
        default=[128],
        help="Token counts to evaluate (tests adaptive tokenization)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=100, help="Diffusion sampling steps")
    parser.add_argument("--cfg_weight", type=float, default=2.0)
    parser.add_argument("--noise_weight", type=float, default=0.45)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per dataset")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--index_dir", type=str, default="data/benchmark",
                        help="Directory containing {dataset}_index.parquet files")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="struct2token")
    args = parser.parse_args()

    # Device
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config: Config = ckpt["config"]

    model = AllAtomDAE(config.model).to(device)
    model.load_state_dict(ckpt["model"])

    # Load EMA weights if available
    if "ema" in ckpt:
        ema = EMA(model, decay=config.training.ema_decay)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()
        print("Using EMA weights")

    model.eval()

    inf_config = InferenceConfig(
        n_steps=args.n_steps,
        cfg_weight=args.cfg_weight,
        noise_weight=args.noise_weight,
    )

    # Optional wandb logging
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"benchmark-{Path(args.checkpoint).stem}",
                config={
                    "checkpoint": args.checkpoint,
                    "datasets": args.datasets,
                    "n_tokens": args.n_tokens,
                    "n_steps": args.n_steps,
                    "cfg_weight": args.cfg_weight,
                },
            )
        except Exception as e:
            print(f"wandb init failed: {e}, continuing without logging")

    # Results container
    all_results = {}
    # For comparison table: keyed by dataset or dataset_nN
    table_results = {}

    n_tokens_sorted = sorted(args.n_tokens)

    for dataset_name in args.datasets:
        index_path = Path(args.index_dir) / f"{dataset_name}_index.parquet"
        if not index_path.exists():
            print(f"\nSkipping {dataset_name}: index not found at {index_path}")
            print(f"  Run: uv run scripts/prepare_benchmark_data.py --datasets {dataset_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name.upper()}")
        print(f"{'='*60}")

        dataset = StructureDataset(
            index_path=str(index_path),
            cache_dir=config.data.cache_dir,
            max_atoms=config.data.max_atoms,
            min_atoms=config.data.min_atoms,
            training=False,
        )

        if args.max_samples and len(dataset) > args.max_samples:
            indices = list(range(args.max_samples))
            dataset = torch.utils.data.Subset(dataset, indices)

        print(f"  Samples: {len(dataset)}")

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_structures,
            num_workers=2,
        )

        dataset_results = {}

        for n_tok in n_tokens_sorted:
            t0 = time.time()
            result = evaluate_dataset(model, loader, n_tok, inf_config, device)
            elapsed = time.time() - t0

            dataset_results[n_tok] = result

            # Key for comparison table
            if len(n_tokens_sorted) > 1:
                table_key = f"{dataset_name}_n{n_tok}"
            else:
                table_key = dataset_name

            table_results[table_key] = {
                "ca_rmsd": result.get("ca_rmsd/mean"),
                "all_atom_rmsd": result.get("all_atom_rmsd/mean"),
                "tm_score": result.get("tm_score/mean"),
                "backbone_rmsd": result.get("backbone_rmsd/mean"),
                "sidechain_rmsd": result.get("sidechain_rmsd/mean"),
            }

            # Print summary
            print(f"\n  n_tokens={n_tok} ({elapsed:.1f}s):")
            print(f"    Cα RMSD:       {result.get('ca_rmsd/mean', 0):.3f} Å "
                  f"(±{result.get('ca_rmsd/std', 0):.3f})")
            print(f"    All-atom RMSD: {result.get('all_atom_rmsd/mean', 0):.3f} Å "
                  f"(±{result.get('all_atom_rmsd/std', 0):.3f})")
            print(f"    Backbone RMSD: {result.get('backbone_rmsd/mean', 0):.3f} Å")
            print(f"    Sidechain RMSD:{result.get('sidechain_rmsd/mean', 0):.3f} Å")
            print(f"    TM-score:      {result.get('tm_score/mean', 0):.4f} "
                  f"(±{result.get('tm_score/std', 0):.4f})")
            print(f"    Dist RMSE:     {result.get('dist_rmse/mean', 0):.3f} Å")
            print(f"    Samples:       {result.get('n_samples', 0)}")

            # Log to wandb
            if wandb_run is not None:
                log_dict = {
                    f"{dataset_name}/n{n_tok}/{k}": v
                    for k, v in result.items()
                    if isinstance(v, (int, float))
                }
                wandb_run.log(log_dict)

        all_results[dataset_name] = dataset_results

    # Print comparison tables
    print(f"\n{'='*60}")
    print("COMPARISON WITH PUBLISHED BASELINES")
    print(f"{'='*60}\n")

    table = format_comparison_table(
        table_results,
        datasets=args.datasets,
        n_tokens_list=n_tokens_sorted,
    )
    print(table)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Make JSON-serializable (remove per-sample data for compact output)
        save_data = {
            "checkpoint": args.checkpoint,
            "datasets": args.datasets,
            "n_tokens": args.n_tokens,
            "config": {
                "n_steps": args.n_steps,
                "cfg_weight": args.cfg_weight,
                "noise_weight": args.noise_weight,
            },
            "results": {},
        }
        for ds, ds_results in all_results.items():
            save_data["results"][ds] = {}
            for n_tok, result in ds_results.items():
                # Save aggregate stats + per-sample
                save_data["results"][ds][str(n_tok)] = result

        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {output_path}")

    if wandb_run is not None:
        # Log comparison table as artifact
        wandb_run.log({"comparison_table": wandb.Html(f"<pre>{table}</pre>")})
        wandb_run.finish()


if __name__ == "__main__":
    main()
