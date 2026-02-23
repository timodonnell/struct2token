#!/usr/bin/env python3
"""Train the all-atom Diffusion Autoencoder tokenizer.

Usage:
    uv run scripts/train_tokenizer.py --config configs/default.yaml
    uv run scripts/train_tokenizer.py --config configs/default.yaml --resume checkpoints/step_10000.pt
    uv run scripts/train_tokenizer.py --config configs/default.yaml --wandb_run_name my-run --batch_size 4
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure print output appears immediately (not buffered until process exit)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
import random

import numpy as np
import torch

from struct2token.config import Config
from struct2token.model.dae import AllAtomDAE
from struct2token.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train struct2token DAE")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    # Training overrides
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Data overrides
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--max_atoms", type=int, default=None)

    # WandB overrides
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    # Apply CLI overrides
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.seed is not None:
        config.training.seed = args.seed
    if args.index_path is not None:
        config.data.index_path = args.index_path
    if args.max_atoms is not None:
        config.data.max_atoms = args.max_atoms
    if args.wandb_project is not None:
        config.training.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.training.wandb_run_name = args.wandb_run_name
    if args.no_wandb:
        config.training.wandb_project = ""  # empty string disables wandb in trainer

    # Seed
    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model
    model = AllAtomDAE(config.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    trainer = Trainer(config, model, device)
    trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()
