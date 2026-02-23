"""Main training loop for the all-atom Diffusion Autoencoder."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.collate import collate_structures
from ..data.dataset import StructureDataset
from ..losses.rmsd import backbone_rmsd, compute_rmsd
from ..model.dae import AllAtomDAE
from .augmentation import apply_random_rotation
from .ema import EMA


def _cosine_schedule_with_warmup(step: int, warmup_steps: int, max_steps: int) -> float:
    """Cosine decay with linear warmup. Returns lr multiplier in [0, 1]."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    """Training loop for AllAtomDAE.

    Handles: optimizer, scheduler, EMA, gradient clipping, checkpointing,
    validation, WandB logging.
    """

    def __init__(self, config: Config, model: AllAtomDAE, device: torch.device):
        self.config = config
        self.tc = config.training
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.tc.lr,
            weight_decay=self.tc.weight_decay,
            betas=self.tc.betas,
        )
        self.ema = EMA(model, decay=self.tc.ema_decay)

        self.global_step = 0
        self.wandb_run = None

    def _get_lr_multiplier(self) -> float:
        return _cosine_schedule_with_warmup(
            self.global_step, self.tc.warmup_steps, self.tc.max_steps
        )

    def _update_lr(self):
        mult = self._get_lr_multiplier()
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.tc.lr * mult

    def _build_dataloader(self, training: bool) -> DataLoader:
        label = "train" if training else "val"
        print(f"Building {label} dataloader...")
        ds = StructureDataset(
            index_path=self.config.data.index_path,
            cache_dir=self.config.data.cache_dir,
            max_atoms=self.config.data.max_atoms,
            min_atoms=self.config.data.min_atoms,
            training=training,
        )
        print(f"  {label} dataset: {len(ds):,} samples")
        num_workers = self.config.data.num_workers
        return DataLoader(
            ds,
            batch_size=self.tc.batch_size,
            shuffle=training,
            num_workers=num_workers,
            collate_fn=collate_structures,
            pin_memory=True,
            drop_last=training,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    def save_checkpoint(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.ema.load_state_dict(ckpt["ema"])
        self.global_step = ckpt["global_step"]

    def _move_batch(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, max_batches: int = 50) -> dict:
        """Run validation: decode with EMA model and compute metrics."""
        self.model.eval()
        with self.ema.average_parameters():
            total_flow = 0.0
            total_size = 0.0
            count = 0

            for i, batch in enumerate(val_loader):
                if i >= max_batches:
                    break
                batch = self._move_batch(batch)
                losses = self.model(batch)
                total_flow += losses["flow_loss"].item()
                total_size += losses["size_loss"].item()
                count += 1

        self.model.train()
        if count == 0:
            return {}
        return {
            "val/flow_loss": total_flow / count,
            "val/size_loss": total_size / count,
        }

    def train(
        self,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        resume_path: str | Path | None = None,
    ):
        """Main training loop."""
        if train_loader is None:
            train_loader = self._build_dataloader(training=True)
        # val_loader built lazily on first validation
        self._val_loader = val_loader

        if resume_path is not None:
            self.load_checkpoint(resume_path)
            print(f"Resumed from step {self.global_step}")

        # Init wandb
        if self.tc.wandb_project:
            import wandb
            self.wandb_run = wandb.init(
                project=self.tc.wandb_project,
                name=self.tc.wandb_run_name,
                config=self.config.to_dict(),
                resume="allow",
            )
        else:
            print("WandB disabled (--no-wandb).")

        ckpt_dir = Path(self.tc.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model.train()
        flow_weight = 1.0
        size_weight = self.config.model.size_loss_weight

        data_iter = iter(train_loader)
        t_start = time.time()
        print("Starting training loop...")

        while self.global_step < self.tc.max_steps:
            # Get batch (loop dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = self._move_batch(batch)
            batch = apply_random_rotation(batch)

            # Forward
            self.optimizer.zero_grad()
            loss_dict = self.model(batch)
            total_loss = (
                flow_weight * loss_dict["flow_loss"]
                + size_weight * loss_dict["size_loss"]
            )

            # Backward
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
            self.optimizer.step()
            self._update_lr()
            self.ema.update()

            self.global_step += 1

            # Logging
            if self.global_step % self.tc.log_every == 0:
                elapsed = time.time() - t_start
                lr = self.optimizer.param_groups[0]["lr"]
                log = {
                    "train/flow_loss": loss_dict["flow_loss"].item(),
                    "train/size_loss": loss_dict["size_loss"].item(),
                    "train/total_loss": total_loss.item(),
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/lr": lr,
                    "train/step": self.global_step,
                    "train/elapsed_s": elapsed,
                }
                print(
                    f"[Step {self.global_step}] "
                    f"flow={log['train/flow_loss']:.4f} "
                    f"size={log['train/size_loss']:.4f} "
                    f"lr={lr:.2e} "
                    f"gnorm={log['train/grad_norm']:.2f}"
                )
                if self.wandb_run:
                    import wandb
                    wandb.log(log, step=self.global_step)

            # Validation
            if self.global_step % self.tc.val_every == 0:
                if self._val_loader is None:
                    self._val_loader = self._build_dataloader(training=False)
                val_metrics = self.validate(self._val_loader)
                if val_metrics:
                    print(f"  [Val] " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                    if self.wandb_run:
                        import wandb
                        wandb.log(val_metrics, step=self.global_step)

            # Checkpointing
            if self.global_step % self.tc.save_every == 0:
                path = ckpt_dir / f"step_{self.global_step}.pt"
                self.save_checkpoint(path)
                print(f"  Saved checkpoint: {path}")

        # Final save
        self.save_checkpoint(ckpt_dir / "final.pt")
        if self.wandb_run:
            import wandb
            wandb.finish()
        print("Training complete.")
