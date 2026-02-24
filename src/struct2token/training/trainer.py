"""Main training loop for the all-atom Diffusion Autoencoder."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config, InferenceConfig
from ..data.collate import collate_structures
from ..data.dataset import StructureDataset
from ..data.tokens import ENTITY_PROTEIN, ENTITY_RNA, ENTITY_SMALL_MOLECULE
from ..inference.decode import roundtrip
from ..inference.metrics import MetricsAccumulator
from ..losses.rmsd import backbone_rmsd, compute_rmsd
from ..model.dae import AllAtomDAE
from .augmentation import apply_random_rotation
from .ema import EMA

NM_TO_ANGSTROM = 10.0

ENTITY_NAMES = {
    ENTITY_PROTEIN: "protein",
    ENTITY_RNA: "rna",
    ENTITY_SMALL_MOLECULE: "small_molecule",
}


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

    @torch.no_grad()
    def _validate_structural(self, val_loader: DataLoader, max_batches: int = 10) -> dict:
        """Roundtrip decode on val batches and compute structural metrics.

        Uses EMA weights and 100 diffusion steps (same as benchmark).
        Returns flat dict of wandb metrics in Angstroms.
        """
        self.model.eval()
        inf_config = InferenceConfig(n_steps=100)

        overall = MetricsAccumulator()
        per_entity: dict[int, MetricsAccumulator] = {
            ENTITY_PROTEIN: MetricsAccumulator(),
            ENTITY_RNA: MetricsAccumulator(),
            ENTITY_SMALL_MOLECULE: MetricsAccumulator(),
        }

        with self.ema.average_parameters():
            for i, batch in enumerate(val_loader):
                if i >= max_batches:
                    break
                batch = self._move_batch(batch)

                pred_coords, _ = roundtrip(self.model, batch, inf_config)

                # Pad pred to match batch padding length (collate pads to multiple of 8)
                N_pred = pred_coords.shape[1]
                N_batch = batch["coords"].shape[1]
                if N_pred < N_batch:
                    pad = torch.zeros(
                        pred_coords.shape[0], N_batch - N_pred, 3,
                        device=pred_coords.device,
                    )
                    pred_coords = torch.cat([pred_coords, pad], dim=1)

                # Update overall accumulator
                overall.update(
                    pred=pred_coords,
                    target=batch["coords"],
                    meta_classes=batch["meta_classes"],
                    residue_ids=batch["residue_ids"],
                    padding_mask=batch["padding_mask"],
                )

                # Update per-entity accumulators (one sample at a time)
                B = pred_coords.shape[0]
                entity_types = batch["entity_types"]
                for b in range(B):
                    etype = entity_types[b].item()
                    if etype in per_entity:
                        per_entity[etype].update(
                            pred=pred_coords[b : b + 1],
                            target=batch["coords"][b : b + 1],
                            meta_classes=batch["meta_classes"][b : b + 1],
                            residue_ids=batch["residue_ids"][b : b + 1],
                            padding_mask=batch["padding_mask"][b : b + 1],
                        )

        self.model.train()

        # Build wandb metrics dict
        result = {}
        summary = overall.summary()

        def _convert(src: dict, prefix: str, dst: dict):
            for k, v in src.items():
                if k == "n_samples":
                    dst[f"{prefix}/n_samples"] = v
                    continue
                metric_name, stat = k.rsplit("/", 1)
                if stat != "mean":
                    continue
                # Convert nm→Å for RMSD/RMSE metrics, leave TM-score as-is
                if "rmsd" in metric_name or "rmse" in metric_name:
                    dst[f"{prefix}/{metric_name}"] = v * NM_TO_ANGSTROM
                else:
                    dst[f"{prefix}/{metric_name}"] = v

        _convert(summary, "val", result)

        for etype, acc in per_entity.items():
            if acc.all_atom_rmsd:
                ename = ENTITY_NAMES[etype]
                _convert(acc.summary(), f"val/{ename}", result)

        return result

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

                # Loss validation (fast, forward-only)
                val_metrics = self.validate(self._val_loader)

                # Structural validation (roundtrip decode)
                struct_metrics = self._validate_structural(self._val_loader)
                val_metrics.update(struct_metrics)

                if val_metrics:
                    # Print summary line
                    fl = val_metrics.get("val/flow_loss", 0)
                    sl = val_metrics.get("val/size_loss", 0)
                    aa = val_metrics.get("val/all_atom_rmsd", 0)
                    ca = val_metrics.get("val/ca_rmsd", 0)
                    bb = val_metrics.get("val/backbone_rmsd", 0)
                    sc = val_metrics.get("val/sidechain_rmsd", 0)
                    tm = val_metrics.get("val/tm_score", 0)
                    ns = val_metrics.get("val/n_samples", 0)
                    print(
                        f"  [Val] flow={fl:.2f} size={sl:.1f} | "
                        f"RMSD: aa={aa:.1f}\u00c5 ca={ca:.1f}\u00c5 "
                        f"bb={bb:.1f}\u00c5 sc={sc:.1f}\u00c5 "
                        f"TM={tm:.3f} ({ns} samples)"
                    )
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
