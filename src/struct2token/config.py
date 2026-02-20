"""Configuration dataclasses for struct2token."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass
class EncoderConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class FSQConfig:
    levels: Tuple[int, ...] = (8, 5, 5, 5)


@dataclass
class DecoderConfig:
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class DAEConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    fsq: FSQConfig = field(default_factory=FSQConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    max_seq_len: int = 8192
    n_tokens: int = 128
    cfg_drop_rate: float = 0.05
    size_loss_weight: float = 0.01


@dataclass
class DataConfig:
    mmcif_dir: str = "~/tim1/helico-data/raw/mmCIF"
    index_path: str = "data/index.parquet"
    cache_dir: str = "data/cache"
    max_atoms: int = 8000
    min_atoms: int = 10
    val_fraction: float = 0.05
    num_workers: int = 4


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 1000
    max_steps: int = 500_000
    batch_size: int = 2
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    val_every: int = 5000
    save_every: int = 10000
    log_every: int = 100
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "struct2token"
    wandb_run_name: str | None = None
    seed: int = 42


@dataclass
class InferenceConfig:
    n_steps: int = 100
    cfg_weight: float = 2.0
    noise_weight: float = 0.45


@dataclass
class Config:
    model: DAEConfig = field(default_factory=DAEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from a YAML file with recursive defaults."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()

        if "model" in raw:
            m = raw["model"]
            if "encoder" in m:
                cfg.model.encoder = EncoderConfig(**m["encoder"])
            if "fsq" in m:
                fsq_d = m["fsq"]
                if "levels" in fsq_d:
                    fsq_d["levels"] = tuple(fsq_d["levels"])
                cfg.model.fsq = FSQConfig(**fsq_d)
            if "decoder" in m:
                cfg.model.decoder = DecoderConfig(**m["decoder"])
            for k in ("max_seq_len", "n_tokens", "cfg_drop_rate", "size_loss_weight"):
                if k in m:
                    setattr(cfg.model, k, m[k])

        if "data" in raw:
            for k, v in raw["data"].items():
                setattr(cfg.data, k, v)

        if "training" in raw:
            t = raw["training"]
            for k, v in t.items():
                if k == "betas":
                    v = tuple(v)
                setattr(cfg.training, k, v)

        if "inference" in raw:
            for k, v in raw["inference"].items():
                setattr(cfg.inference, k, v)

        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert config to a plain dict (for wandb/serialization)."""
        return _dataclass_to_dict(self)


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass tree to plain dicts/lists/scalars."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj
