# struct2token

Adaptive all-atom tokenization of proteins, nucleic acids, and small molecules.

Combines [APT](https://github.com/rdilip/apt) (adaptive protein tokenization via diffusion autoencoder + FSQ + nested dropout) with [Bio2Token](https://github.com/flagshippioneering/bio2token) (all-atom representation). The result is APT's architecture extended to every heavy atom in the structure — not just C-alpha.

**Architecture**: Transformer encoder → FSQ quantization (1000 tokens) → DiT diffusion decoder with conditional flow matching. Nested dropout during training creates a coarse-to-fine hierarchy so any prefix of tokens is a valid reconstruction.

**~79M parameters**. Trains on a single A100/H100 in float32.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
# Install the package and all dependencies
uv sync

# With Flash Attention 2 (recommended for GPU training)
uv sync --extra flash

# With dev dependencies (for running tests)
uv sync --extra dev
```

### WandB

Training logs to [Weights & Biases](https://wandb.ai) by default. Log in before training:

```bash
uv run wandb login
```

To disable wandb, pass `--no-wandb` to the training script.

## Quickstart

### 1. Build the data index

Scan your mmCIF files and create a parquet index:

```bash
uv run python scripts/preprocess_data.py \
    --mmcif_dir ~/tim1/helico-data/raw/mmCIF \
    --output data/index.parquet
```

For a quick test with a subset:

```bash
uv run python scripts/preprocess_data.py \
    --mmcif_dir ~/tim1/helico-data/raw/mmCIF \
    --output data/index.parquet \
    --max_files 100
```

### 2. Train the tokenizer

```bash
uv run python scripts/train_tokenizer.py --config configs/default.yaml
```

Common overrides:

```bash
# Name your wandb run
uv run python scripts/train_tokenizer.py --config configs/default.yaml \
    --wandb_run_name first-run

# Adjust batch size and learning rate
uv run python scripts/train_tokenizer.py --config configs/default.yaml \
    --batch_size 4 --lr 1e-4

# Resume from checkpoint
uv run python scripts/train_tokenizer.py --config configs/default.yaml \
    --resume checkpoints/step_10000.pt

# Train without wandb
uv run python scripts/train_tokenizer.py --config configs/default.yaml \
    --no-wandb

# Use a specific GPU
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_tokenizer.py --config configs/default.yaml
```

All CLI flags (`--batch_size`, `--lr`, `--max_steps`, `--seed`, `--index_path`, `--max_atoms`, `--wandb_project`, `--wandb_run_name`) override the corresponding YAML config values.

### 3. Evaluate

```bash
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/final.pt \
    --output results.json
```

Options:

```bash
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/final.pt \
    --max_samples 500 \
    --n_steps 100 \
    --cfg_weight 2.0 \
    --output results.json
```

### 4. Run tests

```bash
uv run pytest
```

## Project structure

```
struct2token/
├── configs/
│   └── default.yaml              # master config
├── src/struct2token/
│   ├── config.py                 # dataclass configs + YAML loading
│   ├── data/
│   │   ├── tokens.py             # atom-type, residue-type vocabularies
│   │   ├── molecule_conventions.py  # per-residue canonical atom ordering
│   │   ├── mmcif_parser.py       # mmCIF → all-atom features
│   │   ├── dataset.py            # PyTorch Dataset with caching
│   │   └── collate.py            # variable-length batching
│   ├── model/
│   │   ├── embeddings.py         # coord + atom + residue + meta embeddings
│   │   ├── attention.py          # Flash Attention 2 transformer (SDPA fallback)
│   │   ├── rotary.py             # RoPE positional embeddings
│   │   ├── fsq.py                # Finite Scalar Quantization (8,5,5,5 → 1000 codes)
│   │   ├── cfm.py                # Conditional Flow Matching
│   │   ├── dit.py                # DiT decoder with adaLN
│   │   └── dae.py                # main Diffusion Autoencoder
│   ├── losses/
│   │   ├── rmsd.py               # Kabsch-aligned RMSD
│   │   ├── inter_atom_distance.py
│   │   ├── permutation.py        # symmetric sidechain resolution
│   │   └── tm.py                 # TM-score
│   ├── training/
│   │   ├── trainer.py            # training loop + wandb
│   │   ├── ema.py                # exponential moving average
│   │   └── augmentation.py       # random SO(3) rotation
│   └── inference/
│       ├── encode.py
│       ├── decode.py
│       └── metrics.py
├── scripts/
│   ├── preprocess_data.py        # build data index
│   ├── train_tokenizer.py        # training entry point
│   └── evaluate.py               # evaluation entry point
└── tests/
```

## Config

All parameters live in `configs/default.yaml`. Key settings:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model.encoder.d_model` | 256 | Encoder hidden dim |
| `model.decoder.d_model` | 512 | Decoder hidden dim |
| `model.decoder.n_layers` | 12 | DiT depth |
| `model.fsq.levels` | [8,5,5,5] | 1000-token codebook |
| `model.n_tokens` | 128 | Max adaptive tokens |
| `model.max_seq_len` | 8192 | Max atoms per structure |
| `training.lr` | 3e-4 | AdamW learning rate |
| `training.batch_size` | 2 | Per-GPU batch size |
| `training.max_steps` | 500000 | Total training steps |
| `training.wandb_project` | struct2token | WandB project name |

## WandB metrics

During training the following are logged:

- `train/flow_loss` — flow matching MSE (main training signal)
- `train/size_loss` — atom count prediction CE
- `train/total_loss` — weighted sum
- `train/grad_norm` — gradient norm before clipping
- `train/lr` — current learning rate
- `val/flow_loss`, `val/size_loss` — validation losses (EMA model)

## Data

Training data: PDB mmCIF files (gzipped or plain). The preprocessing script scans all files and writes a parquet index with path, chain ID, entity type, and atom count per chain. The dataset lazily parses mmCIF files on access and caches parsed tensors as `.pt` files.

## References

- `references/apt.pdf` — [Adaptive Protein Tokenization](https://github.com/rdilip/apt)
- `references/bio2token.pdf` — [Bio2Token: All-atom tokenization](https://github.com/flagshippioneering/bio2token)
