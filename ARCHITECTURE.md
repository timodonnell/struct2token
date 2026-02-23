# struct2token: Architecture and Training

## Overview

struct2token is a diffusion autoencoder that tokenizes macromolecular structures (proteins, RNA, small molecules) into discrete, variable-length token sequences. It extends the Adaptive Protein Tokenization (APT) architecture from C-alpha-only representation to all heavy atoms, incorporating the all-atom vocabulary and conventions from Bio2Token.

The model has three stages:

1. **Encoder**: embed all atoms, contextualize via transformer, pool to fixed-length tokens, quantize with FSQ
2. **Decoder**: reconstruct 3D coordinates from tokens via a DiT diffusion model with conditional flow matching
3. **Adaptive mechanism**: nested dropout during training so any prefix of tokens gives a valid reconstruction

Total parameters: ~79M. Trains on a single A100/H100 in float32.

---

## 1. Input Representation

### 1.1 Per-atom features

Unlike APT, which operates on a single C-alpha atom per residue, struct2token encodes every heavy atom. Each atom is described by four discrete features:

**Atom type vocabulary** (20 indices): 6 special tokens + 14 elements.

```python
ELEMENTS = ["H", "C", "N", "O", "F", "B", "Al", "Si", "P", "S", "Cl", "As", "Br", "I"]
ATOM_TYPE_VOCAB = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
for i, elem in enumerate(ELEMENTS):
    ATOM_TYPE_VOCAB[elem] = NUM_SPECIAL + i   # 20 total
```

**Residue type vocabulary** (33 indices): 6 special + 22 amino acids (including selenocysteine and unknown) + 5 RNA bases.

**Metastructure class** (4 indices): classifies each atom's structural role.

```python
META_BACKBONE  = 0   # backbone atom (N, C, O for protein; P, sugars for RNA)
META_CREF      = 1   # reference atom (CA for protein, C3' for RNA)
META_SIDECHAIN = 2   # sidechain / base atom
META_PAD       = 3   # padding / missing atom
```

**Coordinates**: (x, y, z) in nanometers, centered per structure.

Atoms are ordered within each residue following Bio2Token's canonical ordering. For protein: backbone first (N, CA, C, O), then sidechain in a fixed order per amino acid type. For RNA: 12 backbone atoms, then base atoms.

### 1.2 Atom embedding

The four features are fused into a single vector by summing four independent embeddings:

```python
class AtomEmbedding(nn.Module):
    def __init__(self, d_model: int):
        self.coord_proj = nn.Sequential(nn.Linear(3, d_model), nn.SiLU())
        self.atom_type_embed = nn.Embedding(20, d_model, padding_idx=0)
        self.residue_type_embed = nn.Embedding(33, d_model, padding_idx=0)
        self.meta_class_embed = nn.Embedding(4, d_model)

    def forward(self, coords, atom_types, residue_types, meta_classes):
        return (self.coord_proj(coords)
                + self.atom_type_embed(atom_types)
                + self.residue_type_embed(residue_types)
                + self.meta_class_embed(meta_classes))
```

This produces a `(B, L, 256)` tensor where L is the number of atoms (up to 8000).

---

## 2. Encoder

### 2.1 Transformer encoder

The atom embeddings are contextualized by a bidirectional transformer: 4 layers, 256 dim, 8 heads. Each block follows a pre-norm architecture:

```
RMSNorm → Self-Attention (with RoPE on Q,K) → Residual
RMSNorm → SwiGLU MLP → Residual
```

The MLP uses the SwiGLU gating pattern:

```python
class MLP(nn.Module):
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))   # SwiGLU
```

Attention uses Flash Attention 2 with `flash_attn_varlen_func` when available, falling back to PyTorch SDPA. Variable-length sequences are handled efficiently by unpadding before the kernel and repadding after:

```python
q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
    rearrange(q, "b l h d -> b l (h d)"), padding_mask)
# ... same for k, v ...
out_unpad = flash_attn_varlen_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
out = pad_input(rearrange(out_unpad, "t h d -> t (h d)"), indices_q, B, L)
```

Both encoder and decoder use **Rotary Position Embeddings** (RoPE), not learned absolute embeddings. The encoder output is `(B, L, 256)`.

### 2.2 Pooling to fixed-length tokens

The variable-length atom sequence (up to 8000 atoms) must be compressed to a fixed-length token sequence (K=128 tokens). This is done by cross-attention from K learnable query vectors into the encoder output:

```python
self.pool_queries = nn.Parameter(torch.randn(1, K, d_enc) * 0.02)
self.pool_attn = nn.MultiheadAttention(d_enc, n_heads, batch_first=True)

# In encode():
queries = self.pool_queries.expand(B, -1, -1)           # (B, 128, 256)
pooled, _ = self.pool_attn(queries, h, h,
                           key_padding_mask=~padding_mask)
pooled = self.pool_norm(pooled + queries)                # residual connection
```

This produces `(B, 128, 256)` — 128 tokens, each 256-dimensional.

### 2.3 Finite Scalar Quantization (FSQ)

The continuous 256-dim tokens are projected to 4 dimensions, then quantized using FSQ with levels `(8, 5, 5, 5)`, yielding a codebook of `8 * 5 * 5 * 5 = 1000` discrete tokens.

FSQ works by applying `tanh` to bound each dimension, scaling to the level range, and rounding with a straight-through estimator for gradient flow:

```python
def _scale(self, x):
    half_levels = (self._levels - 1) / 2
    return torch.tanh(x) * half_levels + half_levels

def _quantize(self, x_scaled):
    x_rounded = x_scaled.round()
    for d in range(self.dim):
        x_rounded[..., d] = x_rounded[..., d].clamp(0, self.levels[d] - 1)
    return x_scaled + (x_rounded - x_scaled).detach()   # straight-through
```

The quantized 4-dim codes are projected back up to the decoder dimension (512) for conditioning.

The full encode path is:

```
atoms (B, L, 3+features)
  → AtomEmbedding → (B, L, 256)
  → TransformerEncoder (4 layers) → (B, L, 256)
  → CrossAttention pooling → (B, 128, 256)
  → Linear → (B, 128, 4)
  → FSQ quantize → (B, 128, 4) codes + (B, 128) integer indices
  → Linear → (B, 128, 512) decoder conditioning
```

---

## 3. Decoder

### 3.1 Conditional Flow Matching (CFM)

The decoder generates coordinates through a flow matching formulation rather than discrete diffusion. The flow defines a linear interpolation between noise and data:

```
x_t = t * x_1 + (1 - t) * x_0       where x_0 ~ N(0, I), x_1 = target coords
u_t = x_1 - x_0                      target velocity field
```

The model learns to predict the velocity field `v_t ≈ u_t` given `x_t` and `t`. Training loss is simply MSE between predicted and target velocity.

**Time sampling** uses a Beta(1.9, 1.0) distribution mixed with 2% uniform, biasing toward later timesteps (closer to data):

```python
def sample_time(self, batch_size, device):
    uniform_mask = torch.rand(batch_size, device=device) < 0.02
    t = torch.distributions.Beta(1.9, 1.0).sample((batch_size,)).to(device)
    uniform_t = torch.rand(batch_size, device=device)
    t = torch.where(uniform_mask, uniform_t, t)
    return t.clamp(1e-5, 1.0 - 1e-5)
```

**Noise is centered** (zero mean per sample) to maintain translation invariance:

```python
x_0 = torch.randn(shape, device=device)
x_0 = x_0 - x_0.mean(dim=-2, keepdim=True)
```

### 3.2 DiT decoder architecture

The decoder is a Diffusion Transformer (DiT) with adaptive layer normalization (adaLN). Configuration: 512 dim, 12 layers, 8 heads.

Each DiT block produces 6 modulation parameters from the timestep conditioning signal, which shift, scale, and gate the two sub-layers:

```python
class DiTBlock(nn.Module):
    def forward(self, x, c, cos, sin, padding_mask):
        # c: (B, 512) timestep embedding
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Modulated attention
        h = x * (1 + scale1) + shift1          # adaLN on norm1(x)
        h = self_attention(h)                    # Flash Attention + RoPE
        x = x + gate1 * h                       # gated residual

        # Modulated MLP
        h = x * (1 + scale2) + shift2          # adaLN on norm2(x)
        h = swiglu_mlp(h)
        x = x + gate2 * h
        return x
```

The decoder forward pass:

1. Embed timestep `t` via sinusoidal encoding + MLP → `c` of shape `(B, 512)`
2. Project noisy coordinates `x_t` from `(B, L, 3)` to `(B, L, 512)` via 2-layer MLP
3. Concatenate with conditioning tokens `z` from the encoder along the sequence dimension: `(B, L+K, 512)`
4. Add a learned `cond_type_embed` to distinguish input positions (type 0) from conditioning tokens (type 1)
5. Process through 12 DiT blocks with RoPE and Flash Attention
6. Slice back to L input positions, apply final adaLN + linear → `(B, L, 3)` predicted velocity

```python
def forward(self, x, t, z, padding_mask, cond_mask):
    c = self.time_embed(t)                           # (B, 512)
    s = self.input_proj(x)                           # (B, L, 512)
    combined = torch.cat([s, z], dim=1)              # (B, L+K, 512)
    combined = combined + self.cond_type_embed(...)   # distinguish input vs cond

    for block in self.blocks:
        combined = block(combined, c, cos, sin, mask)

    h = combined[:, :L, :]                           # slice back to input
    v = self.output_proj(adaLN(self.final_norm(h)))  # (B, L, 3)
    return v
```

**Initialization**: all adaLN modulation layers and the final output projection are zero-initialized, so the model starts by predicting zero velocity (identity function), ensuring stable early training.

### 3.3 Inference: Euler-Maruyama sampling with CFG

At inference, coordinates are generated from noise via 100 Euler steps with classifier-free guidance:

```python
x = centered_noise(B, N, 3)
dt = 1.0 / n_steps

for step in range(n_steps):
    t = step * dt
    v_cond   = decoder(x, t, codes, ...)       # conditional
    v_uncond = decoder(x, t, zeros, ...)        # unconditional
    v = v_uncond + cfg_weight * (v_cond - v_uncond)   # CFG, weight=2.0
    x = x + v * dt

    if step < n_steps - 1:                     # stochastic correction
        noise = centered_randn_like(x)
        x = x + 0.45 * sqrt(dt) * noise
```

The `cfg_weight=2.0` and `noise_weight=0.45` are taken directly from APT.

---

## 4. Adaptive Tokenization via Nested Dropout

The key mechanism that makes tokenization *adaptive* (variable-length) is **nested dropout**. During training, for each sample in the batch, a random cutoff `c ~ Uniform{1, ..., K}` is drawn, and all conditioning tokens beyond position `c` are zeroed out:

```python
# In AllAtomDAE.forward():
cutoffs = torch.randint(1, K + 1, (B,), device=codes.device)
token_mask = torch.arange(K, device=codes.device).unsqueeze(0) < cutoffs.unsqueeze(1)

# CFG masking: 5% of the time, drop ALL tokens (unconditional training)
cfg_mask = torch.rand(B, device=codes.device) > 0.05
token_mask = token_mask & cfg_mask.unsqueeze(1)

masked_codes = codes * token_mask.unsqueeze(-1).float()
```

This creates a strict ordering: token 1 is always present, token 2 is present when cutoff >= 2, etc. The model learns to encode global structure in early tokens and fine detail in later ones. At inference, using the first N tokens (for any N from 1 to 128) produces a valid reconstruction at the corresponding resolution.

---

## 5. Loss Functions

### 5.1 Training losses

**Flow matching loss** (primary): MSE between predicted and target velocity, masked to non-padded positions.

```python
diff = (u_t - v_t) ** 2                                # (B, L, 3)
mask_3d = padding_mask.unsqueeze(-1).float()
flow_loss = (diff * mask_3d).sum() / mask_3d.sum().clamp(min=1) / 3.0
```

**Size loss** (auxiliary): cross-entropy predicting the atom count from the first conditioning token. Weighted at 0.01x the flow loss.

```python
size_logits = self.size_pred(codes[:, 0, :])   # (B, max_seq_len)
size_loss = F.cross_entropy(size_logits, lengths.clamp(0, max_seq_len - 1))
```

**Total loss**: `flow_loss + 0.01 * size_loss`

### 5.2 Evaluation metrics

**All-atom RMSD**: Kabsch-aligned (SVD-based optimal rigid-body superposition) RMSD over all heavy atoms. Computed separately for backbone atoms and sidechain atoms.

**Permutation-symmetric RMSD**: before computing RMSD, symmetric sidechain atoms are resolved by trying both assignments and keeping the one with lower local RMSD. Seven residue types have symmetric atoms:

```python
SYMMETRIC_ATOMS = {
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "ARG": [("NH1", "NH2")],
    "LEU": [("CD1", "CD2")],
    "VAL": [("CG1", "CG2")],
}
```

**TM-score**: computed on C-alpha atoms (protein) or C3' atoms (RNA). Standard formula: `TM = 1/L * sum(1 / (1 + (d_i/d_0)^2))` where `d_0 = 1.24 * (L-15)^(1/3) - 1.8`.

**Inter-atomic distance RMSE**: RMSE of intra-residue pairwise distances between prediction and ground truth. Rotation-invariant (no alignment needed).

---

## 6. Training Scheme

### 6.1 Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999)) |
| Schedule | Linear warmup (1000 steps) + cosine decay |
| Precision | float32 (required for flow matching numerical stability) |
| Batch size | 2 (at up to 8000 atoms per structure) |
| Gradient clipping | max_norm=1.0 |
| EMA decay | 0.999 |
| Max training steps | 500,000 |
| Validation | every 5,000 steps |
| Checkpointing | every 10,000 steps |

### 6.2 Data augmentation

Each training sample gets an independent random SO(3) rotation for stochastic equivariance:

```python
def random_rotation_matrix(batch_size, device):
    M = torch.randn(batch_size, 3, 3, device=device)
    Q, R_tri = torch.linalg.qr(M)
    signs = torch.diagonal(R_tri, dim1=-2, dim2=-1).sign()
    Q = Q * signs.unsqueeze(-2)
    dets = torch.det(Q)
    Q[:, :, -1] *= dets.sign().unsqueeze(-1)
    return Q

batch["coords"] = torch.bmm(coords, R.transpose(1, 2))
```

Coordinates are centered on the centroid of known atoms and converted from Angstroms to nanometers (÷10) during data loading.

### 6.3 Training loop

```python
for step in range(max_steps):
    batch = next(train_loader)
    batch = apply_random_rotation(batch)

    loss_dict = model(batch)       # flow_loss + size_loss
    total = flow_loss + 0.01 * size_loss

    total.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    ema.update()
```

### 6.4 Data pipeline

Training data: PDB mmCIF files (gzipped). A preprocessing script builds a parquet index of all chains with metadata (path, chain_id, entity_type, n_atoms). The dataset lazily parses mmCIF files using BioPython and caches parsed tensors as `.pt` files for fast subsequent access.

Collation pads variable-length atom sequences to the batch maximum, rounded up to the nearest multiple of 8 for Flash Attention alignment.

---

## 7. Detailed Comparison with APT

struct2token follows the APT architecture closely. The table below details every component, noting what is shared, what is modified, and what is new.

### 7.1 What is identical to APT

| Component | Details |
|-----------|---------|
| **FSQ quantizer** | Same levels (8,5,5,5) → 1000 codes, same straight-through estimator |
| **CFM formulation** | Same linear interpolation `x_t = t*x_1 + (1-t)*x_0`, same velocity target `u_t = x_1 - x_0` |
| **Time sampling** | Same Beta(1.9, 1.0) with 2% uniform mixture |
| **Centered noise** | Same zero-mean Gaussian noise for translation invariance |
| **DiT block design** | Same adaLN with 6 modulation params (shift, scale, gate for attention and MLP) |
| **Zero initialization** | Same zero-init of adaLN modulations and output projection |
| **Nested dropout** | Same mechanism: random prefix cutoff per sample, forcing coarse-to-fine ordering |
| **CFG masking** | Same 5% unconditional dropout rate |
| **CFG inference** | Same guidance formula `v = v_uncond + w * (v_cond - v_uncond)`, same weight=2.0 |
| **Stochastic sampling** | Same noise injection during Euler steps, same weight=0.45 |
| **Size prediction** | Same CE loss on atom count from first token |
| **Optimizer** | Same AdamW with same hyperparameters (lr=3e-4, betas=(0.9, 0.999)) |
| **EMA** | Same decay=0.999 |
| **Gradient clipping** | Same max_norm=1.0 |
| **Random rotation augmentation** | Same stochastic SO(3) rotation for equivariance |
| **Encoder dimensions** | Same 256 dim, 4 layers, 8 heads |
| **Decoder dimensions** | Same 512 dim, 12 layers, 8 heads |
| **Max tokens** | Same K=128 adaptive token budget |

### 7.2 What is different from APT

| Aspect | APT | struct2token | Rationale |
|--------|-----|-------------|-----------|
| **Input representation** | One point per residue (C-alpha xyz) | All heavy atoms per residue (N, CA, C, O, sidechain atoms, ...) | Captures full 3D chemistry, not just backbone trace |
| **Input dimension** | 3 (just coordinates) | 3 coords + atom type + residue type + metastructure class | Atoms within a residue need type information to be distinguishable |
| **Sequence length** | ~L residues (typically 100-500) | ~L atoms (typically 500-8000) | All-atom means ~8x more tokens per structure |
| **Encoder positional embedding** | Learned absolute position | RoPE (rotary) | Better length generalization for variable atom counts |
| **Decoder positional embedding** | RoPE | RoPE | Same |
| **Encoder normalization** | LayerNorm | RMSNorm | Slightly faster, no mean centering |
| **Decoder normalization** | LayerNorm (elementwise_affine=False) in adaLN | Same | Identical |
| **MLP type** | Standard Mlp from timm | SwiGLU (gated) | Better parameter efficiency |
| **Atom embedding** | Linear(3, d_model) | Sum of 4 embeddings (coord + atom_type + residue_type + meta_class) | Needed to distinguish atom types within each residue |
| **Pooling mechanism** | Direct encoding (one token per residue, already fixed-length per structure) | Cross-attention from K learned queries into variable-length atom sequence | APT has one encoder output per residue; struct2token needs to compress L atoms to K tokens |
| **Molecule types** | Protein only | Protein + RNA + small molecules | Universal structural tokenizer |
| **Vocabulary** | Residue-level (20 amino acids) | Atom-level (20 atom types, 33 residue types, 4 metastructure classes) | All-atom requires element-level typing |
| **Precision** | float32 | float32 | Same requirement for flow matching stability |
| **Evaluation metric** | C-alpha RMSD, TM-score | All-atom RMSD (with permutation symmetry), backbone RMSD, sidechain RMSD, TM-score, inter-atomic distance RMSE | All-atom reconstruction needs sidechain-aware metrics |
| **Permutation symmetry** | N/A (C-alpha only) | Resolve symmetric sidechains (PHE, TYR, ASP, GLU, ARG, LEU, VAL) before RMSD | Some sidechain atoms are chemically equivalent under permutation |
| **Training data** | Protein structures only | PDB mmCIF files (proteins + RNA + small molecules) | Multi-modal structural data |

### 7.3 Key architectural difference: the pooling bottleneck

The most significant architectural change from APT is the **cross-attention pooling layer**. In APT, the encoder operates at residue-level: each residue produces one token, and the sequence of residue tokens is directly quantized by FSQ. There is no pooling step because the encoder output length equals the number of residues, which is already a manageable sequence length.

In struct2token, the encoder operates at atom-level, producing one vector per atom. A protein of 300 residues has ~2400 heavy atoms. These must be compressed to K=128 tokens. The cross-attention pooling achieves this:

```python
# K=128 learnable query vectors attend into L atom representations
queries = self.pool_queries.expand(B, -1, -1)     # (B, 128, 256)
pooled, _ = self.pool_attn(queries, h, h, ...)    # cross-attend into (B, L, 256)
pooled = self.pool_norm(pooled + queries)          # (B, 128, 256) with residual
```

This is analogous to a Perceiver-style bottleneck. The learnable queries determine what information to extract from the full atom representation, and the nested dropout ensures that earlier queries capture globally important features.

### 7.4 Key difference: all-atom input embedding

APT's input is simply `Linear(3, d_model)` — coordinates directly projected. struct2token must distinguish between different atom types at the same position, so it uses four summed embeddings. The metastructure class embedding is particularly important: it tells the model whether an atom is backbone (structurally rigid), a reference atom (C-alpha/C3'), or sidechain (flexible).

### 7.5 Expected performance implications

| Aspect | Impact |
|--------|--------|
| **Reconstruction accuracy** | All-atom RMSD is harder than C-alpha RMSD. APT achieves ~1.3A on C-alpha. Target: sub-2A all-atom. |
| **Token efficiency** | Same 128-token budget must encode ~8x more spatial information. May need more tokens for equivalent accuracy on large structures. |
| **Compute cost** | Quadratic attention over atoms (not residues) means ~64x more attention FLOPs for the encoder. Mitigated by Flash Attention 2. |
| **Generality** | Handles RNA and small molecules natively, not just proteins. |
| **Sidechain modeling** | Directly models sidechain conformations, which APT cannot represent. |
