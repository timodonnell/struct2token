"""Diffusion Autoencoder (DAE) for all-atom macromolecular structure tokenization.

Integrates: AtomEmbedding → TransformerEncoder → FSQ → DiTDecoder with CFM.
Implements nested dropout for adaptive-length tokenization and classifier-free
guidance masking.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DAEConfig
from .attention import TransformerEncoder
from .cfm import ConditionalFlowMatching
from .dit import DiTDecoder
from .embeddings import AtomEmbedding
from .fsq import FSQ


class AllAtomDAE(nn.Module):
    """All-atom Diffusion Autoencoder.

    Architecture:
        Encoder: AtomEmbedding → TransformerEncoder → FSQ quantization
        Decoder: CFM noise schedule → DiTDecoder → predicted velocity

    Training features:
        - Nested dropout: randomly mask conditioning tokens for adaptive tokenization
        - CFG masking: 5% chance of zeroing all conditioning tokens
        - Size prediction: predict atom count from first conditioning token
    """

    def __init__(self, config: DAEConfig | None = None):
        super().__init__()
        if config is None:
            config = DAEConfig()
        self.config = config

        enc = config.encoder
        dec = config.decoder

        # Encoder
        self.atom_embed = AtomEmbedding(enc.d_model)
        self.encoder = TransformerEncoder(
            d_model=enc.d_model,
            n_layers=enc.n_layers,
            n_heads=enc.n_heads,
            mlp_ratio=enc.mlp_ratio,
            dropout=enc.dropout,
            max_seq_len=config.max_seq_len,
        )

        # Encoder output → FSQ input projection
        self.to_fsq = nn.Linear(enc.d_model, len(config.fsq.levels))

        # FSQ quantizer
        self.fsq = FSQ(config.fsq.levels)

        # FSQ output → decoder dim projection
        self.to_decoder = nn.Linear(len(config.fsq.levels), dec.d_model)

        # Pooling: from variable-length atom sequence to fixed n_tokens
        # We use a learnable query set to cross-attend into the encoder output
        self.pool_queries = nn.Parameter(torch.randn(1, config.n_tokens, enc.d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            enc.d_model, enc.n_heads, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(enc.d_model)

        # Decoder (DiT)
        self.decoder = DiTDecoder(
            d_model=dec.d_model,
            n_layers=dec.n_layers,
            n_heads=dec.n_heads,
            mlp_ratio=dec.mlp_ratio,
            dropout=dec.dropout,
            max_seq_len=config.max_seq_len,
        )

        # CFM
        self.cfm = ConditionalFlowMatching()

        # Size predictor: predict number of atoms from first conditioning token
        self.size_pred = nn.Sequential(
            nn.Linear(dec.d_model, dec.d_model),
            nn.SiLU(),
            nn.Linear(dec.d_model, config.max_seq_len),
        )

        self.cfg_drop_rate = config.cfg_drop_rate
        self.n_tokens = config.n_tokens

    def encode(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        residue_types: torch.Tensor,
        meta_classes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all-atom structure to discrete tokens.

        Args:
            coords: (B, L, 3) centered coordinates in nm
            atom_types: (B, L) element type indices
            residue_types: (B, L) residue type indices
            meta_classes: (B, L) metastructure class indices
            padding_mask: (B, L) bool, True for real atoms

        Returns:
            codes: (B, K, d_decoder) quantized token representations
            indices: (B, K) FSQ codebook indices
        """
        # Atom embeddings
        h = self.atom_embed(coords, atom_types, residue_types, meta_classes)  # (B, L, d_enc)

        # Contextual encoding
        h = self.encoder(h, padding_mask)  # (B, L, d_enc)

        # Pool to fixed-length token sequence via cross-attention
        B = h.shape[0]
        queries = self.pool_queries.expand(B, -1, -1)  # (B, K, d_enc)

        # Key padding mask for cross attention (inverted: True = ignore)
        key_pad = ~padding_mask if padding_mask is not None else None
        pooled, _ = self.pool_attn(queries, h, h, key_padding_mask=key_pad)
        pooled = self.pool_norm(pooled + queries)  # (B, K, d_enc)

        # FSQ quantization
        fsq_input = self.to_fsq(pooled)  # (B, K, fsq_dim)
        quantized, indices = self.fsq(fsq_input)  # (B, K, fsq_dim), (B, K)

        # Project to decoder dimension
        codes = self.to_decoder(quantized)  # (B, K, d_dec)

        return codes, indices

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            batch: dict with coords, atom_types, residue_types, meta_classes,
                   padding_mask, known_mask, lengths

        Returns:
            dict with flow_loss, size_loss
        """
        coords = batch["coords"]        # (B, L, 3)
        padding_mask = batch["padding_mask"]  # (B, L)
        lengths = batch["lengths"]       # (B,)
        B, L, _ = coords.shape

        # 1. Encode
        codes, indices = self.encode(
            coords,
            batch["atom_types"],
            batch["residue_types"],
            batch["meta_classes"],
            padding_mask,
        )  # codes: (B, K, d_dec)

        K = codes.shape[1]

        # 2. Nested dropout: randomly mask conditioning tokens
        # Each batch element gets a random cutoff in [1, K]
        if self.training:
            cutoffs = torch.randint(1, K + 1, (B,), device=codes.device)
            token_mask = torch.arange(K, device=codes.device).unsqueeze(0) < cutoffs.unsqueeze(1)
            # (B, K) bool

            # 3. CFG masking: with cfg_drop_rate, zero all tokens
            cfg_mask = torch.rand(B, device=codes.device) > self.cfg_drop_rate
            # (B,) bool — True means keep tokens
            token_mask = token_mask & cfg_mask.unsqueeze(1)

            # Apply mask
            masked_codes = codes * token_mask.unsqueeze(-1).float()
            cond_mask = token_mask
        else:
            masked_codes = codes
            cond_mask = torch.ones(B, K, dtype=torch.bool, device=codes.device)

        # 4. Sample CFM noise and interpolation
        t, x_t, u_t, x_0 = self.cfm(coords)

        # 5. Predict velocity
        v_t = self.decoder(x_t, t, masked_codes, padding_mask, cond_mask)

        # 6. Flow matching loss (MSE on non-padded positions)
        diff = (u_t - v_t) ** 2  # (B, L, 3)
        # Mask out padded positions
        mask_3d = padding_mask.unsqueeze(-1).float()  # (B, L, 1)
        flow_loss = (diff * mask_3d).sum() / mask_3d.sum().clamp(min=1) / 3.0

        # 7. Size loss: predict atom count from first token
        size_logits = self.size_pred(codes[:, 0, :])  # (B, max_seq_len)
        # Clamp lengths to valid range
        target_lengths = lengths.clamp(0, self.config.max_seq_len - 1)
        size_loss = F.cross_entropy(size_logits, target_lengths)

        return {
            "flow_loss": flow_loss,
            "size_loss": size_loss,
        }

    @torch.no_grad()
    def decode(
        self,
        indices: torch.Tensor,
        n_atoms: int | torch.Tensor,
        n_steps: int = 100,
        cfg_weight: float = 2.0,
        noise_weight: float = 0.45,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode token indices to 3D coordinates via diffusion sampling.

        Args:
            indices: (B, K) FSQ codebook indices
            n_atoms: int or (B,) number of atoms to generate
            n_steps: number of Euler-Maruyama steps
            cfg_weight: classifier-free guidance weight
            noise_weight: stochastic noise injection weight
            cond_mask: (B, K) bool, which tokens to condition on.
                If None, all tokens are used. Set to False for truncated
                positions when evaluating adaptive tokenization.

        Returns:
            coords: (B, N, 3) predicted coordinates in nm
        """
        B = indices.shape[0]
        K = indices.shape[1]
        device = indices.device

        if isinstance(n_atoms, int):
            N = n_atoms
        else:
            N = n_atoms.max().item()

        # Reconstruct codes from indices
        fsq_codes = self.fsq.indices_to_codes(indices)  # (B, K, fsq_dim)
        codes = self.to_decoder(fsq_codes)  # (B, K, d_dec)

        # Apply conditioning mask (zero out truncated tokens)
        if cond_mask is not None:
            codes = codes * cond_mask.unsqueeze(-1).float()

        # Unconditional codes (for CFG)
        uncond_codes = torch.zeros_like(codes)
        uncond_mask = torch.ones(B, K, dtype=torch.bool, device=device)
        if cond_mask is None:
            cond_mask = torch.ones(B, K, dtype=torch.bool, device=device)

        # Padding mask for generated atoms
        if isinstance(n_atoms, torch.Tensor):
            padding_mask = torch.arange(N, device=device).unsqueeze(0) < n_atoms.unsqueeze(1)
        else:
            padding_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Initialize from centered noise
        x = torch.randn(B, N, 3, device=device)
        x = x - x.mean(dim=1, keepdim=True)

        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device)

            # Conditional velocity
            v_cond = self.decoder(x, t, codes, padding_mask, cond_mask)

            # Unconditional velocity (for CFG)
            v_uncond = self.decoder(x, t, uncond_codes, padding_mask, uncond_mask)

            # CFG: v = v_uncond + cfg_weight * (v_cond - v_uncond)
            v = v_uncond + cfg_weight * (v_cond - v_uncond)

            # Euler step
            x = x + v * dt

            # Stochastic correction (except last step)
            if step < n_steps - 1 and noise_weight > 0:
                noise = torch.randn_like(x)
                noise = noise - noise.mean(dim=1, keepdim=True)
                x = x + noise_weight * math.sqrt(dt) * noise

        return x
