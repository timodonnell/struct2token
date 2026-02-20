"""DiT (Diffusion Transformer) decoder blocks with adaptive LayerNorm.

Following APT architecture: adaLN modulation from timestep conditioning,
Flash Attention, and SwiGLU MLP.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import HAS_FLASH_ATTN, MLP, RMSNorm, _flash_attention, _sdpa_attention
from .rotary import RotaryEmbedding, apply_rotary_emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embed scalar timestep via sinusoidal encoding + MLP."""

    def __init__(self, d_model: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.freq_dim = freq_dim

    def _sinusoidal_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding of scalar timesteps.

        t: (B,) float in [0, 1]
        Returns: (B, freq_dim)
        """
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None] * freqs[None, :]  # (B, half)
        return torch.cat([args.cos(), args.sin()], dim=-1)  # (B, freq_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, d_model)"""
        return self.mlp(self._sinusoidal_embed(t))


class DiTBlock(nn.Module):
    """Adaptive LayerNorm (adaLN) transformer block for diffusion.

    Conditioning signal c modulates the block via learned shift/scale/gate
    parameters (6 modulation params per block).

    Architecture:
        adaLN_modulation(c) → (shift1, scale1, gate1, shift2, scale2, gate2)
        h = modulate(norm1(x), shift1, scale1)
        h = flash_self_attention(h)
        x = x + gate1 * h
        h = modulate(norm2(x), shift2, scale2)
        h = mlp(h)
        x = x + gate2 * h
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

        # adaLN modulation: 6 parameters (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, L, D) input sequence
        c: (B, D) conditioning vector (timestep embedding)
        cos, sin: (L, head_dim//2) rotary embeddings
        padding_mask: (B, L) bool
        """
        # Get modulation parameters from conditioning
        mod = self.adaLN_modulation(c)  # (B, 6*D)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
        # Expand to (B, 1, D) for broadcasting
        shift1 = shift1.unsqueeze(1)
        scale1 = scale1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        # Attention branch
        h = modulate(self.norm1(x), shift1, scale1)
        B, L, D = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Apply RoPE
        q = rearrange(q, "b l h d -> (b h) l d")
        k = rearrange(k, "b l h d -> (b h) l d")
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rearrange(q, "(b h) l d -> b l h d", b=B)
        k = rearrange(k, "(b h) l d -> b l h d", b=B)

        if HAS_FLASH_ATTN:
            attn_out = _flash_attention(q, k, v, padding_mask)
        else:
            attn_out = _sdpa_attention(q, k, v, padding_mask)

        attn_out = rearrange(attn_out, "b l h d -> b l (h d)")
        x = x + gate1 * self.out_proj(attn_out)

        # MLP branch
        h = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2 * self.mlp(h)

        return x


class DiTDecoder(nn.Module):
    """Full diffusion transformer decoder.

    Forward(x_BLD, t_B, z_BLD):
        1. Embed timestep t → c via sinusoidal + MLP
        2. Project noisy coords x → s via linear + adaLN
        3. Concatenate s with conditioning tokens z along sequence dim
        4. Add cond_embed distinguishing input vs conditioning positions
        5. Process through DiTBlocks with Flash Attention
        6. Slice back to original length
        7. Project to (B, L, 3) via final linear + adaLN
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        head_dim = d_model // n_heads

        # Timestep embedding
        self.time_embed = TimestepEmbedder(d_model)

        # Input projection (noisy coords → d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Conditioning type embedding (0 = noisy input, 1 = conditioning tokens)
        self.cond_type_embed = nn.Embedding(2, d_model)

        # RoPE
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len * 2)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        # Final output
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),  # shift, scale
        )
        self.output_proj = nn.Linear(d_model, 3)

        self._init_weights()

    def _init_weights(self):
        # Zero-initialize output projection for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        # Zero-initialize adaLN modulations
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, 3) noisy coordinates
            t: (B,) timesteps
            z: (B, K, d_model) conditioning tokens from encoder
            padding_mask: (B, L) bool for input positions
            cond_mask: (B, K) bool for conditioning positions

        Returns:
            v: (B, L, 3) predicted velocity field
        """
        B, L, _ = x.shape
        K = z.shape[1]

        # Timestep conditioning
        c = self.time_embed(t)  # (B, d_model)

        # Project noisy coords
        s = self.input_proj(x)  # (B, L, d_model)

        # Concatenate input and conditioning along sequence dimension
        combined = torch.cat([s, z], dim=1)  # (B, L+K, d_model)

        # Add conditioning type embedding
        cond_type = torch.zeros(B, L + K, dtype=torch.long, device=x.device)
        cond_type[:, L:] = 1  # conditioning positions
        combined = combined + self.cond_type_embed(cond_type)

        # Combined padding mask
        if padding_mask is not None and cond_mask is not None:
            combined_mask = torch.cat([padding_mask, cond_mask], dim=1)
        elif padding_mask is not None:
            ones = torch.ones(B, K, dtype=torch.bool, device=x.device)
            combined_mask = torch.cat([padding_mask, ones], dim=1)
        else:
            combined_mask = None

        # RoPE
        total_len = L + K
        cos, sin = self.rope(total_len)
        cos = cos.to(x.device)
        sin = sin.to(x.device)

        # Process through DiT blocks
        h = combined
        for block in self.blocks:
            h = block(h, c, cos, sin, combined_mask)

        # Slice back to input positions only
        h = h[:, :L, :]

        # Final projection with adaLN
        mod = self.final_adaLN(c)  # (B, 2*d_model)
        shift, scale = mod.chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift.unsqueeze(1), scale.unsqueeze(1))
        v = self.output_proj(h)  # (B, L, 3)

        return v
