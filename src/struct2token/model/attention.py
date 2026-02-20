"""Transformer encoder with Flash Attention 2 and RoPE."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .rotary import RotaryEmbedding, apply_rotary_emb

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class MLP(nn.Module):
    """SwiGLU-style MLP."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def _flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    padding_mask: torch.Tensor | None,
    causal: bool = False,
) -> torch.Tensor:
    """Flash Attention 2 with variable-length support.

    q, k, v: (B, L, H, D)
    padding_mask: (B, L) bool, True for real tokens
    Returns: (B, L, H, D)
    """
    B, L, H, D = q.shape

    if padding_mask is not None:
        # Unpad sequences for Flash Attention
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
            rearrange(q, "b l h d -> b l (h d)"), padding_mask
        )
        k_unpad, _, cu_seqlens_k, max_seqlen_k = unpad_input(
            rearrange(k, "b l h d -> b l (h d)"), padding_mask
        )
        v_unpad, _, _, _ = unpad_input(
            rearrange(v, "b l h d -> b l (h d)"), padding_mask
        )

        q_unpad = rearrange(q_unpad, "t (h d) -> t h d", h=H)
        k_unpad = rearrange(k_unpad, "t (h d) -> t h d", h=H)
        v_unpad = rearrange(v_unpad, "t (h d) -> t h d", h=H)

        out_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            causal=causal,
        )

        out = pad_input(
            rearrange(out_unpad, "t h d -> t (h d)"),
            indices_q, B, L,
        )
        return rearrange(out, "b l (h d) -> b l h d", h=H)
    else:
        q_unpad = rearrange(q, "b l h d -> (b l) h d")
        k_unpad = rearrange(k, "b l h d -> (b l) h d")
        v_unpad = rearrange(v, "b l h d -> (b l) h d")
        seqlens = torch.arange(0, (B + 1) * L, L, device=q.device, dtype=torch.int32)

        out_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            seqlens, seqlens, L, L,
            causal=causal,
        )
        return rearrange(out_unpad, "(b l) h d -> b l h d", b=B)


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    padding_mask: torch.Tensor | None,
    causal: bool = False,
) -> torch.Tensor:
    """Fallback: PyTorch SDPA attention.

    q, k, v: (B, L, H, D) → rearrange to (B, H, L, D) for SDPA.
    """
    q = rearrange(q, "b l h d -> b h l d")
    k = rearrange(k, "b l h d -> b h l d")
    v = rearrange(v, "b l h d -> b h l d")

    attn_mask = None
    if padding_mask is not None:
        # (B, L) → (B, 1, 1, L) for broadcasting
        attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        attn_mask = attn_mask.expand(-1, -1, q.shape[2], -1)
        attn_mask = attn_mask.to(dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~attn_mask.bool(), float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask.bool(), 0.0)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
    return rearrange(out, "b h l d -> b l h d")


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → Attention → RMSNorm → MLP."""

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

        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention
        h = self.norm1(x)
        B, L, D = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, D_head)

        # Apply RoPE to Q and K
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
        x = x + self.out_proj(attn_out)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of TransformerBlocks with RoPE.

    Input: (B, L, d_model) from atom embeddings
    Output: (B, L, d_model) contextualized atom representations
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        L = x.shape[1]
        cos, sin = self.rope(L)
        cos = cos.to(x.device)
        sin = sin.to(x.device)

        for layer in self.layers:
            x = layer(x, cos, sin, padding_mask)

        return self.norm(x)
