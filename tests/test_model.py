"""Tests for model components: FSQ round-trip, CFM, attention, DAE forward."""

import math

import pytest
import torch

from struct2token.config import DAEConfig
from struct2token.model.cfm import ConditionalFlowMatching
from struct2token.model.dae import AllAtomDAE
from struct2token.model.dit import DiTDecoder, TimestepEmbedder
from struct2token.model.embeddings import AtomEmbedding
from struct2token.model.fsq import FSQ
from struct2token.model.attention import TransformerEncoder, RMSNorm
from struct2token.model.rotary import RotaryEmbedding, apply_rotary_emb


class TestFSQ:
    def test_roundtrip(self):
        """Encode → indices → decode should recover the quantized codes."""
        fsq = FSQ(levels=(8, 5, 5, 5))
        x = torch.randn(4, 10, 4)  # (B, L, dim)
        quantized, indices = fsq(x)
        recovered = fsq.indices_to_codes(indices)
        # The recovered codes should match the rounded quantized values
        torch.testing.assert_close(
            quantized.detach().round(),
            recovered,
            atol=0.0, rtol=0.0,
        )

    def test_codebook_size(self):
        fsq = FSQ(levels=(8, 5, 5, 5))
        assert fsq.codebook_size == 1000
        assert fsq.num_codes == 1000

    def test_indices_range(self):
        fsq = FSQ(levels=(8, 5, 5, 5))
        x = torch.randn(2, 5, 4)
        _, indices = fsq(x)
        assert indices.min() >= 0
        assert indices.max() < 1000

    def test_gradient_flows(self):
        """FSQ should pass gradients via straight-through estimator."""
        fsq = FSQ(levels=(8, 5, 5, 5))
        x = torch.randn(2, 3, 4, requires_grad=True)
        quantized, _ = fsq(x)
        loss = quantized.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestCFM:
    def test_interpolation_endpoints(self):
        """At t=0, x_t should be x_0; at t=1, x_t should be x_1."""
        cfm = ConditionalFlowMatching()
        x_0 = torch.randn(2, 10, 3)
        x_1 = torch.randn(2, 10, 3)

        t = torch.zeros(2)
        x_t, u_t = cfm.interpolate(x_0, x_1, t)
        torch.testing.assert_close(x_t, x_0, atol=1e-6, rtol=1e-6)

        t = torch.ones(2)
        x_t, u_t = cfm.interpolate(x_0, x_1, t)
        torch.testing.assert_close(x_t, x_1, atol=1e-6, rtol=1e-6)

    def test_velocity(self):
        """Target velocity should be x_1 - x_0."""
        cfm = ConditionalFlowMatching()
        x_0 = torch.randn(2, 10, 3)
        x_1 = torch.randn(2, 10, 3)
        t = torch.rand(2)
        _, u_t = cfm.interpolate(x_0, x_1, t)
        torch.testing.assert_close(u_t, x_1 - x_0, atol=1e-6, rtol=1e-6)

    def test_noise_centered(self):
        """Sampled noise should be approximately zero-mean per sample."""
        cfm = ConditionalFlowMatching()
        noise = cfm.sample_noise((8, 100, 3), torch.device("cpu"))
        means = noise.mean(dim=1)  # (8, 3)
        assert means.abs().max() < 1e-5

    def test_time_range(self):
        cfm = ConditionalFlowMatching()
        t = cfm.sample_time(1000, torch.device("cpu"))
        assert t.min() >= 1e-5
        assert t.max() <= 1.0 - 1e-5


class TestRoPE:
    def test_output_shape(self):
        rope = RotaryEmbedding(32, max_seq_len=128)
        cos, sin = rope(64)
        assert cos.shape == (64, 16)
        assert sin.shape == (64, 16)

    def test_apply(self):
        rope = RotaryEmbedding(32, max_seq_len=128)
        cos, sin = rope(10)
        x = torch.randn(2, 10, 32)
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape


class TestAttention:
    def test_encoder_forward(self):
        enc = TransformerEncoder(d_model=64, n_layers=2, n_heads=4, max_seq_len=128)
        x = torch.randn(2, 16, 64)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[1, 12:] = False
        out = enc(x, mask)
        assert out.shape == (2, 16, 64)

    def test_rmsnorm(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape


class TestEmbeddings:
    def test_forward(self):
        embed = AtomEmbedding(64)
        coords = torch.randn(2, 10, 3)
        atom_types = torch.randint(0, 20, (2, 10))
        res_types = torch.randint(0, 33, (2, 10))
        meta = torch.randint(0, 4, (2, 10))
        out = embed(coords, atom_types, res_types, meta)
        assert out.shape == (2, 10, 64)


class TestDiT:
    def test_timestep_embedder(self):
        emb = TimestepEmbedder(128)
        t = torch.rand(4)
        out = emb(t)
        assert out.shape == (4, 128)

    def test_decoder_forward(self):
        dec = DiTDecoder(d_model=64, n_layers=2, n_heads=4, max_seq_len=128)
        x = torch.randn(2, 16, 3)  # noisy coords
        t = torch.rand(2)
        z = torch.randn(2, 8, 64)  # conditioning
        padding_mask = torch.ones(2, 16, dtype=torch.bool)
        cond_mask = torch.ones(2, 8, dtype=torch.bool)
        v = dec(x, t, z, padding_mask, cond_mask)
        assert v.shape == (2, 16, 3)


class TestDAE:
    @pytest.fixture
    def small_config(self):
        cfg = DAEConfig()
        cfg.encoder.d_model = 32
        cfg.encoder.n_layers = 1
        cfg.encoder.n_heads = 4
        cfg.decoder.d_model = 64
        cfg.decoder.n_layers = 1
        cfg.decoder.n_heads = 4
        cfg.max_seq_len = 128
        cfg.n_tokens = 8
        return cfg

    def test_encode(self, small_config):
        model = AllAtomDAE(small_config)
        model.eval()
        B, L = 2, 20
        codes, indices = model.encode(
            torch.randn(B, L, 3),
            torch.randint(0, 20, (B, L)),
            torch.randint(0, 33, (B, L)),
            torch.randint(0, 4, (B, L)),
            torch.ones(B, L, dtype=torch.bool),
        )
        assert codes.shape == (B, 8, 64)
        assert indices.shape == (B, 8)

    def test_forward_training(self, small_config):
        model = AllAtomDAE(small_config)
        model.train()
        B, L = 2, 20
        batch = {
            "coords": torch.randn(B, L, 3),
            "atom_types": torch.randint(0, 20, (B, L)),
            "residue_types": torch.randint(0, 33, (B, L)),
            "meta_classes": torch.randint(0, 4, (B, L)),
            "padding_mask": torch.ones(B, L, dtype=torch.bool),
            "known_mask": torch.ones(B, L, dtype=torch.bool),
            "lengths": torch.tensor([L, L]),
        }
        losses = model(batch)
        assert "flow_loss" in losses
        assert "size_loss" in losses
        assert losses["flow_loss"].requires_grad
        assert losses["size_loss"].requires_grad

    def test_decode(self, small_config):
        model = AllAtomDAE(small_config)
        model.eval()
        indices = torch.randint(0, 1000, (2, 8))
        coords = model.decode(indices, n_atoms=20, n_steps=5, cfg_weight=1.0)
        assert coords.shape == (2, 20, 3)
