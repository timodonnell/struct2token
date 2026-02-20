"""Conditional Flow Matching (CFM) for diffusion-based coordinate generation.

Implements linear interpolation paths and Beta-distributed time sampling
following the APT approach.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalFlowMatching(nn.Module):
    """Conditional Flow Matching with linear interpolation paths.

    The flow is defined as:
        x_t = t * x_1 + (1 - t) * x_0
    where x_0 ~ N(0, I) (noise), x_1 is the target, and t in [0, 1].

    Target velocity field:
        u_t = x_1 - x_0

    Time sampling: Beta(alpha, beta) mixture with uniform component.
    """

    def __init__(
        self,
        beta_alpha: float = 1.9,
        beta_beta: float = 1.0,
        uniform_fraction: float = 0.02,
    ):
        super().__init__()
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.uniform_fraction = uniform_fraction

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from Beta distribution mixed with uniform.

        Returns:
            t: (B,) timesteps in (0, 1).
        """
        # Determine which samples use uniform vs beta
        uniform_mask = torch.rand(batch_size, device=device) < self.uniform_fraction

        # Sample from Beta distribution
        beta_dist = torch.distributions.Beta(self.beta_alpha, self.beta_beta)
        t = beta_dist.sample((batch_size,)).to(device)

        # Replace some with uniform
        uniform_t = torch.rand(batch_size, device=device)
        t = torch.where(uniform_mask, uniform_t, t)

        # Clamp to avoid exact 0 or 1
        t = t.clamp(1e-5, 1.0 - 1e-5)
        return t

    def sample_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample centered Gaussian noise.

        Returns:
            x_0: noise with zero mean per sample.
        """
        x_0 = torch.randn(shape, device=device)
        # Center per sample (subtract mean across atom dimension)
        x_0 = x_0 - x_0.mean(dim=-2, keepdim=True)
        return x_0

    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute interpolated state and target velocity.

        Args:
            x_0: (B, L, 3) noise
            x_1: (B, L, 3) target coordinates
            t: (B,) timesteps

        Returns:
            x_t: (B, L, 3) interpolated state
            u_t: (B, L, 3) target velocity field
        """
        t_expanded = t[:, None, None]  # (B, 1, 1)
        x_t = t_expanded * x_1 + (1.0 - t_expanded) * x_0
        u_t = x_1 - x_0
        return x_t, u_t

    def forward(
        self, x_1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample noise, time, and compute interpolation for training.

        Args:
            x_1: (B, L, 3) target coordinates

        Returns:
            t: (B,) sampled timesteps
            x_t: (B, L, 3) noisy interpolated coords
            u_t: (B, L, 3) target velocity
            x_0: (B, L, 3) sampled noise
        """
        B = x_1.shape[0]
        x_0 = self.sample_noise(x_1.shape, x_1.device)
        t = self.sample_time(B, x_1.device)
        x_t, u_t = self.interpolate(x_0, x_1, t)
        return t, x_t, u_t, x_0
