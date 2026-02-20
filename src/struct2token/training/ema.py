"""Exponential Moving Average of model parameters."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """Maintains an exponential moving average of model parameters.

    Usage:
        ema = EMA(model, decay=0.999)
        # In training loop:
        ema.update()
        # For evaluation:
        with ema.average_parameters():
            validate(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_shadow()

    def _init_shadow(self):
        """Initialize shadow parameters as copies of model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update shadow parameters with exponential moving average."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self):
        """Replace model parameters with shadow (EMA) parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model parameters from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    class _AverageContext:
        """Context manager for temporarily using EMA parameters."""
        def __init__(self, ema: "EMA"):
            self.ema = ema

        def __enter__(self):
            self.ema.apply_shadow()
            return self.ema.model

        def __exit__(self, *args):
            self.ema.restore()

    def average_parameters(self) -> _AverageContext:
        """Context manager: temporarily swap model params with EMA params."""
        return self._AverageContext(self)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]
