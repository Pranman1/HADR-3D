"""Seeded random streams, passed explicitly and forked by name, so a run reproduces
regardless of how often any one consumer draws."""

from __future__ import annotations

import zlib

import torch


class Rng:
    """A seeded source of random numbers, carried explicitly."""

    def __init__(self, seed: int = 0, device: str | torch.device = "cpu"):
        self.seed = int(seed)
        self.device = torch.device(device)
        self._gen = torch.Generator(device=self.device).manual_seed(self.seed)

    def uniform(self, *shape: int, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        u = torch.rand(*shape, generator=self._gen, device=self.device)
        return u if (low, high) == (0.0, 1.0) else low + u * (high - low)

    def normal(self, *shape: int, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        return mean + std * torch.randn(*shape, generator=self._gen, device=self.device)

    def integers(self, low: int, high: int, *shape: int) -> torch.Tensor:
        return torch.randint(low, high, shape, generator=self._gen, device=self.device)

    def bernoulli(self, p: torch.Tensor) -> torch.Tensor:
        """Coin flips with per-element probability."""
        return torch.rand(p.shape, generator=self._gen, device=p.device) < p

    def fork(self, tag: str) -> "Rng":
        """An independent stream named by `tag` (hashed with crc32, so it is the same in every process)."""
        mixed = (self.seed * 0x9E3779B1 + zlib.crc32(tag.encode())) & 0x7FFFFFFF
        return Rng(mixed, self.device)

    def state(self) -> torch.Tensor:
        return self._gen.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self._gen.set_state(state)

    def __repr__(self) -> str:
        return f"Rng(seed={self.seed}, device={self.device})"
