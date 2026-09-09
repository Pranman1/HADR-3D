"""Randomness, seeded and passed in.

The 2D simulator's fire draws from the unseeded global numpy generator, so the same
seed does not give the same fire and no run can be reproduced exactly.  That is a
one-line mistake with expensive consequences, and this module exists so it cannot be
repeated: nothing in `hadrsim` may call `torch.rand` or `np.random` directly.

One generator serves the whole batch.  Per-environment streams would be tidier but
buy nothing here -- what reproducibility needs is that the same seed, the same batch
size and the same sequence of calls produce the same numbers, which a single seeded
generator already gives.
"""

from __future__ import annotations

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
        """Coin flips with per-element probability. The fire's whole spread step."""
        return torch.rand(p.shape, generator=self._gen, device=p.device) < p

    def fork(self, tag: str) -> "Rng":
        """An independent stream, named.

        Lets one subsystem's random draws stop perturbing another's: adding a gust of
        wind should not silently change which cells the fire ignites.
        """
        mixed = (self.seed * 0x9E3779B1 + (hash(tag) & 0xFFFFFFFF)) & 0x7FFFFFFF
        return Rng(mixed, self.device)

    def state(self) -> torch.Tensor:
        return self._gen.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self._gen.set_state(state)

    def __repr__(self) -> str:
        return f"Rng(seed={self.seed}, device={self.device})"
