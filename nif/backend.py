"""Tiny helpers that work for both torch tensors and numpy arrays."""
from __future__ import annotations

from typing import List

import numpy as np


def _is_torch(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def to_numpy(x) -> np.ndarray:
    if _is_torch(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def rownorm(x) -> List[float]:
    """Euclidean norm of every row of a (B, d) array."""
    return np.linalg.norm(to_numpy(x), axis=-1).tolist()


def randn_like(x, rng: np.random.Generator):
    """Standard normal noise of the same shape, dtype and device as x."""
    noise = rng.standard_normal(size=tuple(x.shape)).astype(np.float32)
    if _is_torch(x):
        import torch
        return torch.from_numpy(noise).to(device=x.device, dtype=x.dtype)
    return noise


def scale_rows(x, factors: List[float]):
    """Multiply row i of x by factors[i]."""
    f = np.asarray(factors, dtype=np.float32).reshape(-1, 1)
    if _is_torch(x):
        import torch
        return x * torch.from_numpy(f).to(device=x.device, dtype=x.dtype)
    return x * f


def spectral_norm(w) -> float:
    return float(np.linalg.norm(to_numpy(w), 2))


def swap_first_two(w):
    """P w for a batch of comparator outputs (B, 3): swaps columns 0 and 1."""
    return w[:, [1, 0, 2]]
