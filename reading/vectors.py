"""Number vectors for the reading experiment: the frozen NIF encoder, and FoNE with exact digit arithmetic."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nif.model import NIF  # noqa: E402
from nif.numdata import K, strings_to_number  # noqa: E402


class NIFVectors(nn.Module):
    """phi(a) of a trained NIF (frozen, evaluation mode)."""

    def __init__(self, ckpt):
        super().__init__()
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.nif = NIF(**st["cfg"])
        self.nif.load_state_dict(st["model"])
        self.nif.eval().requires_grad_(False)
        self.dim, self.c = self.nif.d, self.nif.c

    @torch.no_grad()
    def encode(self, strings):
        return self.nif.encode(list(strings))


class FoNEVectors(nn.Module):
    """FoNE features computed with exact digit arithmetic: slot k (exponent e = k + E_MIN) holds cos and sin of
    2 pi (x mod 10^(e+1)) / 10^(e+1); one more dimension holds the sign."""

    def __init__(self, terms=16):
        super().__init__()
        self.dim = 2 * K + 1
        self.terms = terms
        self.register_buffer("w", (10.0 ** -(torch.arange(terms, dtype=torch.float64) + 1)).flip(0), persistent=False)

    @torch.no_grad()
    def encode(self, strings):
        sign, dig = strings_to_number(list(strings), self.w.device)
        win = F.pad(dig.double(), (self.terms - 1, 0)).unfold(1, self.terms, 1)      # digits e, e-1, ... per slot
        ang = 2 * math.pi * (win * self.w).sum(-1)
        feat = torch.stack([ang.cos(), ang.sin()], -1).reshape(len(strings), -1)
        return torch.cat([feat, sign.double()[:, None]], 1).float()
