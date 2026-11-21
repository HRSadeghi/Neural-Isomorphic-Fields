"""Helpers of the training loop: exact order of digit arrays, batched encoding, decoder loss, exact match."""
from __future__ import annotations

import torch

from .numdata import PAD, canonical_sign, exact_add, to_tokens


def rel(sa, da, sb, db):
    """0: a < b, 1: a > b, 2: a == b (exact, on digit arrays)."""
    sa, sb = canonical_sign(sa, da), canonical_sign(sb, db)
    s, d, _ = exact_add(sa, da, -sb, db)
    zero = (d == 0).all(1)
    return torch.where(zero, 2, torch.where(s < 0, 0, 1))


def encode_many(model, numbers):
    """Encode several (sign, digits) batches in one encoder call."""
    toks, exps, sizes = [], [], []
    T = 0
    for s, d in numbers:
        tok, exp, _, _, length = to_tokens(s, d)
        toks.append(tok[:, 1:])
        exps.append(exp[:, 1:])
        sizes.append(s.shape[0])
        T = max(T, int(length.max()) - 1)
    tok = torch.cat([t[:, :T] for t in toks])
    exp = torch.cat([e[:, :T] for e in exps])
    z = model.encoder(tok, exp)
    return list(z.split(sizes))


def dec_loss(model, z, s, d):
    tok, exp, n_int, n_frac, length = to_tokens(s, d)
    return model.decoder.loss(z, tok, n_int, n_frac)


def em_mask(model, z, s, d):
    tok = model.decoder.generate(z)
    ref, _, _, _, _ = to_tokens(s, d)
    T = min(tok.shape[1], ref.shape[1])
    return (tok[:, :T] == ref[:, :T]).all(1) & (ref[:, T:] == PAD).all(1) & (tok[:, T:] == PAD).all(1)
