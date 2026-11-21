"""The NIF model: encoder, the two operators, decoder and comparator.

    phi      : number tokens -> one vector z in R^d (Transformer encoder pooled into K place-value slots)
    +, x     : R^d x R^d -> R^d, Abelian Neural Operators (stacked AC layers, commutative by construction)
    phi^dag  : R^d -> number tokens (autoregressive Transformer decoder with a length head, greedy decoding)
    omega    : R^d x R^d -> probabilities of <, >, = (Transformer comparator), antisymmetrized as omega-bar

Every digit token carries an embedding of its exponent, and z is read as K slots of c channels (d = K c).
The decoder cross-attends to the slots and predicts the length of its output itself; in the oracle mode
of the tests it is given the length of the target.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .layers import ANO, Attention, Block, DenseANO, MulHintHeads, RMSNorm, StagedMulANO, rope_cache
from .numdata import DOT, E_MAX, E_MIN, K, MINUS, N_DIGITS, PAD, PLUS, START, VOCAB, mul_bounds, strings_to_number, \
    to_tokens, tokens_to_strings

NO_EXP = 10 ** 6


def ck(module, *args, **kw):
    """Activation checkpointing while training (recompute in the backward pass), plain call otherwise."""
    if module.training and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False, **kw)
    return module(*args, **kw)


def amp(t):
    """bf16 autocast on the GPU, nothing on the CPU."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.is_cuda)
N_INT_MAX, N_FRAC_MAX = E_MAX + 1, -E_MIN          # 26 integer digits, 26 fraction digits
T_MAX = 1 + 1 + N_INT_MAX + 1 + N_FRAC_MAX         # START, sign, digits, '.', digits


def exp_index(exp):
    """Exponent -> embedding index: 0..K-1 for digits, K for non-digit tokens."""
    return torch.where(exp == NO_EXP, torch.full_like(exp, K), (exp - E_MIN).clamp(0, K - 1))


class Encoder(nn.Module):
    def __init__(self, c, D=256, heads=8, layers=4):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(VOCAB, D)
        self.exp = nn.Embedding(K + 1, D)
        self.blocks = nn.ModuleList([Block(D, heads, periodic_frac=0.0) for _ in range(layers)])
        self.slot_q = nn.Parameter(torch.randn(K, D) * 0.02)
        self.pool_norm = RMSNorm(D)
        self.pool = Attention(D, heads)
        self.beta = nn.Parameter(torch.tensor(4.0))          # place-value coupling strength
        self.slot_blocks = nn.ModuleList([Block(D, heads, periodic_frac=0.0) for _ in range(2)])
        self.out_norm = RMSNorm(D)
        self.out = nn.Linear(D, c)

    def forward(self, tok, exp):
        """tok, exp (B, T) of the number (without START) -> z (B, K*c)."""
        B, T = tok.shape
        pad = tok == PAD
        ei = exp_index(exp)
        x = self.tok(tok) + self.exp(ei)
        key_mask = (~pad)[:, None, None, :]
        for blk in self.blocks:
            x = ck(blk, x, mask=key_mask)
        # place-value slot pooling: slot e prefers the digit token of exponent e
        slot_e = torch.arange(K, device=tok.device)
        bias = self.beta * (ei[:, None, :] == slot_e[None, :, None]).float()          # (B, K, T)
        bias = bias.masked_fill(pad[:, None, :], float("-inf"))
        q = self.slot_q[None].expand(B, -1, -1) + self.exp.weight[:K][None]
        s = q + self.pool(self.pool_norm(q), mem=self.pool_norm(x), mask=bias[:, None].to(x.dtype))
        for blk in self.slot_blocks:
            s = ck(blk, s)
        z = self.out(self.out_norm(s))                                                # (B, K, c)
        z = F.rms_norm(z, (self.c,))                                                  # fixed scale per slot
        return z.reshape(B, -1)


class Decoder(nn.Module):
    """Autoregressive transformer decoder phi^dag with a length head."""

    def __init__(self, c, D=256, heads=8, layers=4):
        super().__init__()
        self.c, self.D, self.heads = c, D, heads
        self.mem_in = nn.Linear(c, D)
        self.slot = nn.Parameter(torch.randn(K, D) * 0.02)
        self.mem_blocks = nn.ModuleList([Block(D, heads, periodic_frac=0.0) for _ in range(2)])
        self.len_q = nn.Parameter(torch.randn(1, D) * 0.02)
        self.len_att = Attention(D, heads)
        self.len_norm = RMSNorm(D)
        self.len_int = nn.Linear(D, N_INT_MAX)          # n_int in 1..26
        self.len_frac = nn.Linear(D, N_FRAC_MAX + 1)    # n_frac in 0..26
        self.tok = nn.Embedding(VOCAB, D)
        self.exp = nn.Embedding(K + 1, D)
        self.kind = nn.Embedding(4, D)                  # predicting: sign, digit, '.', end
        self.blocks = nn.ModuleList([Block(D, heads, cross=True, periodic_frac=0.0) for _ in range(layers)])
        self.beta = nn.Parameter(torch.tensor(4.0))
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, VOCAB)
        self.register_buffer("cos", torch.empty(0), persistent=False)

    def _rope(self, device):
        if self.cos.numel() == 0 or self.cos.device != device:
            self.cos, self.sin = rope_cache(T_MAX, self.D // self.heads, device)
        return self.cos, self.sin

    def memory(self, z):
        B = z.shape[0]
        m = self.mem_in(z.view(B, K, self.c)) + self.slot + self.exp.weight[:K][None]
        for blk in self.mem_blocks:
            m = ck(blk, m)
        return m

    def lengths(self, m):
        q = self.len_q[None].expand(m.shape[0], -1, -1)
        h = self.len_norm(q + self.len_att(q, mem=m)).squeeze(1)
        return self.len_int(h), self.len_frac(h)

    @staticmethod
    def skeleton(n_int, n_frac):
        """Exponent and kind of the token predicted at each input position (inputs START..)."""
        B = n_int.shape[0]
        pos = torch.arange(T_MAX, device=n_int.device)[None].expand(B, T_MAX)
        # the target at input position t is token t+1 of [START, sign, int, '.', frac]
        tpos = pos + 1
        is_int = (tpos >= 2) & (tpos < 2 + n_int[:, None])
        dot = 2 + n_int[:, None]
        has_frac = (n_frac > 0)[:, None]
        is_frac = has_frac & (tpos > dot) & (tpos <= dot + n_frac[:, None])
        exp = torch.where(is_int, n_int[:, None] - 1 - (tpos - 2),
                          torch.where(is_frac, dot - tpos, torch.full_like(tpos, NO_EXP)))
        kind = torch.where(tpos == 1, 0, torch.where(is_int | is_frac, 1, torch.where(has_frac & (tpos == dot), 2, 3)))
        length = 2 + n_int + torch.where(n_frac > 0, 1 + n_frac, torch.zeros_like(n_frac))
        return exp, kind, length

    def step_logits(self, m, inp, exp, kind):
        """Teacher-forced logits: inp (B, T) input tokens (START, y1, ...), exp/kind of the targets."""
        B, T = inp.shape
        rope = tuple(r[:T] for r in self._rope(inp.device))
        ei = exp_index(exp)
        x = self.tok(inp) + self.exp(ei) + self.kind(kind)
        slot_e = torch.arange(K, device=inp.device)
        bias = (self.beta * (ei[:, :, None] == slot_e[None, None, :]).float())[:, None].to(x.dtype)
        causal = torch.ones(T, T, dtype=torch.bool, device=inp.device).tril()
        for blk in self.blocks:
            x = ck(blk, x, rope=rope, mask=causal, mem=m, mem_mask=bias)
        return self.head(self.norm(x))

    def loss(self, z, tok, n_int, n_frac):
        """Cross-entropy of the token sequence (teacher forcing) plus the length head."""
        m = self.memory(z)
        li, lf = self.lengths(m)
        exp, kind, length = self.skeleton(n_int, n_frac)
        T = int(length.max())
        logits = self.step_logits(m, tok[:, :T], exp[:, :T], kind[:, :T])
        target = torch.cat([tok[:, 1:T + 1], torch.full_like(tok[:, :1], PAD)], 1)[:, :T]
        valid = torch.arange(T, device=z.device)[None] < (length - 1)[:, None]
        ce = F.cross_entropy(logits.float()[valid], target[valid])
        lce = F.cross_entropy(li.float(), n_int - 1) + F.cross_entropy(lf.float(), n_frac)
        return ce, lce

    @torch.no_grad()
    def generate(self, z, n_int=None, n_frac=None):
        """Greedy decoding. Without n_int / n_frac the length head decides the length."""
        m = self.memory(z)
        if n_int is None:
            li, lf = self.lengths(m)
            n_int, n_frac = li.argmax(-1) + 1, lf.argmax(-1)
        exp, kind, length = self.skeleton(n_int, n_frac)
        B = z.shape[0]
        T = int(length.max())
        inp = torch.full((B, T + 1), PAD, dtype=torch.long, device=z.device)
        inp[:, 0] = START
        for t in range(T - 1):
            logits = self.step_logits(m, inp[:, :t + 1], exp[:, :t + 1], kind[:, :t + 1])[:, -1].float()
            k = kind[:, t]
            # the skeleton fixes the kind of token; choose the best token of that kind
            allowed = torch.zeros(B, VOCAB, dtype=torch.bool, device=z.device)
            allowed[:, :10] = (k == 1)[:, None]
            allowed[:, DOT] = k == 2
            allowed[:, MINUS] = k == 0
            allowed[:, PLUS] = k == 0
            nxt = logits.masked_fill(~allowed, float("-inf")).argmax(-1)
            inp[:, t + 1] = torch.where(t + 1 <= length - 1, nxt, torch.full_like(nxt, PAD))
        return inp


class Comparator(nn.Module):
    """omega(h1, h2): probabilities of [first smaller, first larger, equal]."""

    def __init__(self, c, D=192, heads=6, layers=3):
        super().__init__()
        self.c = c
        self.inp = nn.Linear(c, D)
        self.slot = nn.Parameter(torch.randn(K, D) * 0.02)
        self.side = nn.Parameter(torch.randn(2, D) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, heads, periodic_frac=0.0) for _ in range(layers)])
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, 3)

    def logits(self, h1, h2):
        B = h1.shape[0]
        x1 = self.inp(h1.view(B, K, self.c)) + self.slot + self.side[0]
        x2 = self.inp(h2.view(B, K, self.c)) + self.slot + self.side[1]
        x = torch.cat([self.cls[None].expand(B, -1, -1), x1, x2], 1)
        for blk in self.blocks:
            x = ck(blk, x)
        return self.head(self.norm(x[:, 0]))

    def bar_log_probs(self, h1, h2):
        """log of omega-bar = (omega(h1,h2) + P omega(h2,h1)) / 2, Eq. (11) of the paper."""
        p12 = self.logits(h1, h2).float().softmax(-1)
        p21 = self.logits(h2, h1).float().softmax(-1)[:, [1, 0, 2]]
        return torch.log(0.5 * (p12 + p21) + 1e-9)


class NIF(nn.Module):
    def __init__(self, c=10, enc=(256, 8, 4), dec=(256, 8, 4), add_op=(192, 6, 4, 1),
                 mul_op=dict(c_int=32, D=192, heads=6, stages=(2, 2, 2, 2)), cmp=(192, 6, 3),
                 add_kind="slot", mul_kind="staged", mul_front=None, dense_layers=4, add_fp32=False, n_digits=None):
        super().__init__()
        if n_digits is not None and n_digits != N_DIGITS:
            raise ValueError(f"this checkpoint is for numbers with up to {n_digits} digits, but numdata was imported with "
                             f"N = {N_DIGITS}: set the environment variable NIF_N_DIGITS={n_digits} before starting Python")
        self.c, self.d = c, K * c
        self.cfg = dict(c=c, enc=enc, dec=dec, add_op=add_op, mul_op=mul_op, cmp=cmp, add_kind=add_kind,
                        mul_kind=mul_kind, mul_front=mul_front, dense_layers=dense_layers, add_fp32=add_fp32)
        if N_DIGITS != 25:                     # recorded only for the length study (checkpoints of N = 25 unchanged)
            self.cfg["n_digits"] = N_DIGITS
        planes, s_max, t_max = mul_bounds()
        self.mul_kind = mul_kind
        self.encoder = Encoder(c, *enc)
        self.decoder = Decoder(c, *dec)
        if add_kind == "dense":                 # ablation: perceptron AC layers
            self.add_op = DenseANO(K, c, dense_layers)
        else:
            self.add_op = ANO(K, c, *add_op, bilinear=None, shift=E_MIN, fp32=add_fp32)
        if mul_kind == "dense":
            self.mul_op = DenseANO(K, c, dense_layers, fp32=True)
        elif mul_kind == "coupled":             # ablation: plain ANO with a place-coupled bilinear branch
            self.mul_op = ANO(K, c, 192, 6, 6, 1, bilinear="coupled", bilinear_hidden=32, shift=E_MIN)
        else:
            self.mul_op = StagedMulANO(K, c, **mul_op, shift=E_MIN, front_kw=dict(mul_front or {}, planes=planes))
        if mul_kind == "staged":
            # classes of the hint heads: for N = 25 those of Lemma 3.4 (rare large targets clamped), the full
            # ranges of Lemma 3.4 otherwise
            q_cls, t_cls = (4, 13) if N_DIGITS == 25 else (s_max // 10 + 1, t_max + 1)
            self.mul_hints = MulHintHeads(mul_op["D"], len(mul_op["stages"]), planes=planes, q_classes=q_cls,
                                          t_classes=t_cls)                         # training only
        self.comparator = Comparator(c, *cmp)

    # ---------------------------------------------------------------- interface used by the tests
    def encode_numbers(self, sign, digits):
        tok, exp, n_int, n_frac, length = to_tokens(sign, digits)
        return self.encoder(tok[:, 1:], exp[:, 1:])

    @torch.no_grad()
    def encode(self, strings: List[str]):
        dev = next(self.parameters()).device
        sign, digits = strings_to_number(strings, dev)
        with amp(digits):
            return self.encode_numbers(sign, digits).float()

    @torch.no_grad()
    def decode(self, z, target_strings: Optional[List[str]] = None):
        """Greedy decoding; target_strings, if given, are used only for their length (oracle mode)."""
        n_int = n_frac = None
        if target_strings is not None:
            s, d = strings_to_number(target_strings, z.device)
            _, _, n_int, n_frac, _ = to_tokens(s, d)
        with amp(z):
            out = self.decoder.generate(z, n_int, n_frac)
        return tokens_to_strings(out)

    @torch.no_grad()
    def decoder_logits(self, z, target_strings):
        """Teacher-forced logits for the margin / decoding-radius estimates (Lemma 3.23).
        Rows 0..n-1: token logits for the correct prefix; the last two rows: the two classifiers of the length
        head, so that the margin also covers the length chosen by the length-free decoder. Unused columns and
        the padded positions of shorter strings are constants (-1e4, target 0 with logit 0): they never decide
        the margin and do not change under perturbations."""
        s, d = strings_to_number(target_strings, z.device)
        tok, exp, n_int, n_frac, length = to_tokens(s, d)
        with amp(z):
            m = self.decoder.memory(z)
            li, lf = self.decoder.lengths(m)
            e, kind, L = self.decoder.skeleton(n_int, n_frac)
            T = int(L.max()) - 1
            lg = self.decoder.step_logits(m, tok[:, :T], e[:, :T], kind[:, :T]).float()
        V = max(lg.shape[-1], li.shape[-1], lf.shape[-1])
        pad = lambda x: F.pad(x.float(), (0, V - x.shape[-1]), value=-1e4)
        lg = pad(lg)
        ids = tok[:, 1:T + 1].clone()
        invalid = torch.arange(T, device=z.device)[None] >= (L - 1)[:, None]
        lg[invalid] = -1e4
        lg[..., 0][invalid] = 0.0
        ids[invalid] = 0
        logits = torch.cat([lg, pad(li)[:, None], pad(lf)[:, None]], 1)
        ids = torch.cat([ids, (n_int - 1)[:, None], n_frac[:, None]], 1)
        return logits, ids

    def ac_layer_weights(self, op):
        """W^(i) of the AC layers for the spectral bound of Lemma 3.25 (perceptron layers only)."""
        mod = self.add_op if op == "+" else self.mul_op
        if isinstance(mod, DenseANO):
            return [W.weight for W in mod.W]
        raise NotImplementedError("transformer AC layers: no global Lipschitz bound (estimated empirically)")

    @torch.no_grad()
    def add(self, h1, h2):
        with amp(h1):
            return self.add_op(h1, h2).float()

    @torch.no_grad()
    def mul(self, h1, h2):
        with amp(h1):
            return self.mul_op(h1, h2).float()

    @torch.no_grad()
    def compare(self, h1, h2):
        with amp(h1):
            return self.comparator.logits(h1, h2).float().softmax(-1)
