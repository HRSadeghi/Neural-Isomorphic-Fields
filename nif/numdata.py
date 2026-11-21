"""Numbers of D_N as tensors: random sampling, exact sums and products (used only to build training targets,
never inside the model), the steps of a product used as hint targets, and the token sequences
(15 tokens: 0-9, '.', '-', '+', start, padding).

A number is held as (sign, digits) with digits[:, i] the digit of exponent E_MIN + i.

N, the maximum number of digits, is fixed when this module is imported: 25, or the value of the environment
variable NIF_N_DIGITS (train.py sets it from the run's n_digits). The slot window has K = 2N + 2 places,
exponents -(N+1) .. N.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

# ----------------------------------------------------------------------------- vocabulary
DOT, MINUS, PLUS, START, PAD = 10, 11, 12, 13, 14
VOCAB = 15
N_DIGITS = int(os.environ.get("NIF_N_DIGITS", "25"))   # D_N; N = 25 in the main experiments of the paper
E_MIN, E_MAX = -(N_DIGITS + 1), N_DIGITS                # exponent window: 2N + 2 slots (52 for N = 25)
K = E_MAX - E_MIN + 1


def mul_bounds(N: int = N_DIGITS):
    """Bounds of Lemma 3.4 for operands with at most N digits: L_k <= 9N and H_k <= 8N have at most P
    decimal digits (P = 3 for N <= 111, P = 4 for N <= 1111), S_k is a sum of 2P aligned digits of L and H
    (S_max = 40 for N = 25, 56 for N = 200), T_k <= 9 + floor(S_max / 10), and the carries of the final pass are 0
    or 1 as long as T_max <= 18. Returns (P, S_max, T_max)."""
    P = len(str(9 * N))

    def dmax(v, p):                                       # largest digit at place p of an integer in [0, v]
        return 9 if v >= 10 ** (p + 1) - 1 else v // 10 ** p
    S = sum(dmax(9 * N, p) + dmax(8 * N, p) for p in range(P))
    T = 9 + S // 10
    assert T <= 18, "carries larger than 1: the staged operator would need a further stage"
    return P, S, T


# ----------------------------------------------------------------------------- sampling
def _neg_binomial(shape, r, p, device, g):
    """Failures before the r-th success (numpy's convention), as a sum of r geometrics."""
    u = torch.rand((r,) + tuple(shape), device=device, generator=g).clamp_min(1e-12)
    return torch.floor(torch.log(u) / torch.log(torch.tensor(1.0 - p, device=device))).sum(0).long()


def sample_numbers(B: int, device, g: Optional[torch.Generator] = None, mode: str = "paper",
                   max_len: int = 12, p_neg: float = 0.5):
    """Random elements of D_N.

    mode="paper": integer and fractional lengths ~ negative binomial (r=2, p=0.45), random digits
                  (leading / trailing zeros allowed, as in the paper's sampler), random sign.
    mode="uniform": both lengths uniform in [0, max_len] (to cover the tail of the distribution).
    Returns sign (B,) in {-1, 1} and digits (B, K) int64."""
    if mode == "paper":
        li = _neg_binomial((B,), 2, 0.45, device, g)
        lf = _neg_binomial((B,), 2, 0.45, device, g)
    else:
        li = torch.randint(0, max_len + 1, (B,), device=device, generator=g)
        lf = torch.randint(0, max_len + 1, (B,), device=device, generator=g)
    li = li.clamp(max=E_MAX + 1)
    lf = lf.clamp(max=-E_MIN)
    e = torch.arange(E_MIN, E_MAX + 1, device=device)
    present = (e[None] < li[:, None]) & (e[None] >= -lf[:, None])
    digits = torch.randint(0, 10, (B, K), device=device, generator=g) * present
    sign = torch.where(torch.rand(B, device=device, generator=g) < p_neg, -1, 1)
    return sign, digits


def num_digits(sign, digits):
    """Digits of the canonical string (sign and point excluded), as in nif_numbers.num_digits."""
    e = torch.arange(E_MIN, E_MAX + 1, device=digits.device)
    nz = digits != 0
    hi = torch.where(nz, e, torch.full_like(e, -10 ** 6)).max(1).values
    lo = torch.where(nz, e, torch.full_like(e, 10 ** 6)).min(1).values
    n_int = torch.where(hi >= 0, hi + 1, torch.ones_like(hi))
    n_frac = torch.where(lo < 0, -lo, torch.zeros_like(lo))
    zero = ~nz.any(1)
    return torch.where(zero, torch.ones_like(n_int), n_int + n_frac), n_int, n_frac


# ----------------------------------------------------------------------------- exact arithmetic (targets only)
def _carry(c: torch.Tensor):
    """Integer coefficients (B, P) in exponent order -> (negative flag, digits (B, P)), plus an
    overflow flag. Plain sequential carry; used only to build targets."""
    def run(x):
        B, P = x.shape
        carry = torch.zeros(B, dtype=torch.long, device=x.device)
        d = torch.empty_like(x)
        for p in range(P):
            t = x[:, p] + carry
            d[:, p] = torch.remainder(t, 10)
            carry = torch.div(t - d[:, p], 10, rounding_mode="floor")
        return d, carry
    d, carry = run(c)
    neg = carry < 0
    if neg.any():
        d2, c2 = run(-c[neg])
        d[neg] = d2
        carry = carry.clone()
        carry[neg] = c2
    return neg, d, carry != 0


def exact_add(sa, da, sb, db):
    c = sa[:, None] * da + sb[:, None] * db
    c = torch.cat([c, torch.zeros_like(c[:, :2])], 1)               # room for a carry
    neg, d, ovf = _carry(c)
    ovf = ovf | (d[:, K:] != 0).any(1)
    return torch.where(neg, -1, 1), d[:, :K], ovf


def exact_mul(sa, da, sb, db):
    """Product via a direct convolution in float64 (exact for these sizes)."""
    B = da.shape[0]
    a = da.double().unsqueeze(0)                                     # (1, B, K)
    w = db.double().flip(1).unsqueeze(1)                             # (B, 1, K)
    full = torch.nn.functional.conv1d(a, w, padding=K - 1, groups=B).squeeze(0).round().long()  # (B, 2K-1)
    # full[:, j] has exponent 2*E_MIN + j; keep E_MIN .. E_MAX, check the rest is zero
    lo = -E_MIN                                                       # index of exponent E_MIN
    keep = full[:, lo:lo + K]
    outside = torch.cat([full[:, :lo], full[:, lo + K:]], 1)
    c = torch.cat([keep, torch.zeros_like(keep[:, :3])], 1)
    neg, d, ovf = _carry(c)
    ovf = ovf | (outside != 0).any(1) | (d[:, K:] != 0).any(1)
    return sa * sb, d[:, :K], ovf


def _coupled_conv(x1, x2):
    """sum_{i+j = k - E_MIN} x1[:, i] * x2[:, j] for every channel, kept on the slot window (float64)."""
    n = 2 * K
    y = torch.fft.irfft(torch.fft.rfft(x1.double(), n=n, dim=1) * torch.fft.rfft(x2.double(), n=n, dim=1), n=n, dim=1)
    return y[:, -E_MIN:-E_MIN + K].round().long()


_P = torch.arange(10)
_PROD = _P[:, None] * _P[None, :]                      # 10 x 10 table of digit products


def mul_hints(da, db):
    """Intermediate targets of a multiplication algorithm whose every step is as easy as an
    addition (used only as auxiliary training targets, never at test time):

      stage 1  L_k = sum_{i+j=k} (a_i b_j mod 10),  H_k = sum_{i+j=k} floor(a_i b_j / 10)   (<= 9m, 8m; m <= N digits)
               targets: the P decimal digits of L_k and of H_k (P = 3 for N = 25) -> planes (B, K, 2P)
      stage 2  S_k = sum_p L_p,{k-p} + sum_p H_p,{k-p-1}  (<= 40 for N = 25; mul_bounds)
               targets: S_k mod 10 and floor(S_k / 10)                         -> (B, K, 2)
      stage 3  T_k = (S_k mod 10) + floor(S_{k-1} / 10)  (<= 13), then a carry pass whose
               carries are 0 or 1, exactly as in an addition                  -> final digits
    The value sum_k L_k 10^k + 10 sum_k H_k 10^k = sum_k S_k 10^k = sum_k T_k 10^k = |a b|
    (Lemma 3.4 of the paper). For N = 25 the hint heads (layers.MulHintHeads) have classes for
    floor(S_k/10) <= 3 and T_k <= 12 and clamp larger targets; those need S_k = 40, i.e. H_k >= 200,
    which requires m = 25 and almost only large digits, so the clamp affects only the auxiliary
    targets of practically non-occurring columns, never the training targets of the product. For other N the
    heads cover the full ranges of mul_bounds(N)."""
    dev = da.device
    oa = torch.nn.functional.one_hot(da, 10).double()                  # (B, K, 10)
    ob = torch.nn.functional.one_hot(db, 10).double()
    x1 = oa[:, :, :, None].expand(-1, -1, -1, 10).reshape(da.shape[0], K, 100)
    x2 = ob[:, :, None, :].expand(-1, -1, 10, -1).reshape(db.shape[0], K, 100)
    cnt = _coupled_conv(x1, x2)                                         # (B, K, 100) pair counts
    prod = _PROD.reshape(-1).to(dev)
    L = (cnt * (prod % 10)).sum(-1)
    H = (cnt * (prod // 10)).sum(-1)

    P = mul_bounds()[0]

    def planes(x):                                                      # decimal digits 0 .. P-1 (P = 3 for N = 25)
        return torch.stack([(x // 10 ** p) % 10 if p < P - 1 else x // 10 ** p for p in range(P)], -1)

    PL, PH = planes(L), planes(H)

    def sh(x, s):                                                       # x shifted up by s places
        return torch.cat([torch.zeros_like(x[:, :s]), x[:, :K - s]], 1) if s else x

    S = sum(sh(PL[..., p], p) + sh(PH[..., p], p + 1) for p in range(P))
    r, q = S % 10, S // 10
    T = r + sh(q, 1)
    return {"planes": torch.cat([PL, PH], -1), "rq": torch.stack([r, q], -1), "T": T, "LH": torch.stack([L, H], -1)}


def admissible_pairs(B: int, op: str, device, g=None, p_uniform: float = 0.5, max_len: int = 12,
                     N: int = N_DIGITS):
    """Pairs (a, b) with a, b, and a op b in D_N (the admissible sets of the paper), drawn from a
    mixture of the paper's sampler and uniform lengths. Returns operands and the exact result."""
    out = []
    got = 0
    while got < B:
        n = 2 * (B - got) + 64
        nu = int(n * p_uniform)
        sa1, da1 = sample_numbers(n - nu, device, g, "paper")
        sa2, da2 = sample_numbers(nu, device, g, "uniform", max_len)
        sb1, db1 = sample_numbers(n - nu, device, g, "paper")
        sb2, db2 = sample_numbers(nu, device, g, "uniform", max_len)
        sa, da = torch.cat([sa1, sa2]), torch.cat([da1, da2])
        sb, db = torch.cat([sb1, sb2]), torch.cat([db1, db2])
        perm = torch.randperm(n, device=device, generator=g)
        sa, da, sb, db = sa[perm], da[perm], sb[perm], db[perm]
        sr, dr, ovf = (exact_add if op == "+" else exact_mul)(sa, da, sb, db)
        ok = ~ovf & (num_digits(sa, da)[0] <= N) & (num_digits(sb, db)[0] <= N) & (num_digits(sr, dr)[0] <= N)
        idx = ok.nonzero().squeeze(1)[:B - got]
        out.append((sa[idx], da[idx], sb[idx], db[idx], sr[idx], dr[idx]))
        got += idx.numel()
    return tuple(torch.cat([o[i] for o in out]) for i in range(6))


def canonical_sign(sign, digits):
    """Zero has a plus sign."""
    return torch.where((digits != 0).any(1), sign, torch.ones_like(sign))


# ----------------------------------------------------------------------------- tokens
def to_tokens(sign, digits, max_len: int = 1 + 1 + (E_MAX + 1) + 1 + (-E_MIN)):
    """Canonical token sequences [START, sign, int digits (MSB first), '.', frac digits] padded with
    PAD, the exponent of every token (digits) or a marker, and the lengths (n_int, n_frac)."""
    B = digits.shape[0]
    dev = digits.device
    sign = canonical_sign(sign, digits)
    _, n_int, n_frac = num_digits(sign, digits)
    T = max_len
    pos = torch.arange(T, device=dev)[None].expand(B, T)
    # position 0 START, 1 sign, 2 .. 1+n_int integer digits, then '.', then fraction digits
    int_pos = (pos >= 2) & (pos < 2 + n_int[:, None])
    exp_int = n_int[:, None] - 1 - (pos - 2)
    has_frac = n_frac > 0
    dot_pos = 2 + n_int
    frac_pos = has_frac[:, None] & (pos > dot_pos[:, None]) & (pos <= (dot_pos + n_frac)[:, None])
    exp_frac = dot_pos[:, None] - pos
    exp = torch.where(int_pos, exp_int, torch.where(frac_pos, exp_frac, torch.full_like(pos, 10 ** 6)))
    idx = (exp - E_MIN).clamp(0, K - 1)
    dig = torch.gather(digits, 1, idx)
    tok = torch.full((B, T), PAD, dtype=torch.long, device=dev)
    tok[:, 0] = START
    tok[:, 1] = torch.where(sign < 0, MINUS, PLUS)
    tok = torch.where(int_pos | frac_pos, dig, tok)
    tok = torch.where(has_frac[:, None] & (pos == dot_pos[:, None]), torch.full_like(tok, DOT), tok)
    length = 2 + n_int + torch.where(has_frac, 1 + n_frac, torch.zeros_like(n_frac))
    return tok, exp, n_int, n_frac, length


def tokens_to_strings(tok):
    out = []
    for row in tok.tolist():
        s = ""
        for t in row[1:]:
            if t == PAD:
                break
            s += "." if t == DOT else "-" if t == MINUS else "+" if t == PLUS else ("" if t == START else str(t))
        out.append(s)
    return out


def strings_to_number(strings, device):
    """Canonical (or not) number strings -> (sign, digits)."""
    B = len(strings)
    digits = torch.zeros(B, K, dtype=torch.long)
    sign = torch.ones(B, dtype=torch.long)
    for i, s in enumerate(strings):
        s = s.strip()
        if s[:1] in "+-":
            sign[i] = -1 if s[0] == "-" else 1
            s = s[1:]
        ip, fp = (s.split(".") + [""])[:2] if "." in s else (s, "")
        ip = ip.lstrip("0")
        fp = fp.rstrip("0")
        if len(ip) > E_MAX + 1 or len(fp) > -E_MIN:
            raise ValueError("number outside the exponent window: " + s)
        for j, ch in enumerate(reversed(ip)):
            digits[i, j - E_MIN] = int(ch)
        for j, ch in enumerate(fp):
            digits[i, -(j + 1) - E_MIN] = int(ch)
    return sign.to(device), digits.to(device)
