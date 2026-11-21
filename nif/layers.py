"""Building blocks: RMSNorm, rotary embeddings, attention, SwiGLU with periodic units, the Abelian
Concatenation (AC) layer and the Abelian Neural Operator (ANO),

    t1^i = f_i([t1^{i-1}, t2^{i-1}]),   t2^i = f_i([t2^{i-1}, t1^{i-1}]),   h1 * h2 = t1^k + t2^k,

the staged multiplication operator with its hint heads, and the perceptron AC layers of the ablations.
Each f_i is a small slot Transformer: the two input vectors are read as 2K slot tokens (one per place
value), mixed by self-attention and a SwiGLU MLP with periodic units, and the first K tokens are written
back as the output. The operators are commutative for every parameter value (Lemma 3.3 of the paper),
because the same f_i is applied to both orders and the two streams are summed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

RMS_EPS = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, d, eps=None):
        super().__init__()
        self.eps, self.w = (RMS_EPS if eps is None else eps), nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w.to(x.dtype), self.eps)


def rope_cache(n, dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(n, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    """x (B, H, T, Dh); cos/sin (T, Dh/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, heads, cross=False):
        super().__init__()
        self.h, self.cross = heads, cross
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, mem=None, rope_q=None, rope_k=None, mask=None, causal=False):
        B, T, D = x.shape
        src = x if mem is None else mem
        S = src.shape[1]
        q = self.q(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(src).view(B, S, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if rope_q is not None:
            q = apply_rope(q, *rope_q)
        if rope_k is not None:
            k = apply_rope(k, *rope_k)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)
        return self.o(a.transpose(1, 2).reshape(B, T, D))


class FourierSwiGLU(nn.Module):
    """SwiGLU MLP with an extra periodic branch [sin(Wx), cos(Wx)] (as in Fourier Analysis
    Networks), which lets the MLP represent digit-wise periodic functions such as x mod 10."""

    def __init__(self, d, mult=4, periodic_frac=0.25):
        super().__init__()
        hid = int(mult * d * 2 / 3)
        self.p = max(8, int(hid * periodic_frac)) if periodic_frac > 0 else 0
        self.w12 = nn.Linear(d, 2 * hid, bias=False)
        if self.p:
            self.wp = nn.Linear(d, self.p, bias=True)
        self.w3 = nn.Linear(hid + 2 * self.p, d, bias=False)

    def forward(self, x):
        g, u = self.w12(x).chunk(2, dim=-1)
        h = [F.silu(g) * u]
        if self.p:
            z = self.wp(x)
            h += [torch.sin(z), torch.cos(z)]
        return self.w3(torch.cat(h, dim=-1))


class Block(nn.Module):
    """Pre-norm transformer block: self-attention (+ optional cross-attention) + MLP."""

    def __init__(self, d, heads, cross=False, periodic_frac=0.25):
        super().__init__()
        self.n1, self.att = RMSNorm(d), Attention(d, heads)
        self.cross = cross
        if cross:
            self.nc, self.catt = RMSNorm(d), Attention(d, heads)
        self.n2, self.mlp = RMSNorm(d), FourierSwiGLU(d, periodic_frac=periodic_frac)

    def forward(self, x, rope=None, mask=None, causal=False, mem=None, mem_mask=None, rope_mem=None):
        h = self.n1(x)
        x = x + self.att(h, rope_q=rope, rope_k=rope, mask=mask, causal=causal)
        if self.cross:
            x = x + self.catt(self.nc(x), mem=mem, rope_q=rope if rope_mem is not None else None,
                              rope_k=rope_mem, mask=mem_mask)
        return x + self.mlp(self.n2(x))


# ----------------------------------------------------------------------------- AC layers and ANO
class ACLayer(nn.Module):
    """One Abelian Concatenation layer f_i([h1, h2]) : R^{2d} -> R^d, with d = K * c.

    bilinear = "coupled": place-value-coupled bilinear branch. With learned feature maps A, B,
        y_k = sum_{i + j = k - shift} (A h1_i) * (B h2_j)
    i.e. features of places i and j interact at place i + j. This is the multiplicative analogue
    of position coupling: the connectivity is fixed, as in a convolution, and all weights are
    learned. bilinear = "dense": W_o[(P h1) * (Q h2)] over the whole vectors."""

    def __init__(self, K, c, D, heads, sub_blocks=1, bilinear=None, bilinear_hidden=0, periodic_frac=0.25,
                 shift=0):
        super().__init__()
        self.K, self.c, self.shift = K, c, shift
        self.inp = nn.Linear(c, D)
        self.slot = nn.Parameter(torch.randn(K, D) * 0.02)
        self.side = nn.Parameter(torch.randn(2, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, heads, periodic_frac=periodic_frac) for _ in range(sub_blocks)])
        self.norm = RMSNorm(D)
        self.out = nn.Linear(D, c)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.bil = bilinear
        if bilinear == "dense":
            self.P = nn.Linear(K * c, bilinear_hidden, bias=False)
            self.Q = nn.Linear(K * c, bilinear_hidden, bias=False)
            self.Wb = nn.Linear(bilinear_hidden, K * c, bias=False)
            self.inp_b = nn.Linear(c, D, bias=False)
        elif bilinear == "coupled":
            r = bilinear_hidden
            self.A = nn.Linear(c, r, bias=True)
            self.Bm = nn.Linear(c, r, bias=True)
            self.inp_b = nn.Linear(r, D, bias=False)
            i = torch.arange(K)
            o = (i[:, None] + i[None, :] + shift).reshape(-1)                 # output place of (i, j)
            keep = (o >= 0) & (o < K)
            self.register_buffer("pair_keep", keep.nonzero().squeeze(1), persistent=False)
            self.register_buffer("pair_out", o[keep], persistent=False)

    def coupled(self, h1, h2):
        """sum over i + j = k - shift of x1_i * x2_j, computed as a linear convolution along the
        place axis with an FFT of length 2K (float32; same result as the explicit pair sum up to
        rounding, without materialising the K x K pairs)."""
        x1, x2 = self.A(h1).float(), self.Bm(h2).float()                     # (B, K, r)
        n = 2 * self.K
        y = torch.fft.irfft(torch.fft.rfft(x1, n=n, dim=1) * torch.fft.rfft(x2, n=n, dim=1), n=n, dim=1)
        o = torch.arange(self.K, device=h1.device) - self.shift              # pair-sum index of output place
        return y[:, o.clamp(0, n - 1)] * ((o >= 0) & (o < n - 1)).to(y.dtype)[None, :, None]

    def forward(self, h1, h2, rope):
        B = h1.shape[0]
        x1 = self.inp(h1) + self.slot + self.side[0]
        x2 = self.inp(h2) + self.slot + self.side[1]
        if self.bil == "dense":
            b = self.Wb(self.P(h1.reshape(B, -1)) * self.Q(h2.reshape(B, -1))).view(B, self.K, self.c)
            x1 = x1 + self.inp_b(b)
        elif self.bil == "coupled":
            x1 = x1 + self.inp_b(self.coupled(h1, h2))
        x = torch.cat([x1, x2], dim=1)
        rp = (torch.cat([rope[0], rope[0]]), torch.cat([rope[1], rope[1]]))
        for blk in self.blocks:
            x = blk(x, rope=rp)
        return h1 + self.out(self.norm(x[:, :self.K]))


class ANO(nn.Module):
    """Abelian Neural Operator (Eq. (6) of the paper) on vectors of R^{K*c}, viewed as K slots."""

    def __init__(self, K, c, D=192, heads=6, layers=6, sub_blocks=1, bilinear=None, bilinear_hidden=0,
                 periodic_frac=0.25, shift=0, bilinear_layers=1, fp32=True):
        super().__init__()
        self.K, self.c, self.fp32 = K, c, fp32
        self.layers = nn.ModuleList([
            ACLayer(K, c, D, heads, sub_blocks, bilinear if i < bilinear_layers else None, bilinear_hidden,
                    periodic_frac, shift)
            for i in range(layers)])
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.D, self.heads = D, heads

    def _rope(self, device):
        if self.cos.numel() == 0 or self.cos.device != device:
            c, s = rope_cache(self.K, self.D // self.heads, device)
            self.cos, self.sin = c, s
        return self.cos, self.sin

    def forward(self, h1, h2):
        """h1, h2 (B, K*c) -> (B, K*c). Both orders are evaluated in one batch. With fp32=True the
        operator runs in full float32 (no autocast): its features carry integer-valued quantities
        such as the column sums of a product, which bf16 cannot hold exactly above 256."""
        if self.fp32 and torch.is_autocast_enabled():
            with torch.autocast("cuda", enabled=False):
                return self.forward(h1.float(), h2.float())
        B = h1.shape[0]
        rope = self._rope(h1.device)
        t1 = h1.view(B, self.K, self.c)
        t2 = h2.view(B, self.K, self.c)
        for layer in self.layers:
            a = torch.cat([t1, t2], 0)
            b = torch.cat([t2, t1], 0)
            if self.training and torch.is_grad_enabled():
                # activation checkpointing: recompute each AC layer in the backward pass
                y = torch.utils.checkpoint.checkpoint(layer, a, b, rope, use_reentrant=False)
            else:
                y = layer(a, b, rope)
            t1, t2 = y[:B], y[B:]
        return (t1 + t2).reshape(B, -1)


# ----------------------------------------------------------------------------- staged multiplication ANO
def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


def st_onehot(logits):
    p = logits.softmax(-1)
    hard = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
    return hard + p - p.detach()


def coupled_fft(x1, x2, shift):
    """y_k = sum_{i + j = k - shift} x1_i * x2_j per channel (linear convolution along places)."""
    K = x1.shape[1]
    n = 2 * K
    y = torch.fft.irfft(torch.fft.rfft(x1.float(), n=n, dim=1) * torch.fft.rfft(x2.float(), n=n, dim=1), n=n, dim=1)
    o = torch.arange(K, device=x1.device) - shift
    return y[:, o.clamp(0, n - 1)] * ((o >= 0) & (o < n - 1)).to(y.dtype)[None, :, None]


class DecimalFourier(nn.Module):
    """cos / sin (w_m x) with learnable frequencies initialised at 2 pi / 10^m (the decimal prior of
    Fourier number embeddings, FoNE): digit m of an integer-valued x is linearly readable from them.
    init="random" (ablation): frequencies log-uniform in [2 pi / 2000, 2 pi / 2]."""

    def __init__(self, n_in, periods=(10, 100, 1000), init="decimal"):
        super().__init__()
        if init == "decimal":
            w = torch.tensor([2 * math.pi / p for p in periods]).repeat(n_in, 1)
        else:
            g = torch.Generator().manual_seed(1234)
            w = 2 * math.pi / torch.exp(torch.empty(n_in, len(periods)).uniform_(math.log(2), math.log(2000), generator=g))
        self.w = nn.Parameter(w)

    def forward(self, x):
        z = x[..., None] * self.w
        return torch.cat([torch.cos(z), torch.sin(z)], -1).flatten(-2)


class PairFront(nn.Module):
    """Front end of the first multiplication AC layer.

    A learned classifier reads the digit and the sign of every slot of both inputs. The
    place-coupled product of the two digit distributions gives, for every output place k, the
    expected number of digit pairs (p, q) with i + j = k (an integer when the classifier is sure).
    A learned linear map turns these counts into the two sums L_k and H_k of the hint algorithm,
    and decimal Fourier features make their digits readable."""

    def __init__(self, c_in, D, shift, hidden=128, soft_counts=False, fourier="decimal", refit=True, planes=3):
        super().__init__()
        # planes: decimal digits of the largest L_k, H_k (numdata.mul_bounds; 3 for N = 25, 4 for N = 200)
        self.shift, self.soft_counts, self.fourier, self.planes = shift, soft_counts, fourier, planes
        self.digit = mlp(c_in, hidden, 10)
        self.sign = mlp(c_in, hidden, 1)
        self.lh = nn.Linear(100, 2, bias=False)
        nn.init.zeros_(self.lh.weight)
        # learned by closed-form least squares (refit_pair_front), not by the optimizer: Adam's
        # normalized steps make an exact solution drift once its gradient is tiny
        # (refit=False, ablation: trained by Adam with its own learning rate)
        self.lh.weight.requires_grad_(not refit)
        self.ff = DecimalFourier(2, periods=tuple(10 ** (p + 1) for p in range(planes)),
                                 init="random" if fourier == "random" else "decimal")
        self.proj = nn.Linear(4 * planes + 2 + 2 + 20, D)

    def forward(self, h1, h2):
        B, K, _ = h1.shape
        la, lb = self.digit(h1).float(), self.digit(h2).float()
        # The pair counts use exact one-hot digits (soft probabilities of the many empty slots would
        # leak into the counts) and carry no gradient: the classifier learns from its own target,
        # the map counts -> (L, H) from its own regression target, and each stage stays clean.
        pa_soft, pb_soft = la.softmax(-1), lb.softmax(-1)
        if self.soft_counts:                    # ablation: probabilities leak into the counts
            pa, pb = pa_soft, pb_soft
        else:
            pa = F.one_hot(la.argmax(-1), 10).float()
            pb = F.one_hot(lb.argmax(-1), 10).float()
        sa, sb = self.sign(h1).float().mean(1), self.sign(h2).float().mean(1)          # (B, 1) logits
        x1 = pa[..., :, None].expand(B, K, 10, 10).reshape(B, K, 100)
        x2 = pb[..., None, :].expand(B, K, 10, 10).reshape(B, K, 100)
        counts = coupled_fft(x1, x2, self.shift)
        if K > 64 and not self.soft_counts:
            # long windows (length study, N > 25): the float32 FFT error grows with the window and the counts
            # (integers that carry no gradient) are rounded; for N = 25 (K = 52) the computation is unchanged
            counts = counts.round()
        LH = self.lh(counts)
        # The map counts -> (L, H) is learned from its own regression target only: gradients from
        # later stages would arrive through cos / sin of large arguments and swamp that signal.
        LHd = LH.detach()
        sg = torch.cat([torch.tanh(sa), torch.tanh(sb)], -1)[:, None].expand(B, K, 2)
        ff = self.ff(LHd) if self.fourier != "none" else torch.zeros(B, K, 4 * self.planes, device=h1.device)
        feat = torch.cat([ff, LHd / 100.0, sg, pa_soft, pb_soft], -1)
        aux = torch.cat([la, lb, LH, sa[:, None].expand(B, K, 1), sb[:, None].expand(B, K, 1)], -1)
        return self.proj(feat), aux


class StageLayer(nn.Module):
    """AC layer f : R^{K c_in} x R^{K c_in} -> R^{K c_out} that also returns its slot tokens (and,
    for the first layer, the outputs of the pair front end), so that auxiliary heads can read the
    intermediate state (hint supervision)."""

    def __init__(self, K, c_in, c_out, D, heads, sub_blocks, front=False, shift=0, periodic_frac=0.25,
                 front_kw=None):
        super().__init__()
        self.K, self.skip = K, c_in == c_out
        self.inp = nn.Linear(c_in, D)
        self.slot = nn.Parameter(torch.randn(K, D) * 0.02)
        self.side = nn.Parameter(torch.randn(2, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, heads, periodic_frac=periodic_frac) for _ in range(sub_blocks)])
        self.norm = RMSNorm(D)
        self.out = nn.Linear(D, c_out)
        if self.skip:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        self.front = PairFront(c_in, D, shift, **(front_kw or {})) if front else None

    def forward(self, h1, h2, rope):
        x1 = self.inp(h1) + self.slot + self.side[0]
        x2 = self.inp(h2) + self.slot + self.side[1]
        aux = None
        if self.front is not None:
            f, aux = self.front(h1, h2)
            x1 = x1 + f
        x = torch.cat([x1, x2], dim=1)
        rp = (torch.cat([rope[0], rope[0]]), torch.cat([rope[1], rope[1]]))
        for blk in self.blocks:
            x = blk(x, rope=rp)
        tok = self.norm(x[:, :self.K])
        y = self.out(tok)
        y = h1 + y if self.skip else y
        return (y, tok, aux) if aux is not None else (y, tok)


class StagedMulANO(nn.Module):
    """Abelian Neural Operator for multiplication with a wider internal state (K slots of c_int
    channels) and one AC layer per stage of the hint algorithm (numdata.mul_hints). The structure
    is the paper's: t1^i = f_i([t1, t2]), t2^i = f_i([t2, t1]), output t1^k + t2^k."""

    def __init__(self, K, c, c_int=32, D=192, heads=6, stages=(2, 2, 2, 2), shift=0, periodic_frac=0.25, fp32=True,
                 front_kw=None):
        super().__init__()
        self.K, self.c, self.fp32, self.D, self.heads = K, c, fp32, D, heads
        n = len(stages)
        dims = [c] + [c_int] * (n - 1) + [c]
        self.layers = nn.ModuleList([
            StageLayer(K, dims[i], dims[i + 1], D, heads, stages[i], front=(i == 0), shift=shift,
                       periodic_frac=periodic_frac, front_kw=front_kw) for i in range(n)])
        self.register_buffer("cos", torch.empty(0), persistent=False)

    def _rope(self, device):
        if self.cos.numel() == 0 or self.cos.device != device:
            self.cos, self.sin = rope_cache(self.K, self.D // self.heads, device)
        return self.cos, self.sin

    def forward(self, h1, h2, return_tokens=False):
        if self.fp32 and torch.is_autocast_enabled():
            with torch.autocast("cuda", enabled=False):
                return self.forward(h1.float(), h2.float(), return_tokens)
        B = h1.shape[0]
        rope = self._rope(h1.device)
        t1, t2 = h1.view(B, self.K, -1), h2.view(B, self.K, -1)
        toks, aux = [], None
        for i, layer in enumerate(self.layers):
            a, b = torch.cat([t1, t2], 0), torch.cat([t2, t1], 0)
            # Layer 0 is never recomputed: its exact one-hot digits (argmax) could flip between the forward
            # pass and a checkpoint recomputation when two digits are nearly tied (e.g. for an input that is
            # itself an operator output), and the backward pass would then mix inconsistent activations.
            if self.training and torch.is_grad_enabled() and i > 0:
                res = torch.utils.checkpoint.checkpoint(layer, a, b, rope, use_reentrant=False)
            else:
                res = layer(a, b, rope)
            if i == 0:
                y, tok, aux = res
            else:
                y, tok = res
            toks.append(tok)                                   # (2B, K, D): both orders
            t1, t2 = y[:B], y[B:]
        out = (t1 + t2).reshape(B, -1)
        return (out, toks, aux) if return_tokens else out


class MulHintHeads(nn.Module):
    """Auxiliary heads that read the stages of StagedMulANO (training only)."""

    def __init__(self, D, n_layers, planes=3, q_classes=4, t_classes=13):
        """planes, q_classes, t_classes: digits of L_k / H_k and the classes for floor(S_k / 10) and T_k
        (numdata.mul_bounds; the defaults are those of N = 25, where larger targets are clamped)."""
        super().__init__()
        self.n_planes, self.q_classes, self.t_classes = planes, q_classes, t_classes
        self.planes = nn.Linear(D, 2 * planes * 10)
        self.rq = nn.Linear(D, 10 + q_classes)
        self.T = nn.Linear(D, t_classes)
        self.final = nn.ModuleList([nn.Linear(D, 10) for _ in range(n_layers - 2)])
        self.sign = nn.Linear(D, 2)

    def loss(self, toks, aux, hints, digits, sign, da, db, sa, sb):
        """toks: list of (2B, K, D); aux: (2B, K, 24) from PairFront; hints from numdata.mul_hints;
        digits / sign of the product; da, db, sa, sb: digits and signs of the operands."""
        rep = lambda x: torch.cat([x, x], 0)                       # both orders have the same targets
        P, RQ, T, Dg, S = rep(hints["planes"]), rep(hints["rq"]), rep(hints["T"]), rep(digits), rep(sign)
        ce = lambda logits, y: F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        # front end: operand digits and signs (order 1: (a, b), order 2: (b, a)), L and H
        first, second = torch.cat([da, db], 0), torch.cat([db, da], 0)
        s1, s2 = torch.cat([sa, sb], 0), torch.cat([sb, sa], 0)
        l = ce(aux[..., :10], first) + ce(aux[..., 10:20], second)
        l = l + F.mse_loss(aux[..., 20:22], rep(hints["LH"]).float()) / 100.0
        l = l + F.binary_cross_entropy_with_logits(aux[:, 0, 22], (s1 > 0).float()) + \
            F.binary_cross_entropy_with_logits(aux[:, 0, 23], (s2 > 0).float())
        l = l + ce(self.planes(toks[0]).view(*toks[0].shape[:2], 2 * self.n_planes, 10), P.clamp(max=9))
        rq = self.rq(toks[1])
        l = l + ce(rq[..., :10], RQ[..., 0]) + ce(rq[..., 10:], RQ[..., 1].clamp(max=self.q_classes - 1))
        l = l + ce(self.T(toks[2]), T.clamp(max=self.t_classes - 1))
        for head, tk in zip(self.final, toks[2:]):
            l = l + ce(head(tk), Dg)
        l = l + F.cross_entropy(self.sign(toks[-1].mean(1)).float(), (S > 0).long())
        return l


def param_groups(model, lr, weight_decay=0.01, lr_lh=0.05, lr_freq=1e-5):
    """AdamW parameter groups. The linear map from pair counts to L and H is a convex regression
    whose weights are integers up to 9: with the default learning rate Adam would need ~10^4
    steps to reach them, so it gets a large learning rate and no weight decay. The decimal Fourier
    frequencies get a tiny learning rate so that the base-10 prior is not washed out."""
    lh, freq, rest = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("lh.weight"):
            lh.append(p)
        elif n.endswith("ff.w"):
            freq.append(p)
        else:
            rest.append(p)
    groups = [{"params": rest, "lr": lr, "weight_decay": weight_decay}]
    if lh:
        groups.append({"params": lh, "lr": lr_lh, "weight_decay": 0.0})
    if freq:
        groups.append({"params": freq, "lr": lr_freq, "weight_decay": 0.0})
    return groups


@torch.no_grad()
def refit_pair_front(ano, h1, h2, LH, opt=None):
    """Closed-form least-squares fit of the linear map pair counts -> (L, H) of the first AC layer of
    a StagedMulANO on a batch (h1, h2) with targets LH (B, K, 2). The map is a convex regression, so
    the least-squares solution is its exact optimum; Adam keeps fine-tuning it between refits."""
    front = ano.layers[0].front
    h1 = h1.float().view(h1.shape[0], ano.K, -1)
    h2 = h2.float().view(h2.shape[0], ano.K, -1)
    B, K, _ = h1.shape
    pa = F.one_hot(front.digit(h1).argmax(-1), 10).float()
    pb = F.one_hot(front.digit(h2).argmax(-1), 10).float()
    x1 = pa[..., :, None].expand(B, K, 10, 10).reshape(B, K, 100)
    x2 = pb[..., None, :].expand(B, K, 10, 10).reshape(B, K, 100)
    X = coupled_fft(x1, x2, front.shift)
    if K > 64:                                            # long windows: exact integer counts (see PairFront)
        X = X.round()
    X = X.reshape(-1, 100).double()
    Y = LH.reshape(-1, 2).double()
    W = torch.linalg.lstsq(X.cpu(), Y.cpu(), driver="gelsd").solution.T          # (2, 100)
    front.lh.weight.copy_(W.to(front.lh.weight))
    if opt is not None and front.lh.weight in opt.state:
        for v in opt.state[front.lh.weight].values():
            if torch.is_tensor(v) and v.dim() > 0:
                v.zero_()
    return (X @ W.T.to(X) - Y).abs().max().item()



class DenseANO(nn.Module):
    """Perceptron AC layers (ablation): k stacked AC layers with f([h1, h2]) = ReLU(W [h1, h2] + b),
    W : R^{2d} -> R^d, output t1^k + t2^k.
    The last layer has no ReLU so that the output can take both signs."""

    def __init__(self, K, c, layers=4, fp32=False):
        super().__init__()
        self.K, self.c, self.fp32 = K, c, fp32
        d = K * c
        self.W = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])

    def forward(self, h1, h2):
        if self.fp32 and torch.is_autocast_enabled():
            with torch.autocast("cuda", enabled=False):
                return self.forward(h1.float(), h2.float())
        t1, t2 = h1, h2
        for i, W in enumerate(self.W):
            a = W(torch.cat([t1, t2], -1))
            b = W(torch.cat([t2, t1], -1))
            if i < len(self.W) - 1:
                a, b = F.relu(a), F.relu(b)
            t1, t2 = a, b
        return t1 + t2
