"""One run of the reading experiment (Section 5.3.5, Table 6). A small Transformer (rotary positions, trained from
scratch on the CPU) adds and subtracts non-negative integers written as digit tokens; the training operands have at
most 12 digits, the test goes to 20.

conditions
  text_lsb      digit tokens only; the answer is written from the units digit
  read_nif      text_lsb + place-coupled reading: the position that writes the digit of place p receives the slots p
                and p-1 of the two operand vectors through maps shared by all places (no weight depends on the place)
  read_fone     the same with FoNE features computed with exact digit arithmetic
  read_f64fone  the same with FoNE features computed from the number as a float64
  read_onehot   the same with the one-hot digit of every place (no encoder)

    python reading/experiment.py <condition> <steps> <seed> [threads] [training problems] [vocabulary size]

run.py holds the arguments of the runs of the paper and writes the logs to reading/logs/; make_table.py builds the
table from the last line of each log. The vocabulary size (default 16) only changes the random initialization.
"""
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import torch
import torch.nn as nn
import torch.nn.functional as F

from nif.numdata import E_MIN, K
from nif.paths import checkpoint
from vectors import FoNEVectors, NIFVectors

cond, steps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
torch.set_num_threads(threads)
torch.manual_seed(seed)
assert cond in ("text_lsb", "read_nif", "read_fone", "read_f64fone", "read_onehot"), cond
PLUS, MINUS, EQ, BOS, EOS, PAD = 10, 11, 12, 13, 14, 15
V = int(sys.argv[6]) if len(sys.argv) > 6 else 16          # number of tokens
assert V >= 16
D_MODEL, LAYERS, HEADS, BATCH, TRAIN_MAX = 192, 4, 4, 128, 12
lsb = cond == "text_lsb" or cond.startswith("read_")
kind = cond.split("_")[0]
enc = None


class FoNE64:
    """FoNE with the numerics of the released single-token method: the number is a float64 and the features are
    cos / sin of 2 pi (x mod 10^(e+1)) / 10^(e+1), computed in float64 (exact only up to about 15 digits)."""
    dim = 2 * K + 1

    def encode(self, strings):
        x = torch.tensor([float(s) for s in strings], dtype=torch.float64)
        T = 10.0 ** (torch.arange(K, dtype=torch.float64) + E_MIN + 1)
        ang = 2 * math.pi * torch.fmod(x[:, None], T[None]) / T[None]
        feat = torch.stack([ang.cos(), ang.sin()], -1).reshape(len(strings), -1)
        return torch.cat([feat, torch.ones(len(strings), 1, dtype=torch.float64)], 1).float()


class OneHot:
    """Control: the digit of every place as a one-hot vector (no learned or designed code)."""
    c, dim = 10, 10 * K

    def encode(self, strings):
        out = torch.zeros(len(strings), K, 10)
        out[:, :, 0] = 1.0
        for i, s in enumerate(strings):
            for e, ch in enumerate(reversed(s)):
                out[i, K0 + e] = 0.0
                out[i, K0 + e, int(ch)] = 1.0
        return out.reshape(len(strings), -1)


K0 = -E_MIN
if kind == "read":
    enc = FoNE64() if cond.endswith("f64fone") else OneHot() if cond.endswith("onehot") else \
        FoNEVectors() if cond.endswith("fone") else NIFVectors(checkpoint("main_s0"))
    C = 2 if cond.endswith("fone") else enc.c
K0 = -E_MIN                                               # slot of the units digit


def sample(rng, size):
    a = str(rng.randint(10 ** (size - 1), 10 ** size - 1))
    n = rng.randint(1, size)
    b = str(rng.randint(10 ** (n - 1), 10 ** n - 1)) if n > 1 else str(rng.randint(0, 9))
    if rng.random() < 0.5:
        a, b = b, a
    op = rng.choice("+-")
    if op == "-" and int(a) < int(b):
        a, b = b, a
    c = str(int(a) + int(b) if op == "+" else int(a) - int(b))
    return a, op, b, c


def batchify(items):
    rows, eq, pa, pb = [], [], [], []
    for a, op, b, c in items:
        r = [BOS] + [int(x) for x in a] + [{"+": PLUS, "-": MINUS}[op]] + [int(x) for x in b] + [EQ]
        pa.append(len(a)); pb.append(len(a) + 1 + len(b)); eq.append(len(r) - 1)
        rows.append(r + [int(x) for x in (c[::-1] if lsb else c)] + [EOS])
    T = max(len(r) for r in rows)
    ids = torch.tensor([r + [PAD] * (T - len(r)) for r in rows])
    return ids, torch.tensor(eq), torch.tensor(pa), torch.tensor(pb)


def encode_all(items):
    """The operand vectors of a list of problems, computed once."""
    if enc is None:
        return None
    out = []
    with torch.no_grad():
        for s0 in range(0, len(items), 500):
            part = items[s0:s0 + 500]
            out.append(enc.encode([x[0] for x in part] + [x[2] for x in part]).float().view(2, len(part), -1))
    return torch.cat(out, 1)                               # (2, n, dim)


def vectors(e):
    B = e.shape[1]
    return e[0], e[1], e[0][:, :K * C].reshape(B, K, C), e[1][:, :K * C].reshape(B, K, C)


def rope(T, dim):
    f = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    ang = torch.arange(T).float()[:, None] * f[None]
    return ang.cos(), ang.sin()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.qkv, self.o = nn.Linear(D_MODEL, 3 * D_MODEL), nn.Linear(D_MODEL, D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x, cs):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, HEADS, -1).permute(2, 0, 3, 1, 4)
        cos, sin = cs

        def rot(t):
            t1, t2 = t[..., 0::2], t[..., 1::2]
            return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], -1).flatten(-2)
        a = F.scaled_dot_product_attention(rot(q), rot(k), v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
        return x + self.mlp(self.n2(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm, self.head = nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, V)
        if kind == "read":
            self.ra = nn.ModuleList([nn.Linear(C, D_MODEL) for _ in range(2)])        # first operand: places p, p-1
            self.rb = nn.ModuleList([nn.Linear(C, D_MODEL) for _ in range(2)])        # second operand
            self.flag = nn.Embedding(4, D_MODEL)                                       # is place p above each operand?

    def forward(self, ids, eq, pa, pb, extra):
        B, T = ids.shape
        x = self.emb(ids)
        cs = rope(T, D_MODEL // HEADS)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cs)
            if i == 0 and kind == "read":
                sa, sb = extra[2], extra[3]                                            # (B, K, C)
                p = torch.arange(T)[None] - eq[:, None]                                # place written after position t
                on = (p >= 0) & (K0 + p < K)
                add = 0
                for j in range(2):
                    idx = (K0 + p - j).clamp(0, K - 1)
                    ok = (on & (p - j >= 0)).unsqueeze(-1).float()
                    add = add + ok * (self.ra[j](sa.gather(1, idx[..., None].expand(-1, -1, C)))
                                      + self.rb[j](sb.gather(1, idx[..., None].expand(-1, -1, C))))
                la, lb = pa[:, None], (pb - pa - 1)[:, None]
                fl = ((p >= la).long() * 2 + (p >= lb).long())
                x = x + add + on.unsqueeze(-1).float() * self.flag(fl)
        return self.head(self.norm(x))


def loss_fn(net, items, e):
    ids, eq, pa, pb = batchify(items)
    extra = vectors(e) if enc is not None else None
    logits = net(ids[:, :-1], eq, pa, pb, extra)
    tgt = ids[:, 1:].clone()
    pos = torch.arange(tgt.shape[1])[None]
    tgt[(pos < eq[:, None]) | (tgt == PAD)] = -100
    return F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=-100)


@torch.no_grad()
def accuracy(net, items, e):
    ok = 0
    for s in range(0, len(items), 100):
        part = items[s:s + 100]
        ids, eq, pa, pb = batchify(part)
        extra = vectors(e[:, s:s + 100]) if enc is not None else None
        T0 = int(eq.max()) + 1
        cur = torch.full((len(part), T0 + 24), PAD)
        for b in range(len(part)):                         # left part of every row up to '=' (rows differ in length)
            cur[b, :eq[b] + 1] = ids[b, :eq[b] + 1]
        n = eq.clone() + 1
        done = torch.zeros(len(part), dtype=torch.bool)
        r = torch.arange(len(part))
        for _ in range(23):
            logits = net(cur[:, :int(n.max())], eq, pa, pb, extra)
            nxt = logits[r, n - 1].argmax(-1)
            cur[r[~done], n[~done]] = nxt[~done]
            done |= nxt == EOS
            n = n + (~done).long()
            if done.all():
                break
        for b, (a, op, bb, c) in enumerate(part):
            out = cur[b, eq[b] + 1:].tolist()
            out = out[:out.index(EOS)] if EOS in out else out
            s_out = "".join(str(v) for v in out if v < 10)
            ok += (s_out[::-1] if lsb else s_out) == c
    return 100.0 * ok / len(items)


net = Net()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0, betas=(0.9, 0.99))
N_TRAIN = int(sys.argv[5]) if len(sys.argv) > 5 else 60000
rng = random.Random(1234)                                  # the same problems for every condition
train = [sample(rng, rng.randint(1, TRAIN_MAX)) for _ in range(N_TRAIN)]
test_rng = random.Random(999)
tests = {s: [sample(test_rng, s) for _ in range(200)] for s in (4, 8, 12, 14, 16, 20)}
t0 = time.time()
os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
cache = os.path.join(HERE, "cache", f"vec_{cond.split('_')[-1]}_{N_TRAIN}.pt")
if enc is not None and os.path.exists(cache):
    etrain, etest = torch.load(cache)
else:
    etrain = encode_all(train)
    etest = {s: encode_all(tests[s]) for s in tests}
    if enc is not None:
        torch.save((etrain, etest), cache)
print(f"{cond}: vectors of {N_TRAIN} problems in {(time.time() - t0) / 60:.1f} min", flush=True)
order = random.Random(seed)
perm, ptr = [], 0
t0 = time.time()
for step in range(1, steps + 1):
    lr = 1e-3 * min(1.0, step / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
    for g in opt.param_groups:
        g["lr"] = lr
    if ptr + BATCH > len(perm):
        perm, ptr = list(range(N_TRAIN)), 0
        order.shuffle(perm)
    idx = perm[ptr:ptr + BATCH]
    ptr += BATCH
    loss = loss_fn(net, [train[i] for i in idx], etrain[:, idx] if etrain is not None else None)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    if step % 1000 == 0 or step == steps:
        net.eval()
        n = 100 if step < steps else 200
        acc = {s: accuracy(net, tests[s][:n], etest[s][:, :n] if etest[s] is not None else None) for s in tests}
        net.train()
        print(json.dumps({"cond": cond, "seed": seed, "vocab": V, "threads": threads, "step": step, "loss": round(loss.item(), 4), "acc": acc,
                          "min": round((time.time() - t0) / 60, 1)}), flush=True)
