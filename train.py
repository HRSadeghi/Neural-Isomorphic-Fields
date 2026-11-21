"""Train one NIF model, with complete logs and every checkpoint kept.

    python train.py --name main_s0 --set seed=0 steps=12000 patience=4 min_delta=0.2 min_steps=3000
    python train.py --name aeonly --init-from runs/aeonly_rec/best.pt --set freeze_encoder=true ...

The settings of every run of the paper are in configs/experiments.json (run_experiments.py starts them).

Run folder runs/<name>/:
    config.json        all settings of the run
    metrics.jsonl      every 50 steps: every loss term, gradient norm, learning rate, time, GPU memory
    evals.jsonl        every eval_every steps: validation exact match (overall and by result length) for
                       reconstruction, +, x, order, and the hard cases (cancellation, x * (1/x),
                       distributivity, O1, chains of +), on seeds that the test suite does not use
    ckpt/step_XXXXX.pt full training state (model, optimizer, scheduler, generator) every ckpt_every steps
    last.pt            full state every last_every steps; an interrupted run resumes from it
    final.pt           weights at the last step;  best.pt  weights of the checkpoint with the best validation score
    DONE               written at the end

Losses (Section 3.5 of the paper): l_rec, l_iso, l_ord, the decoder cross-entropy on operator outputs,
compositions, and hint supervision for x. The targeted cases are extra inputs for these same losses:
    cancellation   a + (-a) and a + (-a + delta)                  (long borrow chains, result 0 or small)
    powers of ten  x * y = 10^k with x = +-2^i 5^j and k != 0     (carries through every place; k = 0,
                                                                   i.e. x * (1/x), is left to the test)
    mixed          a * (b + c) and (a * b) + (a * c)              (outputs of + as inputs of x and back)
    order on ops   omega-bar(a + b, a + c)                         (outputs of + as inputs of omega)
    chains of +    ((a + b) + c) + d
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _bootstrap_n_digits():
    """The maximum number of digits N (domain D_N, K = 2N + 2 slots) is fixed per process when numdata is
    imported. It is taken from `--set n_digits=...` or from the run's config.json and passed on through the
    environment variable NIF_N_DIGITS, before numdata is imported below."""
    argv = sys.argv[1:]
    n = None
    if "--name" in argv and argv.index("--name") + 1 < len(argv):
        p = os.path.join(HERE, "runs", argv[argv.index("--name") + 1], "config.json")
        if os.path.exists(p):
            n = json.load(open(p)).get("n_digits")
    for kv in argv:
        if kv.startswith("n_digits="):
            n = int(kv.split("=", 1)[1])
    if n is not None:
        if os.environ.get("NIF_N_DIGITS", str(n)) != str(n):
            raise SystemExit(f"NIF_N_DIGITS={os.environ['NIF_N_DIGITS']} contradicts n_digits={n} of the run")
        os.environ["NIF_N_DIGITS"] = str(n)


_bootstrap_n_digits()

from nif.convergence import convergence_state  # noqa: E402
from nif.layers import param_groups, refit_pair_front  # noqa: E402
from nif.model import NIF  # noqa: E402
from nif.numdata import (N_DIGITS, admissible_pairs, canonical_sign, exact_add, exact_mul, mul_hints,  # noqa: E402
                     num_digits, sample_numbers, strings_to_number)
from nif.train_utils import dec_loss, em_mask, encode_many, rel  # noqa: E402

DEFAULTS = dict(
    steps=6000, batch=64, sub_batch=32, accum=2, lr=5e-4, warmup=500, seed=0,
    w_iso=1.0, w_iso_add=3.0, w_ord=0.5, w_hint=1.0,
    dec_op_loss=True,           # decoder cross-entropy on operator outputs
    chains=True, chain_from=2000,
    targeted=True, p_cancel=0.1, p_near=0.1, pow10_batch=8, mixed_from=1000, cmp_ops=True, chain_add_k=4,
    hints=True, refit=True,
    max_len=None,               # uniform lengths 0..max_len of the integer and of the fractional part (None: N // 2)
    ckpt_every=500, last_every=25, eval_every=500, eval_n=512,
    patience=0, min_delta=0.2, min_steps=0,   # convergence: stop when the validation score stops improving
    freeze_encoder=False,       # keep the encoder fixed (AE-only comparison, second phase)
    uniform_frac=0.5,           # share of training numbers with uniform lengths 0..max_len (ablation: 0)
    n_digits=25,                # N of D_N (length x dimension study: 50, 100, 200; K = 2N + 2 slots)
    eval_uniform=0.0,           # share of uniform-length numbers in the validation sets of rec, + and x
    eval_units=4438,            # pairs (x, 1/x) of the unit-inverse monitor at every evaluation (N = 25: all 4438)
    model={},                   # overrides of the NIF constructor (ablations)
)


# ----------------------------------------------------------------------------- special inputs
def units_pairs(N=N_DIGITS):
    """(x, y) with x = +-2^i 5^j, x, y and x*y = 10^k in D_N, k != 0 (training);  and the test-like
    pairs (x, 1/x) (k = 0), used only for monitoring."""
    from nif.numbers import canonical, num_digits as nd, units_in_Dn
    U = units_in_Dn(N, max_exp=max(40, N + 2))
    if len(U) > 20000:            # long numbers (N = 200: 293k units): a fixed random subset; N = 25 (4438) unchanged
        import random
        U = sorted(random.Random(0).sample(U, 20000))
    train, test = [], []
    for x in U:
        test.append((canonical(x), canonical(1 / x)))
        for k in (-2, -1, 1, 2):
            y = Fraction(10) ** k / x
            if nd(y) <= N and nd(x * y) <= N:
                train.append((canonical(x), canonical(y)))
    if len(test) > 4438:          # the monitor of x * (1/x) at every evaluation: as many pairs as for N = 25
        import random
        test = random.Random(1).sample(test, 4438)
    return train, test


def to_tensors(pairs, dev):
    sa, da = strings_to_number([a for a, _ in pairs], dev)
    sb, db = strings_to_number([b for _, b in pairs], dev)
    return sa, da, sb, db


def mixture(B, dev, g, max_len, frac=0.5):
    nu = int(round(B * frac))
    s0, d0 = sample_numbers(B - nu, dev, g, "paper")
    s1, d1 = sample_numbers(nu, dev, g, "uniform", max_len)
    return torch.cat([s0, s1]), torch.cat([d0, d1])


def ok_N(s, d):
    return num_digits(s, d)[0] <= N_DIGITS


def cancel_pairs(B, dev, g, max_len, near, frac=0.5):
    """(a, -a) or (a, -a + delta): results 0 or small, with borrows through many places."""
    sa, da = mixture(B, dev, g, max_len, frac)
    if not near:
        return sa, da, -sa, da.clone(), torch.ones_like(sa), torch.zeros_like(da)
    sd, dd = sample_numbers(B, dev, g, "paper")
    sb, db, ovf = exact_add(-sa, da, sd, dd)
    keep = ~ovf & ok_N(sb, db)
    return sa[keep], da[keep], sb[keep], db[keep], sd[keep], dd[keep]


def distrib_triples(B, dev, g, max_len, frac=0.5):
    sb, db, sc, dc, sbc, dbc = admissible_pairs(B, "+", dev, g, max_len=max_len, p_uniform=frac)
    sa, da = mixture(B, dev, g, max_len, frac)
    s1, d1, o1 = exact_mul(sa, da, sb, db)
    s2, d2, o2 = exact_mul(sa, da, sc, dc)
    s3, d3, o3 = exact_mul(sa, da, sbc, dbc)
    keep = ~(o1 | o2 | o3) & ok_N(s1, d1) & ok_N(s2, d2) & ok_N(s3, d3) & ok_N(sa, da)
    return [x[keep] for x in (sa, da, sb, db, sc, dc, sbc, dbc, s1, d1, s2, d2, s3, d3)]


def o1_triples(B, dev, g):
    sa, da = sample_numbers(B, dev, g, "paper")
    sb, db = sample_numbers(B, dev, g, "paper")
    sc, dc = sample_numbers(B, dev, g, "paper")
    s1, d1, o1 = exact_add(sa, da, sb, db)
    s2, d2, o2 = exact_add(sa, da, sc, dc)
    keep = ~(o1 | o2) & ok_N(s1, d1) & ok_N(s2, d2)
    return [x[keep] for x in (sa, da, sb, db, sc, dc, s1, d1, s2, d2)]


def add_chain(B, K, dev, g):
    """Operands x_1..x_K with all partial sums in D_N, and the partial sums."""
    xs = [sample_numbers(B, dev, g, "paper") for _ in range(K)]
    s, d = xs[0]
    keep = torch.ones(B, dtype=torch.bool, device=dev)
    for sx, dx in xs[1:]:
        s, d, ovf = exact_add(s, d, sx, dx)
        keep &= ~ovf & ok_N(s, d)
    return [(sx[keep], dx[keep]) for sx, dx in xs], (s[keep], d[keep])


# ----------------------------------------------------------------------------- evaluation
LEN_BINS = [(i * N_DIGITS // 5 + 1, (i + 1) * N_DIGITS // 5) for i in range(5)]   # N = 25: 1-5, ..., 21-25


def by_len(ok, s, d):
    L = num_digits(s, d)[0]
    return {f"{lo}-{hi}": round(100 * ok[(L >= lo) & (L <= hi)].float().mean().item(), 2)
            for lo, hi in LEN_BINS if ((L >= lo) & (L <= hi)).any()}


@torch.no_grad()
def evaluate_run(model, dev, n=512, seed=777, units_test=None, chunk=256, max_len=12, eval_uniform=0.0):
    """Validation metrics (seeds differ from those of the tests). Everything length-free. With eval_uniform > 0
    (length study) that share of the numbers of rec, + and x has uniform lengths 0..max_len per part, so that
    model selection and the convergence criterion also see long numbers; 0 reproduces the main experiments."""
    model.eval()
    g = torch.Generator(device=dev).manual_seed(seed)
    out = {}
    pct = lambda ok: round(100 * ok.float().mean().item(), 2)

    def run(fn_z, s, d):
        oks = []
        for i in range(0, s.shape[0], chunk):
            oks.append(em_mask(model, fn_z(i, i + chunk), s[i:i + chunk], d[i:i + chunk]))
        return torch.cat(oks) if oks else torch.zeros(0, dtype=torch.bool, device=dev)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        s, d = sample_numbers(n, dev, g, "paper") if not eval_uniform else mixture(n, dev, g, max_len, eval_uniform)
        ok = run(lambda i, j: encode_many(model, [(s[i:j], d[i:j])])[0], s, d)
        out["rec"] = pct(ok)
        if eval_uniform:
            out["rec_by_len"] = by_len(ok, s, d)
        for op, fn in (("+", model.add_op), ("x", model.mul_op)):
            sa, da, sb, db, sr, dr = admissible_pairs(n, op, dev, g, p_uniform=eval_uniform, max_len=max_len)
            ok = run(lambda i, j: fn(*encode_many(model, [(sa[i:j], da[i:j]), (sb[i:j], db[i:j])])), sr, dr)
            out[op] = pct(ok)
            out[op + "_by_len"] = by_len(ok, sr, dr)
        # order
        sa, da = sample_numbers(n, dev, g, "paper")
        sb, db = sample_numbers(n, dev, g, "paper")
        eq = torch.arange(n, device=dev) % 10 == 0
        sb, db = torch.where(eq, sa, sb), torch.where(eq[:, None], da, db)
        hits = []
        for i in range(0, n, chunk):
            za, zb = encode_many(model, [(sa[i:i + chunk], da[i:i + chunk]), (sb[i:i + chunk], db[i:i + chunk])])
            hits.append(model.comparator.bar_log_probs(za, zb).argmax(-1) ==
                        rel(sa[i:i + chunk], da[i:i + chunk], sb[i:i + chunk], db[i:i + chunk]))
        out["order"] = pct(torch.cat(hits))
        # hard cases
        sa, da, sb, db, sr, dr = cancel_pairs(n // 2, dev, g, max_len, near=False)
        ok = run(lambda i, j: model.add_op(*encode_many(model, [(sa[i:j], da[i:j]), (sb[i:j], db[i:j])])), sr, dr)
        out["cancel"] = pct(ok)
        sa, da, sb, db, sr, dr = cancel_pairs(n // 2, dev, g, max_len, near=True)
        ok = run(lambda i, j: model.add_op(*encode_many(model, [(sa[i:j], da[i:j]), (sb[i:j], db[i:j])])), sr, dr)
        out["near_cancel"] = pct(ok)
        if units_test is not None:
            sa, da, sb, db = units_test
            sr, dr, _ = exact_mul(sa, da, sb, db)
            ok = run(lambda i, j: model.mul_op(*encode_many(model, [(sa[i:j], da[i:j]), (sb[i:j], db[i:j])])), sr, dr)
            out["unit_inverse"] = pct(ok)
        T = distrib_triples(n // 2, dev, g, max_len)
        sa, da, sb, db, sc, dc, sbc, dbc, s1, d1, s2, d2, s3, d3 = T

        def lhs(i, j):
            za, zb, zc = encode_many(model, [(sa[i:j], da[i:j]), (sb[i:j], db[i:j]), (sc[i:j], dc[i:j])])
            return model.mul_op(za, model.add_op(zb, zc))

        def rhs(i, j):
            za, zb, zc = encode_many(model, [(sa[i:j], da[i:j]), (sb[i:j], db[i:j]), (sc[i:j], dc[i:j])])
            return model.add_op(model.mul_op(za, zb), model.mul_op(za, zc))
        out["distrib"] = pct(run(lhs, s3, d3) & run(rhs, s3, d3))
        sa, da, sb, db, sc, dc, s1, d1, s2, d2 = o1_triples(n, dev, g)
        hits = []
        for i in range(0, sa.shape[0], chunk):
            sl = slice(i, i + chunk)
            za, zb, zc = encode_many(model, [(sa[sl], da[sl]), (sb[sl], db[sl]), (sc[sl], dc[sl])])
            pred = model.comparator.bar_log_probs(model.add_op(za, zb), model.add_op(za, zc)).argmax(-1)
            hits.append(pred == rel(s1[sl], d1[sl], s2[sl], d2[sl]))
        out["O1"] = pct(torch.cat(hits))
        xs, (sk, dk) = add_chain(n // 2, 4, dev, g)

        def chain(i, j):
            zs = encode_many(model, [(sx[i:j], dx[i:j]) for sx, dx in xs])
            z = zs[0]
            for zz in zs[1:]:
                z = model.add_op(z, zz)
            return z
        out["chain+4"] = pct(run(chain, sk, dk))
    model.train()
    return out


def val_score(ev):
    """Model selection: mean of the validation metrics (the x * (1/x) monitor is excluded, since its
    pairs are the test pairs)."""
    keys = ["rec", "+", "x", "order", "cancel", "near_cancel", "distrib", "O1", "chain+4"]
    return sum(ev[k] for k in keys if k in ev) / len(keys)


# ----------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (JSON values)")
    ap.add_argument("--init-from", default=None, help="start from the weights of this checkpoint")
    a = ap.parse_args()
    run = os.path.join(HERE, "runs", a.name)
    os.makedirs(os.path.join(run, "ckpt"), exist_ok=True)
    cfg = json.loads(json.dumps(DEFAULTS))
    cfg_path = os.path.join(run, "config.json")
    if os.path.exists(cfg_path):
        cfg.update(json.load(open(cfg_path)))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[k] = v
    if a.init_from:
        cfg["init_from"] = a.init_from
    if cfg.get("max_len") is None:
        cfg["max_len"] = cfg["n_digits"] // 2          # N = 25: 12, as in the main experiments
    if cfg["n_digits"] != N_DIGITS:
        raise SystemExit(f"n_digits={cfg['n_digits']} but numdata was imported with N={N_DIGITS}")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    c = argparse.Namespace(**cfg)
    if os.path.exists(os.path.join(run, "DONE")):
        print("DONE (already finished)", flush=True)
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = os.environ.get("NIF_DEVICE", "cuda")      # NIF_DEVICE=cpu: smoke tests of the training loop only
    torch.manual_seed(c.seed)
    model_kw = dict(c.model)
    if not c.refit:
        model_kw["mul_front"] = dict(model_kw.get("mul_front") or {}, refit=False)
    model = NIF(**model_kw).to(dev)
    staged = model.mul_kind == "staged"
    if getattr(c, "freeze_encoder", False):       # comparison protocol: a fixed (pre-trained) encoder
        for p_ in model.encoder.parameters():
            p_.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model, c.lr), betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / c.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / c.steps)))))
    g = torch.Generator(device=dev).manual_seed(c.seed)
    step = 0
    last = os.path.join(run, "last.pt")
    if os.path.exists(last):
        st = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        g.set_state(st["g"].cpu())
        step = st["step"]
        print(f"resumed from step {step}", flush=True)
    elif getattr(c, "init_from", None):
        src = c.init_from if os.path.isabs(c.init_from) else os.path.join(HERE, c.init_from)
        if not os.path.exists(src) and c.init_from.replace("\\", "/").startswith("runs/"):
            src = os.path.join(HERE, "checkpoints", *c.init_from.replace("\\", "/").split("/")[1:])
        st = torch.load(src, map_location=dev, weights_only=False)
        missing, unexpected = model.load_state_dict(st["model"], strict=False)
        print(f"initialised from {c.init_from} (missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
    print("parameters:", {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children()}, flush=True)

    utrain, utest = units_pairs()
    U_tr = to_tensors(utrain, dev)
    U_te = to_tensors(utest[:c.eval_units], dev)
    print(f"powers-of-ten training pairs: {len(utrain)}; unit-inverse monitor pairs: {U_te[0].shape[0]} of {len(utest)}",
          flush=True)

    def full_state():
        return {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "g": g.get_state(), "step": step, "cfg": model.cfg, "run_cfg": cfg}

    def save(path):
        torch.save(full_state(), path + ".tmp")
        os.replace(path + ".tmp", path)

    def log(fname, rec):
        with open(os.path.join(run, fname), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def after_step():
        if step % c.last_every == 0:
            save(last)
        if step % c.ckpt_every == 0:
            save(os.path.join(run, "ckpt", f"step_{step:05d}.pt"))

    t0 = time.time()
    B, Bs = c.batch, c.sub_batch
    evaluated = {json.loads(l)["step"] for l in open(os.path.join(run, "evals.jsonl"))} \
        if os.path.exists(os.path.join(run, "evals.jsonl")) else set()
    stop_reason = f"step cap {c.steps} reached"
    if os.path.exists(os.path.join(run, "evals.jsonl")):      # a resume right after the stopping evaluation
        _ev = [json.loads(l) for l in open(os.path.join(run, "evals.jsonl"))]
        if _ev and _ev[-1]["step"] == step and _ev[-1].get("convergence", {}).get("stop"):
            stop_reason = f"converged at step {step}: {_ev[-1]['convergence']['reason']}"
            step = c.steps + 1
    while step <= c.steps:
        if (step % c.eval_every == 0 or step == c.steps) and step not in evaluated:
            te = time.time()
            ev = evaluate_run(model, dev, n=c.eval_n, units_test=U_te, max_len=c.max_len, eval_uniform=c.eval_uniform)
            ev.update(step=step, score=round(val_score(ev), 3), eval_seconds=round(time.time() - te, 1),
                      wall=round(time.time() - t0, 1))
            prev = [json.loads(l) for l in open(os.path.join(run, "evals.jsonl"))] \
                if os.path.exists(os.path.join(run, "evals.jsonl")) else []
            stop, conv = convergence_state(prev + [ev], "score", c.min_delta, c.patience, c.min_steps)
            ev["convergence"] = conv
            log("evals.jsonl", ev)
            evaluated.add(step)
            print(f"[eval] step {step}: " + json.dumps(ev), flush=True)
            if stop:
                stop_reason = f"converged at step {step}: {conv['reason']}"
                print("[convergence] " + stop_reason, flush=True)
                break
        if step == c.steps:
            break
        opt.zero_grad(set_to_none=True)
        for micro in range(c.accum):                  # gradient accumulation keeps peak memory low
            logs = {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # ---------------------------------------------------------------- inputs of this step
                sn, dn = mixture(B, dev, g, c.max_len, c.uniform_frac)
                P = list(admissible_pairs(B, "+", dev, g, max_len=c.max_len, p_uniform=c.uniform_frac))
                if c.targeted:                      # exact and near cancellations among the + pairs
                    nc, nn_ = int(B * c.p_cancel), int(B * c.p_near)
                    C0 = cancel_pairs(nc, dev, g, c.max_len, near=False, frac=c.uniform_frac)
                    C1 = cancel_pairs(nn_, dev, g, c.max_len, near=True, frac=c.uniform_frac)
                    P = [torch.cat([x[:B - nc - nn_], y, z]) for x, y, z in zip(P, C0, C1)]
                M = list(admissible_pairs(B, "x", dev, g, max_len=c.max_len, p_uniform=c.uniform_frac))
                if c.targeted and c.pow10_batch:   # products that are powers of ten (k != 0)
                    idx = torch.randint(0, U_tr[0].shape[0], (c.pow10_batch,), device=dev, generator=g)
                    sa_, da_, sb_, db_ = (t[idx] for t in U_tr)
                    sr_, dr_, _ = exact_mul(sa_, da_, sb_, db_)
                    M = [torch.cat([x[:B - c.pow10_batch], y]) for x, y in zip(M, (sa_, da_, sb_, db_, sr_, dr_))]
                sa, da = sample_numbers(B, dev, g, "paper")
                sb, db = sample_numbers(B, dev, g, "paper")
                eq = torch.rand(B, device=dev, generator=g) < 0.1
                sb, db = torch.where(eq, sa, sb), torch.where(eq[:, None], da, db)
                nums = [(sn, dn), (P[0], P[1]), (P[2], P[3]), (P[4], P[5]), (M[0], M[1]), (M[2], M[3]), (M[4], M[5]),
                        (sa, da), (sb, db)]
                use_chain = c.chains and step >= c.chain_from
                use_mixed = c.targeted and step >= c.mixed_from
                extra = {}
                if use_chain:
                    for op in ("+", "x"):
                        A_ = list(admissible_pairs(Bs, op, dev, g, p_uniform=c.uniform_frac))
                        sc_, dc_ = sample_numbers(Bs, dev, g, "paper")
                        s2_, d2_, ov = (exact_add if op == "+" else exact_mul)(A_[4], A_[5], sc_, dc_)
                        k_ = ~ov & ok_N(s2_, d2_)
                        extra["chain" + op] = len(nums)
                        nums += [(A_[0][k_], A_[1][k_]), (A_[2][k_], A_[3][k_]), (sc_[k_], dc_[k_]), (s2_[k_], d2_[k_])]
                    if c.targeted and c.chain_add_k > 3:
                        xs, (sk, dk) = add_chain(Bs, c.chain_add_k, dev, g)
                        extra["chain+K"] = len(nums)
                        nums += xs + [(sk, dk)]
                if use_mixed:
                    T = distrib_triples(Bs, dev, g, c.max_len, c.uniform_frac)
                    extra["distrib"] = len(nums)
                    nums += [(T[0], T[1]), (T[2], T[3]), (T[4], T[5]), (T[12], T[13])]
                    if c.cmp_ops:
                        O = o1_triples(Bs, dev, g)
                        extra["o1"] = len(nums)
                        nums += [(O[0], O[1]), (O[2], O[3]), (O[4], O[5])]
                Z = encode_many(model, nums)
                if staged and c.refit and step > 0 and step % (100 if step <= 1000 else 500) == 0:
                    res = refit_pair_front(model.mul_op, Z[4].detach(), Z[5].detach(), mul_hints(M[1], M[3])["LH"], opt)
                    logs["refit_residual"] = res
                # ---------------------------------------------------------------- losses
                ce, lce = dec_loss(model, Z[0], sn, dn)
                loss = ce + lce
                terms = {"rec": ce + lce}
                logs.update(rec=ce.item(), len=lce.item())

                def op_terms(name, out, zr, sr, dr, w=1.0):
                    nonlocal loss
                    w_iso = c.w_iso_add if name.startswith("+") else c.w_iso
                    iso = F.mse_loss(out.float(), zr.float().detach())
                    term = w_iso * iso
                    logs["iso_" + name] = iso.item()
                    if c.dec_op_loss:
                        dce, dlce = dec_loss(model, out, sr, dr)
                        term = term + dce + dlce
                        logs["dec_" + name] = dce.item()
                    loss = loss + w * term
                    terms[name] = w * term

                def mul_with_hints(name, za, zb, sa_, da_, sb_, db_, sr_, dr_):
                    nonlocal loss
                    if staged and c.hints:
                        out, toks, aux = model.mul_op(za, zb, return_tokens=True)
                        hint = model.mul_hints.loss(toks, aux, mul_hints(da_, db_), dr_, canonical_sign(sr_, dr_),
                                                    da_, db_, canonical_sign(sa_, da_), canonical_sign(sb_, db_))
                        loss = loss + c.w_hint * hint
                        terms["hint_" + name] = c.w_hint * hint
                        logs["hint_" + name] = hint.item()
                        return out
                    return model.mul_op(za, zb)

                op_terms("+", model.add_op(Z[1], Z[2]), Z[3], P[4], P[5])
                op_terms("x", mul_with_hints("x", Z[4], Z[5], *M), Z[6], M[4], M[5])
                lp = model.comparator.bar_log_probs(Z[7], Z[8])
                ordl = F.nll_loss(lp, rel(sa, da, sb, db))
                loss = loss + c.w_ord * ordl
                terms["ord"] = c.w_ord * ordl
                logs["ord"] = ordl.item()
                if use_chain:
                    for op in ("+", "x"):
                        i = extra["chain" + op]
                        fn = model.add_op if op == "+" else model.mul_op
                        z2 = fn(fn(Z[i], Z[i + 1]), Z[i + 2])
                        op_terms("chain" + op, z2, Z[i + 3], *nums[i + 3], w=0.5)
                    if "chain+K" in extra:
                        i = extra["chain+K"]
                        z = Z[i]
                        for j in range(1, c.chain_add_k):
                            z = model.add_op(z, Z[i + j])
                        op_terms("chain+K", z, Z[i + c.chain_add_k], *nums[i + c.chain_add_k], w=0.5)
                if use_mixed:
                    i = extra["distrib"]
                    za, zb, zc, zr = Z[i:i + 4]
                    lhs = mul_with_hints("dist", za, model.add_op(zb, zc), T[0], T[1], T[6], T[7], T[12], T[13])
                    rhs = model.add_op(model.mul_op(za, zb), model.mul_op(za, zc))
                    op_terms("dist_l", lhs, zr, T[12], T[13], w=0.5)
                    op_terms("dist_r", rhs, zr, T[12], T[13], w=0.5)
                    if "o1" in extra:
                        i = extra["o1"]
                        za, zb, zc = Z[i:i + 3]
                        lp = model.comparator.bar_log_probs(model.add_op(za, zb), model.add_op(za, zc))
                        o1 = F.nll_loss(lp, rel(O[6], O[7], O[8], O[9]))
                        loss = loss + c.w_ord * o1
                        terms["ord_ops"] = c.w_ord * o1
                        logs["ord_ops"] = o1.item()
            (loss / c.accum).backward()
        grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
        gn = torch.stack([gr.detach().double().norm() for _, gr in grads]).norm()
        scale = min(1.0, 1.0 / (gn.item() + 1e-6))
        if scale < 1.0:
            for _, gr in grads:
                gr.mul_(scale)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            rec = {"step": step, "loss": round(loss.item(), 5), "gn": round(gn.item(), 3), "lr": sched.get_last_lr()[0],
                   "wall": round(time.time() - t0, 1), "mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
            rec.update({k: round(v, 5) for k, v in logs.items()})
            log("metrics.jsonl", rec)
            print(f"step {step} loss {loss.item():.4f} " + " ".join(f"{k} {v:.4f}" for k, v in logs.items())
                  + f" gn {gn.item():.2f} ({time.time() - t0:.0f}s)", flush=True)
        after_step()
    save(last)
    torch.save({"model": model.state_dict(), "cfg": model.cfg, "step": step}, os.path.join(run, "final.pt"))
    evs = [json.loads(l) for l in open(os.path.join(run, "evals.jsonl"))]
    best = max((e for e in evs if os.path.exists(os.path.join(run, "ckpt", f"step_{e['step']:05d}.pt")) or e["step"] == step),
               key=lambda e: e["score"])
    src = os.path.join(run, "ckpt", f"step_{best['step']:05d}.pt")
    sd = torch.load(src, map_location="cpu", weights_only=False)["model"] if os.path.exists(src) else model.state_dict()
    torch.save({"model": sd, "cfg": model.cfg, "step": best["step"], "score": best["score"]}, os.path.join(run, "best.pt"))
    print(f"best checkpoint: step {best['step']} (validation score {best['score']})", flush=True)
    open(os.path.join(run, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + stop_reason + "\n")
    print("DONE", stop_reason, flush=True)


if __name__ == "__main__":
    main()
