"""Run the tests of the paper on a trained model, in two decoding modes:

  * length-free  : the length head of the decoder chooses the length of the output
  * oracle length: the decoder is given the length of the target

    python evaluate.py --run main_s0                       # checkpoints/main_s0/best.pt -> runs/main_s0/test_results.json
    python evaluate.py --ckpt path/to/best.pt --out results.json

Besides the tests of Table 3 it computes the results by operand length, the protocol of the comparison
with other embeddings (N = 10, 20, 30, Table 5) and, in the length-free mode, the quantities of
Section 3.6 (defects, estimated decoding radii, empirical Lipschitz constants). The 0/1 outcome of every
test item is written to <out>.items.json.gz (McNemar's test between models). Results are saved after
every test, and an interrupted evaluation continues where it stopped.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

from nif.model import NIF  # noqa: E402
from nif.paths import RUNS, checkpoint  # noqa: E402

GEOMETRY_R = 6.0      # perturbation norm delta_0 for the radius estimates (||phi(a)|| = sqrt(520) = 22.8)


def load(path, dev="cuda"):
    st = torch.load(path, map_location=dev, weights_only=False)
    model = NIF(**st["cfg"]).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    return model


LENGTH_BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25)]


def number_with_digits(rng, l):
    """A random element of D whose canonical string has exactly l digits (random split and sign)."""
    li = int(rng.integers(1, l + 1))
    lf = l - li
    ip = [int(rng.integers(0 if li == 1 else 1, 10))] + [int(v) for v in rng.integers(0, 10, size=li - 1)]
    fp = [int(v) for v in rng.integers(0, 10, size=lf)]
    if lf:
        fp[-1] = int(rng.integers(1, 10))
    x = Fraction(int("".join(map(str, ip)))) + (Fraction(int("".join(map(str, fp))), 10 ** lf) if lf else 0)
    return -x if rng.random() < 0.5 else x


def length_tests(T, m, n_per_bin, seed, max_digits=25):
    """Exact match by the number of digits of the (longer) operand: reconstruction, closure of + and x.
    The longer operand has a uniformly chosen length in the bin; the other one comes from the paper's
    sampler and is at most as long; pairs must be admissible."""
    from nif.numbers import NumberSampler, in_Dn, num_digits
    rng = np.random.default_rng(seed)
    other = NumberSampler(max_digits=max_digits, seed=seed + 1)
    out = {"bins": [f"{lo}-{hi}" for lo, hi in LENGTH_BINS]}
    rec, items = [], {}
    for lo, hi in LENGTH_BINS:
        xs = [number_with_digits(rng, int(rng.integers(lo, hi + 1))) for _ in range(n_per_bin)]
        r = T.reconstruction(m, xs)
        rec.append({"n": r["n"], "em": r["em"]})
        items[f"reconstruction_{lo}-{hi}"] = r["per_item"]
    out["reconstruction"] = rec
    for op in "+x":
        rows = []
        for lo, hi in LENGTH_BINS:
            pairs, tries = [], 0
            while len(pairs) < n_per_bin and tries < 200 * n_per_bin:
                tries += 1
                a = number_with_digits(rng, int(rng.integers(lo, hi + 1)))
                b = other.sample()
                if num_digits(b) > num_digits(a) or not in_Dn(a + b if op == "+" else a * b, max_digits):
                    continue
                pairs.append((a, b) if rng.random() < 0.5 else (b, a))
            r = T.closure(m, pairs, op) if pairs else {"n": 0, "em": float("nan"), "per_item": []}
            rows.append({"n": r["n"], "em": r["em"]})
            items[f"closure{op}_{lo}-{hi}"] = r["per_item"]
        out["closure" + op] = rows
    return out, items


def fits(x):
    """True if x lies in the exponent window of the slots (at most 26 integer and 26 fractional digits)."""
    from nif.numbers import canonical
    from nif.numdata import E_MAX, E_MIN
    t = canonical(abs(x))
    ip, fp = (t.split(".") + [""])[:2]
    return len(ip.lstrip("0")) <= E_MAX + 1 and len(fp) <= -E_MIN


def comparison_tests(T, m, n, seed=4242):
    """Protocol of the comparison with existing embeddings (Table 5): closure of + and x
    and the order relation for maximum lengths N = 10, 20, 30. N = 30 exceeds the training range N = 25;
    operands outside the slot window are not drawn, and pairs whose result lies outside the window are
    counted as errors (their number is reported)."""
    from nif.numbers import NumberSampler

    class WindowSampler(NumberSampler):
        def sample(self):
            while True:
                x = super().sample()
                if fits(x):
                    return x

    out = {}
    for N in (10, 20, 30):
        S = WindowSampler(max_digits=N, seed=seed + N)
        row = {}
        for op in "+x":
            pairs = S.pairs(n, op)
            ok = [q for q in pairs if fits(q[0] + q[1] if op == "+" else q[0] * q[1])]
            r = T.closure(m, ok, op) if ok else {"em": 0.0, "acc": 0.0}
            f = len(ok) / max(1, len(pairs))
            row["closure" + op] = {"n": len(pairs), "em": r["em"] * f, "acc": r["acc"] * f,
                                   "outside_window": len(pairs) - len(ok)}
        o = T.order_tests(m, S, min(n, 5000))
        row["order"] = {"n": o["order"]["n"], "em": o["order"]["em"]}
        out[f"N{N}"] = row
    return out


def paper_tests(model, max_digits, n, oracle, done, save, log=print, items=None, geometry=False):
    from nif import algebra_tests as T
    from nif.numbers import NumberSampler, units_in_Dn

    class Adapter:
        def encode(self, strings): return model.encode(strings)
        def decode(self, z, target_strings=None): return model.decode(z, target_strings)
        def add(self, h1, h2): return model.add(h1, h2)
        def mul(self, h1, h2): return model.mul(h1, h2)
        def compare(self, h1, h2): return model.compare(h1, h2)
        def decoder_logits(self, z, t): return model.decoder_logits(z, t)
        def ac_layer_weights(self, op): return model.ac_layer_weights(op)

    T.ORACLE_LENGTH = oracle
    m = Adapter()
    S = lambda off=0: NumberSampler(max_digits=max_digits, seed=12345 + off)
    jobs = [
        ("reconstruction", lambda: T.reconstruction(m, S(10).sample_many(n))),
        ("identity+", lambda: T.identity(m, S(11).sample_many(n), "+")),
        ("closure+", lambda: T.closure(m, S(12).pairs(n, "+"), "+")),
        ("invertibility+", lambda: T.invertibility(m, S(13).sample_many(n), "+")),
        ("assoc+", lambda: T.associativity(m, S(14).triples(n, "assoc+"), "+")),
        ("identityx", lambda: T.identity(m, S(15).sample_many(n), "x")),
        ("closurex", lambda: T.closure(m, S(16).pairs(n, "x"), "x")),
        ("invertibilityx", lambda: T.invertibility(m, units_in_Dn(max_digits), "x")),
        ("assocx", lambda: T.associativity(m, S(17).triples(n, "assocx"), "x")),
        ("distrib", lambda: T.distributivity(m, S(18).triples(n, "distrib"))),
        ("order", lambda: T.order_tests(m, S(1), n)),
        ("fraction", lambda: T.fraction_field_tests(m, n, max_int_digits=6)),
        ("chains+", lambda: T.chains(m, S(4), "+", n=500, operand_digits=(1, 6))),
        ("chainsx", lambda: T.chains(m, S(5), "x", n=500, operand_digits=(1, 3))),
        ("by_length", lambda: length_tests(T, m, min(n, 2000), seed=777, max_digits=max_digits)),
        ("comparison", lambda: comparison_tests(T, m, n)),
    ]
    if geometry:   # Section 3.6 quantities (decoder-logit based; independent of the decoding mode)
        jobs.append(("geometry", lambda: T.table10(m, S(30), min(n, 2000), r=GEOMETRY_R, n_dir=50)))

    def strip(v):
        if isinstance(v, dict):
            return {kk: strip(vv) for kk, vv in v.items() if kk != "per_item"}
        return v

    def collect(v, prefix):
        if isinstance(v, dict):
            if "per_item" in v and items is not None:
                items[prefix] = v["per_item"]
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    collect(vv, f"{prefix}.{kk}")

    for name, job in jobs:
        if name in done or (name == "order" and "order_order" in done) or (name == "fraction" and "fraction_add" in done):
            continue
        t0 = time.time()
        r = job()
        if name == "by_length":
            r, it = r
            if items is not None:
                items.update({f"by_length.{k}": v for k, v in it.items()})
        collect(r, name)
        r = strip(r)
        if name in ("order", "fraction"):
            done.update({name + "_" + k: v for k, v in r.items()})
            summary = ", ".join(f"{k} {v['em']:.2f}%" for k, v in r.items() if isinstance(v, dict) and "em" in v)
        else:
            done[name] = r
            if name == "by_length":
                summary = "; ".join(f"{k}: " + " ".join(f"{row['em']:.1f}" for row in r[k])
                                    for k in ("reconstruction", "closure+", "closurex"))
            elif name == "comparison":
                summary = "; ".join(f"{N}: +{v['closure+']['em']:.2f} x{v['closurex']['em']:.2f} <{v['order']['em']:.2f}"
                                    for N, v in r.items())
            elif name == "geometry":
                summary = ", ".join(f"{op}: defect {r[op]['defect']['median']:.3f}, rho {r[op]['rho']['median']:.3f}, "
                                    f"L_hat {r[op]['L_hat']:.2f}" for op in "+x")
            else:
                summary = f"EM {r['em']:.2f}%" if "em" in r else \
                    ", ".join(f"K={k}: {v['em']:.1f}" for k, v in r.get("per_K", {}).items())
        log(f"[{'oracle' if oracle else 'length-free'} N={max_digits}] {name}: {summary} ({time.time() - t0:.1f}s)")
        save()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="name of a run: checkpoint and output of that run")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=10000, help="tuples per test")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results in --out")
    ap.add_argument("--no-geometry", action="store_true", help="skip the Section 3.6 quantities")
    a = ap.parse_args()
    if a.run:
        a.ckpt = a.ckpt or checkpoint(a.run)
        a.out = a.out or os.path.join(RUNS, a.run, "test_results.json")
    if not a.ckpt or not a.out:
        ap.error("give --run, or --ckpt and --out")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load(a.ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) and not a.fresh else {}

    def save():
        with open(a.out + ".tmp", "w") as f:
            json.dump(res, f, indent=2)
        os.replace(a.out + ".tmp", a.out)

    items_path = a.out.replace(".json", "") + ".items.json.gz"
    items = json.load(gzip.open(items_path, "rt")) if os.path.exists(items_path) and not a.fresh else {}

    def save_all():
        save()
        with gzip.open(items_path + ".tmp", "wt") as f:
            json.dump(items, f)
        os.replace(items_path + ".tmp", items_path)

    log = lambda s: print(s, flush=True)
    for oracle in (False, True):
        key = f"N25_{'oracle' if oracle else 'length_free'}"
        res.setdefault(key, {})
        it = items.setdefault(key, {})
        paper_tests(model, 25, a.n, oracle, res[key], save_all, log, items=it, geometry=not oracle and not a.no_geometry)
    save_all()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
