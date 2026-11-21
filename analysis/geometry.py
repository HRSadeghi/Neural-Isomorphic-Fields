"""Analyses of the selected model (main_s0) for Figure 10 of the paper (CPU is enough).

    python analysis/geometry.py                      # the selected model (main_s0), CPU
    python analysis/geometry.py --run main_s1 --device cuda --n 800 --n-dir 20

Writes results/analysis/model_analysis.json (selected model) or model_analysis_<run>.json, with
  * meta: checkpoint, step, validation score, n, n_dir, device, duration;
  * slot_geometry: the c-dimensional vector of one place-value slot for every digit 0..9 (all other digits
    random), for several exponents, with a common 2-D PCA projection (what the encoder learned per place);
  * items: per-tuple defect ||phi(a)*phi(b) - phi(a*b)||, estimated decoding radius of the target
    (Lemma 3.23, delta_0 = 6) and exact-match outcome, for closure pairs of + and x;
  * norms: ||phi(a)|| for the tested numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
sys.path.insert(0, HERE)

import evaluate as E  # noqa: E402
from nif import algebra_tests as T  # noqa: E402
from nif.numbers import NumberSampler, canonical  # noqa: E402
from nif.paths import checkpoint  # noqa: E402


class Adapter:
    def __init__(self, m):
        self.m = m

    def encode(self, s): return self.m.encode(s)
    def decode(self, z, target_strings=None): return self.m.decode(z, target_strings)
    def add(self, a, b): return self.m.add(a, b)
    def mul(self, a, b): return self.m.mul(a, b)
    def compare(self, a, b): return self.m.compare(a, b)
    def decoder_logits(self, z, t): return self.m.decoder_logits(z, t)


LOG = os.path.join(HERE, "logs", "analysis_geometry.log")


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def slot_geometry(model, exps=(-3, 0, 3, 8), per_digit=40, seed=0):
    from nif.numdata import E_MIN
    rng = np.random.default_rng(seed)
    c = model.c
    rows = []
    for e in exps:
        for d in range(10):
            strs = []
            for _ in range(per_digit):
                # a number with digits at exponents -4..9 (random), digit d at exponent e
                ip = [int(v) for v in rng.integers(0, 10, size=10)]
                fp = [int(v) for v in rng.integers(0, 10, size=4)]
                if e >= 0:
                    ip[9 - e] = d
                else:
                    fp[-e - 1] = d
                ip[0] = int(rng.integers(1, 10)) if e != 9 else d
                fp[-1] = fp[-1] or 1 if e != -4 else fp[-1]
                strs.append("".join(map(str, ip)) + "." + "".join(map(str, fp)))
            z = model.encode(strs).view(len(strs), -1, c)
            slot = z[:, e - E_MIN, :].numpy()
            rows += [(e, d, v) for v in slot]
    X = np.stack([r[2] for r in rows])
    Xc = X - X.mean(0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ vt[:2].T
    var = (s ** 2 / (s ** 2).sum())[:2]
    return {"exps": list(exps), "explained_variance": var.tolist(),
            "points": [{"exp": int(r[0]), "digit": int(r[1]), "x": float(p[0]), "y": float(p[1])} for r, p in zip(rows, P)]}


def items(model, m, op, n, n_dir, r=6.0, seed=31):
    S = NumberSampler(max_digits=25, seed=seed)
    pairs = S.pairs(n, op)
    eps = T.defects(m, pairs, op)
    targets = [a + b if op == "+" else a * b for a, b in pairs]
    rho = T.radius_estimates(m, targets, r, n_dir)
    T.ORACLE_LENGTH = False
    em = T.closure(m, pairs, op)["per_item"]
    lens = [max(len(canonical(a).replace("-", "").replace(".", "")), len(canonical(b).replace("-", "").replace(".", "")))
            for a, b in pairs]
    return [{"defect": float(e), "radius": float(p), "em": int(k), "digits": int(l)} for e, p, k, l in zip(eps, rho, em, lens)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="analyse runs/<run>/best.pt instead of the selected model")
    ap.add_argument("--ckpt", default=checkpoint("main_s0"))
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--n-dir", type=int, default=20)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    if a.device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    ck = checkpoint(a.run) if a.run else a.ckpt
    t0 = time.time()
    meta = torch.load(ck, map_location="cpu", weights_only=False)
    log(f"start: {ck} (step {meta.get('step')}), n {a.n}, n_dir {a.n_dir}, device {a.device}")
    model = E.load(ck, dev=a.device)
    m = Adapter(model)
    out = {"meta": {"ckpt": os.path.relpath(ck, HERE), "run": a.run, "step": meta.get("step"), "validation_score": meta.get("score"),
                    "n": a.n, "n_dir": a.n_dir, "delta0": 6.0, "device": a.device}}
    out["slot_geometry"] = slot_geometry(model)
    log("slot geometry done: explained variance " + str([round(v, 3) for v in out["slot_geometry"]["explained_variance"]]))
    out["items"] = {}
    for op in "+x":
        out["items"][op] = items(model, m, op, a.n, a.n_dir)
        it = out["items"][op]
        log(f"{op}: EM {100 * np.mean([i['em'] for i in it]):.2f}%, certified {100 * np.mean([i['defect'] < i['radius'] for i in it]):.2f}%, "
            f"median defect {np.median([i['defect'] for i in it]):.3f}, median radius {np.median([i['radius'] for i in it]):.3f}")
    out["meta"]["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.join(HERE, "results", "analysis"), exist_ok=True)
    p = os.path.join(HERE, "results", "analysis", f"model_analysis_{a.run}.json" if a.run else "model_analysis.json")
    json.dump(out, open(p, "w"))
    log(f"written {p} ({out['meta']['seconds']} s)")


if __name__ == "__main__":
    main()
