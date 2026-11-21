"""Exact match by number length for trained models, for any maximum length N of the training domain D_N.

    python analysis/lengths.py                                   # three seeds and two ablations, CPU
    python analysis/lengths.py --runs all --device cuda          # every finished full-model run, on the GPU
    python analysis/lengths.py --runs main_s0,emb_c4 --n 200
    python analysis/lengths.py --runs len_n200_c4 --device cuda  # length x dimension study (N = 200)

For every length l, `n` numbers with exactly l digits (reconstruction) and `n` pairs whose RESULT has exactly
l digits (closure of + and x: the number the decoder has to produce) are drawn with fixed seeds that depend
only on l (identical for every model, so the models are compared on the same items). In the + pairs the
second operand is, with probability 1/2 each, a number of the paper's sampler or a number with a uniform
number of digits up to l. All numbers fit the slot window of the run (at most N+1 integer and N+1 fractional
digits); l > N lies outside the training domain. Length-free decoding.

Lengths (auto): 1..N+5 for N <= 30 (N = 25: 1..30, every length), otherwise 1, 2, 3, 5 and every N/20 up to
1.2 N (N = 200: 10, 20, ..., 240).

N of a run is read from runs/<run>/config.json (n_digits, default 25). numdata fixes N per process, so runs
with another N than the current process are analysed in a child process with NIF_N_DIGITS set.

Everything is logged and the script resumes after an interruption (per model and per length). Results are
kept next to the run:
  runs/<run>/lengths.json            exact match per test and length + metadata (N, c, d, K, checkpoint, ...)
  runs/<run>/lengths_items.json.gz   0/1 outcome of every item (for McNemar's test between models)
  logs/lengths_<run>.log             progress with timestamps
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
sys.path.insert(0, HERE)

import evaluate as E  # noqa: E402
from nif import algebra_tests as T  # noqa: E402
from nif.numbers import NumberSampler, num_digits  # noqa: E402
from nif.numdata import E_MAX, N_DIGITS  # noqa: E402
from nif.paths import checkpoint  # noqa: E402

DEFAULT_RUNS = "main_s0,main_s1,main_s2,full_no_length_mix,full_no_targeted"


def run_dir(run):
    return os.path.join(HERE, "runs", run)


def run_n(run):
    p = os.path.join(run_dir(run), "config.json")
    return int(json.load(open(p)).get("n_digits", 25)) if os.path.exists(p) else 25


def log(run, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + f"[{run}] " + msg
    print(line, flush=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", f"lengths_{run}.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def auto_lengths(N):
    if N <= 30:
        return list(range(1, N + 6))
    step = max(1, N // 20)
    return sorted({1, 2, 3, 5, N} | set(range(step, int(1.2 * N) + 1, step)))


def number(rng, l):
    """A random number with exactly l digits whose parts fit the slot window (W = N + 1 digits per part)."""
    W = E_MAX + 1
    li = int(rng.integers(max(1, l - W), min(l, W) + 1))
    lf = l - li
    ip = [int(rng.integers(0 if li == 1 else 1, 10))] + [int(v) for v in rng.integers(0, 10, size=li - 1)]
    fp = [int(v) for v in rng.integers(0, 10, size=lf)]
    if lf:
        fp[-1] = int(rng.integers(1, 10))
    x = Fraction(int("".join(map(str, ip)))) + (Fraction(int("".join(map(str, fp))), 10 ** lf) if lf else 0)
    return -x if rng.random() < 0.5 else x


def items_for_length(l, n):
    """The test items of one length; the seeds depend only on l, so every model sees the same items."""
    rng, other = np.random.default_rng(1000 + l), NumberSampler(max_digits=25, seed=2000 + l)
    xs = [number(rng, l) for _ in range(n)]
    pairs = {}
    for op in "+x":
        P = []
        while len(P) < n:
            if op == "+":
                x = number(rng, l)
                y = other.sample() if rng.random() < 0.5 else number(rng, int(rng.integers(1, l + 1)))
                if num_digits(y) > l:
                    continue
            else:
                la = int(rng.integers(1, l + 1))
                lb = int(rng.integers(max(1, l - la), l - la + 2))
                x, y = number(rng, la), number(rng, lb)
            r = x + y if op == "+" else x * y
            if num_digits(r) != l or not E.fits(r):
                continue
            P.append((x, y) if rng.random() < 0.5 else (y, x))
        pairs[op] = P
    return xs, pairs


class Adapter:
    def __init__(self, m):
        self.m = m

    def encode(self, s): return self.m.encode(s)
    def decode(self, z, target_strings=None): return self.m.decode(z, target_strings)
    def add(self, a, b): return self.m.add(a, b)
    def mul(self, a, b): return self.m.mul(a, b)


def finished_full_runs():
    exps = json.load(open(os.path.join(HERE, "configs", "experiments.json")))["experiments"]
    return [e["name"] for e in exps if e["kind"] == "full" and os.path.exists(checkpoint(e["name"]))]


def analyse(run, a):
    out, items_path = os.path.join(run_dir(run), "lengths.json"), os.path.join(run_dir(run), "lengths_items.json.gz")
    ck = checkpoint(run)
    if not os.path.exists(ck):
        log(run, "no best.pt, skipped")
        return
    lengths = [int(x) for x in a.lengths.split(",")] if a.lengths != "auto" else auto_lengths(N_DIGITS)
    too_long = [l for l in lengths if l > 2 * (E_MAX + 1)]
    if too_long:
        log(run, f"lengths {too_long} do not fit the slot window of N = {N_DIGITS} (at most {2 * (E_MAX + 1)} digits): skipped")
        lengths = [l for l in lengths if l not in too_long]
    r = json.load(open(out)) if os.path.exists(out) and not a.fresh else None
    items = json.load(gzip.open(items_path, "rt")) if os.path.exists(items_path) and not a.fresh else {}
    if r and r.get("n") == a.n and all(l in r["lengths"] for l in lengths):
        log(run, "already done")
        return
    if not r or r.get("n") != a.n:
        meta = torch.load(ck, map_location="cpu", weights_only=False)
        c = meta["cfg"]["c"]
        r = {"run": run, "n": a.n, "n_digits": N_DIGITS, "c": c, "K": 2 * N_DIGITS + 2, "d": (2 * N_DIGITS + 2) * c,
             "lengths": [], "reconstruction": [], "closure+": [], "closurex": [], "seconds": [],
             "checkpoint": os.path.relpath(ck, HERE), "step": meta.get("step"), "validation_score": meta.get("score"),
             "device": a.device, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        items = {}
    log(run, f"N {N_DIGITS}, c {r['c']}, d {r['d']}; lengths {lengths}; n {a.n}; device {a.device}")
    m = Adapter(E.load(ck, dev=a.device))
    t_run = time.time()
    for l in lengths:
        if l in r["lengths"]:
            continue
        t0 = time.time()
        xs, pairs = items_for_length(l, a.n)
        rec = T.reconstruction(m, xs)
        cl = {op: T.closure(m, pairs[op], op) for op in "+x"}
        r["lengths"].append(l)
        r["reconstruction"].append(rec["em"])
        r["closure+"].append(cl["+"]["em"])
        r["closurex"].append(cl["x"]["em"])
        r["seconds"].append(round(time.time() - t0, 1))
        order = sorted(range(len(r["lengths"])), key=lambda i: r["lengths"][i])      # keep the lists sorted by length
        for k in ("lengths", "reconstruction", "closure+", "closurex", "seconds"):
            r[k] = [r[k][i] for i in order]
        items[str(l)] = {"reconstruction": rec["per_item"], "closure+": cl["+"]["per_item"], "closurex": cl["x"]["per_item"]}
        with open(out + ".tmp", "w") as f:
            json.dump(r, f, indent=1)
        os.replace(out + ".tmp", out)
        with gzip.open(items_path + ".tmp", "wt") as f:
            json.dump(items, f)
        os.replace(items_path + ".tmp", items_path)
        log(run, f"length {l:3d}: rec {rec['em']:6.2f}  cl+ {cl['+']['em']:6.2f}  clx {cl['x']['em']:6.2f}  "
                 f"({time.time() - t0:.0f} s)")
    log(run, f"done in {(time.time() - t_run) / 60:.1f} min; results in {os.path.relpath(out, HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="comma-separated run names, or 'all' (every full run with best.pt)")
    ap.add_argument("--n", type=int, default=200, help="items per test and length")
    ap.add_argument("--lengths", default="auto", help="'auto' or a comma-separated list of lengths")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results")
    a = ap.parse_args()
    if a.device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    T.ORACLE_LENGTH = False
    runs = finished_full_runs() if a.runs == "all" else [r for r in a.runs.split(",") if r]
    groups = {}
    for run in runs:
        groups.setdefault(run_n(run), []).append(run)
    for n_digits, group in sorted(groups.items()):
        if n_digits == N_DIGITS:
            for run in group:
                analyse(run, a)
        else:                                  # another N: a child process with its own slot window
            args = [sys.executable, "-u", os.path.abspath(__file__), "--runs", ",".join(group), "--n", str(a.n),
                    "--lengths", a.lengths, "--device", a.device] + (["--fresh"] if a.fresh else [])
            rc = subprocess.call(args, env=dict(os.environ, NIF_N_DIGITS=str(n_digits)))
            if rc != 0:
                raise SystemExit(f"length analysis of {group} (N = {n_digits}) failed with code {rc}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
