"""Fourier analysis of the digit code of a place-value slot, and reconstruction with the true length beyond the
training lengths (CPU, no training).

  code     For numbers whose digit at one place is d = 0..9 (all other digits random), the mean slot vector m_d in
           R^c is computed at several exponents. The discrete Fourier transform of (m_0, ..., m_9) over d gives the
           share of the between-digit variance carried by each harmonic k = 1..5 (period 10/k). A ring in the order
           0, 1, ..., 9 is a pure first harmonic; the report also gives the within-digit spread (how much the slot
           depends on the other digits), the agreement of the code between exponents and the dimension of the code
           (participation ratio of the between-digit covariance).
  oracle   Reconstruction of the items of analysis/lengths.py outside the training lengths with length-free decoding
           and with the true length given to the decoder: if the errors are errors of the length head, the true
           length repairs them.

    NIF_N_DIGITS=<N> python analysis/digit_code.py --runs main_s0,main_s1 [--what code,oracle]

Results are merged into results/analysis/extra.json (keys "fourier" and "oracle").
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

from analysis import lengths as A  # noqa: E402
import evaluate as E  # noqa: E402
from nif.numbers import canonical  # noqa: E402
from nif.numdata import E_MIN, N_DIGITS  # noqa: E402
from nif.paths import checkpoint  # noqa: E402

OUT = os.path.join(HERE, "results", "analysis", "extra.json")


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "analysis_extra.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def slot_vectors(model, exps=(-3, 0, 3, 8), per_digit=40, seed=0):
    """X[e][d]: (per_digit, c) slot vectors of numbers with digit d at exponent e (other digits random)."""
    rng = np.random.default_rng(seed)
    X = {}
    for e in exps:
        X[e] = []
        for d in range(10):
            strs = []
            for _ in range(per_digit):
                ip = [int(v) for v in rng.integers(0, 10, size=10)]
                fp = [int(v) for v in rng.integers(0, 10, size=4)]
                if e >= 0:
                    ip[9 - e] = d
                else:
                    fp[-e - 1] = d
                ip[0] = int(rng.integers(1, 10)) if e != 9 else d
                fp[-1] = fp[-1] or 1 if e != -4 else fp[-1]
                strs.append("".join(map(str, ip)) + "." + "".join(map(str, fp)))
            z = model.encode(strs).view(len(strs), -1, model.c)
            X[e].append(z[:, e - E_MIN, :].double().numpy())
    return X


def fourier_code(model):
    X = slot_vectors(model)
    exps = sorted(X)
    M = {e: np.stack([x.mean(0) for x in X[e]]) for e in exps}                    # (10, c) digit means per exponent
    Mall = np.mean([M[e] for e in exps], 0)
    C = Mall - Mall.mean(0)
    F = np.fft.fft(C, axis=0)                                                       # over the digit index
    power = (np.abs(F) ** 2).sum(1)                                                 # per frequency 0..9
    harm = [float((power[k] + power[10 - k]) / power[1:].sum()) if k < 5 else float(power[5] / power[1:].sum()) for k in range(1, 6)]
    between = float((C ** 2).sum() / 10)
    within = float(np.mean([((x - x.mean(0)) ** 2).sum(1).mean() for e in exps for x in X[e]]))
    ev = np.linalg.eigvalsh(C.T @ C)
    ev = ev[ev > 1e-12]
    # agreement of the centred codes of different exponents (1 = identical code at every place)
    cs = []
    for i in range(len(exps)):
        for j in range(i + 1, len(exps)):
            a, b = M[exps[i]] - M[exps[i]].mean(0), M[exps[j]] - M[exps[j]].mean(0)
            cs.append(float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())))
    # nearest neighbours of each digit mean: are they the digits d - 1 and d + 1 (mod 10)?
    D = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(10) * 1e9
    nn2 = sum(set(np.argsort(D[d])[:2]) == {(d - 1) % 10, (d + 1) % 10} for d in range(10))
    return {"harmonic_share": harm, "within_over_between": within / between, "code_dimension": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "between_exponent_cosine_min": min(cs), "digits_with_cyclic_neighbours": int(nn2),
            "min_digit_distance_over_rms": float(D.min() / np.sqrt(between))}


def oracle_test(model, lengths, n, h):
    m = A.Adapter(model)
    acc = {r: {"n": 0, "length_free": 0, "oracle": 0} for r in ("out", "beyond")}
    for l in lengths:
        xs, _ = A.items_for_length(l, n)
        t = [canonical(x) for x in xs]
        for i in range(0, len(t), 100):
            b = t[i:i + 100]
            z = m.encode(b)
            free, orc = m.decode(z), m.decode(z, b)
            for ti, f, o in zip(b, free, orc):
                ip, fp = (ti.lstrip("+-").split(".") + [""])[:2]
                reg = "beyond" if l > N_DIGITS else ("out" if max(len(ip), len(fp)) > h else None)
                if reg:
                    acc[reg]["n"] += 1
                    acc[reg]["length_free"] += int(f == ti)
                    acc[reg]["oracle"] += int(o == ti)
    for a in acc.values():
        a["em_length_free"] = 100.0 * a["length_free"] / max(1, a["n"])
        a["em_oracle"] = 100.0 * a["oracle"] / max(1, a["n"])
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--what", default="code,oracle")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    N = N_DIGITS
    lengths = list(range(20, 31)) if N <= 30 else sorted(l for l in A.auto_lengths(N) if 0.8 * N <= l <= 1.2 * N)
    for run in a.runs.split(","):
        assert A.run_n(run) == N, f"{run} has N = {A.run_n(run)}: set NIF_N_DIGITS"
        cfg = json.load(open(os.path.join(HERE, "runs", run, "config.json")))
        model = E.load(checkpoint(run), dev="cpu")
        res = {}
        if "code" in a.what:
            res["fourier"] = fourier_code(model)
            log(f"{run} fourier: {json.dumps(res['fourier'])}")
        if "oracle" in a.what:
            res["oracle"] = oracle_test(model, lengths, a.n, cfg.get("max_len") or N // 2)
            log(f"{run} oracle: {json.dumps(res['oracle'])}")
        old = json.load(open(OUT)) if os.path.exists(OUT) else {}
        old.setdefault(run, {"n_digits": N, "c": model.c}).update(res)
        with open(OUT + ".tmp", "w") as f:
            json.dump(old, f, indent=1)
        os.replace(OUT + ".tmp", OUT)


if __name__ == "__main__":
    main()
