"""Three further analyses of trained models on the CPU (no training):

  inverse   x * (1/x) on the 4438 units of D_25, split by whether the digit columns of the product stay inside
            the slot window of the model (n_frac(x) + n_frac(1/x) <= number of fractional slots);
  errors    kinds of reconstruction errors on the items of analysis/lengths.py near and beyond the training
            lengths: wrong predicted length, or right length with wrong digits (and where the wrong digits are);
  code      the digit code of a place-value slot: variance explained by two principal components, whether the
            digits 0..9 lie in cyclic order on a ring, and the spread of their radii.

    NIF_N_DIGITS=<N of the runs> python analysis/extra.py --runs main_s0,emb_c4 [--what inverse,errors,code]

All runs of one call must have the same N (the slot window is fixed per process). Results are merged into
results/analysis/extra.json; progress in logs/analysis_extra.log.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
sys.path.insert(0, HERE)

from analysis import lengths as A  # noqa: E402
import evaluate as E  # noqa: E402
from analysis.geometry import slot_geometry  # noqa: E402
from nif.numbers import canonical, units_in_Dn  # noqa: E402
from nif.numdata import E_MIN, N_DIGITS  # noqa: E402
from nif.paths import checkpoint  # noqa: E402

OUT = os.path.join(HERE, "results", "analysis", "extra.json")


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "analysis_extra.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parts(s):
    s = s.lstrip("+-")
    ip, fp = (s.split(".") + [""])[:2]
    return ip, fp


def inverse_test(m):
    U = units_in_Dn(25)
    low = [len(parts(canonical(x))[1]) + len(parts(canonical(1 / x))[1]) for x in U]     # lowest column: 10^-low
    ok = []
    for i in range(0, len(U), 256):
        b = U[i:i + 256]
        z = m.mul(m.encode([canonical(x) for x in b]), m.encode([canonical(1 / x) for x in b]))
        ok += [int(s == canonical(Fraction(1))) for s in m.decode(z)]
    ins = [o for o, l in zip(ok, low) if l <= -E_MIN]
    outs = [o for o, l in zip(ok, low) if l > -E_MIN]
    pct = lambda v: 100.0 * sum(v) / len(v) if v else None
    return {"lowest_slot": E_MIN, "em": pct(ok), "n": len(ok), "em_inside_window": pct(ins), "n_inside": len(ins),
            "em_outside_window": pct(outs), "n_outside": len(outs)}


def error_kinds(m, lengths, n, h):
    """Reconstruction errors by region: 'in' (both parts <= h, total <= N), 'out' (longer part, total <= N),
    'beyond' (total > N)."""
    acc = {r: {"n": 0, "errors": 0, "wrong_length": 0, "wrong_sign": 0, "wrong_digits": 0, "n_wrong_digits": [],
               "first_wrong_from_top": [], "wrong_in_top3": 0, "wrong_only_low_half": 0} for r in ("in", "out", "beyond")}
    for l in lengths:
        xs, _ = A.items_for_length(l, n)
        t = [canonical(x) for x in xs]
        p = []
        for i in range(0, len(xs), 100):
            p += m.decode(m.encode(t[i:i + 100]))
        for ti, pi in zip(t, p):
            (ip, fp), (jp, gp) = parts(ti), parts(pi)
            reg = "beyond" if l > N_DIGITS else ("in" if max(len(ip), len(fp)) <= h else "out")
            a = acc[reg]
            a["n"] += 1
            if pi == ti:
                continue
            a["errors"] += 1
            if len(ip) != len(jp) or len(fp) != len(gp):
                a["wrong_length"] += 1
            elif ti[0] != pi[0] and ip == jp and fp == gp:
                a["wrong_sign"] += 1
            else:
                d1, d2 = ip + fp, jp + gp
                wrong = [k for k, (u, v) in enumerate(zip(d1, d2)) if u != v]
                a["wrong_digits"] += 1
                a["n_wrong_digits"].append(len(wrong))
                a["first_wrong_from_top"].append(wrong[0] / max(1, len(d1) - 1) if wrong else 0.0)
                a["wrong_in_top3"] += int(any(k < 3 for k in wrong))
                a["wrong_only_low_half"] += int(all(k >= len(d1) / 2 for k in wrong))
    for a in acc.values():
        a["median_wrong_digits"] = float(np.median(a["n_wrong_digits"])) if a["n_wrong_digits"] else None
        a["median_first_wrong_position"] = float(np.median(a["first_wrong_from_top"])) if a["first_wrong_from_top"] else None
        del a["n_wrong_digits"], a["first_wrong_from_top"]
    return {"lengths": lengths, "n_per_length": n, "regions": acc}


def digit_code(model):
    g = slot_geometry(model)
    P = g["points"]
    cen = np.array([[np.mean([p["x"] for p in P if p["digit"] == d]), np.mean([p["y"] for p in P if p["digit"] == d])]
                    for d in range(10)])
    c0 = cen.mean(0)
    ang = np.unwrap(np.arctan2(cen[:, 1] - c0[1], cen[:, 0] - c0[0]))
    steps = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi * round((ang[-1] - ang[0]) / (2 * math.pi) + np.sign(ang[-1] - ang[0]))]]))
    rad = np.linalg.norm(cen - c0, axis=1)
    spread = np.mean([np.linalg.norm(np.array([[p["x"], p["y"]] for p in P if p["digit"] == d]) - cen[d], axis=1).mean()
                      for d in range(10)])
    total = abs(ang[-1] - ang[0]) / (2 * math.pi)
    return {"explained_variance_2pc": float(sum(g["explained_variance"])),
            "cyclic_order": bool(np.all(np.diff(ang) > 0) or np.all(np.diff(ang) < 0)), "turns_0_to_9": float(total),
            "radius_cv": float(rad.std() / rad.mean()), "spread_over_radius": float(spread / rad.mean()),
            "min_step_deg": float(np.degrees(np.abs(np.diff(ang)).min())), "max_step_deg": float(np.degrees(np.abs(np.diff(ang)).max()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--what", default="inverse,errors,code")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    what = a.what.split(",")
    N = N_DIGITS
    lengths = list(range(20, 31)) if N <= 30 else sorted(l for l in A.auto_lengths(N) if 0.8 * N <= l <= 1.2 * N)
    for run in a.runs.split(","):
        assert A.run_n(run) == N, f"{run} has N = {A.run_n(run)}: set NIF_N_DIGITS"
        cfg = json.load(open(os.path.join(HERE, "runs", run, "config.json")))
        h = cfg.get("max_len") or N // 2
        model = E.load(checkpoint(run), dev="cpu")
        m = A.Adapter(model)
        res = {"n_digits": N, "c": model.c}
        t0 = time.time()
        if "code" in what:
            res["code"] = digit_code(model)
            log(f"{run} code: {json.dumps(res['code'])}")
        if "inverse" in what:
            res["inverse"] = inverse_test(m)
            log(f"{run} inverse: {json.dumps(res['inverse'])}")
        if "errors" in what:
            res["errors"] = error_kinds(m, lengths, a.n, h)
            log(f"{run} errors: {json.dumps(res['errors']['regions'])}")
        old = json.load(open(OUT)) if os.path.exists(OUT) else {}
        old.setdefault(run, {}).update(res)
        with open(OUT + ".tmp", "w") as f:
            json.dump(old, f, indent=1)
        os.replace(OUT + ".tmp", OUT)
        log(f"{run}: done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
