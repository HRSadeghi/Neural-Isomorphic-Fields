"""Join the per-item outcomes of analysis/lengths.py with the shape of every test number: the numbers of
integer and fractional digits of the number that the decoder has to produce (the number itself for
reconstruction, the result for the closure tests).

    python analysis/parts.py                 # every run with runs/<run>/lengths.json
    python analysis/parts.py --runs main_s0,len_n50_c4

No model is evaluated: the test items are regenerated from their seeds (analysis.lengths.items_for_length) and
matched with runs/<run>/lengths_items.json.gz. Output: runs/<run>/lengths_parts.json with, per test, the lists
total (digits), n_int, n_frac and ok (0/1), and the training range of the run (uniform lengths 0..max_len per
part). The slot window depends on N, so runs with another N are handled in a child process (NIF_N_DIGITS).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
sys.path.insert(0, HERE)

from analysis import lengths as A  # noqa: E402
from nif.numbers import canonical  # noqa: E402
from nif.numdata import N_DIGITS  # noqa: E402

TESTS = ("reconstruction", "closure+", "closurex")


def parts(x):
    ip, fp = (canonical(abs(x)).lstrip("+-").split(".") + [""])[:2]
    return len(ip), len(fp)


def analysed_runs():
    root = os.path.join(HERE, "runs")
    return sorted(r for r in os.listdir(root) if os.path.exists(os.path.join(root, r, "lengths.json"))
                  and os.path.exists(os.path.join(root, r, "lengths_items.json.gz")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all")
    a = ap.parse_args()
    runs = analysed_runs() if a.runs == "all" else [r for r in a.runs.split(",") if r]
    groups = {}
    for run in runs:
        groups.setdefault(A.run_n(run), []).append(run)
    cache = {}
    for n_digits, group in sorted(groups.items()):
        if n_digits != N_DIGITS:
            rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "--runs", ",".join(group)],
                                 env=dict(os.environ, NIF_N_DIGITS=str(n_digits)))
            if rc != 0:
                raise SystemExit(f"analysis/parts.py failed for {group} (N = {n_digits})")
            continue
        for run in group:
            d = os.path.join(HERE, "runs", run)
            r = json.load(open(os.path.join(d, "lengths.json")))
            items = json.load(gzip.open(os.path.join(d, "lengths_items.json.gz"), "rt"))
            cfg = json.load(open(os.path.join(d, "config.json")))
            out = {"run": run, "n_digits": N_DIGITS, "c": r["c"], "d": r["d"], "n": r["n"],
                   "max_part_trained": cfg.get("max_len") or N_DIGITS // 2, "uniform_frac": cfg.get("uniform_frac", 0.5),
                   "tests": {k: {"total": [], "n_int": [], "n_frac": [], "ok": []} for k in TESTS}}
            for l in r["lengths"]:
                if (l, r["n"]) not in cache:
                    xs, pairs = A.items_for_length(l, r["n"])
                    cache[(l, r["n"])] = {"reconstruction": [parts(x) for x in xs],
                                          "closure+": [parts(x + y) for x, y in pairs["+"]],
                                          "closurex": [parts(x * y) for x, y in pairs["x"]]}
                for k in TESTS:
                    ok, shape = items[str(l)][k], cache[(l, r["n"])][k]
                    assert len(ok) == len(shape), (run, l, k)
                    t = out["tests"][k]
                    t["total"] += [l] * len(ok)
                    t["n_int"] += [s[0] for s in shape]
                    t["n_frac"] += [s[1] for s in shape]
                    t["ok"] += [int(o) for o in ok]
            with open(os.path.join(d, "lengths_parts.json"), "w") as f:
                json.dump(out, f)
            print(f"{run}: N {N_DIGITS}, {sum(len(t['ok']) for t in out['tests'].values())} items", flush=True)


if __name__ == "__main__":
    main()
