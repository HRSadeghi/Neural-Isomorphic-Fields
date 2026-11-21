"""Inference cost of NIF: time per number and peak GPU memory of the encoder, the operators, the comparator and the
length-free decoder, for numbers with a given number of digits (row "Inference cost" of Table A.1).

    python analysis/measure_cost.py --ckpt checkpoints/main_s0/best.pt --digits 25
    NIF_N_DIGITS=200 python analysis/measure_cost.py --ckpt checkpoints/len_n200_c4/best.pt --digits 200

Prints one JSON line (microseconds per number at the given batch size, peak memory in MB per step).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
sys.path.insert(0, HERE)

from evaluate import load, number_with_digits  # noqa: E402
from nif.numbers import canonical  # noqa: E402


def timed(fn, reps, batch):
    for _ in range(3):                                   # warm-up
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / reps / batch * 1e6
    return round(us, 2), round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--digits", type=int, required=True, help="digits of every test number")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load(a.ckpt)
    rng = np.random.default_rng(0)
    xs = [canonical(number_with_digits(rng, a.digits)) for _ in range(2 * a.batch)]
    s1, s2 = xs[:a.batch], xs[a.batch:]
    out = {"ckpt": a.ckpt, "digits": a.digits, "batch": a.batch, "gpu": torch.cuda.get_device_name(0),
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    with torch.no_grad():
        z1, z2 = model.encode(s1), model.encode(s2)
        out["dim"] = int(z1.shape[-1])
        for name, fn in (("encode", lambda: model.encode(s1)), ("add", lambda: model.add(z1, z2)),
                         ("mul", lambda: model.mul(z1, z2)), ("compare", lambda: model.compare(z1, z2)),
                         ("decode", lambda: model.decode(z1))):
            out[name + "_us"], out[name + "_MB"] = timed(fn, a.reps if name != "decode" else max(3, a.reps // 4), a.batch)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
