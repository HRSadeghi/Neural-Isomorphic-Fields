"""The runs of the language-model experiment (Section 5.4), one after the other.

    python downstream/run.py                          # list the runs and what is finished
    python downstream/run.py operator_layers_test     # tests of the released model (no training)
    python downstream/run.py base_model operator_layers operator_layers_10000
    python downstream/run.py operator_layers --out my_runs

Runs
    base_model              Qwen3.5-0.8B as released, all tests (Table 7, Figure 11)
    operator_layers         training with NIF operator layers, 6000 steps; the complete test every 500 steps, the
                            final test and the interventions (Table 7, Figures 11 and 12)
    operator_layers_10000   operator_layers continued to 10000 steps from its checkpoint (Figure 11c)
    operator_layers_test    the final test and the interventions of the released model
                            (checkpoints/language_model/operator_layers_6000.pt), without training

A run writes downstream/runs/<name>/ (or <out>/<name>/); a finished run (DONE) is skipped, an interrupted one resumes
from its checkpoint. The results of the paper are in downstream/runs/; to repeat a run, give another --out.
Every run needs one GPU with about 70 GB of memory (the batches are split automatically when less is free).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = {
    "base_model": ("zeroshot", []),
    "operator_layers": ("op_nif", []),
    "operator_layers_10000": ("op_nif", ["steps=10000", "init_ckpt={out}/operator_layers/ckpt.pt",
                                         "restart_at=6000", "restart_peak=0.4"]),
    "operator_layers_test": ("op_nif", ["init_ckpt=checkpoints/language_model/operator_layers_6000.pt"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="names of runs (none: list them)")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"), help="folder of the run folders")
    ap.add_argument("--set", nargs="*", default=[], help="further settings of downstream/train.py")
    a = ap.parse_args()
    if not a.runs:
        for name, (method, settings) in RUNS.items():
            done = os.path.exists(os.path.join(a.out, name, "DONE"))
            print(f"{'finished' if done else 'to run  '}  {name:22s} {method} {' '.join(x.format(out=a.out) for x in settings)}")
        return
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for name in a.runs:
        method, settings = RUNS[name]
        out = os.path.join(a.out, name)
        cmd = [sys.executable, "-u", os.path.join(HERE, "train.py"), "--method", method, "--out", out,
               "--set"] + [x.format(out=a.out) for x in settings] + a.set
        print(" ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=ROOT, env=env)
        if rc:
            raise SystemExit(f"{name} stopped with exit code {rc}")


if __name__ == "__main__":
    main()
