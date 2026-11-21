"""Runs of the reading experiment (Section 5.3.5, Table 6) on the CPU.

    python reading/run.py                      the five runs of the table, seed 0 (finished runs are skipped)
    python reading/run.py --seeds 0,1,2        more seeds (make_table.py then reports mean +- SD)
    python reading/run.py --jobs 3             three runs at the same time (default 1)
    python reading/run.py --list               show the commands and what is finished, run nothing

Every run is one process of reading/experiment.py; its output is the log reading/logs/<name>_s<seed>.log, and a run
counts as finished when the log contains the line of its last step. Then:

    python reading/make_table.py               -> paper/generated/tab_reader.tex, reader_numbers.tex

Arguments of a run: (condition, steps, threads, training problems, vocabulary size). The NIF vectors come from the
checkpoint of main_s0; the vectors of every condition are cached in reading/cache/.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "experiment.py")
RUNS = os.path.join(HERE, "logs")

# name of the log -> (condition, steps, threads, training problems, vocabulary size)
PAPER = {
    "text_lsb":     ("text_lsb",     4000, 2, 30000, 16),
    "read_onehot":  ("read_onehot",  4000, 2, 30000, 16),
    "read_f64fone": ("read_f64fone", 4000, 2, 30000, 16),
    "read_fone":    ("read_fone",    4000, 2, 30000, 16),
    "read_nif":     ("read_nif",     4000, 2, 30000, 16),
}
SEED0_VOCAB = {"read_onehot": 17, "read_f64fone": 17}       # vocabulary size of these two runs in the table


def finished(log, steps):
    if not os.path.exists(log):
        return False
    for line in open(log, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try:
                if json.loads(line).get("step") == steps:
                    return True
            except ValueError:
                pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--only", default="", help="comma-separated names of runs")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--steps", type=int, default=0, help="for tests only: other number of steps (use another --runs)")
    ap.add_argument("--train", type=int, default=0, help="for tests only: other number of training problems")
    args = ap.parse_args()
    runs = dict(PAPER)
    if args.only:
        runs = {k: v for k, v in PAPER.items() if k in args.only.split(",")}
    if (args.steps or args.train) and os.path.abspath(args.runs) == os.path.abspath(RUNS):
        raise SystemExit("--steps / --train are for tests: give another folder with --runs")
    os.makedirs(args.runs, exist_ok=True)

    todo = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for name, (cond, steps, threads, n_train, vocab) in runs.items():
            if seed == 0:
                vocab = SEED0_VOCAB.get(name, vocab)
            steps, n_train = args.steps or steps, args.train or n_train
            log = os.path.join(args.runs, f"{name}_s{seed}.log")
            cmd = [sys.executable, "-u", SCRIPT, cond, str(steps), str(seed), str(threads), str(n_train), str(vocab)]
            done = finished(log, steps)
            print(("done   " if done else "to run ") + f"{name}_s{seed}: " + " ".join(cmd[3:]))
            if not done:
                todo.append((log, cmd))
    if args.list or not todo:
        return

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")           # CPU only
    active, t0 = [], time.time()
    while todo or active:
        while todo and len(active) < args.jobs:
            log, cmd = todo.pop(0)
            f = open(log, "w", encoding="utf-8")
            active.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE), f, log))
            print(f"[{(time.time() - t0) / 60:6.1f} min] started {os.path.basename(log)}", flush=True)
        time.sleep(5)
        for item in list(active):
            p, f, log = item
            if p.poll() is not None:
                f.close()
                active.remove(item)
                print(f"[{(time.time() - t0) / 60:6.1f} min] {'finished' if p.returncode == 0 else 'FAILED'} "
                      f"{os.path.basename(log)}", flush=True)


if __name__ == "__main__":
    main()
