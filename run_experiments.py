"""Run the experiments of configs/experiments.json one after the other.

    python run_experiments.py                       # everything (finished runs are skipped)
    python run_experiments.py --only main_s0,main_s1

Each experiment runs in its own process. Finished experiments (runs/<name>/DONE) are skipped, and an interrupted
one continues from its own last.pt when it is started again. After a full-model run the tests of the paper run on
runs/<name>/best.pt and are written to runs/<name>/test_results.json; the runs of the length study are tested by
analysis/lengths.py. The output of every process is written to logs/.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(HERE, "logs", "queue.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, logfile):
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}\n")
        f.flush()
        return subprocess.call(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated names")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    exps = json.load(open(os.path.join(HERE, "configs", "experiments.json")))["experiments"]
    only = set(x for x in a.only.split(",") if x)
    for e in exps:
        name = e["name"]
        if only and name not in only:
            continue
        rundir = os.path.join(HERE, "runs", name)
        script = "train.py" if e["kind"] == "full" else "train_operator.py"
        if not os.path.exists(os.path.join(rundir, "DONE")):
            log(f"start {name}")
            t0 = time.time()
            rc = run([PY, "-u", script, "--name", name] + e["args"], os.path.join(HERE, "logs", f"{name}.log"))
            if not os.path.exists(os.path.join(rundir, "DONE")):
                log(f"{name} stopped (exit code {rc}); see logs/{name}.log")
                continue
            log(f"finished {name} after {(time.time() - t0) / 3600:.2f} h")
        if e["kind"] == "full":
            lengths = e.get("test") == "lengths"           # length x dimension study: exact match by length
            res = os.path.join(rundir, "lengths.json" if lengths else "test_results.json")
            cmd = [PY, "-u", "-m", "analysis.lengths", "--runs", name, "--device", "cuda"] if lengths else \
                [PY, "-u", "evaluate.py", "--ckpt", os.path.join(rundir, "best.pt"), "--out", res, "--n", "10000"]
            tlog = os.path.join(HERE, "logs", f"{name}_test.log")
            if os.path.exists(res) and os.path.exists(tlog) and "DONE" in open(tlog, encoding="utf-8", errors="ignore").read():
                continue                                    # tested already
            if not os.path.exists(os.path.join(rundir, "best.pt")):
                continue                                    # a run of the paper: its results are in the repository
            log(f"tests of {name}")
            rc = run(cmd, tlog)
            log(f"tests of {name} {'done' if rc == 0 else f'stopped (exit code {rc})'}")
    log("all experiments done")


if __name__ == "__main__":
    main()
