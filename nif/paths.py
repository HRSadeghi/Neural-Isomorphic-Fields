"""Folders of the repository."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")                  # training logs and test results (in git)
CHECKPOINTS = os.path.join(ROOT, "checkpoints")    # model weights (not in git)
RESULTS = os.path.join(ROOT, "results")


def checkpoint(run, name="best.pt"):
    """Weights of a run: runs/<run>/<name> after training here, otherwise checkpoints/<run>/<name>."""
    own = os.path.join(RUNS, run, name)
    if os.path.exists(own):
        return own
    return os.path.join(CHECKPOINTS, run, name)
