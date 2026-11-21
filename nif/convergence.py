"""Stopping rule for training until convergence.

A run stops at an evaluation when, after at least `min_steps` steps, the validation score has not improved
by more than `min_delta` over its best value for `patience` consecutive evaluations, or when the score has
reached `saturate` at the last two evaluations. The run's `steps` is always an upper bound.
"""


def convergence_state(evals, key, min_delta, patience, min_steps, saturate=None):
    """evals: list of evaluation records (with 'step' and `key`), any order. Returns (stop, info)."""
    evs = sorted(evals, key=lambda e: e["step"])
    if not evs or patience <= 0:
        return False, {"enabled": False}
    best, best_i = float("-inf"), 0
    for i, e in enumerate(evs):
        if e[key] > best + min_delta:
            best, best_i = e[key], i
    since = len(evs) - 1 - best_i
    step = evs[-1]["step"]
    info = {"enabled": True, "best_score": round(best, 4), "best_step": evs[best_i]["step"],
            "evals_since_best": since, "patience": patience, "min_delta": min_delta, "min_steps": min_steps}
    stop, reason = False, ""
    if step >= min_steps and since >= patience:
        stop, reason = True, f"no improvement > {min_delta} for {since} evaluations (best {best:.3f} at step {evs[best_i]['step']})"
    elif saturate is not None and step >= min_steps and len(evs) >= 2 and all(e[key] >= saturate for e in evs[-2:]):
        stop, reason = True, f"saturated: {key} >= {saturate} at the last two evaluations"
    info.update(stop=stop, reason=reason)
    return stop, info
