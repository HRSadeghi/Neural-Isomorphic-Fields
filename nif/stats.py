"""McNemar's exact test and the Holm-Bonferroni correction (Section 5.2 of the paper).
"""
from __future__ import annotations

from math import comb
from typing import Dict, List, Sequence


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> float:
    """Two-sided exact McNemar test on paired 0/1 outcomes of two models."""
    assert len(a) == len(b)
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
