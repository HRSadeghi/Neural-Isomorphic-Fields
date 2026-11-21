"""Word problems from various domains (the share "further word problems" of the training mixture): every problem has
a question with numbers, a numeric answer, the number of decimal places of the answer and a difficulty. The problems
are in data/domain_problems.json.gz; the model's own solutions of them, which are the training data, are in
data/domain_solutions.json.gz (make_solutions.py --source domains).

Only the Python standard library is needed here.
"""
from __future__ import annotations

import gzip
import json
import os
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation

import data as D

DIFF = ("easy", "medium", "hard", "very_hard")
PROBLEM_FILE = "domain_problems.json.gz"


def load(data_dir, path=""):
    """{'train': [problems]}."""
    path = path or os.path.join(data_dir, PROBLEM_FILE)
    return json.load(gzip.open(path, "rt", encoding="utf-8"))


def round_hint(d):
    if d is None:
        return ""
    return " Give the final answer as a whole number." if d <= 0 else \
        f" Give the final answer rounded to {d} decimal place{'s' if d > 1 else ''}."


def example(r):
    """A problem as an example of the task domain_cot: the numbers of the question are Num parts; the target is only
    the final answer (make_solutions.py writes the solutions)."""
    return dict(task="domain_cot", size=DIFF.index(r["difficulty"]) + 1,
                user=D.split_numbers(r["question"]) + [round_hint(r["decimals"]), D.GSM_COT],
                asst=["#### ", D.Num(r["answer"])], answer=r["answer"], answer_alt=r.get("answer_alt"),
                decimals=r["decimals"], question=r["question"], id=r["id"], field=r["field"])


def correct(pred, answer, decimals, alt=None):
    """The final number is right if it equals the answer (or the half-up alternative of solve()), or rounds to one
    of them at the answer's decimal places."""
    try:
        p = Decimal(D.canon(pred))
        goals = [Decimal(answer)] + ([Decimal(alt)] if alt else [])
    except (InvalidOperation, TypeError, ValueError):
        return False
    if p in goals:
        return True
    if decimals is None:
        return False
    q = Decimal(1).scaleb(-max(0, decimals))
    return p.quantize(q, ROUND_HALF_UP) in goals or p.quantize(q, ROUND_HALF_EVEN) in goals
