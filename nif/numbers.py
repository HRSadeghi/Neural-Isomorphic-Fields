"""Exact number handling for the tests: numbers are fractions.Fraction, `canonical` gives the canonical
string of Section 3.1 of the paper. Pure Python and NumPy; nothing here depends on the model.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Canonical strings and the domain D = Z[1/10]
# ---------------------------------------------------------------------------

# Set to False if the model's canonical strings carry no "+" for non-negative numbers.
WRITE_PLUS_SIGN = True


def strip_2_5(q: int) -> int:
    """Remove all factors 2 and 5 from a positive integer."""
    while q % 2 == 0:
        q //= 2
    while q % 5 == 0:
        q //= 5
    return q


def in_D(x: Fraction) -> bool:
    """True iff x has a finite decimal expansion (Lemma 3.1(b))."""
    return strip_2_5(Fraction(x).denominator) == 1


def is_unit_of_D(x: Fraction) -> bool:
    """True iff x and 1/x are both in D, i.e. x = +-2^i 5^j (Lemma 3.1(c))."""
    x = Fraction(x)
    return x != 0 and strip_2_5(abs(x.numerator)) == 1 and strip_2_5(x.denominator) == 1


def canonical(x: Fraction, plus_sign: Optional[bool] = None) -> str:
    """Canonical string c(a) of a decimal fraction (Section 3.1)."""
    plus_sign = WRITE_PLUS_SIGN if plus_sign is None else plus_sign
    x = Fraction(x)
    if not in_D(x):
        raise ValueError(f"{x} has no finite decimal representation")
    sign = "-" if x < 0 else ("+" if plus_sign else "")
    a = abs(x)
    ip = a.numerator // a.denominator
    frac = a - ip
    s = sign + str(ip)
    if frac:
        k, f = 0, frac
        while f.denominator != 1:  # minimal k with frac * 10^k integer -> no trailing zeros
            f *= 10
            k += 1
        s += "." + str(f.numerator).rjust(k, "0")
    return s


def parse(s: str) -> Optional[Fraction]:
    """Parse a (possibly non-canonical) decoded string; None if malformed."""
    s = s.strip()
    if not s:
        return None
    sign = 1
    if s[0] in "+-":
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    if s.count(".") > 1 or not s or not all(c.isdigit() or c == "." for c in s) or s == ".":
        return None
    if "." in s:
        ip, fp = s.split(".")
        val = Fraction(int(ip or "0")) + (Fraction(int(fp), 10 ** len(fp)) if fp else 0)
    else:
        val = Fraction(int(s))
    return sign * val


def num_digits(x: Fraction) -> int:
    """Number of digits of the canonical string (sign and decimal point excluded)."""
    return sum(c.isdigit() for c in canonical(x))


def in_Dn(x: Fraction, n: int) -> bool:
    return in_D(x) and num_digits(x) <= n


# ---------------------------------------------------------------------------
# The length-based sampler of Section 5.1 (Lemma 5.1)
# ---------------------------------------------------------------------------

class NumberSampler:
    """Integer- and fractional-part lengths ~ negative binomial (failures before the
    r-th success, success probability p), i.e. g(l) = C(l+r-1, l) p^r (1-p)^l.
    With r=2, p=0.45: g(0)=0.2025, g(1)=0.2228 (as in Figure 4 of the paper).
    Pairs of zero lengths are redrawn. Numbers with more than `max_digits` digits are
    rejected (truncation to D_n)."""

    def __init__(self, r: int = 2, p: float = 0.45, max_digits: int = 25,
                 sign_prob_negative: float = 0.5, seed: int = 0):
        self.r, self.p, self.max_digits = r, p, max_digits
        self.q_neg = sign_prob_negative
        self.rng = np.random.default_rng(seed)

    def set_max_digits(self, n: int) -> None:  # used by the length curriculum
        self.max_digits = n

    def sample(self) -> Fraction:
        rng = self.rng
        while True:
            l1 = int(rng.negative_binomial(self.r, self.p))
            l2 = int(rng.negative_binomial(self.r, self.p))
            if l1 == 0 and l2 == 0:
                continue
            ip = "".join(map(str, rng.integers(0, 10, size=l1))) if l1 else "0"
            fp = "".join(map(str, rng.integers(0, 10, size=l2))) if l2 else ""
            x = Fraction(int(ip)) + (Fraction(int(fp), 10 ** l2) if l2 else 0)
            if rng.random() < self.q_neg:
                x = -x
            if num_digits(x) <= self.max_digits:
                return x

    def sample_many(self, k: int) -> List[Fraction]:
        return [self.sample() for _ in range(k)]

    # --- admissible tuples (Eq. (3) of the paper) -------------------------
    def pairs(self, k: int, op: str, max_tries: int = 10 ** 7) -> List[Tuple[Fraction, Fraction]]:
        out, tries = [], 0
        while len(out) < k and tries < max_tries:
            tries += 1
            a, b = self.sample(), self.sample()
            r = a + b if op == "+" else a * b
            if num_digits(r) <= self.max_digits:
                out.append((a, b))
        return out

    def triples(self, k: int, kind: str) -> List[Tuple[Fraction, Fraction, Fraction]]:
        """kind in {'assoc+', 'assocx', 'distrib'}; all intermediate results in D_n."""
        n, out = self.max_digits, []
        while len(out) < k:
            a, b, c = self.sample(), self.sample(), self.sample()
            if kind == "assoc+":
                inter = [a + b, b + c, a + b + c]
            elif kind == "assocx":
                inter = [a * b, b * c, a * b * c]
            else:
                inter = [b + c, a * b, a * c, a * (b + c)]
            if all(num_digits(v) <= n for v in inter):
                out.append((a, b, c))
        return out

    def chains(self, k: int, K: int, op: str, operand_digits: Tuple[int, int]) -> List[List[Fraction]]:
        """Chains x_1..x_K whose operands have digits in operand_digits and whose
        partial results all lie in D_n (Proposition 3.26)."""
        lo, hi = operand_digits
        out = []
        while len(out) < k:
            xs = []
            while len(xs) < K:
                x = self.sample()
                if lo <= num_digits(x) <= hi:
                    xs.append(x)
            acc, ok = xs[0], True
            for x in xs[1:]:
                acc = acc + x if op == "+" else acc * x
                if num_digits(acc) > self.max_digits:
                    ok = False
                    break
            if ok:
                out.append(xs)
        return out


# ---------------------------------------------------------------------------
# Special test sets
# ---------------------------------------------------------------------------

def units_in_Dn(n: int, max_exp: int = 40) -> List[Fraction]:
    """All units a = +-2^i 5^j of D with a and 1/a in D_n (multiplicative invertibility test)."""
    out = set()
    for i in range(-max_exp, max_exp + 1):
        for j in range(-max_exp, max_exp + 1):
            a = Fraction(2) ** i * Fraction(5) ** j
            if num_digits(a) <= n and num_digits(1 / a) <= n:
                out.add(a)
                out.add(-a)
    return sorted(out)


def fraction_pairs(k: int, max_int_digits: int, seed: int = 0) -> List[Tuple[int, int]]:
    """Rationals p/q in lowest terms with q > 0 divisible by a prime other than 2 and 5,
    i.e. p/q not in D (fraction-field tests, Section 4)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < k:
        dp = int(rng.integers(1, max_int_digits + 1))
        dq = int(rng.integers(1, max_int_digits + 1))
        p = int(rng.integers(10 ** (dp - 1) if dp > 1 else 1, 10 ** dp))
        q = int(rng.integers(10 ** (dq - 1) if dq > 1 else 2, 10 ** dq))
        if rng.random() < 0.5:
            p = -p
        if q > 1 and math.gcd(abs(p), q) == 1 and strip_2_5(q) != 1:
            out.append((p, q))
    return out


def length_bin(x: Fraction, bins: Sequence[Tuple[int, int]]) -> Optional[int]:
    d = num_digits(x)
    for i, (lo, hi) in enumerate(bins):
        if lo <= d <= hi:
            return i
    return None


if __name__ == "__main__":  # quick self-test
    assert canonical(Fraction(-314, 100)) == "-3.14"
    assert canonical(Fraction(1, 2)) == "+0.5"
    assert canonical(Fraction(0)) == "+0"
    assert parse("007.50") == Fraction(15, 2)
    assert not in_D(Fraction(1, 3)) and in_D(Fraction(1, 40))
    assert is_unit_of_D(Fraction(4, 25)) and not is_unit_of_D(Fraction(3))
    s = NumberSampler(seed=1)
    xs = s.sample_many(2000)
    assert all(in_D(x) and num_digits(x) <= 25 for x in xs)
    assert all((a + b) == parse(canonical(a + b)) for a, b in s.pairs(200, "+"))
    print("units with up to 6 digits:", len(units_in_Dn(6)), "e.g.", [str(u) for u in units_in_Dn(6)[:5]])
    print("nif.numbers self-test passed")
