"""Tests of the paper (Table 3): reconstruction, identity, inverses, closure, associativity, distributivity,
chains, order and the field of fractions, and the quantities of Section 3.6 (defects, decoding radii,
Lipschitz constants, capacity).

Every test returns a dict with
    'em'       : exact match (%), the success rate of Table 3,
    'acc'      : token accuracy (%) where defined,
    'per_item' : list of 0/1 outcomes (for McNemar's test between two models),
    'n'        : number of evaluated tuples.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .backend import randn_like, rownorm, scale_rows, spectral_norm, swap_first_two, to_numpy
from .numbers import NumberSampler, canonical, fraction_pairs, length_bin, num_digits, parse, units_in_Dn

ORACLE_LENGTH = True     # pass the target strings to decode() (see nif_interface.decode)
BATCH = 512

SMALLER, LARGER, EQUAL = 0, 1, 2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _chunks(seq: Sequence, size: int = BATCH):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def enc(model, xs: Sequence[Fraction]):
    return model.encode([canonical(x) for x in xs])


def dec(model, z, targets: Sequence[str]) -> List[str]:
    return model.decode(z, target_strings=list(targets) if ORACLE_LENGTH else None)


def op_fn(model, op: str) -> Callable:
    return model.add if op == "+" else model.mul


def token_acc(pred: str, target: str) -> float:
    return sum(1 for i, c in enumerate(target) if i < len(pred) and pred[i] == c) / len(target)


def summarize(em_items: List[int], acc_items: List[float] = None) -> Dict:
    out = {"n": len(em_items), "per_item": em_items,
           "em": 100.0 * float(np.mean(em_items)) if em_items else float("nan")}
    if acc_items is not None:
        out["acc"] = 100.0 * float(np.mean(acc_items)) if acc_items else float("nan")
    return out


def _true_rel(a: Fraction, b: Fraction) -> int:
    return SMALLER if a < b else (LARGER if a > b else EQUAL)


def _strict_class(w) -> List[int]:
    """Relation of Definition 3.7: class i only if w_i is strictly the largest;
    ties and 'equal' map to EQUAL."""
    out = []
    for p0, p1, p2 in to_numpy(w).tolist():
        if p0 > max(p1, p2):
            out.append(SMALLER)
        elif p1 > max(p0, p2):
            out.append(LARGER)
        else:
            out.append(EQUAL)
    return out


def omega_bar(model, h1, h2):
    """Antisymmetrized comparator, Eq. (11)."""
    return 0.5 * (model.compare(h1, h2) + swap_first_two(model.compare(h2, h1)))


def rel(model, h1, h2, antisym: bool = True) -> List[int]:
    return _strict_class(omega_bar(model, h1, h2) if antisym else model.compare(h1, h2))


# ---------------------------------------------------------------------------
# Table 4: identity, invertibility, closure, associativity, distributivity
# ---------------------------------------------------------------------------

def reconstruction(model, xs) -> Dict:
    em, acc = [], []
    for b in _chunks(xs):
        t = [canonical(x) for x in b]
        p = dec(model, enc(model, b), t)
        em += [int(pi == ti) for pi, ti in zip(p, t)]
        acc += [token_acc(pi, ti) for pi, ti in zip(p, t)]
    return summarize(em, acc)


def identity(model, xs, op: str) -> Dict:
    e = Fraction(0) if op == "+" else Fraction(1)
    em, acc = [], []
    for b in _chunks(xs):
        z = op_fn(model, op)(enc(model, b), enc(model, [e] * len(b)))
        t = [canonical(x) for x in b]
        p = dec(model, z, t)
        em += [int(pi == ti) for pi, ti in zip(p, t)]
        acc += [token_acc(pi, ti) for pi, ti in zip(p, t)]
    return summarize(em, acc)


def invertibility(model, xs, op: str) -> Dict:
    """op '+': a and -a for the given xs. op 'x': xs must be units of D (units_in_Dn)."""
    e = Fraction(0) if op == "+" else Fraction(1)
    em, acc = [], []
    for b in _chunks(xs):
        inv = [-x for x in b] if op == "+" else [1 / x for x in b]
        z = op_fn(model, op)(enc(model, b), enc(model, inv))
        t = [canonical(e)] * len(b)
        p = dec(model, z, t)
        em += [int(pi == ti) for pi, ti in zip(p, t)]
        acc += [token_acc(pi, ti) for pi, ti in zip(p, t)]
    return summarize(em, acc)


def closure(model, pairs, op: str) -> Dict:
    em, acc = [], []
    for b in _chunks(pairs):
        A, B = [x for x, _ in b], [y for _, y in b]
        z = op_fn(model, op)(enc(model, A), enc(model, B))
        t = [canonical(x + y if op == "+" else x * y) for x, y in b]
        p = dec(model, z, t)
        em += [int(pi == ti) for pi, ti in zip(p, t)]
        acc += [token_acc(pi, ti) for pi, ti in zip(p, t)]
    return summarize(em, acc)


def associativity(model, triples, op: str) -> Dict:
    f = op_fn(model, op)
    em, acc = [], []
    for b in _chunks(triples):
        A, B, C = (enc(model, [t[i] for t in b]) for i in range(3))
        zl, zr = f(f(A, B), C), f(A, f(B, C))
        t = [canonical(a + bb + c if op == "+" else a * bb * c) for a, bb, c in b]
        pl, pr = dec(model, zl, t), dec(model, zr, t)
        em += [int(l == ti and r == ti) for l, r, ti in zip(pl, pr, t)]
        acc += [0.5 * (token_acc(l, ti) + token_acc(r, ti)) for l, r, ti in zip(pl, pr, t)]
    return summarize(em, acc)


def distributivity(model, triples) -> Dict:
    em, acc = [], []
    for b in _chunks(triples):
        A, B, C = (enc(model, [t[i] for t in b]) for i in range(3))
        zl = model.mul(A, model.add(B, C))
        zr = model.add(model.mul(A, B), model.mul(A, C))
        t = [canonical(a * (bb + c)) for a, bb, c in b]
        pl, pr = dec(model, zl, t), dec(model, zr, t)
        em += [int(l == ti and r == ti) for l, r, ti in zip(pl, pr, t)]
        acc += [0.5 * (token_acc(l, ti) + token_acc(r, ti)) for l, r, ti in zip(pl, pr, t)]
    return summarize(em, acc)


def by_length(model, sampler: NumberSampler, bins, n_per_bin: int) -> Dict:
    """By number length: identity (x1), closure (x) and reconstruction by digits of the longer operand."""
    rows = {"identity_x": [], "closure_x": [], "reconstruction": []}
    for lo, hi in bins:
        xs, pairs = [], []
        while len(xs) < n_per_bin:
            x = sampler.sample()
            if lo <= num_digits(x) <= hi:
                xs.append(x)
        while len(pairs) < n_per_bin:
            a, b = sampler.pairs(1, "x")[0]
            if lo <= max(num_digits(a), num_digits(b)) <= hi:
                pairs.append((a, b))
        rows["identity_x"].append(identity(model, xs, "x")["em"])
        rows["closure_x"].append(closure(model, pairs, "x")["em"])
        rows["reconstruction"].append(reconstruction(model, xs)["em"])
    return rows


# ---------------------------------------------------------------------------
# Table 4: order and fraction-field tests
# ---------------------------------------------------------------------------

def order_tests(model, sampler: NumberSampler, n: int, equal_frac: float = 0.1, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    out = {}
    # (a) correct relation with the antisymmetrized comparator, incl. equal pairs
    pairs = []
    for _ in range(n):
        a = sampler.sample()
        b = a if rng.random() < equal_frac else sampler.sample()
        pairs.append((a, b))
    em, consist = [], []
    for b in _chunks(pairs):
        A, B = enc(model, [x for x, _ in b]), enc(model, [y for _, y in b])
        pred = rel(model, A, B)
        em += [int(p == _true_rel(x, y)) for p, (x, y) in zip(pred, b)]
        raw_ab, raw_ba = rel(model, A, B, False), rel(model, B, A, False)
        swap = {SMALLER: LARGER, LARGER: SMALLER, EQUAL: EQUAL}
        consist += [int(r1 == swap[r2]) for r1, r2 in zip(raw_ab, raw_ba)]
    out["order"] = summarize(em)
    out["antisymmetry_consistency_raw"] = summarize(consist)
    # (b) transitivity on triples whose two premises are predicted
    trans = []
    triples = [(sampler.sample(), sampler.sample(), sampler.sample()) for _ in range(6 * n)]
    for b in _chunks(triples):
        A, B, C = (enc(model, [t[i] for t in b]) for i in range(3))
        ab, bc, ac = rel(model, A, B), rel(model, B, C), rel(model, A, C)
        for x, y, z in zip(ab, bc, ac):
            if x == SMALLER and y == SMALLER:
                trans.append(int(z == SMALLER))
            elif x == LARGER and y == LARGER:
                trans.append(int(z == LARGER))
    out["transitivity"] = summarize(trans)
    # (c) compatibility with addition (O1): b < c  =>  a+b < a+c
    o1, trip = [], []
    while len(trip) < n:
        a, b, c = sampler.sample(), sampler.sample(), sampler.sample()
        if b != c and max(num_digits(a + b), num_digits(a + c)) <= sampler.max_digits:
            trip.append((a, min(b, c), max(b, c)))
    for tt in _chunks(trip):
        A, B, C = (enc(model, [x[i] for x in tt]) for i in range(3))
        pred = rel(model, model.add(A, B), model.add(A, C))
        o1 += [int(p == SMALLER) for p in pred]
    out["compat_O1"] = summarize(o1)
    # (d) compatibility with multiplication (O2): 0 < a, 0 < b  =>  0 < a*b
    o2 = []
    pos = [(abs(a), abs(b)) for a, b in sampler.pairs(n, "x") if a != 0 and b != 0]
    for b in _chunks(pos):
        A, B = enc(model, [x for x, _ in b]), enc(model, [y for _, y in b])
        Z = enc(model, [Fraction(0)] * len(b))
        pred = rel(model, Z, model.mul(A, B))
        o2 += [int(p == SMALLER) for p in pred]
    out["compat_O2"] = summarize(o2)
    return out


def _decode_fraction(model, num_z, den_z, num_t: List[int], den_t: List[int]):
    """Decode numerator and denominator; ORACLE_LENGTH uses the true lengths."""
    pn = dec(model, num_z, [canonical(Fraction(v)) for v in num_t])
    pd = dec(model, den_z, [canonical(Fraction(v)) for v in den_t])
    vals = []
    for a, b in zip(pn, pd):
        va, vb = parse(a), parse(b)
        vals.append(None if va is None or vb is None or vb == 0 else va / vb)
    return vals


def fraction_field_tests(model, n: int, max_int_digits: int = 4, seed: int = 0) -> Dict:
    """Corollary 3.17: arithmetic with rationals outside D via pairs (phi(p), phi(q))."""
    fr1 = fraction_pairs(n, max_int_digits, seed)
    fr2 = fraction_pairs(n, max_int_digits, seed + 1)
    out = {k: [] for k in ("add", "mul", "order")}
    for i in range(0, n, BATCH):
        b1, b2 = fr1[i:i + BATCH], fr2[i:i + BATCH]
        P1, Q1 = enc(model, [Fraction(p) for p, _ in b1]), enc(model, [Fraction(q) for _, q in b1])
        P2, Q2 = enc(model, [Fraction(p) for p, _ in b2]), enc(model, [Fraction(q) for _, q in b2])
        # addition, Eq. (22)
        num = model.add(model.mul(P1, Q2), model.mul(P2, Q1))
        den = model.mul(Q1, Q2)
        nt = [p1 * q2 + p2 * q1 for (p1, q1), (p2, q2) in zip(b1, b2)]
        dt = [q1 * q2 for (_, q1), (_, q2) in zip(b1, b2)]
        vals = _decode_fraction(model, num, den, nt, dt)
        out["add"] += [int(v is not None and v == Fraction(p1, q1) + Fraction(p2, q2))
                       for v, (p1, q1), (p2, q2) in zip(vals, b1, b2)]
        # multiplication, Eq. (23)
        num, den = model.mul(P1, P2), model.mul(Q1, Q2)
        nt = [p1 * p2 for (p1, _), (p2, _) in zip(b1, b2)]
        vals = _decode_fraction(model, num, den, nt, dt)
        out["mul"] += [int(v is not None and v == Fraction(p1, q1) * Fraction(p2, q2))
                       for v, (p1, q1), (p2, q2) in zip(vals, b1, b2)]
        # order with positive denominators, Eq. (24)
        pred = rel(model, model.mul(P1, Q2), model.mul(P2, Q1))
        out["order"] += [int(p == _true_rel(Fraction(p1, q1), Fraction(p2, q2)))
                         for p, (p1, q1), (p2, q2) in zip(pred, b1, b2)]
    return {k: summarize(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Sections 3.6 and 5.3.3: defects, decoding radii, Lipschitz constants, chains (Table A.5, Figure 10)
# ---------------------------------------------------------------------------

def defects(model, pairs, op: str) -> List[float]:
    out = []
    for b in _chunks(pairs):
        A, B = enc(model, [x for x, _ in b]), enc(model, [y for _, y in b])
        T = enc(model, [x + y if op == "+" else x * y for x, y in b])
        out += rownorm(op_fn(model, op)(A, B) - T)
    return out


def empirical_lipschitz(model, sampler: NumberSampler, op: str, n: int) -> Dict:
    """Largest ratio ||u*v - u'*v|| / ||u - u'|| over test triples (a lower bound of L)."""
    ratios = []
    for _ in range(0, n, BATCH):
        k = min(BATCH, n - len(ratios))
        U, U2, V = (enc(model, sampler.sample_many(k)) for _ in range(3))
        f = op_fn(model, op)
        num, den = rownorm(f(U, V) - f(U2, V)), rownorm(U - U2)
        ratios += [a / b for a, b in zip(num, den) if b > 0]
    return {"max": float(np.max(ratios)), "p99": float(np.percentile(ratios, 99))}


def spectral_bound(model, op: str) -> float:
    """prod_i (||W1^(i)||_2 + ||W2^(i)||_2), Eq. (31)."""
    prod = 1.0
    for w in model.ac_layer_weights(op):
        w = to_numpy(w)
        m = w.shape[0]
        prod *= spectral_norm(w[:, :m]) + spectral_norm(w[:, m:])
    return prod


def radius_estimates(model, xs: Sequence[Fraction], r: float, n_dir: int = 100, seed: int = 0) -> List[float]:
    """rho_hat(a) = min(r, m(a) / (2 L_hat)) (Lemma 3.23) with L_hat estimated from
    n_dir random perturbations of norm r. This is an ESTIMATE, not a certificate."""
    rng = np.random.default_rng(seed)
    by_len: Dict[int, List[int]] = {}
    strings = [canonical(x) for x in xs]
    for i, s in enumerate(strings):
        by_len.setdefault(len(s), []).append(i)
    rho = [0.0] * len(xs)
    for idx in by_len.values():
        for b in _chunks(idx, 64):
            t = [strings[i] for i in b]
            z = model.encode(t)
            logits, ids = model.decoder_logits(z, t)
            lg, ids = to_numpy(logits), to_numpy(ids).astype(int)
            true = np.take_along_axis(lg, ids[..., None], -1)[..., 0]
            other = lg.copy()
            np.put_along_axis(other, ids[..., None], -np.inf, -1)
            margin = (true - other.max(-1)).min(-1)                       # m(a)
            L = np.zeros(len(b))
            for _ in range(n_dir):
                d = randn_like(z, rng)
                d = scale_rows(d, [r / max(v, 1e-12) for v in rownorm(d)])
                lg2 = to_numpy(model.decoder_logits(z + d, t)[0])
                L = np.maximum(L, np.abs(lg2 - lg).max(axis=(1, 2)) / r)
            for j, i in enumerate(b):
                rho[i] = float(min(r, margin[j] / (2 * L[j]))) if margin[j] > 0 and L[j] > 0 else 0.0
    return rho


def certified_wrong(model, pairs, op: str, max_digits: int, r: float, n_dir: int = 100) -> float:
    """Proposition 3.24(b): percentage of pairs whose result embedding z lies inside the
    estimated decoding ball of the (wrong) number c it is decoded to."""
    from .numbers import in_Dn
    wrong_c, wrong_z = [], []
    for b in _chunks(pairs):
        A, B_ = enc(model, [x for x, _ in b]), enc(model, [y for _, y in b])
        Z = op_fn(model, op)(A, B_)
        t = [canonical(x + y if op == "+" else x * y) for x, y in b]
        p = dec(model, Z, t)
        for i, (pi, ti) in enumerate(zip(p, t)):
            if pi != ti:
                c = parse(pi)
                if c is not None and in_Dn(c, max_digits):
                    wrong_c.append(c)
                    wrong_z.append(Z[i:i + 1])
    if not wrong_c:
        return 0.0
    rho_c = radius_estimates(model, wrong_c, r, n_dir)
    count = 0
    for c, z, rc in zip(wrong_c, wrong_z, rho_c):
        if rc > 0 and rownorm(z - enc(model, [c]))[0] < rc:
            count += 1
    return 100.0 * count / len(pairs)


def capacity(model, xs: Sequence[Fraction], rho: Sequence[float]) -> Dict:
    """Quantities of Proposition B.5: B = max ||phi(a)||, smallest positive estimated radius,
    and the implied bound N <= d log10(1 + B / rho_min) for this embedding dimension d."""
    norms, d = [], None
    for b in _chunks(list(xs)):
        Z = enc(model, b)
        d = to_numpy(Z).shape[1]
        norms += rownorm(Z)
    B = float(np.max(norms))
    pos = [v for v in rho if v > 0]
    rho_min = float(np.min(pos)) if pos else float("nan")
    n_max = float(d * np.log10(1 + B / rho_min)) if pos else float("nan")
    return {"B": B, "rho_min": rho_min, "B_over_rho_min": B / rho_min if pos else float("nan"),
            "zero_radius_pct": 100.0 * (1 - len(pos) / max(1, len(rho))), "N_max_bound": n_max, "d": d}


def table10(model, sampler: NumberSampler, n: int, r: float, n_dir: int = 100) -> Dict:
    out = {}
    L_hat = {op: empirical_lipschitz(model, sampler, op, n)["max"] for op in "+x"}
    for op in "+x":
        pairs = sampler.pairs(n, op)
        eps = defects(model, pairs, op)
        targets = [a + b if op == "+" else a * b for a, b in pairs]
        rho = radius_estimates(model, targets, r, n_dir)
        out[op] = {"defect": _q(eps), "rho": _q(rho),
                   "bound_pct": 100.0 * float(np.mean([e < p for e, p in zip(eps, rho)])),
                   "cert_wrong_pct": certified_wrong(model, pairs, op, sampler.max_digits, r, n_dir),
                   "em": closure(model, pairs, op)["em"], "L_hat": L_hat[op]}
        out[op].update(capacity(model, targets, rho))
        try:
            out[op]["L_spectral"] = spectral_bound(model, op)
        except NotImplementedError:
            out[op]["L_spectral"] = None
    # associative (+) and distributive bounds, Eqs. (33)-(34)
    Lp, Lx = L_hat["+"], L_hat["x"]
    tri = sampler.triples(n, "assoc+")
    e = lambda pr, op: defects(model, pr, op)
    ab, bc = e([(a, b) for a, b, c in tri], "+"), e([(b, c) for a, b, c in tri], "+")
    ab_c, a_bc = e([(a + b, c) for a, b, c in tri], "+"), e([(a, b + c) for a, b, c in tri], "+")
    side_l = [Lp * x + y for x, y in zip(ab, ab_c)]
    side_r = [Lp * x + y for x, y in zip(bc, a_bc)]
    rho = radius_estimates(model, [a + b + c for a, b, c in tri], r, n_dir)
    out["assoc+"] = {"bound_median": float(np.median([x + y for x, y in zip(side_l, side_r)])),
                     "bound_pct": 100.0 * float(np.mean([max(x, y) < p for x, y, p in zip(side_l, side_r, rho)])),
                     "em": associativity(model, tri, "+")["em"],
                     "defect": _q([x + y for x, y in zip(ab, bc)]), "rho": _q(rho)}
    tri = sampler.triples(n, "distrib")
    bc = e([(b, c) for a, b, c in tri], "+")
    a_bc = e([(a, b + c) for a, b, c in tri], "x")
    a_b, a_c = e([(a, b) for a, b, c in tri], "x"), e([(a, c) for a, b, c in tri], "x")
    ab_ac = e([(a * b, a * c) for a, b, c in tri], "+")
    side_l = [Lx * x + y for x, y in zip(bc, a_bc)]
    side_r = [Lp * (x + y) + z for x, y, z in zip(a_b, a_c, ab_ac)]
    rho = radius_estimates(model, [a * (b + c) for a, b, c in tri], r, n_dir)
    out["distrib"] = {"bound_median": float(np.median([x + y for x, y in zip(side_l, side_r)])),
                      "bound_pct": 100.0 * float(np.mean([max(x, y) < p for x, y, p in zip(side_l, side_r, rho)])),
                      "em": distributivity(model, tri)["em"],
                      "defect": _q([x + y for x, y in zip(a_b, a_c)]), "rho": _q(rho)}
    return out


def chains(model, sampler: NumberSampler, op: str, Ks=range(2, 11), n: int = 1000,
           operand_digits=(1, 6)) -> Dict:
    """Figure 10d: exact match, error ||F_K - phi(s_K)|| and the bound of Eq. (32)."""
    L = empirical_lipschitz(model, sampler, op, n)["max"]
    f = op_fn(model, op)
    res = {}
    for K in Ks:
        cs = sampler.chains(n, K, op, operand_digits)
        em, err, bound = [], [], []
        for b in _chunks(cs, 256):
            F = enc(model, [c[0] for c in b])
            s = [c[0] for c in b]
            epss = []
            for j in range(1, K):
                xj = [c[j] for c in b]
                X = enc(model, xj)
                # defect of the exact partial result with the next operand
                epss.append(rownorm(f(enc(model, s), X) - enc(model, [u + v if op == "+" else u * v for u, v in zip(s, xj)])))
                F = f(F, X)
                s = [u + v if op == "+" else u * v for u, v in zip(s, xj)]
            t = [canonical(v) for v in s]
            p = dec(model, F, t)
            em += [int(pi == ti) for pi, ti in zip(p, t)]
            err += rownorm(F - enc(model, s))
            for i in range(len(b)):
                bound.append(sum(L ** (K - 1 - j) * epss[j - 1][i] for j in range(1, K)))
        res[K] = {"em": 100.0 * float(np.mean(em)), "err": _q(err), "bound": _q(bound)}
    return {"L_hat": L, "per_K": res}


def _q(v: Sequence[float]) -> Dict:
    v = np.asarray(v, dtype=float)
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
