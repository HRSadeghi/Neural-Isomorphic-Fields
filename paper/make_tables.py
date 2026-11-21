r"""Tables, text macros and CSV files of the paper, built from the logs and test results in runs/.

    python paper/make_tables.py [--main main_s0,main_s1,main_s2] [--out paper/generated]

Inputs (written by train.py, train_operator.py, evaluate.py and analysis/):
    runs/<name>/config.json, metrics.jsonl, evals.jsonl, test_results.json, lengths_parts.json
    results/analysis/*.json
Outputs:
    <out>/tab_main.tex           Table 4: all tests, both decoding modes, mean +- SD over the seeds
    <out>/tab_abl_operator.tex   Table A.2: operators trained alone
    <out>/tab_abl_full.tex       Table A.3: ablations of the full model, with McNemar's tests
    <out>/tab_lengthdim.tex      Table A.4: number length and embedding dimension
    <out>/tab_geometry.tex       Table A.5: defects, decoding radii, Lipschitz constants
    <out>/fig_training.pdf       Figure 6: validation exact match during training
    <out>/numbers.tex            macros for the numbers quoted in the text
    results/csv/*.csv            every number behind the tables and figures, and the training logs as CSV
A table cell whose run is missing is written as \tbd.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import gzip
import statistics as st
import sys

import length_tables as length_assets

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, "runs")

# reference categorical palette (dataviz skill, light mode), fixed order; secondary encoding by line style
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

MAIN_ROWS = [
    ("Reconstruction", [("reconstruction", "Reconstruction")]),
    ("Addition", [("identity+", "Identity"), ("closure+", "Closure"), ("invertibility+", "Inverse"),
                  ("assoc+", "Associativity")]),
    ("Multiplication", [("identityx", "Identity"), ("closurex", "Closure"), ("invertibilityx", "Inverse (units)"),
                        ("assocx", "Associativity")]),
    ("Both operations", [("distrib", "Distributivity")]),
    ("Order", [("order_order", r"Relation $\bar\omega$"), ("order_antisymmetry_consistency_raw", r"Antisymmetry of $\omega$"),
               ("order_transitivity", "Transitivity"), ("order_compat_O1", r"Compatibility with $+$ (O1)"),
               ("order_compat_O2", r"Compatibility with $\times$ (O2)")]),
    ("Fraction field", [("fraction_add", "Addition"), ("fraction_mul", "Multiplication"),
                        ("fraction_order", "Order")]),
    ("Chains", [("chains+@5", r"Sums, $M=5$"), ("chains+@10", r"Sums, $M=10$"),
                ("chainsx@5", r"Products, $M=5$"), ("chainsx@10", r"Products, $M=10$")]),
]
MODES = [("N25_length_free", "length-free"), ("N25_oracle", "oracle length")]

PROBES = [
    ("abl_x_full", r"$\dot{\times}$: staged operator (Section~\ref{sec:staged})"),
    ("abl_x_nohints", r"\quad without hint supervision"),
    ("abl_x_norefit", r"\quad without least-squares fit of the counts"),
    ("abl_x_fourier_random", r"\quad random instead of decimal frequencies"),
    ("abl_x_fourier_none", r"\quad without periodic features"),
    ("abl_x_softcounts", r"\quad soft instead of one-hot digit counts"),
    ("abl_x_bf16", r"\quad bfloat16 instead of float32"),
    ("abl_x_coupled", r"$\dot{\times}$: generic AC layers with a place-coupled bilinear branch, $k=6$"),
    ("abl_x_dense", r"$\dot{\times}$: perceptron AC layers, $f(x)=\sigma(Wx+b)$, $k=4$"),
    ("abl_add_slot", r"$\dot{+}$: slot-Transformer AC layers (Section~\ref{sec:ano}), $k=6$"),
    ("abl_add_dense", r"$\dot{+}$: perceptron AC layers, $f(x)=\sigma(Wx+b)$, $k=4$"),
    ("depth_x_dense_k1", r"$\dot{\times}$: perceptron AC layers, $k=1$"),
    ("depth_x_dense_k2", r"$\dot{\times}$: perceptron AC layers, $k=2$"),
    ("depth_x_dense_k6", r"$\dot{\times}$: perceptron AC layers, $k=6$"),
    ("depth_x_coupled_k2", r"$\dot{\times}$: generic AC layers with the bilinear branch, $k=2$"),
    ("depth_x_coupled_k4", r"$\dot{\times}$: generic AC layers with the bilinear branch, $k=4$"),
    ("depth_x_coupled_k8", r"$\dot{\times}$: generic AC layers with the bilinear branch, $k=8$"),
]
FULL_ABL = [
    (None, "NIF (all terms)"),
    ("full_paper_losses", r"only $\ell_{\mathrm{rec}}$, $\ell_{\mathrm{iso}}$, $\ell_{\mathrm{ord}}$"),
    ("full_no_hints", "without hint supervision"),
    ("full_no_targeted", "without targeted pairs and terms on operator outputs"),
    ("full_no_length_mix", "training numbers only from the sampler (no uniform lengths)"),
    ("full_dense_ops", "perceptron AC layers in both operators"),
    ("aeonly", "AE-only encoder (trained with $\\ell_{\\mathrm{rec}}$ only, then frozen)"),
    ("emb_c2", "embedding size $c=2$ ($d=104$)"),
    ("emb_c4", "embedding size $c=4$ ($d=208$)"),
    ("emb_c6", "embedding size $c=6$ ($d=312$)"),
    ("emb_c16", "embedding size $c=16$ ($d=832$)"),
]
FULL_COLS = [("reconstruction", "Rec."), ("closure+", r"Cl.\,$\dot{+}$"), ("closurex", r"Cl.\,$\dot{\times}$"),
             ("invertibility+", r"Inv.\,$\dot{+}$"), ("invertibilityx", r"Inv.\,$\dot{\times}$"), ("assoc+", r"Ass.\,$\dot{+}$"),
             ("assocx", r"Ass.\,$\dot{\times}$"), ("distrib", "Distr."), ("order_compat_O1", "O1"),
             ("chains+@5", r"$\dot{+}$, $M{=}5$")]
TBD = r"\tbd"


# ----------------------------------------------------------------------------- reading
def jload(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def jlines(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return out


def tests(name):
    return jload(os.path.join(RUNS, name, "test_results.json")) or {}


def items(name):
    """Per-item 0/1 outcomes written by evaluate.py (for McNemar's test)."""
    try:
        with gzip.open(os.path.join(RUNS, name, "test_results.items.json.gz"), "rt") as f:
            out = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return out


ITEM_KEY = {"order_compat_O1": "order.compat_O1", "order_order": "order.order"}


def metric(res, mode, key, field="em"):
    r = res.get(mode, {})
    if "@" in key:
        name, K = key.split("@")
        v = r.get(name, {}).get("per_K", {}).get(K, {})
        return v.get(field)
    return r.get(key, {}).get(field)


def agg(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None, None, 0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else None), len(v)


def fmt(m, s=None, digits=2):
    if m is None:
        return TBD
    out = f"{m:.{digits}f}"
    if s is not None:
        out += rf"\,$\pm$\,{s:.{digits}f}"
    return out


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------------- tables
def table_main(main, out, csvdir):
    R = [tests(n) for n in main]
    have = [r for r in R if r]
    seeds = len(have)
    rows, csv_rows = [], []
    for group, items in MAIN_ROWS:
        rows.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{group}}}}} \\")
        for key, label in items:
            ns = [metric(r, MODES[0][0], key, "n") for r in have]
            n = next((x for x in ns if x is not None), None)
            acc = agg([metric(r, MODES[0][0], key, "acc") for r in have])
            em_lf = agg([metric(r, MODES[0][0], key) for r in have])
            em_or = agg([metric(r, MODES[1][0], key) for r in have])
            if "@" in key:
                n = 500
            acc_s = fmt(*acc[:2]) if acc[0] is not None else ("--" if have else TBD)
            rows.append(rf"\quad {label} & {n if n is not None else TBD} & {acc_s} & {fmt(*em_lf[:2])} & {fmt(*em_or[:2])} \\")
            for name, r in zip(main, R):
                for mode, _ in MODES:
                    csv_rows.append([name, mode, group, key, metric(r, mode, key, "n"), metric(r, mode, key),
                                     metric(r, mode, key, "acc")])
    seed_txt = f"mean $\\pm$ SD over {seeds} seeds" if seeds > 1 else ("one seed" if seeds == 1 else r"\tbd{} seeds")
    tex = rf"""% generated by paper/make_tables.py -- do not edit
\begin{{table}}[pos=htbp]
	\centering
	\caption{{Results of NIF on the tests of Table~\ref{{tab:test_definitions}} (test sets, {seed_txt}).
	Accuracy is the token-level accuracy and EM the exact match (\%), both with the length chosen by the decoder (length-free);
	the last column gives the exact match when the decoder is given the true length (the protocol of Table~\ref{{tab:performance}}).
	$n$ is the number of test items. Chains: left folds of $M$ operands (Proposition~\ref{{prop:chains}}). Inverse (units): all units of $\Dn$; the units whose digit products fit the slot window reach \invInside\% (Section~\ref{{subsec:results}}). Results by number of digits: Figure~\ref{{fig:lengths}}.}}
	\label{{tab:main}}
	\renewcommand{{\arraystretch}}{{1.08}}
	\footnotesize
	\begin{{tabular}}{{@{{}}lrccc@{{}}}}
		\toprule
		\textbf{{Test}} & $n$ & \textbf{{Accuracy}} & \textbf{{EM}} & \textbf{{EM, oracle length}} \\
		\midrule
		{chr(10).join(chr(9) * 2 + r for r in rows)}
		\bottomrule
	\end{{tabular}}
\end{{table}}
"""
    write(os.path.join(out, "tab_main.tex"), tex)
    write_csv(os.path.join(csvdir, "main_results.csv"), csv_rows, ["run", "mode", "group", "test", "n", "em", "acc"])
    return have


def probe_summary(name):
    ev = jlines(os.path.join(RUNS, name, "evals.jsonl"))
    if not ev:
        return None
    ev.sort(key=lambda e: e["step"])
    last = max(ev, key=lambda e: e["em"])
    first99 = next((e["step"] for e in ev if e["em"] >= 99.0), None)
    stg = last.get("stages") or {}
    return {"final": last["em"], "best": max(e["em"] for e in ev), "long": last.get("by_len", {}).get("21-25"),
            "first99": first99, "steps": ev[-1]["step"], "digits": stg.get("digits"), "LH": stg.get("LH_within_0.5"),
            "T": stg.get("T"), "done": os.path.exists(os.path.join(RUNS, name, "DONE")), "curve": ev}


def table_ablation_operator(out, csvdir):
    rows, csv_rows = [], []
    for name, label in PROBES:
        s = probe_summary(name)
        if s is None:
            rows.append(rf"{label} & {TBD} & {TBD} & {TBD} & {TBD} & {TBD} & {TBD} & {TBD} \\")
            continue
        f = lambda v, d=2: TBD if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
        mark = "" if s["done"] else r"$^{*}$"
        first = f(s["first99"]) if s["first99"] is not None else "--"
        g = lambda v: "--" if v is None else f(v)          # stage accuracies exist only for the staged operator
        rows.append(rf"{label} & {f(s['final'])}{mark} & {f(s['long'])} & {s['steps']} & {first} & {g(s['digits'])} & {g(s['LH'])} & {g(s['T'])} \\")
        csv_rows.append([name, s["steps"], s["final"], s["best"], s["long"], s["first99"], s["digits"], s["LH"], s["T"], s["done"]])
        if name in ("abl_x_dense", "abl_x_bf16", "abl_add_dense"):
            rows.append(r"\midrule")
    unfinished = r" $^{*}$Run not finished." if any("$^{*}$" in r for r in rows) else ""
    tex = rf"""% generated by paper/make_tables.py -- do not edit
\begin{{table}}[pos=htbp]
	\centering
	\caption{{Operator ablations. Each operator is trained in isolation until convergence (Section~\ref{{subsec:setup}}; at most 8000 steps) on fixed one-hot digit slots (no encoder or decoder), and a linear head reads the digits of the result. EM: exact match (\%) on 4096 held-out pairs of the sampler of Section~\ref{{subsec:distribution_and_numerical_dataset}}, overall and for results with 21--25 digits, at the best evaluation; steps: training steps until convergence; first $\ge99\%$: first evaluation (every 250 steps) with $\text{{EM}}\ge 99\%$ (--: never). Stage accuracies of $\dot{{\times}}$ (\%, read by the hint heads): operand digits, $(L_k,H_k)$ within $0.5$, and $T_k$ (--: operator without these stages). The full model uses $k=4$ layers for $\dot{{+}}$ (Table~\ref{{tab:hparams}}).{unfinished}}}
	\label{{tab:abl_operator}}
	\renewcommand{{\arraystretch}}{{1.08}}
	\footnotesize
	\resizebox{{\textwidth}}{{!}}{{%
	\begin{{tabular}}{{@{{}}lccccccc@{{}}}}
		\toprule
		\textbf{{Operator}} & \textbf{{EM}} & \textbf{{EM, 21--25 digits}} & \textbf{{Steps}} & \textbf{{First $\ge99\%$}} & \textbf{{Digits}} & $(L_k,H_k)$ & $T_k$ \\
		\midrule
		{chr(10).join(chr(9) * 2 + r for r in rows)}
		\bottomrule
	\end{{tabular}}%
	}}
\end{{table}}
"""
    write(os.path.join(out, "tab_abl_operator.tex"), tex)
    write_csv(os.path.join(csvdir, "ablation_operator.csv"), csv_rows,
              ["run", "steps", "em_final", "em_best", "em_21_25", "first_step_99", "stage_digits", "stage_LH", "stage_T", "done"])
    curves = []
    for name, _ in PROBES:
        for e in jlines(os.path.join(RUNS, name, "evals.jsonl")):
            curves.append([name, e["step"], e["em"]] + [e.get("by_len", {}).get(b) for b in ("1-5", "6-10", "11-15", "16-20", "21-25")]
                          + [(e.get("stages") or {}).get(k) for k in ("digits", "LH_within_0.5", "S_mod10", "T")])
    write_csv(os.path.join(csvdir, "ablation_operator_curves.csv"), curves,
              ["run", "step", "em", "em_1_5", "em_6_10", "em_11_15", "em_16_20", "em_21_25", "digits", "LH", "S_mod10", "T"])


def table_ablation_full(main, out, csvdir):
    from nif.stats import holm, mcnemar_exact
    rows, csv_rows = [], []
    ref = items(main[0]).get(MODES[0][0], {}) if main else {}
    pvals = {}
    for name, _ in FULL_ABL:
        if name is None:
            continue
        it = items(name).get(MODES[0][0], {})
        for key, _ in FULL_COLS:
            k = ITEM_KEY.get(key, key)
            if k in it and k in ref and len(it[k]) == len(ref[k]):
                pvals[(name, key)] = mcnemar_exact(ref[k], it[k])
    padj = holm({f"{a}|{b}": v for (a, b), v in pvals.items()}) if pvals else {}
    for name, label in FULL_ABL:
        names = main if name is None else [name]
        R = [r for r in (tests(n) for n in names) if r]
        if name is not None and not R:              # ablation not run (yet): no row
            continue
        cells = []
        for key, _ in FULL_COLS:
            m, s, _ = agg([metric(r, MODES[0][0], key) for r in R])
            pa = padj.get(f"{name}|{key}")
            cells.append(fmt(m, None, 2) + (r"$^{*}$" if pa is not None and pa < 0.05 else ""))
            for n_, r in zip(names, R):
                csv_rows.append([n_, key, metric(r, MODES[0][0], key), pvals.get((name, key)), pa])
        rows.append(rf"{label} & " + " & ".join(cells) + r" \\")
        if name is None:
            r0 = tests(main[0]) if main else None
            if r0:                                   # the run that the McNemar tests compare with
                rows.append(r"NIF, seed 0 (reference of the tests) & " + " & ".join(
                    fmt(metric(r0, MODES[0][0], key), None, 2) for key, _ in FULL_COLS) + r" \\")
            rows.append(r"\midrule")
    head = " & ".join(h for _, h in FULL_COLS)
    tex = rf"""% generated by paper/make_tables.py -- do not edit
\begin{{table}}[pos=htbp]
	\centering
	\caption{{Ablations of the full model: exact match (\%, length-free decoding) on the test sets. Each variant changes one component of NIF and is trained with the same budget and seed 0; the first row is the mean over the seeds of Table~\ref{{tab:main}}, the second the seed-0 run. $^{{*}}$ (all columns but the last): different from the seed-0 run (McNemar's test, Holm--Bonferroni-corrected $p<0.05$ within the table). Rec.: reconstruction; Cl.: closure; Inv.: inverse; Ass.: associativity; Distr.: distributivity; O1: compatibility of the order with $\dot{{+}}$; last column: chains of 5 sums (500 chains). The other tests of Figure~\ref{{fig:profile}} are released with the code.}}
	\label{{tab:abl_full}}
	\renewcommand{{\arraystretch}}{{1.08}}
	\footnotesize
	\resizebox{{\textwidth}}{{!}}{{%
	\begin{{tabular}}{{@{{}}l{"c" * len(FULL_COLS)}@{{}}}}
		\toprule
		\textbf{{Variant}} & {head} \\
		\midrule
		{chr(10).join(chr(9) * 2 + r for r in rows)}
		\bottomrule
	\end{{tabular}}%
	}}
\end{{table}}
"""
    write(os.path.join(out, "tab_abl_full.tex"), tex)
    write_csv(os.path.join(csvdir, "ablation_full.csv"), csv_rows,
              ["run", "test", "em_length_free", "p_mcnemar_vs_seed0", "p_holm"])


def table_geometry(main, out, csvdir):
    R = [r.get(MODES[0][0], {}).get("geometry") for r in (tests(n) for n in main)]
    G = [g for g in R if g]
    rows, csv_rows = [], []

    def a(path, digits=2):
        vals = []
        for g in G:
            v = g
            for p in path:
                v = v.get(p) if isinstance(v, dict) else None
            vals.append(v)
        m, s, _ = agg(vals)
        return fmt(m, None, digits) if m is not None else ("--" if G else TBD)

    for key, label in (("+", r"$\dot{+}$, closure"), ("x", r"$\dot{\times}$, closure"),
                       ("assoc+", r"Associativity of $\dot{+}$"), ("distrib", "Distributivity")):
        if key in ("+", "x"):
            rows.append(rf"{label} & {a([key, 'em'])} & {a([key, 'defect', 'median'])} / {a([key, 'defect', 'p90'])} & "
                        rf"{a([key, 'rho', 'median'])} & {a([key, 'bound_pct'], 1)} & {a([key, 'cert_wrong_pct'], 1)} & {a([key, 'L_hat'])} \\")
        else:
            rows.append(rf"{label} & {a([key, 'em'])} & {a([key, 'defect', 'median'])} / {a([key, 'defect', 'p90'])} & "
                        rf"{a([key, 'rho', 'median'])} & {a([key, 'bound_pct'], 1)} & -- & -- \\")
    # embedding sizes: one row per c, each cell "+ / x" (single runs)
    def dig(v, path):
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
        return v

    extra = []
    for run, c in (("emb_c2", 2), ("emb_c4", 4), ("emb_c6", 6), ("emb_c16", 16)):
        g = tests(run).get(MODES[0][0], {}).get("geometry")
        if g:
            pair = lambda path, d=2: " / ".join(fmt(dig(g[k], path), None, d) for k in ("+", "x"))
            extra.append(rf"$c={c}$: closure of $\dot{{+}}$ / $\dot{{\times}}$ & {pair(['em'])} & {pair(['defect', 'median'])} & "
                         rf"{pair(['rho', 'median'])} & {pair(['bound_pct'], 1)} & {pair(['cert_wrong_pct'], 1)} & {pair(['L_hat'])} \\")
            main, R = list(main) + [run], list(R) + [g]          # also written to the CSV below
    if extra:
        rows += [r"\midrule"] + extra
    for n_, g in zip(main, R):
        if g:
            for key, v in g.items():
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        for q, x in vv.items():
                            csv_rows.append([n_, key, f"{kk}.{q}", x])
                    else:
                        csv_rows.append([n_, key, kk, vv])
    cap = (rf"Over the results of the closure tests, $B=\max\norm{{\varphi(a)}}={a(['x', 'B'])}$, the smallest estimated radius is "
           rf"$\hat\rho_{{\min}}={a(['x', 'rho_min'])}$ ($\dot{{\times}}$) and ${a(['+', 'rho_min'])}$ ($\dot{{+}}$), and Eq.~(\ref{{eq:capacity}}) gives "
           rf"$N\le{a(['x', 'N_max_bound'], 0)}$ for $d=520$.")
    tex = rf"""% generated by paper/make_tables.py -- do not edit
\begin{{table}}[pos=htbp]
	\centering
	\caption{{Quantities of Section~\ref{{sec:approximate}} on 2000 test tuples per test (length-free decoding, mean over seeds; EM: exact match on these tuples, a subset of the test sets of Table~\ref{{tab:main}}). Defect: $\epsadd$, $\epsmul$ (for the laws: unweighted sum of the defects in the bound of Proposition~\ref{{prop:assoc_distrib}}), median / 90th percentile; $\hat\rho$: median estimated decoding radius (Lemma~\ref{{lem:margin}}, 50 random perturbations of norm $\delta_0=6$, with the length head included in the margin); in bound: tuples for which Proposition~\ref{{prop:decoding_error}}(a), respectively the bounds of Proposition~\ref{{prop:assoc_distrib}} weighted with $\hat L$, give the correct result for the estimated radius; wrong ball: the result lies in the decoding ball of a wrong number, Proposition~\ref{{prop:decoding_error}}(b); $\hat L$: empirical Lipschitz constant (the rows of the laws use $\hat L_+$ and $\hat L_\times$ of the first two rows). The radii and these shares are estimates, not guarantees. {cap} Rows $c=2,\dots,16$: the models with $c$ channels per slot of Figure~\ref{{fig:profile}} (single runs); each cell gives the values for $\dot{{+}}$ / $\dot{{\times}}$, the defect column the medians.}}
	\label{{tab:geometry}}
	\renewcommand{{\arraystretch}}{{1.08}}
	\footnotesize
	\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
		\toprule
		\textbf{{Test}} & \textbf{{EM (\%)}} & \textbf{{Defect}} & $\hat\rho$ & \textbf{{In bound (\%)}} & \textbf{{Wrong ball (\%)}} & $\hat L$ \\
		\midrule
		{chr(10).join(chr(9) * 2 + r for r in rows)}
		\bottomrule
	\end{{tabular}}
\end{{table}}
"""
    write(os.path.join(out, "tab_geometry.tex"), tex)
    write_csv(os.path.join(csvdir, "geometry.csv"), csv_rows, ["run", "test", "quantity", "value"])


# ----------------------------------------------------------------------------- figures
def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    return plt


def series(main, key):
    """Validation curve of one metric: steps, mean, min, max over the seeds."""
    by_step = {}
    for n in main:
        for e in jlines(os.path.join(RUNS, n, "evals.jsonl")):
            if key in e:
                by_step.setdefault(e["step"], []).append(e[key])
    steps = sorted(by_step)
    return steps, [st.mean(by_step[s]) for s in steps], [min(by_step[s]) for s in steps], [max(by_step[s]) for s in steps]


def placeholder(ax, text="run pending"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=INK2)


def fig_training(main, out, csvdir):
    import figstyle as FS                          # the style of all figures of the paper (STIX, text width)
    plt = FS.setup()
    fig = FS.figure(2.15)
    axes = fig.subplots(1, 2, sharey=True)
    panels = [("Single operations", [("rec", "reconstruction"), ("+", r"closure of $\dot{+}$"), ("x", r"closure of $\dot{\times}$"),
                                     ("order", "order relation")]),
              ("Composed operations", [("distrib", "distributivity"), ("O1", "order compatible with $\\dot{+}$"),
                                       ("chain+4", "chain of 4 sums"), ("cancel", r"$a\dot{+}(-a)$")])]
    csv_rows = []
    for (ax, (title, keys)), letter in zip(zip(axes, panels), "ab"):
        FS.light_grid(ax)
        any_ = False
        for i, (k, label) in enumerate(keys):
            s, m, lo, hi = series(main, k)
            if not s:
                continue
            any_ = True
            ax.fill_between(s, lo, hi, color=FS.C[i], alpha=0.12, linewidth=0)
            ax.plot(s, m, color=FS.C[i], linestyle=STYLES[i], linewidth=1.3, label=label)
            csv_rows += [[k, a, b, c, d] for a, b, c, d in zip(s, m, lo, hi)]
        if not any_:
            placeholder(ax)
        FS.panel(ax, letter, title, x=-0.13 if letter == "a" else -0.05)
        ax.set_xlabel("training step")
        ax.set_ylim(0, 103)
        if any_:
            ax.legend(loc="lower right", handlelength=2.4)
    for x in (1000, 2000):                         # start of the targeted terms and of the compositions (caption)
        axes[1].axvline(x, color=FS.MUTED, lw=0.6, zorder=0)
    axes[0].set_ylabel("validation exact match (%)")
    fig.savefig(os.path.join(out, "fig_training.pdf"))
    os.makedirs(os.path.join(out, "png"), exist_ok=True)
    fig.savefig(os.path.join(out, "png", "fig_training.png"), dpi=250)
    plt.close(fig)
    write_csv(os.path.join(csvdir, "training_curves_mean.csv"), csv_rows, ["metric", "step", "mean", "min", "max"])


def run_logs(csvdir):
    summ, train, ev_rows = [], [], []
    keys_seen = set()
    for name in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
        d = os.path.join(RUNS, name)
        if not os.path.isdir(d) or name.startswith("_"):
            continue
        M = jlines(os.path.join(d, "metrics.jsonl"))
        E = jlines(os.path.join(d, "evals.jsonl"))
        reg = [m for m in M if "loss" in m]
        last = reg[-1] if reg else {}
        wall, prev = 0.0, 0.0                      # `wall` restarts from zero when a run is resumed: sum the segments
        for m in reg:
            w = float(m.get("wall") or 0)
            if w < prev:
                wall += prev
            prev = w
        wall += prev
        summ.append([name, last.get("step"), round(wall / 3600, 2) if last else None,
                     os.path.exists(os.path.join(d, "DONE"))])
        for m in reg:
            keys_seen.update(m)
        train += [(name, m) for m in reg]
        for e in E:
            ev_rows.append([name, e["step"], json.dumps({k: v for k, v in e.items() if k != "step"})])
    keys = ["step"] + sorted(k for k in keys_seen if k != "step")
    write_csv(os.path.join(csvdir, "training_metrics.csv"), [[n] + [m.get(k) for k in keys] for n, m in train], ["run"] + keys)
    write_csv(os.path.join(csvdir, "run_summary.csv"), summ,
              ["run", "last_step", "hours", "done"])
    write_csv(os.path.join(csvdir, "validation_evals.csv"), ev_rows, ["run", "step", "metrics_json"])
    return summ


MACROS = [("reconstruction", "Rec"), ("identity+", "IdAdd"), ("closure+", "ClAdd"), ("invertibility+", "InvAdd"),
          ("assoc+", "AssAdd"), ("identityx", "IdMul"), ("closurex", "ClMul"), ("invertibilityx", "InvMul"),
          ("assocx", "AssMul"), ("distrib", "Distr"), ("order_order", "Ord"), ("order_transitivity", "Trans"),
          ("order_compat_O1", "Oone"), ("order_compat_O2", "Otwo"), ("fraction_add", "FrAdd"), ("fraction_mul", "FrMul"),
          ("fraction_order", "FrOrd"), ("chains+@5", "ChAddFive"),
          ("chains+@10", "ChAddTen"), ("chainsx@10", "ChMulTen")]


def macros(main, summ, out):
    R = [r for r in (tests(n) for n in main) if r]
    lines = ["% generated by paper/make_tables.py -- do not edit",
             r"% \res<Test>: exact match (%) of the revised NIF, length-free, mean over the seeds; \reso<Test>: oracle length"]
    for key, mname in MACROS:
        for mode, pre in ((MODES[0][0], "res"), (MODES[1][0], "reso")):
            m, s, _ = agg([metric(r, mode, key) for r in R])
            lines.append(rf"\providecommand{{\{pre}{mname}}}{{{fmt(m)}}}")
    lines.append(rf"\providecommand{{\nseeds}}{{{len(R) if R else TBD}}}")
    # operator probes: \abl<Name> = final held-out exact match
    for name, mname in (("abl_x_full", "XFull"), ("abl_x_nohints", "XNoHints"), ("abl_x_norefit", "XNoRefit"),
                        ("abl_x_fourier_random", "XRandFreq"), ("abl_x_fourier_none", "XNoFourier"),
                        ("abl_x_softcounts", "XSoft"), ("abl_x_bf16", "XBf"), ("abl_x_coupled", "XCoupled"),
                        ("abl_x_dense", "XDense"), ("abl_add_slot", "AddSlot"), ("abl_add_dense", "AddDense"),
                        ("depth_x_dense_k1", "DenseKone"), ("depth_x_dense_k2", "DenseKtwo"), ("depth_x_dense_k6", "DenseKsix"),
                        ("depth_x_coupled_k2", "CoupledKtwo"), ("depth_x_coupled_k4", "CoupledKfour"),
                        ("depth_x_coupled_k8", "CoupledKeight")):
        ps = probe_summary(name)
        lines.append(rf"\providecommand{{\abl{mname}}}{{{fmt(ps['final']) if ps else TBD}}}")
    # full-model ablations: \fab<Run><Test> (length-free exact match)
    for name, rname in (("full_paper_losses", "Paper"), ("full_no_hints", "NoHints"), ("full_no_targeted", "NoTargeted"),
                        ("full_no_length_mix", "NoLengthMix"), ("full_dense_ops", "Dense"),
                        ("emb_c2", "CTwo"), ("emb_c4", "CFour"), ("emb_c6", "CSix"), ("emb_c16", "CSixteen")):
        r = tests(name)
        for key, mname in MACROS:
            v = metric(r, MODES[0][0], key) if r else None
            lines.append(rf"\providecommand{{\fab{rname}{mname}}}{{{fmt(v)}}}")
    # comparison protocol (oracle length): \cmp<Op><Acc|EM><N>
    for mode, pre in ((MODES[1][0], "cmp"), (MODES[0][0], "cmplf")):
        for N, nname in (("N10", "Ten"), ("N20", "Twenty"), ("N30", "Thirty")):
            for test, tname in (("closure+", "Add"), ("closurex", "Mul"), ("order", "Ord")):
                for field, fname in (("acc", "Acc"), ("em", "EM")):
                    if test == "order" and field == "acc":
                        continue
                    vals = [r.get(mode, {}).get("comparison", {}).get(N, {}).get(test, {}).get(field) for r in R]
                    m, sd, _ = agg(vals)
                    lines.append(rf"\providecommand{{\{pre}{tname}{fname}{nname}}}{{{fmt(m, sd)}}}")
    # AE-only row of the comparison table (one run, oracle length as the other rows): \cmpae<Op><Acc|EM><N>
    ae = tests("aeonly")
    for N, nname in (("N10", "Ten"), ("N20", "Twenty"), ("N30", "Thirty")):
        for test, tname in (("closure+", "Add"), ("closurex", "Mul"), ("order", "Ord")):
            for field, fname in (("acc", "Acc"), ("em", "EM")):
                if test == "order" and field == "acc":
                    continue
                v = ae.get(MODES[1][0], {}).get("comparison", {}).get(N, {}).get(test, {}).get(field) if ae else None
                lines.append(rf"\providecommand{{\cmpae{tname}{fname}{nname}}}{{{fmt(v)}}}")
    hours = [row[2] for row in summ if row[0] in main and row[2]]
    lines.append(rf"\providecommand{{\hoursperrun}}{{{fmt(st.mean(hours), None, 1) if hours else TBD}}}")
    lines += length_assets.length_macros(RUNS, fmt, TBD)
    lines += length_assets.extra_macros(os.path.join(HERE, "results", "analysis"), fmt, TBD)
    write(os.path.join(out, "numbers.tex"), "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="main_s0,main_s1,main_s2", help="runs of the main result (seeds)")
    ap.add_argument("--out", default=os.path.join(HERE, "paper", "generated"))
    ap.add_argument("--csv", default=os.path.join(HERE, "results", "csv"))
    a = ap.parse_args()
    main_runs = [n for n in a.main.split(",") if n]
    os.makedirs(a.out, exist_ok=True)
    have = table_main(main_runs, a.out, a.csv)
    table_ablation_operator(a.out, a.csv)
    table_ablation_full(main_runs, a.out, a.csv)
    table_geometry(main_runs, a.out, a.csv)
    length_assets.table_lengthdim(RUNS, a.out, a.csv, write, write_csv)
    summ = run_logs(a.csv)
    macros(main_runs, summ, a.out)
    fig_training(main_runs, a.out, a.csv)
    print(f"main result from {main_runs} ({len(have)} tested); assets in {os.path.abspath(a.out)}, CSV in {os.path.abspath(a.csv)}")


if __name__ == "__main__":
    main()
