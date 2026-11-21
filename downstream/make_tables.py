r"""Table 7, Figures 11 and 12 and the macros of the language-model experiment (Section 5.4).

    python downstream/make_tables.py [--runs downstream/runs] [--out paper/generated]

Reads the run folders
    base_model              the language model as released
    operator_layers         the model with NIF operator layers after 6000 steps: final test and interventions
    operator_layers_10000   the same run continued to 10000 steps
and writes <out>/tab_llm.tex, fig_llm.pdf, fig_llm_causal.pdf, llm_numbers.tex (macros \llm...), results/csv/llm.csv
and results/csv/llm_causal.csv. Every comparison is paired on the same test problems (exact two-sided McNemar test);
the p-values of a table are adjusted with the Holm-Bonferroni procedure over its rows.
"""
import argparse
import csv
import gzip
import json
import os
from math import comb

import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
sys.path.insert(0, os.path.join(HERE, "paper"))
sys.path.insert(0, os.path.join(HERE, "downstream"))
ZS, NIF, LONG = "base_model", "operator_layers", "operator_layers_10000"
TRAIN_MAX = 12                                           # digits of the operands in training (max_size)

# (label, macro tag, task, sizes or None = all, in the Holm family of the table)
MAIN = [
    ("GSM-Ranges, level 1", "ROne", "ranges_cot", [1], True),
    ("GSM-Ranges, level 2", "RTwo", "ranges_cot", [2], True),
    ("GSM-Ranges, level 3", "RThree", "ranges_cot", [3], True),
    ("GSM-Ranges, level 4", "RFour", "ranges_cot", [4], True),
    ("GSM-Ranges, level 5", "RFive", "ranges_cot", [5], True),
    ("GSM-Ranges, level 6", "RSix", "ranges_cot", [6], True),
    ("Templated problems, numbers of 1--12 digits", "WpIn", "wp_cot", "in", True),
    ("Templated problems, numbers of 14--20 digits", "WpOut", "wp_cot", "out", True),
]
EXTRA = [("GSM-Ranges, levels 1-6", "RAll", "ranges_cot", [1, 2, 3, 4, 5, 6]),          # for the text only
         ("GSM-Ranges, levels 4-6", "RHigh", "ranges_cot", [4, 5, 6])]
CAUSAL_ROWS = [
    ("Addition, 1--12 digits", "AddIn", "add", "in"),
    ("Addition, 14--20 digits", "AddOut", "add", "out"),
    ("Subtraction, 1--12 digits", "SubIn", "sub", "in"),
    ("Subtraction, 14--20 digits", "SubOut", "sub", "out"),
    ("Multiplication, factors of 1--6 digits", "MulIn", "mul", [1, 2, 3, 4, 5, 6]),
    ("Multiplication, factors of 7--8 digits", "MulOut", "mul", [7, 8]),
    ("Templated problems, numbers of 1--12 digits", "WpIn", "wp_cot", "in"),
    ("Templated problems, numbers of 14--20 digits", "WpOut", "wp_cot", "out"),
    ("GSM-Ranges, levels 1--3", "RLow", "ranges_cot", [1, 2, 3]),
    ("GSM-Ranges, levels 4--6", "RHigh", "ranges_cot", [4, 5, 6]),
]


def load_items(path, fields=("ok",)):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["task"], int(r["size"]), r["i"])] = r[fields[0]] if len(fields) == 1 else {k: r[k] for k in fields}
    return out


def keep(key, task, sizes):
    if key[0] != task:
        return False
    if sizes is None:
        return True
    if sizes == "in":
        return key[1] <= TRAIN_MAX
    if sizes == "out":
        return key[1] > TRAIN_MAX
    return key[1] in sizes


def mcnemar(x, y):
    """Exact two-sided McNemar test of two lists of outcomes on the same problems: (only x, only y, p)."""
    a = sum(1 for u, v in zip(x, y) if u and not v)
    b = sum(1 for u, v in zip(x, y) if v and not u)
    n = a + b
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0
    return a, b, p


def holm(ps):
    """Holm-Bonferroni adjusted p-values (same order)."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def pct(xs):
    return 100.0 * sum(xs) / max(1, len(xs))


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def p_text(p):
    """A p-value for the running text (a complete formula)."""
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$" if p < 0.1 else f"$p={p:.2f}$"


def gsm_lengths(path):
    """Share of the numbers of the GSM8K test questions with at most three digits, their mean number of digits, and
    the share of the reference results with at most four digits."""
    g = json.load(open(path, encoding="utf-8"))["test"]
    dig = lambda s: len(s.lstrip("-").replace(".", ""))
    q = [dig(p[1]) for r in g for p in r["user"] if isinstance(p, list)]
    a = [dig(p[1]) for r in g for p in r["asst"] if isinstance(p, list)]
    return pct([d <= 3 for d in q]), sum(q) / len(q), pct([d <= 4 for d in a])


def wilson(k, n, z=1.96):
    """95% Wilson interval of a proportion, in %."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def series(items, task, sizes):
    """[(size, exact match %, low, high)] of a task by cell."""
    out = []
    for s in sizes:
        v = [ok for k, ok in items.items() if k[0] == task and k[1] == s]
        lo, hi = wilson(sum(v), len(v))
        out.append((s, pct(v), lo, hi))
    return out


def save_fig(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    prev = os.path.join(out, "png")
    os.makedirs(prev, exist_ok=True)
    fig.savefig(os.path.join(prev, name + ".png"), dpi=250)


def two_lines(ax, xs, base, nif, S):
    """Base model and model with NIF, with 95% intervals."""
    sers = [(base, S.C[1], "Base model", "s"), (nif, S.C[0], "Trained with NIF", "o")]
    for ser, col, lab, mk in sers:
        ys, lo, hi = [p[1] for p in ser], [p[2] for p in ser], [p[3] for p in ser]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.13, linewidth=0)
        ax.plot(xs, ys, color=col, marker=mk, markersize=3.6, markeredgecolor=S.SURFACE, markeredgewidth=0.6,
                label=lab)


def figure_results(z, t, runs, out):
    """fig_llm.pdf: (a) GSM-Ranges by level, (b) templated problems by number length, (c) exact match during
    training (complete test every 500 steps) with the base model as reference."""
    import figstyle as S
    plt = S.setup()
    fig = S.figure(2.25)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.35])
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

    lv = list(range(1, 7))
    two_lines(a, lv, series(z, "ranges_cot", lv), series(t, "ranges_cot", lv), S)
    a.set_xticks(lv)
    a.set_xlabel("GSM-Ranges level (larger numbers $\\rightarrow$)")
    a.set_ylabel("exact match (%)")
    a.set_ylim(0, 100)
    S.light_grid(a)
    S.panel(a, "a", "GSM-Ranges", x=-0.27, gap=0.12)
    a.legend(loc="upper right", fontsize=6.6, handlelength=1.5, borderaxespad=0.1)

    ln = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20]
    b.axvspan(13, 21, color="#f4f4f4", linewidth=0, zorder=0)
    b.text(17, 96, "longer than in\ntraining", ha="center", va="top", fontsize=6.6, color=S.INK2, linespacing=1.0)
    two_lines(b, ln, series(z, "wp_cot", ln), series(t, "wp_cot", ln), S)
    b.set_xticks([1, 4, 8, 12, 16, 20])
    b.set_xlim(0.3, 20.7)
    b.set_xlabel("digits of the numbers")
    b.set_ylim(0, 100)
    S.light_grid(b)
    S.panel(b, "b", "Templated problems", x=-0.19, gap=0.09)

    rows = []
    for name in (NIF, LONG):
        p = os.path.join(runs, name, "evals.jsonl")
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    seen, pts = set(), []
    for r in rows:
        if r["step"] not in seen and "wp_cot_beyond" in r:
            seen.add(r["step"])
            pts.append(r)
    pts.sort(key=lambda r: r["step"])
    zsum = json.load(open(os.path.join(runs, ZS, "results.json"), encoding="utf-8"))["summary"]
    high = lambda r: (r["ranges_cot_L4"] + r["ranges_cot_L5"] + r["ranges_cot_L6"]) / 3
    curves = [("GSM-Ranges, levels 4–6", high, S.C[0]),
              ("Templated, 14–20 digits", lambda r: r["wp_cot_beyond"], S.C[4])]
    xs = [r["step"] for r in pts]
    last = max(xs)
    if last > 6000:
        c.axvspan(6000, last + 300, color="#f4f4f4", linewidth=0, zorder=0)
        c.text((6000 + last) / 2, 58, "training\ncontinued", ha="center", va="bottom", fontsize=6.6, color=S.INK2,
               linespacing=1.0)
    for lab, f, col in curves:
        c.plot(xs, [f(r) for r in pts], color=col, marker="o", markersize=2.6, markeredgecolor=S.SURFACE,
               markeredgewidth=0.4, label=lab)
        c.axhline(f(zsum), color=col, linewidth=0.9, linestyle=(0, (3, 2)))
    c.plot([], [], color=S.INK2, linewidth=0.9, linestyle=(0, (3, 2)), label="base model (dashed)")
    c.set_xlim(0, last + 300)
    c.set_xticks([0, 2000, 4000, 6000, 8000, 10000][: 1 + last // 2000])
    c.set_xticklabels([f"{x // 1000}k" if x else "0" for x in c.get_xticks().astype(int)])
    c.set_xlabel("training step")
    c.set_ylabel("exact match (%)")
    c.set_ylim(0, 80)
    S.light_grid(c)
    S.panel(c, "c", "During training", x=-0.16, gap=0.08)
    c.legend(loc="upper left", fontsize=6.3, handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)
    save_fig(fig, out, "fig_llm")
    plt.close(fig)


MODES = (("on", "Trained with NIF", 0, "o"), ("off", "… operator layers off", 3, "s"),
         ("shuffle", "… vectors of other numbers", None, "v"))


def figure_causal(cells, out):
    """fig_llm_causal.pdf: exact match of the trained model as trained, with the operator layers off and with the
    vectors of other numbers. cells: {task: {size: {"n", "on", "off", "shuffle"}}} (percentages)."""
    import figstyle as S
    from matplotlib.patches import Patch
    plt = S.setup()
    fig = S.figure(2.0)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.25])
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    col = lambda i: S.MUTED if i is None else S.C[i]

    def lines(ax, task_sizes, xs):
        for mode, lab, ci, mk in MODES:
            if not all(mode in v for tk in task_sizes for v in cells.get(tk, {}).values()):
                continue
            ys = []
            for s in xs:
                vals = [(cells[tk][str(s)][mode], cells[tk][str(s)]["n"]) for tk in task_sizes if str(s) in cells.get(tk, {})]
                ys.append(sum(v * n for v, n in vals) / sum(n for _, n in vals))
            ax.plot(xs, ys, color=col(ci), marker=mk, markersize=3.2, markeredgecolor=S.SURFACE, markeredgewidth=0.5,
                    label=lab, linestyle="-" if ci is not None else (0, (3, 1.5)))

    ln = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20]
    a = axs[0]
    a.axvspan(13, 21, color="#f4f4f4", linewidth=0, zorder=0)
    lines(a, ["add", "sub"], ln)
    a.set_xticks([1, 4, 8, 12, 16, 20])
    a.set_xlim(0.3, 20.7)
    a.set_xlabel("digits of the operands")
    a.set_ylabel("exact match (%)")
    S.panel(a, "a", "Addition, subtraction", x=-0.2, gap=0.09)
    b = axs[1]
    b.axvspan(6.5, 8.5, color="#f4f4f4", linewidth=0, zorder=0)
    lines(b, ["mul"], list(range(1, 9)))
    b.set_xticks([1, 4, 8])
    b.set_xlim(0.5, 8.5)
    b.set_xlabel("digits of the factors")
    S.panel(b, "b", "Multiplication", x=-0.16, gap=0.08)
    c = axs[2]
    c.axvspan(13, 21, color="#f4f4f4", linewidth=0, zorder=0)
    lines(c, ["wp_cot"], ln)
    c.set_xticks([1, 4, 8, 12, 16, 20])
    c.set_xlim(0.3, 20.7)
    c.set_xlabel("digits of the numbers")
    S.panel(c, "c", "Templated word problems", x=-0.12, gap=0.06)
    for ax in (a, b, c):
        ax.set_ylim(0, 102)
        S.light_grid(ax)

    modes = [m for m in MODES if all(m[0] in v for t_ in cells.values() for v in t_.values())]
    handles = [plt.Line2D([], [], color=col(ci), marker=mk, markersize=3.2, linestyle="-" if ci is not None else (0, (3, 1.5)),
                          label=lab) for _, lab, ci, mk in modes]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(modes), fontsize=7, handlelength=2.0, columnspacing=1.4)
    save_fig(fig, out, "fig_llm_causal")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "downstream", "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "paper", "generated"))
    ap.add_argument("--data", default=os.path.join(HERE, "downstream", "data"))
    a = ap.parse_args()
    run = lambda name, f: os.path.join(a.runs, name, f)
    z, t = load_items(run(ZS, "items.jsonl.gz")), load_items(run(NIF, "items.jsonl.gz"))
    res = json.load(open(run(NIF, "results.json"), encoding="utf-8"))
    cfg = res["config"]
    macros, csv_rows = {}, []

    # ------------------------------------------------------------------ base model and model with NIF
    rows = []
    cfile = run(NIF, "causal_items.jsonl.gz")                # operators switched off on the same problems
    coff = load_items(cfile, ("on", "off")) if os.path.exists(cfile) else None
    for label, tag, task, sizes, fam in MAIN:
        ks = sorted(k for k in t if keep(k, task, sizes) and k in z)
        x, y = [z[k] for k in ks], [t[k] for k in ks]
        rows.append(dict(label=label, tag=tag, n=len(ks), base=pct(x), nif=pct(y), p=mcnemar(x, y)[2]))
        if coff is not None:
            assert all(k in coff for k in ks), label
            assert [coff[k]["on"] for k in ks] == y, label       # the intervention run reproduces the final test
            rows[-1]["off"] = pct([coff[k]["off"] for k in ks])
    for r, q in zip(rows, holm([r["p"] for r in rows])):
        r["padj"] = q
    body = []
    pcell = lambda q: rf"\textbf{{{fmt_p(q)}}}" if q < 0.05 else fmt_p(q)
    for r in rows:
        body.append(f"\t\t{r['label']} & {r['n']} & {r['base']:.1f} & {r['nif']:.1f} & {r['off']:.1f} & {pcell(r['padj'])} "
                    + r"\\")
        for k in ("base", "nif", "off"):
            macros[f"llm{r['tag']}{k.capitalize()}"] = f"{r[k]:.1f}"
        macros[f"llm{r['tag']}P"] = p_text(r["padj"])
        macros[f"llm{r['tag']}N"] = f"{r['n']:,}".replace(",", "{,}")
        csv_rows.append([r["label"], r["n"], f"{r['base']:.2f}", f"{r['nif']:.2f}", f"{r['off']:.2f}", f"{r['p']:.6f}",
                         f"{r['padj']:.6f}"])
    by = {r["tag"]: r for r in rows}                     # loss from level 1 to level 6 of GSM-Ranges
    for k in ("base", "nif"):
        macros[f"llmRDrop{k.capitalize()}"] = f"{by['ROne'][k] - by['RSix'][k]:.0f}"
    tex = rf"""% generated by downstream/make_tables.py -- do not edit
\begin{{table}}[pos=htbp]
	\centering
	\caption{{Exact match (\%) of the final answer on word problems (Section~\ref{{subsec:downstream}}): the language model as released (base model) and after training with the NIF operator layers, with the same prompts. Operators off: the model trained with NIF, with the operator layers and register outputs switched off (Figure~\ref{{fig:llmcausal}}). $p$: exact McNemar test of the base model and the model trained with NIF on the same problems, adjusted with the Holm--Bonferroni procedure over the rows; bold: $p<0.05$.}}
	\label{{tab:llm}}
	\renewcommand{{\arraystretch}}{{1.08}}
	\footnotesize
	\begin{{tabular}}{{@{{}}lrcccc@{{}}}}
		\toprule
		\textbf{{Test set}} & \textbf{{Problems}} & \textbf{{Base model}} & \textbf{{Trained with NIF}} & \textbf{{Operators off}} & $p$ \\
		\midrule
{chr(10).join(body[:6])}
		\midrule
{chr(10).join(body[6:])}
		\bottomrule
	\end{{tabular}}
\end{{table}}
"""
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "tab_llm.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)

    # ------------------------------------------------------------------ facts of the setup quoted in the text
    import data as D
    D.load_gsm8k(a.data)                                 # GSM8K, downloaded and processed once (numbers of its questions)
    q3, qmean, a4 = gsm_lengths(os.path.join(a.data, "gsm8k_processed.json"))
    macros.update(llmGsmShort=f"{q3:.0f}", llmGsmMean=f"{qmean:.1f}", llmGsmResShort=f"{a4:.0f}")
    macros.update(llmSteps=f"{cfg['steps']:,}".replace(",", "{,}"), llmBatch=str(cfg["batch"]), llmTrainMax=str(cfg["max_size"]),
                  llmParams=f"{res['train']['base_params'] / 1e6:.0f}",
                  llmNewParams=f"{(res['train']['trained_params'] - res['train']['base_params']) / 1e6:.1f}",
                  llmHours=f"{res['train']['hours']:.1f}")
    sol, st = D.load_self_cot(os.path.join(a.data, "gsm8k_solutions.json.gz"))     # the model's own GSM8K solutions
    macros.update(llmGsmSol=f"{len(sol):,}".replace(",", "{,}"), llmGsmSolved=f"{st['solved']:,}".replace(",", "{,}"),
                  llmGsmTrain=f"{st['problems']:,}".replace(",", "{,}"))

    # ------------------------------------------------------------------ the run continued to 10000 steps
    if os.path.exists(run(LONG, "items.jsonl.gz")):
        u = load_items(run(LONG, "items.jsonl.gz"))
        ps = []
        for label, tag, task, sizes, fam in MAIN:
            ks = sorted(k for k in t if keep(k, task, sizes) and k in u)
            ps.append(mcnemar([t[k] for k in ks], [u[k] for k in ks])[2])
        macros["llmLongMinP"] = f"{min(ps):.2f}"
        macros["llmLongSteps"] = "10{,}000"
        print("10000 steps against 6000 steps, smallest unadjusted p over the rows of the table:", min(ps))

    # ------------------------------------------------------------------ interventions on the operator layers
    cpath = run(NIF, "causal_items.jsonl.gz")
    if os.path.exists(cpath):
        c = load_items(cpath, ("on", "off", "shuffle"))
        crow = []
        for label, tag, task, sizes in CAUSAL_ROWS:
            ks = sorted(k for k in c if keep(k, task, sizes))
            on, off, sh = ([c[k][m] for k in ks] for m in ("on", "off", "shuffle"))
            crow.append(dict(label=label, tag=tag, n=len(ks), on=pct(on), off=pct(off), sh=pct(sh), p=mcnemar(on, off)[2]))
        for r, p in zip(crow, holm([r["p"] for r in crow])):
            r["padj"] = p
        ks1 = [k for k in c if k[0] == "ranges_cot" and k[1] == 1]        # level 1: the lengths of GSM8K (text only)
        macros["llmCROneOn"] = f"{pct([c[k]['on'] for k in ks1]):.1f}"
        macros["llmCROneOff"] = f"{pct([c[k]['off'] for k in ks1]):.1f}"
        ccsv = []
        for r in crow:
            for k, nm in (("on", "On"), ("off", "Off"), ("sh", "Shuf")):
                macros[f"llmC{r['tag']}{nm}"] = f"{r[k]:.1f}"
            macros[f"llmC{r['tag']}P"] = p_text(r["padj"])
            ccsv.append([r["label"], r["n"], f"{r['on']:.2f}", f"{r['off']:.2f}", f"{r['sh']:.2f}", f"{r['p']:.6f}", f"{r['padj']:.6f}"])
        agg = {}
        for key, v in c.items():
            task, size, _ = key
            cell = agg.setdefault(task, {}).setdefault(str(size), {"n": 0, "on": 0, "off": 0, "shuffle": 0})
            cell["n"] += 1
            for m in ("on", "off", "shuffle"):
                cell[m] += bool(v[m])
        for task in agg.values():
            for cell in task.values():
                for m in ("on", "off", "shuffle"):
                    cell[m] = 100.0 * cell[m] / cell["n"]
        figure_causal(agg, a.out)
        with open(os.path.join(HERE, "results", "csv", "llm_causal.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["test set", "problems", "as trained", "operators off", "other vectors", "p (McNemar, on vs off)", "p (Holm)"])
            w.writerows(ccsv)
        for r in crow:
            print(f"   {r['label'][:46]:46s} n={r['n']:5d} on {r['on']:5.1f} off {r['off']:5.1f} other {r['sh']:5.1f} p={r['padj']:.4f}")
    else:
        print(f"no {cpath}: interventions not written")

    figure_results(z, t, a.runs, a.out)
    with open(os.path.join(a.out, "llm_numbers.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("% generated by downstream/make_tables.py -- do not edit\n")
        f.write("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    os.makedirs(os.path.join(HERE, "results", "csv"), exist_ok=True)
    with open(os.path.join(HERE, "results", "csv", "llm.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test set", "problems", "base model", "trained with NIF", "operators off", "p", "p (Holm)"])
        w.writerows(csv_rows)
    for r in rows:
        print(f"   {r['label'][:46]:46s} n={r['n']:5d} base model {r['base']:5.1f} NIF {r['nif']:5.1f} p={r['padj']:.4f}")
    print("macros:", len(macros))


if __name__ == "__main__":
    main()
