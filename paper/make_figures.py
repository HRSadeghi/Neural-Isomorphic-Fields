"""Figures of the paper (vector PDF, and PNG copies in paper/generated/png), built from the logs of the runs and
from analysis/. Style: figstyle.py (STIX fonts, the text width of the journal template).

    python paper/make_figures.py

  fig_distributions.pdf  Figure 4: candidate distributions of the number of digits
  fig_lengths.pdf        Figure 5: errors by the shape of the number to decode
  fig_operators.pdf      Figure 7: operators trained alone
  fig_profile.pdf        Figure 8: tests x models, exact match and error rate on a log scale
  fig_lengthdim.pdf      Figure 9: number length and embedding dimension
  fig_geometry.pdf       Figure 10: digit code of a slot, defect vs decoding radius, chains
"""
from __future__ import annotations

import json
import math
import os
import statistics as st

import numpy as np

import figstyle as S

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "paper", "generated")
PREVIEW = os.path.join(OUT, "png")
ANALYSIS = os.path.join(HERE, "results", "analysis")
C, INK, INK2, MUTED, RULE, SURFACE = S.C, S.INK, S.INK2, S.MUTED, S.RULE, S.SURFACE


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PREVIEW, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=250)


def jl(path):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def tests(run):
    p = os.path.join(RUNS, run, "test_results.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("N25_length_free", {})


def em(lf, key):
    if "@" in key:
        name, K = key.split("@")
        return lf.get(name, {}).get("per_K", {}).get(K, {}).get("em")
    return lf.get(key, {}).get("em")


def mean_em(runs, keys):
    R = [t for t in (tests(r) for r in runs) if t]
    if not R:
        return None
    return [st.mean(v) if (v := [em(t, k) for t in R if em(t, k) is not None]) else np.nan for k in keys]


# ----------------------------------------------------------------------------- (1) profile
COLS = [("Single operations", [("reconstruction", "rec."), ("closure+", "cl.$\\,+$"), ("closurex", "cl.$\\,\\times$"),
                               ("identityx", "id.$\\,\\times$")]),
        ("Inverses", [("invertibility+", "$+$"), ("invertibilityx", "$\\times$")]),
        ("Assoc.", [("assoc+", "$+$"), ("assocx", "$\\times$")]),
        ("Distr.", [("distrib", "$\\times,+$")]),
        ("Order", [("order_order", "rel."), ("order_compat_O1", "O1"), ("order_compat_O2", "O2")]),
        ("Fraction field", [("fraction_add", "$+$"), ("fraction_mul", "$\\times$"),
                            ("fraction_order", "$<$")]),
        ("Chains ($M\\!=\\!10$)", [("chains+@10", "$+$"), ("chainsx@10", "$\\times$")])]
ROWS = [(["main_s0", "main_s1", "main_s2"], "NIF (mean of 3 seeds)"),
        (["full_no_length_mix"], "$-$ uniform-length numbers"),
        (["full_no_targeted"], "$-$ targeted pairs, output terms"),
        (["full_no_hints"], "$-$ hint supervision"),
        (["full_paper_losses"], "only $\\ell_{\\mathrm{rec}},\\ell_{\\mathrm{iso}},\\ell_{\\mathrm{ord}}$"),
        (["full_dense_ops"], "perceptron AC layers"),
        (["aeonly"], "AE-only encoder (frozen)"),
        (["emb_c2"], "$c=2$ channels per slot ($d=104$)"), (["emb_c4"], "$c=4$ ($d=208$)"),
        (["emb_c6"], "$c=6$ ($d=312$)"), (["emb_c16"], "$c=16$ ($d=832$)")]
GROUP_STARTS = ("$-$ uniform", "AE-only", "$c=2$")        # a gap before these rows


def fig_profile():
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Rectangle
    plt = S.setup()
    cmap = LinearSegmentedColormap.from_list("err", S.BLUES)
    norm = LogNorm(vmin=0.01, vmax=100)
    keys = [k for _, g in COLS for k, _ in g]
    rows, labels = [], []
    for runs, label in ROWS:
        v = mean_em(runs, keys)
        if v is not None:
            rows.append(v)
            labels.append(label)
    if not rows:
        return
    gap_c, gap_r, cw, rh = 0.38, 0.32, 1.0, 1.0
    xs, x = [], 0.0
    for gi, (_, g) in enumerate(COLS):
        for _ in g:
            xs.append(x)
            x += cw
        x += gap_c
    width_units = x - gap_c
    ys, y = [], 0.0
    for label in labels:
        if ys and label.startswith(GROUP_STARTS):
            y += gap_r
        ys.append(y)
        y += rh
    height_units = ys[-1] + rh
    fig = S.figure(0.55 + 0.205 * height_units + 0.22)
    ax = fig.add_subplot(111)
    for i, (r, y) in enumerate(zip(rows, ys)):
        for j, (v, x0) in enumerate(zip(r, xs)):
            if np.isnan(v):
                continue
            e = min(100.0, max(0.01, 100.0 - v))
            ax.add_patch(Rectangle((x0, y), cw, rh, facecolor=cmap(norm(e)), edgecolor=SURFACE, linewidth=0.9))
            txt = "100" if v >= 99.995 else (f"{v:.2f}" if v >= 99 else f"{v:.1f}")
            ax.text(x0 + cw / 2, y + rh / 2, txt, ha="center", va="center", fontsize=6.6,
                    color=SURFACE if e >= 8 else INK)
    for label, y in zip(labels, ys):
        ax.text(-0.25, y + rh / 2, label, ha="right", va="center", fontsize=7.8, color=INK)
    j = 0
    for title, g in COLS:                                   # booktabs-like two-level header
        x0, x1 = xs[j], xs[j + len(g) - 1] + cw
        ax.text((x0 + x1) / 2, -1.55, title, ha="center", va="bottom", fontsize=7.8, color=INK)
        ax.plot([x0 + 0.08, x1 - 0.08], [-1.42, -1.42], color=INK2, linewidth=0.6, solid_capstyle="butt")
        for k, (_, lab) in enumerate(g):
            ax.text(xs[j + k] + cw / 2, -0.62, lab, ha="center", va="center", fontsize=7.2, color=INK2)
        j += len(g)
    ax.set_xlim(-0.1, width_units + 0.1)
    ax.set_ylim(height_units + 0.15, -2.05)
    ax.axis("off")
    cax = ax.inset_axes([0.72, -0.075, 0.28, 0.04])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0.01, 0.1, 1, 10, 100], labels=["0.01", "0.1", "1", "10", "100"])
    cb.ax.tick_params(labelsize=6.8, length=2, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.set_label("error rate, $100-\\mathrm{EM}$ (%), log scale", fontsize=7.2, labelpad=2)
    save(fig, "fig_profile")
    plt.close(fig)


# ----------------------------------------------------------------------------- (2) operators
def best_em(run):
    ev = jl(os.path.join(RUNS, run, "evals.jsonl"))
    return (max(ev, key=lambda e: e["em"]), ev) if ev else (None, [])


def fig_operators():
    plt = S.setup()
    fig = S.figure(2.25)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.32])
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    series = [("perceptron AC layers", {1: "depth_x_dense_k1", 2: "depth_x_dense_k2", 4: "abl_x_dense", 6: "depth_x_dense_k6"}),
              ("Transformer AC layers", {2: "depth_x_coupled_k2", 4: "depth_x_coupled_k4", 6: "abl_x_coupled",
                                         8: "depth_x_coupled_k8"})]
    S.light_grid(a)
    staged = best_em("abl_x_full")[0]
    if staged:
        a.axhline(staged["em"], color=C[2], linewidth=1.4)
        a.text(8.3, staged["em"] - 3.5, f"staged operator ($k=4$): {staged['em']:.2f}%", ha="right", va="top",
               fontsize=7.2, color=INK)
    for i, (lab, runs) in enumerate(series):
        pts = [(k, best_em(r)[0]["em"]) for k, r in sorted(runs.items()) if best_em(r)[0]]
        if pts:
            a.plot([p[0] for p in pts], [p[1] for p in pts], color=C[i], marker="os"[i], markersize=4.2,
                   markeredgecolor=SURFACE, markeredgewidth=0.8, label=lab)
    a.set_xticks([1, 2, 4, 6, 8])
    a.set_xlim(0.6, 8.4)
    a.set_ylim(0, 104)
    a.set_xlabel("number of AC layers $k$")
    a.set_ylabel("held-out exact match of $\\dot{\\times}$ (%)")
    a.legend(loc="center right", bbox_to_anchor=(1.0, 0.56))
    S.panel(a, "a", "Depth, each trained to convergence", x=-0.2)
    variants = [("abl_x_full", "staged operator"), ("abl_x_softcounts", "soft digit counts"),
                ("abl_x_fourier_random", "random frequencies"), ("abl_x_norefit", "no least-squares fit"),
                ("abl_x_fourier_none", "no periodic features"), ("abl_x_bf16", "bfloat16"),
                ("abl_x_nohints", "no hint supervision")]
    rows = []
    for r, lab in variants:
        bst, ev = best_em(r)
        if bst:
            first = next((e["step"] for e in sorted(ev, key=lambda e: e["step"]) if e["em"] >= 99.0), None)
            rows.append((lab, bst["em"], first, ev[-1]["step"]))
    y = np.arange(len(rows))[::-1]
    S.light_grid(b, "x")
    for yi, (lab, v, first, last) in zip(y, rows):
        b.barh(yi, v, height=0.58, color=C[0] if v >= 99 else C[1], linewidth=0)
        note = f"{v:.2f}%,  $\\geq$99% at step {first:,}" if first is not None else f"{v:.1f}%,  never 99% ({last:,} steps)"
        inside = v > 60
        b.text(v - 1.5 if inside else v + 1.5, yi, note, ha="right" if inside else "left", va="center", fontsize=7,
               color=SURFACE if inside else INK)
    b.set_yticks(y, [r[0] for r in rows])
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.set_xlim(0, 100)
    b.set_xlabel("held-out exact match of $\\dot{\\times}$ (%)")
    S.panel(b, "b", "Staged operator without one component", x=-0.36)
    save(fig, "fig_operators")
    plt.close(fig)


# ----------------------------------------------------------------------------- (3) lengths
def load_lengths():
    """Results of analysis/lengths.py: runs/<run>/lengths.json of every analysed run."""
    out = {}
    if os.path.isdir(RUNS):
        for run in sorted(os.listdir(RUNS)):
            p = os.path.join(RUNS, run, "lengths.json")
            if os.path.exists(p):
                out[run] = json.load(open(p))
    return out


def placeholder(name, text, height=1.9):
    plt = S.setup()
    print(f"{name}: data missing ({text}); a placeholder figure is written")
    fig = S.figure(height)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, "pending: " + text, ha="center", va="center", color=INK2)
    save(fig, name)
    plt.close(fig)


def load_parts():
    """Results of analysis/parts.py: per-item outcomes with the numbers of integer and fractional digits."""
    out = {}
    if os.path.isdir(RUNS):
        for run in sorted(os.listdir(RUNS)):
            p = os.path.join(RUNS, run, "lengths_parts.json")
            if os.path.exists(p):
                d = json.load(open(p))
                for t in d["tests"].values():
                    for k in t:
                        t[k] = np.array(t[k])
                out[run] = d
    return out


LD_TESTS = (("reconstruction", "Reconstruction"), ("closure+", "Closure of $\\dot{+}$"), ("closurex", "Closure of $\\dot{\\times}$"))


def split_em(d, key, inside=True):
    """Exact match (%) on the test numbers of D_N whose two parts are within (inside=True) or beyond (False) the
    part lengths of the uniform training half; returns (EM, number of items)."""
    t, N, h = d["tests"][key], d["n_digits"], d["max_part_trained"]
    longest = np.maximum(t["n_int"], t["n_frac"])
    m = (t["total"] <= N) & ((longest <= h) if inside else (longest > h))
    return (100.0 * t["ok"][m].mean() if m.any() else np.nan), int(m.sum())


def err_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("err", S.BLUES)


def fig_lengths():
    """Exact match of NIF (N = 25, the three seeds pooled) over the shape of the number to decode:
    integer digits x fractional digits, with the training lengths and the domain D_25 marked."""
    from matplotlib.colors import Normalize
    P = load_parts()
    seeds = [r for r in ("main_s0", "main_s1", "main_s2") if r in P]
    if not seeds:
        placeholder("fig_lengths", "python analysis/lengths.py; python analysis/parts.py; python make_figures.py")
        return
    plt = S.setup()
    cmap, norm = err_cmap(), Normalize(0, 100)
    N, h = P[seeds[0]]["n_digits"], P[seeds[0]]["max_part_trained"]
    W = N + 1
    fig = S.figure(2.42)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax, (key, title), letter in zip(axes, LD_TESTS, "abc"):
        n_ok = np.zeros((W + 1, W + 1))
        n_all = np.zeros((W + 1, W + 1))
        for r in seeds:
            t = P[r]["tests"][key]
            np.add.at(n_all, (t["n_frac"], t["n_int"]), 1)
            np.add.at(n_ok, (t["n_frac"], t["n_int"]), t["ok"])
        err = np.where(n_all >= 6, 100.0 * (1 - n_ok / np.maximum(n_all, 1)), np.nan)
        ax.set_facecolor("#f4f4f4")
        ax.pcolormesh(np.arange(W + 2) - 0.5, np.arange(W + 2) - 0.5, np.ma.masked_invalid(err), cmap=cmap, norm=norm,
                      linewidth=0, rasterized=True)
        ax.plot([0.5, h + 0.5, h + 0.5], [h + 0.5, h + 0.5, -0.5], color=C[1], linewidth=1.0)            # training lengths
        ax.plot([0.5, N + 0.5], [N - 0.5, -0.5], color=INK, linewidth=0.8, linestyle=(0, (3, 2)))         # n_int + n_frac = N
        ax.set_xlim(0.5, W + 0.5)
        ax.set_ylim(-0.5, W + 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([1, 5, 10, 15, 20, 25])
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_xlabel("integer digits")
        if letter == "a":
            ax.set_ylabel("fractional digits")
            ax.plot([12.6, 14.8], [24.4, 24.4], color=C[1], linewidth=1.0)
            ax.text(15.4, 24.4, "training lengths", ha="left", va="center", fontsize=6.4, color=INK)
            ax.plot([12.6, 14.8], [22.2, 22.2], color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
            ax.text(15.4, 22.2, "25 digits in all", ha="left", va="center", fontsize=6.4, color=INK)
        else:
            ax.set_yticklabels([])
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(True)
        S.panel(ax, letter, title, x=-0.2 if letter == "a" else -0.04)
    cax = fig.add_subplot(gs[3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=6.8, length=2, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.set_label("errors (%)", fontsize=7.2, labelpad=2)
    save(fig, "fig_lengths")
    plt.close(fig)


# ----------------------------------------------------------------------------- (3b) length x dimension
GRID_N, GRID_C = (25, 50, 100, 200), (2, 4, 6, 10)


def grid_runs(N, c):
    """Runs of one cell of the length x dimension grid (N = 25: the main runs and the embedding sizes)."""
    if N == 25:
        return ["main_s0", "main_s1", "main_s2"] if c == 10 else [f"emb_c{c}"]
    return [f"len_n{N}_c{c}"]


def curve_by_part(runs, P, key, edges):
    """Exact match by the longest part relative to the trained part length (pooled over runs)."""
    ok, x = [], []
    for r in runs:
        t = P[r]["tests"][key]
        ok.append(t["ok"])
        x.append(np.maximum(t["n_int"], t["n_frac"]) / P[r]["max_part_trained"])
    ok, x = np.concatenate(ok), np.concatenate(x)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x > a) & (x <= b)
        if m.sum() >= 30:
            xs.append(x[m].mean())
            ys.append(100.0 * ok[m].mean())
    return xs, ys


def fig_lengthdim():
    """(a) N x c grid: exact match inside the trained part lengths (worst test) and beyond them; (b, c) exact
    match of reconstruction and of the closure of x against the longest part relative to the trained length."""
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    P = load_parts()
    if not any(f"len_n{N}_c{c}" in P for N in GRID_N[1:] for c in GRID_C):
        placeholder("fig_lengthdim", "length x dimension runs (len_n*_c*), then python make_figures.py", 2.2)
        return
    plt = S.setup()
    cmap, norm = err_cmap(), LogNorm(vmin=0.01, vmax=100)
    fig = S.figure(2.45)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1, 1])
    a, b, c_ = (fig.add_subplot(gs[i]) for i in range(3))
    keys = [k for k, _ in LD_TESTS]
    fmt = lambda v: "100" if v >= 99.995 else (f"{v:.2f}" if v >= 99 else f"{v:.1f}")
    for i, N in enumerate(GRID_N):
        for j, c in enumerate(GRID_C):
            runs = [r for r in grid_runs(N, c) if r in P]
            if runs:
                v_in = min(np.mean([split_em(P[r], k, True)[0] for r in runs]) for k in keys)
                v_out = min(np.mean([split_em(P[r], k, False)[0] for r in runs]) for k in keys)
                e = min(100.0, max(0.01, 100.0 - v_in))
                a.add_patch(Rectangle((j, i), 1, 1, facecolor=cmap(norm(e)), edgecolor=SURFACE, linewidth=1.0))
                col = SURFACE if e >= 8 else INK
                a.text(j + 0.5, i + 0.4, fmt(v_in), ha="center", va="center", fontsize=7.4, color=col)
                a.text(j + 0.5, i + 0.76, f"beyond: {v_out:.0f}", ha="center", va="center", fontsize=5.6, color=col)
            else:
                a.add_patch(Rectangle((j, i), 1, 1, facecolor="#f4f4f4", edgecolor=SURFACE, linewidth=1.0))
                a.text(j + 0.5, i + 0.5, "not\nfinished", ha="center", va="center", fontsize=5.6, color=MUTED, linespacing=0.95)
    a.set_xlim(0, len(GRID_C))
    a.set_ylim(len(GRID_N), 0)
    a.set_xticks(np.arange(len(GRID_C)) + 0.5, labels=[str(c) for c in GRID_C])
    a.set_yticks(np.arange(len(GRID_N)) + 0.5, labels=[f"{N}\n$K\\!=\\!{2 * N + 2}$" for N in GRID_N])
    a.tick_params(length=0, pad=3)
    a.tick_params(axis="y", labelsize=6.8)
    for s_ in a.spines.values():
        s_.set_visible(False)
    a.set_xlabel("channels per slot $c$  ($d=Kc$)")
    a.set_ylabel("maximum digits $N$")
    S.panel(a, "a", "Exact match, worst test", x=-0.36)
    shades = {25: S.BLUES[3], 50: S.BLUES[5], 100: S.BLUES[7], 200: C[1]}
    edges = np.concatenate([np.linspace(0, 1, 6), np.linspace(1, 2.2, 10)[1:]])
    for ax, key, letter, title in ((b, "reconstruction", "b", "Reconstruction, $c=4$"),
                                   (c_, "closurex", "c", "Closure of $\\dot{\\times}$, $c=4$")):
        S.light_grid(ax)
        ax.axvspan(1.0, 2.3, color="#f4f4f4", linewidth=0, zorder=0)
        for N in GRID_N:
            runs = [r for r in grid_runs(N, 4) if r in P]
            if runs:
                xs, ys = curve_by_part(runs, P, key, edges)
                ax.plot(xs, ys, color=shades[N], marker="o", ms=2.2, linewidth=1.3, label=f"$N={N}$")
        ax.set_xlim(0, 2.25)
        ax.set_ylim(0, 103)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_xlabel("longest part / trained part length")
        S.panel(ax, letter, title, x=-0.2 if letter == "b" else -0.06)
    b.set_ylabel("exact match (%)")
    c_.set_yticklabels([])
    c_.text(1.06, 5, "beyond the\ntraining lengths", ha="left", va="bottom", fontsize=6.6, color=INK2, linespacing=1.0)
    b.legend(loc="lower left", handlelength=1.5, borderaxespad=0.3)
    save(fig, "fig_lengthdim")
    plt.close(fig)


# ----------------------------------------------------------------------------- (3c) length distributions
def fig_distributions():
    """Six of the candidate distributions of the number of digits per part (Section 5.1); the selected one,
    the negative binomial distribution with r = 2 and p = 0.45, is highlighted."""
    plt = S.setup()
    l = np.arange(0, 31)
    lf = np.array([math.lgamma(v + 1) for v in l])

    def nb(r, p):
        return np.exp(np.array([math.lgamma(v + r) for v in l]) - lf - math.lgamma(r) + r * math.log(p) + l * math.log(1 - p))

    panels = [("Geometric, $p=0.3$", 0.3 * 0.7 ** l, False), ("Poisson, $\\lambda=3$", np.exp(-3 + l * math.log(3) - lf), False),
              ("Negative binomial, $r=2$, $p=0.45$", nb(2, 0.45), True),
              ("Geometric, $p=0.01$", 0.01 * 0.99 ** l, False), ("Poisson, $\\lambda=7$", np.exp(-7 + l * math.log(7) - lf), False),
              ("Negative binomial, $r=10$, $p=0.3$", nb(10, 0.3), False)]
    fig = S.figure(2.7)
    axes = fig.subplots(2, 3, sharex=True)
    for ax, (title, g, sel), letter in zip(axes.flat, panels, "abcdef"):
        S.light_grid(ax)
        ax.bar(l, g, width=0.72, color=C[1] if sel else C[0], linewidth=0)
        ax.set_xlim(-0.8, 30.8)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_ylim(0, 1.12 * g.max())
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        S.panel(ax, letter, title, x=-0.16)
    for ax in axes[1]:
        ax.set_xlabel("number of digits $l$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$g(l)$")
    save(fig, "fig_distributions")
    plt.close(fig)


# ----------------------------------------------------------------------------- (4) geometry
def fig_geometry():
    plt = S.setup()
    p = os.path.join(ANALYSIS, "model_analysis.json")
    if not os.path.exists(p):
        print("model_analysis.json missing: run analysis/geometry.py")
        return
    A = json.load(open(p))
    fig = S.figure(2.2)
    gs = fig.add_gridspec(1, 4, width_ratios=[0.86, 0.95, 0.88, 0.95])
    a, sp, b, c = (fig.add_subplot(gs[i]) for i in range(4))
    g, pts = A["slot_geometry"], A["slot_geometry"]["points"]
    # (b) Fourier spectrum of the digit code (analysis/digit_code.py): share of the between-digit variance per harmonic
    px = os.path.join(ANALYSIS, "extra.json")
    X = json.load(open(px)) if os.path.exists(px) else {}
    models = [("main_s0", "seed 0"), ("main_s1", "seed 1"), ("main_s2", "seed 2"), ("emb_c2", "$c=2$"), ("emb_c4", "$c=4$"),
              ("emb_c6", "$c=6$"), ("emb_c16", "$c=16$"), ("len_n50_c10", "$N=50$"), ("len_n100_c10", "$N=100$")]
    models = [(r, lab) for r, lab in models if "fourier" in X.get(r, {})]
    shades = [S.BLUES[7], S.BLUES[4], "#d9d9d9"]
    for i, (r, lab) in enumerate(models):
        h = X[r]["fourier"]["harmonic_share"]
        parts = [100 * h[0], 100 * h[1], 100 * sum(h[2:])]
        left = 0.0
        for v, col in zip(parts, shades):
            sp.barh(i, v, left=left, height=0.7, color=col, edgecolor=SURFACE, linewidth=0.6)
            left += v
    sp.set_yticks(range(len(models)), labels=[lab for _, lab in models])
    sp.set_ylim(len(models) - 0.45, -1.75)
    sp.set_xlim(0, 100)
    sp.set_xticks([0, 50, 100])
    sp.tick_params(axis="y", length=0)
    sp.set_xlabel("variance between digits (%)")
    from matplotlib.patches import Patch
    sp.legend(handles=[Patch(color=shades[0], label="period 10"), Patch(color=shades[1], label="5"), Patch(color=shades[2], label="shorter")],
              loc="upper left", ncol=3, handlelength=0.9, handletextpad=0.35, columnspacing=0.7, borderaxespad=0.1, fontsize=6.6)
    S.panel(sp, "b", "Harmonics of the code", x=-0.3, gap=0.12)
    from matplotlib.colors import LinearSegmentedColormap
    ramp = LinearSegmentedColormap.from_list("digits", S.BLUES[2:])
    cols = [ramp(0.08 + 0.92 * d / 9) for d in range(10)]          # evenly spaced, distinguishable steps
    for d in range(10):
        q = [p_ for p_ in pts if p_["digit"] == d]
        a.scatter([p_["x"] for p_ in q], [p_["y"] for p_ in q], s=3, color=cols[d], linewidth=0, alpha=0.5)
    cen = [(np.mean([p_["x"] for p_ in pts if p_["digit"] == d]), np.mean([p_["y"] for p_ in pts if p_["digit"] == d]))
           for d in range(10)]
    ring = cen + [cen[0]]
    a.plot([x for x, _ in ring], [y for _, y in ring], color=MUTED, linewidth=0.7, zorder=1)
    cx, cy = np.mean([x for x, _ in cen]), np.mean([y for _, y in cen])
    R = max(math.hypot(x - cx, y - cy) for x, y in cen)
    for d, (x, y) in enumerate(cen):
        a.scatter([x], [y], s=22, color=cols[d], edgecolor=SURFACE, linewidth=0.8, zorder=3)
        dx, dy = x - cx, y - cy
        r = math.hypot(dx, dy) or 1
        a.text(x + 0.24 * R * dx / r, y + 0.24 * R * dy / r, str(d), ha="center", va="center", fontsize=7.6, color=INK, zorder=4)
    ring_view = (cx, cy, 1.42 * R)        # applied after the layout: centred, circular, same box height as b, c
    ev = g["explained_variance"]
    a.set_xlabel(f"PC 1 ({100 * ev[0]:.0f}%)")
    a.set_ylabel(f"PC 2 ({100 * ev[1]:.0f}%)")
    a.set_xticks([])
    a.set_yticks([])
    S.panel(a, "a", "Digit code (seed 0)", x=-0.2, gap=0.13)
    m = 6.2
    b.fill_between([0, m], [0, 0], [0, m], color="#eef4ea", linewidth=0, zorder=0)
    b.plot([0, m], [0, m], color=INK2, linewidth=0.7)
    for i, (op, lab) in enumerate((("+", "$\\dot{+}$"), ("x", "$\\dot{\\times}$"))):
        it = A["items"][op]
        rad, dfc = [i_["radius"] for i_ in it], [i_["defect"] for i_ in it]
        b.scatter(rad, dfc, s=3.5, color=C[i], linewidth=0, alpha=0.5, label=f"{lab} closure", zorder=2)
    b.text(5.95, 3.7, "defect\n< radius", ha="right", va="center", fontsize=6.8, color=INK2)
    b.set_xlim(0, m)
    b.set_ylim(0, m)
    b.set_xticks([0, 2, 4, 6])
    b.set_xlabel("decoding radius $\\hat\\rho(a*b)$")
    b.set_ylabel("defect $\\varepsilon_*(a,b)$")
    b.legend(loc="upper left", handletextpad=0.2, markerscale=2.4, borderaxespad=0.1, labelspacing=0.25)
    S.panel(b, "c", "Defect and radius", x=-0.2, gap=0.12)
    R = [t for t in (tests(r) for r in ("main_s0", "main_s1", "main_s2")) if t]
    Ks = list(range(2, 11))
    S.light_grid(c)
    for i, (op, lab) in enumerate((("chains+", "$\\dot{+}$"), ("chainsx", "$\\dot{\\times}$"))):
        err = [st.mean(t[op]["per_K"][str(K)]["err"]["median"] for t in R) for K in Ks]
        bnd = [st.mean(t[op]["per_K"][str(K)]["bound"]["median"] for t in R) for K in Ks]
        c.plot(Ks, err, color=C[i], label=lab)
        c.plot(Ks, bnd, color=C[i], linewidth=1.0, linestyle=(0, (1.0, 1.3)))
    c.set_yscale("log")
    c.set_xticks(range(2, 11, 2))
    c.set_xlabel("number of operands $M$")
    c.set_ylabel("median $\\|z_M-\\varphi(\\bar a_M)\\|$")
    c.legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.4, borderaxespad=0.1)
    c.text(9.9, 2.6, "measured", ha="right", va="bottom", fontsize=6.8, color=INK2)
    c.text(6.4, 2.2e3, "bound", ha="right", va="bottom", fontsize=6.8, color=INK2)
    c.set_ylim(0.4, 3e5)
    S.panel(c, "d", "Error in chains", x=-0.24, gap=0.12)
    fig.canvas.draw()                                     # final layout, then a circular, centred view in panel a
    bb = a.get_window_extent()
    rx, ry, half = ring_view
    ratio = bb.width / bb.height
    hx, hy = (half * ratio, half) if ratio >= 1 else (half, half / ratio)      # the ring fits the shorter side
    a.set_ylim(ry - hy, ry + hy)
    a.set_xlim(rx - hx, rx + hx)
    save(fig, "fig_geometry")
    plt.close(fig)


if __name__ == "__main__":
    fig_profile()
    fig_operators()
    fig_geometry()
    fig_lengths()
    fig_lengthdim()
    fig_distributions()
    print("figures in", os.path.abspath(OUT), "previews in", os.path.abspath(PREVIEW))
