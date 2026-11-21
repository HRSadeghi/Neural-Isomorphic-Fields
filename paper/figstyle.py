"""Shared figure style of the paper: STIX fonts (identical to the text and formulas of the paper), figures
exactly as wide as the text block (6.48 in, so they are never scaled), 8 pt type, thin axes with outward
ticks, bold panel letters, a fixed colour-blind-safe categorical order and single-hue ramps for magnitudes."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXTWIDTH = 6.48                     # \textwidth of the cas-sc layout, inches
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, RULE, SURFACE = "#1a1a1a", "#555555", "#8c8c8c", "#d9d9d9", "#ffffff"
BLUES = ["#f3f7fc", "#dce9f8", "#b9d3f1", "#8db8e8", "#5e98db", "#3779c6", "#245ea6", "#174680", "#0e2f59"]


def setup():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.8, "ytick.major.size": 2.8, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round", "legend.frameon": False, "legend.handlelength": 1.8,
        "axes.titlelocation": "left", "axes.titlepad": 4, "pdf.fonttype": 42, "savefig.dpi": 300,
        "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.03})
    return plt


def figure(height, **kw):
    """A figure exactly as wide as the text block."""
    return plt.figure(figsize=(TEXTWIDTH, height), **kw)


def panel(ax, letter, title=None, x=-0.13, y=1.0, gap=0.07):
    """Bold panel letter outside the axes (Nature style) and an optional short title (gap: distance between the
    letter and the title in axes widths; larger for narrow panels)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="left")
    if title:
        ax.set_title(title, loc="left", fontsize=8, pad=4, x=x + gap if x < 0 else 0.0)


def light_grid(ax, axis="y"):
    ax.grid(True, axis=axis, color="#ececec", linewidth=0.5)
    ax.set_axisbelow(True)
