"""10 control and 10 signal patients, drawn as scatter plots of the cell readings.

    python cellbench/plot_patients.py
Writes data/pilot/patients_overview.png (all cells, four scatter views) and
data/pilot/patients_neutrophils.png (neutrophils only, strong and weak signal).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import generator as G

OUT = G.ROOT / "data" / "pilot"
gen = G.Generator()
P = gen.P
N = 10
BLUE, RED, ORANGE, GREY = "#2a78d6", "#d03b3b", "#eb6834", "#b9b8b0"
R = {n: i for i, n in enumerate(G.READINGS)}


def cohort(f, d):
    s = dict(seed=1, noise="shared", patients_per_group=N, signal_type="subgroup", affected_fraction=f, shift_size=d, cohort_id="plot")
    return gen.cohort(s)


def draw(ax, cells, pop, planted, bg, rows, xr, yr, only_neut=False):
    X, Y = R[xr], R[yr]
    c = cells[rows].reshape(-1, 4).astype(float)
    p, pl, b = pop[rows].reshape(-1), planted[rows].reshape(-1), bg[rows].reshape(-1)
    jit = np.random.default_rng(0).uniform(-0.5, 0.5, (len(c), 2))  # values are whole numbers: jitter so dots do not stack
    nm = p == P.neut
    if not only_neut:
        o = ~nm
        ax.scatter(c[o, X] + jit[o, 0], c[o, Y] + jit[o, 1], s=1.2, c=GREY, alpha=.35, linewidths=0, rasterized=True)
    ax.scatter(c[nm & ~pl & ~b, X] + jit[nm & ~pl & ~b, 0], c[nm & ~pl & ~b, Y] + jit[nm & ~pl & ~b, 1], s=1.5, c=BLUE, alpha=.45, linewidths=0, rasterized=True)
    ax.scatter(c[b, X] + jit[b, 0], c[b, Y] + jit[b, 1], s=4, c=ORANGE, alpha=.9, linewidths=0, rasterized=True)
    ax.scatter(c[pl, X] + jit[pl, 0], c[pl, Y] + jit[pl, 1], s=4, c=RED, alpha=.9, linewidths=0, rasterized=True)
    ax.set_xlabel(xr); ax.set_ylabel(yr)
    ax.tick_params(labelsize=8)


VIEWS = [("SSC", "SFL"), ("SSC", "FSC"), ("FSC", "SFL"), ("FSCW", "FSC")]
cells, pop, planted, bg, lab = cohort(0.20, 0.20)
ctrl, sig = np.arange(N), np.arange(2 * N)[N:]
fig, ax = plt.subplots(2, 4, figsize=(18, 8.2), constrained_layout=True)
for j, (xr, yr) in enumerate(VIEWS):
    draw(ax[0, j], cells, pop, planted, bg, ctrl, xr, yr)
    draw(ax[1, j], cells, pop, planted, bg, sig, xr, yr)
    ax[0, j].set_title(f"{xr} vs {yr}")
    for r in (0, 1):
        ax[r, j].set_xlim(0, 255); ax[r, j].set_ylim(0, 255)
ax[0, 0].annotate("10 control patients\n(20,000 cells)", (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=11, weight="bold")
ax[1, 0].annotate("10 signal patients\n(20% of neutrophils, 20% SSC drop)", (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=11, weight="bold")
h = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
     [(GREY, "other cell groups"), (BLUE, "neutrophil"), (RED, "neutrophil, planted hypogranular"), (ORANGE, "neutrophil, background dysplasia")]]
fig.legend(handles=h, loc="outside lower center", ncol=4, frameon=False)
fig.suptitle("Control vs signal patients, all cells. Shared noise, seed 1, blood-like mix. Synthetic data.", fontsize=13)
fig.savefig(OUT / "patients_overview.png", dpi=120)

NV = [("SSC", "SFL"), ("SSC", "FSC"), ("SFL", "FSC")]
weak = cohort(0.10, 0.10)
rows = [("10 control patients", cells, ctrl), ("signal: 20% of neutrophils, 20% SSC drop", cells, sig),
        ("signal: 10% of neutrophils, 10% SSC drop", weak[0], sig)]
srcs = {0: (cells, pop, planted, bg), 1: (cells, pop, planted, bg), 2: (weak[0], weak[1], weak[2], weak[3])}
fig, ax = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
for i, (title, _, rr) in enumerate(rows):
    c, p, pl, b = srcs[i]
    for j, (xr, yr) in enumerate(NV):
        draw(ax[i, j], c, p, pl, b, rr, xr, yr, only_neut=True)
        ax[i, j].set_xlim(50, 210) if xr == "SSC" else None
        if i == 0: ax[i, j].set_title(f"neutrophils: {xr} vs {yr}")
    ax[i, 0].annotate(title, (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=10, weight="bold")
fig.suptitle("Neutrophils only: how the planted subgroup separates from normal cells (10 patients per row)", fontsize=13)
fig.savefig(OUT / "patients_neutrophils.png", dpi=120)
print("saved patients_overview.png and patients_neutrophils.png")
