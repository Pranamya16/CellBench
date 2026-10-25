"""Figures for review: the three signal types on neutrophil SSC, and one normal vs one signal patient on SSC x SFL."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import generator as G

OUT = G.ROOT / "data" / "pilot"
OUT.mkdir(parents=True, exist_ok=True)
gen = G.Generator()
P = gen.P
CASES = [("subgroup", 0.20, 0.20), ("shift", 1.0, 0.20), ("spread", 0.20, 0.20)]


def cohort(kind, f, d, n=100):
    s = dict(seed=1, noise="shared", patients_per_group=n, signal_type=kind, affected_fraction=f, shift_size=d, cohort_id="plot")
    return gen.cohort(s)


fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
bins = np.arange(60, 215, 3)
for a, (kind, f, d) in zip(ax, CASES):
    cells, pop, planted, bg, lab = cohort(kind, f, d)
    n = len(cells) // 2
    for sl, name, c in [(slice(0, n), "normal", "#2a78d6"), (slice(n, None), "signal", "#d03b3b")]:
        v = cells[sl][..., 0][pop[sl] == P.neut]
        a.hist(v, bins=bins, density=True, histtype="step", lw=2, color=c, label=name)
    a.set(title=f"{kind}: {int(f * 100)}% affected, {int(d * 100)}% SSC drop" if kind != "shift" else f"shift: all neutrophils, {int(d * 100)}% SSC drop",
          xlabel="neutrophil SSC (granularity)", yticks=[])
    a.legend(frameon=False)
fig.savefig(OUT / "signal_types.png", dpi=130)

cells, pop, planted, bg, lab = cohort("subgroup", 0.20, 0.20)
n = len(cells) // 2
fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
for a, i, name in [(ax[0], 0, "Normal patient"), (ax[1], n, "Signal patient: 20% of neutrophils, 20% SSC drop")]:
    x, p = cells[i].astype(float), pop[i]
    for j, lbl in enumerate(P.labels):
        m = p == j
        if j == P.neut:
            a.scatter(x[m & ~planted[i] & ~bg[i], 0], x[m & ~planted[i] & ~bg[i], 1], s=5, c="#2a78d6", label="neutrophil")
            a.scatter(x[m & bg[i], 0], x[m & bg[i], 1], s=9, c="#eb6834", label="neutrophil, background dysplasia")
            a.scatter(x[planted[i], 0], x[planted[i], 1], s=9, c="#d03b3b", label="neutrophil, planted hypogranular")
        else:
            a.scatter(x[m, 0], x[m, 1], s=2, c="#b9b8b0", label="other groups" if j == 0 else None)
    a.set(title=name, xlabel="SSC (granularity)", ylabel="SFL (dye glow)", xlim=(0, 255), ylim=(0, 255))
    a.legend(frameon=False, markerscale=2, loc="upper left", fontsize=8)
fig.savefig(OUT / "sanity_normal_vs_signal.png", dpi=130)
print("saved", OUT / "signal_types.png", OUT / "sanity_normal_vs_signal.png")
