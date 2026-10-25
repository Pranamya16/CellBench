"""How much do control patients differ from each other? Small multiples plus a variation summary.

    python cellbench/plot_variation.py
Writes to data/pilot/: controls_SSC_SFL.png, controls_SSC_FSC.png, controls_variation.png, and prints a table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import generator as G

OUT = G.ROOT / "data" / "pilot"
gen = G.Generator()
P = gen.P
NP = 60
COL = {"neutrophil": "#2a78d6", "lymphocyte": "#1baf7a", "monocyte": "#eb6834", "eosinophil": "#6250d6", "ig": "#e87ba4", "debris": "#b9b8b0"}
R = {n: i for i, n in enumerate(G.READINGS)}


def controls(preset):
    """The first NP control (normal) patients of a null cohort, seed 1."""
    (xn, pn, _, _), _ = gen.base(1, preset, NP)
    return G.finalize(xn), pn


def small_multiples(cells, pop, xr, yr, fname, title):
    fig, ax = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    X, Y = R[xr], R[yr]
    for k, a in enumerate(ax.ravel()):
        c, p = cells[k].astype(float), pop[k]
        jit = np.random.default_rng(k).uniform(-0.5, 0.5, (len(c), 2))
        for j in np.argsort([-(p == j).sum() for j in range(len(P.names))])[::-1]:  # rare groups drawn last, on top
            m = p == j
            a.scatter(c[m, X] + jit[m, 0], c[m, Y] + jit[m, 1], s=2.2, c=COL[P.names[j]], alpha=.5, linewidths=0, rasterized=True)
        sh = {P.names[j]: (p == j).mean() * 100 for j in range(len(P.names))}
        a.set_title(f"patient {k + 1}:  neut {sh['neutrophil']:.0f}%  lymph {sh['lymphocyte']:.0f}%  mono {sh['monocyte']:.0f}%\n"
                    f"eos {sh['eosinophil']:.1f}%  IG {sh['ig']:.1f}%  debris {sh['debris']:.0f}%", fontsize=9)
        a.set_xlim(0, 255); a.set_ylim(0, 255); a.set_xlabel(xr, fontsize=9); a.set_ylabel(yr, fontsize=9); a.tick_params(labelsize=8)
    h = [plt.Line2D([], [], marker="o", ls="", color=COL[n], label=l) for n, l in
         zip(P.names, ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "immature granulocytes", "debris / RBC ghosts"])]
    fig.legend(handles=h, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.savefig(OUT / fname, dpi=115)
    plt.close(fig)


cells_s, pop_s = controls("shared")
cells_i, pop_i = controls("independent")
small_multiples(cells_s, pop_s, "SSC", "SFL", "controls_SSC_SFL.png", "12 control patients, same settings (shared noise, seed 1), SSC vs SFL. Synthetic data.")
small_multiples(cells_s, pop_s, "SSC", "FSC", "controls_SSC_FSC.png", "The same 12 control patients, SSC vs FSC")


def stats(cells, pop):
    """Per-patient mean of each group and reading (rows: patient, group, reading)."""
    out = np.full((len(cells), len(P.names), 4), np.nan)
    for i in range(len(cells)):
        for j in range(len(P.names)):
            m = pop[i] == j
            if m.sum() > 5:
                out[i, j] = cells[i][m].mean(0)
    shares = np.stack([(pop == j).mean(1) * 100 for j in range(len(P.names))], 1)
    return out, shares


mu_s, sh_s = stats(cells_s, pop_s)
mu_i, sh_i = stats(cells_i, pop_i)

fig, ax = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
# 1. composition across patients vs the table's range
a = ax[0, 0]
names = [n for n in P.names]
order = [P.names.index(n) for n in ["neutrophil", "lymphocyte", "monocyte", "debris", "eosinophil", "ig"]]
bp = a.boxplot([sh_s[:, j] for j in order], positions=range(len(order)), widths=.55, patch_artist=True, showfliers=True,
               medianprops=dict(color="black"))
for patch, j in zip(bp["boxes"], order):
    patch.set_facecolor(COL[P.names[j]]); patch.set_alpha(.7)
exp = P.expected_shares() * 100
for k, j in enumerate(order):
    a.plot([k - .35, k + .35], [exp[j]] * 2, color="black", ls="--", lw=1)
a.set_xticks(range(len(order))); a.set_xticklabels([P.labels[j].replace(" (IG)", "").replace("Immature granulocytes", "IG") for j in order], fontsize=9)
a.set_ylabel("share of all events (%), log-like scale"); a.set_yscale("symlog", linthresh=1)
a.set_ylim(0, 90); a.set_yticks([0, 1, 2, 5, 10, 20, 50]); a.set_yticklabels(["0", "1", "2", "5", "10", "20", "50"])
a.set_title(f"Cell mix across {NP} control patients (dashed = target)")
# 2/3. where each patient's group centres sit
for a, mu, nm in [(ax[0, 1], mu_i, "independent"), (ax[1, 0], mu_s, "shared")]:
    for j in range(len(P.names)):
        if P.names[j] in ("debris",): continue
        a.scatter(mu[:, j, 0], mu[:, j, 1], s=22, c=COL[P.names[j]], alpha=.8, label=P.labels[j].replace(" (IG)", ""))
    a.set_xlabel("group mean SSC"); a.set_ylabel("group mean SFL"); a.set_xlim(40, 210); a.set_ylim(20, 140)
    a.set_title(f"Each dot = one patient's group centre ({nm} noise)")
ax[0, 1].legend(frameon=False, fontsize=8, loc="upper left")
# 4. is the variation shared across groups?
a = ax[1, 1]
jn, jl = P.names.index("neutrophil"), P.names.index("lymphocyte")
for mu, nm, c in [(mu_i, "independent", "#eb6834"), (mu_s, "shared", "#2a78d6")]:
    r = np.corrcoef(mu[:, jn, 1], mu[:, jl, 1])[0, 1]
    a.scatter(mu[:, jl, 1], mu[:, jn, 1], s=26, c=c, alpha=.8, label=f"{nm}: correlation {r:.2f}")
a.set_xlabel("lymphocyte mean SFL"); a.set_ylabel("neutrophil mean SFL")
a.set_title("Do patients' groups move together? (one dot per patient)"); a.legend(frameon=False)
fig.suptitle(f"Variation between {NP} control patients (seed 1, synthetic)", fontsize=14)
fig.savefig(OUT / "controls_variation.png", dpi=115)

rows = []
for nm, mu in [("independent", mu_i), ("shared", mu_s)]:
    for j in [P.names.index(n) for n in ["neutrophil", "lymphocyte", "monocyte"]]:
        sd = np.nanstd(mu[:, j], 0, ddof=1)
        rows.append(dict(preset=nm, group=P.labels[j], **{f"{r}_mean": round(np.nanmean(mu[:, j, i]), 1) for i, r in enumerate(["SSC", "SFL", "FSC"])},
                         **{f"{r}_CV%": round(100 * sd[i] / np.nanmean(mu[:, j, i]), 1) for i, r in enumerate(["SSC", "SFL", "FSC"])}))
print(pd.DataFrame(rows).to_string(index=False))
print("\nshare of events, mean (min - max) across patients, vs target:")
for j in order:
    print(f"  {P.labels[j]:28s} {sh_s[:, j].mean():5.1f}% ({sh_s[:, j].min():4.1f} - {sh_s[:, j].max():5.1f})  target {exp[j]:5.1f}%  table range {P.low[j]}-{P.high[j]}")
for nm, mu in [("independent", mu_i), ("shared", mu_s)]:
    print(f"{nm}: correlation of neutrophil and lymphocyte mean SFL across patients {np.corrcoef(mu[:, jn, 1], mu[:, jl, 1])[0, 1]:.2f}, "
          f"SSC {np.corrcoef(mu[:, jn, 0], mu[:, jl, 0])[0, 1]:.2f}")
