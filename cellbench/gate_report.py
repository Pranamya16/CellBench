"""How good is the auto-gate, and how much do the summaries lose without free labels? Light, single process."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import generator as G, gating, evaluate as E

gen = G.Generator(); P = gen.P
t = time.time(); gate = gating.get_gate(gen); print(f"gate trained in {time.time() - t:.0f}s on reference seed {gating.GATE_SEED}")

def cohort(kind, f, d, seed=902, noise="shared", n=60):
    return gen.cohort(dict(seed=seed, noise=noise, patients_per_group=n, signal_type=kind, affected_fraction=f, shift_size=d, cohort_id="g"))

cells, pop, planted, bg, lab = cohort("none", 0, 0)
pred = gating.assign(gate, cells)
print("\nGate accuracy on held-out NORMAL patients (seed 902), per true group:")
for j, name in enumerate(P.labels):
    m = pop == j
    print(f"  {name:28s} {m.sum():>7d} cells, {100 * (pred[m] == j).mean():5.1f}% assigned to the right group")
print(f"  overall {100 * (pred == pop).mean():.1f}%")

print("\nWhere do the PLANTED hypogranular neutrophils go? (signal patients only)")
for kind, f, d in [("subgroup", 0.20, 0.20), ("subgroup", 0.10, 0.10), ("subgroup", 0.05, 0.10)]:
    c, p, pl, b, l = cohort(kind, f, d)
    pr = gating.assign(gate, c)
    m = pl
    share = [(pr[m] == j).mean() * 100 for j in range(len(P.names))]
    print(f"  {kind} {int(f*100)}%/{int(d*100)}%: " + ", ".join(f"{P.labels[j].split(' (')[0]} {share[j]:.0f}%" for j in range(len(P.names)) if share[j] >= 0.5))

print("\nOracle vs auto-gated summaries, AUROC (1 seed, 150+150 patients, shared noise):")
rows = []
for kind, f, d in [("subgroup", 0.20, 0.20), ("subgroup", 0.10, 0.20), ("subgroup", 0.10, 0.10), ("subgroup", 0.05, 0.20), ("spread", 0.20, 0.10), ("none", 0.0, 0.0)]:
    s = dict(seed=102, noise="shared", patients_per_group=150, signal_type=kind, affected_fraction=f, shift_size=d, cohort_id="o")
    r = E.run_cohort(s, gen, models=E.ORACLE + E.AUTO)
    rows.append(dict(case=f"{kind} {int(f*100)}%/{int(d*100)}%", **{x["model"]: round(x["auroc"], 2) for x in r}))
print(pd.DataFrame(rows).set_index("case").to_string())
