"""Time-boxed dry run: race the model options on a few small tuning cohorts (seeds outside the sweep).

    python cellbench/bakeoff.py [--skip-nn] [--n 150] [--seed 101]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import generator as G
import models as M

n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 150
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 101
names = [k for k in M.OPTIONS if not (("--skip-nn" in sys.argv) and k in ("cells_CNN", "cells_DeepSets"))]

gen = G.Generator()
CASES = [("easy: subgroup 20% / 20%", "subgroup", 0.20, 0.20), ("medium: subgroup 10% / 10%", "subgroup", 0.10, 0.10),
         ("spread 20% / 10%", "spread", 0.20, 0.10), ("null: 0% (must be about 0.5)", "none", 0.0, 0.0)]
rows, t_all = [], time.time()
for label, kind, f, d in CASES:
    s = dict(seed=seed, noise="shared", patients_per_group=n, signal_type=kind, affected_fraction=f, shift_size=d, cohort_id="dry")
    cells, pop, planted, bg, lab = gen.cohort(s)
    data = M.Data(cells, pop, lab["label"].to_numpy(), len(gen.P.names))
    for name in names:
        t0 = time.time()
        auc = M.cv_auroc(name, data)
        rows.append(dict(case=label, option=name, auroc=round(auc, 3), seconds=round(time.time() - t0, 1)))
        print(f"{label:32s} {name:15s} AUROC {auc:.3f}  ({time.time() - t0:5.1f}s)  [total {time.time() - t_all:4.0f}s]", flush=True)

df = pd.DataFrame(rows)
df.to_csv(G.ROOT / "data" / "bakeoff_dryrun.csv", index=False)
print("\nLEADERBOARD (AUROC; null column should be about 0.5)")
print(df.pivot(index="option", columns="case", values="auroc").to_string())
print("\ntime per option (s, summed over cohorts)")
print(df.groupby("option")["seconds"].sum().to_string())
