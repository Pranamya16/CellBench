"""Print the sanity checks for the blood-like generator (Step 3 of the revision)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind

sys.path.insert(0, str(Path(__file__).parent))
import generator as G

gen = G.Generator()
P, N = gen.P, 300


def st(kind="none", f=0.0, d=0.0, noise="shared", seed=1):
    return dict(seed=seed, noise=noise, patients_per_group=N, signal_type=kind, affected_fraction=f, shift_size=d, cohort_id="t")


def nsum(cells, pop, r):
    return np.array([cells[i][pop[i] == P.neut, r].mean() for i in range(len(cells))])


for noise in ["independent", "shared"]:
    print(f"\n===== noise preset: {noise} =====")
    cells, pop, planted, bg, lab = gen.cohort(st(noise=noise))
    got = np.array([(pop == j).mean() for j in range(len(P.names))])
    want = P.expected_shares()
    ev = cells.shape[1]
    print(f"1. group shares of all events (600 null patients, {ev} events each)")
    print(pd.DataFrame(dict(group=P.labels, target_pct=(want * 100).round(1), generated_pct=(got * 100).round(1))).to_string(index=False))
    nn = (pop == P.neut).sum(1)
    print(f"2. neutrophils per patient: mean {nn.mean():.0f}, 5th-95th percentile {np.percentile(nn, 5):.0f}-{np.percentile(nn, 95):.0f} (expected about {want[P.neut] * ev:.0f})")
    m, s = nsum(cells, pop, 0), np.array([cells[i][pop[i] == P.neut, 0].std() for i in range(2 * N)])
    print("3. 0% affected, normal vs signal group (leakage check):")
    print(f"   neutrophil SSC mean  {m[:N].mean():.2f} vs {m[N:].mean():.2f}  (t-test p = {ttest_ind(m[:N], m[N:]).pvalue:.2f})")
    print(f"   neutrophil SSC SD    {s[:N].mean():.2f} vs {s[N:].mean():.2f}  (KS on patient SDs p = {ks_2samp(s[:N], s[N:]).pvalue:.2f})")
    print(f"   neutrophil count     {nn[:N].mean():.0f} vs {nn[N:].mean():.0f}  (t-test p = {ttest_ind(nn[:N], nn[N:]).pvalue:.2f})")
    print(f"   background dysplasia carriers {bg[:N].any(1).mean():.0%} vs {bg[N:].any(1).mean():.0%}; max hypogranular cells in one patient {bg.sum(1).max()}")
    print("4. report-summary means of neutrophils (NE-SSC / NE-SFL / NE-FSC), normal -> signal")
    for kind, f, d in [("subgroup", 0.20, 0.20), ("subgroup", 0.10, 0.10), ("subgroup", 0.02, 0.10), ("shift", 1.0, 0.10), ("spread", 0.20, 0.10)]:
        c, p, pl, b, _ = gen.cohort(st(kind, f, d, noise))
        mm = [nsum(c, p, r) for r in (0, 1, 2)]
        t = ttest_ind(mm[0][:N], mm[0][N:]).pvalue
        print(f"   {kind:8s} f={f:4.0%} d={d:3.0%}: " + "  ".join(f"{x[:N].mean():6.1f} -> {x[N:].mean():6.1f}" for x in mm) + f"   (NE-SSC p = {t:.3g}, planted cells/patient {pl[N:].sum(1).mean():.0f})")
