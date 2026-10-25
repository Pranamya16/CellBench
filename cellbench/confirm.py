"""Confirmation bake-off on tuning seeds (never the sweep seeds 1-5), in parallel on this machine.

    python cellbench/confirm.py --models mlp_report mlp_rich mlp_cells [--seeds 101 102 103] [--n 300] [--jobs 3] [--out name]
Each cohort's rows are saved as soon as it finishes (data/confirm_parts/), so stopping it never loses finished work.
Writes data/confirmation_<out>.csv and prints the tables.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
import generator as G
import evaluate as E
import tasks as T


def opt(args, flag, cast=str, default=None, many=False):
    if flag not in args:
        return default
    i = args.index(flag) + 1
    if not many:
        return cast(args[i])
    out = []
    while i < len(args) and not args[i].startswith("--"):
        out.append(cast(args[i])); i += 1
    return out


args = sys.argv[1:]
seeds = opt(args, "--seeds", int, [101, 102, 103], many=True)
n = opt(args, "--n", int, 300)
jobs = opt(args, "--jobs", int, 3)
models = opt(args, "--models", str, E.MAIN, many=True)
name = opt(args, "--out", str, "all" if models == E.MAIN else "_".join(m.replace("mlp_", "") for m in models))
PARTS = G.ROOT / "data" / "confirm_parts" / name
PARTS.mkdir(parents=True, exist_ok=True)
tasks = T.confirmation_tasks(seeds, n)


def run(i, s, shuffle, tag):
    import warnings
    warnings.filterwarnings("ignore")
    f = PARTS / f"{i:03d}.csv"
    if f.exists():
        return
    pd.DataFrame(E.run_cohort(s, G.Generator(), models=models, shuffle_labels=shuffle, tag=tag)).to_csv(f, index=False)


t0 = time.time()
print(f"{len(tasks)} cohorts x {len(models)} models ({', '.join(models)}) on {jobs} workers", flush=True)
Parallel(n_jobs=jobs, verbose=5)(delayed(run)(i, *t) for i, t in enumerate(tasks))
df = pd.concat([pd.read_csv(p) for p in sorted(PARTS.glob("*.csv"))], ignore_index=True)
df.to_csv(G.ROOT / "data" / f"confirmation_{name}.csv", index=False)
print(f"\ndone in {(time.time() - t0) / 60:.1f} min, {len(df)} rows")

sig = df[df.tag == "signal"].copy()
sig["case"] = sig.signal_type + " f=" + (sig.affected_fraction * 100).round().astype(int).astype(str) + "% d=" + (sig.shift_size * 100).round().astype(int).astype(str) + "%"
pd.set_option("display.width", 250)
for noise in ["shared", "independent"]:
    print(f"\n=== AUROC, {noise} noise (mean over seeds) ===")
    print(sig[sig.noise == noise].pivot_table(index="case", columns="model", values="auroc").round(2)[models].to_string())
print("\n=== NULL cohorts: AUROC mean / sd / min / max over seeds and noise presets (should be about 0.5) ===")
print(df[df.tag.isin(["no_signal", "null_shuffled"])].groupby(["tag", "model"])["auroc"].agg(["mean", "std", "min", "max"]).round(3).to_string())
