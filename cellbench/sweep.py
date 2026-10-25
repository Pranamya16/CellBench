"""Run every cohort of a manifest through all main models, in parallel, resumable.

    python cellbench/sweep.py [--manifest full|budget] [--jobs 9] [--models report_GB rich_GB ...]
Writes data/results/part_<cohort_id>.csv per cohort (skipped if it exists) and merges them into data/results.csv.
Cohorts are rebuilt from the seed and config, so the numbers match the stored data/synthetic cohorts exactly.
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

args = sys.argv[1:]
which = args[args.index("--manifest") + 1] if "--manifest" in args else "full"
jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 9
models = None
if "--models" in args:
    i = args.index("--models") + 1
    models = []
    while i < len(args) and not args[i].startswith("--"):
        models.append(args[i]); i += 1

OUT = G.ROOT / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)
manifest = pd.read_csv(G.ROOT / "data" / "synthetic" / f"manifest_{which}.csv")
todo = [r for r in manifest.to_dict("records") if not (OUT / f"part_{r['cohort_id']}.csv").exists()]
print(f"{len(manifest)} cohorts in the {which} manifest, {len(todo)} still to run, {jobs} workers", flush=True)


def one(s):
    import warnings
    warnings.filterwarnings("ignore")
    rows = E.run_cohort(s, G.Generator(), models=models)
    pd.DataFrame(rows).to_csv(OUT / f"part_{s['cohort_id']}.csv", index=False)
    return s["cohort_id"]


t0 = time.time()
if todo:
    Parallel(n_jobs=jobs, verbose=5)(delayed(one)(s) for s in todo)
parts = [pd.read_csv(p) for p in sorted(OUT.glob("part_*.csv")) if p.stem.split("_", 1)[1].startswith(which + "_")]
res = pd.concat(parts, ignore_index=True)
res.to_csv(G.ROOT / "data" / f"results_{which}.csv", index=False)
print(f"merged {len(parts)} cohorts, {len(res)} rows into data/results_{which}.csv in {(time.time() - t0) / 60:.1f} min")
