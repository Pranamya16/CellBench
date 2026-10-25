"""Write the manifests, the full cohort set and a small pilot set with ground truth.

    python cellbench/make_data.py                 # everything, 2,000 events per patient
    python cellbench/make_data.py --pilot         # manifests + pilot only
    python cellbench/make_data.py --events 10000  # more cells per patient (config value is overridden)
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import generator as G
from cohort_io import load_cohort

ROOT = G.ROOT
OUT = ROOT / "data" / "synthetic"
PILOT = ROOT / "data" / "pilot"
pilot_only = "--pilot" in sys.argv
events = int(sys.argv[sys.argv.index("--events") + 1]) if "--events" in sys.argv else None

gen = G.Generator(events=events)
cfg = gen.cfg
manifests = G.build_manifest(cfg)
OUT.mkdir(parents=True, exist_ok=True)
PILOT.mkdir(parents=True, exist_ok=True)
(OUT / "config.json").write_text(json.dumps(cfg, indent=2))
shutil.copy(ROOT / "data" / "population_table.csv", OUT / "population_table.csv")

# the budget plan is the first N patients of each group of the matching full cohort
full = pd.DataFrame(manifests["full"])
key = ["signal_type", "affected_fraction", "shift_size", "noise", "seed"]
budget = pd.DataFrame(manifests["budget"]).merge(
    full[key + ["cohort_id"]].rename(columns={"cohort_id": "source_cohort"}), on=key, how="left")
assert budget["source_cohort"].notna().all()
full.to_csv(OUT / "manifest_full.csv", index=False)
budget.to_csv(OUT / "manifest_budget.csv", index=False)
print(f"manifests: full {len(full)} cohorts, budget {len(budget)} cohorts "
      f"(each cohort = {manifests['full'][0]['patients_per_group']} normal + {manifests['full'][0]['patients_per_group']} signal patients)")

# pilot: a few cohorts with per-cell ground truth (group, planted flag, background-dysplasia flag)
pilot_rows = [("none", 0.0, 0.0), ("subgroup", 0.10, 0.10), ("subgroup", 0.20, 0.20), ("shift", 1.0, 0.10), ("spread", 0.20, 0.10)]
pilot = []
for noise in cfg["grid"]["noise"]:
    for kind, f, d in pilot_rows:
        pilot.append(dict(signal_type=kind, affected_fraction=f, shift_size=d, noise=noise, seed=1, patients_per_group=300,
                          cohort_id=f"pilot_{kind}_f{int(f * 100):03d}_d{int(d * 100):02d}_{noise}"))
t0 = time.time()
for s in pilot:
    cells, pop, planted, bg, lab = gen.cohort(s)
    d = PILOT / s["cohort_id"]
    d.mkdir(exist_ok=True)
    np.save(d / "cells.npy", cells)
    np.savez_compressed(d / "truth.npz", population=pop, planted=planted, background=bg, population_names=np.array(gen.P.names))
    lab.to_csv(d / "labels.csv", index=False)
    (d / "settings.json").write_text(json.dumps(s, indent=2))
pd.DataFrame(pilot).to_csv(PILOT / "pilot_manifest.csv", index=False)
print(f"pilot: {len(pilot)} cohorts in {time.time() - t0:.1f}s")

# one cohort as a plain CSV for people who want to open it in Excel or R
demo = next(s for s in pilot if s["cohort_id"] == "pilot_subgroup_f020_d20_shared")
cells, _, _, _, lab = gen.cohort(demo)
flat = pd.DataFrame(cells.reshape(-1, 4), columns=G.READINGS)
flat.insert(0, "label", np.repeat(lab["label"].to_numpy(), cells.shape[1]))
flat.insert(0, "patient_id", np.repeat(lab["patient_id"].to_numpy(), cells.shape[1]))
flat.to_csv(PILOT / "demo_cohort_cells.csv", index=False)
print("demo CSV:", len(flat), "rows")

if not pilot_only:
    t0 = time.time()
    for n, s in enumerate(manifests["full"], 1):
        cells, _, _, _, lab = gen.cohort(s)
        d = OUT / "cohorts" / s["cohort_id"]
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "cells.npy", cells)
        lab.to_csv(d / "labels.csv", index=False)
        if n % 30 == 0 or n == len(manifests["full"]):
            print(f"  {n}/{len(manifests['full'])} cohorts, {time.time() - t0:.0f}s", flush=True)
    # the budget view must equal a cohort generated directly with the budget patient count
    b = next(r for r in manifests["budget"] if r["signal_type"] == "subgroup" and r["seed"] == 2)
    direct, *_ = gen.cohort(b)
    src = budget.loc[budget.cohort_id == b["cohort_id"], "source_cohort"].iloc[0]
    sliced, _ = load_cohort(src, patients_per_group=b["patients_per_group"])
    assert np.array_equal(direct, sliced), "budget slice differs from a direct build"
    print("budget slice check passed")

(OUT / "README.txt").write_text(
    "Synthetic Sysmex XN WDF cohorts, peripheral-blood-like mix (neutrophils, lymphocytes, monocytes, eosinophils,\n"
    "immature granulocytes, debris/RBC ghosts). Cell positions are BORROWED from body-fluid WDF files. Not clinical data.\n"
    f"Events per patient: {cfg['events_per_patient']}.\n"
    "cohorts/<id>/cells.npy  uint8 (patients, events, 4): SSC, SFL, FSC, FSCW. First half normal, second half signal.\n"
    "cohorts/<id>/labels.csv one row per patient: label (0 normal, 1 signal), settings, n_neutrophils, n_planted, n_background.\n"
    "manifest_full.csv       all cohorts of the full plan (300 + 300 patients, seeds 1-5).\n"
    "manifest_budget.csv     budget plan (200 + 200 patients, seeds 1-3); source_cohort is the matching full cohort,\n"
    "                        load it with load_cohort(source_cohort, patients_per_group=200).\n"
    "signal_type: none (0% affected, leakage check), subgroup, shift (every neutrophil; affected_fraction is fixed at 1.0\n"
    "and ignored), spread. affected_fraction and shift_size replace the old 'strength' parameter.\n"
    "Ground truth (true group, planted flag, background-dysplasia flag per cell) is only saved for the pilot set in data/pilot.\n"
    "Any cohort can be rebuilt exactly from the code and config.json. See DATA_DESIGN.md.\n"
    "The superseded body-fluid / SFL-signal data is in archive/v1_body_fluid_sfl_signal/.\n")
print("done")
