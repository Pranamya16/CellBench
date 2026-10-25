"""Build the CellBench prototype page from the real result files (formatting only, no model runs).

    python cellbench/build_prototype.py <page_fragment.html> <standalone.html> [final_dir]
Reads data/results_full_sweep_summary.csv and data/detection_limits_all.csv; if final_dir holds the final-model outputs
(final_metrics.csv, demo_patients.json, demo_heatmaps.json for pair "cnn"), the patient card is filled too.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
frag_path, full_path = Path(sys.argv[1]), Path(sys.argv[2])
final_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None

s = pd.read_csv(ROOT / "data" / "results_full_sweep_summary.csv")
RES = {}
for r in s.itertuples():
    k = f"{r.noise}|{r.signal_type}|{round(r.affected_fraction * 100)}|{round(r.shift_size * 100)}|{r.model}"
    RES[k] = [round(r.auroc_mean, 3), round(r.auroc_sd, 3), round(r.accuracy_mean, 3)]
dl = pd.read_csv(ROOT / "data" / "detection_limits_all.csv")
LIM = {f"{r.noise}|{r.type}|{r.drop.rstrip('%')}|{r.view}": r.limit for r in dl.itertuples()}

DEMO, HEAT, FINAL = None, None, None
if final_dir and (final_dir / "demo_patients.json").exists():
    DEMO = json.loads((final_dir / "demo_patients.json").read_text())
    if (final_dir / "demo_heatmaps.json").exists():
        HEAT = json.loads((final_dir / "demo_heatmaps.json").read_text())["heatmaps"]
    fm = pd.read_csv(final_dir / "final_metrics.csv")
    FINAL = [dict(view=r.view, noise=r.noise, type=r.signal_type, f=None if pd.isna(r.affected_fraction) else round(r.affected_fraction * 100),
                  d=None if pd.isna(r.shift_size) else round(r.shift_size * 100), auroc=round(r.auroc, 3), acc=round(r.accuracy, 3))
             for r in fm.itertuples() if r.noise in ("shared", "all")]

page = (Path(__file__).parent / "prototype_template.html").read_text()
page = (page.replace("__RES__", json.dumps(RES)).replace("__LIM__", json.dumps(LIM))
        .replace("__DEMO__", json.dumps(DEMO)).replace("__HEAT__", json.dumps(HEAT)).replace("__FINAL__", json.dumps(FINAL)))
frag_path.write_text(page)
i, j = page.index("<style>"), page.index("</style>") + 8
full_path.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                     + page[:i] + page[i:j] + "\n</head>\n<body>\n" + page[j:] + "\n</body>\n</html>\n")
print(f"built: {len(RES)} result cells, {len(LIM)} limits, demo patients: {0 if DEMO is None else len(DEMO['patients'])}")
