"""One-off: measure what NORMAL synthetic patients look like on the neutrophil numbers used to explain a result.

    python cellbench/serve/make_reference.py
Writes data/final/cnn/normal_reference.json (ranges from 160 normal patients, both noise presets, seeds the models never saw). The live page compares an
uploaded or generated patient with these ranges so the explanation can say, in words, what in the cells looks different from normal.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import gating
import generator as G
from inference import Scorer

N = 160


def tail_share(ssc):
    """Share of a patient's neutrophils sitting far BELOW that patient's own centre, measured against the width of the upper half (which a planted
    drop does not touch). About 5% for a normal patient; larger when a group of neutrophils has lower side scatter. Independent of a patient-wide shift."""
    med = np.median(ssc)
    up = ssc[ssc >= med]
    return float((ssc < med - 1.645 * np.sqrt(np.mean((up - med) ** 2))).mean())

sc = Scorer.__new__(Scorer)                 # only the generator and the gate are needed, not the networks
sc.gen = G.Generator()
sc.P = sc.gen.P
sc.gate = gating.get_gate(sc.gen)
sc.n_groups = len(sc.P.names)
rows, pool = [], []
for i in range(N):
    cells, _, _ = Scorer.make_patient(sc, "subgroup", 0, 0, ("shared", "independent")[i % 2], 40000 + i, signal=False)
    pred = gating.assign(sc.gate, cells[None])[0]
    ne = cells[pred == sc.P.neut].astype(float)
    pool.append(ne[:, 0])
    rows.append(dict(share=len(ne) / len(cells), ssc=ne[:, 0].mean(), sd=ne[:, 0].std(), sfl=ne[:, 1].mean(), fsc=ne[:, 2].mean(), ne=ne))
for r in rows:
    r["low"] = tail_share(r.pop("ne")[:, 0])
ref = dict(n=N)
for k in ("share", "ssc", "sd", "sfl", "fsc", "low"):
    v = np.array([r[k] for r in rows])
    ref[k] = {q: round(float(np.percentile(v, p)), 4) for q, p in (("q05", 5), ("q50", 50), ("q95", 95))}
out = Path(__file__).resolve().parents[2] / "data" / "final" / "cnn" / "normal_reference.json"
out.write_text(json.dumps(ref, indent=1))
print(json.dumps(ref, indent=1))
