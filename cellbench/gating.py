"""Auto-gating: assign every cell to a group from its four readings only, like an analyser does.

The gate is taught on REFERENCE patients: normal patients from a seed that is used nowhere else (901). The generator's true
labels are used only for that teaching. Afterwards the gate sees test patients without any labels, so the report and rich
summaries built from its groups ("auto-gated") no longer get free labels. The labelled versions stay as "oracle" best cases.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import generator as G

GATE_SEED = 901  # never a sweep seed (1-5) or a tuning seed (101-103)
_CACHE = {}


def train_gate(gen, seed=GATE_SEED, patients=60, cells=150_000):
    """Fit the gate on normal reference patients from both noise presets. Returns a fitted classifier."""
    X, y = [], []
    for preset in gen.cfg["grid"]["noise"]:
        (xn, pn, _, _), _ = gen.base(seed, preset, patients)
        X.append(G.finalize(xn).reshape(-1, 4)); y.append(pn.reshape(-1))
    X, y = np.concatenate(X), np.concatenate(y)
    rng = np.random.default_rng(0)
    keep = rng.choice(len(X), min(cells, len(X)), replace=False)
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, early_stopping=False, random_state=0).fit(X[keep], y[keep])


def get_gate(gen):
    key = (gen.cfg["events_per_patient"], GATE_SEED)
    if key not in _CACHE:
        _CACHE[key] = train_gate(gen)
    return _CACHE[key]


def assign(gate, cells):
    """cells: (patients, events, 4) uint8. Returns the predicted group of every cell, same shape without the last axis."""
    n, e, _ = cells.shape
    return gate.predict(cells.reshape(-1, 4)).reshape(n, e).astype(np.uint8)
