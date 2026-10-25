"""Score every model on one cohort: AUROC with a bootstrap interval, sensitivity at 95% specificity, and plain accuracy.

All numbers come from patient-level cross-validation (each patient is scored by a model that never saw them).
Plain accuracy uses a cutoff chosen on the OTHER folds' out-of-fold scores, so it is not tuned on the patients it scores.
"""
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import gating
import generator as G
import models as M

ORACLE = ["report_LR", "report_GB", "rich_GB", "mlp_report", "mlp_rich"]
AUTO = ["report_LR_auto", "report_GB_auto", "rich_GB_auto", "mlp_report_auto", "mlp_rich_auto"]
CELLS = ["cells_hist_GB", "cells_CNN", "mlp_cells"]
MAIN = AUTO + CELLS + ORACLE


def sens_at_spec(y, s, spec=0.95):
    return float((s[y == 1] > np.quantile(s[y == 0], spec)).mean())


def cv_accuracy(y, oof, fold):
    correct = 0
    for k in np.unique(fold):
        te = fold == k
        grid = np.unique(np.quantile(oof[~te], np.linspace(0.02, 0.98, 49)))
        thr = grid[np.argmax([((oof[~te] > t) == (y[~te] == 1)).mean() for t in grid])]
        correct += ((oof[te] > thr) == (y[te] == 1)).sum()
    return correct / len(y)


def bootstrap_auc(y, s, n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def run_cohort(s, gen=None, models=None, folds=5, shuffle_labels=False, tag=""):
    """s: a settings dict as in the manifest. Returns one row per model."""
    gen = gen or G.Generator()
    cells, pop, _, _, lab = gen.cohort(s)
    pop_auto = gating.assign(gating.get_gate(gen), cells) if any(m.endswith("_auto") for m in (models or MAIN)) else None
    d = M.Data(cells, pop, lab["label"].to_numpy(), len(gen.P.names), pop_auto=pop_auto)
    if shuffle_labels:
        d.y = np.random.default_rng(s["seed"] + 12345).permutation(d.y)
    rows = []
    for name in models or MAIN:
        t0 = time.time()
        oof, fold = M.cv_scores(name, d, folds, seed=s["seed"])
        lo, hi = bootstrap_auc(d.y, oof)
        rows.append(dict(cohort_id=s.get("cohort_id", ""), signal_type=s["signal_type"], affected_fraction=s["affected_fraction"],
                         shift_size=s["shift_size"], noise=s["noise"], seed=s["seed"], patients_per_group=s["patients_per_group"],
                         tag=tag, model=name, view=M.VIEW[name], auroc=roc_auc_score(d.y, oof), auroc_lo=lo, auroc_hi=hi,
                         sens_at_95spec=sens_at_spec(d.y, oof), accuracy=cv_accuracy(d.y, oof, fold), seconds=time.time() - t0))
    return rows
