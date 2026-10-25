"""Read cohorts written by make_data.py.

    cells, labels = load_cohort("full_0123")                       # 300 + 300 patients
    cells, labels = load_cohort("full_0123", patients_per_group=200)  # the budget plan's slice
cells: uint8 array (patients, events, 4) with readings SSC, SFL, FSC, FSCW. The first half of the
patients are normal (label 0), the second half signal (label 1).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SYNTH = ROOT / "data" / "synthetic"


def load_cohort(cohort_id, patients_per_group=None, folder=SYNTH / "cohorts"):
    d = Path(folder) / cohort_id
    cells = np.load(d / "cells.npy", mmap_mode="r")
    labels = pd.read_csv(d / "labels.csv")
    if patients_per_group is not None:
        n = len(labels) // 2
        keep = np.r_[np.arange(patients_per_group), n + np.arange(patients_per_group)]
        return np.asarray(cells[keep]), labels.iloc[keep].reset_index(drop=True)
    return np.asarray(cells), labels
