"""Tests for the data views and models. Run: .venv/bin/python -m pytest tests -q"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cellbench"))
import features as F
import generator as G
import models as M

N = 80  # patients per group in these tests


@pytest.fixture(scope="module")
def gen():
    return G.Generator()


def data(gen, kind="none", f=0.0, d=0.0, noise="shared", seed=101):
    s = dict(seed=seed, noise=noise, patients_per_group=N, signal_type=kind, affected_fraction=f, shift_size=d, cohort_id="t")
    cells, pop, _, _, lab = gen.cohort(s)
    return M.Data(cells, pop, lab["label"].to_numpy(), len(gen.P.names))


def test_per_cell_features_never_see_group_labels():
    assert list(inspect.signature(F.histogram_features).parameters) == ["cells", "bins"]


def test_histogram_features_sum_to_one_per_map(gen):
    d = data(gen)
    h = d.hist.reshape(len(d.y), 3, -1)
    assert np.allclose(h.sum(-1), 1.0, atol=1e-5)


def test_feature_widths(gen):
    d = data(gen)
    assert d.report.shape[1] == 5 * len(gen.P.names)
    assert d.rich.shape[1] == 25 * len(gen.P.names)
    assert d.hist.shape[1] == 3 * 32 * 32


def test_split_is_by_patient_and_covers_everyone(gen):
    d = data(gen)
    seen = []

    def spy(dd, tr, te, seed):
        assert np.intersect1d(tr, te).size == 0          # a patient is never in both sets
        seen.append(te)
        return np.zeros(len(te))

    M.OPTIONS["_spy"] = spy
    try:
        oof, fold = M.cv_scores("_spy", d, folds=5)
    finally:
        del M.OPTIONS["_spy"]
    assert sorted(np.concatenate(seen).tolist()) == list(range(len(d.y)))
    assert set(fold) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize("name", ["report_GB", "rich_GB", "cells_hist_GB", "mlp_cells"])
def test_null_cohort_scores_about_chance(gen, name):
    # 160 patients: the standard error of an AUROC is about 0.04, so 0.35 to 0.65 is a wide, safe band
    auc = M.cv_auroc(name, data(gen))
    assert 0.35 < auc < 0.65, auc


def test_strong_subgroup_is_found_by_distribution_views_but_averages_lag(gen):
    d = data(gen, "subgroup", 0.20, 0.20)
    cells = M.cv_auroc("cells_hist_GB", d)
    rich = M.cv_auroc("rich_GB", d)
    assert cells > 0.9 and rich > 0.9


def test_medium_subgroup_averages_miss_what_the_rich_summary_finds(gen):
    d = data(gen, "subgroup", 0.10, 0.10)
    assert M.cv_auroc("rich_GB", d) > M.cv_auroc("report_GB", d) + 0.2


def test_same_mlp_runs_on_all_three_views_and_returns_probabilities(gen):
    d = data(gen, "subgroup", 0.20, 0.20)
    tr, te = np.arange(0, 2 * N, 2), np.arange(1, 2 * N, 2)
    for name in ["mlp_report", "mlp_rich", "mlp_cells"]:
        p = M.OPTIONS[name](d, tr, te, 0)
        assert p.shape == (len(te),) and (p >= 0).all() and (p <= 1).all()


# ---- auto-gating -------------------------------------------------------------------------------------------------
import gating


def test_gate_is_taught_on_a_seed_used_nowhere_else():
    assert gating.GATE_SEED not in (1, 2, 3, 4, 5, 101, 102, 103)


def test_gate_sees_only_the_four_readings_not_labels(gen):
    assert list(inspect.signature(gating.assign).parameters) == ["gate", "cells"]
    assert gating.get_gate(gen).n_features_in_ == 4


def test_gate_is_accurate_on_held_out_normal_patients_and_keeps_most_hypogranular_cells_as_neutrophils(gen):
    gate = gating.get_gate(gen)
    s = dict(seed=902, noise="shared", patients_per_group=30, signal_type="subgroup", affected_fraction=0.2, shift_size=0.2, cohort_id="t")
    cells, pop, planted, bg, lab = gen.cohort(s)
    pred = gating.assign(gate, cells)
    assert (pred[:30] == pop[:30]).mean() > 0.93                       # normal patients
    assert (pred[planted] == gen.P.neut).mean() > 0.75                 # planted cells mostly stay in the neutrophil group


def test_auto_gated_features_have_the_same_shape_as_oracle_and_need_no_true_groups(gen):
    s = dict(seed=101, noise="shared", patients_per_group=20, signal_type="none", affected_fraction=0.0, shift_size=0.0, cohort_id="t")
    cells, pop, *_ , lab = gen.cohort(s)
    d = M.Data(cells, pop, lab["label"].to_numpy(), len(gen.P.names), pop_auto=gating.assign(gating.get_gate(gen), cells))
    assert d.report_auto.shape == d.report.shape and d.rich_auto.shape == d.rich.shape


def test_folds_are_shuffled_and_balanced_even_though_the_file_is_ordered(gen):
    """cells.npy stores all normal patients first and all signal patients last. The cross-validation must not care."""
    d = data(gen)
    seen = {}

    def spy(dd, tr, te, seed):
        seen[len(seen)] = (tr, te)
        return np.zeros(len(te))

    M.OPTIONS["_spy"] = spy
    try:
        M.cv_scores("_spy", d, folds=5)
    finally:
        del M.OPTIONS["_spy"]
    for tr, te in seen.values():
        assert 0.4 < d.y[tr].mean() < 0.6 and 0.4 < d.y[te].mean() < 0.6   # both classes in every train and test set
        assert abs(d.y[te].mean() - 0.5) < 0.02                             # stratified: each test fold is balanced
    first_block = np.arange(10)                                             # the first 10 patients are all "normal" in the file
    folds_of_first_block = {k for k, (tr, te) in seen.items() if np.intersect1d(te, first_block).size}
    assert len(folds_of_first_block) >= 3                                   # ...but they are spread over the test folds


def test_report_cnn_grid_is_one_row_per_group_and_one_column_per_statistic(gen):
    s = dict(seed=101, noise="shared", patients_per_group=20, signal_type="none", affected_fraction=0.0, shift_size=0.0, cohort_id="t")
    cells, pop, *_, lab = gen.cohort(s)
    d = M.Data(cells, pop, lab["label"].to_numpy(), len(gen.P.names), pop_auto=gating.assign(gating.get_gate(gen), cells))
    assert d.report_auto.shape[1] == len(gen.P.names) * 5      # reshapes to (groups, 5 statistics)
    assert "cnn_report_auto" in M.OPTIONS and M.VIEW["cnn_report_auto"] == "report_auto"
