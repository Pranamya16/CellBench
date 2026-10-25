"""Tests for the blood-like generator and the MDS hypogranular signal. Run: .venv/bin/python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ks_2samp, ttest_ind

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cellbench"))
import generator as G

N = 120  # patients per group in the tests


@pytest.fixture(scope="module")
def gen():
    return G.Generator()


def settings(kind="none", f=0.0, d=0.0, noise="shared", seed=1, n=N, **kw):
    return dict(seed=seed, noise=noise, patients_per_group=n, signal_type=kind, affected_fraction=f, shift_size=d,
                cohort_id="t", **kw)


def neut_mean(cells, pop, P, reading, rows):
    return np.array([cells[i][pop[i] == P.neut, reading].mean() for i in rows])


def test_groups_are_blood_like(gen):
    assert gen.P.names == ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "ig", "debris"]
    for gone in ["macrophage", "hfbf_macrophage", "mesothelial", "aggregate"]:
        assert gone not in gen.P.names
    t = gen.P.table
    assert (t["provenance"].str.len() > 20).all()                       # every group states where its position comes from
    assert t["provenance"].str.contains("body-fluid").sum() >= 5         # borrowed positions say so
    assert t.set_index("population").loc["ig", "position_status"].startswith("ASSUMED")
    assert t.set_index("population").loc["eosinophil", "position_status"].startswith("ASSUMED")


def test_config_has_no_strength(gen):
    assert "strength" not in json.dumps(gen.cfg).lower()


def test_deterministic_and_seed_sensitive(gen):
    a = gen.cohort(settings("subgroup", 0.1, 0.1))[0]
    b = G.Generator().cohort(settings("subgroup", 0.1, 0.1))[0]
    c = gen.cohort(settings("subgroup", 0.1, 0.1, seed=2))[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_event_count_floor_and_range(gen):
    cells = gen.cohort(settings())[0]
    assert cells.shape == (2 * N, gen.cfg["events_per_patient"], 4)
    assert cells[..., 2].min() >= G.FSC_FLOOR and cells.max() <= 255


def test_events_per_patient_is_a_parameter():
    g = G.Generator(events=500)
    assert g.cohort(settings(n=5))[0].shape[1] == 500


def test_group_proportions_match_targets(gen):
    cells, pop, *_ = gen.cohort(settings())
    got = np.array([(pop == j).mean() for j in range(len(gen.P.names))])
    want = gen.P.expected_shares()
    assert np.allclose(got, want, atol=0.025), (got.round(3), want.round(3))
    assert 1000 < (pop == gen.P.neut).sum(1).mean() < 1200  # about 1,100 neutrophils per 2,000 events


def test_normal_and_signal_share_the_same_composition(gen):
    _, pop, *_ = gen.cohort(settings("subgroup", 0.2, 0.2))
    nn = (pop == gen.P.neut).sum(1)
    assert ttest_ind(nn[:N], nn[N:]).pvalue > 0.01  # composition must not reveal the label


def test_zero_percent_is_a_clean_leakage_check(gen):
    cells, pop, planted, bg, lab = gen.cohort(settings("none"))
    assert planted.sum() == 0
    P = gen.P
    m = neut_mean(cells, pop, P, 0, range(2 * N))
    assert ttest_ind(m[:N], m[N:]).pvalue > 0.01
    # compare patients, not pooled cells: cells within a patient are not independent, so a pooled KS test
    # on ~100,000 cells flags differences far too small to matter
    sd = np.array([cells[i][pop[i] == P.neut, 0].std() for i in range(2 * N)])
    assert ks_2samp(m[:N], m[N:]).pvalue > 0.01 and ks_2samp(sd[:N], sd[N:]).pvalue > 0.01
    # same carrier rates of background dysplasia in both groups
    assert abs(bg[:N].any(1).mean() - bg[N:].any(1).mean()) < 0.25


def test_zero_fraction_leaves_cells_identical(gen):
    (xn, pn, _, bn), (xs, ps, infos, bs) = gen.base(1, "shared", N)
    for kind in ["subgroup", "spread"]:
        x2, aff = G.apply_signal(gen.P, gen.cfg, xs, ps, infos, bs, dict(type=kind, affected_fraction=0.0, shift_size=0.2), 1, 1)
        assert np.array_equal(x2, xs) and aff.sum() == 0


def test_subgroup_plants_the_right_fraction_and_drop(gen):
    P, f, d = gen.P, 0.20, 0.20
    (xn, pn, _, bn), (xs, ps, infos, bs) = gen.base(1, "shared", N)
    x2, aff = G.apply_signal(P, gen.cfg, xs, ps, infos, bs, dict(type="subgroup", affected_fraction=f, shift_size=d), 1, 1)
    n_neut = (ps == P.neut).sum(1)
    assert abs(aff.sum() / n_neut.sum() - f) < 0.01
    assert not aff[ps != P.neut].any()                      # only neutrophils are touched
    assert np.array_equal(x2[ps != P.neut], xs[ps != P.neut])
    ratio = x2[aff][:, 0] / xs[aff][:, 0]
    assert np.allclose(ratio, 1 - d, atol=1e-4)
    assert np.allclose(x2[aff][:, 1] / xs[aff][:, 1], 1 - gen.cfg["link"]["sfl"] * d, atol=1e-4)
    assert np.allclose(x2[aff][:, 2] / xs[aff][:, 2], 1 - gen.cfg["link"]["fsc"] * d, atol=1e-4)
    assert not (aff & bs).any()                              # no cell is hit by background and planted signal


def test_subgroup_is_a_distinct_low_ssc_group(gen):
    cells, pop, planted, bg, _ = gen.cohort(settings("subgroup", 0.2, 0.2))
    P = gen.P
    low = cells[N:][..., 0][planted[N:]]
    rest = cells[N:][..., 0][(pop[N:] == P.neut) & ~planted[N:] & ~bg[N:]]
    assert low.mean() < rest.mean() - 20  # about 0.2 x 154 = 31 units lower


def test_shift_hits_every_neutrophil_and_ignores_fraction(gen):
    P = gen.P
    (xn, pn, _, bn), (xs, ps, infos, bs) = gen.base(1, "shared", N)
    a = G.apply_signal(P, gen.cfg, xs, ps, infos, bs, dict(type="shift", affected_fraction=1.0, shift_size=0.1), 1, 1)
    b = G.apply_signal(P, gen.cfg, xs, ps, infos, bs, dict(type="shift", affected_fraction=0.02, shift_size=0.1), 1, 1)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[1][ps == P.neut].all()


def test_spread_widens_ssc_without_moving_the_centre(gen):
    P = gen.P
    (xn, pn, _, bn), (xs, ps, infos, bs) = gen.base(1, "shared", N)
    x2, aff = G.apply_signal(P, gen.cfg, xs, ps, infos, bs, dict(type="spread", affected_fraction=0.2, shift_size=0.2), 1, 1)
    m0, m1 = neut_mean(xs, ps, P, 0, range(N)), neut_mean(x2, ps, P, 0, range(N))
    s0 = np.array([xs[i][ps[i] == P.neut, 0].std() for i in range(N)])
    s1 = np.array([x2[i][ps[i] == P.neut, 0].std() for i in range(N)])
    assert abs((m1 - m0).mean()) < 0.1
    assert s1.mean() > 1.3 * s0.mean()


def test_report_averages_move_the_expected_way(gen):
    P = gen.P
    cells, pop, *_ = gen.cohort(settings("subgroup", 0.2, 0.2, noise="independent"))
    for r in (0, 1, 2):  # NE-SSC, NE-SFL, NE-FSC all lower in signal patients
        m = neut_mean(cells, pop, P, r, range(2 * N))
        assert m[N:].mean() < m[:N].mean()


def test_background_dysplasia_is_in_both_groups_and_can_be_switched_off(gen):
    _, _, _, bg, _ = gen.cohort(settings(noise="shared"))
    assert bg[:N].any(1).mean() > 0.2 and bg[N:].any(1).mean() > 0.2
    assert bg.sum(1).max() <= 0.02 * 1500 + 1
    _, _, _, bg0, _ = gen.cohort(settings(noise="shared_clean"))
    assert bg0.sum() == 0


def test_neutropenia_knob_is_off_by_default_and_leaks_when_on(gen):
    P = gen.P
    assert gen.cfg["neutropenia"]["enabled"] is False
    off = gen.cohort(settings(noise="independent_clean"))
    on = gen.cohort(settings(noise="independent_clean", neutropenia=True))
    n_off, n_on = (off[1] == P.neut).sum(1), (on[1] == P.neut).sum(1)
    assert ttest_ind(n_off[:N], n_off[N:]).pvalue > 0.01
    assert n_on[N:].mean() < 0.85 * n_on[:N].mean()


def test_manifest_counts_and_no_duplicates(gen):
    m = G.build_manifest(gen.cfg)
    assert len(m["full"]) == 180 and len(m["budget"]) == 108
    keys = [(r["signal_type"], r["affected_fraction"], r["shift_size"], r["noise"], r["seed"]) for r in m["full"]]
    assert len(set(keys)) == len(keys)
    assert sum(r["signal_type"] == "shift" for r in m["full"]) == 10   # one per noise preset and seed
    assert sum(r["signal_type"] == "none" for r in m["full"]) == 10


def test_budget_slice_equals_direct_build(gen):
    big = gen.cohort(settings("subgroup", 0.1, 0.1, n=N))[0]
    small = gen.cohort(settings("subgroup", 0.1, 0.1, n=40))[0]
    keep = np.r_[np.arange(40), N + np.arange(40)]
    assert np.array_equal(big[keep], small)
