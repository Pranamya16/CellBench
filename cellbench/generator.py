"""Synthetic Sysmex XN WDF patients (peripheral-blood-like) with a planted MDS-style hypogranular
neutrophil signal.

Pipeline for one patient (see DATA_DESIGN.md):
  1. draw the white-cell mix (logistic-normal around the typical shares, clipped to the table's range);
     normal and signal patients use the SAME distribution
  2. draw patient-level noise (multiplicative gain per group and reading; 'independent' or 'shared')
  3. draw the cells of each group from its centre / SD / correlations
  4. background dysplasia (noise preset option): some patients, normal or signal, carry a small
     hypogranular neutrophil fraction
  5. signal patients only: make neutrophils hypogranular (subgroup / shift / spread)
  6. round, apply the FSC trigger floor, clip to 0-255, store as uint8

Base patients depend only on (seed, preset, group, index, neutropenia on/off). The planted signal uses
its own random stream, so the same base patient can be shown with and without a signal.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
READINGS = ["SSC", "SFL", "FSC", "FSCW"]
FSC_FLOOR = 10
TOP = 255
SIGNAL_TYPES = ("none", "subgroup", "shift", "spread")


class Populations:
    def __init__(self, table=ROOT / "data" / "population_table.csv"):
        df = pd.read_csv(table)
        self.table = df
        self.names = df["population"].tolist()
        self.labels = df["label"].tolist()
        self.deb = self.names.index("debris")
        self.neut = self.names.index("neutrophil")
        self.centre = df[[f"{r}_centre" for r in READINGS]].to_numpy(float)
        self.sd = df[[f"{r}_sd" for r in READINGS]].to_numpy(float)
        self.chol = np.stack([self._chol(row) for _, row in df.iterrows()])
        self.typ = df["pct_typical"].to_numpy(float)
        self.low = df["pct_low"].to_numpy(float)
        self.high = df["pct_high"].to_numpy(float)
        lo = np.where(self.low <= 0, self.typ / 10, self.low)  # a zero lower bound has no finite log
        self.log_sd = (np.log(self.high) - np.log(lo)) / (2 * 1.645)

    @staticmethod
    def _chol(row):
        r = np.eye(4)
        for (i, j), key in {(0, 1): "r_SSC_SFL", (0, 2): "r_SSC_FSC", (0, 3): "r_SSC_FSCW",
                            (1, 2): "r_SFL_FSC", (1, 3): "r_SFL_FSCW", (2, 3): "r_FSC_FSCW"}.items():
            r[i, j] = r[j, i] = row[key]
        w, v = np.linalg.eigh(r)  # nudge to positive definite if the rounded table values need it
        r = v @ np.diag(np.maximum(w, 1e-3)) @ v.T
        d = np.sqrt(np.diag(r))
        return np.linalg.cholesky(r / np.outer(d, d))

    def expected_shares(self):
        """Expected share of ALL events per group at the typical values (nucleated groups renormalised)."""
        nuc = np.where(np.arange(len(self.names)) == self.deb, 0, self.typ)
        deb = self.typ[self.deb] / 100
        p = nuc / nuc.sum() * (1 - deb)
        p[self.deb] = deb
        return p


def _draw_mix(rng, P, neut_scale=1.0):
    z = rng.standard_normal(len(P.names))
    raw = np.clip(P.typ * np.exp(P.log_sd * z - P.log_sd ** 2 / 2), P.low, P.high)  # mean stays at typical
    raw[P.neut] *= neut_scale
    deb = raw[P.deb] / 100
    nuc = np.where(np.arange(len(raw)) == P.deb, 0, raw)
    p = nuc / nuc.sum() * (1 - deb)  # basophils are not in the WDF mix, so the rest renormalise to 100%
    p[P.deb] = deb
    return p


def _draw_noise(rng, P, kind, sizes):
    n = len(P.names)
    if kind == "independent":
        eps = rng.normal(0, sizes["independent_sd"], (n, 4))
    else:
        common = rng.normal(0, sizes["shared_sd"], 4)
        common[3] = common[2]  # FSCW follows FSC (pulse width tracks size)
        eps = common[None, :] + rng.normal(0, sizes["shared_residual_sd"], (n, 4))
    eps[P.sd == 0] = 0  # readings pinned at the ceiling stay pinned
    return eps


def _draw_cells(rng, n, mean, sd, L, eps):
    out, got = [], 0
    for _ in range(200):
        z = (rng.standard_normal((max(2 * (n - got), 8), 4)) @ L.T) * sd + mean
        z = z * (1 + eps)
        z = z[z[:, 2] >= FSC_FLOOR - 0.5]  # trigger floor: events below it are never recorded
        out.append(z[: n - got])
        got += len(out[-1])
        if got >= n:
            return np.vstack(out)
    raise RuntimeError("rejection sampling failed")


def hypogranular(x, idx, drop, link):
    """Make the cells x[idx] hypogranular in place: SSC down by `drop`, SFL and FSC by a linked, smaller amount."""
    x[idx, 0] *= 1 - drop
    x[idx, 1] *= 1 - link["sfl"] * drop
    x[idx, 2] *= 1 - link["fsc"] * drop


def make_base_patient(P, cfg, seed, preset, group, index, events, neutropenia=False):
    """Returns (cells float32 (events, 4), group index uint8, info dict, background-dysplasia mask)."""
    kind = cfg["noise_presets"][preset]["kind"]
    ss = np.random.SeedSequence([seed, list(cfg["noise_presets"]).index(preset), group, index])
    rng = np.random.default_rng(ss)
    scale = 1 - cfg["neutropenia"]["relative_reduction"] if (neutropenia and group == 1) else 1.0
    mix = _draw_mix(rng, P, scale)
    counts = rng.multinomial(events, mix)
    eps = _draw_noise(rng, P, kind, cfg["noise_sizes"])
    parts, pops = [], []
    for j, k in enumerate(counts):
        if k:
            parts.append(_draw_cells(rng, k, P.centre[j], P.sd[j], P.chol[j], eps[j]))
            pops.append(np.full(k, j, dtype=np.uint8))
    x, pop = np.vstack(parts), np.concatenate(pops)
    o = rng.permutation(len(x))  # real files carry no order information
    x, pop = x[o].astype(np.float32), pop[o]
    bg = np.zeros(len(x), dtype=bool)
    bd = cfg["background_dysplasia"]
    if cfg["noise_presets"][preset]["background_dysplasia"]:
        r2 = np.random.default_rng(np.random.SeedSequence([seed, list(cfg["noise_presets"]).index(preset), group, index, 7]))
        carrier = r2.random() < bd["carrier_share"]
        frac = r2.uniform(0, bd["max_fraction"]) if carrier else 0.0
        idx = np.flatnonzero(pop == P.neut)
        k = int(np.floor(frac * len(idx) + r2.random()))
        sel = r2.choice(idx, k, replace=False) if k else idx[:0]
        hypogranular(x, sel, bd["shift_size"], cfg["link"])
        bg[sel] = True
    info = dict(neut_ssc_mu=P.centre[P.neut, 0] * (1 + eps[P.neut, 0]),
                neut_ssc_sd=P.sd[P.neut, 0] * (1 + eps[P.neut, 0]))
    return x, pop, info, bg


def make_base_group(P, cfg, seed, preset, group, n_patients, events, neutropenia=False):
    xs, pops, infos, bgs = [], [], [], []
    for i in range(n_patients):
        x, pop, info, bg = make_base_patient(P, cfg, seed, preset, group, i, events, neutropenia)
        xs.append(x); pops.append(pop); infos.append(info); bgs.append(bg)
    return np.stack(xs), np.stack(pops), infos, np.stack(bgs)


def apply_signal(P, cfg, x, pop, infos, bg, signal, seed, group):
    """Plant the signal. x: (patients, events, 4) float. Returns (x', planted-cell mask); input is not modified.

    subgroup: a fraction f of each patient's neutrophils forms a low-SSC group (SSC x (1 - d)).
    shift:    EVERY neutrophil moves down by d. The affected fraction is ignored (fixed at 100%).
    spread:   neutrophil SSC widens, centre unchanged, to the same variance as the subgroup mixture at (f, d).
    """
    x = x.copy()
    affected = np.zeros(pop.shape, dtype=bool)
    kind, f, d = signal["type"], signal["affected_fraction"], signal["shift_size"]
    if kind == "none" or d == 0 or (kind != "shift" and f == 0):
        return x, affected
    for i in range(len(x)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, group, i, 99]))
        idx = np.flatnonzero(pop[i] == P.neut)
        if kind == "subgroup":
            k = int(np.floor(f * len(idx) + rng.random()))  # stochastic rounding keeps the expected fraction
            pool = idx[~bg[i, idx]]  # background-dysplastic cells are not hit twice
            sel = rng.choice(pool, min(k, len(pool)), replace=False) if k else pool[:0]
            hypogranular(x[i], sel, d, cfg["link"])
            affected[i, sel] = True
        elif kind == "shift":
            hypogranular(x[i], idx, d, cfg["link"])
            affected[i, idx] = True
        elif kind == "spread":
            mu, sd = infos[i]["neut_ssc_mu"], infos[i]["neut_ssc_sd"]
            k = np.sqrt(1 + f * (1 - f) * (d * mu / sd) ** 2)
            x[i, idx, 0] = mu + k * (x[i, idx, 0] - mu)
            affected[i, idx] = True
        else:
            raise ValueError(kind)
    return x, affected


def finalize(x):
    x = np.rint(x)
    x[..., 2] = np.maximum(x[..., 2], FSC_FLOOR)
    return np.clip(x, 0, TOP).astype(np.uint8)


class Generator:
    def __init__(self, config_path=ROOT / "configs" / "experiment.json", events=None):
        self.cfg = json.loads(Path(config_path).read_text())
        if events:
            self.cfg["events_per_patient"] = int(events)
        self.P = Populations()
        self._base = {}

    def base(self, seed, preset, n_patients, neutropenia=False):
        key = (seed, preset, n_patients, neutropenia, self.cfg["events_per_patient"])
        if key not in self._base:
            e = self.cfg["events_per_patient"]
            self._base[key] = [make_base_group(self.P, self.cfg, seed, preset, g, n_patients, e, neutropenia) for g in (0, 1)]
        return self._base[key]

    def cohort(self, s):
        """s: one settings dict from the manifest. Returns cells, group index, planted mask, background mask, labels."""
        P, n = self.P, s["patients_per_group"]
        nt = s.get("neutropenia", self.cfg["neutropenia"]["enabled"])
        (xn, pn, _, bn), (xs, ps, infos, bs) = self.base(s["seed"], s["noise"], n, nt)
        signal = dict(type=s["signal_type"], affected_fraction=s["affected_fraction"], shift_size=s["shift_size"])
        xs2, aff = apply_signal(P, self.cfg, xs, ps, infos, bs, signal, s["seed"], 1)
        cells = np.concatenate([finalize(xn), finalize(xs2)])
        pop, bg = np.concatenate([pn, ps]), np.concatenate([bn, bs])
        planted = np.concatenate([np.zeros_like(aff), aff])
        labels = pd.DataFrame(dict(
            patient_id=[f"N{i + 1:04d}" for i in range(n)] + [f"S{i + 1:04d}" for i in range(n)],
            label=[0] * n + [1] * n, group=["normal"] * n + ["signal"] * n,
            n_neutrophils=(pop == P.neut).sum(1), n_planted=planted.sum(1), n_background=bg.sum(1)))
        for k in ("cohort_id", "signal_type", "affected_fraction", "shift_size", "noise", "seed"):
            labels[k] = s.get(k)
        labels["events"] = cells.shape[1]
        labels["neutropenia"] = nt
        return cells, pop, planted, bg, labels


def build_manifest(cfg):
    """All cohort settings per plan. Per (noise preset, seed): one 0% leakage-check cohort plus the grid."""
    out = {}
    for plan, p in cfg["plans"].items():
        rows = []
        for noise in cfg["grid"]["noise"]:
            for seed in p["seeds"]:
                rows.append(dict(signal_type="none", affected_fraction=0.0, shift_size=0.0, noise=noise, seed=seed))
                for kind, g in cfg["grid"]["types"].items():
                    for f in g["fractions"]:
                        for d in g["sizes"]:
                            rows.append(dict(signal_type=kind, affected_fraction=f, shift_size=d, noise=noise, seed=seed))
        for i, r in enumerate(rows):
            r.update(cohort_id=f"{plan}_{i:04d}", patients_per_group=p["patients_per_group"])
        out[plan] = rows
    return out
