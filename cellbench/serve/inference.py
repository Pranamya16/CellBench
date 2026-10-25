"""Live scoring with the FINAL CNN models (trained on Kaggle, saved in data/final/cnn).

One synthetic patient in, two calibrated detector scores out (full cell maps, and the auto-gated report summary), plus the
"where the model looked" overlay. Everything the training job did to the inputs is repeated here exactly:
    cells view : log1p(event count) / 5 on the three 32x32 count maps, 3 networks averaged, isotonic calibration
    report view: auto-gated report summary (6 groups x 5 statistics), median-impute, standardise, clip, 3 networks, calibration
"""
import json
from pathlib import Path

import numpy as np

import features as F
import gating
import generator as G
import models as M

ROOT = G.ROOT
FINAL = ROOT / "data" / "final" / "cnn"
SERVE_SEED = 701      # patients made here use a seed no training, tuning or sweep job ever used
NOTE = "Detector score on a synthetic patient: calibrated on a 50/50 mix of synthetic normal and signal patients. Not a chance of disease."


class Scorer:
    def __init__(self, final_dir=FINAL):
        import torch
        torch.set_num_threads(2)
        self.torch = torch
        self.gen = G.Generator()
        self.P = self.gen.P
        self.gate = gating.get_gate(self.gen)
        meta = json.loads((Path(final_dir) / "calibration.json").read_text())
        self.cal = {v: (np.array(c["x"]), np.array(c["y"])) for v, c in meta["calibration"].items()}
        p = meta["prep"]["report"]
        self.prep = {k: np.array(v) for k, v in p.items()}
        _, nn, _, CNN = M._torch_models()
        self.nets = {}
        for view, make in (("cells", CNN), ("report", M.grid_cnn_class())):
            states = torch.load(Path(final_dir) / f"weights_cnn_{view}.pt", map_location="cpu")
            nets = []
            for s in states:
                net = make()
                net.load_state_dict(s)
                nets.append(net.eval())
            self.nets[view] = nets
        self.n_groups = len(self.P.names)
        ref = Path(final_dir) / "normal_reference.json"
        self.ref = json.loads(ref.read_text()) if ref.exists() else None

    # ---- inputs -------------------------------------------------------------------------------------------------
    def _inputs(self, cells):
        """cells: (events, 4) uint8 -> the two network inputs, exactly as in training."""
        c = cells[None]
        E = self.gen.cfg["events_per_patient"]   # the size the networks were trained on; shares are scaled to it for any file size
        maps = (np.log1p(F.histogram_features(c) * E).astype(np.float32) / 5.0).reshape(1, 3, 32, 32)
        rep = F.report_features(c, gating.assign(self.gate, c), self.n_groups)
        pr = self.prep
        z = np.clip((np.where(np.isnan(rep), pr["med"], rep) - pr["mu"]) / pr["sd"], -10, 10).astype(np.float32)
        return {"cells": maps, "report": z.reshape(1, 1, self.n_groups, 5)}

    def _calibrate(self, view, raw):
        x, y = self.cal[view]
        return float(np.interp(raw, x, y))

    def score(self, cells):
        """Returns calibrated scores for both views (0 to 1)."""
        torch = self.torch
        X, out = self._inputs(cells), {}
        with torch.no_grad():
            for view in ("cells", "report"):
                xt = torch.tensor(X[view])
                raw = float(np.mean([torch.sigmoid(n(xt)).item() for n in self.nets[view]]))
                out[view] = dict(raw=raw, score=self._calibrate(view, raw))
        return out

    def heatmap(self, cells):
        """Gradient x input of the cell-map ensemble on the SSC-SFL map (32x32, scaled 0-1): where the model looked."""
        torch = self.torch
        x = torch.tensor(self._inputs(cells)["cells"], requires_grad=True)
        sum(n(x).sum() for n in self.nets["cells"]).backward()
        s = (x.grad * x).detach().abs().numpy()[0, 0]
        return (s / (s.max() + 1e-12)).round(3)

    # ---- patients -----------------------------------------------------------------------------------------------
    def make_patient(self, kind, frac, drop, noise, seed, signal=True):
        """One fresh synthetic patient. kind: subgroup | spread | shift. frac and drop are fractions (0.1 = 10%).
        Returns cells (events, 4) uint8, per-cell group, planted flag, and the truth."""
        cfg, E = self.gen.cfg, self.gen.cfg["events_per_patient"]
        group = 1 if signal else 0
        x, pop, info, bg = G.make_base_patient(self.P, cfg, SERVE_SEED, noise, group, int(seed), E)
        planted = np.zeros(E, dtype=bool)
        if signal:
            sig = dict(type=kind, affected_fraction=frac, shift_size=drop)
            x2, aff = G.apply_signal(self.P, cfg, x[None], pop[None], [info], bg[None], sig, SERVE_SEED, 1)
            x, planted = x2[0], aff[0]
        return G.finalize(x), pop, planted

    def lab_values(self, cells, seed):
        """What an analyser would print for this synthetic patient: a differential from the auto-gated groups and the neutrophil
        averages (NE-SSC, NE-SFL, NE-FSC). WBC is a made-up total (4.5 to 9.5) so that counts can be shown. All on the 0-255 scale."""
        pred = gating.assign(self.gate, cells[None])[0]
        names = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "ig"]
        idx = [self.P.names.index(n) for n in names]
        counts = np.array([(pred == i).sum() for i in idx], dtype=float)
        wbc = float(np.round(np.random.default_rng(int(seed) + 17).uniform(4.5, 9.5), 2))
        pct = 100 * counts / counts.sum()
        ne = cells[pred == self.P.neut].astype(float)
        return dict(wbc=wbc, groups=names, pct=[round(float(x), 1) for x in pct], absolute=[round(float(x) * wbc / 100, 2) for x in pct],
                    ne_ssc=round(float(ne[:, 0].mean()), 1), ne_sfl=round(float(ne[:, 1].mean()), 1), ne_fsc=round(float(ne[:, 2].mean()), 1),
                    ne_events=int(len(ne)), events=int(len(cells)))

    def evidence(self, cells):
        """Readable clues about the neutrophils in this file, each placed against the range seen in normal synthetic patients (5th to 95th percentile)."""
        if self.ref is None:
            return None
        pred = gating.assign(self.gate, cells[None])[0]
        ssc = cells[pred == self.P.neut].astype(float)[:, 0]
        if len(ssc) < 100:
            return None
        med = np.median(ssc)
        up = ssc[ssc >= med]
        tail = float((ssc < med - 1.645 * np.sqrt(np.mean((up - med) ** 2))).mean())
        out = {}
        for key, val in (("ssc", float(ssc.mean())), ("tail", tail), ("sd", float(ssc.std()))):
            r = self.ref["low" if key == "tail" else key]
            pos = "below" if val < r["q05"] else "above" if val > r["q95"] else "within"
            s = 100 if key == "tail" else 1
            hit = pos == ("below" if key == "ssc" else "above")        # the direction a planted less-granular group would push it
            out[key] = dict(hit=hit, value=round(val * s, 1 if key != "tail" else 0), lo=round(r["q05"] * s, 1 if key != "tail" else 0), hi=round(r["q95"] * s, 1 if key != "tail" else 0), pos=pos)
        return out

    def package(self, kind, frac, drop, noise, seed, signal=True, n_show=800):
        cells, pop, planted = self.make_patient(kind, frac, drop, noise, seed, signal)
        sc, hm = self.score(cells), self.heatmap(cells)
        lab = self.lab_values(cells, seed)
        rng = np.random.default_rng(int(seed))
        show = rng.choice(len(cells), n_show, replace=False)
        neut = int((pop == self.P.neut).sum())
        patient = dict(
            signal=bool(signal), kind=kind if signal else "none", frac=float(frac) if signal else 0.0, drop=float(drop) if signal else 0.0,
            noise=noise, seed=int(seed), neutrophils=neut, planted=int(planted.sum()),
            score_cells=round(sc["cells"]["score"], 3), score_report=round(sc["report"]["score"], 3),
            cells=[[int(cells[i, 0]), int(cells[i, 1]), int(planted[i]), int(pop[i] == self.P.neut)] for i in show],
            heat=hm.tolist(), lab=lab, note=NOTE, evidence=self.evidence(cells))
        return patient

    def package_cells(self, cells, name, warnings=()):
        """Score an uploaded cell file. Same outputs as package(); the truth is not known."""
        import zlib
        sc, hm = self.score(cells), self.heatmap(cells)
        lab = self.lab_values(cells, zlib.crc32(name.encode()) % 100000)
        pred = gating.assign(self.gate, cells[None])[0]
        rng = np.random.default_rng(0)
        show = rng.choice(len(cells), min(800, len(cells)), replace=False)
        warnings = list(warnings)
        share = float((pred == self.P.neut).mean())
        if share < 0.30 or share > 0.80:
            warnings.insert(0, f"Out of range: only {round(share * 100)}% of cells look like neutrophils, against about 50% in the simulated patients. The scores are not meaningful.")
        return dict(
            signal=None, upload=True, name=name, kind="none", frac=0.0, drop=0.0, noise="uploaded", seed=0,
            neutrophils=int((pred == self.P.neut).sum()), planted=0,
            score_cells=round(sc["cells"]["score"], 3), score_report=round(sc["report"]["score"], 3),
            cells=[[int(cells[i, 0]), int(cells[i, 1]), 0, int(pred[i] == self.P.neut)] for i in show],
            heat=hm.tolist(), lab=lab, warnings=warnings, out_of_range=bool(warnings and warnings[0].startswith('Out of range')), note=NOTE, evidence=self.evidence(cells))
