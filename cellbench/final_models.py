"""Final models for the demo. Runs on Kaggle (GPU). Never trained locally.

One model per data view, trained once on MIXED scenarios with fresh seeds, calibrated, then tested on new patients:
    pair "cnn": CNN on the full cell maps  +  CNN on the auto-gated report summary (6 x 5 grid)
    pair "mlp": MLP on the full cell maps  +  MLP on the auto-gated report summary
Seeds: train 301-310, validation 401-402 (early stopping + calibration), test 501-503. None were used before.
The 5% SSC-drop scenarios are left out of training (undetectable in the sweep) but kept in the test.
Outputs, written in this order so a late crash still leaves the numbers:
    final_metrics.csv, shortcut_check.csv, weights_*.pt, calibration.json, demo_patients.json, demo_heatmaps.json
Environment: CB_PAIR = cnn | mlp, CB_SMOKE = 1 for a tiny run, CB_OUT = output folder.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import features as F
import gating
import generator as G
import models as M

TRAIN_SCEN = ([("subgroup", f, d) for d in (0.10, 0.20) for f in (0.02, 0.05, 0.10, 0.20)]
              + [("spread", f, 0.10) for f in (0.02, 0.05, 0.10, 0.20)] + [("shift", 1.0, 0.10)])
TEST_SCEN = TRAIN_SCEN + [("subgroup", f, 0.05) for f in (0.02, 0.05, 0.10, 0.20)] + [("none", 0.0, 0.0)]
DEMO = [("subgroup", 0.20, 0.20), ("subgroup", 0.10, 0.10), ("subgroup", 0.05, 0.20), ("subgroup", 0.02, 0.20),
        ("spread", 0.20, 0.10), ("shift", 1.0, 0.10)]
MLP_RECIPE = dict(M.MLP_CONFIG, batch=256, max_epochs=100, patience=10)  # mini-batches: the pooled set is much larger
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def build_set(gen, seeds, presets, scenarios, n, tag):
    """Normal and signal patients for every (seed, preset, scenario); distinct people per scenario."""
    P, cfg, E = gen.P, gen.cfg, gen.cfg["events_per_patient"]
    cells, pops, planted, meta = [], [], [], []
    for seed in seeds:
        for preset in presets:
            for k, (kind, f, d) in enumerate(scenarios):
                off = (k + 1) * 10_000
                g0 = [G.make_base_patient(P, cfg, seed, preset, 0, off + i, E) for i in range(n)]
                g1 = [G.make_base_patient(P, cfg, seed, preset, 1, off + i, E) for i in range(n)]
                x1, aff = G.apply_signal(P, cfg, np.stack([a[0] for a in g1]), np.stack([a[1] for a in g1]), [a[2] for a in g1],
                                         np.stack([a[3] for a in g1]), dict(type=kind, affected_fraction=f, shift_size=d), seed * 100 + k, 1)
                cells.append(G.finalize(np.concatenate([np.stack([a[0] for a in g0]), x1])))
                pops.append(np.concatenate([np.stack([a[1] for a in g0]), np.stack([a[1] for a in g1])]))
                planted.append(np.concatenate([np.zeros((n, E), dtype=bool), aff]))
                meta += [dict(seed=seed, noise=preset, signal_type=kind, affected_fraction=f, shift_size=d, label=lab) for lab in [0] * n + [1] * n]
    log(f"{tag}: {sum(len(c) for c in cells)} patients")
    return np.concatenate(cells), np.concatenate(pops), np.concatenate(planted), pd.DataFrame(meta)


def report_auto(gen, gate, cells, chunk=1000):
    out = []
    for i in range(0, len(cells), chunk):  # chunked: the gate's predict is memory-hungry on millions of cells
        c = cells[i:i + chunk]
        out.append(F.report_features(c, gating.assign(gate, c), len(gen.P.names)))
    return np.concatenate(out)


class Prep:
    """Median-impute and standardise, fitted on training patients only. Unlike the sweep, the spread has a floor (0.1)
    and values are clipped to +/-10: a map box that is always empty in training would otherwise turn one test cell
    into a value in the millions (seen in the smoke test)."""

    def fit(self, X):
        med = np.nanmedian(X, 0)
        self.med = np.where(np.isnan(med), 0.0, med)
        Xf = np.where(np.isnan(X), self.med, X)
        self.mu, self.sd = Xf.mean(0), np.maximum(Xf.std(0), 0.1)
        return self

    def __call__(self, X):
        return np.clip((np.where(np.isnan(X), self.med, X) - self.mu) / self.sd, -10, 10).astype(np.float32)

    def to_dict(self):
        return dict(med=self.med.tolist(), mu=self.mu.tolist(), sd=self.sd.tolist())


def views(pair, gen, gate, cells, prep=None):
    """Returns {view: model input} and the fitted preparations (fitted on this set when prep is None)."""
    E = cells.shape[1]
    logmap = np.log1p(F.histogram_features(cells) * E).astype(np.float32)
    rep = report_auto(gen, gate, cells)
    fit = prep is None
    prep = prep or {}
    if fit:
        prep["report"] = Prep().fit(rep)
        if pair == "mlp":
            prep["cells"] = Prep().fit(logmap)
    if pair == "cnn":
        X = {"cells": (logmap / 5.0).reshape(len(cells), 3, 32, 32), "report": prep["report"](rep).reshape(len(cells), 1, len(gen.P.names), 5)}
    else:
        X = {"cells": prep["cells"](logmap), "report": prep["report"](rep)}
    return X, prep


def net_maker(pair, view, in_dim, n_groups):
    torch, nn, _, CNN = M._torch_models()
    if pair == "cnn":
        return CNN if view == "cells" else M.grid_cnn_class()
    c = M.MLP_CONFIG

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            h1, h2 = c["hidden"]
            self.f = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(c["dropout"]), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))

        def forward(self, x):
            return self.f(x).squeeze(-1)

    return MLP


def fit_nets(make_net, Xtr, ytr, Xva, yva, r, seed, dev):
    import torch
    import torch.nn as nn
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xv, yv = torch.tensor(Xva, device=dev), torch.tensor(yva, dtype=torch.float32, device=dev)
    nets = []
    for s in range(r["starts"]):
        torch.manual_seed(seed * 31 + s)
        rng = np.random.default_rng(seed * 31 + s)
        net = make_net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=r["lr"], weight_decay=r["weight_decay"])
        best, bad, state, used = 1e9, 0, None, 0
        for ep in range(r["max_epochs"]):
            net.train()
            order = rng.permutation(len(ytr))
            for b in np.array_split(order, max(1, len(order) // r["batch"])):
                bi = torch.as_tensor(b, device=dev)
                loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[bi]), yt[bi])
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():  # mean validation loss, computed in chunks
                v = sum(nn.functional.binary_cross_entropy_with_logits(net(Xv[i:i + 2048]), yv[i:i + 2048], reduction="sum").item()
                        for i in range(0, len(yv), 2048)) / len(yv)
            used = ep + 1
            if v < best - 1e-4:
                best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= r["patience"]:
                    break
        net.load_state_dict(state)
        nets.append(net.eval())
        log(f"  start {s + 1}: {used} epochs, best validation loss {best:.4f}")
    return nets


def predict(nets, X, dev, chunk=2048):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xb = torch.tensor(X[i:i + chunk], device=dev)
            out.append(np.mean([torch.sigmoid(n(xb)).cpu().numpy() for n in nets], axis=0))
    return np.concatenate(out)


def saliency(nets, x, dev):
    """Gradient x input of the ensemble's logit for one patient: 'where the model looked'."""
    import torch
    xt = torch.tensor(x[None], device=dev, requires_grad=True)
    sum(n(xt).sum() for n in nets).backward()
    return (xt.grad * xt).detach().abs().cpu().numpy()[0]


def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def add_nuisance(cells, rng, frac=0.01):
    """Replace ~1% of each patient's events with events spread over the near-empty low-SSC, low-SFL area."""
    c = cells.copy()
    k = max(1, int(frac * c.shape[1]))
    for i in range(len(c)):
        idx = rng.choice(c.shape[1], k, replace=False)
        c[i, idx] = np.column_stack([rng.integers(105, 141, k), rng.integers(20, 51, k), rng.integers(40, 111, k), rng.integers(16, 27, k)])
    return c


def main(out_dir=None):
    import torch
    pair = os.environ.get("CB_PAIR", "cnn")
    smoke = os.environ.get("CB_SMOKE", "0") == "1"
    out = os.environ.get("CB_OUT", out_dir or "final_out")
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    log(f"pair={pair} smoke={smoke} device={dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")
    gen = G.Generator()
    gate = gating.get_gate(gen)
    presets = gen.cfg["grid"]["noise"]
    if smoke:
        tr_seeds, va_seeds, te_seeds, n_tr, n_va, n_te = [301], [401], [501], 6, 4, 6
        presets, train_scen, test_scen = presets[:1], TRAIN_SCEN[:2] + TRAIN_SCEN[-1:], TEST_SCEN[:2] + TEST_SCEN[-2:]
        recipe = {"cnn": dict(M.CNN_CONFIG, starts=1, max_epochs=2, patience=1), "mlp": dict(MLP_RECIPE, starts=1, max_epochs=2, patience=1)}[pair]
    else:
        tr_seeds, va_seeds, te_seeds, n_tr, n_va, n_te = list(range(301, 311)), [401, 402], [501, 502, 503], 40, 40, 100
        train_scen, test_scen = TRAIN_SCEN, TEST_SCEN
        recipe = {"cnn": M.CNN_CONFIG, "mlp": MLP_RECIPE}[pair]

    ctr, _, _, mtr = build_set(gen, tr_seeds, presets, train_scen, n_tr, "train")
    cva, _, _, mva = build_set(gen, va_seeds, presets, train_scen, n_va, "validation")
    cte, pte, plte, mte = build_set(gen, te_seeds, presets, test_scen, n_te, "test")
    Xtr, prep = views(pair, gen, gate, ctr)
    Xva, _ = views(pair, gen, gate, cva, prep)
    Xte, _ = views(pair, gen, gate, cte, prep)
    log("features ready")

    nets, cal, raw_te = {}, {}, {}
    for view in ("cells", "report"):
        log(f"training {pair} on {view}")
        mk = net_maker(pair, view, Xtr[view].shape[1] if Xtr[view].ndim == 2 else None, len(gen.P.names))
        nets[view] = fit_nets(mk, Xtr[view], mtr.label.to_numpy(), Xva[view], mva.label.to_numpy(), recipe, 7 if view == "cells" else 11, dev)
        cal[view] = IsotonicRegression(out_of_bounds="clip").fit(predict(nets[view], Xva[view], dev), mva.label.to_numpy())
        raw_te[view] = predict(nets[view], Xte[view], dev)

    # 1) metrics first
    rows, y = [], mte.label.to_numpy()
    trained = {(k, round(f, 4), round(d, 4)) for k, f, d in train_scen}
    for view in ("cells", "report"):
        p = cal[view].predict(raw_te[view])
        for (noise, kind, f, d), g in mte.groupby(["noise", "signal_type", "affected_fraction", "shift_size"]):
            i = g.index.to_numpy()
            rows.append(dict(pair=pair, view=view, noise=noise, signal_type=kind, affected_fraction=f, shift_size=d,
                             in_training=(kind, round(f, 4), round(d, 4)) in trained, n=len(i),
                             auroc=roc_auc_score(y[i], raw_te[view][i]), accuracy=float(((p[i] > 0.5) == (y[i] == 1)).mean()),
                             mean_score_normal=float(p[i][y[i] == 0].mean()), mean_score_signal=float(p[i][y[i] == 1].mean())))
        m = mte.apply(lambda r: (r.signal_type, round(r.affected_fraction, 4), round(r.shift_size, 4)) in trained, axis=1).to_numpy()
        rows.append(dict(pair=pair, view=view, noise="all", signal_type="CALIBRATION_ERROR_trained_scenarios", n=int(m.sum()),
                         auroc=roc_auc_score(y[m], raw_te[view][m]), accuracy=float(((p[m] > 0.5) == (y[m] == 1)).mean()), ece=ece(y[m], p[m])))
    pd.DataFrame(rows).to_csv(os.path.join(out, "final_metrics.csv"), index=False)
    log("final_metrics.csv written")

    # shortcut check: extra events in the near-empty area, equally in normal and signal patients
    rng = np.random.default_rng(2026)
    pick = mte[(mte.seed == te_seeds[0]) & mte.signal_type.isin(["none", "subgroup"])].index.to_numpy()
    Xn, _ = views(pair, gen, gate, add_nuisance(cte[pick], rng), prep)
    srows = []
    for view in ("cells", "report"):
        after = predict(nets[view], Xn[view], dev)
        for key, g in mte.loc[pick].groupby(["noise", "signal_type", "affected_fraction", "shift_size"]):
            loc = np.searchsorted(pick, g.index.to_numpy())
            yy = y[g.index.to_numpy()]
            srows.append(dict(pair=pair, view=view, noise=key[0], signal_type=key[1], affected_fraction=key[2], shift_size=key[3],
                              auroc_before=roc_auc_score(yy, raw_te[view][g.index.to_numpy()]), auroc_after=roc_auc_score(yy, after[loc])))
    pd.DataFrame(srows).to_csv(os.path.join(out, "shortcut_check.csv"), index=False)
    log("shortcut_check.csv written")

    # 2) weights, calibration, preparation
    for view in ("cells", "report"):
        torch.save([n.state_dict() for n in nets[view]], os.path.join(out, f"weights_{pair}_{view}.pt"))
    json.dump(dict(pair=pair, recipe={k: (list(v) if isinstance(v, tuple) else v) for k, v in recipe.items()},
                   calibration={v: dict(x=cal[v].X_thresholds_.tolist(), y=cal[v].y_thresholds_.tolist()) for v in cal},
                   prep={k: v.to_dict() for k, v in prep.items()}, seeds=dict(train=tr_seeds, validation=va_seeds, test=te_seeds),
                   note="Detector score = isotonic-calibrated on a 50/50 normal/signal mix of synthetic patients. Not a chance of disease."),
              open(os.path.join(out, "calibration.json"), "w"))
    log("weights and calibration written")

    # 3) demo patients (test seed, shared noise when available)
    noise0 = "shared" if "shared" in presets else presets[0]
    choice = []
    for kind, f, d in DEMO:
        g = mte[(mte.seed == te_seeds[0]) & (mte.noise == noise0) & (mte.signal_type == kind) & np.isclose(mte.affected_fraction, f)
                & np.isclose(mte.shift_size, d) & (mte.label == 1)]
        choice += g.index[:3].tolist()
    normals = mte[(mte.seed == te_seeds[0]) & (mte.noise == noise0) & (mte.label == 0)]
    choice += normals.index[:: max(1, len(normals) // 6)][:6].tolist()
    E = cte.shape[1]
    demo, rng = [], np.random.default_rng(7)
    for j, i in enumerate(choice):
        r = mte.loc[i]
        sub = rng.choice(E, 800, replace=False)
        demo.append(dict(id=f"P{j + 1:02d}", label="signal" if r.label == 1 else "normal", signal_type=r.signal_type if r.label == 1 else "none",
                         affected_fraction=float(r.affected_fraction) if r.label == 1 else 0.0, shift_size=float(r.shift_size) if r.label == 1 else 0.0,
                         noise=r.noise, planted_cells=int(plte[i].sum()), neutrophils=int((pte[i] == gen.P.neut).sum()),
                         score_report=round(float(cal["report"].predict(raw_te["report"][i:i + 1])[0]), 3),
                         score_cells=round(float(cal["cells"].predict(raw_te["cells"][i:i + 1])[0]), 3),
                         ssc_sfl_map=(F.histogram_features(cte[i:i + 1])[0][:1024] * E).round().astype(int).tolist(),
                         cells=[[int(a), int(b), int(c), int(g)] for (a, b), c, g in zip(cte[i][sub][:, :2], plte[i][sub], pte[i][sub])]))
    json.dump(dict(pair=pair, note="Synthetic patients. Scores are detector scores (50/50 mix), not a chance of disease. "
                                   "cells = [SSC, SFL, planted (known because the data is synthetic), true group index].",
                   groups=gen.P.labels, patients=demo), open(os.path.join(out, "demo_patients.json"), "w"))
    log("demo_patients.json written")

    # 4) heatmaps last: 'where the model looked' on the SSC-SFL map
    heat = {}
    for j, i in enumerate(choice):
        x = Xte["cells"][i]
        s = saliency(nets["cells"], x, dev)
        s = s.reshape(3, 32, 32)[0] if s.size == 3 * 1024 else s[:1024].reshape(32, 32) if s.ndim == 1 else s[0]
        heat[f"P{j + 1:02d}"] = (s / (s.max() + 1e-12)).round(3).tolist()
    json.dump(dict(pair=pair, note="Gradient x input on the SSC-SFL map: where the model looked, not why a patient is ill.", heatmaps=heat),
              open(os.path.join(out, "demo_heatmaps.json"), "w"))
    log("demo_heatmaps.json written")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
