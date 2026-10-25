"""Candidate models for the three data views. Every model gets the same patient-level cross-validation.

Views and options (see the plan):
  report_LR, report_GB     report averages
  rich_GB                  richer summaries
  cells_hist_GB  (A)       2D count maps + gradient boosting      (no group labels)
  cells_hist_LR  (B)       2D count maps + logistic regression    (no group labels)
  cells_MIL_GB   (C)       cell-level boosting, patient score = mean of the top 5% of cell scores
  cells_CNN      (D)       small CNN on the count maps            (needs torch)
  cells_DeepSets (E)       Deep Sets on raw cells                 (needs torch)
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F


def gb(seed=0):
    # identical settings for every view, no per-view tuning
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=seed)


class Data:
    """One cohort prepared for modelling. `pop` (true group per cell) is used only by the summary views."""

    def __init__(self, cells, pop, y, n_groups, pop_auto=None):
        self.cells, self.pop, self.y, self.n_groups = cells, pop, np.asarray(y), n_groups
        self.report = F.report_features(cells, pop, n_groups)      # oracle: true group of every cell (best case)
        self.rich = F.rich_features(cells, pop, n_groups)
        if pop_auto is not None:                                   # auto-gated: groups assigned from the 4 readings only
            self.report_auto = F.report_features(cells, pop_auto, n_groups)
            self.rich_auto = F.rich_features(cells, pop_auto, n_groups)
        self.hist = F.histogram_features(cells)


def _tab(model_fn, X):
    def run(d, tr, te, seed):
        m = model_fn(seed)
        m.fit(X(d)[tr], d.y[tr])
        return m.predict_proba(X(d)[te])[:, 1]
    return run


def _lr(seed):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


def _hist_lr(d, tr, te, seed):
    X = np.log1p(d.hist * d.cells.shape[1])
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
    m.fit(X[tr], d.y[tr])
    return m.predict_proba(X[te])[:, 1]


def _mil(d, tr, te, seed, per_patient=300, top=0.05):
    rng = np.random.default_rng(seed)
    n_cells = d.cells.shape[1]
    idx = np.stack([rng.choice(n_cells, per_patient, replace=False) for _ in tr])
    Xc = d.cells[tr][np.arange(len(tr))[:, None], idx].reshape(-1, 4)
    yc = np.repeat(d.y[tr], per_patient)
    m = gb(seed).fit(Xc, yc)
    p = m.predict_proba(d.cells[te].reshape(-1, 4))[:, 1].reshape(len(te), n_cells)
    k = max(1, int(top * n_cells))
    return np.sort(p, axis=1)[:, -k:].mean(1)


def _torch_models():
    import torch
    import torch.nn as nn

    class DeepSets(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            h = self.phi(x)
            return self.rho(torch.cat([h.mean(1), h.max(1).values], 1)).squeeze(-1)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                   nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            h = self.f(x)
            return self.head(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)).squeeze(-1)

    return torch, nn, DeepSets, CNN


def _fit_torch(net, make_batch, y_tr, n_tr, predict_inputs, epochs, seed, lr=2e-3, bs=32):
    import torch
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(n_tr), max(1, n_tr // bs)):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(make_batch(b)), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(predict_inputs(net)).numpy()


def _deepsets(d, tr, te, seed, n_cells=1000, epochs=30):
    torch, nn, DeepSets, _ = _torch_models()
    rng = np.random.default_rng(seed)
    X = torch.tensor(d.cells.astype(np.float32) / 255.0)
    E = X.shape[1]
    Xtr = X[tr]

    def batch(b):
        idx = torch.tensor(np.stack([rng.choice(E, n_cells, replace=False) for _ in b]))
        return Xtr[torch.tensor(b)[:, None], idx]

    def predict(net):
        outs = []
        for r in range(3):  # average over three random cell subsets
            g = np.random.default_rng(1000 + r)
            idx = torch.tensor(np.stack([g.choice(E, n_cells, replace=False) for _ in te]))
            outs.append(net(X[torch.tensor(te)[:, None], idx]))
        return torch.stack(outs).mean(0)

    return _fit_torch(DeepSets(), batch, d.y[tr], len(tr), predict, epochs, seed)


# FROZEN before the main sweep (the earlier dry-run CNN used a fixed 30 epochs and no early stopping). Never tuned on sweep seeds.
CNN_CONFIG = dict(channels=(16, 32, 32), lr=2e-3, weight_decay=1e-4, batch=64, max_epochs=60, patience=8, starts=3,
                  val_fraction=0.2, input="log1p(event count) / 5 on three 32x32 maps (SSC-SFL, SSC-FSC, FSC-SFL)")


def _train_cnn(make_net, H, y, tr, te, seed):
    """Shared training recipe (CNN_CONFIG): 3 random starts averaged, early stopping on an inner validation split of the
    training patients, GPU when available. H: (patients, channels, rows, cols) on the right device."""
    torch, nn, _, _ = _torch_models()
    c = CNN_CONFIG
    dev = H.device
    outs = []
    for r in range(c["starts"]):
        rng = np.random.default_rng(seed * 31 + r)
        perm = rng.permutation(len(tr))
        nv = max(8, int(len(tr) * c["val_fraction"]))
        va, fi = tr[perm[:nv]], tr[perm[nv:]]
        torch.manual_seed(seed * 31 + r)
        net = make_net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        best, bad, state = 1e9, 0, None
        for _ in range(c["max_epochs"]):
            net.train()
            order = fi[rng.permutation(len(fi))]
            for b in np.array_split(order, max(1, len(order) // c["batch"])):
                loss = nn.functional.binary_cross_entropy_with_logits(net(H[b]), y[b])
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = nn.functional.binary_cross_entropy_with_logits(net(H[va]), y[va]).item()
            if v < best - 1e-4:
                best, bad, state = v, 0, {k: s.clone() for k, s in net.state_dict().items()}
            else:
                bad += 1
                if bad >= c["patience"]:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            outs.append(torch.sigmoid(net(H[te])).cpu().numpy())
    return np.mean(outs, axis=0)


def _device():
    import torch
    torch.backends.cudnn.benchmark = False
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cnn(d, tr, te, seed):
    """CNN on the three 32x32 count maps of every cell (no group labels)."""
    torch, nn, _, CNN = _torch_models()
    dev = _device()
    H = torch.tensor(np.log1p(d.hist * d.cells.shape[1]).reshape(len(d.y), 3, 32, 32).astype(np.float32) / 5.0, device=dev)
    return _train_cnn(CNN, H, torch.tensor(d.y, dtype=torch.float32, device=dev), tr, te, seed)


def grid_cnn_class():
    """The CNN used on the report summary laid out as a (groups x 5 statistics) grid."""
    torch, nn, _, _ = _torch_models()

    class GridCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            h = self.f(x)
            return self.head(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)).squeeze(-1)

    return GridCNN


def _cnn_report(d, tr, te, seed):
    """CNN on the auto-gated report summary. The ~30 numbers are laid out as a small grid: one row per cell group, one
    column per statistic (mean SSC, SFL, FSC, FSCW, and the share of events), so neighbouring cells of the grid are related.
    Same standardisation as the MLP (median-impute, standardise on training patients only)."""
    torch, nn, _, _ = _torch_models()
    dev = _device()
    n_groups = d.n_groups
    Z = _prep(d.report_auto, tr, te).reshape(len(d.y), 1, n_groups, 5)
    return _train_cnn(grid_cnn_class(), torch.tensor(Z, device=dev), torch.tensor(d.y, dtype=torch.float32, device=dev), tr, te, seed)


def _prep(X, tr, te, log_scale=None):
    """Same recipe for every view: log1p on counts (histograms only), median-impute, standardise (train stats only)."""
    X = np.asarray(X, dtype=np.float64)
    if log_scale:
        X = np.log1p(X * log_scale)
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


# FROZEN before the main sweep. Identical for every data view; only the input width differs. Never tuned on any seed.
MLP_CONFIG = dict(hidden=(64, 32), dropout=0.3, lr=1e-3, weight_decay=1e-3, max_epochs=200, patience=20, starts=3,
                  val_fraction=0.2, preprocessing="log1p on map counts (cells only), median-impute, standardise on training patients")


def _mlp(get_X, log_scale=None):
    """The SAME small MLP for every view (settings in MLP_CONFIG); only the input width differs."""
    c = MLP_CONFIG
    starts, max_epochs, patience = c["starts"], c["max_epochs"], c["patience"]

    def run(d, tr, te, seed):
        import torch
        import torch.nn as nn
        torch.set_num_threads(1)
        Z = torch.tensor(_prep(get_X(d), tr, te, log_scale(d) if log_scale else None))
        y = torch.tensor(d.y, dtype=torch.float32)
        outs = []
        for r in range(starts):
            g = torch.Generator().manual_seed(seed * 31 + r)
            perm = torch.randperm(len(tr), generator=g).numpy()
            nv = max(8, int(len(tr) * c["val_fraction"]))
            va, fi = tr[perm[:nv]], tr[perm[nv:]]
            torch.manual_seed(seed * 31 + r)
            h1, h2 = c["hidden"]
            net = nn.Sequential(nn.Linear(Z.shape[1], h1), nn.ReLU(), nn.Dropout(c["dropout"]), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))
            opt = torch.optim.Adam(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
            best, bad, state = 1e9, 0, None
            for _ in range(max_epochs):
                net.train()
                loss = nn.functional.binary_cross_entropy_with_logits(net(Z[fi]).squeeze(-1), y[fi])
                opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    v = nn.functional.binary_cross_entropy_with_logits(net(Z[va]).squeeze(-1), y[va]).item()
                if v < best - 1e-4:
                    best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
            net.load_state_dict(state)
            net.eval()
            with torch.no_grad():
                outs.append(torch.sigmoid(net(Z[te]).squeeze(-1)).numpy())
        return np.mean(outs, axis=0)
    return run


OPTIONS = {
    "report_LR": _tab(_lr, lambda d: d.report),
    "report_GB": _tab(gb, lambda d: d.report),
    "rich_GB": _tab(gb, lambda d: d.rich),
    "cells_hist_GB": _tab(gb, lambda d: d.hist),
    "cells_hist_LR": _hist_lr,
    "cells_MIL_GB": _mil,
    "cells_CNN": _cnn,
    "cnn_report_auto": _cnn_report,
    "cells_DeepSets": _deepsets,
    "mlp_report": _mlp(lambda d: d.report),
    "mlp_rich": _mlp(lambda d: d.rich),
    "mlp_cells": _mlp(lambda d: d.hist, log_scale=lambda d: d.cells.shape[1]),
    # auto-gated summaries (no free labels)
    "report_LR_auto": _tab(_lr, lambda d: d.report_auto),
    "report_GB_auto": _tab(gb, lambda d: d.report_auto),
    "rich_GB_auto": _tab(gb, lambda d: d.rich_auto),
    "mlp_report_auto": _mlp(lambda d: d.report_auto),
    "mlp_rich_auto": _mlp(lambda d: d.rich_auto),
}
VIEW = {"report_LR": "report_oracle", "report_GB": "report_oracle", "mlp_report": "report_oracle", "rich_GB": "rich_oracle", "mlp_rich": "rich_oracle",
        "report_LR_auto": "report_auto", "report_GB_auto": "report_auto", "mlp_report_auto": "report_auto",
        "rich_GB_auto": "rich_auto", "mlp_rich_auto": "rich_auto",
        "cells_hist_GB": "cells", "cells_hist_LR": "cells", "cells_MIL_GB": "cells", "cells_CNN": "cells", "cnn_report_auto": "report_auto",
        "cells_DeepSets": "cells", "mlp_cells": "cells"}


def cv_scores(name, d, folds=5, seed=0):
    """Patient-level stratified K-fold. Returns the out-of-fold score of every patient and its fold number."""
    oof, fold = np.zeros(len(d.y)), np.zeros(len(d.y), dtype=int)
    for k, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(d.y)), d.y)):
        oof[te] = OPTIONS[name](d, tr, te, seed + k)
        fold[te] = k
    return oof, fold


def cv_auroc(name, d, folds=5, seed=0):
    return roc_auc_score(d.y, cv_scores(name, d, folds, seed)[0])
