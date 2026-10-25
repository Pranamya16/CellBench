"""The three data views. Report and rich summaries use the true group of each cell ("oracle gating", generous to
the summaries). The full per-cell view never sees group labels."""
import numpy as np

READINGS = ["SSC", "SFL", "FSC", "FSCW"]
PAIRS = [(0, 1), (0, 2), (2, 1)]  # SSC-SFL, SSC-FSC, FSC-SFL


def report_features(cells, pop, n_groups):
    """Per group: mean of each reading and share of events. Like NE-SSC, NE-SFL, NE-FSC plus the differential."""
    n, e, _ = cells.shape
    X = np.full((n, n_groups * 5), np.nan, dtype=np.float32)
    for g in range(n_groups):
        m = pop == g
        cnt = m.sum(1)
        for r in range(4):
            tot = np.where(m, cells[..., r], 0).sum(1, dtype=np.float64)
            X[:, g * 5 + r] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        X[:, g * 5 + 4] = cnt / e
    return X


def rich_features(cells, pop, n_groups, min_cells=20):
    """Per group: percentiles (5, 25, 50, 75, 95), SD of each reading, and the share of events."""
    n, e, _ = cells.shape
    X = np.full((n, n_groups * 25), np.nan, dtype=np.float32)
    for i in range(n):
        for g in range(n_groups):
            c = cells[i][pop[i] == g].astype(np.float32)
            X[i, g * 25 + 24] = len(c) / e
            if len(c) >= min_cells:
                q = np.percentile(c, [5, 25, 50, 75, 95], axis=0)  # (5, 4)
                X[i, g * 25: g * 25 + 20] = q.T.ravel()
                X[i, g * 25 + 20: g * 25 + 24] = c.std(0)
    return X


def histogram_features(cells, bins=32):
    """Every cell, no labels: 2D count maps (share of events per box) for SSC-SFL, SSC-FSC, FSC-SFL."""
    n, e, _ = cells.shape
    w = 256 // bins
    out = []
    for a, b in PAIRS:
        idx = (cells[..., a].astype(np.int64) // w) * bins + (cells[..., b].astype(np.int64) // w)
        flat = (np.arange(n)[:, None] * bins * bins + idx).ravel()
        out.append(np.bincount(flat, minlength=n * bins * bins).reshape(n, bins * bins) / e)
    return np.concatenate(out, axis=1).astype(np.float32)
