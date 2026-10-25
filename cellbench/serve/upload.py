"""Read an uploaded per-cell CSV (one patient) into the (cells, 4) uint8 array the models expect.

Accepted: a header row naming SSC, SFL, FSC and FSCW (any order, any case, extra columns such as 'event' are ignored), or no header and exactly
four columns in that order. Separator: comma, semicolon or tab. Values must be numbers from 0 to 255 (the analyser's 8-bit scale).
"""
import io
import re

import numpy as np
import pandas as pd

COLS = ["SSC", "SFL", "FSC", "FSCW"]
MIN_ROWS, WARN_ROWS, MAX_ROWS = 500, 1000, 200_000


def safe_name(name):
    """File names go into the explanation prompt, so keep only harmless characters."""
    return (re.sub(r"[^A-Za-z0-9._-]", "_", str(name))[:40] or "file.csv")


def parse_cells(text):
    """Returns (cells uint8 (N, 4), warnings list). Raises ValueError with a message meant for the user."""
    text = text.replace("\r", "").strip("﻿\n ")
    if not text.strip():
        raise ValueError("The file is empty.")
    first = text.split("\n", 1)[0]
    sep = max([",", ";", "\t"], key=first.count)
    cells0 = [c.strip().strip('"') for c in first.split(sep)]
    header = any(re.search(r"[A-Za-z]", c) for c in cells0)
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, header=0 if header else None, nrows=MAX_ROWS + 1)
    except Exception:
        raise ValueError("The file could not be read as a table. Use comma-separated values, one cell per row.")
    if header:
        low = {str(c).strip().lower(): c for c in df.columns}
        missing = [c for c in COLS if c.lower() not in low]
        if missing:
            raise ValueError(f"The header must contain the columns {', '.join(COLS)}. Missing: {', '.join(missing)}. Found: {', '.join(map(str, df.columns))[:80]}.")
        df = df[[low[c.lower()] for c in COLS]]
    else:
        if df.shape[1] != 4:
            raise ValueError(f"Without a header the file must have exactly 4 columns (SSC, SFL, FSC, FSCW); this one has {df.shape[1]}.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"The file has more than {MAX_ROWS:,} cells. Upload one patient at a time.")
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).any():
        raise ValueError("Every row needs four numbers. Some values are missing or are not numbers.")
    if len(arr) < MIN_ROWS:
        raise ValueError(f"The file has only {len(arr)} cells. At least {MIN_ROWS} are needed (about 2,000 is typical).")
    if arr.min() < -0.5 or arr.max() > 255.5:
        raise ValueError(f"Values must be between 0 and 255 (the analyser's 8-bit scale). Found {arr.min():g} to {arr.max():g}.")
    warns = []
    if not np.allclose(arr, np.rint(arr), atol=1e-6):
        warns.append("Values were rounded to whole numbers.")
    if len(arr) < WARN_ROWS:
        warns.append(f"Only {len(arr)} cells: scores will be noisier than for about 2,000.")
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8), warns
