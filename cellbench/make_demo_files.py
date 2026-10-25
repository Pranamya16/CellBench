"""Write synthetic patient cell files in the upload format (SSC,SFL,FSC,FSCW) for the live demo.

    python cellbench/make_demo_files.py --n 2224 --out demo_uploads

Half are normal patients and half have a planted neutrophil change (all scenario types and sizes, both noise presets). File names are
neutral (patient_0001.csv, ...) so the demo can be blind; the answer key goes in a separate folder. Patients are made exactly as the live
page makes them (cohort seed 701, which no training, tuning or sweep job used). All data is synthetic.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generator as G

SERVE_SEED = 701


def scenarios():
    """Signal scenarios (kind, fraction, drop). 5% drops are included on purpose: the models cannot see them, and the demo should show that."""
    out = [("subgroup", f, d) for f in (0.02, 0.05, 0.10, 0.20) for d in (0.05, 0.10, 0.20)]
    out += [("spread", f, 0.10) for f in (0.02, 0.05, 0.10, 0.20)]
    out += [("shift", 1.0, 0.10)]
    return out


def make(gen, kind, frac, drop, noise, seed, signal):
    cfg, E, P = gen.cfg, gen.cfg["events_per_patient"], gen.P
    x, pop, info, bg = G.make_base_patient(P, cfg, SERVE_SEED, noise, 1 if signal else 0, int(seed), E)
    planted = 0
    if signal:
        sig = dict(type=kind, affected_fraction=frac, shift_size=drop)
        x2, aff = G.apply_signal(P, cfg, x[None], pop[None], [info], bg[None], sig, SERVE_SEED, 1)
        x, planted = x2[0], int(aff[0].sum())
    return G.finalize(x), planted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2224)
    ap.add_argument("--out", default="demo_uploads")
    ap.add_argument("--first_seed", type=int, default=10000)
    a = ap.parse_args()
    out = Path(a.out)
    key_dir = out.parent / (out.name + "_answer_key")
    out.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    gen = G.Generator()
    sc = scenarios()
    rng = np.random.default_rng(2224)
    n_sig = a.n // 2
    plan = [dict(signal=False, kind="none", frac=0.0, drop=0.0) for _ in range(a.n - n_sig)]
    plan += [dict(signal=True, kind=sc[i % len(sc)][0], frac=sc[i % len(sc)][1], drop=sc[i % len(sc)][2]) for i in range(n_sig)]
    order = rng.permutation(len(plan))              # shuffled, so the file number gives nothing away
    rows = []
    for i, k in enumerate(order, 1):
        p = plan[k]
        noise = ("shared", "independent")[i % 2]
        seed = a.first_seed + i
        cells, planted = make(gen, p["kind"], p["frac"], p["drop"], noise, seed, p["signal"])
        name = f"patient_{i:04d}.csv"
        with open(out / name, "w", newline="") as f:
            f.write("SSC,SFL,FSC,FSCW\n")
            np.savetxt(f, cells, fmt="%d", delimiter=",")
        rows.append(dict(file=name, truth="planted change" if p["signal"] else "normal", kind=p["kind"], affected_fraction=p["frac"], ssc_drop=p["drop"],
                         noise=noise, seed=seed, planted_cells=planted, cells=len(cells)))
        if i % 200 == 0:
            print(i, "files", flush=True)
    with open(key_dir / "answer_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("done:", len(rows), "files in", out, "| answer key:", key_dir / "answer_key.csv")


if __name__ == "__main__":
    main()
