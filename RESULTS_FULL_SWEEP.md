# CellBench: results of the full sweeps

*Written for: teammates who need to review the numbers. All data is **synthetic** (no real patient data). Generated 2026-10-24.*

## Headline: same model, same patients, only the data changes
Two model pairs were trained on both data views: CNN on report summary vs CNN on full cell maps, and MLP on report summary vs MLP on full cell maps. The difference inside each pair is the data, not the model.

| Shared noise | CNN: report averages | CNN: full cell data | MLP: report averages | MLP: full cell data |
|---|---|---|---|---|
| Subgroup, 10% of neutrophils, 10% SSC drop | 0.52 | **0.97** | 0.52 | 0.67 |
| Subgroup, 20% of neutrophils, 10% SSC drop | 0.61 | **1.00** | 0.78 | 0.83 |
| Wider neutrophil cluster (spread), 20% | 0.51 | **0.99** | 0.52 | 0.75 |
| Detection limit at a 20% SSC drop | 10% of neutrophils | **2%** | 10% | 5% |

Plain meaning: full cell data detects **smaller** cell changes, and changes that only widen the cluster, which the averages miss. Say "detects smaller cell changes", not "detects early disease". The richer summary (percentiles, MLP) also recovers part of the signal; it is kept below as a footnote row.

**Known handicap of the MLP on full cell maps in this sweep:** its standardising step divides by the spread of each map box measured on training patients; a box that is almost always empty in training gets a tiny spread, so a single test cell there becomes a huge value. This may have lowered the MLP-on-maps scores. It does not affect the CNNs (they use a fixed log scale) or the report views. The final models use a spread floor and clipping.

## 1. What is included
Only the **full sweeps**: the 180 cohorts of the plan (5 seeds, 300 normal + 300 signal patients each, 2,000 events per patient, both noise presets), run once per model. **Not included:** all dry runs, the tuning checks on seeds 101 to 103, the labelled "oracle" summaries, and the boosting models (they were only run in dry runs).

| Run | Model | Data view | Sweep cohorts | Extra no-signal cohorts | Where it ran |
|---|---|---|---|---|---|
| 1 | MLP | Report summary (auto-gated) | 180 | 30 | Kaggle CPU, 8 kernels |
| 1 | MLP | Rich summary (auto-gated) | 180 | 30 | Kaggle CPU, 8 kernels |
| 1 | MLP | Full per-cell maps | 180 | 30 | Kaggle CPU, 8 kernels |
| 2 | CNN | Full per-cell maps | 180 | 30 | Kaggle T4 GPU, 8 kernels |
| 3 | CNN | Report summary (auto-gated, 6x5 grid) | 180 | 30 | Kaggle T4 GPU, 8 kernels |

Each sweep has 180 cohorts: 170 with a planted signal and 10 with none. Every case below is the mean of **5 seeds**. Typical seed-to-seed spread of AUROC is 0.02 (largest 0.06), so differences under about 0.05 are not reliable.

## 2. How to read the tables
- **Score = AUROC:** 0.5 is a coin flip, 1.0 is perfect. Accuracy tables show the plain % of patients classified correctly.
- **Columns** "a%/b%" mean: a% of a signal patient's neutrophils were changed, with an SSC drop of b%. "Spread x%/10%" means the neutrophil cluster is widened to the same variance as a subgroup of x% with a 10% drop. "Shift all/10%" means every neutrophil drops 10%.
- **Report and rich summaries** are built from auto-gated groups (no free labels). The full per-cell models see all cells without any group labels.

## 3. No-signal check (must be about 0.5)
Over the 10 plan cohorts with no signal plus 30 extra no-signal cohorts on spare seeds:

| Data view | Model | Cohorts | Mean AUROC | SD | Range |
|---|---|---|---|---|---|
| Report summary | MLP | 40 | 0.503 | 0.023 | 0.45 to 0.55 |
| Report summary | CNN | 40 | 0.506 | 0.026 | 0.43 to 0.55 |
| Rich summary | MLP | 40 | 0.503 | 0.029 | 0.43 to 0.57 |
| Full per-cell maps | MLP | 40 | 0.496 | 0.024 | 0.45 to 0.55 |
| Full per-cell maps | CNN | 40 | 0.504 | 0.023 | 0.45 to 0.57 |

No model is meaningfully above 0.5, so nothing leaks the answer.

## 4. AUROC, shared noise
### Subgroup signal
| Data view | Model | 2%/5% | 5%/5% | 10%/5% | 20%/5% | 2%/10% | 5%/10% | 10%/10% | 20%/10% | 2%/20% | 5%/20% | 10%/20% | 20%/20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Report summary | MLP | 0.48 | 0.48 | 0.49 | 0.53 | 0.49 | 0.49 | 0.52 | 0.78 | 0.53 | 0.76 | 0.91 | 0.99 |
| Report summary | CNN | 0.49 | 0.49 | 0.50 | 0.52 | 0.49 | 0.50 | 0.52 | 0.61 | 0.52 | 0.63 | 0.82 | 0.98 |
| Rich summary | MLP | 0.51 | 0.53 | 0.57 | 0.70 | 0.54 | 0.67 | 0.85 | 0.96 | 0.82 | 0.98 | 1.00 | 1.00 |
| Full per-cell maps | MLP | 0.49 | 0.50 | 0.51 | 0.56 | 0.50 | 0.55 | 0.67 | 0.83 | 0.73 | 0.95 | 1.00 | 1.00 |
| Full per-cell maps | CNN | 0.50 | 0.50 | 0.53 | 0.70 | 0.54 | 0.81 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Spread and shift
| Data view | Model | Spread 2%/10% | Spread 5%/10% | Spread 10%/10% | Spread 20%/10% | Shift all/10% |
|---|---|---|---|---|---|---|
| Report summary | MLP | 0.48 | 0.49 | 0.49 | 0.52 | 1.00 |
| Report summary | CNN | 0.49 | 0.49 | 0.50 | 0.51 | 1.00 |
| Rich summary | MLP | 0.54 | 0.62 | 0.80 | 0.93 | 1.00 |
| Full per-cell maps | MLP | 0.49 | 0.51 | 0.59 | 0.75 | 1.00 |
| Full per-cell maps | CNN | 0.50 | 0.60 | 0.91 | 0.99 | 1.00 |

## 5. AUROC, independent noise
### Subgroup signal
| Data view | Model | 2%/5% | 5%/5% | 10%/5% | 20%/5% | 2%/10% | 5%/10% | 10%/10% | 20%/10% | 2%/20% | 5%/20% | 10%/20% | 20%/20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Report summary | MLP | 0.52 | 0.52 | 0.53 | 0.55 | 0.52 | 0.53 | 0.55 | 0.62 | 0.57 | 0.68 | 0.79 | 0.90 |
| Report summary | CNN | 0.52 | 0.52 | 0.53 | 0.54 | 0.52 | 0.53 | 0.54 | 0.58 | 0.53 | 0.59 | 0.69 | 0.82 |
| Rich summary | MLP | 0.51 | 0.52 | 0.53 | 0.57 | 0.52 | 0.57 | 0.72 | 0.88 | 0.71 | 0.95 | 0.99 | 1.00 |
| Full per-cell maps | MLP | 0.52 | 0.52 | 0.53 | 0.56 | 0.53 | 0.56 | 0.62 | 0.74 | 0.71 | 0.96 | 0.99 | 1.00 |
| Full per-cell maps | CNN | 0.53 | 0.53 | 0.54 | 0.61 | 0.55 | 0.78 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Spread and shift
| Data view | Model | Spread 2%/10% | Spread 5%/10% | Spread 10%/10% | Spread 20%/10% | Shift all/10% |
|---|---|---|---|---|---|---|
| Report summary | MLP | 0.52 | 0.52 | 0.53 | 0.55 | 0.95 |
| Report summary | CNN | 0.52 | 0.51 | 0.52 | 0.52 | 0.94 |
| Rich summary | MLP | 0.52 | 0.55 | 0.68 | 0.87 | 0.95 |
| Full per-cell maps | MLP | 0.52 | 0.54 | 0.58 | 0.71 | 0.90 |
| Full per-cell maps | CNN | 0.52 | 0.58 | 0.89 | 0.99 | 0.96 |

## 6. Plain accuracy (% of patients classified correctly), shared noise
### Subgroup signal
| Data view | Model | 2%/5% | 5%/5% | 10%/5% | 20%/5% | 2%/10% | 5%/10% | 10%/10% | 20%/10% | 2%/20% | 5%/20% | 10%/20% | 20%/20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Report summary | MLP | 49% | 49% | 49% | 51% | 49% | 51% | 52% | 72% | 53% | 70% | 83% | 95% |
| Report summary | CNN | 49% | 49% | 49% | 50% | 50% | 49% | 50% | 58% | 52% | 60% | 73% | 93% |
| Rich summary | MLP | 51% | 52% | 54% | 64% | 52% | 62% | 78% | 90% | 74% | 93% | 97% | 99% |
| Full per-cell maps | MLP | 50% | 51% | 51% | 55% | 50% | 53% | 63% | 76% | 67% | 88% | 97% | 100% |
| Full per-cell maps | CNN | 50% | 50% | 52% | 64% | 52% | 73% | 91% | 98% | 98% | 100% | 100% | 100% |

### Spread and shift
| Data view | Model | Spread 2%/10% | Spread 5%/10% | Spread 10%/10% | Spread 20%/10% | Shift all/10% |
|---|---|---|---|---|---|---|
| Report summary | MLP | 49% | 49% | 50% | 51% | 99% |
| Report summary | CNN | 49% | 50% | 50% | 52% | 99% |
| Rich summary | MLP | 52% | 58% | 73% | 87% | 100% |
| Full per-cell maps | MLP | 50% | 52% | 57% | 69% | 98% |
| Full per-cell maps | CNN | 50% | 56% | 83% | 95% | 100% |

## 7. Plain accuracy, independent noise
### Subgroup signal
| Data view | Model | 2%/5% | 5%/5% | 10%/5% | 20%/5% | 2%/10% | 5%/10% | 10%/10% | 20%/10% | 2%/20% | 5%/20% | 10%/20% | 20%/20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Report summary | MLP | 51% | 52% | 51% | 51% | 51% | 51% | 54% | 58% | 54% | 63% | 72% | 82% |
| Report summary | CNN | 52% | 52% | 52% | 52% | 51% | 53% | 53% | 54% | 52% | 56% | 64% | 74% |
| Rich summary | MLP | 50% | 51% | 52% | 56% | 51% | 54% | 66% | 82% | 66% | 88% | 96% | 98% |
| Full per-cell maps | MLP | 51% | 51% | 52% | 55% | 51% | 54% | 59% | 67% | 66% | 90% | 96% | 98% |
| Full per-cell maps | CNN | 52% | 51% | 52% | 58% | 53% | 68% | 91% | 97% | 97% | 100% | 100% | 100% |

### Spread and shift
| Data view | Model | Spread 2%/10% | Spread 5%/10% | Spread 10%/10% | Spread 20%/10% | Shift all/10% |
|---|---|---|---|---|---|---|
| Report summary | MLP | 52% | 50% | 50% | 54% | 87% |
| Report summary | CNN | 51% | 51% | 51% | 52% | 87% |
| Rich summary | MLP | 51% | 52% | 64% | 80% | 87% |
| Full per-cell maps | MLP | 52% | 53% | 55% | 66% | 83% |
| Full per-cell maps | CNN | 51% | 54% | 80% | 96% | 88% |

## 8. Detection limits
Defined before the sweep: the smallest tested % of neutrophils changed whose mean AUROC is at least 0.80 and stays at or above 0.80 for every larger tested fraction. "Not reached" means no tested fraction up to 20% qualifies.

### Shared noise
| Signal | SSC drop | Report summary, MLP | Report summary, CNN | Rich summary, MLP | Full per-cell maps, MLP | Full per-cell maps, CNN |
|---|---|---|---|---|---|---|
| spread | 10% | not reached | not reached | 10% | not reached | 10% |
| subgroup | 5% | not reached | not reached | not reached | not reached | not reached |
| subgroup | 10% | not reached | not reached | 10% | 20% | 5% |
| subgroup | 20% | 10% | 10% | 2% | 5% | 2% |

### Independent noise
| Signal | SSC drop | Report summary, MLP | Report summary, CNN | Rich summary, MLP | Full per-cell maps, MLP | Full per-cell maps, CNN |
|---|---|---|---|---|---|---|
| spread | 10% | not reached | not reached | 20% | not reached | 10% |
| subgroup | 5% | not reached | not reached | not reached | not reached | not reached |
| subgroup | 10% | not reached | not reached | 20% | not reached | 10% |
| subgroup | 20% | 20% | 20% | 5% | 5% | 2% |

## 9. Caveats
- **Synthetic data.** Cell positions were measured from four body-fluid files and borrowed for a blood-like mix; the cell mix, noise size, shift sizes and the linked SFL/FSC drop are assumptions.
- **Our fake cells leave the key area of the cell map nearly empty,** which makes the full-cell CNN's near-perfect scores optimistic for real data.
- **A 5% SSC drop is undetectable by every view.**
- **The report summary is a stand-in** built from auto-gated simulated cells, not a real lab report.
- **GPU runs are not bit-for-bit repeatable.** A rerun of the CNNs would differ slightly.
- Full data: `data/results_all_models.csv`; limits: `data/detection_limits_all.csv`.
