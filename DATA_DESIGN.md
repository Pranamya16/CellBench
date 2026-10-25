# CellBench data design (blood-like, MDS hypogranular neutrophil signal)

*Version 2, 2026-10-24. Replaces the body-fluid / SFL-signal version, which is archived in `archive/v1_body_fluid_sfl_signal/`.*
*Plain-English document for the whole team and the biomedical reviewer. Anything marked "our proposal" needs your review.*

## 1. The idea in plain words

A Sysmex analyser measures tens of thousands of cells per blood sample, with four readings per cell. The lab report keeps only a few dozen summary numbers (counts, percentages, averages). We want to know how much useful signal that summary throws away.

We build a **detection ruler**:

1. Generate realistic **synthetic** patients, cell by cell.
2. Plant a small, known change in half of them.
3. Ask models to tell the two halves apart, using (a) report-style averages, (b) richer summaries, (c) every cell.
4. Shrink the change until each version fails. The smallest change it still finds is its **detection limit**.

This is a benchmark, not a diagnostic tool. We do not claim that any real disease produces changes of the sizes we test.

## 2. Why this signal: MDS and hypogranular neutrophils

- In myelodysplastic syndrome (MDS), abnormal neutrophil development (dysgranulopoiesis) produces neutrophils with fewer granules. These scatter less light sideways, so **SSC (granularity) goes down**.
- Published Sysmex XN data also report lower SFL (dye glow) and smaller FSC (size) in MDS patients with dysplasia.
- The analyser reports these only as **averages across all neutrophils** (NE-SSC, NE-SFL, NE-FSC). As MDS markers they perform only moderately (published AUCs roughly 0.63 to 0.76 *[CITATION NEEDED]*).
- **Hypothesis:** the averages dilute the signal because only some neutrophils are abnormal. A model that looks at individual cells can find a small abnormal subgroup that the averages miss.
- A recent preprint on Sysmex single-cell data ([medRxiv 2026](https://www.medrxiv.org/content/10.64898/2026.03.24.26349221v1)) found its strongest clinical associations in the **variance and tails** of cell distributions rather than in means. That is why we plant three kinds of change (section 5).

### Early-detection framing

- The WHO diagnosis of MDS needs dysplasia in **more than 10% of cells** in a myeloid lineage. Precursor states (CHIP, CCUS, ICUS) sit below that line.
- So an **affected fraction** of 2% to 5% models the sub-diagnostic precursor range, 10% the diagnostic line, and 20% established MDS.
- This is **not a measured trajectory** of any patient over time. It is only a way to place our test points against the clinical thresholds.
- The 10% count is mainly a bone-marrow assessment. The analyser sees **peripheral blood**, so the match is loose.

## 3. The synthetic patients: peripheral-blood-like

Each patient has 2,000 events by default (`events_per_patient` in `configs/experiment.json`; raise it for more cells). Shares below are drawn **per patient** and are the same distribution for normal and signal patients, so only neutrophil morphology differs.

| Group | Typical % | Range | Position source | Status |
|---|---|---|---|---|
| Neutrophil | 55 | 40-75 | Ai et al. 2018, XN **body-fluid** WDF (S2+S3 gate), borrowed for blood | measured elsewhere |
| Lymphocyte | 30 | 20-45 | same files (S1+S2 gate), borrowed | measured elsewhere |
| Monocyte | 7 | 2-10 | same files (S2+S4 gate), borrowed | measured elsewhere |
| Eosinophil | 3 | 1-6 | high-SSC end of the neutrophil gate | **ASSUMED** |
| Immature granulocytes (IG) | 0.5 | 0-1 | placed between the neutrophil cluster and monocytes (lower SSC, higher SFL than mature neutrophils) | **ASSUMED / placeholder** |
| Debris / RBC ghosts | 8 of all events | 3-15 | low-SFL region, kept from v1 | measured elsewhere, blood share **not verified** |

- Percentages are shares of nucleated cells, drawn per patient and renormalised to 100% (the typical values add to 95.5% because basophils are left out). Debris is a share of all events.
- **Basophils are omitted.** On the XN they are counted mainly in the WNR channel, not WDF.
- **Reference ranges:** *[CITATION NEEDED]* adult peripheral blood WBC differential. The numbers above are placeholders from the project brief.
- **Provenance.** The positions (centres, spreads, correlations of the four readings) were **measured from XN body-fluid-mode WDF files** and are **borrowed** for blood. They should be re-measured from real blood WDF files when we have them. Each group carries a `provenance` field in `data/population_table.csv`.
- With 2,000 events a patient has about **1,070 neutrophils** (5th to 95th percentile about 860 to 1,270). Planting 2% of them means about 21 cells; 5% about 53; 10% about 107; 20% about 214.
- Readings are whole numbers 0 to 255 (8-bit). Values are clipped to that range. In the blood-like mix about 2.5% of events pile up at SFL = 0 (debris) and nothing reaches 255, unlike the body-fluid files. Cells below the FSC trigger floor of 10 are never recorded.

### Optional confounder: neutropenia (off by default)

`neutropenia` in the config lowers the neutrophil share in **signal patients only**. Real MDS often comes with low neutrophil counts, but turning this on lets a model separate the groups by neutrophil count alone, which leaks the label. It is a deliberate, separate confounder and stays **off** unless a run asks for it.

## 4. Terms

| Term | Meaning |
|---|---|
| SSC (side scatter) | Light scattered sideways: internal complexity and **granularity** |
| SFL (side fluorescence) | How much the dye glows: nucleic-acid content and activity |
| FSC / FSCW | Forward scatter: cell size / pulse width: how long the cell takes to cross the beam |
| Normal | A patient made from the plain recipe, nothing planted |
| Signal | The same recipe with a known fraction of neutrophils made hypogranular |
| Affected fraction | Percentage of a signal patient's neutrophils that are abnormal: 0%, 2%, 5%, 10%, 20%. 0% means nothing planted and is the leakage check |
| Shift size | How far the abnormal neutrophils move, as a percentage drop in SSC (5%, 10%, 20%). SFL and FSC drop alongside at a smaller, linked amount |
| Signal type | What the change does: subgroup, shift or spread |
| Noise | Natural differences between patients that can hide the signal |
| Seed | Starting number of the random generator. Same seed, same patients |
| Cohort | One complete training dataset: 600 patients (300 normal + 300 signal) made with one fixed combination of affected fraction, shift size, type, noise preset and seed |
| Report summary | Averages and counts only, like the routine lab report (e.g. NE-SSC, NE-SFL, NE-FSC) |
| Rich summary | Percentiles, spread and correlations of each cell group |
| Full per-cell data | Every cell, shown to a model as a 2D histogram or as a set of cells |
| AUROC | Score from 0.5 (coin flip) to 1.0 (perfect) for how well a model separates normal from signal |

## 5. The planted signal

Only signal patients are changed, and within them only a set fraction of neutrophils (for `subgroup`). The rest stay normal. *These definitions are our proposal and need your review.*

| Type | What it does | Parameters |
|---|---|---|
| `subgroup` (main case) | A fraction of neutrophils forms a separate low-granularity group, below and slightly left of the normal cluster. Closest to what is seen under the microscope in MDS | affected fraction **and** shift size |
| `shift` | **Every** neutrophil moves down in SSC. The comparison case, where averages should do best. The affected fraction is **ignored and fixed at 100%** | shift size only |
| `spread` | The neutrophil SSC cluster widens without its centre moving, standing for a mixed population with no clear separate group. Links to the Sysmex width parameters (NE-WX/WY/WZ) | affected fraction (sets how much it widens) and shift size |

How the numbers are applied (all on a continuous scale, rounded last):

- **Linked drop.** Affected cells have SSC multiplied by (1 − shift size). SFL and FSC are multiplied by (1 − 0.5 × shift size). The 0.5 ratios are **placeholders** (`link` in the config).
- **Spread matching.** The widening is set so that the neutrophil SSC variance equals that of the subgroup case at the same fraction and shift size. This is our choice and needs review.
- **Shift sizes 5%, 10% and 20%** are placeholders for exploring detection limits, not clinical estimates. For scale: the neutrophil SSC centre is 154 with an SD of 9.5, so a 5% drop is 0.8 SD, 10% is 1.6 SD and 20% is 3.2 SD.
- **Background dysplasia.** Some patients, normal **and** signal, carry a small hypogranular neutrophil fraction (0 to 2%), as seen in reactive states and in healthy older people. About half of patients are carriers. It is part of the noise presets, so early-detection results are not flattered. We apply it to both groups because applying it to normals only would make the 0% leakage cohort distinguishable.
- Planted cells are chosen among neutrophils that are not already background-dysplastic.

## 6. Noise presets

Every patient draws its own cell mix and its own gain per group and reading. Both presets give a 5% spread on every reading, so they differ only in structure.

| Preset | What happens |
|---|---|
| `independent` | Each cell group varies on its own from patient to patient. Hard to cancel out |
| `shared` | One patient-level factor per reading moves all groups together, plus a 1% leftover per group. Can be cancelled by comparing against a reference group |
| `independent_clean`, `shared_clean` | The same without background dysplasia. Not in the default grid |

An earlier toy simulation scored the same 1% shift at AUROC about 0.53 under independent noise and about 0.93 under shared noise. The 5% size is our assumption. Sample age and other factors are not modelled yet.

## 7. The grid and how much data

Default grid (editable under `grid` in `configs/experiment.json`):

| Signal type | Affected fractions | Shift sizes |
|---|---|---|
| `subgroup` | 2%, 5%, 10%, 20% | 5%, 10%, 20% (full crossing) |
| `spread` | 2%, 5%, 10%, 20% | 10% |
| `shift` | fixed at 100% | 10% |
| none (leakage check) | 0% | none |

For each of 2 noise presets and 5 seeds: 1 leakage cohort + 12 subgroup + 4 spread + 1 shift = **18 cohorts**. That gives **180 cohorts** (10 + 120 + 40 + 10) in the full plan, each with 600 patients of 2,000 events: **216 million cells**, about 830 MB stored.

**Budget plan:** seeds 1 to 3 and 200 + 200 patients per cohort: 108 cohorts, 86 million cells. Each budget cohort is the first 200 patients of each group of the matching full cohort, so nothing extra is stored.

Seeds 1 to 5 (full) and 1 to 3 (budget) are the same seed numbers as the earlier version.

## 8. Open questions and limitations

- **Borrowed positions.** All cell positions come from body-fluid-mode files, not blood. They should be re-measured from real blood WDF files from our lab's Sysmex.
- **Placeholders.** Shift sizes (5/10/20%), the linked SFL and FSC drop (0.5), the 5% noise, the IG position, the eosinophil position and the debris share are all assumed.
- **Shift is not "slight" at 10%.** Moving every neutrophil down by 10% shifts NE-SSC by about 15 units, much more than a 10% subgroup does. We may want a 5% shift cohort for a fairer comparison.
- **Neutropenia** confounder: off by default, because it would leak the label.
- **Sample age** can change neutrophil scatter in real samples. Not modelled yet.
- **Not a longitudinal model.** Affected fractions place test points against the 10% WHO line. They do not describe how a real patient progresses.
- **Not real patients.** All data is synthetic, and the report-average view is a proxy built from the simulated cells.
- **Citations needed:** the adult blood reference ranges and the published NE-SSC/SFL/FSC AUCs. The data paper is cited as Ai et al. 2018 in the code repository and as Ai et al. 2018 in the pitch deck (same DOI 10.1371/journal.pone.0190886); check the first author.

## 9. Reproduce

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # versions pinned; requirements.lock.txt has the full freeze
.venv/bin/python -m pytest tests -q               # 18 tests
.venv/bin/python cellbench/make_data.py           # manifests + pilot + 180 cohorts into data/synthetic
.venv/bin/python cellbench/make_data.py --events 10000   # more cells per patient
.venv/bin/python cellbench/sanity_report.py       # printed sanity checks
.venv/bin/python cellbench/plot_patients.py       # 10 control vs 10 signal patients, scatter plots
```

## 10. Evaluation protocol (fixed before the main sweep)

- **Data views:** report summary, rich summary and full per-cell maps, all read by the **same MLP** (`MLP_CONFIG` in `cellbench/models.py`: hidden layers 64 and 32, dropout 0.3, Adam, learning rate 1e-3, weight decay 1e-3, early stopping on an inner validation split, 3 random starts averaged). It is frozen and was never tuned on any seed.
- **Gating:** the report and rich summaries are built from **auto-gated** groups (a gate taught on reference patients from seed 901, which sees only the 4 readings). The labelled "oracle" versions are a best case only and are not the headline.
- **Splits:** patient-level, shuffled and stratified 5-fold cross-validation inside each cohort. Scaling and the gate use training or reference patients only. Sweep seeds are 1 to 5; seeds 101 to 103 and 201 to 215 are tuning and null-check seeds and are never in the sweep.
- **Metrics:** AUROC (with a bootstrap interval), sensitivity at 95% specificity, and plain accuracy at a cutoff chosen on the other folds.
- **Detection limit (defined before seeing the sweep):** for one signal type, shift size and noise preset, the smallest tested affected fraction whose **mean AUROC over the seeds is at least 0.80** and that stays at or above 0.80 for every larger tested fraction. If no tested fraction qualifies, it is reported as "not reached".
- **Honesty checks:** no-signal cohorts must score about 0.5. A label-shuffled copy of each no-signal cohort and 30 extra no-signal cohorts on spare seeds give the baseline. A small, consistent dip below 0.5 is a known property of cross-validation with no signal and is not treated as leakage.
