# CellBench synthetic Sysmex WDF cohorts

**Fully synthetic data.** No real patient data. Made for the Leeds AI Society "AI in Healthcare" hackathon (24-25 Oct 2026) to measure how small a planted change in neutrophils different data views (report averages, richer summaries, every cell) can still detect. It is a benchmark, **not** a diagnostic dataset: no claim is made that any real disease produces changes of these sizes.

## What is inside
- `cohorts/<cohort_id>/cells.npy`: `uint8` array `(patients, events, 4)`, readings **SSC, SFL, FSC, FSCW** (whole numbers 0-255, like the 8-bit analyser export). 600 patients per cohort, 2,000 events each.
- `cohorts/<cohort_id>/labels.csv`: one row per patient: `label` (0 normal, 1 signal), `group`, `n_neutrophils`, `n_planted`, `n_background`, and the cohort settings (`signal_type`, `affected_fraction`, `shift_size`, `noise`, `seed`).
- `manifest_full.csv`: the 180 cohorts of the full plan. `manifest_budget.csv`: the 108-cohort budget plan (seeds 1-3, first 200 patients of each group, see `source_cohort`).
- `config.json`, `population_table.csv`: the exact generator settings and the cell-group recipe.

## IMPORTANT: shuffle before you split
In every `cells.npy` the **first half of the patients are normal and the second half are signal**. That is only storage order. Always split by patient with shuffling and stratification, for example `StratifiedKFold(shuffle=True)`. Splitting by position would give a training set of only normal patients and a test set of only signal patients. Never split by cell: cells of one patient are not independent.

## How it was made
Peripheral-blood-like mix (neutrophil 55%, lymphocyte 30%, monocyte 7%, eosinophil 3%, immature granulocytes 0.5%, plus debris / RBC ghosts), with a different mix and noise for every patient. In signal patients a set share of neutrophils is made hypogranular (MDS-style): lower SSC, with SFL and FSC dropping by a smaller linked amount. Signal types: `subgroup` (a fraction of neutrophils), `shift` (all neutrophils), `spread` (wider SSC). `signal_type = none` with 0% affected is the leakage check. Some normal **and** signal patients carry a small background hypogranular fraction (0-2%). Full design: `DATA_DESIGN.md` in the code repository.

## Limitations (please read)
- **Borrowed positions.** Cell positions (centres, spreads, correlations) were measured from four **body-fluid-mode** XN WDF files (Ai et al. 2018, PLOS ONE, doi:10.1371/journal.pone.0190886, CC BY) and borrowed for a blood-like mix. They should be re-measured from real blood WDF files.
- **Assumed numbers.** Cell-mix shares, the 5% patient-to-patient noise, the shift sizes (5/10/20%), the linked SFL and FSC drop (0.5), the eosinophil and immature-granulocyte positions and the debris share are assumptions. Reference ranges: [CITATION NEEDED].
- Basophils are left out (counted mainly in WNR on the XN). Only the WDF channel is modelled. Not tested in Sysmex software.
- The planted signal is a modelling choice, not a measured disease trajectory.

## Rebuild exactly
Everything is regenerated from the code and the seeds, so this upload is a convenience copy: `python cellbench/make_data.py`.

## Licence
Synthetic data derived from positions measured in CC BY files. Marked "other" on Kaggle; please credit "CellBench, Leeds AI Society hackathon 2026" and the Ai et al. 2018 source.
