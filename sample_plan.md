**The big idea:** to win, ship one tight demo. It should show a single picture that answers one question: how small can a planted MDS-style change in neutrophils get before AI stops seeing it, and does per-cell data see it earlier than the lab report's averages? Everything below serves that.

*Version 2 (2026-10-24): peripheral-blood-like patients and an MDS hypogranular-neutrophil signal. The earlier body-fluid / SFL version is archived in `archive/v1_body_fluid_sfl_signal/`. The full data design is in [DATA_DESIGN.md](DATA_DESIGN.md).*

## 1. What the resources told me

- **The medRxiv preprint (UW, 242,623 samples) is our recipe.** It uses the same hospital machine. Each sample is a set of 512 cells, fed to a set-based network. The summary version uses percentiles, spread and covariances per cell type. Its strongest signals came from spread and tails, not averages. That is why we plant three kinds of change.
- **The PMC COVID paper is our list of traps.** Its authors blame over-fitting for AUROC 0.98 in training, and warn that repeat samples let models memorise a person. Also, one plain number (HFLC%) matched the fancy model. So we split by sample, run a zero-signal check, and include a simple baseline.
- **Kanishq's dataset list is mostly the wrong kind of data.** Those are research cytometers with fluorescent antibody stains, not Sysmex files. We use only the 4 public files already in the repo, and only for cell positions.
- **Not yet seen:** the "Pretrained models" PDF and Sam's private slide link.

## 2. What we build

A small app with **three competitors** on the same fake patients:

1. **Report-level model:** averages and counts only, like the lab report (NE-SSC, NE-SFL, NE-FSC).
2. **Rich-summary model:** percentiles, spread and covariances (the preprint's recipe).
3. **Per-cell model:** a 2D histogram of all cells (no group labels), or a small set-based network.

The question: how small a **share of abnormal neutrophils** can each one still find? That detection limit is the main result. If the rich summary matches the per-cell model, that is an honest and useful finding too.

## 3. Status and next steps

1. **DONE: blood-like generator.** Six groups (neutrophil, lymphocyte, monocyte, eosinophil, immature granulocytes, debris / RBC ghosts), about 1,070 neutrophils per 2,000-event patient. Positions are borrowed from body-fluid files and marked as such.
2. **DONE: MDS signal.** Types `subgroup`, `shift`, `spread` on neutrophil SSC, with SFL and FSC dropping alongside. Affected fraction 0/2/5/10/20% and shift size 5/10/20%. Background dysplasia in some normal and signal patients. The old "strength" parameter is gone.
3. **DONE: noise as a toggle.** `independent` or `shared`, both 5% per reading.
4. **DONE: data.** 180 cohorts (600 patients each, 216 million cells) in `data/synthetic/`; the budget plan is 108 cohorts. 18 tests pass. See `data/pilot/` for plots of 10 control and 10 signal patients.
5. **NEXT: turn each patient into the three versions** (report summary, rich summary, full per-cell histogram) and **train the same model (gradient boosting) on each**, with a patient-level split. The per-cell version gets all cells without group labels.
6. **NEXT: run the sweep** over the 180 cohorts (or the budget 108) on Kaggle: one self-contained script per seed and noise preset, split across the four accounts. Save one results row per cohort and data version.
7. **NEXT: build the app (Streamlit).** Controls for signal type, shift size and noise, and the curves of score against affected fraction. Add a claim checker: type a claimed affected fraction and shift size, and it says "below the detection limit" or "detectable".
8. **NEXT: write the data page** for the Responsible Data award, and the limitations slide.

**Open points for the expert** are in section 8 of DATA_DESIGN.md: borrowed positions, placeholder shift sizes and linkage ratios, the IG and eosinophil positions, the 5% noise, and the missing reference citations.

## 3b. Doctor-facing score (with guardrails)

- **In the prototype:** a "synthetic patient scorer". Pick or generate a synthetic patient, and the model returns a **pattern score from 0 to 100%**: how likely the patient carries the planted pattern (a hypogranular neutrophil subgroup). It also shows a map of which cells the model used. Everything on screen is labelled synthetic.
- **Not in the prototype:** a "probability of disease" for a real patient, or suggested tests. The models learn "planted pattern or not" on simulated data, so a disease percentage would be a made-up number. It would also be a clinical claim, which the deck and the hackathon brief rule out.
- **Roadmap slide (the vision):** with approved real data, labelled outcomes and calibration, the same method could give a validated risk score that helps a clinician decide who to test, for example for an MDS work-up. That needs real-cohort validation, ethics and consent, and regulatory approval (clinical decision-support software can count as a medical device).

## 4. What gets us the points

- **Creativity:** the ruler idea. We aren't claiming a new diagnosis.
- **Technical:** three model tiers, leakage controls, a zero-signal check and background dysplasia in the noise.
- **Impact:** the claim checker, which gives labs a quick credibility test.
- **Design:** one page, one curve, no jargon.
- **Responsible data:** all synthetic. Say that the positions come from body-fluid files of cancer patients and are borrowed for a blood-like mix, and that the ruler shows what is detectable, not that real diseases produce such signals.

## 5. Cut these if time runs short

- The other channels (WNR, RET, PLT-F), the reflex rules and the Docker packaging.
- The set network. Use 2D histograms plus gradient boosting.
- FCS export. Keep it as a stretch. The data is stored as arrays and CSV for now.
- Seeds 4 and 5 and the second 100 patients per group: use the budget plan (3 seeds, 200 + 200 patients).

I can't promise a win. This plan gives you a clear question, a working demo and honest limits, which is what the judging criteria reward.

## How to remember it

- "Three models, one curve, one claim checker."
- "Planted MDS-style neutrophils: how few abnormal cells can each view still find?"
- "Split by patient, check zero signal, and say the data's limits."
