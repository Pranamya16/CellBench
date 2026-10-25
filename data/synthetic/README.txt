Synthetic Sysmex XN WDF cohorts, peripheral-blood-like mix (neutrophils, lymphocytes, monocytes, eosinophils,
immature granulocytes, debris/RBC ghosts). Cell positions are BORROWED from body-fluid WDF files. Not clinical data.
Events per patient: 2000.
cohorts/<id>/cells.npy  uint8 (patients, events, 4): SSC, SFL, FSC, FSCW. First half normal, second half signal.
cohorts/<id>/labels.csv one row per patient: label (0 normal, 1 signal), settings, n_neutrophils, n_planted, n_background.
manifest_full.csv       all cohorts of the full plan (300 + 300 patients, seeds 1-5).
manifest_budget.csv     budget plan (200 + 200 patients, seeds 1-3); source_cohort is the matching full cohort,
                        load it with load_cohort(source_cohort, patients_per_group=200).
signal_type: none (0% affected, leakage check), subgroup, shift (every neutrophil; affected_fraction is fixed at 1.0
and ignored), spread. affected_fraction and shift_size replace the old 'strength' parameter.
Ground truth (true group, planted flag, background-dysplasia flag per cell) is only saved for the pilot set in data/pilot.
Any cohort can be rebuilt exactly from the code and config.json. See DATA_DESIGN.md.
The superseded body-fluid / SFL-signal data is in archive/v1_body_fluid_sfl_signal/.
