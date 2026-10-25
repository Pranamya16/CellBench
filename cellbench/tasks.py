"""The confirmation tasks: tuning seeds only (never the sweep seeds 1-5)."""


def confirmation_tasks(seeds=(101, 102, 103), n=300):
    """List of (settings, shuffle_labels, tag). Per noise preset and seed: 8 subgroup cohorts, spread, shift, a null
    cohort and the same null with shuffled labels."""
    tasks = []
    for noise in ["independent", "shared"]:
        for seed in seeds:
            base = dict(seed=seed, noise=noise, patients_per_group=n)
            for f in (0.02, 0.05, 0.10, 0.20):
                for d in (0.10, 0.20):
                    tasks.append((dict(base, signal_type="subgroup", affected_fraction=f, shift_size=d,
                                       cohort_id=f"c_{noise}_{seed}_sub_{f}_{d}"), False, "signal"))
            tasks.append((dict(base, signal_type="spread", affected_fraction=0.20, shift_size=0.10,
                               cohort_id=f"c_{noise}_{seed}_spread"), False, "signal"))
            tasks.append((dict(base, signal_type="shift", affected_fraction=1.0, shift_size=0.10,
                               cohort_id=f"c_{noise}_{seed}_shift"), False, "signal"))
            null = dict(base, signal_type="none", affected_fraction=0.0, shift_size=0.0, cohort_id=f"c_{noise}_{seed}_null")
            tasks.append((null, False, "no_signal"))  # not "null": pandas reads that text as a missing value
            tasks.append((null, True, "null_shuffled"))
    return tasks
