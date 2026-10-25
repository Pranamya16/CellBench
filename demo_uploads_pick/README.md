# Demo files (all synthetic)

Upload these on the page (http://localhost:8000, 'Your patient file'). Scores are the detector scores (0-100) the live page gave when these were chosen; the report-averages score can move by about +/-10 between machines.

## normal/ (nothing planted: expect low scores)

| file | full cell data | report averages |
|---|---|---|
| normal_1.csv | 2 | 42 |
| normal_2.csv | 9 | 34 |
| normal_3.csv | 7 | 37 |
| normal_4.csv | 9 | 42 |
| normal_5.csv | 4 | 32 |

## planted/ (a change was planted: expect a high full cell score)

| file | what was planted | full cell data | report averages |
|---|---|---|---|
| planted_1_subgroup.csv | 20% of neutrophils made 20% less granular | 100 | 88 |
| planted_2_subgroup.csv | 10% of neutrophils made 20% less granular | 100 | 56 |
| planted_3_subgroup.csv | 20% of neutrophils made 10% less granular | 100 | 62 |
| planted_4_spread.csv | neutrophil cluster widened in 10% of neutrophils | 99 | 56 |
| planted_5_shift.csv | every neutrophil 10% less granular | 100 | 98 |

## What to say
- Full cell data scores 99 to 100 on all five planted files and 2 to 9 on the five normal files.
- Report averages are weaker on the same files: 56 to 62 on planted_2, planted_3 and planted_4 (not clearly separated from the normal files, which score 32 to 42). They do see the big changes (planted_1 and planted_5).
- These are detector scores on synthetic patients, not diagnoses.
