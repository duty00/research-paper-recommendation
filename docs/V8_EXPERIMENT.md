# V8: local semantic ensemble

V8 candidate 1 scored **0.66699** on the Public leaderboard, as reported in the user-provided screenshot on 27 October 2026. This improves on V6 candidate 2 (**0.66504**) by **0.00195**. The screenshot shows Team DMM at rank 3. The Private score is not available.

## Model

The selected addition is `bge_neighbors30`, blended at weight **0.15** with the frozen V6 ensemble. Selection uses the top **62.5%** by predicted gain plus 0.5 times probability of a worthwhile paper; the selected papers are ordered by gain.

The search compared RBF kernel ridge regressors and distance-weighted semantic neighbors. Both predict expected gain and the probability of a rating above 1. Embedding centering, venue encoding and training targets are fitted inside each training fold. The neighbor estimate uses training papers only, with a smoothed global prior; no query labels or Public scores enter these calculations.

## Validation

Seed 361 shortlisted one configuration per encoder and model family. These four configurations were frozen before seed 1361. The best two-seed mean selected one configuration, which was frozen before seed 2361. The third result did not trigger retuning. The run was configured to export this experimental candidate even if the third comparison was negative.

| Seven-fold seed | V6 candidate 2 | V8 candidate 1 | Difference |
|---|---:|---:|---:|
| 361 | 0.629567 | 0.629311 | -0.000256 |
| 1361 | 0.628207 | 0.629261 | +0.001054 |
| 2361 | 0.629501 | 0.631199 | +0.001698 |
| Mean | 0.629092 | 0.629924 | +0.000832 |

These are repeated validations on the same labeled training set. The third seed is another partition, not an independent dataset. Small differences are uncertain and do not establish a Kaggle improvement. Scores are computed with the locally implemented Evaluation-page metric.

The policy-only probe did not give a material gain. A separate cross-encoder metadata probe is recorded under `outputs/v8_cross_probe/`; an unavailable cached head was not confirmed. Neither probe changes this frozen candidate.

## Reproduce the CSV

Use Python 3.12 and the pinned dependencies. No Docker, historical CV outputs, or saved classifier bundle is needed for the frozen replay. Obtain the original competition files separately. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
.\.venv\Scripts\python.exe -X utf8 reproduce.py prepare
.\.venv\Scripts\python.exe -X utf8 v8.py reproduce --data-dir .\assignment-data-mining-261 --config .\configs\v8_candidate_1.json --out .\outputs\v8_replayed
```

Expected SHA-256: `a35b53f97a9982296deee437a2145ba5d9a965be2e684ceddd7ddcd0664a1163`.
The command independently refits the frozen classifiers, validates all IDs and priorities, and fails if the CSV hash differs from this run. Cross-platform exact agreement has not been checked.

To repeat the model search as well, retain `outputs/v6_metadata_run01` and run `v8.py run --data-dir .\assignment-data-mining-261 --out .\outputs\v8_new_search`. This search command verifies the historical parent OOF; the frozen replay above does not need that directory.

## Files

- `configs/v8_candidate_1.json`: frozen model parameters, assets, dependencies and hashes.
- `results/v8_validation.json`: scores and fold-level measurements.
- `outputs/v8_run01/reports/`: full search plan, frozen decisions and results.
- `outputs/v8_run01/submissions/v8_candidate_1.csv`: upload manually to Kaggle.
- `outputs/v8_run01/models/v8_candidate_1.joblib`: fitted classifier bundle.

No Kaggle upload or final-submission selection is performed by these scripts.
