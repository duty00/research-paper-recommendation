# V7 experiments

Parent: **V6 candidate 2**, reported Public **0.66504**. New V7 Public scores are not available.

## SVC refinement

Six fixed variants changed C, kernel gamma, author scale, venue scale or encoder. The first CV seed did not yield a candidate above the parent; none passed the predefined two-seed gates. No CSV was exported from this branch. Its outputs are under `outputs/v7_run01/`.

## Author/venue models

This follow-up fits a separate metadata model: Ridge regression, ordinal Logistic Regression, or smoothed author offsets around venue preferences. Each is blended with the parent at a weight of 0.10, 0.25 or 0.50. The selection rule is fixed at fraction=0.60 and read coefficient=0.50.

All author statistics and vocabularies are fitted only on the outer training fold. The author model never uses parent OOF residuals as a target.

### Selection protocol

1. Select one candidate per family on seven-fold seed361 and freeze the shortlist.
2. Confirm on seed1361. Require positive deltas on both seeds, at least4/7 paired fold wins per seed, and mean delta>=0.001. Freeze at most2 candidates.
3. Recompute parent and frozen candidates on seed2361. Require nonnegative delta, at least4/7 paired wins, and three-seed mean delta>=0.001.
4. Do not retune after seed2361. Export passing candidates in the already frozen order.

These are adaptive checks on reused labels. The third seed is a new partition, not a new dataset or an untouched Private test. The parent was chosen using prior Public results; all new numerical settings use training CV.

### Results

| Candidate | Seed361 delta | Seed1361 delta | Mean delta | Passed confirmation |
|---|---:|---:|---:|---|
| author_ordinal_c01_mix10 | -0.001298 | -0.000823 | -0.001060 | False |
| author_ridge_a100_mix10 | -0.001640 | -0.001213 | -0.001427 | False |
| author_prior_s2_mix10 | -0.001376 | -0.002737 | -0.002056 | False |

No author/venue candidate passed every gate; no new CSV was produced. V6 candidate2 remains the recorded Public baseline.

Author-branch run artifacts are in `outputs/v7_author_run01/` and are ignored by Git. Search plans, frozen shortlists and per-seed scores are retained there.
