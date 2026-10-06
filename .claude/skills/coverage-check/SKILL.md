---
name: coverage-check
description: Run the in-distribution repeated-split coverage validity test for one score and compare it with the theoretical Beta law. Invoke as /coverage-check <score-id> [convention].
#disable-model-invocation: true
argument-hint: "[score-id] [abstain_allowed|answer_required]"
---

Run the repeated-split validity test for: $ARGUMENTS

1. Load `configs/conformal.yaml`; use `val_cal ∪ val_test` only (never `val_tune`, never HIL).
   Default convention `abstain_allowed` if none given.
2. For each α in the config's grid: R re-splits (config `repeated_splits`), calibrate on one half,
   evaluate on the other; record coverage per split.
3. Compare with Beta(n+1−l, l), l = ⌊(n+1)α⌋ (see the `conformal-math` skill): mean, MC s.e.,
   KS p-value, and whether the mean lies in [1−α, 1−α+1/(n+1)] ± 3 s.e.
4. Write `results/validity/<score>_<convention>.json` (schema in `results-provenance`) and a
   histogram-vs-Beta figure to `assets/validity_<score>_<convention>.png`.
5. Report a one-line verdict per α (VALID / DEVIATES) and the table. If any α deviates, do not
   adjust anything — hand the result to `conformal-validity-auditor`.
