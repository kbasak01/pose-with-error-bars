# Progress

| Phase | Status | Key numbers | Tag |
|---|---|---|---|
| 0 Scaffold | **complete** 2026-10-06 | 4/4 P1 checkpoint SHA-256 match the release manifest ([`results/p1_checkpoints.json`](../results/p1_checkpoints.json)); P1 pinned at `0f81b426` (same file, `provenance.p1_commit`) | `phase-0-complete` |
| 1 Conformal core ⚠️ | **complete** 2026-10-06 | synthetic validity 16/16 cases VALID; split CP Beta-law KS p ∈ [0.033, 0.773] over 12 cases, R = 2,000 each; weighted CP with true weights 0.9050 vs split 0.7502 at α = 0.10 under a known shift ([`results/validity/conformal_core_synthetic.json`](../results/validity/conformal_core_synthetic.json)); `conformal-validity-auditor` SOUND | `phase-1-complete` |
| 2 Splits + Level A | not started | | |
| 3 Dump + P1 parity | not started | | |
| 4 Level B | not started | | |
| 5 Variance head | not started | | |
| 6 Coverage under shift ⭐ | not started | | |
| 7 Head, ONNX, latency | not started | | |
| 8 Validation sweep | not started | | |
| 9 Publication | not started | | |

Update at the end of each phase via `/phase-gate N`.

## Phase 0 — Scaffold (complete, 2026-10-06)

- Package skeleton, `p1_adapter` (sole `speedpose` import), `provenance`, checkpoint verification,
  CI with `submodules: recursive`. Gate: `make lint && make test && make smoke` green locally and in
  CI on `976894b`; `import speedpose, poseconf` works.
- Checkpoints: `detector`, `keypoint_a1`, `keypoint_a2`, `direct_regression_a1` all match the
  `phase-9-complete` `SHA256SUMS.txt` — [`results/p1_checkpoints.json`](../results/p1_checkpoints.json).
- Split sizes confirmed against P1's audit (`external/spacecraft-pose-baseline/results/dataset_audit.json`,
  `counts`): synthetic train 47,966 / validation 11,994; lightbox 6,740; sunlamp 2,791. The
  `keypoint_a2` synthetic sidecar covers all 11,994 validation frames (`tests/test_adapter.py`).
- Carried into Phase 1: TorchCP is not installable alongside P1's numpy pin; MAPIE 1.5.0 is the
  remaining library oracle (`docs/DECISIONS.md`). Sidecar box columns exist only for
  `predicted_crop` (IoU) / `pipeline_gt_crop` (boxes), so the IoU slice is `predicted_crop`-only.

## Phase 1 — Conformal core (complete, 2026-10-06)

- **Modules.** `poseconf.conformal.{split,weighted,mondrian,so3,scores,metrics}` are numpy/scipy only.
  `docs/SCORES.md` covers all seven scores, A1–C2. `p1_adapter.pose_quaternion_order()` reads
  P1's quaternion order (`scalar_first`).
- **Gate.** `make lint && make test`: 279 passed. `make smoke`: 232 passed. `conformal-validity-auditor`
  returned SOUND at `0243839`, after its one must-fix item was closed: M1, the semantics of `contains()` on
  failed frames.
- **Synthetic validity**
  ([`results/validity/conformal_core_synthetic.json`](../results/validity/conformal_core_synthetic.json),
  generated at `aca39c6`): 16/16 cases VALID.
  - Split CP, 3 score laws × n ∈ {100, 1000} × α ∈ {0.05, 0.10}, R = 2,000 resamples each.
    Exact conditional coverage was tested against Beta(n+1−l, l). KS p ranged from 0.0332 to 0.7730.
    Example: Gaussian, n = 1000, α = 0.10 gave mean 0.90029 (MC s.e. 0.00021) against a Beta mean of 0.90010.
  - Failure mixture, 5 % failures at n = 1000, α = 0.10: mean coverage 0.90018 (`answer_required`)
    and 0.90027 (`abstain_allowed`).
  - Known covariate shift with true weights: weighted CP 0.9050 vs plain split 0.7502 at α = 0.10,
    and 0.8029 vs 0.6134 at α = 0.20. Mean ESS was 192.7 of 500.
- **Oracles.** MAPIE 1.5.0 agrees on every (n, α) it accepts; it refuses small n. The exact-rational
  brute force agrees for n ∈ {1, 2, 9, 10, 99, 4798}. TorchCP was dropped (see `docs/DECISIONS.md`).
- **Carried into Phase 2**, listed under "Phase 1 audit: carried items" in `docs/DECISIONS.md`.
  The most urgent is S1: the SPEED+ re-split reference law must be Beta-Binomial, not Beta, and
  failure atoms need handling.
