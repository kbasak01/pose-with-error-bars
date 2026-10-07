# Progress

| Phase | Status | Key numbers | Tag |
|---|---|---|---|
| 0 Scaffold | **complete** 2026-10-06 | 4/4 P1 checkpoint SHA-256 match the release manifest ([`results/p1_checkpoints.json`](../results/p1_checkpoints.json)); P1 pinned at `0f81b426` (same file, `provenance.p1_commit`) | `phase-0-complete` |
| 1 Conformal core ⚠️ | **complete** 2026-10-06 | synthetic validity 16/16 cases VALID; split CP Beta-law KS p ∈ [0.033, 0.773] over 12 cases, R = 2,000 each; weighted CP with true weights 0.9050 vs split 0.7502 at α = 0.10 under a known shift ([`results/validity/conformal_core_synthetic.json`](../results/validity/conformal_core_synthetic.json)); `conformal-validity-auditor` SOUND | `phase-1-complete` |
| 2 Splits + Level A | **complete** 2026-10-07 | splits 2,399 / 4,798 / 4,797 + HIL 3,370/3,370, 1,395/1,396, byte-identical ([`splits/SHA256SUMS`](../splits/SHA256SUMS)); A1–A3 × 2 conventions × 8 α: 39 VALID, 9 DEGENERATE (answer_required α ≤ 0.05, q = ∞), 0 DEVIATES vs the exact re-split law, R = 1,000 ([`results/level_a/keypoint_a2_predicted_crop_synthetic.json`](../results/level_a/keypoint_a2_predicted_crop_synthetic.json)); `split-leakage-auditor` no critical, `conformal-validity-auditor` SOUND | `phase-2-complete` |
| 3 Dump + P1 parity | **complete** 2026-10-07 | 6/6 domain × arm cells reproduce P1: success-flag agreement 1.0 (0 disagreements), solved counts equal P1's JSON (synthetic 11,159 / 11,146; lightbox 3,630 / 4,399; sunlamp 362 / 675, predicted / GT box), median \|Δe_r\| ≤ 1.38e-09 rad ([`results/dump_summary.json`](../results/dump_summary.json)); `p1-parity-auditor` REPRODUCED | `phase-3-complete` |
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

## Phase 2 — Splits + Level A (complete, 2026-10-07)

- **Splits.** `splits/*.txt` + `SHA256SUMS`, seed 20261006. Sizes are synthetic `val_tune` / `val_cal` / `val_test`
  2,399 / 4,798 / 4,797, lightbox poolA/poolB 3,370 / 3,370, and sunlamp 1,395 / 1,396.
  `make_splits.py --check` is byte-identical, both from the dataset listing and from the committed
  HIL union (the latter is in CI). HIL names come from an image-directory listing only. No HIL label,
  sidecar or image was opened in this phase.
- **Level A** ([`results/level_a/keypoint_a2_predicted_crop_synthetic.json`](../results/level_a/keypoint_a2_predicted_crop_synthetic.json)):
  `keypoint_a2`, `predicted_crop`, arm `split`, synthetic `val_test` only, with n_cal = 4,798 and n = 4,797.
  - **Fit.** Normalisers were fitted on 2,251 solved `val_tune` frames: c_R = 0.01208 rad, c_t = 0.003281,
    c_z = 0.003094, c_xy = 0.000572. g(conf̄) is a decreasing isotonic fit with 70 knots.
  - **Answer rate.** 0.9285 on `val_test`.
  - **α = 0.10, `abstain_allowed`** (`val_test` coverage, then the Clopper–Pearson interval, rotation radius, translation radius):

    | Score | Coverage | 95 % CI | Rotation radius (median) | Translation radius |
    |---|---|---|---|---|
    | A1 | 0.8941 | [0.8850, 0.9027] | 2.33° | 1.10 % of ‖t̂‖ |
    | A2 | 0.8993 | [0.8904, 0.9077] | 2.83° | boresight 1.27 %, lateral 0.23 % |
    | A3 | 0.8891 | [0.8799, 0.8978] | 2.13° (p90 3.15°) | 1.01 % (p90 1.50 %) |

    Silent-failure rate is 0.106 / 0.101 / 0.111.
    The fixed-split coverage is one draw from the re-split law (`resplits.fixed_split_law_cdf` in
    the JSON). A3's 0.8891, with a CI below 0.90, has a law CDF of 0.039. That is consistent with
    marginal coverage under exchangeability of `val_cal` and `val_test`; the Clopper–Pearson
    interval conditions on this one calibration set. A3's re-split mean is 0.90002.
  - **α = 0.10, `answer_required`** (`val_test`, n = 4,797; translation radius as % of ‖t̂‖):

    | Score | Coverage | 95 % CI | Rotation radius | Translation radius | Silent-failure rate |
    |---|---|---|---|---|---|
    | A1 | 0.8989 | [0.8900, 0.9073] | 3.99° | 1.89 % | 0.0296 |
    | A2 | 0.8991 | [0.8902, 0.9075] | 4.83° | boresight 2.16 %, lateral 0.40 % | 0.0294 |
    | A3 | 0.9014 | [0.8926, 0.9097] | 3.27° (median; p90 4.83°) | 1.55 % (p90 2.29 %) | 0.0271 |

  - **`answer_required`, α ≤ 0.05.** q = +∞, so the set is the whole space and coverage is 1. This is
    reported, not fixed: the pool has 7.2 % failures, which exceeds α.
- **Validity over R = 1,000 re-splits of `val_cal ∪ val_test`.** The reference is the exact
  atom-aware Beta-Binomial law (`conformal.metrics.resplit_coverage_law`), which closes carried item S1.
  - Results: 39 / 48 cells VALID, 9 DEGENERATE, 0 DEVIATES.
  - Law KS p ranges from 0.037 to 0.999. Observed/law sd ratio ranges from 0.96 to 1.04.
  - The plain Beta KS rejects everywhere (max p 1.4e-7), because its sd ignores the finite test
    half (0.00433 vs 0.00612 at α = 0.10). This is documented as a deviation in `docs/DECISIONS.md`.
  - The plan's literal mean band holds for all 39 non-degenerate cells.
  - Per-score files and figures: [`results/validity/A*_*.json`](../results/validity/) and
    `assets/validity_*.png`.
- **What the re-split test cannot show.** Random re-splits of one pool are exchangeable by
  construction. The test validates the implementation, not §0(4) selection dependence
  (`docs/DECISIONS.md`).
- **Audits.** `split-leakage-auditor`: no critical findings, 2 warnings carried to Phase 6.
  `conformal-validity-auditor`: SOUND. Its 4 must-fix items (wording and provenance, no number
  changed) are closed. See "Phase 2 audits: carried items" in `docs/DECISIONS.md`.
- **Carried into Phase 6.** A code-level guard on HIL sidecar loading (leakage W1).

## Phase 3 — Dump + P1 parity (complete, 2026-10-07)

- **Dump.** `poseconf.data.dump` runs P1's own stages: `prepare_frame`, `detect_boxes`, `crop_one`,
  `normalise_crop`, `keypoints` and `solve_single`. It uses P1's batch size (48) and P1's PnP chunk
  seeding. Forward hooks capture heatmaps, the pooled 512-d encoder feature, and decoder features
  (the latter for a fixed 256-frame subset).
  - Per frame it stores keypoints (crop and full frame), confidence, the heatmap-moment covariance
    mapped to the full frame, and entropy. It also stores the PnP outcome, with NaN pose and a reason
    on failure.
  - The moment mean equals P1's coordinate on every non-empty channel. The maximum deviation is in
    each dump's meta (`info.max_moment_mean_dev_crop_px`).
  - Dumps live in gitignored `dumps/keypoint_a2/`, 3.3 GB. They were made at `f9ee848` with TF32
    disabled and `cudnn.benchmark` off.
- **HIL labels (invariant 4).** Prediction files are label-free. Ground truth is split physically by
  pool, and poolA can be read only under an `oracle_*` tag. HIL `gt_crop` is the oracle arm
  `oracle_gt_box`. `p1_adapter.load_sidecar` now guards label-derived HIL columns; this closes Phase 2
  leakage item W1 for the sidecar and label paths.
- **Gate** ([`results/dump_summary.json`](../results/dump_summary.json), parity run at `6736c85`):
  every domain × arm cell meets it, with numbers in the table below.
  - On HIL, \|Δe_r\| and \|Δe_t\| are measured on poolB only. poolA is gated by the label-free bound
    angle(q̂_dump, q̂_P1), whose median ranges from 2.46e-08 to 2.88e-08 rad.

  | Domain × arm | Flag agreement | Solved, dump = P1 JSON | Median \|Δe_r\| (rad), scope |
  |---|---|---|---|
  | synthetic × predicted_crop | 1.0 | 11,159 / 11,994 | 2.12e-10, all |
  | synthetic × gt_crop | 1.0 | 11,146 / 11,994 | 2.06e-10, all |
  | lightbox × predicted_crop | 1.0 | 3,630 / 6,740 | 6.45e-10, poolB |
  | lightbox × gt_crop (oracle) | 1.0 | 4,399 / 6,740 | 6.30e-10, poolB |
  | sunlamp × predicted_crop | 1.0 | 362 / 2,791 | 1.38e-09, poolB |
  | sunlamp × gt_crop (oracle) | 1.0 | 675 / 2,791 | 1.34e-09, poolB |
- **Audits.** `p1-parity-auditor`: REPRODUCED. `split-leakage-auditor`: no critical findings. Two
  warnings are closed by new tests and three are carried into Phase 6 (`docs/DECISIONS.md`, "Phase 3
  audits: carried items").
- **Carried into Phase 4.** B2 must treat `heatmap_empty` channels, which have zero covariance, explicitly.

