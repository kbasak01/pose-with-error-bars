# Limitations

Candid, and written against the final code. An accepted, documented limitation is a pass in the
Phase 8 sweep; an undocumented one is not.

Phase 8 sweep, **2026-10-08**. Every line of `VALIDATION_CHECKLIST.md` was walked against the
tree at `3e1b8d0` by four independent audits, run in this order: `split-leakage-auditor`,
`conformal-validity-auditor`, `p1-parity-auditor`, `eval-reproducibility-auditor`. Direct checks
covered the lines no auditor owns. The ranked fix list was recorded first (commit `0bfe555`), then
every fix was made, and the changed areas were re-audited. The follow-up audits were
`conformal-validity-auditor` (SOUND), `split-leakage-auditor` (no critical, no must-fix) and
`uncertainty-head-diagnostician` (seed 2026 TRUSTWORTHY).

| | ✅ | ⚠️ | ❌ |
|---|---|---|---|
| Before fixes (`3e1b8d0`) | 41 | 9 | 8 |
| **After fixes** | **48** | **4** | **6** |

Of the six ❌ that remain, one is real and five are owned by Phase 9:
- **G2** is the fp32 variance parity gate. It is root-caused and deliberately not loosened.
- **H1, H3, H4, H5 and I4** need a README and a release, which are Phase 9 by plan.

No leakage was found and no coverage number is invalid. `results/TABLES.md` regenerates
byte-identically from the JSON.

---

## Resolved at the sweep

These were ranked before any fix (critical → must → should) and are recorded rather than deleted.
As in P1's sweep, `results/` was almost never wrong. Most problems were prose that had drifted from
correct artifacts, and claims stated more strongly than their evidence.

| # | Rank | What was wrong | How it was closed |
|---|---|---|---|
| 1 | critical | This file claimed `keypoint_a1` served "as a check"; no such result exists | Removed; scope is `keypoint_a2` only (below) |
| 2 | critical | `docs/SCORES.md` and this file said the Phase 2 Beta-law test "measures" the P1 selection effect (B8) | Both now say no test on validation frames can, citing `docs/DECISIONS.md` Phase 2 |
| 3 | must | D6: one variance-head seed | Seed 2026 trained and evaluated; `results/level_c/variance_head_seed_spread.json`; diagnostician TRUSTWORTHY. See *Variance head* |
| 4 | must | J2: no worked recalibration example | [`docs/RECALIBRATION_EXAMPLE.md`](RECALIBRATION_EXAMPLE.md) + `scripts/recalibration_example.py`, run in CI |
| 5 | must | E1/E3/E8: α-grid tables gave no n; slice and grid cells had no silent-failure rate; PROGRESS quoted Mondrian and oracle sunlamp coverage without answer rate or n | `make_tables.py` cells now carry `ans (n answered) · sf`, and the grid has an n column. No number changed (re-audited line by line); pinned by a `TABLES.md` test. PROGRESS lines corrected |
| 6 | must | PROGRESS Phase 4 runtimes were stale | Re-copied from the propagation JSON, with the correction stated in place |
| 7 | must | DECISIONS: A3 law CDF 0.025/0.064 (actually 0.024/0.063), and "epochs 13–20 within 0.01 nats" (epoch 14 is 0.039 above) | Corrected in place, marked as Phase 8 corrections |
| 8 | must | Seven records stamped `12033fa…-dirty`; `dump_summary` and `p1_checkpoints` from different commits | Re-produced at a clean tree and compared exactly: all 10 identical outside provenance ([`results/reproducibility/rerun_check.json`](../results/reproducibility/rerun_check.json)) |
| 9 | must | `coverage_check.py` overwrote the source result's git SHA | It now keeps `source_poseconf_git_sha` |
| 10 | should | Nothing asserted synthetic `train` ∩ `val_*` = ∅ | `assert_train_disjoint` in variance-head training, plus a dataset test (47,966 train frames, overlap 0) |
| 11 | should | The empty-channel export path had never seen an empty channel | `test_a_real_empty_channel_survives_export`: a forced-empty channel through ORT, mask equal to torch, C1 radius ∞ |
| 12 | should | `conformal/aci.py` was a one-line stub | Deleted |
| 13 | should | `rational_alpha` let α round to 0 or 1 (Phase 1 item S4) | Raises; tested |
| 14 | should | The MAPIE oracle was `slow`, so CI never ran it | Unmarked (about 1.5 s); CI runs 44 MAPIE cases |
| 15 | should | The HIL label guard accepted any `oracle_` prefix and an empty frame list | Exact allow-list `p1_adapter.ORACLE_TAGS`; empty list refused; tested |
| 16 | should | `results/calibration/index.json` held an absolute local path | Rebuilt with a relative path; all 12 `calibration_id`s unchanged, replay `all_match` |
| 17 | should | No check-only splits target | `make splits-check` (temp dir, byte compare) |
| 18 | should | PROGRESS said synthetic is the *only* guaranteed domain; DECISIONS pilot numbers were unlabelled | Wording aligned with `results/TABLES.md`; pilot marked as having no file |

**Found by the follow-up audits and closed:**
- **The seed-2026 DEVIATES cell is now reported** (below), with a permutation-seed sensitivity
  record beside it.
- **`rerun_check` records the re-run side**: its SHA-256 and git SHA, refused if `-dirty`, plus
  every input-provenance key that differs.
- **The example compares like with like.** Coverage given answered sits beside the naive-Gaussian
  baseline.
- **`src/poseconf.egg-info/` is untracked.**

## Accepted limitations, documented

### Scope

**Single pose model.** All conclusions are about P1 `keypoint_a2`, with its detector, on SPEED+.
No other keypoint network was dumped or calibrated, so nothing here generalises to keypoint
pipelines as a class.

**HIL is not flight.** `lightbox` and `sunlamp` are laboratory hardware-in-the-loop imagery of a
mock-up. Measured coverage on them is evidence about *this* synthetic → HIL shift, not about orbit.

**Exchangeability holds only approximately in distribution, and that cannot be tested here.**
P1 chose its `best.pt` with one scalar computed over all 11,994 synthetic validation frames. Those
frames stay exchangeable *with each other*, so `val_cal` → `val_test` coverage is still valid
between them. Selection may break exchangeability with a *fresh* synthetic draw, and no test using
validation frames can see that (`docs/DECISIONS.md`, Phase 2, "Limits of this test"). The Phase 2
re-split test ([`results/level_a/keypoint_a2_predicted_crop_synthetic.json`](../results/level_a/keypoint_a2_predicted_crop_synthetic.json))
shows the production path matches the exact re-split law: 39 VALID, 9 DEGENERATE, 0 DEVIATES. It
says nothing about §0(4).

**The committed split runs low for A3.** The fixed `val_cal → val_test` split sits low in the
re-split law for A3: `fixed_split_law_cdf` 0.0386 at `abstain_allowed`, α = 0.10 (same file). Its
coverage CI lies below 0.90. This is one draw from a valid law, not a validity failure. Any quote
of that row carries the CDF beside it.

### Coverage under shift

**Synthetic-calibrated HIL coverage is measured, never guaranteed.** Every HIL number comes with
a Clopper–Pearson interval, n and the answer rate (`results/TABLES.md`).

**Sunlamp is small after PnP failures.** On `sunlamp_poolB`, 175 of 1,396 frames are answered
(A1, `split`, α = 0.10, `abstain_allowed`, predicted_crop,
[`results/shift/keypoint_a2_sunlamp_predicted_crop_split.json`](../results/shift/keypoint_a2_sunlamp_predicted_crop_split.json)).
Its near-nominal marginal coverage comes from abstentions, and coverage given answered is the
number to read beside it.

**The weighted arm is degenerate at this shift.** The domain classifier separates the domains
almost perfectly: 5-fold AUC 0.9834 on lightbox and 0.9982 on sunlamp. The ESS of the 4,798
`val_cal` weights is 1.04 and 1.60, and every answered set is ∞
([`results/shift/index.json`](../results/shift/index.json), `classifiers`). Its coverage is that
of ∞ sets plus abstentions, not a recovery. Clipping or other features would be new design
choices, and they may not be tuned on HIL, so none were tried.

### Variance head

**σ̂ ranks error. It is not a calibrated Gaussian.** On `val_test` the raw 1σ / 2σ ellipse
reliability is 0.520 / 0.881, against 0.393 / 0.865 for a 2-D Gaussian
([`results/level_c/keypoint_a2_predicted_crop_synthetic.json`](../results/level_c/keypoint_a2_predicted_crop_synthetic.json),
`head_quality`). The conformal step is what makes its sets valid.

**Two seeds, one selected head.** `vhead_a2_s1337` was selected in Phase 5 and is the only head
behind any calibration, shift or export result. `vhead_a2_s2026` was trained at Phase 8 with an
identical config except the seed, as a robustness check; nothing is selected between seeds. Two
seeds give a range, not a variance estimate. On `val_test`, from
[`results/level_c/variance_head_seed_spread.json`](../results/level_c/variance_head_seed_spread.json):

| | s1337 (selected) | s2026 |
|---|---|---|
| NLL, nats/keypoint | 3.054 | 3.073 |
| Spearman(σ̂, ‖r‖) | 0.520 | 0.509 |
| 1σ / 2σ ellipse reliability | 0.520 / 0.881 | 0.497 / 0.863 |
| C1 `abstain_allowed` α = 0.10 coverage | 0.9079 [0.8993, 0.9159] | 0.9087 [0.9002, 0.9167] |
| C2 `abstain_allowed` α = 0.10 coverage | 0.8981 [0.8892, 0.9065] | 0.9014 [0.8926, 0.9097] |
| C1 median keypoint radius, `abstain_allowed` | 19.97 px | 19.89 px |
| Re-split verdicts (VALID / DEGENERATE / DEVIATES) | 26 / 6 / 0 | 25 / 6 / 1 |

Answer rate is 0.9285 for both, n = 4,797. Every Phase 5 conclusion holds for both seeds: the head
beats heatmap moments by about 2 nats, ranks error better than confidence, and is not a calibrated
Gaussian (`uncertainty-head-diagnostician`, TRUSTWORTHY for both).

**One re-split cell of seed 2026 fails the per-cell gate. It belongs to the permutation set, not to
the cell.** The cell is C1 `answer_required` α = 0.5:
- law KS p = 0.0059 against the 0.01 gate;
- mean +2.43 MC s.e. *above* the exact law, which is the safe direction;
- mean inside the plan band, sd ratio 0.99.

The committed verdict stays DEVIATES. Every cell and both heads share one set of 1,000
permutations, and the gate is per cell with no multiplicity correction. 26 non-degenerate cells at
0.01 expect about 0.26 false rejections per run. Under independent permutation seeds the cell
does not recur: 0 DEVIATES in 5 runs at R = 1,000 and one at R = 20,000 for s2026, and 0 in 5
runs for s1337
([`results/level_c/seed_2026/resplit_seed_sensitivity.json`](../results/level_c/seed_2026/resplit_seed_sensitivity.json)).
The gate was not changed. A multiplicity-aware gate would have to be fixed before a future run, not
after this one.

**The σ clamp values are not stored in the s1337 training record.**
`results/level_c/variance_head_training.json` records the clamp *hit* fractions per epoch (0 at
the selected epoch 15). The bounds themselves (σ ∈ [0.05, 64] px, |ρ| ≤ 0.99) are reachable only
through `config_sha256` → `configs/variance_head.yaml`; rewriting the record would mean
retraining. The s2026 record stores them (`head`).

### Export and deployment

**The fp32 variance parity gate is not met, and it stays unmet (G2).** In
[`results/export/onnx_parity.json`](../results/export/onnx_parity.json) `gate_status`, `cov_chol`
max |Δ| is 2.857e-3 against the 1e-4 gate (TF32 off on both sides). A CPU-to-CPU control also
misses it, at 7.02e-4. The shipped quantity, the C1 set radius, agrees to 6.95e-5 relative. The
mechanism is a hypothesis (`docs/DECISIONS.md`, ONNX parity). Loosening the gate, or gating on the
relative number instead, would be moving a gate to pass it (invariant 11).

**Inherited ONNX parity miss.** P1's keypoint `coords` parity gate is unmet upstream: max 3.23 px
with TF32 on, per P1's committed record. It is quoted in `onnx_parity.json` → `p1_inherited`.

**fp16 is not a deployment option for the variance outputs.** The C1 radius moves by up to 8.3 %.

**The frame budget gives p50/p99 only and records no PnP config.** It is P1's
`end_to_end_frame_budget`, which returns those two percentiles (`stats_note` in
[`results/latency/frame_budget.json`](../results/latency/frame_budget.json)). The PnP config is
the pinned one used in every other record. Re-benchmarking to add fields was rejected, for P1's
reason: a second run on this card moves the fast providers by more than the differences being
read. In the same record, the fp16 keypoint stage is slower than fp32, and the cause is not
established.

**The A4000 is not a Jetson.** Every latency JSON and table carries the caveat.

### Ship-deck reuse

**`ConformalPoseHead` names its locks after P1 (J1).** The numerics take only scores, predictions
and fitted constants, and `conformal/` has no SPEED+ constant. But the artifact requires a
`p1_checkpoint_sha256` lock and `context.crop_source ∈ {predicted_crop, gt_crop}`. A ship-deck
pipeline can fill both, and [`docs/RECALIBRATION_EXAMPLE.md`](RECALIBRATION_EXAMPLE.md) shows the
mapping on a 6-marker deck target. It cannot rename them without a schema bump, and that bump
would re-hash all 12 committed artifacts.

### Publication (owned by Phase 9)

There is no `README.md` and no GitHub release yet, so H1, H3, H4, H5 and I4 are ❌ by plan and
are not critical findings.

---

## Phase 8 checklist sweep

✅ met · ⚠️ met with a caveat documented above · ❌ unmet

### A. Conformal correctness

| # | Item | | Evidence |
|---|---|---|---|
| A1 | `⌈(n+1)(1−α)⌉`-th order statistic, brute force at n ∈ {1,2,9,10,99,4798} | ✅ | `split.py:77-126` uses an exact `Fraction` index. `tests/test_conformal_split.py:20-67` enumerates the definition at those n × 20 α and pins float traps (n = 9, α = 0.1 → k = 9) |
| A2 | `+∞` when k > n; never clips | ✅ | `split.py:124-125`; `test_index_past_n_gives_inf_not_the_max`, `test_empty_calibration_gives_inf`. No `np.quantile` touches calibration scores |
| A3 | NaN raises; ±∞ per convention | ✅ | `as_scores` raises on NaN (`split.py:101`); `metrics.outcomes` raises on a convention mismatch; `test_infinite_scores_sort_and_are_not_dropped` |
| A4 | Agrees with MAPIE and TorchCP | ⚠️ | MAPIE 1.5.0 agrees at rel 1e-12 on 40 cases (`tests/test_conformal_oracles.py`), and since this sweep it runs in CI (no longer `slow`). TorchCP is absent: it is not installable beside P1's numpy pin, and exact brute force replaces it (`docs/DECISIONS.md` 2026-10-06 ×2) |
| A5 | ≥ 2,000 resamples inside the Beta 99 % band, three noise models | ✅ | `results/validity/conformal_core_synthetic.json`: 16/16 VALID, R = 2,000, KS p ≥ 0.033; reproduced into scratch by the auditor |
| A6 | Weighted CP: test mass at `+∞`; known shift restored; ESS | ✅ | `weighted.py:68`; `test_test_point_mass_sits_at_inf`; same JSON: 0.9050 weighted vs 0.7502 split at α = 0.10, ESS 192.7 of 500 |
| A7 | Small Mondrian groups → `∞`, never borrow | ✅ | `mondrian.py:47`; `test_missing_group_is_inf_not_borrowed`, `test_small_group_threshold` |
| A8 | No set function takes a label | ✅ | `test_set_functions_take_no_labels` (all 7 scores); data-level `test_c1_c2_sets_and_sigma_ignore_labels`, `test_level_a::test_sets_never_see_labels` |
| A9 | `conformal/` imports no torch | ✅ | `tests/test_no_torch_in_conformal.py`: AST scan plus a fresh-interpreter runtime check |

### B. Splits and leakage

| # | Item | | Evidence |
|---|---|---|---|
| B1 | Manifests with SHA-256; regeneration byte-identical | ✅ | `sha256sum -c splits/SHA256SUMS` 7/7 OK; `make_splits.py --check` "byte-identical: 7 manifests + SHA256SUMS"; `tests/test_splits.py:130` |
| B2 | `val_tune`/`val_cal`/`val_test` disjoint, exact cover | ✅ | 2,399 / 4,798 / 4,797, union 11,994, every intersection 0; `tests/test_splits.py:151` |
| B3 | HIL poolA/poolB disjoint, exact cover | ✅ | lightbox 3,370 + 3,370, sunlamp 1,395 + 1,396, overlap 0; `tests/test_splits.py:142` |
| B4 | No HIL label read outside poolB or `oracle_*` | ✅ | Every read site goes through `p1_adapter.hil_label_access_allowed` / `HILLabelAccessError` (`load_sidecar`, `load_eval_labels`, `load_dump_labels`). Since this sweep the guard is an exact allow-list (`p1_adapter.ORACLE_TAGS`) and refuses an empty frame list; `tests/test_dump.py`, `tests/test_shift.py:299,321,342`; re-audited |
| B5 | Only `weighted_unlabeled_target` reads poolA images; never labels | ✅ | `run_shift_matrix.py:413-470` reads `enc_feat` only via `dump_features`, which refuses oracle dumps; the classifier rows are asserted disjoint from cal and eval; `tests/test_shift.py:330,342` |
| B6 | All fitting on `val_tune` only | ✅ | `configs/conformal.yaml:96` `fit_split: synthetic_val_tune`; `level_a.py:222`; `shift.py:393,443`; `weight_clip_quantile: null`; `train_variance_head.py:117-124` raises on a non-tune frame; `run_shift_matrix.py:260-269` asserts the data roles |
| B7 | Nothing chosen by looking at HIL coverage | ✅ | Matrix code was last changed in the matrix commit `1103ba2` and the figures commit `b3a2795`; later `configs/` changes are synthetic-only (`f6f9720`); post-run commits are presentation-only |
| B8 | P1 selection caveat disclosed with the Phase 2 evidence | ✅ | *Scope* above, citing `results/level_a/…synthetic.json` and stating what that evidence cannot show; `docs/SCORES.md` corrected (fix 2) |

### C. Reproduction of P1

| # | Item | | Evidence |
|---|---|---|---|
| C1 | Pinned at `phase-9-complete`, clean | ✅ | `git submodule status` → `0f81b426… (phase-9-complete)`, `status --porcelain` empty, one pin commit in history |
| C2 | Checkpoint SHA-256 match the release manifest | ✅ | Re-hashed live into scratch: 4/4 match, identical to `results/p1_checkpoints.json` |
| C3 | Flags ≥ 99.9 % per domain; median \|Δe_r\| ≤ 1e-6 rad | ✅ | `check_p1_parity.py` re-run into scratch equals `results/dump_summary.json`: agreement 1.0 in 6/6 cells, median \|Δe_r\| ≤ 1.4e-9 rad; dump hashes unchanged since Phase 3 |
| C4 | Solve rates match P1's JSON | ✅ | `solved_count_exact` true in all 6 cells (synthetic 11,159 / 11,146; lightbox 3,630 / 4,399; sunlamp 362 / 675) |

### D. Variance head

| # | Item | | Evidence |
|---|---|---|---|
| D1 | Trunk hash-identical after training | ✅ | `results/level_c/variance_head_training.json`: `p1_state_hash_before == p1_state_hash_after`; `test_frozen_parameter_hash_unchanged_after_a_training_step` |
| D2 | Coordinates bit-identical with/without the head | ✅ | `test_outputs_are_p1_outputs_bit_for_bit`, `test_real_checkpoint_bit_identical_under_p1_autocast`; all 6 sidecars `kp_pred_crop_bit_equal_to_base = True` |
| D3 | Σ PD for every output; σ clamps recorded | ⚠️ | Cholesky with clamped σ and ρ (`test_covariances_positive_definite_and_clamped`); hit fractions per epoch, 0 at the selected epoch for both seeds. The clamp *values* sit in the s2026 record but only via `config_sha256` for s1337 (accepted above) |
| D4 | Overfit-one-batch gate | ✅ | `results/level_c/variance_head_smoke.json`: NLL 3.3525 vs homoscedastic floor 7.8065, `passed: true` |
| D5 | Select on `val_tune`, report on `val_test` | ✅ | `data.selection.split = val_tune` (2,399), best epoch 15; `head_quality` on `val_test` |
| D6 | Two seeds; spread reported | ✅ | Seeds 1337 and 2026, `results/level_c/variance_head_seed_spread.json`; table under *Variance head*; diagnostician TRUSTWORTHY for s2026 |
| D7 | NLL and set size vs heatmap moments | ✅ | `val_test` NLL 3.054 vs 5.432 nats/keypoint; set size mixed (C1 vs B2 at α = 0.10: 20.0 vs 16.4 px `abstain_allowed`, 25.6 vs 30.8 px `answer_required`), level_c JSON `comparison` |

### E. Evaluation rigour

| # | Item | | Evidence |
|---|---|---|---|
| E1 | Every table states convention, arm, α, n | ✅ | Every `results/TABLES.md` section header states convention, arm and α; the α grid now has an n column (fix 5) |
| E2 | Clopper–Pearson 95 % everywhere | ✅ | `metrics.clopper_pearson` (exact Beta); every coverage cell in `results/TABLES.md` |
| E3 | Answer and silent-failure rate beside every coverage | ✅ | Every coverage cell in `results/TABLES.md` carries `ans … (n answered) · sf …`, pinned by `test_committed_coverage_cells_carry_answered_n_and_silent_failure`; PROGRESS Phase 6 lines corrected |
| E4 | `∞` set sizes reported | ✅ | Weighted rows print `∞ / ∞`; the α grid flags **set ∞**; DEGENERATE rows kept |
| E5 | Oracle rows in separate sections | ✅ | `results/TABLES.md` "Non-oracle rows" / "Oracle rows — HIL `gt_crop`" / few-label section; every oracle JSON has `oracle: true`; `tests/test_shift.py:367,474` |
| E6 | Repeated-split coverage vs Beta law | ✅ | Levels A/B/C (selected head): 39/9/0, 26/6/0, 26/6/0 VALID/DEGENERATE/DEVIATES against the exact re-split law. Plain-Beta KS rejection explained by finite m (`docs/DECISIONS.md` Phase 2). Seed 2026's one DEVIATES is explained and does not recur under independent permutations (`results/level_c/seed_2026/resplit_seed_sensitivity.json`) |
| E7 | Conditional slices for every domain | ✅ | `results/TABLES.md` "Conditional coverage slices": range, confidence, IoU × 3 domains × 2 conventions |
| E8 | Sunlamp always with n answered and CI | ✅ | Headline 175 / 1,396 with CI; every α-grid and slice cell now gives n answered; PROGRESS Mondrian and oracle sunlamp lines carry the answer rate, n answered and CI |
| E9 | Same PnP config in every row | ✅ | 66 of 76 result JSONs carry it, all identical (EPnP, 5.0 px, 6 inliers, 1,000 iterations, 0.99) and equal to the pinned P1 config |

### F. Propagation and geometry

| # | Item | | Evidence |
|---|---|---|---|
| F1 | PURSE ≡ joint keypoint coverage | ✅ | `test_purse_membership_is_joint_keypoint_coverage` (B1/B2 × both conventions); level_b JSON `checks.purse_agreement` 142,528 / 142,528 |
| F2 | Jacobian vs finite differences | ✅ | `test_projection_jacobian_matches_finite_differences`, `test_cv2_rvec_columns_match_finite_differences`, `test_left_jacobian_first_order` |
| F3 | Sampled radius labelled inner / lower bound | ✅ | `docs/SCORES.md:170`, `run_level_b.py:95-100` definitions, `propagate.py:42-46`, `results/latency/postprocess.json` definitions |
| F4 | `pose-geometry-verifier` sign-off on `propagate.py` | ✅ | VERIFIED in Phase 4, then again on the residual-aware linearisation (`docs/DECISIONS.md`, Phase 4 audits); `propagate.py` code unchanged since (`1175135`) |

### G. Export and deployment

| # | Item | | Evidence |
|---|---|---|---|
| G1 | Opset 17, `.eval()`, named I/O, checker, no forbidden ops | ✅ | `results/export/onnx_export.json`; auditor re-ran `onnx.checker` full check on the live fp32/fp16 graphs, hashes equal the record; `tests/test_export.py::test_graph_meets_the_export_checklist` |
| G2 | Variance tensor parity < 1e-4 fp32 | ❌ | `cov_chol` 2.857e-3; documented above, gate not moved |
| G3 | Inherited P1 miss stated | ✅ | `onnx_parity.json` `p1_inherited.all_met: false`; above and `docs/PROGRESS.md` Phase 7 |
| G4 | ≥ 50 / ≥ 500, p50/p90/p99/max, provider per row | ⚠️ | `keypoint_stage.json` and `postprocess.json` meet it per row; `frame_budget.json` is p50/p99 only (accepted above) |
| G5 | Calibration round-trips; mismatched hash refused | ✅ | Auditor loaded all 12 artifacts: quantiles equal their source rows; wrong P1 hash and edited quantile refused 12/12; `results/calibration/index.json` replay 12/12 |
| G6 | A4000 caveat on every latency table | ✅ | `hardware_caveat` in all 3 latency JSONs and in each `results/TABLES.md` latency caption |

### H. Claims and honesty

| # | Item | | Evidence |
|---|---|---|---|
| H1 | README numbers trace to `results/` | ❌ | No README (Phase 9). `docs/` traced number by number; mismatches are fixes 6–7 |
| H2 | No "guaranteed / certified / provably / safe" on HIL | ✅ | Grep over `docs/`, `results/`, `src/`, `scripts/`, `configs/`: every "guarantee" qualified, HIL always "measured"; "safe" only in code comments about label-free reads. README re-checked in Phase 9 |
| H3 | "What the guarantee says and does not say" box | ❌ | No README (Phase 9). The content exists in `docs/SCORES.md` and `results/TABLES.md:5` |
| H4 | Prior art cited, contribution stated relative to it | ❌ | `SOURCES.md` only; no README (Phase 9) |
| H5 | README limitations section | ❌ | This file has them; README is Phase 9 |

### I. Repository quality

| # | Item | | Evidence |
|---|---|---|---|
| I1 | `make smoke` green on a clean clone (CI) | ✅ | CI run 37697251527 on `3e1b8d0`: CPU-only, `submodules: recursive`, no dataset, runs `make smoke`; locally 461 passed, 61 deselected |
| I2 | Licences; SPEED+ attributed, not redistributed; HIL imagery terms | ✅ | `LICENSE` MIT; `SOURCES.md:11`; both image figures carry "SPEED+ imagery, CC BY-NC-SA 4.0" on the figure and show synthetic frames only |
| I3 | No dumps, checkpoints, ONNX, large binaries | ✅ | Largest tracked files are two PNGs (2.7 MB, 1.9 MB) and the ~370 KB re-split `.npz` results; no `.pt`/`.onnx` in any commit. `src/poseconf.egg-info/` (build output) untracked at this sweep |
| I4 | Release assets with `SHA256SUMS.txt` | ❌ | No release (Phase 9) |

### J. Ship-deck reusability

| # | Item | | Evidence |
|---|---|---|---|
| J1 | `ConformalPoseHead` from scores + predictions only; no SPEED+ constants | ⚠️ | `FrameEstimate` has no label field (`test_predict_takes_no_label`); K, wireframe, camera are inputs; the J2 example runs it on a 6-marker target. Lock slots are P1-named (accepted above) |
| J2 | Worked recalibration example in `docs/` | ✅ | [`docs/RECALIBRATION_EXAMPLE.md`](RECALIBRATION_EXAMPLE.md), `scripts/recalibration_example.py`, `results/validity/recalibration_example.json`; `tests/test_recalibration_example.py` runs it in CI |

### Sweep totals

| | A | B | C | D | E | F | G | H | I | J | Total | Before fixes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ✅ | 8 | 8 | 4 | 6 | 9 | 4 | 4 | 1 | 3 | 1 | **48** | 41 |
| ⚠️ | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | **4** | 9 |
| ❌ | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 1 | 0 | **6** | 8 |
| *items* | *9* | *8* | *4* | *7* | *9* | *4* | *6* | *5* | *4* | *2* | ***58*** | *58* |

The four ⚠️ are accepted limitations, each written up above:
- A4: no TorchCP;
- D3: s1337's clamp values reachable only via the config hash;
- G4: frame budget p50/p99 only;
- J1: P1-named lock slots.

The ❌ that stays open on its merits is G2, which is real and root-caused. The other five are the
README and release that Phase 9 owns.
