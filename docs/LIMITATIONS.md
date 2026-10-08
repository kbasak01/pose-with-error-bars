# Limitations

Candid, and written against the final code. An accepted, documented limitation is a pass in the
Phase 8 sweep; an undocumented one is not.

Phase 8 sweep, **2026-10-08**, at `3e1b8d0`: every line of `VALIDATION_CHECKLIST.md` was walked
against the final tree by four independent audits, run in this order: `split-leakage-auditor`,
`conformal-validity-auditor`, `p1-parity-auditor`, `eval-reproducibility-auditor`. Direct checks
covered the lines that no auditor owns. This first pass records the verdicts *before* any fix; the
ranked fix list follows it.

**Sweep outcome before fixes: 41 ✅ / 9 ⚠️ / 8 ❌ over 58 items.** Audit verdicts:

| Audit | Verdict |
|---|---|
| `split-leakage-auditor` | no critical finding |
| `conformal-validity-auditor` | **SOUND** |
| `p1-parity-auditor` | **REPRODUCED** |
| `eval-reproducibility-auditor` | two critical *wording* findings, no wrong number in `results/` |

No leakage was found, and no coverage number is invalid. `results/TABLES.md` regenerates
byte-identically from the JSON.

---

## Ranked fixes (open at the start of the sweep)

**Critical: claims stronger than their evidence**

1. `docs/LIMITATIONS.md` said "(with `keypoint_a1` as a check)". No `keypoint_a1` result exists.
2. `docs/SCORES.md:55` and `docs/LIMITATIONS.md` said "the Phase 2 Beta-law test measures" the
   P1 checkpoint-selection effect. It cannot. `docs/DECISIONS.md` (Phase 2, "Limits of this test")
   already says that nothing using validation frames can test it (B8).

**Must fix**

3. D6. Only one variance-head seed exists (`vhead_a2_s1337`). The plan asks for seeds 1337 and
   2026 with the spread reported.
4. J2. There is no worked recalibration example for a different wireframe.
5. E1, E3, E8 in tables and progress notes:
   - The α-grid tables in `results/TABLES.md` state no n.
   - The slice and α-grid cells give no silent-failure rate.
   - `docs/PROGRESS.md` Phase 6 quotes Mondrian and oracle sunlamp coverage without the answer
     rate or n answered.
6. `docs/PROGRESS.md` Phase 4 runtimes are stale against
   `results/level_b/keypoint_a2_predicted_crop_synthetic_propagation.json`.
7. `docs/DECISIONS.md` has three numbers that do not match their JSON:
   - A3 `fixed_split_law_cdf` 0.025 / 0.064 is really 0.0245 / 0.0635.
   - "epochs 13–20 within about 0.01 nats" is wrong, because epoch 14 is 0.039 above the best epoch.
8. Seven result files carry the dirty provenance SHA `12033fa…-dirty`:
   `results/level_a/keypoint_a2_predicted_crop_synthetic.json` and
   `results/validity/A{1,2,3}_*.json`.
9. `scripts/coverage_check.py` overwrites the source result's `poseconf_git_sha`. This was carried
   from Phase 5.

**Should fix**

10. Nothing asserts that synthetic `train` is disjoint from `val_*`. This was carried from Phase 5.
    It holds today: the overlap is 0, recounted by `split-leakage-auditor`.
11. The empty-heatmap-channel export path has never run on a real empty channel. This was carried
    from Phase 7.
12. `src/poseconf/conformal/aci.py` is a one-line stub for a stretch arm, and reads as working code.
13. `split.rational_alpha` lets α round to 0 or 1, and α → 1 returns the max score. Phase 1 item
    S4 is still open. It cannot be reached from the config grid.
14. The MAPIE oracle test is marked `slow`, so CI never runs it.
15. HIL label guard:
    - It accepts any `oracle_` prefix, not an allow-list.
    - It allows an empty frame list.
16. `results/calibration/index.json` records an absolute local path.
17. `make splits` rewrites in place, and there is no check-only target.
18. Two wording fixes in the docs:
    - `docs/PROGRESS.md` says synthetic is the *only* domain where the guarantee applies. The
      oracle poolA→poolB rows also hold under exchangeability.
    - `docs/DECISIONS.md` pilot numbers have no backing file and are not labelled as a pilot.

**Accepted, not fixed** (each documented below): G2, G4's frame budget percentiles, J1's
P1-named locks, the σ clamp values not stored in the s1337 training JSON, H1–H5 and I4 (Phase 9).

---

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

**The σ clamp values are not stored in the s1337 training record.**
`results/level_c/variance_head_training.json` records the clamp *hit* fractions per epoch (0 at
the selected epoch 15). The bounds themselves (σ ∈ [0.05, 64] px, |ρ| ≤ 0.99) are reachable only
through `config_sha256` → `configs/variance_head.yaml`. Rewriting the record would mean retraining.

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
pipeline can fill both, by putting its keypoint model's hash in the P1 slot and its crop policy
in `crop_source`. It cannot rename them without a schema bump, and that bump would re-hash all 12
committed artifacts.

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
| A4 | Agrees with MAPIE and TorchCP | ⚠️ | MAPIE 1.5.0: 40 cases at rel 1e-12, 0 skips (`tests/test_conformal_oracles.py`). TorchCP was dropped and replaced by exact brute force (`docs/DECISIONS.md` 2026-10-06 ×2). The MAPIE test is `slow`, so CI does not run it |
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
| B4 | No HIL label read outside poolB or `oracle_*` | ⚠️ | All read sites are guarded by `p1_adapter.hil_label_access_allowed` / `HILLabelAccessError` (`load_sidecar`, `load_eval_labels`, `load_dump_labels`); `tests/test_shift.py:299,321,342`, `tests/test_dump.py:378-417`. The guard is a tag *prefix* and allows an empty frame list (fix 15) |
| B5 | Only `weighted_unlabeled_target` reads poolA images; never labels | ✅ | `run_shift_matrix.py:413-470` reads `enc_feat` only via `dump_features`, which refuses oracle dumps; the classifier rows are asserted disjoint from cal and eval; `tests/test_shift.py:330,342` |
| B6 | All fitting on `val_tune` only | ✅ | `configs/conformal.yaml:96` `fit_split: synthetic_val_tune`; `level_a.py:222`; `shift.py:393,443`; `weight_clip_quantile: null`; `train_variance_head.py:117-124` raises on a non-tune frame; `run_shift_matrix.py:260-269` asserts the data roles |
| B7 | Nothing chosen by looking at HIL coverage | ✅ | Matrix code was last changed in the matrix commit `1103ba2` and the figures commit `b3a2795`; later `configs/` changes are synthetic-only (`f6f9720`); post-run commits are presentation-only |
| B8 | P1 selection caveat disclosed with the Phase 2 evidence | ⚠️ | Disclosed above, but `docs/SCORES.md:55` and the previous LIMITATIONS overstated what the Beta-law test shows (fixes 1–2) |

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
| D3 | Σ PD for every output; σ clamps recorded | ⚠️ | Cholesky with clamped σ and ρ (`test_covariances_positive_definite_and_clamped`); hit fractions per epoch, 0 at epoch 15. Clamp *values* only via `config_sha256` (accepted above) |
| D4 | Overfit-one-batch gate | ✅ | `results/level_c/variance_head_smoke.json`: NLL 3.3525 vs homoscedastic floor 7.8065, `passed: true` |
| D5 | Select on `val_tune`, report on `val_test` | ✅ | `data.selection.split = val_tune` (2,399), best epoch 15; `head_quality` on `val_test` |
| D6 | Two seeds; spread reported | ❌ | Only `vhead_a2_s1337` (fix 3) |
| D7 | NLL and set size vs heatmap moments | ✅ | `val_test` NLL 3.054 vs 5.432 nats/keypoint; set size mixed (C1 vs B2 at α = 0.10: 20.0 vs 16.4 px `abstain_allowed`, 25.6 vs 30.8 px `answer_required`), level_c JSON `comparison` |

### E. Evaluation rigour

| # | Item | | Evidence |
|---|---|---|---|
| E1 | Every table states convention, arm, α, n | ⚠️ | Headline, oracle, slice and few-label tables do; the α-grid tables give no n (fix 5) |
| E2 | Clopper–Pearson 95 % everywhere | ✅ | `metrics.clopper_pearson` (exact Beta); every coverage cell in `results/TABLES.md` |
| E3 | Answer and silent-failure rate beside every coverage | ⚠️ | Headline and few-label tables do; slice and α-grid cells lack silent failure; `docs/PROGRESS.md` Phase 6 Mondrian/oracle lines lack the answer rate (fix 5) |
| E4 | `∞` set sizes reported | ✅ | Weighted rows print `∞ / ∞`; the α grid flags **set ∞**; DEGENERATE rows kept |
| E5 | Oracle rows in separate sections | ✅ | `results/TABLES.md` "Non-oracle rows" / "Oracle rows — HIL `gt_crop`" / few-label section; every oracle JSON has `oracle: true`; `tests/test_shift.py:367,474` |
| E6 | Repeated-split coverage vs Beta law | ✅ | Levels A/B/C: 39/9/0, 26/6/0, 26/6/0 VALID/DEGENERATE/DEVIATES against the exact re-split law; plain-Beta KS rejection explained by finite m (`docs/DECISIONS.md` Phase 2) |
| E7 | Conditional slices for every domain | ✅ | `results/TABLES.md` "Conditional coverage slices": range, confidence, IoU × 3 domains × 2 conventions |
| E8 | Sunlamp always with n answered and CI | ⚠️ | Headline table gives 175 / 1,396 with CI; α-grid and `docs/PROGRESS.md` Phase 6 Mondrian/oracle lines do not (fix 5) |
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
| I3 | No dumps, checkpoints, ONNX, large binaries | ✅ | Largest tracked files are two PNGs (2.7 MB, 1.9 MB) and three ~370 KB re-split `.npz` results; no `.pt`/`.onnx` in any commit |
| I4 | Release assets with `SHA256SUMS.txt` | ❌ | No release (Phase 9) |

### J. Ship-deck reusability

| # | Item | | Evidence |
|---|---|---|---|
| J1 | `ConformalPoseHead` from scores + predictions only; no SPEED+ constants | ⚠️ | `FrameEstimate` has no label field (`test_predict_takes_no_label`); K, wireframe, camera are inputs. Locks are P1-named (accepted above) |
| J2 | Worked recalibration example in `docs/` | ❌ | None (fix 4) |

### Sweep totals (before fixes)

| | A | B | C | D | E | F | G | H | I | J | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ✅ | 8 | 6 | 4 | 5 | 6 | 4 | 4 | 1 | 3 | 0 | **41** |
| ⚠️ | 1 | 2 | 0 | 1 | 3 | 0 | 1 | 0 | 0 | 1 | **9** |
| ❌ | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 4 | 1 | 1 | **8** |
| *items* | *9* | *8* | *4* | *7* | *9* | *4* | *6* | *5* | *4* | *2* | ***58*** |
