# Nonconformity scores

One section per score in `IMPLEMENTATION_PLAN.md` §1.3. Code: `src/poseconf/conformal/scores.py`
(registry `SCORES`). Tests: `tests/test_conformal_scores.py` unless stated.

## Common to every score

**Shape.** Each score is a pair:

* `score_X(predictions, labels, valid, …, convention) → s` — one score per frame, used on
  calibration frames and to *evaluate* test frames;
* `set_X(predictions, valid, q, …) → set` — the set drawn at test time. It takes **no label**: its
  parameters must all be on a prediction-side allowlist and none may be named like a label
  (`gt`, `label`, `true`, `incl`, `range`, `iou`, `pixel_error`). Enforced by
  `test_set_functions_take_no_labels`.

A set is checked against the truth only through `set.contains(labels…)`, for evaluation. `contains`
reuses the score's own normalised-error helper and the per-frame scales stored in the set. On
**valid** frames, that makes `truth ∈ set(q)` exactly `score ≤ q`, bit for bit, at every q
including ±∞. On **failed** frames, `contains` answers the literal question "is the truth in the
set?": True only when q = +∞ (the set is the whole space). That equals `score ≤ q` under
`answer_required`. It does **not** equal it under `abstain_allowed`, where a failure scores −∞ and
counts as covered by abstaining. So `contains` is never the coverage indicator on its own.
**Coverage must be computed with `metrics.outcomes`** (equivalently, `set.abstain | contains` under
`abstain_allowed`, and `contains` under `answer_required`). `test_set_inverts_score_exactly` (every
score × both conventions) and `test_contains_alone_is_not_abstain_allowed_coverage` pin this.

**Quantile.** `q = conformal_quantile(cal_scores, α)` is the ⌈(n+1)(1−α)⌉-th smallest calibration
score, computed in exact rationals, and `+∞` when that index exceeds n (`split.py`;
`tests/test_conformal_split.py`). Weighted and Mondrian variants are in `weighted.py` and
`mondrian.py`.

**PnP-failure conventions** (CLAUDE.md invariant 6). `valid = False` marks a frame without a point
estimate:

| Convention | Failure score | Set on a failed frame |
|---|---|---|
| `answer_required` | `+∞` | the whole space if q = +∞, otherwise none (a miss) |
| `abstain_allowed` | `−∞` | none: abstention |

Two rules fix what "covered" means (`metrics.outcomes`; `tests/test_conformal_metrics.py`):

* **Answered** means `valid ∧ q > −∞`. The set at q = −∞ is empty, and an empty set counts as an
  abstention. An all-failure calibration set under `abstain_allowed` therefore gives an
  always-abstain predictor.
* **Covered**: under `answer_required`, `s ≤ q`. Under `abstain_allowed`, `¬answered ∨ s ≤ q`.

**Silent failure** means answered and not covered. `outcomes` raises when the scores don't match
the stated convention, so two conventions cannot be mixed in one evaluation.

**What every guarantee here is.** Split CP gives marginal coverage ≥ 1 − α over the draw of the
calibration set and the test frame. It requires calibration and test frames to be exchangeable,
and it is finite-sample. It is not conditional coverage and not a bound on error. Between `val_cal`
and `val_test` exchangeability holds only approximately, because P1 selected its checkpoint on all
of synthetic validation (IMPLEMENTATION_PLAN §0(4)). The Phase 2 Beta-law test measures that. On
`lightbox`/`sunlamp` exchangeability does not hold, and coverage there is a *measured* number with
a Clopper–Pearson interval and n. It is never a guarantee.

**Fitted inputs.** The normalisers `c_R, c_t, c_z, c_xy` are medians of the error components from
`pose_error_components` over `val_tune` solved frames (`median_normaliser`). `g` (A3), Σ̂ (B2/C1)
and σ̂ (C2) are passed in as arrays. They are fitted or predicted elsewhere, and none may use the
frame's label. Fitting any of them on `val_cal`, `val_test` or HIL data breaks the guarantee.

**Units.** Rotations are in radians (reported in degrees). Translations are in metres, in the
camera frame, with z along the optical axis. Keypoints are in full-frame pixels.

---

## A1: pose, rotation ball × translation ball

* **Formula.** `s = max(E_R / c_R, ‖t̂ − t‖ / (‖t̂‖ · c_t))`. Here `E_R = 2·arccos(|⟨q̂, q⟩|)` in
  rad, the same formula as P1, with quaternions renormalised after a tolerance check (`so3.py`). `c_R`
  is in rad; `c_t` is dimensionless. Translation is normalised by the **predicted** range ‖t̂‖, never
  by the GT ‖t‖.
* **Set** (`set_a1`). A geodesic ball `{R : d(R, R̂) ≤ q·c_R}` × `{t : ‖t − t̂‖ ≤ q·c_t·‖t̂‖}`.
  Because the score is a `max`, a single q covers both components simultaneously.
* **Failures.** `+∞` / `−∞` per convention. Under `answer_required` the set is `∞` whenever the
  failure rate exceeds α (IMPLEMENTATION_PLAN §0(2)).
* **Valid when** `val_cal` and test frames are exchangeable, and `c_R, c_t` come from `val_tune`
  only.
* **Breaks when** the normalisers are fitted on calibration or test frames, ‖t‖ replaces ‖t̂‖, or
  failures are dropped.
* **Tests:** `test_a1_formula`, `test_translation_is_normalised_by_predicted_range`,
  `test_set_inverts_score_exactly[A1-*]`, `test_failures_follow_convention[A1]`.

## A2: pose, rotation ball × boresight/lateral cylinder

* **Formula.** `s = max(E_R / c_R, |Δz| / (‖t̂‖ c_z), ‖Δxy‖ / (‖t̂‖ c_xy))`, with `Δ = t̂ − t` in the
  camera frame. z is the optical axis (boresight) and xy is lateral.
* **Set** (`set_a2`). A geodesic ball × a cylinder: half-length `q c_z ‖t̂‖` along z and radius
  `q c_xy ‖t̂‖` in xy. Monocular range error is mostly along the boresight, so the cylinder should
  be tighter than A1's ball laterally.
* **Failures.** As A1.
* **Valid / breaks.** As A1, with `c_z, c_xy` fitted on `val_tune` only. If the target leaves the
  principal-point region, boresight ≈ line of sight is weaker, but coverage still holds; only the
  set's efficiency suffers.
* **Tests:** `test_set_inverts_score_exactly[A2-*]`, `test_failures_follow_convention[A2]`.

## A3: pose, A1 scaled by a confidence-difficulty function

* **Formula.** `s = A1 / g(conf̄)`, where `g > 0` is a monotone fit of pose error on mean keypoint
  confidence, fitted on `val_tune`. The score receives `difficulty = g(conf̄)` per frame. Fitting
  happens outside `conformal/` so this package stays numpy/scipy only.
* **Set** (`set_a3`). A1's radii multiplied per frame by `g(conf̄)`. Both `conf̄` and `g` are
  predictions.
* **Failures.** As A1.
* **Valid when** g is fixed before calibration, using `val_tune` only. Coverage stays marginal:
  better ranking by g gives tighter sets on easy frames, but does not give conditional coverage.
* **Breaks when** g is fitted on `val_cal`, is non-positive (raises), or uses `pixel_error_mean`
  (GT keypoints). P1's `pixel_error_mean` ranks error better than confidence, and it is still
  unusable.
* **Tests:** `test_a3_is_a1_over_difficulty`, `test_set_inverts_score_exactly[A3-*]`.

## B1: keypoints, 11 discs scaled by the predicted box

* **Formula.** `s = max_{k ∈ incl} ‖ŷ_k − y_k‖ / d̂`, in full-frame px divided by the **predicted**
  box diagonal d̂. `incl` contains the keypoints whose GT projection lies inside the full frame. That
  is a label-side mask, applied by the same rule at calibration and evaluation. A frame with no
  included keypoint scores 0, so it is covered for every q ≥ 0.
* **Set** (`set_b1`). Eleven discs, centred on `ŷ_k`, each of radius `q·d̂`. Taking the max over k
  gives simultaneous coverage of every included keypoint, which is the event the PURSE argument
  (Phase 4) needs.
* **Failures.** `valid` is the caller's failure mask. The default, from §1.3, is PnP failure, so
  that Levels A and B share one answer rate. It is decided per table and stated there.
* **Valid when** d̂ comes from the detector, never the GT box.
* **Breaks when** the GT box or GT IoU enters d̂, or the inclusion rule differs between calibration
  and evaluation.
* **Tests:** `test_keypoint_scores`, `test_set_inverts_score_exactly[B1-*]`.

## B2: keypoints, 11 Mahalanobis ellipses from the heatmap moment

* **Formula.** `s = max_{k ∈ incl} √(r_kᵀ Σ̂_k⁻¹ r_k)` with `r_k = ŷ_k − y_k`. Here Σ̂_k is the
  heatmap's second moment, mapped from crop to full frame (`Σ_full = A⁻¹ Σ_crop A⁻ᵀ`; Phase 4).
  Units: Mahalanobis, dimensionless.
* **Set** (`set_mahalanobis`). Eleven ellipses `{y : (y − ŷ_k)ᵀ Σ̂_k⁻¹ (y − ŷ_k) ≤ q²}`. The
  reported size is the largest semi-axis `q·√λ_max` (`KeypointSet.radius_px`).
* **Failures.** As B1.
* **Valid when** Σ̂ is a function of the prediction alone; it needs no calibration of its own,
  because the conformal q absorbs any scale error. Σ̂ must be symmetric positive-definite on every
  valid frame, otherwise the function raises.
* **Breaks when** the crop-to-full-frame mapping uses the GT box, or Σ̂ is regularised using
  residuals from `val_cal`.
* **Tests:** `test_keypoint_scores`, `test_non_pd_covariance_raises`,
  `test_set_inverts_score_exactly[B2-*]`.

## C1: keypoints, B2 with the learned variance head

* **Formula / set.** Identical code to B2 (`SCORES["C1"].score_fn is SCORES["B2"].score_fn`). Only
  Σ̂_k changes: it comes from the Phase 5 heteroscedastic head, mapped to full frame.
* **Failures.** As B1.
* **Valid when** the head is trained on `train` and selected on `val_tune` only; its P1
  coordinates must be bit-identical (invariant 10).
* **Breaks when** head selection or early stopping looks at `val_cal`, `val_test` or HIL.
* **Tests:** `test_registry_is_exactly_the_seven_scores`, `test_set_inverts_score_exactly[C1-*]`.

## C2: pose, normalised by the linearised pose std

* **Formula.** `s = max(E_R / σ̂_R, ‖t̂ − t‖ / σ̂_t)`. Here σ̂_R (rad) and σ̂_t (m) are per-frame
  predicted standard deviations, obtained by propagating C1's Σ̂_k through the PnP Jacobian
  (Phase 4/5). They are passed in as arrays.
* **Set** (`set_c2`). A geodesic ball of radius `q·σ̂_R` × a translation ball of radius `q·σ̂_t`.
  Each frame's set is scaled by its own predicted uncertainty.
* **Failures.** As A1. A frame with no σ̂ (for example, a degenerate Jacobian) must be marked
  invalid, not given a finite placeholder.
* **Valid when** σ̂ is a function of the prediction alone and is positive on every valid frame
  (otherwise the function raises).
* **Breaks when** σ̂ is rescaled using calibration residuals, or failed frames get a finite σ̂
  instead of being marked invalid.
* **Tests:** `test_set_inverts_score_exactly[C2-*]`, `test_failures_follow_convention[C2]`.
