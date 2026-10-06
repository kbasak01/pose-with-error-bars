# `pose-with-error-bars` — Sequential Implementation Plan

Distribution-free uncertainty for the `spacecraft-pose-baseline` (Project 1) pose pipeline:
split-conformal pose and keypoint sets with finite-sample coverage guarantees, a learned
heteroscedastic keypoint head, and an honest measurement of where those guarantees break under
the SPEED+ synthetic → hardware-in-the-loop shift. Single RTX A4000 (16 GB). Builds on P1 at tag
`phase-9-complete` (`0f81b426e0a7a7ec26369fb31ad1ea19161434a1`); never modifies it.

---

## 0. Read this first: the five things that decide whether this project works

**(1) A conformal guarantee is only as good as its exchangeability assumption — and the
headline of this project is the place it fails.** Split conformal prediction guarantees
`P(y_test ∈ C(x_test)) ≥ 1 − α` when calibration and test points are exchangeable. Synthetic
validation → synthetic test satisfies that (approximately; see (4)). Synthetic → `lightbox` /
`sunlamp` does not. The deliverable is not "we have 90 % coverage"; it is a measured
coverage-under-shift curve plus the cheapest fix that buys coverage back. Exactly as P1's
headline was the domain gap, this project's headline is the *coverage gap*.

**(2) PnP failure is the elephant in the calibration set.** P1's best rung (`keypoint_a2`,
predicted box) produces **no pose at all** on 7.0 % of synthetic validation, 46.1 % of
`lightbox` and 87.0 % of `sunlamp` frames (`results/keypoint_a2/*_predicted_crop.json` in P1).
A conformal set must say what it does with a frame that has no point estimate. Two valid
conventions, both implemented, never mixed in one table:

| Convention | Score on PnP failure | What "covered" means | When it is the right one |
|---|---|---|---|
| **answer-required** | `+∞` | the true pose is inside a finite set | a downstream consumer that cannot tolerate abstention |
| **abstain-allowed** | `−∞` | the system either abstains *or* its set contains the truth | GNC with a fallback (IMU propagation, hold, wave-off) — the realistic one |

Under *answer-required*, the quantile is infinite whenever the failure rate exceeds α. Pilot
check on P1's committed **synthetic** sidecar only (2026-10-06, 200 random half-splits,
no test domain touched): at α = 0.05 the set is `∞` because 7.0 % > 5 %; at α = 0.10 it is a
3.89° rotation ball × 1.9 %-of-range translation ball. Under *abstain-allowed* the same α = 0.05
gives 3.16° × 1.5 %. On `lightbox`, *answer-required* coverage is bounded above by the 53.9 %
solve rate no matter what is calibrated. **That is a finding to report, not a bug to fix.**

The safety metric that unifies both: **silent-failure rate** = `P(answered ∧ truth ∉ set)`.
A GNC stack can survive an abstention; it cannot survive a confident wrong set.

**(3) Every score must be computable at test time without the label.** The SPEC metric
normalises translation error by the *ground-truth* range `‖t‖`. A set defined that way cannot be
drawn at test time. Normalise by the *predicted* range `‖t̂‖`; the set is then
`{t : ‖t − t̂‖ ≤ q·‖t̂‖}`. Same for anything else in a score's normaliser: predicted box, predicted
confidence, predicted covariance — never GT box, GT IoU, GT range. The pilot also shows the trap:
P1's `pixel_error_mean` correlates with rotation error better (Spearman 0.57) than
`confidence_mean` does (0.41), but `pixel_error_mean` uses GT keypoints and is unusable.

**(4) The calibration data is not pristine.** P1 selected `best.pt` on all of synthetic
validation (one capped scalar, 75 epochs). Synthetic validation is therefore mildly dependent
on the model — not exchangeable with a truly fresh synthetic draw. There is no unseen synthetic
data, and retraining P1 to carve out a clean calibration split is out of scope. Mitigate and
disclose: (a) all *new* fitting (variance head selection, score normalisers, Mondrian bins)
uses only a fixed `val_tune` subset; (b) `val_cal` and `val_test` are disjoint and never touched
by fitting; (c) the repeated-split coverage distribution (Phase 3) is tested against its
theoretical Beta law — if selection had materially broken exchangeability it would show there.

**(5) Prior art is close; position against it honestly.** Yang & Pavone (CVPR 2023) do
conformal keypoint sets propagated to a pose uncertainty set (PURSE) on LineMOD-Occlusion; Wang et
al. (ICCV 2025) do conformal keypoints + implicit-function-theorem propagation to ellipsoidal pose
regions on LMO and the original **SPEED**. Neither, as far as a search on 2026-10-06 found,
measures coverage across the **SPEED+ synthetic → HIL** shift, treats PnP failure / abstention
explicitly, or compares shift-aware calibration (weighted, group-conditional, few-label
target recalibration) with deployment cost. Those four are this project's contribution. Do not
claim novelty for conformal keypoints or for propagation to pose.

---

## 1. System design (decided up front, so Claude Code isn't improvising)

### 1.1 Architecture

```
                         P1 (frozen, pinned submodule)                        P6 (this repo)
1920×1200 ─▶ detector ─▶ crop ─▶ KeypointNet ─┬─ coords (11×2), conf (11) ──▶ PnP ──▶ (R̂, t̂) or FAIL
                                              │                                       │
                                              ├─ heatmaps ─▶ moment Σ̂ₖ (Level B0)      │
                                              └─ decoder features ─▶ VarianceHead ─▶ Σ̂ₖ (Level C)
                                                                                      ▼
                     ConformalPoseHead (numpy, framework-free, JSON calibration artifact)
          ┌──────────────────────────┬──────────────────────────┬───────────────────────────┐
          │ Level A: pose scores     │ Level B: keypoint discs  │ Level C: Mahalanobis       │
          │ rot ball × trans ball    │ → PURSE membership +     │ ellipses from Σ̂ₖ → pose    │
          │                          │   propagated pose radius │   via linearised Jacobian  │
          └──────────────────────────┴──────────────────────────┴───────────────────────────┘
                                         │
            output: ANSWER(set, α, calibration_id)   or   ABSTAIN(reason)
```

`poseconf.p1_adapter` is the **only** module that imports `speedpose`. Everything under
`poseconf/conformal/` is pure numpy/scipy (no torch), mirroring P1's rule for `geometry/`, because
it is the piece that ports to the ship-deck pipeline.

### 1.2 Data splits (fixed, seeded 20261006, written to `splits/` with SHA-256 manifests)

| Split | Source | Size | Used for | Labels read? |
|---|---|---:|---|---|
| `train` | synthetic/train.json | 47,966 | variance-head training only | yes |
| `val_tune` | 20 % of synthetic/validation | ~2,399 | variance-head selection, score normalisers `c_R, c_t`, Mondrian bin edges, weight-clip choice | yes |
| `val_cal` | 40 % of synthetic/validation | ~4,798 | conformal calibration | yes |
| `val_test` | 40 % of synthetic/validation | ~4,797 | in-distribution coverage | yes (eval only) |
| `{lightbox,sunlamp}_poolA` | 50 % of each HIL domain | 3,370 / 1,395 | (i) *images only* for the weighted-CP domain classifier; (ii) *labels* only in arms tagged `oracle_target_labels` | **tagged arms only** |
| `{lightbox,sunlamp}_poolB` | other 50 % | 3,370 / 1,396 | **evaluation only, for every arm, forever** | eval only |

Synthetic train/validation counts come from P1's dataset audit; confirm against
`results/dataset_audit.json` in Phase 0 rather than trusting this table. `val_cal ∪ val_test`
may be re-split at random R = 1,000 times for coverage distributions; `val_tune` never moves.

### 1.3 Nonconformity scores (the complete list; nothing else gets a table row)

All scores are per frame; `s = −∞` (abstain-allowed) or `+∞` (answer-required) on PnP failure.

| ID | Level | Score | Set at test time | Needs |
|---|---|---|---|---|
| `A1` | pose | `max(E_R / c_R, ‖t̂ − t‖ / (‖t̂‖ c_t))` | geodesic ball radius `q c_R` × translation ball radius `q c_t ‖t̂‖` | sidecar only |
| `A2` | pose | A1 with translation split boresight/lateral: `max(E_R/c_R, |Δz|/(‖t̂‖c_z), ‖Δxy‖/(‖t̂‖c_xy))` | ball × cylinder | sidecar only |
| `A3` | pose | A1 ÷ `g(conf̄)`, `g` = monotone fit of error on mean keypoint confidence on `val_tune` | A1 sets scaled per frame by `g` | sidecar only |
| `B1` | keypoint | `maxₖ ‖ŷₖ − yₖ‖ / d̂` (full-frame px ÷ predicted box diagonal) | 11 discs, radius `q d̂` | dump |
| `B2` | keypoint | `maxₖ √(rₖᵀ Σ̂ₖ⁻¹ rₖ)`, `Σ̂ₖ` = heatmap second moment mapped to full frame | 11 ellipses | dump |
| `C1` | keypoint | B2 with `Σ̂ₖ` from the learned variance head | 11 ellipses | Phase 5 |
| `C2` | pose | A1 normalised by the linearised pose std from C1's `Σ̂ₖ` | per-frame-scaled ball × ball | Phase 5 |

`c_R, c_t, c_z, c_xy` = medians of the respective errors on `val_tune` solved frames. `max` over
keypoints gives *simultaneous* coverage of all 11 (the property the PURSE argument needs).
Keypoints included in the coverage event: those whose GT projection is inside the full frame —
the same rule at calibration and evaluation.

### 1.4 Calibration arms (every table row carries exactly one tag)

| Tag | What it consumes beyond synthetic | Method |
|---|---|---|
| `split` | nothing | standard split CP, finite-sample quantile `⌈(n+1)(1−α)⌉/n` |
| `mondrian` | nothing | group-conditional CP; groups = tertiles of `‖t̂‖` × tertiles of `conf̄`, edges from `val_tune` |
| `weighted_unlabeled_target` | **unlabeled** poolA images | weighted CP (Tibshirani et al. 2019); weights from a logistic domain classifier on frozen P1 encoder features; report effective sample size |
| `oracle_target_labels_n{25,50,100,250,500,1000}` | **labeled** poolA frames | split CP calibrated on n target frames (5 random draws per n). Answers "how many labeled real frames buy coverage back?" |
| `aci_online` *(stretch)* | labels revealed online, random order | adaptive conformal inference (Gibbs & Candès 2021); HIL frames are not a trajectory, so state that the order is synthetic |

### 1.5 Metrics (implement exactly)

- **Coverage** (empirical), with a Clopper–Pearson 95 % interval, per domain × score × arm × convention × α.
- **Set size**: median and p90 rotation radius (deg), translation radius (m and fraction of `‖t̂‖`), keypoint radius (px). Report `∞` as `∞`, never drop it.
- **Answer rate** and **silent-failure rate** `P(answered ∧ not covered)`.
- **Conditional coverage** by GT range tertile, by GT bbox IoU bin (<0.5, 0.5–0.8, >0.8), by confidence quintile — GT quantities are allowed *for slicing the evaluation*, never inside a score.
- **Coverage calibration curve**: empirical vs nominal coverage over α ∈ {0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5}.
- **In-distribution validity test**: coverage over R = 1,000 re-splits vs Beta(n+1−l, l), `l = ⌊(n+1)α⌋` (Angelopoulos & Bates). KS p-value and the mean.
- **Uncertainty ranking**: Spearman(σ̂, error), AURC of the risk–coverage curve when abstaining by σ̂.
- **Heteroscedastic head**: Gaussian NLL on `val_tune`/`val_test`, fraction of residuals inside the 1σ/2σ ellipses (uncalibrated reliability).
- **Weighted CP**: effective sample size `(Σw)²/Σw²`, max normalised weight.
- **Latency**: p50/p90/p99/max of the uncertainty post-process alone and of the full frame, P1 methodology (≥50 warmup, ≥500 timed, sync-bracketed).

### 1.6 Variance head (Level C)

- Input: P1 `keypoint_a2` decoder output before the final 1×1 conv (B, C, 64, 64), **frozen**.
- Head: 3×3 conv (C→64) → ReLU → 1×1 conv (64→3K) giving per-pixel `(log σx, log σy, atanh ρ)`;
  read out per keypoint as the heatmap-weighted average over the soft-argmax window (export-safe:
  multiply + reduce, no `ArgMax`/`TopK`). Σ̂ₖ built from the Cholesky form, `σ` clamped to
  [0.05, 64] crop px.
- Loss: β-NLL (Seitzer et al. 2022), β = 0.5, on residuals of the **stop-gradient** P1 coordinates
  in crop pixels, masked exactly as P1's coordinate loss is.
- Training: synthetic train, P1's A2 augmentation, AdamW 1e-3, 20 epochs, bf16, seed 1337. P1
  measured 78 s/epoch for the full network; the frozen trunk should be faster. Selection on
  `val_tune` NLL only.
- Invariant: P1 point predictions are bit-identical with and without the head (frozen-weight hash
  check + coordinate equality test). The head may change uncertainty, never the pose.

### 1.7 Stretch (only after Phase 8 passes)

- CQR on a quantile version of P1's direct-regression head (`direct_regression_a1`).
- `aci_online`.
- Conformal risk control (Angelopoulos et al. 2022) on capped `E_pose` instead of set coverage.

---

## 2. Phase plan

Each phase ends with an exit gate. Run `/phase-gate N` (skill) before starting the next one.
Long GPU jobs are run by the user; Claude prepares and prints the command.

| Phase | Name | Days | GPU | Exit gate in one line |
|---|---|---:|:-:|---|
| 0 | Scaffold, submodule, environment | 0.5 | – | `make smoke` green in CI with P1 importable |
| 1 | Conformal core + score spec ⚠️ the crux | 1 | – | coverage law reproduced on synthetic data; `conformal-validity-auditor` SOUND |
| 2 | Splits + Level A on P1 sidecars | 0.5 | – | Level A in-distribution coverage within MC error of 1−α |
| 3 | Prediction dump + P1 reproduction gate | 1 | ✔ | dumped solves/errors reproduce P1's committed sidecars |
| 4 | Level B: keypoint sets + propagation | 1.5 | – | joint keypoint coverage valid; propagated radii measured |
| 5 | Level C: variance head | 1.5 | ✔ | P1 coords bit-identical; NLL beats heatmap-moment baseline or the miss is reported |
| 6 | Coverage under shift ⭐ headline | 1.5 | – | full matrix filled from JSON; every arm tagged |
| 7 | `ConformalPoseHead`, ONNX, latency | 1 | ✔ | parity + latency JSON committed; artifact loader refuses mismatched checkpoints |
| 8 | Validation sweep | 1 | – | every `VALIDATION_CHECKLIST.md` line has a verdict |
| 9 | Publication | 0.5–1 | – | `repo-publication-reviewer`: SHIP |

**Total ≈ 9.5 working days.** Fast path (≈5 days): 0 → 1 → 2 → 6 (Level A only) → 9; this needs no
GPU at all because Level A runs on P1's committed sidecars, and still produces the headline
coverage-under-shift result for pose-level sets.

---

### Phase 0 — Scaffold, submodule, environment (0.5 day)

1. `git init pose-with-error-bars`; copy this starter kit in (it holds `CLAUDE.md`, this plan,
   `REPO_SETUP.md`, `VALIDATION_CHECKLIST.md`, `SOURCES.md`, `KICKSTART.md`, `.claude/`).
2. `git submodule add https://github.com/kbasak01/spacecraft-pose-baseline external/spacecraft-pose-baseline`
   then check out tag `phase-9-complete` inside it and commit the pin.
3. Venv per `REPO_SETUP.md`: P1's pinned `requirements.txt` + this repo's additions;
   `pip install -e external/spacecraft-pose-baseline --no-deps` (editable, so P1's
   module-level `REPO_ROOT` resolves to the submodule and its configs/assets load unchanged);
   `pip install -e . --no-deps`.
4. Package skeleton `src/poseconf/` (layout in `REPO_SETUP.md`), `configs/conformal.yaml`
   (draft provided), `Makefile`, `pyproject.toml` (ruff + pytest markers identical to P1),
   CI workflow (CPU, `submodules: recursive`).
5. `poseconf/p1_adapter.py`: the single import boundary. Exposes `load_p1_config(name)`,
   `load_pipeline(...)`, `load_sidecar(run, domain, crop_source)`, `p1_paths()`. Paths to the
   dataset and P1 checkpoints come from `configs/paths.local.yaml` (gitignored), never hardcoded.
6. Checkpoint verification script: SHA-256 of `keypoint_a2/best.pt`, `keypoint_a1/best.pt`,
   `detector/best.pt`, `direct_regression_a1/best.pt` vs the `SHA256SUMS.txt` from P1's
   `phase-9-complete` release; written to `results/p1_checkpoints.json`.

**Exit gate:** `make lint && make test && make smoke` green locally and in CI; `python -c "import
speedpose, poseconf"` works; `results/p1_checkpoints.json` shows four matching hashes (on the
workstation; CI skips it with the `gpu`/`dataset` markers).

---

### Phase 1 — Conformal core and score specification (1 day) ⚠️ the crux

Everything downstream is a call into this module. A wrong quantile index silently produces
coverage 1/(n+1) off nominal, which no plot will reveal at n = 4,798.

1. `poseconf/conformal/split.py`: `conformal_quantile(scores, alpha)` with the
   `⌈(n+1)(1−α)⌉`-th order statistic and explicit `+∞` when that index exceeds n; handles
   `±∞` scores; refuses NaN loudly.
2. `weighted.py`: weighted quantile with the test-point mass at `+∞` (Tibshirani et al. 2019
   eq. for normalised weights), effective sample size, optional clipping (clip value chosen on
   `val_tune` only and recorded).
3. `mondrian.py`: per-group quantiles; groups with n < ⌈1/α⌉−1 return `∞` rather than
   borrowing from neighbours.
4. `so3.py` (numpy): geodesic distance using `2·arccos(|⟨q₁,q₂⟩|)` with clamping, ball
   membership, the convention from P1's `pose_convention.yaml` via the adapter.
5. `scores.py`: every score in §1.3 as a pure function `(predictions, labels) → s` plus its
   inverse `(predictions, q) → set` — the inverse must not take labels as an argument (enforced by
   signature and a test).
6. `docs/SCORES.md`: one section per score — formula, test-time set, what makes it valid, what
   breaks it.
7. Tests (no dataset, no GPU):
   - quantile index vs a brute-force enumeration for n ∈ {1, 2, 9, 10, 99, 4798};
   - coverage over 2,000 resamples of Gaussian / heavy-tailed / heteroscedastic synthetic data
     falls in the Beta law's 99 % band; repeat for weighted CP under a known covariate shift where
     the true weights are known;
   - cross-check against MAPIE ≥ 1.0 `SplitConformalRegressor(prefit=True)` and TorchCP split
     regression on the same residuals: identical quantiles (oracles only — neither is imported by
     `src/`);
   - `±∞` conventions: an all-failure calibration set gives `∞` (answer-required) and an
     always-abstain predictor (abstain-allowed).

**Exit gate:** all tests green; `conformal-validity-auditor` verdict **SOUND**; `docs/SCORES.md`
complete for all seven scores.

---

### Phase 2 — Splits and Level A on P1's committed sidecars (0.5 day)

No GPU. P1 committed per-sample `e_r`, `e_t`, `t_pred`, `success`, `confidence_mean`,
`bbox_used` for every run × domain × crop arm. That is enough for Level A.

1. `scripts/make_splits.py`: seeded manifests in `splits/` (filename lists + SHA-256), including
   HIL poolA/poolB. Committed. Re-running must be byte-identical (test).
2. `scripts/run_level_a.py --run keypoint_a2 --crop-source predicted_crop`: fits `c_R, c_t,
   c_z, c_xy` and `g(conf̄)` on `val_tune`, calibrates on `val_cal`, evaluates on `val_test`
   **only** (HIL evaluation waits for Phase 6), both conventions, α grid; R = 1,000 re-splits.
3. Output `results/level_a/keypoint_a2_predicted_crop_synthetic.json` + the per-split
   coverage array as `.npz`.

**Exit gate:** for A1–A3 under both conventions, mean coverage over re-splits within
`[1−α, 1−α+1/(n+1)] ± 3·MC s.e.`, KS test vs Beta not rejected at 0.01 — or the deviation is
documented in `docs/DECISIONS.md` as evidence about (4) in §0. Set sizes recorded.

---

### Phase 3 — Prediction dump and P1 reproduction gate (1 day, GPU)

Level B/C need per-keypoint predictions and heatmap moments that P1 did not store.

1. `scripts/dump_predictions.py --run keypoint_a2 --domain {synthetic,lightbox,sunlamp}
   --crop-source {predicted_crop,gt_crop}` writes `dumps/<run>/<domain>_<arm>.npz` (gitignored):
   filename, bbox_pred, bbox_gt, affine, kp_pred_crop (11×2), kp_pred_full (11×2),
   kp_gt_full (11×2), in_frame mask, confidence (11), heatmap moments (11×2×2, full-frame px),
   heatmap peak/entropy (11), pooled encoder feature (512, fp16, for the domain classifier),
   PnP success/q/t/inliers/reprojection RMSE, GT q/t.
   Uses P1's own `PosePipeline` stages and `engine.pose.solve_single` so a frame cannot be solved
   differently from P1. Heatmaps themselves are not stored (≈1 GB per domain); a 256-frame subset
   is, for figures.
2. `scripts/check_p1_parity.py`: compare the dump's success flags and `e_r`, `e_t` with P1's
   committed `*_samples.npz`.
3. Commit a distilled `results/dump_summary.json` (counts, solve rates, parity stats) — the dumps
   themselves never enter git.

**Exit gate:** success-flag agreement ≥ 99.9 % per domain; median |Δe_r| ≤ 1e-6 rad on frames
both solved; solve rates match P1's JSON to the reported precision. P1 documented a bistable
soft-argmax under TF32 — if parity misses, run with TF32 disabled before anything else, and
report rather than loosen. Then `p1-parity-auditor` verdict **REPRODUCED**.

---

### Phase 4 — Level B: keypoint sets and propagation to pose (1.5 days)

1. Calibrate B1 and B2 (synthetic only) and evaluate in-distribution, same protocol as Phase 2.
2. **PURSE membership** (exact): the GT pose is in the PURSE iff every in-frame GT keypoint
   projection is inside its set — which is precisely the joint-keypoint coverage event. Test that
   equivalence numerically on fixtures.
3. **Pose-space extent**, two estimators, both labelled for what they are:
   - *linearised*: Jacobian of projection w.r.t. (rvec, t) at the PnP solution
     (`cv2.projectPoints` returns it), keypoint ellipses → pose covariance → rotation/translation
     radii. Fast; an approximation, not a bound.
   - *sampled*: M = 256 keypoint configurations drawn uniformly inside the sets, iterative PnP
     warm-started at the estimate, max geodesic / translation deviation. An **inner**
     approximation of the PURSE extent — a lower bound on the true radius. Offline only.
4. Compare Level-B-propagated radii against Level A radii at equal α. Expectation from the
   literature: B is looser. Measure, don't assume.
5. Use `pose-geometry-verifier` (from P1) on the Jacobian and sampling code.

**Exit gate:** joint keypoint coverage valid in-distribution (Phase 2 criterion); equivalence test
green; `results/level_b/*.json` with both radius estimators and their runtime.

---

### Phase 5 — Level C: learned heteroscedastic keypoint head (1.5 days, GPU)

1. `poseconf/models/variance_head.py` + `VarianceKeypointNet` wrapping a frozen P1 `KeypointNet`
   (§1.6). Forward returns P1's `(coords, confidence, heatmaps)` unchanged plus `cov_chol (B,K,3)`.
2. `scripts/train_variance_head.py`; `make train-var-smoke` (overfit one batch: NLL must fall
   below the analytic floor for the batch's residuals) and `make train-var`.
3. Tests: frozen-parameter hash identical before/after training; coords bit-identical vs bare P1
   model on fixtures; covariance positive-definite for all outputs; export-safe op set.
4. Evaluate on `val_tune` then (after selection is frozen) `val_test`: NLL, 1σ/2σ ellipse
   reliability, Spearman(σ̂, error) vs heatmap-moment baseline (B2's Σ̂) and vs `conf̄`.
5. Calibrate C1 and C2; in-distribution coverage per Phase 2 protocol.
6. `uncertainty-head-diagnostician` on the run.

**Exit gate:** bit-identity tests green; diagnostician verdict TRUSTWORTHY; C1 coverage valid
in-distribution; whether the learned Σ̂ beats heatmap moments on NLL and on set size is
*reported either way*.

---

### Phase 6 — Coverage under shift (1.5 days) ⭐ the headline result

1. `scripts/run_shift_matrix.py`: for each score ∈ {A1,A2,A3,B1,B2,C1,C2} × arm (§1.4) ×
   convention × α grid × domain ∈ {synthetic `val_test`, `lightbox_poolB`, `sunlamp_poolB`} ×
   crop arm ∈ {predicted_crop, gt_crop(**oracle**)}: one JSON per cell under `results/shift/`.
2. Weighted CP: domain classifier = logistic regression on the 512-d pooled encoder features,
   synthetic `val_tune` vs poolA images, 5-fold CV AUC reported. AUC near 1.0 means near-disjoint
   supports: weights explode, ESS collapses, sets go to `∞` — an informative outcome.
3. Oracle few-label curve: coverage and set size vs n for n ∈ {25,50,100,250,500,1000}, 5 draws
   each, mean ± range.
4. Conditional coverage slices (§1.5) for the `split` arm on every domain.
5. `scripts/make_tables.py` → `results/TABLES.md`; `scripts/make_figures.py` → headline
   figures: (a) coverage-vs-nominal curves per domain, (b) silent-failure rate per arm × domain,
   (c) few-label recovery curve, (d) set size vs coverage scatter, (e) a gallery of predicted sets
   drawn on P1's wireframe for 8 `val_test` frames (synthetic imagery only in the README; HIL
   imagery carries CC BY-NC-SA — see `SOURCES.md`).
6. `shift-study-runner` builds tables; `conformal-validity-auditor` re-checks every arm's
   tagging and that poolB was never used for fitting.

**Exit gate:** matrix complete or every empty cell explained; no oracle row in the same table
section as a non-oracle row; leakage audit clean.

---

### Phase 7 — `ConformalPoseHead`, ONNX, latency (1 day, GPU)

1. `poseconf/conformal/head.py`: `ConformalPoseHead.from_json(path)`; `predict(estimate,
   cov=None) → PoseSet | Abstain`. JSON artifact carries α, convention, score ID + version, n_cal,
   quantile(s), normalisers, split-manifest SHA, P1 checkpoint SHA-256, poseconf git SHA. The
   loader **refuses** a checkpoint whose hash differs.
2. Export `VarianceKeypointNet` to ONNX (opset 17, fp32/fp16) — outputs `coords, confidence,
   cov_chol`. Parity on ≥200 fixed inputs; tensor gate 1e-4 fp32 with TF32 off. P1's keypoint
   parity gate is **inherited unmet** (bistable soft-argmax); the variance outputs are gated on
   their own, and the README must say both.
3. Latency (reuse `speedpose.export.bench` methodology): head-on vs head-off keypoint stage on
   ORT CUDA/TensorRT; conformal post-process alone; linearised propagation; sampled propagation
   M ∈ {16, 64, 256}. Full-frame p50/p99 with and without uncertainty.
4. `onnx-parity-auditor` (from P1, extended).

**Exit gate:** parity + latency JSON committed under `results/`; artifact round-trip and
mismatched-hash refusal tests green.

---

### Phase 8 — Validation sweep (1 day)

Walk `VALIDATION_CHECKLIST.md` line by line, verdict + evidence for each, written into
`docs/LIMITATIONS.md` in P1's format (met / met-with-caveat / unmet). Run
`eval-reproducibility-auditor` with the guarantee-language extension and `conformal-validity-auditor`
once more on the final tree. Two seeds for the variance head (1337, 2026) — report spread.

**Exit gate:** every line has a verdict; no critical finding open.

### Phase 9 — Publication (0.5–1 day)

README in P1's style: headline coverage-under-shift figure and table above the fold, every number
traced to `results/`, a "what the guarantee does and does not say" box, limitations, attribution.
`repo-publication-reviewer`; tag `phase-9-complete`; calibration JSONs + variance-head checkpoint +
ONNX as release assets with `SHA256SUMS.txt`.

---

## 3. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Dump does not reproduce P1 (TF32 bistability, crop path drift) | medium | high | Phase 3 gate; TF32 off; use P1's own stage functions; stop rather than recalibrate on a different model |
| Submodule path assumptions (`REPO_ROOT`, relative configs) | medium | medium | editable install of the submodule; adapter resolves every path; smoke test exercises it |
| Domain classifier AUC ≈ 1 → weighted CP useless | high | low | report ESS and `∞` sets; this *is* the finding about covariate shift severity |
| Selection-on-validation breaks exchangeability visibly | low | medium | Beta-law test in Phase 2; disclose either way |
| Sunlamp poolB has ~180 answered frames → wide CIs | high | medium | Clopper–Pearson everywhere; never quote sunlamp coverage without its CI and n |
| Variance head learns nothing beyond heatmap moments | medium | low | it is an ablation; report it |
| Sampled PURSE radius is slow (PnP tail 50.9 ms p99 in P1) | high | low | linearised estimator for deployment; sampling offline only |
| Over-claiming "guaranteed safe" language | medium | high | claim-trace skill + auditor word list; §0(1) |

## 4. Calendars

- **Full (≈9.5 days):** D1 P0+P1(start) · D2 P1 · D3 P2+P3(start) · D4 P3 · D5–6 P4 · D6–7 P5 · D8–9 P6 · D9–10 P7 · D10–11 P8+P9.
- **Fast (≈5 days, no GPU):** D1 P0+P1(start) · D2 P1 · D3 P2 + Phase 6 for Level A · D4 tables/figures · D5 P9 trimmed.

## 5. How this feeds the ship-deck work

- `poseconf/conformal/` is framework-free: it takes scores and predictions, not models. Swap the
  Tango wireframe for a deck-marker wireframe and the same calibration code produces deck-pose sets.
- The abstain-allowed convention and the silent-failure metric are exactly what a landing
  decision needs: commit only when the set is answered and smaller than the deck's tolerance;
  otherwise wave off.
- The few-label recalibration curve answers the practical question for the real deck: how many
  labeled frames in the new visibility condition (fog/spray/glare) are needed before the set is
  trustworthy again.
- The calibration-artifact contract (hash-locked to a checkpoint) is the deployment pattern for a
  Jetson: a model update without recalibration fails loudly instead of silently mis-covering.
