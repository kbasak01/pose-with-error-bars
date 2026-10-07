# Decisions

One entry per non-obvious choice: date, decision, alternatives considered, reason. This file is the
project's memory across Claude Code sessions. Newest at the bottom.

---

## 2026-10-06 — Depend on P1 through a pinned submodule, not a fork or a copy

**Decision.** `external/spacecraft-pose-baseline` at `phase-9-complete`, installed editable;
`poseconf.p1_adapter` is the sole import boundary.
**Alternatives.** Fork P1 and add the head there; copy the needed modules.
**Reason.** P1's published numbers stay valid only if its code does not move; a fork invites
edits, a copy loses provenance. The adapter confines the coupling to one file.

## 2026-10-06 — Two PnP-failure conventions, never mixed

**Decision.** Every score is computed under `answer_required` (failure → `+∞`) and
`abstain_allowed` (failure → `−∞`); tables state which.
**Alternatives.** Calibrate on solved frames only and report coverage on solved frames.
**Reason.** "Solved only" silently conditions on an event whose probability collapses under shift
(93.0 % synthetic → 13.0 % sunlamp in P1). Both conventions are valid split CP; together with the
silent-failure rate they make the trade explicit.

## 2026-10-06 — Translation normalised by predicted range

**Decision.** Translation scores divide by `‖t̂‖`, not the SPEC metric's `‖t‖`.
**Reason.** A set must be drawable without the label.

## 2026-10-06 — Own the conformal core; MAPIE and TorchCP as test oracles

**Decision.** Implement split/weighted/Mondrian CP in numpy; use MAPIE ≥ 1.0 and TorchCP only in tests.
**Reason.** The core is small, must be framework-free for the ship-deck port, and independent
implementations are worth more as cross-checks than as dependencies.

## 2026-10-06 — Pilot on P1's synthetic sidecar only

**Decision.** The planning pilot (split CP on `keypoint_a2` synthetic validation, 200 half-splits)
used no HIL data. Result: α = 0.05 `answer_required` gives `∞` (7.0 % failures); α = 0.10 gives a
3.89° × 1.9 %-of-range set; `abstain_allowed` α = 0.05 gives 3.16° × 1.5 %. Mean keypoint
confidence has Spearman 0.41 with `E_R` on solved frames.
**Reason.** HIL coverage must not inform any design choice before Phase 6.

## 2026-10-06 — TorchCP dropped as a test oracle; MAPIE remains

**Decision.** `torchcp` is not installed. Phase 1's split-CP cross-check uses MAPIE 1.5.0 plus the
brute-force order-statistic enumeration; what (if anything) replaces the TorchCP oracle is a
Phase 1 decision.
**Alternatives.** (a) torchcp 1.2.1, which requires `transformers` and resolves only with numpy < 2,
so P1's numpy 2.4.4 pin would have to move. (b) torchcp 1.2.0, which declares no dependencies but
whose top-level `__init__` imports `classification` (needs `torchsort`, a compiled extension),
`graph` (`torch_geometric`) and `llm` (`transformers`), so it cannot be imported in this
environment either. (c) Stubbing those modules in `sys.modules` inside tests.
**Reason.** P1's pins are not ours to move (invariant 1). An oracle that needs import hacks is not an
independent check.

## 2026-10-06 — Extras installed with P1's resolution as constraints

**Decision.** `requirements-extra.txt` is installed with `-c <P1 frozen env>` locally and
`-c requirements-ci.txt` on CI. Exact versions are pinned in the file.
**Reason.** Without constraints a transitive dependency of an extra could silently upgrade torch or
numpy and move P1's numbers.

## 2026-10-06 — `poseconf/provenance.py` added to the layout

**Decision.** One module outside REPO_SETUP §4 holds the hashing, git SHA, timestamp,
SHA256SUMS parsing and `inf`-preserving result-JSON writer that every results script needs.
**Alternatives.** Duplicating them per script; putting them in `p1_adapter` (wrong boundary — they
are not P1-specific) or in `conformal/` (not conformal).
**Reason.** One implementation of the provenance block, testable without a dataset.

## 2026-10-06 — Result kind `checkpoints`

**Decision.** `results/p1_checkpoints.json` uses `schema: poseconf.result.v1` with a new
`kind: "checkpoints"`; it carries per-run local/release SHA-256 and `match`, plus provenance. It
records `<run>/best.pt` relative paths only, never machine paths.
**Reason.** The results-provenance kinds list predates the Phase 0 checkpoint check.

## 2026-10-06 — Release manifest location is a local-paths key

**Decision.** `p1_release_sums` in `configs/paths.local.yaml` names the `phase-9-complete`
`SHA256SUMS.txt`; `--sums` overrides it. The user places the file; nothing downloads it. Manifest
entries are matched as `<run>/best.pt`, `<run>_best.pt` or `<run>.pt`; anything else fails loudly,
because P1's repository does not record the release asset names.

## 2026-10-06 — Adapter rewrites P1 paths instead of creating `data/` in the submodule

**Decision.** `load_p1_config` makes P1's committed-file paths absolute inside the submodule. With
local paths it points `dataset_root`/`camera_json` at `speedplus_root` and `output_root` at
`p1_runs`; without them it *removes* the dataset keys (plus `crop_cache`, `keypoint_labels`,
`output_root`), so P1 code that needs them raises `KeyError`. `crop_cache` and `keypoint_labels`
stay removed until Phase 3 shows they are needed.
**Alternatives.** A `data/speedplus` symlink inside the submodule (writes into P1, invariant 1).
**Reason.** A config that silently resolves to a nonexistent directory fails far from its cause.

## 2026-10-06 — Sidecar box columns differ by crop arm

**Observation.** SOURCES.md lists `bbox_used`, `bbox_gt`, `bbox_iou` as available in every sidecar.
In P1's committed `keypoint_a2/synthetic` sidecars, `gt_crop` has none of the three,
`pipeline_gt_crop` has `bbox_used`/`bbox_gt` but no `bbox_iou`, and `predicted_crop` has all three.
The conditional-coverage-by-IoU slice (§1.5) therefore exists for `predicted_crop` only.
`tests/test_adapter.py` encodes this.

## 2026-10-06 — `dataset` pytest marker covers checkpoints too

**Decision.** Marker names are identical to P1 (`slow`, `gpu`, `dataset`); the `dataset` description
now says "SPEED+ dataset or P1 checkpoints via configs/paths.local.yaml", because here those come
from local paths, not P1's `data/speedplus` symlink.

## 2026-10-06 — Release manifest naming verified: `<run>-best.pt` only

**Decision.** Supersedes the matching rule in "Release manifest location is a local-paths key".
P1's `phase-9-complete` `SHA256SUMS.txt` (fetched on request with
`gh release download phase-9-complete -p SHA256SUMS.txt`, manifest only, no weights) names
checkpoints `<run>-best.pt` — none of the three provisional spellings. `match_release_entry` now
accepts exactly `<run>-best.pt`; the guesses were removed rather than kept alongside.
**Reason.** Once the real naming is known, extra accepted spellings are only ways to match the wrong
entry.

## 2026-10-06 — TorchCP replacement: MAPIE + exact brute force (closes the open item above)

**Decision.** There is no third-party replacement for TorchCP. Phase 1 has two oracles:
(a) MAPIE 1.5.0 `SplitConformalRegressor(prefit=True, conformity_score="absolute")` on residuals
from a constant-zero estimator (`tests/test_conformal_oracles.py`, `slow`);
(b) a brute-force enumeration of the definition in `fractions.Fraction`
(`tests/test_conformal_split.py`). MAPIE refuses `n ≤ 1/α`, where it raises "too low" instead of
returning `+∞`. The test records that refusal rather than forcing the two to agree there.
**Alternatives.** Run TorchCP in a throwaway numpy<2 venv and commit a fixture; transcribe TorchCP's
rule into the test.
**Reason.** User choice (Phase 1 planning). A transcription is not independent, and the fixture
route needs network installs for little extra assurance beyond the exact enumeration.

## 2026-10-06 — α handled as an exact rational

**Decision.** `split.rational_alpha` = `Fraction(α).limit_denominator(10**6)`. The index
`⌈(n+1)(1−α)⌉` and the Beta parameter `l = ⌊(n+1)α⌋` are computed exactly from it.
**Reason.** In floats, `ceil(10 * (1 - 0.1)) = ceil(9.000000000000002) = 10`, one order statistic
too high (conservative by 1/(n+1)), and `floor(10 * 0.1)` has the same problem.
`test_float_traps_are_pinned` pins both cases.

## 2026-10-06 — Weighted CP mass comparison uses a 1e-12 relative tolerance

**Decision.** `weighted_conformal_quantiles` accepts cumulative mass ≥ (1−α)(W+w)·(1 − 1e-12).
**Reason.** Float cumulative sums of equal weights can land one ulp under the exact threshold, and
then the reduction to split CP fails. `test_equal_weights_reduce_to_split` pins exact equality for
n ∈ {1, 2, 9, 10, 99, 4798} × the α grid × three weight scales.

## 2026-10-06 — Empty set = abstention; answered/covered semantics

**Decision.** A frame is answered when `valid ∧ q > −∞`. Covered means `s ≤ q` (`answer_required`)
or `¬answered ∨ s ≤ q` (`abstain_allowed`). Silent failure means answered and not covered. A failed
frame's set is the whole space at q = +∞ and none otherwise. `metrics.outcomes` raises if the
scores don't match the stated convention.
**Reason.** q = −∞ (failures ≥ k under `abstain_allowed`) gives an empty set for every solved
frame. Treating that as an abstention is what "always-abstain predictor" means, and it keeps
`set.abstain` and `outcomes` identical. `test_set_inverts_score_exactly` checks they agree.

## 2026-10-06 — Score inputs fitted elsewhere are passed in as arrays

**Decision.** A3 receives `difficulty = g(conf̄)` and C2 receives `sigma_R`, `sigma_t` per frame.
B2/C1 receive `cov` in full-frame px². `conformal/` fits none of them.
**Alternatives.** Fit the isotonic `g` inside `scores.py`.
**Reason.** `conformal/` stays numpy/scipy (no sklearn) for the ship-deck port, and the role of the
fitting data (`val_tune`) stays visible at the call site, not hidden inside a score.

## 2026-10-06 — A2 boresight = camera z axis

**Decision.** A2 splits `Δt` into `|Δz|` along the optical axis and `‖Δxy‖` lateral, as §1.3
writes it, not along the line of sight `t̂/‖t̂‖`.
**Reason.** It matches the plan's formula. SPEED+ keeps the target near the principal point, where
the two nearly coincide, and coverage holds either way; only set shape is affected.

## 2026-10-06 — Keypoint-score failure mask is the caller's; empty inclusion scores 0

**Decision.** B1/B2/C1 take a `valid` mask like the pose scores. Per §1.3 ("all scores … on PnP
failure") the default is the PnP-failure mask, so Levels A and B share one answer rate. Each
table states its mask. A frame whose GT keypoints all fall outside the image scores 0, which is
vacuously covered for any q ≥ 0.
**Reason.** "All included keypoints inside their sets" is vacuously true for an empty inclusion
set. Scoring −∞ instead would confuse it with an abstention.

## 2026-10-06 — Result kind `validity`

**Decision.** `results/validity/conformal_core_synthetic.json` uses
`schema: poseconf.result.v1`, `kind: "validity"`. It holds per-case Beta-law statistics for the
synthetic study that `scripts/run_conformal_synthetic_validity.py` and the tests share (same
seeds, via `poseconf.synthetic_validity`). The `/coverage-check` results on SPEED+ will reuse the
kind under `results/validity/<score>_<convention>.json`.

## 2026-10-06 — `product_groups` takes explicit level counts

**Decision.** Mondrian product groups are built with an explicit radix (`n_levels`) and never one
inferred from the data.
**Reason.** An inferred radix can encode the same (range, confidence) tuple differently in
calibration and test batches, which silently assigns test frames to the wrong group's quantile.

## 2026-10-06 — Phase 1 audit: carried items (Should-consider, not fixed in Phase 1)

The `conformal-validity-auditor` returned SOUND at `0243839`. These items are deferred on purpose,
each to the phase where it starts to matter.
- **S1 (before `make level-a`).** On SPEED+ re-splits, coverage is measured on a finite `val_test`
  of size m. It therefore follows Beta-Binomial(m; n+1−l, l)/m, which is wider than the Beta law
  (about 1.4× at n = m ≈ 2400). Re-splits of a single pool are also not independent draws. When
  failure atoms make q = +∞, coverage is identically 1. Phase 2 must state the reference law and
  how it handles atoms before it runs.
- **S2.** At n ∈ {100, 1000} the Beta-law study cannot tell `np.quantile(..., method="higher")` apart
  from the correct index. The brute-force test (n = 4798, random α) does catch it. Add an n where
  the two differ (e.g. 101).
- **S3.** The weighted-CP equal-weight reduction is exact only up to n ≲ 5e4. Above that it is off
  by one order statistic, in the conservative direction. Use `cum[-1] + w_test` or `fsum` for the
  total if larger n ever occurs.
- **S4.** `rational_alpha` rounds α < 5e-7 to 0 and α > 1 − 5e-7 to 1. α = 1 then yields the max
  score, which is wrong. Unreachable from the config grid; it should raise instead.
- **S5 (before Phase 6).** `set_*` and `metrics.outcomes` accept only a scalar q. Weighted and
  Mondrian arms need per-frame q.
- **S6 (Phase 2 onwards).** Every SPEED+/HIL result JSON must carry its arm tag. Synthetic
  true-weight rows must never be labelled `weighted_unlabeled_target`.
- **Correction.** The TorchCP entry above says MAPIE "refuses `n ≤ 1/α`". MAPIE's actual rule is
  `n < max(1/α, 1/(1−α))`, evaluated in floats (`mapie/utils.py`). For example, n = 100 at α = 0.01
  is answered. The oracle test's acceptance condition is looser than this, which is harmless.
- **Reporting note.** A keypoint frame with no included keypoint scores 0 and is vacuously covered.
  Phase 4 tables report how many such frames there are.

## 2026-10-07 — Split rule: sorted names, per-domain seed streams, round-half-down sizes

**Decision.** `poseconf.data.splits.assign_splits` sorts filenames, permutes them with
`default_rng(SeedSequence([20261006, key]))`, key = synthetic 0 / lightbox 1 / sunlamp 2, and takes
consecutive blocks. Every split but the last gets `N·f` rounded half-down (exact `Fraction`); the last
takes the remainder. This reproduces IMPLEMENTATION_PLAN §1.2 exactly: 2,399 / 4,798 / 4,797;
3,370 / 3,370; 1,395 / 1,396. Manifests list names sorted, LF, trailing newline; `splits/SHA256SUMS`
pins them and `load_split` refuses a file whose hash disagrees. Results carry
`split_manifest_sha256` = SHA-256 of `SHA256SUMS`.
**Alternatives.** Floor everywhere (2,398 / 4,797 / 4,799); one global stream for all domains.
**Reason.** Matches the plan's table; per-domain streams mean a domain can be regenerated alone.
Byte-identity depends on numpy's `Generator.permutation`, pinned by P1 (numpy 2.4.4).

## 2026-10-07 — Manifest name sources (no label is read)

**Decision.** Synthetic validation names come from the `filename` column of P1's committed
`keypoint_a2/synthetic_predicted_crop_samples.npz` (so CI can regenerate them without the dataset;
a `dataset` test checks they equal `synthetic/validation.json`). HIL names come from a directory
listing of `<speedplus_root>/{lightbox,sunlamp}/images/*.jpg` (`p1_adapter.list_image_filenames`):
no `test.json`, no HIL sidecar, no pixels. CI regenerates HIL from the committed poolA ∪ poolB
(`--hil-names-from-manifest`), which verifies the permutation, not the listing; the listing check is
a `dataset` test. Counts are asserted against P1's committed `results/dataset_audit.json`.
**Reason.** User choice (Phase 2 planning): nothing on lightbox/sunlamp is opened before Phase 6.

## 2026-10-07 — Synthetic labels for Level A via `p1_adapter.load_synthetic_labels`

**Observation.** P1's sidecars store `e_r`, `e_t`, `q_pred`, `t_pred` but no GT pose; A2 needs the
vector Δt. **Decision.** `load_synthetic_labels(split, paths)` wraps P1's `load_split` for
`synthetic/{train,validation}` only and has no domain argument. `run_level_a.py` re-derives `e_r`,
`e_t` from predictions + labels and fails if they differ from P1's committed values by > 1e-4
(measured max 8.1e-8 rad, 4.8e-7 m), which proves the filename join and the quaternion order.

## 2026-10-07 — A3 difficulty g(conf̄): decreasing isotonic fit of the A1 score

**Decision.** g = sklearn `IsotonicRegression(increasing=False, out_of_bounds="clip")` of the A1
score (with the val_tune c_R, c_t) on `confidence_mean`, solved `val_tune` frames only (2,251).
Stored as knots and evaluated with `np.interp` (numpy-only, serialisable for the Phase 7 artifact).
No floor: every knot is > 0 (asserted). **Alternatives.** Fit E_R alone; a log-linear fit.
**Reason.** A3 = A1 / g is then "A1 relative to its expected value at this confidence", the
standard normalised-residual construction; isotonic makes no shape assumption beyond monotonicity.

## 2026-10-07 — Re-split reference law (closes S1) and what the re-split test can and cannot detect

**Decision.** The primary reference is `conformal.metrics.resplit_coverage_law`: conditional on the
pool, a uniformly random cal/test partition makes the covered count X_tb Beta-Binomial(m; k, n+1−k)
(negative-hypergeometric rank law; exact for distinct scores). PnP-failure atoms are exact too:
answer_required, X = m iff k + X_tb > N − F (q = +∞); abstain_allowed, X = m iff k + X_tb ≤ F
(q = −∞, always abstain). Verified by exhaustive enumeration and Monte Carlo
(`tests/test_conformal_metrics.py`). KS uses D = max over the integers of |F_emp − F| with the
continuous Kolmogorov p-value. Under a discrete null that p is an *upper bound* on the exact p, so
the test is conservative (the auditor's simulation of the size under the exact null confirmed it is
below nominal; that simulation is not a committed result). Verdicts: VALID (KS p ≥
0.01 and mean within 3 MC s.e. of the law mean), DEGENERATE (q = +∞ or −∞ in every draw — coverage
≡ 1 by construction, reported with set = ∞, not a validity test), DEVIATES. R re-splits of one pool
are iid draws from this conditional law: the null law is exact, the p-value is conservative.
DEGENERATE additionally requires the law to predict the all-infinite quantile (KS not rejected), so
a bug that returned q = +∞ everywhere would read DEVIATES.
**Deviation from the gate's literal wording (documented as the gate allows).** "KS vs Beta not
rejected at 0.01" fails for every non-degenerate cell (max p 1.4e-7), because Beta(n+1−l, l) ignores
the finite test half: at n = 4,798, m = 4,797, α = 0.10 the Beta sd is 0.00433 and the re-split law
sd 0.00612 (×1.41); observed/law sd ratios are 0.96–1.04. The literal mean band
[1−α, 1−α+1/(n+1)] ± 3 s.e. holds for every non-degenerate cell. It fails for the 9 DEGENERATE
cells (answer_required, α ∈ {0.01, 0.02, 0.05}: 7.2 % failures in the pool > α, so q = +∞ and
coverage is 1 with an ∞ set) — the plan's §0(2) prediction, reported, not fixed.
**Limits of this test** (sharpened by the Phase 2 `conformal-validity-auditor`).
- *It cannot see §0(4), and nothing using validation frames can.* P1 chose `best.pt` with one scalar
  over all 11,994 validation frames, a symmetric function of the pool; given that choice the
  validation frames stay exchangeable *with each other*. What selection may break is exchangeability
  with a fresh synthetic draw, testable only with fresh data. IMPLEMENTATION_PLAN §0(4)(c)
  overstates this ("if selection had materially broken exchangeability it would show there"); the
  plan is a root planning doc and is not edited — this entry corrects it.
- *It says nothing about data roles.* Scores are fixed across re-splits, so normalisers fitted on
  `val_cal` would pass too. Data roles rest on the code paths and the `split-leakage-auditor`.
- *Resolution.* At n ≈ 4,800, R = 1,000, 3 MC s.e. ≈ 0.0006 ≈ 3/(n+1): an off-by-one quantile index
  is below what the run resolves. Index correctness rests on the brute-force / enumeration unit
  tests. What the run adds is that the production vectorised path, on real arrays with real failure
  atoms, matches the exact law (observed/law sd 0.96–1.04).

## 2026-10-07 — `level_a` result layout and the `level_a` config block

**Decision.** `results/level_a/<run>_<crop>_synthetic.json` (`kind: level_a`) holds one provenance
block, the val_tune fit, the join check, the re-split protocol and `rows` (score × convention × α),
each row carrying every results-provenance field plus a `resplits` summary. Per-draw arrays live in
the sibling `_resplits.npz`, whose SHA-256 is in the JSON; `scripts/coverage_check.py` refuses an
npz that does not match. `results/validity/<score>_<convention>.json` (`kind: validity`) is derived
from those two files without recomputation. `configs/conformal.yaml` gained `level_a:` (score list,
split roles, join-check tolerances, gate thresholds `ks_min_p: 0.01`, `mean_band_se: 3`).
The same R = 1,000 permutations (seed 1337) serve every score, convention and α (common random
numbers), so cells are correlated with each other; each cell's test is valid on its own.

## 2026-10-07 — Phase 2 audits: carried items

- **Leakage W1 (before Phase 6).** `p1_adapter.load_sidecar` accepts lightbox/sunlamp and returns
  label columns; the only guard today is that Phase 2 callers pass `"synthetic"` (tested by a spy in
  `tests/test_level_a.py`). Phase 6 must add a code-level guard (poolB-evaluation or `oracle_*` tag,
  or a label-free loader for the weighted arm).
- **Leakage W2.** The weighted / domain-classifier / oracle paths do not exist yet; re-audit in Phase 6.
- **Validity audit (SOUND).** Must-fix items closed: the `ks_against_pmf` docstring had the
  conservativeness direction inverted (fixed; the JSON now says "upper bound on the exact p"); KS
  wording above; PROGRESS guarantee wording; the fixed split's position in the re-split law is now
  stored per row (`resplits.fixed_split_law_cdf`). Should-consider items adopted: DEGENERATE requires
  law agreement, the sd ratio is null when degenerate, and an abstain_allowed Monte Carlo test
  covers a mixed q = −∞ branch.
- **The committed `val_cal → val_test` split runs low for A3.** `fixed_split_law_cdf` is 0.039
  (abstain_allowed, α = 0.10), 0.025 (abstain_allowed, 0.15), 0.064 (answer_required, 0.15) and
  0.043 (answer_required, 0.20); their Clopper–Pearson intervals lie below 1 − α. The cells share
  one split, so they are correlated, and are consistent with the re-split law. Any README sentence
  quoting a fixed-split coverage cites `fixed_split_law_cdf` beside it. The split is not changed
  (invariant 3).

## 2026-10-07 — Phase 3 dump calls P1's stages in P1's batch and chunk order

**Decision.** `poseconf.data.dump.run_dump` calls `PosePipeline.prepare_frame`, `detect_boxes`,
`crop_one`, `normalise_crop` and `keypoints`, then `engine.pose.solve_single` (through
`p1_adapter.solve_frames`), in the order and batch size P1's `_evaluate_through_pipeline` uses:
detect and keypoint batches of `train.batch_size` = 48, one PnP pass over the whole domain.
`solve_frames` reproduces `solve_many`'s chunking exactly: `min(num_workers, cpu_count)` chunks
from `np.linspace`, chunk i seeded `cv2.setRNGSeed(1337 + i)`, sequential below 256 frames.
**Alternatives.** Call `solve_many` itself (needs GT poses, so it is not label-free on HIL, and it
drops the rotation matrix); solve sequentially with one seed.
**Reason.** RANSAC draws from OpenCV's process-global RNG, so a frame's solve depends on its chunk.
`tests/test_dump.py::test_solve_frames_equals_p1_solve_many` pins bit-equality with `solve_many`
on 320 frames in 4 chunks. An uncroppable box is carried through exactly as P1 does (blank crop,
identity affine, solved, then charged as "detector box was degenerate").

## 2026-10-07 — Dump `gt_crop` is P1's `pipeline_gt_crop`; tagged `oracle_gt_box` on HIL

**Decision.** The dump's `gt_crop` arm feeds the tight GT box through the pipeline, which is P1's
`pipeline_gt_crop` code path; its parity reference is `*_pipeline_gt_crop_samples.npz`, not P1's
crop-cache `gt_crop`. On lightbox/sunlamp its input box is a test label, so the arm's tag is
`oracle_gt_box` and it reads GT boxes under that tag. The box column is named `bbox_used`
(P1's name) rather than the plan's `bbox_pred`, because on this arm it is not a prediction.

## 2026-10-07 — HIL labels in Phase 3: strict split, label-free parity on poolA

**Decision (user choice, Phase 3 planning).** Prediction files are label-free. Ground truth goes
to separate label files: `synthetic_labels.npz`, and per HIL domain `<d>_labels_poolA.npz` and
`<d>_labels_poolB.npz`, split by the committed manifests. `load_dump_labels` refuses poolA unless
the tag starts with `oracle_`; poolA labels are written under the tag `oracle_target_labels`, the
only arms that will read them. Parity recomputes `e_r`/`e_t` on synthetic and HIL **poolB** only.
On HIL poolA the gate uses the label-free bound angle(q̂_dump, q̂_P1) ≥ |e_r(q̂_dump) − e_r(q̂_P1)|
(triangle inequality on SO(3)), which is the stronger check. `p1_adapter.load_sidecar` now returns
label-derived HIL columns only for poolB frames or an `oracle_*` tag, and `p1_result_counts` reads
only solve counts from P1's results JSON. This is the code-level guard leakage item W1 asked for,
on the sidecar and label paths; Phase 6 still re-audits the weighted/oracle paths (W2).
**Alternatives.** Read HIL labels for every frame under a documented exemption; defer HIL dumps.
**Reason.** Invariant 4, without weakening the gate.

## 2026-10-07 — Heatmap moments: the soft-argmax refine distribution

**Decision.** `heatmap_cov_*` is the second central moment of `relu(h)^2 · window / sum`, the exact
distribution P1's two-pass `SoftArgmax2d` takes its coordinate from (locate with `relu^8`, Gaussian
window σ = 4 hm px). Both centroids go through the head's own `_centroid`, and the dump raises if
the mean departs from P1's coordinate by more than `dump.moment_mean_tol_crop_px` on any non-empty
channel. Covariance maps hm px → crop px (× stride²) → full frame (`L Σ Lᵀ`, L = linear part of
P1's inverse affine). `heatmap_entropy` is the Shannon entropy (nats) of the whole normalised
`relu(h)`. `heatmap_peak` is the raw max and equals P1's confidence by P1's definition (tested).
A channel with no positive activation decodes to the origin in P1. It gets `heatmap_empty = True`
and a zero covariance, and B2 must treat it explicitly.
**Alternatives.** The moment of the unwindowed map; a fitted Gaussian.
**Reason.** B2's ellipse should describe the spread of the distribution that produced the point,
centred on that point. The window shrinks the variance (precisions add: σ² = 1/(1/σ_h² + 1/σ_w²)),
which `test_moment_covariance_of_a_known_gaussian` pins.

## 2026-10-07 — Decoder features for a 256-frame subset only

**Decision (user choice).** Full decoder maps (input to the final 1×1 conv, 256×64×64, fp16) and raw
heatmaps are stored for a fixed 256-frame subset per domain (`default_rng(1337)`, sorted, the same
frames in both arms) in `<domain>_<arm>_subset.npz`. The pooled 512-d encoder feature is stored
for every frame. Hooks are forward hooks returning None. `test_hooks_leave_p1_outputs_bit_identical`
checks that coordinates and confidence are unchanged.
**Reason.** 2 MB per frame for all frames is ~90 GB with no Phase 4–6 consumer. Phase 5 runs the
frozen trunk live.

## 2026-10-07 — `p1_keypoint_labels` local path; result kind `dump_summary`

**Decision.** `configs/paths.local.yaml` gains the optional `p1_keypoint_labels`, P1's Phase 2d
label products. It closes the "crop_cache/keypoint_labels stay removed until Phase 3" note above:
labels are needed (frame order, GT boxes, keypoint inclusion), and the crop cache is still not.
`results/dump_summary.json` uses `kind: dump_summary`, one cell per domain × arm, with the gate
criteria per cell and the dumps' SHA-256 and runtime in provenance. Dumps never enter git.

## 2026-10-07 — I run the Phase 3 GPU dump myself, on the user's instruction

**Decision.** CLAUDE.md says long GPU jobs are run by the user. For Phase 3 the user explicitly asked
me to generate the commands and run them. The dump ran as a background job with TF32 disabled
(`cuda.matmul.allow_tf32 = cudnn.allow_tf32 = False`) and `cudnn.benchmark = False`. The flags are
recorded in every dump's meta and in `dump_summary.json`.

## 2026-10-07 — Phase 3 audits: carried items

`p1-parity-auditor`: REPRODUCED (6/6 cells). It flagged its own import grep as loose, and a
precise `^\s*(from|import) speedpose` grep confirms `p1_adapter.py` is the only importer.
`split-leakage-auditor`: no critical findings.
- **Closed.** (W4) `results/dump_summary.json` is committed with the gate commit. (W5) New tests pin that
  `eval_frames` decompresses only `filenames` and that the `predicted_crop` arm reads no GT box.
- **L1 (Phase 6).** `dumps/keypoint_a2/<hil>_labels_poolA.npz` exists, as the recorded strict-split
  decision specifies (written under `oracle_target_labels`). `load_dump_labels` guards it, but the
  file itself is plain npz in a gitignored directory. Every Phase 6 consumer must go through
  `load_dump_labels`, and the auditor re-checks that (with W2).
- **L2 (Phase 6).** The HIL `gt_crop` prediction dumps are oracle-conditioned on every frame:
  `bbox_used` is the GT box, and poses come from GT crops. `load_dump` has no tag guard. Phase 6
  consumers must check `meta["tag"]` / `meta["oracle"]` and keep these out of non-oracle and
  `weighted_unlabeled_target` arms.
- **L3 (note).** `load_eval_labels` parses the full HIL label JSON and arrays before slicing to the
  permitted rows. Only the slice is returned, so nothing leaks, but the guarantee is in the slicing.

## 2026-10-07 — OpenCV in `conformal/propagate.py` only

**Decision (user choice, Phase 4 planning).** `poseconf.conformal.propagate` imports `cv2`
(`projectPoints` for the Jacobian, `solvePnP` for sampled propagation, `Rodrigues`). No other
`conformal/` module may: `test_cv2_only_in_propagate` enforces this, and the torch-family ban is
unchanged. The projection and its visibility rule stay numpy, as a port of P1's hand-rolled
`project_points`, which P1 wrote because `cv2.projectPoints` mirrors points behind the camera.
**Alternatives.** Keep `conformal/` numpy/scipy and put the cv2 calls in `engine/level_b.py`;
use an analytic numpy Jacobian and a numpy Gauss–Newton PnP.
**Reason.** The ship-deck port needs OpenCV for PnP anyway. Keeping the propagation in one module
keeps it auditable. CLAUDE.md's "pure numpy/scipy" line still holds for every other conformal
module.

## 2026-10-07 — B2: an empty heatmap channel is an unconstrained keypoint

**Decision (user choice).** `score_mahalanobis` / `set_mahalanobis` take an optional `unconstrained`
(n, K) mask, which for B2 is `heatmap_empty`. Such a keypoint's set is the whole image. It is left out
of the joint max, it needs no positive-definite covariance, and it drops out of the PURSE and the
propagation. `radius_px` is `∞` on its frame.
**Alternatives.** Treat the frame as a failure (±∞), which would break the shared answer rate with
Level A. Floor the covariance, which leaves the keypoint at P1's decoded origin and makes the frame
effectively uncovered.
**Reason.** The network said "not found", so the set says "anywhere". The mask is a prediction, so
the score stays label-free. The synthetic `predicted_crop` dump has 0 empty channels, so no Phase 4
number depends on this rule. It is fixed now, before Phase 6 looks at any HIL frame.

## 2026-10-07 — Level B label keypoints: P1's projection of the label pose

**Observation.** P1's stored keypoint label product (`kp_gt_full`) differs from P1's own
`project_points` of the label pose by up to 0.003 px on synthetic `val_cal ∪ val_test` (exact values
per split in `checks.label_projection` of `results/level_b/*_synthetic.json`). The
`pose-geometry-verifier` traced the gap to P1's label product being built from un-normalised
quaternions. The visibility masks are identical.
**Decision.** B1/B2 score against `propagate.project(quat_to_matrix(q_gt), t_gt)`, which is
bit-identical to P1's `project_points`. `include` is P1's stored `in_frame`, and the join raises if
it differs from the projection's visibility. The run fails if the gap to the stored product exceeds
`level_b.label_projection_max_px` (0.01).
**Reason.** PURSE membership projects candidate poses. Scoring against the same projection makes
"label pose ∈ PURSE ⇔ joint score ≤ q" exact rather than true up to a 0.003 px boundary band.

## 2026-10-07 — Pose-space extent: residual-aware linearised inner/outer and sampled inner

**The PURSE is unbounded.** A pose that projects no constrained keypoint into the frame (behind the
camera, far off-axis) meets no constraint, so it is a member at every q ≥ 0. Both auditors
demonstrated this. Its global extent is therefore π in rotation and ∞ in translation. Every radius
we report describes the PURSE *near the PnP estimate*, and the result definitions say so. This
leaves "label pose ∈ PURSE ⇔ joint score ≤ q" and the PURSE's coverage untouched.
**Alternative (not taken; a score-definition change).** Take the inclusion set from the estimate's
visibility rather than the label's. That bounds the PURSE and stays label-free, but it changes
B1/B2 away from the §1.3 inclusion rule.

**Linearisation.** The tangent perturbation is the left one, `R = Exp(ω) R̂`, so ‖ω‖ is the
geodesic angle. OpenCV's d/d(rvec) is mapped by `J_l(r̂)⁻¹`; translation is unchanged. U is the set
of keypoints visible under the PnP estimate and not unconstrained, which is label-free. Unit shapes
are S_k = Σ̂_k (B2) or d̂² I (B1).
- Linearise around the PnP residual r_k = π_k(θ̂) − ŷ_k, which is predictions only. Then
  Σ_k‖S_k^-½(r_k + J_k δ)‖² = (δ−δ*)ᵀH(δ−δ*) + c, with δ* = −H⁻¹g and c ≥ 0.
- *Inner* radius √(‖o‖² + (q² − c)λ). It is attained in E_in = {sum ≤ q²}, which lies inside the
  linearised PURSE. It is NaN when c > q².
- *Outer* radius ‖o‖ + √((|U|q² − c)λ). It bounds E_out = {sum ≤ |U|q²}, which contains the
  linearised PURSE. It is NaN when c > |U|q², i.e. the linearised PURSE is empty.
- Both are labelled "approximation, not a bound". |U| < 3 or a numerically singular H gives ∞.
- The first version assumed zero residual (inner q√λ, outer √|U|·q√λ). The `pose-geometry-verifier`
  measured that the estimate lies outside its own PURSE on about 10 % of headline frames, where
  that "inner" radius is not inner. The residual-aware form above reduces to the zero-residual
  version when r = 0, and it replaced it before any number was committed. The fraction of
  estimates inside their own PURSE, and residual/q, are reported for every row.

**Sampled.** Draw M = 256 configurations uniform in each set (y = ŷ + q L u). Solve each with
`cv2.solvePnP(SOLVEPNP_ITERATIVE, useExtrinsicGuess=True)` from the estimate, and keep only PURSE
members.
- It is an inner approximation of the PURSE's extent near the estimate. A frame where no sample is
  accepted (mostly frames whose estimate is outside its PURSE) gets NaN and is counted in `n_nan`.
- It runs at α = 0.10, both conventions, on all answered `val_test` frames, seeded per frame with
  `SeedSequence([1337, score, convention, row])`.
- The 50-frame pilot measured about 35 ms per frame, so the full run takes about 1.5 min on 16
  workers. That is below the 20-minute threshold, so no subset is used.

**Measured ball coverage.** We report E_R ≤ r_R ∧ ‖Δt‖ ≤ r_t on answered `val_test` frames, with a
Clopper–Pearson interval, the answer rate, `n_nan`, `n_inf` and a NaN-as-uncovered variant. It is a
measurement and carries no guarantee.

**Alternatives.** A convex-program maximum over the intersection of cylinders (exact for the
linearised set, but needs an SOCP solver); χ²-scaled radii, which presume a Gaussian that conformal
sets do not have.

## 2026-10-07 — Level B data roles, re-split reuse and result layout

**Decision.** Level B's valid mask is PnP success, so it shares Level A's answer rate. B1/B2 fit
nothing, so `val_tune` is not read. The re-split protocol is Phase 2's (R = 1,000, seed 1337,
`resplit_coverage_law`, VALID/DEGENERATE/DEVIATES). `level_a.resplit_draws` gained `size_factor`
and `summarise_draws` gained `size_name`; their defaults leave Level A's output unchanged.
Outputs:
- `results/level_b/<run>_<crop>_synthetic.json` (`kind: level_b`, rows score × convention × α with
  `purse_agreement`) and `_resplits.npz`;
- `_propagation.json` (`kind: level_b_propagation`, new kind), with definitions, linearised and
  sampled radii, runtimes, and a Level A comparison. The comparison reads
  `results/level_a/keypoint_a2_predicted_crop_synthetic.json` and records its SHA-256.

The gate fails (exit 1) on any DEVIATES or on PURSE agreement below 1.0.

## 2026-10-07 — Phase 4 audits: carried items

`pose-geometry-verifier`: VERIFIED, no must-fix. `conformal-validity-auditor`: SOUND, no critical
findings; its three must-fix items are closed.
- **Closed (must-fix).**
  - PURSE unboundedness is now stated in `propagate.py`, the result `definitions` and `docs/SCORES.md`.
  - `measured_ball_coverage` now carries a CI, the answer rate, `n_total`, `n_nan` and `n_inf`.
  - The label-projection number above is corrected to 0.003 px.
- **Closed (should-consider).**
  - The linearisation is residual-aware (entry above).
  - `estimate_in_purse` and `residual_over_q` are reported per row.
  - `n_vacuous` is reported per row: valid frames whose included keypoints are all unconstrained,
    scored 0.
  - `n_inf` is counted.
  - The comparison note says the sampled medians exclude `n_nan`.
  - `projection_geometry` refuses a P1 convention with `transpose_rotation: true`.
  - The projection-parity test is bit-exact.
- **Carried to Phase 6: B1 vs B2 on empty heatmap channels.** B2 treats an empty channel as
  unconstrained. B1 does not: P1 decodes the channel to the crop origin, so a B1 frame with an
  included empty keypoint is almost surely uncovered. Both are valid split CP. Synthetic has 0 empty
  channels, so no Phase 4 number depends on it. On HIL the two scores will diverge for this reason
  alone. Whether B1 should share the rule is a user decision before Phase 6; `n_vacuous` must be
  read beside B2's HIL coverage.
- **Follow-up `pose-geometry-verifier` on the residual-aware linearisation: VERIFIED.**
  - Inner ≤ brute-force linearised extent ≤ outer held on every frame of its 400-frame sample.
  - The inner radius is a loose lower estimate and the outer is the closer one. The verifier's
    measured ratios are in its report, not in a committed file.
  - Inner medians exclude the high-residual frames where inner is NaN.
  - The outer radius can be finite when the linearised PURSE is empty.
  - The definitions now say all of this. No table may use "linearised inner" as the size of the
    pose set.
- **Notes.**
  - The 100 % PURSE agreement holds by construction: same projection path, same error helper. It
    confirms frame order, quaternion order and mask plumbing. The independent evidence is the exact
    `in_frame` match and the ≤ 0.003 px gap to P1's stored label product.
  - The npz column `*_median_kp_px` is filled from `ResplitDraws.median_rot_deg`. The value is
    right (`size_factor = 1`); only the field name is generic.
  - `so3.UNIT_NORM_TOL` (1e-3) is looser than P1's 1e-5. That is harmless for P1 outputs, which
    are unit to 1e-15.

## 2026-10-07 — B1 shares B2's empty-channel rule (closes the Phase 4 carried item)

**Decision (user choice, after the Phase 4 gate).** `score_b1` / `set_b1` take the same optional
`unconstrained` mask as B2, and Level B passes `heatmap_empty` to both scores. On HIL, B1 and B2
therefore no longer differ merely because a channel is empty.
**Effect on committed numbers.** None: the synthetic `predicted_crop` dump has 0 empty channels.
`results/level_b/` was regenerated at `55ad03b`, which contains the change. Every coverage, quantile,
verdict and re-split summary is unchanged; the scoped `conformal-validity-auditor` (SOUND) compared
the files with provenance removed and found them equal.
**Still required in Phase 6.** Read `n_vacuous` and `n_frames_with_unconstrained_keypoint` beside
any HIL B1/B2 coverage.

## 2026-10-07 — `scripts/make_figures.py` and the Level B set-overlay gallery

**Decision.** `make figures` (`--figure all`) runs a registry of figures. It skips a figure whose
inputs are absent, with a message, and raises when the figure is named explicitly. The first entry
is `level_b_overlay`, which writes `assets/level_b_overlay_<score>.png`.
- It shows 8 seeded answered `val_test` frames: 5 whose PnP estimate is inside its own PURSE and 3
  outside.
- Each frame shows the keypoint sets at q from the committed `level_b` JSON (α = 0.10,
  `abstain_allowed`), the true keypoints, and the estimate's projection.
- A set that the estimate's keypoint leaves is drawn bold.
- Colours are the reference palette's dark-surface categorical slots 1–3, each with its own
  marker shape. The palette validator could not be run (no `node` on the workstation); the slots
  are the reference palette's pre-validated ones.
- The caption carries the SPEED+ CC BY-NC-SA 4.0 notice.
- Unconstrained keypoints get their own marker, so on HIL they do not silently disappear.
**Reason.** This is the visual check both geometry reviews asked for. Sets offset from the
wireframe, or far too large, would show here first.


## 2026-10-07 — Variance head: readout, parametrisation, frozen trunk

**Decision.** `VarianceHead` is a 3×3 conv (256→64), ReLU, then a 1×1 conv (64→33). It predicts
`(log σx, log σy, atanh ρ)` per heatmap pixel and keypoint.
- **Readout.** Each keypoint's parameters are averaged under the exact distribution P1's
  soft-argmax refines over, `relu(h)^2 · window / Σ`. That is the same distribution as
  `heatmap_moments` (tested), so Σ̂ describes the point P1 returned. The ops are multiply, exp and
  reduce, so the exported graph has no ArgMax, TopK or NonZero (tested).
- **Parametrisation.** σ is clamped to [0.05, 64] crop px and |ρ| ≤ 0.99. The output is the lower
  Cholesky factor `(σx, ρσy, σy√(1−ρ²))`, so Σ̂ is PD for every input.
- **Initialisation.** The final conv starts at zero weight with bias log 2 px.
- **Empty channel.** An empty heatmap channel has zero readout weight, which gives σ = 1, ρ = 0.
  That is still PD. C1 marks the keypoint unconstrained (B2's rule), so the value is never used.
- **Frozen trunk.** `VarianceKeypointNet.forward` calls P1's own `forward` under `no_grad`. It
  reads the decoder input through a pre-hook that it removes after every call. `train()` keeps the
  trunk in eval mode, so BatchNorm buffers never move. `p1_state_hash` hashes every state-dict
  tensor. Training checks the hash before and after the run, and `load_variance_net` refuses a
  checkpoint whose P1 SHA-256 or state hash differs.
**Alternatives.**
- Plain per-keypoint regression from pooled features. This would discard the spatial
  correspondence with the heatmap.
- An unconstrained 2×2 output with an eigenvalue clamp. This is not export-friendly.
- Softplus σ. We chose clamped log σ so the clamp fractions can be audited directly.

## 2026-10-07 — Variance-head training: numerics, loss, data, selection

**Decision.**
- **Numerics.** The trunk runs under P1's own autocast (`runtime.amp_dtype` = bf16; the script
  checks that it matches), exactly as `PosePipeline.keypoints` runs it. TF32 is off. Training
  residuals are therefore the inference residuals. This corrects the plan's draft, which said
  "trunk fp32".
- **Loss.** 2-D β-NLL with β = 0.5. The weight is `stopgrad(det Σ^{β/2}) = (l0·l2)^β`, the 2-D
  analogue of `σ^{2β}`. The loss is masked by P1's `batch["mask"]` (visible ∧ in crop) and reduced
  with P1's `_masked_mean` through the adapter.
- **Data.** P1's `SpeedPlusCrop` on synthetic `train`, with A2, served from P1's crop cache. That
  cache is a new optional local path, `p1_crop_cache`, and is read-only. Optimiser settings follow
  P1: AdamW 1e-3, wd 0.01, ExponentialLR 0.95, clip 1.0, batch 48, seed 1337, 20 epochs.
- **Selection.** We keep the epoch with the lowest plain NLL on `val_tune`, served as P1's a0 eval
  crops (GT box from the cache, margin 1.3). `val_cal`, `val_test` and HIL are never loaded; the
  script raises if a `val_cal`/`val_test` name is served. Selection crops are GT-box crops, while
  the Level C numbers use predicted-box crops. This difference is disclosed, not corrected.
- **Overfit gate** (`make train-var-smoke`). One fixed A2 batch, 400 steps. The "analytic floor"
  is the mean NLL of the best homoscedastic zero-mean Gaussian for that batch's masked residuals
  (its MLE, `½(2 + log det S) + log 2π`). The head must fall below it.
**Note.** The A2 training residuals are heavy-tailed. The first batch's homoscedastic floor is
about 7.8 nats, while `val_tune` NLL is about 3, so σ̂ is fitted to a harder distribution than the
clean validation one. Conformal calibration absorbs the overall scale. The head-quality report
gives NLL both raw and after one global rescaling fitted on `val_tune`.

## 2026-10-07 — Variance sidecar instead of rewriting the parity-gated dump

**Decision.** `scripts/dump_variance.py` writes `<domain>_<arm>_vhead.npz`. It re-runs only P1's
keypoint stage, on the dump's own boxes, in the dump's batch order and size. The head reads the
decoder features through `keypoint_hooks` under the pipeline's autocast.
- The script raises unless every affine and every P1 coordinate equals the Phase 3 dump bit for bit.
- The sidecar records the base dump's SHA-256 and the head checkpoint's SHA-256.
- All domains × arms are produced now. The outputs are predictions only, and saves a Phase 6 GPU
  run. Phase 5 opens only the synthetic one.
**Alternatives.** Re-run the full dump with the head (detector, PnP) and re-gate parity; or store
decoder features for every frame (~90 GB).

## 2026-10-07 — C1 and C2 data roles

**Decision.**
- **C1** is B2's code with `cov_key="vhead_cov_full"` (`level_b.join_dump`), using B2's
  empty-channel rule. Its valid mask is PnP success.
- **C2's σ̂.** σ̂_R = √λ_max((H⁻¹)_ωω) and σ̂_t = √λ_max((H⁻¹)_tt). H uses C1's Σ̂ₖ at the PnP
  estimate, over keypoints visible under the estimate and constrained. This is Phase 4's q-free
  linearisation and uses predictions only.
- **Undefined σ̂.** A solved frame with undefined σ̂ (|U| < 3 or singular H) is invalid for C2. It
  abstains or scores +∞ by convention, and its count is reported per row (`n_solved_sigma_undefined`).
- **`val_tune` in Level C.** It is read only for the head-quality NLL rescaling factor and the σ̂
  ranking diagnostic. It never feeds a score.
- **Re-splits** share Level B's seed, so C1 and B2 see the same R permutations.

## 2026-10-07 — Variance-head run `vhead_a2_s1337`: diagnostician verdict

**`uncertainty-head-diagnostician`: TRUSTWORTHY, as a ranking signal and conformal normaliser. σ̂ is
not a calibrated Gaussian.**
- **How the verdict was reached.** The agent's own shell calls were blocked, so its first pass was a
  provisional SUSPECT: frozen trunk not re-verified. I then ran its recommended check.
  - Both test files passed: 24 tests, including the real-checkpoint GPU bit-identity test.
  - `load_variance_net(best.pt)` matched the P1 SHA-256 and the state hash.
  - `torch.equal` of coordinates, confidences and heatmaps against bare P1 held under bf16 autocast
    on 192 real `val_tune` crops.
  - The agent re-judged that evidence and returned TRUSTWORTHY.
- **Caveats it attached.** These are carried into the Phase 5 report.
  - Raw ellipse reliability on `val_tune` is 0.529 / 0.891 at 1σ / 2σ, against the 2-D Gaussian
    0.393 / 0.865. It is reported as measured.
  - The selected epoch (15) is a flat-minimum pick: epochs 13–20 are within about 0.01 nats of it.
  - Train NLL is about 0.85 nats above `val_tune`. The gap is stable, so it is not overfitting; A2 is
    the plausible cause, but this was not verified.
- **Values.** The values are in [`results/level_c/variance_head_training.json`](../results/level_c/variance_head_training.json)
  (`diagnostics_val_tune`).

## 2026-10-07 — Phase 5 audits: carried items

`conformal-validity-auditor` on the Level C work (HEAD `9dee943`): **SOUND**, no critical findings.
- **Closed (must-fix).**
  - *Set paths never take labels; nothing enforced it.* The set and σ̂ paths receive a
    `KeypointFrames` that also carries labels. `test_c1_c2_sets_and_sigma_ignore_labels` now
    permutes and perturbs `q_gt`, `t_gt`, `y_gt` and `include`. It checks that σ̂, the C2 radii and
    the C1 radii stay bit-identical. The auditor ran the same check on the real dumps.
  - *Unlabelled reliability numbers in `variance_head_training.json`.* The file gains `definitions`
    ("uncalibrated Gaussian reliability, not conformal coverage"), `crop_source:
    gt_box_crop_cache_a0` and `oracle: false`. It also gets an `annotation` saying these are
    labelling-only additions; no value changed. The training script now emits them itself.
- **Closed (should-consider).**
  - `test_c2_sigma_undefined_frames_are_failure_atoms_in_resplits` checks that σ̂-undefined frames
    are failure atoms in the re-splits. A PnP-only mask raises. On synthetic, 0 solved frames have
    an undefined σ̂, so C2's atom count equals Level A's.
- **Carried.**
  - *Beta KS.* It rejects in every cell (the sd is about 1.41× too narrow at n ≈ m), as documented
    since Phase 2. Write-ups cite the law KS only.
  - *answer_required, α ≤ 0.05.* These cells are reported as "set = whole space", not as coverage 1.
  - *Validity JSON provenance.* `results/validity/*.json` overwrite `poseconf_git_sha` with the
    coverage-check run's sha. The source result's sha is recoverable through
    `source_result_sha256`. A separate field is a cleanup for Phase 8.
  - *Disjointness of training and calibration frames.* `train` is disjoint from
    `val_cal`/`val_test` because SPEED+ ships separate official splits. No runtime check enforces
    it. Phase 8 can add a manifest-level assertion.

## 2026-10-07 — Shift-matrix result layout: one JSON per domain × crop × arm tag

**Decision (user choice, Phase 6 planning).** `results/shift/<run>_<domain>_<crop>_<arm>.json`
(`kind: coverage`) holds `rows` over score × convention × α. Every row is self-contained and
carries the full results-provenance field set. Each `oracle_target_labels_n<n>` is its own arm tag,
so it gets its own file. `results/shift/index.json` lists every expected domain × crop × arm cell
with its file or the reason it is empty. `make_tables.py` and the gate read the index, and
`run_shift_matrix.py` exits 1 on any unexplained cell.
**Alternatives.** One file per score × arm × convention × α × domain × crop (about 4,300 files);
one file per domain × crop, which would mix oracle and non-oracle rows.

## 2026-10-07 — Shift matrix reads the dumps for every level; per-crop fits

**Decision.**
- **Frames.** All frames (A1–C2) come from the Phase 3 dumps plus the Phase 5 variance sidecar,
  not P1's sidecars, so every level sees one frame set on every domain × crop.
- **Fitting.** Each crop source is its own pipeline. Normalisers, g(conf̄), Mondrian edges and
  slice edges are fitted on that crop's synthetic `val_tune`, and calibration uses that crop's
  synthetic `val_cal`.
- **Fit check.** The `predicted_crop` refit is checked against the committed Phase 2 fit
  (`index.json`, `checks.predicted_crop_fit_vs_phase2`).
  - The two are not bit-equal: P1's sidecar stores `q_pred`, `t_pred` and `confidence_mean` as
    float32, while the dump stores float64.
  - The first draft compared g's isotonic knots at rtol 1e-6 and failed on a run that was not
    committed.
  - g is now compared as a function on the `val_tune` confidences, at rtol 1e-4, a float32-level
    tolerance. The measured differences are in the same `checks` record
    (`rel_diff`, `g_max_rel_diff_on_val_tune_conf`).
  - Synthetic `split` rows reproduce the Phases 2/4/5 `val_test` counts (pinned by
    `test_shift_matrix_end_to_end` for A1).
- **Scope.** `keypoint_a1`, the config's `secondary_runs`, was never dumped, so it is outside the
  matrix.

## 2026-10-07 — Shift-matrix arms: Mondrian failure group, weighted logits, oracle draws

**Decision.**
- **Per-frame q.** Each arm returns one quantile per test frame. `metrics.outcomes` accepts an
  (m,) q, which closes Phase 1 item S5; the scalar behaviour is unchanged and pinned by tests.
- **`mondrian`.**
  - Groups are predicted-‖t̂‖ tertiles × mean-confidence tertiles, with edges from `val_tune`
    solved frames.
  - Frames without a point estimate for the score form a tenth group: for C2, a solved frame with
    an undefined σ̂ also falls there.
  - That group's quantile is +∞ under `answer_required` (the whole-space set) and −∞ under
    `abstain_allowed` (abstain). Both count as covered, and the other nine groups are identical.
    So Mondrian rows are the same under both conventions, by construction; the result
    `definitions` say so.
  - Alternative rejected: pooling failures into a range group. A failed frame has no t̂.
- **`weighted_unlabeled_target`.**
  - **Classifier.** Standardised L2 logistic regression (C = 1, max_iter 5000) on P1's pooled
    512-d encoder features: synthetic `val_tune` (source) vs `<hil>_poolA` (target), both
    `predicted_crop`.
  - **Validation.** 5-fold stratified CV AUC (seed 1337).
  - **Weights.** The model is refit on every training row. Weights are
    `exp(logit − c)·n_s/n_t`, with one shift c = the max logit over `val_cal` ∪ poolB. A common
    scale cancels in every normalised mass, so the quantiles are those of Tibshirani et al.
    without overflow when AUC ≈ 1 (`test_logit_shift_leaves_weighted_quantiles_unchanged`).
  - **Fixed in advance.** The hyperparameters and the clip (none) were set in config before any
    HIL number existed, and are not tuned.
  - **Disjointness.** Training rows are asserted disjoint from `val_cal`, `val_test` and every
    poolB, keyed on (domain, filename), because filenames repeat across domains.
- **`gt_crop` × weighted on HIL.** Not run, and explained in the index (Phase 3 item L2). The
  features of a GT-box crop are label-conditioned. `require_label_free_dump` raises if a
  non-oracle arm reads an oracle dump.
- **`oracle_target_labels_n<n>`.**
  - For n ∈ {25, 50, 100, 250, 500, 1000}, 5 draws without replacement from poolA, every frame
    kept (failures included). The seed is `SeedSequence([20261007, DOMAIN_KEYS[domain], n, draw])`.
  - Normalisers and σ̂ stay the synthetic fits.
  - **Row summary.**
    - The top-level row gives the mean over draws, plus min and max.
    - `coverage_ci95` is the envelope of the per-draw Clopper–Pearson intervals.
    - `set_size` is each statistic's median over draws.
    - Every draw is listed in full.
- **Synthetic.** Only `split` and `mondrian` run. The weighted and oracle arms have no target
  domain there and are explained in the index.
- **Slices (§1.5).** These are for the `split` arm at the headline α only, to keep files small:
  - GT-range tertiles and confidence quintiles, with edges from `val_tune`;
  - GT-bbox IoU bins (<0.5, 0.5–0.8, >0.8) on `predicted_crop` only (on `gt_crop` the IoU is 1).
- **`coverage_given_answered`.** It is reported beside every marginal coverage and labelled as a
  slice, not a conformal quantity. Under `abstain_allowed` an abstention counts as covered, so a
  near-nominal marginal coverage at a low answer rate can hide a poor answered-frame coverage.

## 2026-10-07 — Shift figures and P1's wireframe edges

**Decision.**
- **Figures.** `make figures` gains five Phase 6 entries:
  - `shift_coverage_vs_nominal_<convention>`, `shift_silent_failure`, `shift_few_label_recovery`
    and `shift_size_vs_coverage` are light-surface data charts. They use the reference palette's
    categorical slots in fixed order, each series with its own marker. No `node` is available, so
    the validator was not run; the slots are the pre-validated ones. ∞ sets are drawn hollow or on
    a labelled rail, never dropped.
  - `shift_gallery` shows 8 seeded synthetic `val_test` frames on the dark surface:
    - P1's wireframe at the estimate;
    - 24 sampled boundary poses of the A1 set (rotation exactly q·c_R about random axes,
      translation q·c_t·‖t̂‖ in random directions);
    - C1 ellipses and the true pose.
    It uses synthetic imagery only, and its caption carries the CC BY-NC-SA notice.
- **Wireframe edges.** They come from P1 through the new `p1_adapter.wireframe_edges()`. P1 marks
  `WIREFRAME_EDGES` as figures-only.

## 2026-10-07 — Phase 6 leakage audit: carried items closed

`split-leakage-auditor` on the whole repo at `1103ba2`: **clean**, no critical findings.
- **Closed.**
  - W1/W2: the weighted, domain-classifier and oracle paths exist and are guarded.
  - L1: the only consumer of `<hil>_labels_poolA.npz` is `shift.oracle_target_labels`, through
    `load_dump_labels` with the `oracle_target_labels` tag.
  - L2: HIL `gt_crop` rows carry `oracle: true`, and the weighted arm refuses those dumps.
- **Adopted.** W-c: a runtime assertion that `val_tune` and `val_cal` are disjoint (poolA vs poolB
  was already asserted through the evaluation-set check).
- **Carried, notes.**
  - W-a: the oracle guard is a tag-prefix convention, pinned by tests (one poolA request in
    `shift.py`, one oracle call in the script), not a hard barrier.
  - L3: `load_eval_labels` parses before slicing; it is producer-side only.
  - N3: `hil_label_access_allowed` returns True for an empty frame list, which returns no rows.

## 2026-10-07 — Phase 6 validity audit: items closed

`conformal-validity-auditor` at `8c5493e`: **SOUND** for the computation. Quantiles, arm tags,
data roles and label-free sets were all verified, with no critical finding. Its four must-fix
items were about presentation and are closed; no number changed.
- **M1 (figures without answer rate).**
  - `shift_coverage_vs_nominal_*` draws the bound and gives the A1 answer rate in each panel
    title: the floor 1 − answer rate under `abstain_allowed`, the ceiling (answer rate) under
    `answer_required`.
  - `shift_size_vs_coverage` labels every point with its answer rate.
  - `shift_few_label_recovery` prints the mean A1 answer rate per n, and states how many draws
    answer where fewer than all do.
  - `shift_silent_failure` puts the answer rate under each domain.
- **M2 (Mondrian under `answer_required`).** Every such row has a footnote. Its arm cell gives the
  number of frames that get the whole-space set.
- **M3 (slice tables mixed conventions).** There is now one slice table per convention, and every
  cell gives coverage [CI] · answer rate · n.
- **M4 (wording).** Figure titles now say "Marginal, finite-sample coverage ≥ 1 − α holds on
  synthetic val_test (exchangeable with val_cal); on HIL poolB it is measured, not guaranteed".
  The size figure says "beside set size and answer rate".
- **Should-consider items adopted.**
  - S1: synthetic `gt_crop` rows carry a "not deployable" footnote. `oracle` stays HIL-only, as
    the results-provenance skill defines it. IMPLEMENTATION_PLAN's "gt_crop(**oracle**)" is read
    as the HIL case.
  - S2: TABLES.md explains degenerate weights (∞ or −∞ quantiles).
  - S3: the label-scramble test also asserts that the per-frame Mondrian quantiles are unchanged.
  - S4: TABLES.md states that the oracle arm's marginal coverage does hold under poolA/poolB
    exchangeability.
  - S5: the empty heading is removed.
- **Gate re-audit (`1103ba2..2691a15`): clean, with one note recorded here.**
  - The weighted arm's overflow shift c = max logit over `val_cal` ∪ poolB reads poolB *features*,
    never labels.
  - Test-time weights need poolB features anyway.
  - c is a common factor that cancels in every normalised mass, so no quantile depends on it
    (`test_logit_shift_leaves_weighted_quantiles_unchanged`).
  - It is not a fitted normaliser.

## 2026-10-07 — `ConformalPoseHead`: artifact, locks and output types

**Decision.** `conformal/head.py` (numpy only) loads one JSON artifact per (score, convention, α).
- The `payload` holds everything that changes a set:
  - score and score version, α, convention, quantile, n_cal;
  - normalisers (and A3's g knots);
  - the camera geometry (C2 only) and the quaternion order;
  - the locks: P1 keypoint, detector (predicted crop) and variance-head (C1/C2) SHA-256;
  - a `context` with run, crop source, arm, calibration subset, P1 commit and PnP config.
- `calibration_id` = SHA-256 of the canonical payload.
- `from_json` hashes the checkpoint files actually in use and refuses a mismatch. It also refuses
  an edited payload, an unknown schema or score version, and any payload failing
  `build_artifact`'s validation.
- `predict(estimate, cov)` returns `Answer(set, …, calibration_id)` or `Abstain(reason,
  covered_if_abstained)`, following the plan's ANSWER/ABSTAIN diagram rather than returning a bare
  `PoseSet`. A frame is answered iff `metrics.outcomes` would count it answered;
  `covered_if_abstained` carries that function's convention rule.

**Alternatives.**
- Recompute quantiles from the dumps inside the build: rejected, because it duplicates Phases 2,
  4 and 5 and invites drift.
- Lock only the keypoint checkpoint (the plan's literal list): rejected after the
  conformal-validity audit. For predicted crops the detector decides every crop and sets B1's
  radius.
- A keyed MAC: unnecessary. The id detects edits; it is not presented as forgery-proof.

**Reason.**
- Artifacts copy each quantile, n_cal and normaliser from the committed Level A/B/C row (invariant 8).
- `--verify` replays `val_test` frame by frame through `predict` and reproduces every row's
  n_covered and n_answered exactly ([`results/calibration/index.json`](../results/calibration/index.json)).

**Score versions.** `SCORE_VERSIONS` starts at 1 for every deployable score. Any change to a score's
definition, normaliser or set construction bumps its version, and old artifacts are then refused.

**B2 is not deployable.** It needs heatmap second moments, which the exported graph does not output.
The deployable scores are A1–A3, B1, C1 and C2.

**C2 in the head.** σ̂ comes from the linearisation of `engine.level_b.linearised_frames` +
`engine.level_c.pose_sigma`, and its covariances are validated by the C1 set constructor. The camera
geometry lives in the artifact because σ̂, and so the calibration, depends on it.

## 2026-10-07 — Exported variance graph: `keypoint_empty` output, no `heatmaps`

**Decision.** The plan names the outputs `coords, confidence, cov_chol`. The graph instead emits
`coords, confidence, cov_chol, keypoint_empty` and drops P1's `heatmaps`.

**Reason.**
- C1 and C2 treat an empty heatmap channel as unconstrained: its set is the whole image, and it is
  excluded from C2's U.
- On such a channel the readout returns σ = 1 px. A deployed head without the mask would read that
  as a real covariance, a fabricated tight ellipse.
- `keypoint_empty` is computed by the same `refine_weights` the readout uses: mass 0 in P1's refine
  window, the dump's `heatmap_empty`.
- No deployed score needs the heatmaps.
- Graph checks: opset 17, no forbidden ops, static non-batch shapes
  ([`results/export/onnx_export.json`](../results/export/onnx_export.json)).

**Head-off baseline.** P1's own `keypoint_a2` graph, re-exported from the pinned checkpoint with
P1's `spec_for_config` into gitignored `exports/`; the detector graph likewise.

## 2026-10-07 — ONNX parity: the fp32 variance gate is missed; reported, not moved

**Measured** ([`results/export/onnx_parity.json`](../results/export/onnx_parity.json)): 512
`val_test` crops; ORT CUDA with `use_tf32 = 0` against torch with TF32 off.
- **`cov_chol`:** max |Δ| 2.857e-3, mean 1.44e-6, p99 1.29e-5, against the plan's gate of 1e-4.
  **Unmet.**
- **Σ̂:** max |Δ| 0.233 px². Unmet.
- **`keypoint_empty`:** 0 mismatches.
- **Relative σ_max:** max 6.95e-5.
- **C1 set radius (what ships):** relative max 6.95e-5, p99 4.74e-6.
- **Decode stability:** 5,630 of 5,632 keypoints decode to within 1e-3 px. On those, `cov_chol`
  max |Δ| is 6.71e-4, still above 1e-4.

**Diagnosis.** It is not TF32 or cuDNN. The CPU control (torch CPU vs ORT CPU, same graph, 256
crops) also misses: `cov_chol` max 7.02e-4, C1 radius relative max 1.76e-5. The TF32-off/off cell
of P1's four-cell ablation gives 2.94e-3, against 0.268 with both on. What is left is the two
frameworks' fp32 arithmetic in a sharp readout: `relu(h)^p` × window, normalised over 4,096 pixels.
That arithmetic amplifies heatmap differences of order 1e-6. Set radii agree to < 1e-4 relative.

**fp16.** Halving the trunk flips P1's decode on most keypoints (112 of 5,632 stable). The C1
radius differs by up to 8.3 % (p99 1.69 %), and `cov_chol` misses P1's stated fp16 relaxation (5e-2).
**fp16 is not a faithful export of the variance outputs.**

**Coords.** These are P1's outputs, and P1's keypoint parity gate is inherited unmet. With TF32 off
here, the coords max |Δ| is 6.1e-3 px. P1's committed record (TF32 on) gives 3.23 px. Both numbers
are in the parity file.

**Not done.** Loosening the 1e-4 gate, or gating on the relative or set-level number instead
(invariant 11). The gate stays as planned; Phase 8 records G2 as unmet, with this evidence.

## 2026-10-07 — Latency: the head and propagation timed on CPU with P1's minimums

Network rows use P1's `benchmark_onnx` unchanged.
- The conformal head and the propagation estimators have no P1 counterpart. `export.bench.time_calls`
  times them, one frame per call, 64 real `val_test` frames from the dumps cycled, under P1's
  minimums (≥ 50 warmup, ≥ 500 timed).
- The full frame uses P1's `end_to_end_frame_budget` on the ORT pipeline. The head-on row adds one
  `uncertainty` stage (covariance mapping + C1 and C2 `predict`) through `UncertaintyFrame`, a
  callable with P1's pipeline signature.
- Every file carries the A4000-is-not-a-Jetson caveat.

## 2026-10-07 — Result kinds `calibration` and `onnx_export`

Added to `poseconf.result.v1`:
- `calibration`: the head replay index;
- `onnx_export`: per-graph SHA-256 + `graph_summary`.

Parity and latency use the existing `parity` and `latency` kinds. The artifacts themselves use
their own schema, `poseconf.calibration.v1`.

## 2026-10-07 — I run the Phase 7 GPU jobs myself, on the user's instruction

As in Phase 3: the user asked for the GPU commands to be generated *and then run*. `make export`,
`make parity-onnx` and `make bench` print the commands; I ran them in that order on a clean tree.
The phase came to about 3,300 lines of src, scripts and tests, well beyond the 200–800 guide. Most of that is the
replay check, the CPU control and the tests that pin the head to the evaluation path.
