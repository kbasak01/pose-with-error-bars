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
