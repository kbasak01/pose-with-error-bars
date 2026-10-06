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
