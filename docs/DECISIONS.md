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
