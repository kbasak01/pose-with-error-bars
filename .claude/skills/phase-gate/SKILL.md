---
name: phase-gate
description: Check whether a phase's exit gate in IMPLEMENTATION_PLAN.md is met, then update docs and prepare the tag. Invoke as /phase-gate <phase number>.
#disable-model-invocation: true
argument-hint: "[phase-number]"
---

Check the exit gate for phase $ARGUMENTS as defined in `IMPLEMENTATION_PLAN.md` §2.

1. Quote the phase's exit gate verbatim from the plan.
2. For each criterion, find concrete evidence (test output, `results/` file, figure, log). Run what
   is needed to produce it now. Use the subagent the plan names for this phase.
3. Print a table: criterion | evidence (file or command) | verdict.
4. If ALL pass: update `docs/PROGRESS.md` (status + key numbers, each with its source file), append
   any decisions to `docs/DECISIONS.md`, run `make lint && make test`, commit with a conventional
   message, and print (do not run) `git tag phase-$ARGUMENTS-complete`.
5. If ANY fails: stop. State what failed and the smallest next action. Do not start the next phase.
   Do not edit the criteria.

A failing gate is information, not an obstacle.
