---
name: verify-p1-parity
description: Check that this repo's prediction dump reproduces Project 1's committed per-sample results and that the P1 submodule is pinned and clean. Invoke as /verify-p1-parity [run].
#disable-model-invocation: true
argument-hint: "[run, default keypoint_a2]"
---

Verify P1 parity for run: $ARGUMENTS (default `keypoint_a2` if empty).

1. Confirm the submodule commit matches `configs/conformal.yaml` and `git status` inside it is clean.
2. Confirm `dumps/<run>/` exists for every domain × crop arm. If not, print the `make dump` command
   for the user to run and stop — do not launch the GPU job yourself.
3. Run `make parity-p1` and read `results/dump_summary.json`.
4. Delegate the judgement to the `p1-parity-auditor` subagent and relay its verdict and table.
