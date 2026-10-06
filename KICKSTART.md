# Kickstart — running this project with Claude Code

How to drive `IMPLEMENTATION_PLAN.md` phase by phase: setup, what each agent and skill is for, and a
copy-paste prompt for every phase. The prompts assume you are in the repo root in Claude Code.

---

## 1. One-time setup (≈20 min, you do this, not Claude)

```bash
# 1. Repo + submodule + kit  (REPO_SETUP.md §1)
mkdir pose-with-error-bars && cd pose-with-error-bars && git init -b main
cp -r /path/to/pose-with-error-bars-kit/. .
chmod +x .claude/hooks/*.sh            # hooks need jq: sudo apt install jq
git submodule add https://github.com/kbasak01/spacecraft-pose-baseline external/spacecraft-pose-baseline
git -C external/spacecraft-pose-baseline checkout phase-9-complete
cp configs/paths.local.example.yaml configs/paths.local.yaml   # edit the three paths
git add -A && git commit -m "chore: starter kit + P1 submodule at phase-9-complete"

# 2. Environment (REPO_SETUP.md §2) — or let the Phase 0 prompt do it
```

Then start Claude Code in the repo root (`claude`, or the desktop app's Code tab pointed at the
folder). Because `.claude/agents/` is new, **restart the session once** so the agents load. Check:

- `/agents` → 9 project agents listed
- type `/` → `phase-gate`, `coverage-check`, `verify-p1-parity`, `claim-trace`, `conformal-math`,
  `results-provenance` appear
- `/hooks` → one PreToolUse (guard) and one PostToolUse (ruff) hook

## 2. What you are deploying

### Subagents (`.claude/agents/`)

| Agent | Model | Writes? | When | Invoke |
|---|---|:-:|---|---|
| `conformal-validity-auditor` | opus | no | Phase 1 gate (mandatory), any change to `conformal/` | `@"conformal-validity-auditor (agent)" audit src/poseconf/conformal` |
| `split-leakage-auditor` | sonnet | no | Phase 2, Phase 6, any split/HIL/classifier change | "use the split-leakage-auditor subagent …" |
| `p1-parity-auditor` | sonnet | no | Phase 3 gate | via `/verify-p1-parity` |
| `uncertainty-head-diagnostician` | sonnet | no | after every variance-head run | "use the uncertainty-head-diagnostician on runs/variance_a2" |
| `shift-study-runner` | sonnet | yes | Phase 6 matrix + tables | "use the shift-study-runner to …" |
| `pose-geometry-verifier` | opus | yes | Phase 4 propagation code | `@"pose-geometry-verifier (agent)"` |
| `onnx-parity-auditor` | sonnet | yes | Phase 7 | "use the onnx-parity-auditor …" |
| `eval-reproducibility-auditor` | opus | no | Phase 8, before README claims | via `/claim-trace` or directly |
| `repo-publication-reviewer` | sonnet | no | Phase 9 | "use the repo-publication-reviewer …" |

Auditors are read-only on purpose: the agent that finds a problem is not the one that fixes it. The
main session fixes; the auditor re-checks. The `@"name (agent)"` form guarantees the agent runs;
plain-language mentions let Claude decide.

### Skills (`.claude/skills/`)

| Skill | Who invokes | Purpose |
|---|---|---|
| `/phase-gate N` | you | evidence table for phase N's exit gate; updates PROGRESS/DECISIONS; prints the tag command |
| `/coverage-check A1 abstain_allowed` | you | repeated-split validity test vs the Beta law |
| `/verify-p1-parity keypoint_a2` | you | dump vs P1 sidecars, via `p1-parity-auditor` |
| `/claim-trace README.md` | you | every number and "guarantee" word traced to `results/` |
| `conformal-math` | Claude (auto) or you | formulas + pitfalls; preloaded into the validity auditor |
| `results-provenance` | Claude (auto) or you | JSON schema + table/figure rules; preloaded into the runner and the claims auditor |

The four `/…` workflow skills have `disable-model-invocation: true` — they run only when you type them.

### Hooks (`.claude/settings.json`)

- **guard_paths.sh** (PreToolUse): blocks edits inside `external/` (P1) and the dataset tree,
  git write operations in the submodule, `rm`/`mv` on data, re-downloading SPEED+, and bumping the
  submodule pin. Tested against 14 sample calls.
- **ruff_on_edit.sh** (PostToolUse): lints + format-checks each edited `.py`; findings go back to
  Claude in the same turn.

## 3. Session habits that matter

- **Plan mode per phase.** Press Shift+Tab to cycle to plan mode (or start the prompt with "Plan
  first:") and approve before code is written. Phases are 200–800 lines; do not one-shot them.
- **One phase per session.** `/clear` between phases. The memory between sessions is
  `docs/PROGRESS.md` + `docs/DECISIONS.md`; every phase prompt starts by reading them.
- **You run the GPU jobs.** Claude prints the command (`make dump`, `make train-var`, export/bench);
  you run it in a terminal and paste back the last lines or say "done".
- **Gate before moving on.** `/phase-gate N` → read the table → `git tag phase-N-complete` yourself.
- **Never let Claude adjust a threshold to pass.** If a gate fails, the next prompt is "diagnose",
  not "fix the test".

## 4. Phase prompts (copy-paste)

### Phase 0 — Scaffold

```
Read CLAUDE.md, IMPLEMENTATION_PLAN.md (§0, §1, Phase 0), REPO_SETUP.md and SOURCES.md fully.
Plan first, then implement Phase 0:
- pyproject.toml (copy P1's ruff/pytest blocks from external/spacecraft-pose-baseline/pyproject.toml,
  package name poseconf), Makefile from REPO_SETUP.md §8, .gitignore §5, requirements-extra.txt,
  .github/workflows/ci.yml §7 with submodules: recursive.
- src/poseconf/ skeleton per REPO_SETUP.md §4 (empty modules with docstrings only).
- src/poseconf/p1_adapter.py: the only speedpose import. load_p1_config, load_sidecar,
  p1_paths, reading configs/paths.local.yaml; fail loudly on missing paths.
- scripts/verify_p1_checkpoints.py writing results/p1_checkpoints.json (sha256 vs the release
  SHA256SUMS.txt I will place at <path>; mark the test @pytest.mark.dataset).
- tests: test_adapter.py (sidecar loads from the committed P1 results, no dataset needed),
  test_no_torch_in_conformal.py.
Print the venv install commands for me instead of running long installs. Finish with
make lint && make test && make smoke, then /phase-gate 0 is mine to run.
```

### Phase 1 — Conformal core ⚠️

```
Read CLAUDE.md, docs/PROGRESS.md, docs/DECISIONS.md, IMPLEMENTATION_PLAN.md §0, §1.3–§1.5 and
Phase 1. Load the conformal-math skill. Plan first.
Implement src/poseconf/conformal/{split,weighted,mondrian,so3,scores,metrics}.py and
docs/SCORES.md exactly as Phase 1 specifies. numpy/scipy only.
Tests first, then code: brute-force quantile index for n in {1,2,9,10,99,4798}; Beta-law coverage
property tests (Gaussian, Student-t df=3, heteroscedastic) with 2,000 resamples; weighted CP under a
known covariate shift with true weights; ±inf conventions; set functions take no labels
(signature test); MAPIE v1 and TorchCP oracle cross-checks marked @pytest.mark.slow.
When green, run the conformal-validity-auditor subagent on src/poseconf/conformal and docs/SCORES.md
and fix only what it marks critical or must-fix, then re-run it until SOUND.
```

### Phase 2 — Splits + Level A (CPU, P1 sidecars)

```
Read CLAUDE.md, PROGRESS, DECISIONS, Phase 2. Load results-provenance. Plan first.
1) scripts/make_splits.py + src/poseconf/data/splits.py: seeded manifests in splits/ with
   SHA256SUMS, per configs/conformal.yaml; regeneration byte-identical (test).
2) scripts/run_level_a.py: scores A1–A3 from P1's committed keypoint_a2 predicted_crop sidecars via
   the adapter; normalisers and g(conf) on val_tune; calibrate val_cal; evaluate val_test ONLY (no
   HIL in this phase); both conventions; full alpha grid; R=1000 re-splits.
3) JSON per results-provenance schema under results/level_a/.
Then run the split-leakage-auditor, and /coverage-check is mine to run for A1, A2, A3.
Do not open or compute anything on lightbox or sunlamp yet.
```

### Phase 3 — Dump + P1 parity (GPU)

```
Read CLAUDE.md, PROGRESS, DECISIONS, Phase 3. Plan first.
Read external/spacecraft-pose-baseline/src/speedpose/pipeline.py, engine/pose.py and
engine/evaluate.py (read-only) and design src/poseconf/data/dump.py to call P1's own stage
functions (prepare_frame, detect_boxes, crop_one, normalise_crop, model forward, solve_single) —
no re-implementation of crop, decode or PnP. Add a forward hook for decoder features and the
pooled encoder feature. Store per-frame fields listed in Phase 3; heatmap moments mapped to full
frame; heatmaps only for a 256-frame subset.
scripts/dump_predictions.py and scripts/check_p1_parity.py; a fixture test that runs the dump on
tests/fixtures with a random-init model (no dataset). Then STOP and print the exact GPU commands
for me (all three domains, predicted_crop and gt_crop, TF32 disabled).
```
After you run it:
```
The dump finished (output pasted below). Run /verify-p1-parity keypoint_a2 and report.
```

### Phase 4 — Level B

```
Read CLAUDE.md, PROGRESS, DECISIONS, Phase 4. Load conformal-math. Plan first.
Implement B1/B2 calibration (synthetic only) and src/poseconf/conformal/propagate.py: covariance
crop→full-frame mapping, PURSE-membership ≡ joint-keypoint-coverage test, linearised propagation
via cv2.projectPoints Jacobian (finite-difference test), sampled propagation (M=256, offline,
labelled inner approximation, timed). scripts/run_level_b.py → results/level_b/.
Then run the pose-geometry-verifier on so3.py and propagate.py, and the conformal-validity-auditor
on the new scores.
```

### Phase 5 — Variance head (GPU)

```
Read CLAUDE.md, PROGRESS, DECISIONS, IMPLEMENTATION_PLAN §1.6 and Phase 5. Plan first.
Implement src/poseconf/models/variance_head.py (VarianceKeypointNet wrapping a frozen P1
KeypointNet; heatmap-weighted readout, Cholesky covariance, clamps), beta-NLL loss,
scripts/train_variance_head.py with P1's dataset and A2 augmentation via the adapter, configs/
variance_head.yaml. Tests: frozen-parameter hash unchanged after a training step; coords
bit-identical vs bare P1 model; PD covariances; no ArgMax/TopK/NonZero in a traced graph.
Run make train-var-smoke yourself (CPU/GPU small). Then print the full training command for me.
```
After training:
```
Training finished: runs/<name>. Run the uncertainty-head-diagnostician on it. If TRUSTWORTHY,
re-dump with the head (print the command), then calibrate C1/C2 on synthetic and run
/coverage-check C1 — no HIL yet.
```

### Phase 6 — Coverage under shift ⭐

```
Read CLAUDE.md, PROGRESS, DECISIONS, IMPLEMENTATION_PLAN §1.4 and Phase 6. Load
results-provenance. Plan first.
Implement scripts/run_shift_matrix.py and src/poseconf/engine/shift.py for every
score × arm × convention × alpha × domain × crop source in configs/conformal.yaml. HIL evaluation on
*_poolB only. weighted_unlabeled_target uses poolA features only (report 5-fold AUC and ESS).
oracle_target_labels draws from poolA labels only, tagged oracle. Then:
1) split-leakage-auditor on the whole repo;
2) shift-study-runner to execute and build results/TABLES.md;
3) scripts/make_figures.py for the five Phase 6 figures;
4) conformal-validity-auditor on tagging and language.
Report the headline in five lines, every number with its file.
```

### Phase 7 — Head, ONNX, latency (GPU)

```
Read CLAUDE.md, PROGRESS, DECISIONS, Phase 7. Plan first.
Implement ConformalPoseHead (src/poseconf/conformal/head.py) with the hash-locked calibration JSON,
round-trip and mismatched-hash refusal tests; export VarianceKeypointNet to ONNX (opset 17, fp32/fp16)
reusing speedpose.export conventions via the adapter; parity script; latency script reusing P1's
bench methodology. Print the GPU commands for me. After I run them, invoke the onnx-parity-auditor.
```

### Phase 8 — Validation sweep

```
Read CLAUDE.md, PROGRESS, DECISIONS, VALIDATION_CHECKLIST.md. Walk every line; for each, produce
verdict + evidence (file or command) and write docs/LIMITATIONS.md in P1's format. Run in order:
split-leakage-auditor, conformal-validity-auditor, p1-parity-auditor, then
eval-reproducibility-auditor. Do not fix anything inside this prompt — list the fixes, ranked.
```

### Phase 9 — Publication

```
Read CLAUDE.md, PROGRESS, LIMITATIONS, results/TABLES.md and P1's README.md for style. Load
results-provenance. Draft README.md: headline coverage-under-shift figure + table above the fold,
"what the guarantee says and does not say" box, method summary vs Yang & Pavone 2023 and
Wang et al. 2025, reproduction, latency with the A4000 caveat, limitations, attribution.
Then /claim-trace is mine to run, then the repo-publication-reviewer.
```

### Resume prompt (any session)

```
Read CLAUDE.md, docs/PROGRESS.md and the last five entries of docs/DECISIONS.md. Tell me in five
lines: current phase, last gate result, open blockers, and the next concrete step. Do not edit.
```

### When a gate fails

```
Phase N gate failed on <criterion>. Diagnose only: find the root cause with evidence, propose the
smallest fix, and say whether the fix changes any already-committed result. Do not change
thresholds, splits, or tests.
```

## 5. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Agents not listed | session predates `.claude/agents/` | restart Claude Code once |
| Hook errors `jq: not found` | jq missing | `sudo apt install jq` |
| Every edit blocked | path contains `external/` or `datasets/` legitimately | rename your folder, or narrow `guard_paths.sh` |
| `import speedpose` fails | submodule not initialised / not editable | `git submodule update --init`; `uv pip install -e external/spacecraft-pose-baseline --no-deps` |
| P1 configs not found | P1 resolves paths from its own `REPO_ROOT` | must be an editable install of the submodule, not a wheel |
| Parity miss ~0.1 % of frames | TF32 / soft-argmax bistability (P1 documented) | disable TF32 both sides; report residual, do not recalibrate on it |
| Weighted CP sets all `∞` | domain classifier AUC ≈ 1, ESS tiny | expected under severe shift; report AUC and ESS |
