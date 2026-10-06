# CLAUDE.md

Project instructions for Claude Code. Read this fully before your first edit in any session.

## What this project is

Distribution-free uncertainty for an existing monocular 6DOF pose pipeline. Project 1
(`spacecraft-pose-baseline`, pinned as a git submodule at `external/spacecraft-pose-baseline`, tag
`phase-9-complete`) produces poses on SPEED+. This repo (`poseconf`) adds split-conformal pose and
keypoint sets with finite-sample coverage guarantees, a learned heteroscedastic keypoint head, and a
measurement of how coverage degrades under the synthetic → hardware-in-the-loop shift.

The headline result is the **coverage gap under shift**, not a coverage number. A guarantee that
holds on synthetic and fails on `lightbox` is the finding. The plan of record is
`IMPLEMENTATION_PLAN.md`; follow its phase order and do not start a phase whose predecessor's exit
gate is unmet.

## Non-negotiable invariants

1. **P1 is read-only.** Never edit, commit to, or move files inside `external/spacecraft-pose-baseline`.
   Never change its pinned commit without an entry in `docs/DECISIONS.md` and a re-run of the
   Phase 3 parity gate. If P1 code looks wrong, write it down; do not patch it.
2. **`poseconf/p1_adapter.py` is the only module that imports `speedpose`.** Everything else goes
   through the adapter.
3. **Splits are fixed and committed** (`splits/*.txt` + SHA-256). Never re-split, never move a frame
   between splits. `val_tune` is the only synthetic-validation subset any fitting or selection may
   see. `val_cal` is for calibration only. `val_test` and every `*_poolB` are evaluation only.
4. **HIL labels are tagged or untouched.** A `lightbox`/`sunlamp` label may be read only (a) when
   evaluating on `*_poolB`, or (b) inside an arm whose tag starts with `oracle_`. Unlabeled poolA
   *images* may be used only by the `weighted_unlabeled_target` arm. Anything else reading a HIL
   label is test-set leakage — stop and report it.
5. **Scores must be computable at test time.** The set-construction function of every score takes
   predictions and a quantile, never labels. Normalise translation by predicted `‖t̂‖`, never GT
   `‖t‖`. GT quantities may *slice* an evaluation; they may never enter a score.
6. **PnP failures are never dropped.** Every frame gets a score: `+∞` (answer-required) or `−∞`
   (abstain-allowed). Every table states its convention. Never mix conventions in one table.
7. **Never claim a guarantee you have not got.** Coverage is marginal, finite-sample, and conditional
   on exchangeability. It is not conditional coverage, not a bound on error, and not "certified" or
   "safe". Under shift, report *measured* coverage with a Clopper–Pearson interval and n.
8. **No number without a file.** Every number in README or any summary must come from a committed
   JSON in `results/`. `∞` is a number; report it, do not omit the row.
9. **The dataset and P1 checkpoints are already on the workstation — never re-acquire them.** Paths
   come from `configs/paths.local.yaml` (gitignored). Never download, extract, move, delete, or write
   into the dataset. If something is missing, say so and stop.
10. **The variance head never changes the pose.** P1 coordinates are bit-identical with and without
    it; the test for this must stay green.
11. **Do not move a gate to pass it.** If coverage misses, report the miss. A failing gate is
    information.
12. **Never move, rename, or delete the six root planning docs** (`CLAUDE.md`,
    `IMPLEMENTATION_PLAN.md`, `REPO_SETUP.md`, `VALIDATION_CHECKLIST.md`, `SOURCES.md`, `KICKSTART.md`).

## Working agreements

- **Plan mode before each phase.** Get the plan approved, then implement. Phases are 200–800 lines.
- **Tests with the code.** `poseconf/conformal/` gets property tests against synthetic data with
  known coverage before it touches SPEED+ numbers.
- **Append to `docs/DECISIONS.md`** for every non-obvious choice: date, decision, alternatives, reason.
- **Update `docs/PROGRESS.md`** at the end of every phase with status and the key numbers.
- **`make lint && make test` before claiming anything is done.**
- **Small conventional commits.**
- **Unsure between two designs → ask.** Don't build both.
- Load the `conformal-math` skill before writing or reviewing any calibration code, and
  `results-provenance` before writing any table, figure or README sentence with a number in it.

## Commands

```bash
make lint            # ruff check + format check
make test            # full pytest
make smoke           # no dataset, no GPU: conformal core + adapter + fixture dump
make splits          # regenerate split manifests (must be byte-identical)
make level-a         # Phase 2, from P1 sidecars, CPU only
make dump            # Phase 3, GPU — print the command, the user runs it
make parity-p1       # Phase 3 gate
make level-b         # Phase 4
make train-var-smoke # Phase 5 overfit gate
make train-var       # Phase 5, GPU — user runs it
make shift-matrix    # Phase 6
make tables figures  # regenerate results/TABLES.md and assets/ from JSON
make export bench    # Phase 7
```

## Environment

- Single RTX A4000 16 GB under WSL2. bf16 AMP, seed 1337 (splits seed 20261006).
- Long jobs (dump, variance-head training, export benchmarks) are run by the user. Prepare the
  command, print it, and stop.
- `dumps/`, `runs/`, `exports/`, `configs/paths.local.yaml` are gitignored.

## Code style

- Type hints on public functions; Google-style docstrings in `src/`.
- Config in YAML. No magic numbers in function bodies.
- `src/poseconf/conformal/` has **zero torch imports** (pure numpy/scipy) — it ports to the
  ship-deck project. A test enforces this.
- No notebooks. Scripts with CLI args. Fail loudly; no bare `except`, no silent `None`.

## Subagents

| Agent | Use when |
|---|---|
| `conformal-validity-auditor` | Phase 1 (mandatory), and after any change to `conformal/` or a score |
| `split-leakage-auditor` | Phase 2, Phase 6, and any change touching splits, HIL data, or the domain classifier |
| `p1-parity-auditor` | Phase 3 gate, and after any change to the dump or the submodule pin |
| `uncertainty-head-diagnostician` | after every variance-head training run |
| `shift-study-runner` | Phase 6 matrix and tables |
| `onnx-parity-auditor` | Phase 7 |
| `eval-reproducibility-auditor` | Phase 8, and before any README claim |
| `repo-publication-reviewer` | Phase 9 |
| `pose-geometry-verifier` | Phase 4 Jacobian/sampling code |

## Anti-patterns specific to this project

- Reporting coverage on solved frames only without saying so, or without the answer rate beside it.
- Calibrating on `val_cal ∪ val_tune`, or choosing α, bins, or normalisers by looking at HIL coverage.
- Using `np.quantile(scores, 1 - alpha)` — that is not the conformal quantile.
- Quoting sunlamp coverage without n and a confidence interval.
- Saying "guaranteed" about a HIL number.
- Treating near-nominal HIL coverage as success without checking set size — an `∞` set covers everything.
- Retraining P1, adding domain adaptation, or "improving" PnP. Out of scope; note it in `docs/DECISIONS.md`.

## Attribution

SPEED+ is **CC BY-NC-SA 4.0** (Stanford Digital Repository, purl.stanford.edu/wv398fc4383) — cite it,
do not redistribute it, and treat anything derived from its imagery (figures, weights) as carrying NC
and SA. P1 code is MIT. Cite Park et al. 2022 (SPEED+), Yang & Pavone 2023, Wang et al. 2025,
Tibshirani et al. 2019, Angelopoulos & Bates, Seitzer et al. 2022 — see `SOURCES.md`.
