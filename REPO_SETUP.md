# Repo setup — `pose-with-error-bars`

Everything Phase 0 needs. The layout below is the *intended* one; once built, `README.md` describes
the real one.

## 1. Create the repo

```bash
mkdir pose-with-error-bars && cd pose-with-error-bars
git init -b main
# copy the starter kit in (root docs + .claude/ + configs/ + docs/ + splits/)
cp -r /path/to/pose-with-error-bars-kit/. .

# Pin Project 1 as a read-only submodule at its published tag.
git submodule add https://github.com/kbasak01/spacecraft-pose-baseline external/spacecraft-pose-baseline
git -C external/spacecraft-pose-baseline checkout phase-9-complete     # 0f81b426e0a7…
git add .gitmodules external/spacecraft-pose-baseline
git commit -m "chore: scaffold from starter kit; pin P1 at phase-9-complete"

gh repo create kbasak01/pose-with-error-bars --public --source . --push   # when ready
```

Why a submodule and not a fork or a copy: P1 is published with phase tags and committed results;
forking invites edits that would make its numbers stale, and copying loses provenance. The
submodule gives P6 P1's code, configs, wireframe asset and committed per-sample sidecars at an exact
commit, and an editable install makes P1's module-level `REPO_ROOT` resolve inside the submodule so
its relative config paths keep working.

## 2. Environment

```bash
uv venv --python 3.11 && source .venv/bin/activate

# P1's pinned stack first (torch 2.11 cu128, onnxruntime-gpu CUDA-12 wheel, TensorRT 10.9, …).
uv pip install -r external/spacecraft-pose-baseline/requirements.txt --index-strategy unsafe-best-match
uv pip install -e external/spacecraft-pose-baseline --no-deps

# This repo's additions, then the package itself.
uv pip install -r requirements-extra.txt
uv pip install -e . --no-deps
python -c "import speedpose, poseconf; print('ok')"
```

`requirements-extra.txt` (pin exact versions at install time and record them in `docs/ENVIRONMENT.md`):

```
scikit-learn          # logistic domain classifier, isotonic fit of g(conf)
statsmodels           # Clopper–Pearson intervals (or implement via scipy.stats.beta)
mapie>=1.0            # TEST ORACLE ONLY — never imported from src/ (v1 API: SplitConformalRegressor)
torchcp>=1.2          # TEST ORACLE ONLY — never imported from src/
```

On 2026-10-06 PyPI listed MAPIE 1.5.0 and TorchCP 1.2.1. CI installs P1's `requirements-ci.txt`
(CPU wheels) instead of `requirements.txt`, plus the same extras.

## 3. Local paths (gitignored)

`configs/paths.local.yaml` — copy from `configs/paths.local.example.yaml`:

```yaml
speedplus_root: /home/<you>/datasets/speedplusv2          # the existing 17 GB tree; read-only
p1_runs: /home/<you>/code/spacecraft-pose-baseline/runs   # your existing P1 checkpoints
dumps_root: dumps
```

If the P1 `runs/` tree is not on this machine, download the `phase-9-complete` release assets
(four `best.pt` + `SHA256SUMS.txt`) into `p1_runs/<run>/best.pt` and let Phase 0 verify the hashes.

## 4. Layout

```
pose-with-error-bars/
├── CLAUDE.md  IMPLEMENTATION_PLAN.md  REPO_SETUP.md  VALIDATION_CHECKLIST.md  SOURCES.md  KICKSTART.md
├── README.md                       # written in Phase 9
├── LICENSE                         # MIT for code
├── pyproject.toml  Makefile  requirements-extra.txt
├── .github/workflows/ci.yml
├── .claude/
│   ├── settings.json               # permissions + hooks
│   ├── hooks/                      # guard_paths.sh, ruff_on_edit.sh
│   ├── agents/                     # 9 subagents
│   └── skills/                     # conformal-math, results-provenance, phase-gate, coverage-check,
│                                   # verify-p1-parity, claim-trace
├── external/spacecraft-pose-baseline/   # submodule, read-only
├── configs/
│   ├── conformal.yaml              # scores, arms, α grid, R, seeds (draft provided)
│   ├── variance_head.yaml          # Phase 5
│   ├── paths.local.example.yaml
│   └── paths.local.yaml            # gitignored
├── splits/                         # committed manifests + SHA256SUMS
├── src/poseconf/
│   ├── __init__.py
│   ├── p1_adapter.py               # the ONLY speedpose import
│   ├── conformal/                  # numpy/scipy only — no torch
│   │   ├── split.py  weighted.py  mondrian.py  aci.py
│   │   ├── so3.py  scores.py  propagate.py  head.py  metrics.py
│   ├── data/      dump.py  splits.py  sidecars.py
│   ├── models/    variance_head.py
│   ├── engine/    train_variance.py  evaluate.py  shift.py
│   └── export/    to_onnx.py  bench.py
├── scripts/       make_splits.py  run_level_a.py  dump_predictions.py  check_p1_parity.py
│                  run_level_b.py  train_variance_head.py  run_shift_matrix.py
│                  export_onnx.py  bench.py  make_tables.py  make_figures.py
├── tests/         test_conformal_core.py  test_scores.py  test_so3.py  test_splits.py
│                  test_adapter.py  test_propagate.py  test_variance_head.py  test_head.py
│                  test_no_torch_in_conformal.py  test_onnx_parity.py  fixtures/
├── results/       every committed number lives here (JSON / small NPZ)
├── assets/        generated figures
├── docs/          DECISIONS.md  PROGRESS.md  LIMITATIONS.md  ENVIRONMENT.md  SCORES.md
├── dumps/  runs/  exports/          # gitignored
```

## 5. `.gitignore`

```
/dumps/
/runs/
/exports/
configs/paths.local.yaml
*.pt
*.pth
*.onnx
*.engine
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
```

## 6. `pyproject.toml` essentials

Copy P1's `[tool.ruff]` and `[tool.pytest.ini_options]` blocks verbatim (same markers: `slow`,
`gpu`, `dataset`), set `name = "poseconf"`, `packages.find.where = ["src"]`,
`known-first-party = ["poseconf"]`, `dependencies = []`.

## 7. CI (`.github/workflows/ci.yml`)

CPU-only, mirrors P1's workflow with two differences: `actions/checkout` with
`submodules: recursive`, and the install step:

```yaml
- run: |
    pip install -r external/spacecraft-pose-baseline/requirements-ci.txt
    pip install -e external/spacecraft-pose-baseline --no-deps
    pip install -r requirements-extra.txt
    pip install -e . --no-deps
- run: ruff check src tests scripts && ruff format --check src tests scripts
- run: pytest -q -m "not slow and not gpu and not dataset" tests
- run: make smoke
```

## 8. Makefile skeleton

```make
VENV   := $(wildcard .venv/bin)
PY     := $(if $(VENV),.venv/bin/python,python)
PYTEST := $(if $(VENV),.venv/bin/pytest,pytest)
RUFF   := $(if $(VENV),.venv/bin/ruff,ruff)

lint:   ; $(RUFF) check src tests scripts && $(RUFF) format --check src tests scripts
test:   ; $(PYTEST) -q tests
smoke:  ; $(PYTEST) -q tests -m "not slow and not gpu and not dataset"
splits: ; $(PY) scripts/make_splits.py --config configs/conformal.yaml
level-a: ; $(PY) scripts/run_level_a.py --config configs/conformal.yaml --run keypoint_a2 --crop-source predicted_crop
dump:
	@echo "Run this yourself (GPU, ~minutes per domain):"
	@echo "  $(PY) scripts/dump_predictions.py --run keypoint_a2 --all-domains --all-arms"
parity-p1: ; $(PY) scripts/check_p1_parity.py --run keypoint_a2
level-b:   ; $(PY) scripts/run_level_b.py --config configs/conformal.yaml
train-var-smoke: ; $(PY) scripts/train_variance_head.py --config configs/variance_head.yaml --overfit-batch --max-steps 400
train-var:
	@echo "Run this yourself (GPU):"
	@echo "  $(PY) scripts/train_variance_head.py --config configs/variance_head.yaml"
shift-matrix: ; $(PY) scripts/run_shift_matrix.py --config configs/conformal.yaml
tables:  ; $(PY) scripts/make_tables.py
figures: ; $(PY) scripts/make_figures.py --figure all
export:  ; $(PY) scripts/export_onnx.py --config configs/variance_head.yaml --fp16
bench:   ; $(PY) scripts/bench.py --all
```

## 9. Claude Code setup

1. Open the repo in Claude Code (`claude` in the repo root, or the desktop app's Code tab).
2. The `.claude/agents/` directory is new — **restart the session once** after copying the kit so the
   agents load. Later edits to agent files are picked up without a restart.
3. `/agents` lists the nine agents; `/` lists the six skills.
4. `.claude/hooks/*.sh` must be executable: `chmod +x .claude/hooks/*.sh`. They need `jq`.
5. Start with the Phase 0 prompt in `KICKSTART.md`.
