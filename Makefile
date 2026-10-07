.PHONY: lint test smoke splits level-a dump parity-p1 level-b train-var-smoke train-var shift-matrix tables figures export bench verify-p1

# Use the project venv when it exists, otherwise whatever is on PATH (CI has no .venv).
VENV   := $(wildcard .venv/bin)
PY     := $(if $(VENV),.venv/bin/python,python)
PYTEST := $(if $(VENV),.venv/bin/pytest,pytest)
RUFF   := $(if $(VENV),.venv/bin/ruff,ruff)

lint:   ; $(RUFF) check src tests scripts && $(RUFF) format --check src tests scripts
test:   ; $(PYTEST) -q tests
# No dataset, no GPU, no checkpoints: the clean-clone gate. Same filter as CI.
smoke:  ; $(PYTEST) -q tests -m "not slow and not gpu and not dataset"

# Phase 0. SHA-256 of the four P1 checkpoints vs the phase-9-complete release manifest.
verify-p1: ; $(PY) scripts/verify_p1_checkpoints.py --config configs/conformal.yaml --paths configs/paths.local.yaml --out results/p1_checkpoints.json

splits: ; $(PY) scripts/make_splits.py --config configs/conformal.yaml
level-a: ; $(PY) scripts/run_level_a.py --config configs/conformal.yaml --run keypoint_a2 --crop-source predicted_crop
dump:
	@echo "GPU, ~minutes per domain x arm; TF32 off; writes dumps_root/keypoint_a2/ (gitignored):"
	@echo "  $(PY) scripts/dump_predictions.py --run keypoint_a2 --all-domains --all-arms --device cuda --tf32 off"
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
