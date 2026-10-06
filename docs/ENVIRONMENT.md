# Environment

Captured 2026-10-06 (Phase 0) from live tool output on the workstation that runs every experiment
here. Regenerate when the environment changes.

## Hardware and OS

| | |
|---|---|
| GPU | NVIDIA RTX A4000, 16,376 MiB |
| Driver | 596.71 (P1 recorded 595.95 on 2026-08-04, so the driver has changed since P1's numbers) |
| CUDA (driver-reported) | 13.2 |
| CPU / RAM | 36 logical cores / 62 GiB |
| OS | Ubuntu 24.04.4 LTS on WSL2, kernel 6.6.114.1-microsoft-standard-WSL2 |
| git / uv | 2.43.0 / 0.12.1 |

## Python stack (`.venv`, Python 3.11.15)

P1's pinned `external/spacecraft-pose-baseline/requirements.txt`, installed unchanged:

| Package | Version |
|---|---|
| torch | 2.11.0+cu128 (CUDA 12.8, cuDNN 91900, `cuda.is_available()` True) |
| numpy / scipy | 2.4.4 / 1.17.1 |
| opencv-python-headless | 5.0.0.93 |
| timm | 1.0.28 |
| onnx / onnxruntime | 1.22.0 / 1.28.0 |
| pytest / ruff | 9.1.1 / 0.16.1 |

This repo's extras (`requirements-extra.txt`), resolved with P1's frozen environment as constraints:

| Package | Version | Role |
|---|---|---|
| scikit-learn | 1.9.1 | domain classifier, isotonic fit |
| statsmodels | 0.15.0 | Clopper–Pearson cross-check |
| mapie | 1.5.0 | test oracle only |
| torchcp | — | not installable alongside P1's numpy pin (docs/DECISIONS.md, 2026-10-06) |

## Install (as run on 2026-10-06)

```bash
uv venv --python 3.11
uv pip install -r external/spacecraft-pose-baseline/requirements.txt --index-strategy unsafe-best-match
uv pip install -e external/spacecraft-pose-baseline --no-deps
uv pip freeze | grep -v speedpose > /tmp/p1-freeze.txt
uv pip install -r requirements-extra.txt -c /tmp/p1-freeze.txt --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install -e . --no-deps
python -c "import speedpose, poseconf; print('ok')"
```

CI (`.github/workflows/ci.yml`) installs P1's `requirements-ci.txt` (CPU wheels) instead, and uses it
as the constraint file for the extras.
