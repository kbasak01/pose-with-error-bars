"""The J2 worked example runs, recalibrates a different wireframe, and refuses a retrained model."""

from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "recalibration_example.py"


def _example():
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("recalibration_example")


def test_example_uses_only_the_framework_free_core():
    """A port needs `poseconf.conformal` and provenance only: no torch, no P1, no SPEED+ data."""
    imported = set()
    for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    poseconf = [m for m in imported if m.startswith("poseconf")]
    assert all(m == "poseconf.provenance" or m.startswith("poseconf.conformal") for m in poseconf)
    assert not {m for m in imported if m.split(".")[0] in {"torch", "speedpose", "cv2"}}


def test_example_end_to_end(tmp_path):
    example = _example()
    out = tmp_path / "example.json"
    code = example.main(
        ["--n-cal", "500", "--n-test", "1000", "--repeats", "40", "--out", str(out)]
    )
    result = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and result["refuses_retrained_model"] is True
    assert result["n_keypoints"] == 6 and result["score"] == "C1"
    test = result["test"]
    assert test["n_total"] == 1000 and 0.0 < test["answer_rate"] < 1.0
    rep = result["repeated_draws"]
    # Marginal guarantee over draws: mean coverage >= 1 - alpha, within Monte Carlo error.
    assert rep["mean_coverage"] >= rep["nominal"] - 4 * rep["mc_se"]
    # The uncalibrated covariance under-covers badly; that is what the example exists to show.
    assert result["naive_gaussian"]["coverage_given_answered"] < rep["nominal"] - 0.2
