"""`poseconf.conformal` must stay framework-free (numpy/scipy only) so it ports to the ship-deck project.

Two checks: a static AST scan (catches imports in code paths no test executes) and a runtime check
in a fresh interpreter (catches transitive imports through a dependency).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

CONFORMAL_DIR = Path(__file__).resolve().parents[1] / "src" / "poseconf" / "conformal"

#: Top-level packages the conformal core may not import, directly or via poseconf modules that do.
FORBIDDEN = {
    "torch",
    "torchvision",
    "timm",
    "onnx",
    "onnxruntime",
    "tensorrt",
    "speedpose",
    "mapie",
    "torchcp",
}

#: poseconf modules outside `conformal/` that pull in a forbidden package.
FORBIDDEN_POSECONF = {
    "poseconf.p1_adapter",
    "poseconf.models",
    "poseconf.engine",
    "poseconf.export",
}


def _imports(path: Path) -> set[str]:
    """Fully qualified module names imported by a file (relative imports resolved to poseconf)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add("poseconf." + (node.module or ""))
            elif node.module:
                names.add(node.module)
    return names


def test_conformal_sources_exist() -> None:
    assert sorted(p.name for p in CONFORMAL_DIR.glob("*.py")), "conformal/ has no modules"


def test_no_forbidden_imports_in_conformal() -> None:
    offenders = {}
    for path in sorted(CONFORMAL_DIR.rglob("*.py")):
        bad = {
            name
            for name in _imports(path)
            if name.split(".")[0] in FORBIDDEN
            or any(name == m or name.startswith(m + ".") for m in FORBIDDEN_POSECONF)
        }
        if bad:
            offenders[path.name] = sorted(bad)
    assert not offenders, f"framework imports in poseconf.conformal: {offenders}"


def test_importing_conformal_does_not_load_torch() -> None:
    modules = sorted(f"poseconf.conformal.{p.stem}" for p in CONFORMAL_DIR.glob("*.py"))
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"loaded = sorted(m for m in {sorted(FORBIDDEN)!r} if m in sys.modules)\n"
        "print(','.join(loaded))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", f"loaded via poseconf.conformal: {completed.stdout}"
