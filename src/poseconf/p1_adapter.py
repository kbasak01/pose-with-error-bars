"""The single import boundary to Project 1 (`speedpose`). Nothing else in poseconf imports it.

P1 is a read-only submodule at `external/spacecraft-pose-baseline`, installed editable so its
module-level `REPO_ROOT` resolves inside the submodule. Its configs write every path relative to
that root — including `data/speedplus`, which does not exist inside the submodule and must not be
created there. This module rewrites those paths from `configs/paths.local.yaml` instead, and fails
loudly whenever something it needs is missing.

Labels: HIL sidecars (`lightbox`, `sunlamp`) carry ground-truth-derived columns (`e_r`, `e_t`,
`range_m`, `bbox_gt`, ...). Loading one is allowed by this module, but callers are bound by
CLAUDE.md invariant 4: HIL labels may be read only when evaluating on `*_poolB` or inside an arm
tagged `oracle_*`.
"""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import speedpose
import yaml
from numpy.typing import NDArray
from speedpose.config import load_config

from poseconf.provenance import POSECONF_ROOT

__all__ = [
    "CROP_SOURCES",
    "DOMAINS",
    "P1_ROOT",
    "P1Paths",
    "Sidecar",
    "load_p1_config",
    "load_pipeline",
    "load_sidecar",
    "p1_commit",
    "p1_paths",
    "pose_quaternion_order",
]

#: Where the submodule must live, relative to this repository.
_SUBMODULE_REL = Path("external") / "spacecraft-pose-baseline"

#: The P1 repository root, derived from the imported package. `src/speedpose/__init__.py` -> two up.
P1_ROOT = Path(speedpose.__file__).resolve().parents[2]

_pinned_root = (POSECONF_ROOT / _SUBMODULE_REL).resolve()
if _pinned_root != P1_ROOT:
    raise ImportError(
        f"speedpose is imported from {P1_ROOT}, not this repo's pinned submodule "
        f"{_pinned_root}. Reinstall it with "
        f"`uv pip install -e {_SUBMODULE_REL} --no-deps` so results come from the pinned P1."
    )

#: Evaluation domains P1 produced sidecars for.
DOMAINS = ("synthetic", "lightbox", "sunlamp")

#: Crop arm -> sidecar filename suffix. P1 writes the gt_crop arm without a suffix
#: (`<domain>_samples.npz`); its `meta.crop_source` says `gt_crop`, which `load_sidecar` checks.
CROP_SOURCES = {
    "gt_crop": "",
    "predicted_crop": "_predicted_crop",
    "pipeline_gt_crop": "_pipeline_gt_crop",
}

#: Keys required in `configs/paths.local.yaml`.
_LOCAL_KEYS = ("speedplus_root", "p1_runs", "dumps_root", "p1_release_sums")

#: The placeholder the example file ships with; a copy that still has it was never edited.
_PLACEHOLDER = "/home/YOU/"

#: Default local-paths file.
DEFAULT_LOCAL_PATHS = POSECONF_ROOT / "configs" / "paths.local.yaml"

#: P1 `paths:` keys that point at the dataset or at data products derived from it. Without local
#: paths they are removed, so P1 code that needs them raises KeyError instead of silently resolving
#: to a nonexistent `external/.../data/` directory.
_DATA_PATH_KEYS = ("dataset_root", "camera_json", "crop_cache", "keypoint_labels")

#: P1 `paths:` keys that point at committed files inside the submodule.
_P1_RELATIVE_KEYS = ("wireframe", "pose_convention", "results_root")


@dataclass(frozen=True)
class P1Paths:
    """Machine-local paths, from `configs/paths.local.yaml`.

    Attributes:
        speedplus_root: SPEED+ tree (contains `camera.json`). Read-only.
        p1_runs: P1 checkpoint tree, `<run>/best.pt`.
        dumps_root: Where Phase 3 writes prediction dumps (gitignored); need not exist yet.
        release_sums: P1 `phase-9-complete` release `SHA256SUMS.txt`; checked by the verify script.
    """

    speedplus_root: Path
    p1_runs: Path
    dumps_root: Path
    release_sums: Path


@dataclass(frozen=True)
class Sidecar:
    """One P1 per-sample sidecar, fully loaded.

    Attributes:
        fields: Column name -> per-frame array (all with the same leading length).
        meta: P1's provenance record (run, domain, split, crop_source, checkpoint, git_sha, ...).
        path: The file it came from.
    """

    fields: dict[str, NDArray[Any]]
    meta: dict[str, Any]
    path: Path

    def __len__(self) -> int:
        """Number of frames."""
        return len(self.fields["filename"])


def p1_paths(path: str | Path = DEFAULT_LOCAL_PATHS) -> P1Paths:
    """Read and validate `configs/paths.local.yaml`.

    Args:
        path: The local-paths YAML.

    Returns:
        Resolved, validated paths.

    Raises:
        FileNotFoundError: If the file, the dataset's `camera.json`, or the runs directory is missing.
        KeyError: If a required key is absent.
        ValueError: If the file is not a mapping or still holds the example's placeholder paths.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Copy configs/paths.local.example.yaml to it and set the real paths."
        )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    missing = [key for key in _LOCAL_KEYS if key not in loaded]
    if missing:
        raise KeyError(f"{path} is missing required key(s) {missing}")
    placeholders = [key for key in _LOCAL_KEYS if _PLACEHOLDER in str(loaded[key])]
    if placeholders:
        raise ValueError(f"{path} still holds the example placeholder for {placeholders}")

    def absolute(value: Any) -> Path:
        candidate = Path(str(value)).expanduser()
        return candidate if candidate.is_absolute() else (POSECONF_ROOT / candidate).resolve()

    resolved = P1Paths(
        speedplus_root=absolute(loaded["speedplus_root"]),
        p1_runs=absolute(loaded["p1_runs"]),
        dumps_root=absolute(loaded["dumps_root"]),
        release_sums=absolute(loaded["p1_release_sums"]),
    )
    if not (resolved.speedplus_root / "camera.json").is_file():
        raise FileNotFoundError(
            f"speedplus_root={resolved.speedplus_root} has no camera.json; the dataset is not "
            "where paths.local.yaml says. Do not re-download it — fix the path."
        )
    if not resolved.p1_runs.is_dir():
        raise FileNotFoundError(f"p1_runs={resolved.p1_runs} is not a directory")
    return resolved


def load_p1_config(name: str, *, paths: P1Paths | None = None) -> dict[str, Any]:
    """Load a P1 config by name, with its `paths:` block made absolute.

    Paths to committed P1 files (wireframe, pose convention, results) resolve inside the submodule.
    With `paths`, the dataset paths come from `speedplus_root` and `output_root` from `p1_runs`.
    Without it, the data-dependent keys are removed so that any P1 code touching the dataset fails
    with a KeyError rather than resolving to a directory that does not exist.

    Args:
        name: Config stem in P1's `configs/`, e.g. `"keypoint_a2"`.
        paths: Local paths from `p1_paths()`, or None for a dataset-free config.

    Returns:
        The merged config (P1's `inherits:` resolved). A fresh copy; mutating it is safe.

    Raises:
        FileNotFoundError: If no such config exists, or a committed P1 file it names is missing.
    """
    config_path = P1_ROOT / "configs" / f"{name}.yaml"
    if not config_path.is_file():
        available = sorted(p.stem for p in (P1_ROOT / "configs").glob("*.yaml"))
        raise FileNotFoundError(f"no P1 config {name!r}; available: {available}")
    config = copy.deepcopy(load_config(config_path))
    block = config["paths"]

    for key in _P1_RELATIVE_KEYS:
        resolved = (P1_ROOT / block[key]).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"P1 config {name!r} paths.{key} -> {resolved} does not exist")
        block[key] = str(resolved)

    if paths is None:
        for key in _DATA_PATH_KEYS:
            block.pop(key, None)
        block.pop("output_root", None)
    else:
        block["dataset_root"] = str(paths.speedplus_root)
        block["camera_json"] = str(paths.speedplus_root / "camera.json")
        block["output_root"] = str(paths.p1_runs)
        # Derived data products (crop cache, keypoint labels) are not needed before Phase 3.
        block.pop("crop_cache", None)
        block.pop("keypoint_labels", None)
    return config


def load_sidecar(run: str, domain: str, crop_source: str) -> Sidecar:
    """Load a P1 committed per-sample sidecar `results/<run>/<domain>[_<arm>]_samples.npz`.

    No dataset or GPU is needed: sidecars are committed in the submodule.

    Args:
        run: P1 run, e.g. `"keypoint_a2"`.
        domain: One of `DOMAINS`. HIL domains carry labels — see the module docstring.
        crop_source: One of `CROP_SOURCES`.

    Returns:
        The loaded sidecar.

    Raises:
        ValueError: On an unknown domain or crop source, ragged columns, or a `meta` record that
            disagrees with the request.
        FileNotFoundError: If the sidecar does not exist (lists what the run has).
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    if crop_source not in CROP_SOURCES:
        raise ValueError(
            f"unknown crop_source {crop_source!r}; expected one of {list(CROP_SOURCES)}"
        )

    run_dir = P1_ROOT / "results" / run
    path = run_dir / f"{domain}{CROP_SOURCES[crop_source]}_samples.npz"
    if not path.is_file():
        available = (
            sorted(p.name for p in run_dir.glob("*_samples.npz")) if run_dir.is_dir() else []
        )
        raise FileNotFoundError(f"no P1 sidecar {path}; {run!r} has {available or 'no sidecars'}")

    with np.load(path, allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files if key != "meta"}
        if "meta" not in archive.files:
            raise ValueError(f"{path} has no meta record")
        meta = json.loads(str(archive["meta"]))

    expected = {"run": run, "domain": domain, "crop_source": crop_source}
    disagree = {k: (v, meta.get(k)) for k, v in expected.items() if meta.get(k) != v}
    if disagree:
        raise ValueError(f"{path} meta disagrees with the request (wanted, found): {disagree}")
    lengths = {key: len(value) for key, value in fields.items()}
    if "filename" not in fields or len(set(lengths.values())) != 1:
        raise ValueError(f"{path} has ragged or missing columns: {lengths}")
    return Sidecar(fields=fields, meta=meta, path=path)


def load_pipeline(
    run: str,
    *,
    crop_source: str,
    paths: P1Paths,
    device: Any = None,
) -> Any:
    """Build P1's `PosePipeline` for a run, from the local checkpoints.

    Args:
        run: Keypoint run, e.g. `"keypoint_a2"`; its checkpoint is `p1_runs/<run>/best.pt`.
        crop_source: `"predicted_crop"` uses `p1_runs/detector/best.pt`; `"gt_crop"` runs without a
            detector.
        paths: Local paths from `p1_paths()`.
        device: A `torch.device`, or None for P1's default.

    Returns:
        A `speedpose.pipeline.PosePipeline`.

    Raises:
        ValueError: On a crop source the pipeline does not support.
        FileNotFoundError: If a checkpoint is missing.
    """
    # Imported here: the pipeline pulls in torch, timm and cv2, which nothing else here needs.
    from speedpose.pipeline import PosePipeline

    if crop_source not in ("predicted_crop", "gt_crop"):
        raise ValueError(f"load_pipeline supports predicted_crop or gt_crop, got {crop_source!r}")
    keypoint_checkpoint = paths.p1_runs / run / "best.pt"
    detector_checkpoint = (
        paths.p1_runs / "detector" / "best.pt" if crop_source == "predicted_crop" else None
    )
    for checkpoint in (keypoint_checkpoint, detector_checkpoint):
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"P1 checkpoint {checkpoint} not found")
    return PosePipeline(
        config=load_p1_config(run, paths=paths),
        detector_checkpoint=detector_checkpoint,
        keypoint_checkpoint=keypoint_checkpoint,
        dataset_root=paths.speedplus_root,
        device=device,
    )


def pose_quaternion_order(run: str = "keypoint_a2") -> str:
    """P1's quaternion component order, read through P1's own convention loader.

    `poseconf.conformal` cannot import this module, so callers pass the returned string into
    `poseconf.conformal.so3` functions that need it. No dataset is touched: the convention file is
    committed in the submodule.

    Args:
        run: P1 config whose `paths.pose_convention` to read.

    Returns:
        `"scalar_first"` or `"scalar_last"`.
    """
    from speedpose.geometry.conventions import load_convention

    convention_path = load_p1_config(run)["paths"]["pose_convention"]
    return load_convention(convention_path).quaternion_order


def p1_commit() -> str:
    """The submodule's checked-out commit (read-only `git rev-parse`).

    Raises:
        RuntimeError: If git fails.
    """
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=P1_ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git rev-parse failed in {P1_ROOT}: {completed.stderr.strip()}")
    return completed.stdout.strip()
