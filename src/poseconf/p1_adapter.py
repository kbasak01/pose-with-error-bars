"""The single import boundary to Project 1 (`speedpose`). Nothing else in poseconf imports it.

P1 is a read-only submodule at `external/spacecraft-pose-baseline`, installed editable so its
module-level `REPO_ROOT` resolves inside the submodule. Its configs write every path relative to
that root — including `data/speedplus`, which does not exist inside the submodule and must not be
created there. This module rewrites those paths from `configs/paths.local.yaml` instead, and fails
loudly whenever something it needs is missing.

Labels: HIL sidecars (`lightbox`, `sunlamp`) carry ground-truth-derived columns (`e_r`, `e_t`,
`range_m`, `bbox_gt`, ...), and HIL label files carry poses, keypoints and boxes. CLAUDE.md
invariant 4 allows a HIL label to be read only when evaluating on `*_poolB` or inside an arm tagged
`oracle_*`. Since Phase 3 this module enforces that: `load_sidecar` returns only label-free columns
for a HIL domain unless the requested frames are all poolB or the tag is `oracle_*`, and
`load_eval_labels` applies the same rule.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import speedpose
import yaml
from numpy.typing import NDArray
from speedpose.config import load_config

from poseconf.data.splits import load_split as load_split_manifest
from poseconf.provenance import POSECONF_ROOT

__all__ = [
    "CROP_SOURCES",
    "DOMAINS",
    "HIL_DOMAINS",
    "P1_RESULT_LABEL_FREE_KEYS",
    "P1_ROOT",
    "SIDECAR_LABEL_FREE_COLUMNS",
    "EvalFrames",
    "EvalLabels",
    "HILLabelAccessError",
    "HookCapture",
    "P1Paths",
    "Sidecar",
    "SolveResults",
    "SyntheticLabels",
    "crop_to_full",
    "dataset_audit_counts",
    "eval_frames",
    "eval_split",
    "hil_label_access_allowed",
    "invert_affine",
    "keypoint_hooks",
    "keypoint_labels_file",
    "list_image_filenames",
    "load_eval_labels",
    "load_p1_config",
    "load_pipeline",
    "load_sidecar",
    "load_synthetic_labels",
    "p1_commit",
    "p1_paths",
    "p1_result_counts",
    "pose_errors",
    "pose_quaternion_order",
    "project_keypoints",
    "solve_chunks",
    "solve_frames",
    "solve_many_reference",
    "write_random_init_checkpoints",
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

#: Hardware-in-the-loop domains, whose labels are guarded by CLAUDE.md invariant 4.
HIL_DOMAINS = ("lightbox", "sunlamp")

#: The official split each domain is evaluated on (P1's `evaluate_domain` rule).
_EVAL_SPLITS = {"synthetic": "validation", "lightbox": "test", "sunlamp": "test"}

#: Sidecar columns that are predictions or solver diagnostics, never derived from a label. On a HIL
#: domain these are the only columns `load_sidecar` returns outside poolB / `oracle_*`. `bbox_used`
#: is absent on purpose: on the GT-box arms it *is* the label box.
SIDECAR_LABEL_FREE_COLUMNS = (
    "filename",
    "success",
    "n_inliers",
    "failure_reason",
    "reprojection_rmse",
    "q_pred",
    "t_pred",
    "confidence_mean",
)

#: Tag prefix that licenses reading HIL labels outside poolB (CLAUDE.md invariant 4).
_ORACLE_PREFIX = "oracle_"

#: `solve_many` solves sequentially below this many frames; mirrored by `solve_chunks`.
_P1_PARALLEL_MIN_FRAMES = 256

#: Crop arm -> sidecar filename suffix. P1 writes the gt_crop arm without a suffix
#: (`<domain>_samples.npz`); its `meta.crop_source` says `gt_crop`, which `load_sidecar` checks.
CROP_SOURCES = {
    "gt_crop": "",
    "predicted_crop": "_predicted_crop",
    "pipeline_gt_crop": "_pipeline_gt_crop",
}

#: Image directory inside each SPEED+ domain directory.
_IMAGE_DIRNAME = "images"

#: Image filename pattern inside `_IMAGE_DIRNAME`.
_IMAGE_GLOB = "*.jpg"

#: Synthetic official splits with labels P1 can load.
_SYNTHETIC_SPLITS = ("train", "validation")

#: Keys required in `configs/paths.local.yaml`.
_LOCAL_KEYS = ("speedplus_root", "p1_runs", "dumps_root", "p1_release_sums")

#: Optional key: P1's Phase 2d keypoint-label products (`<domain>_<split>.npz`), needed from Phase 3.
_LABELS_KEY = "p1_keypoint_labels"

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
        keypoint_labels: P1's keypoint-label directory (`<domain>_<split>.npz`), or None when the
            optional `p1_keypoint_labels` key is absent.
    """

    speedplus_root: Path
    p1_runs: Path
    dumps_root: Path
    release_sums: Path
    keypoint_labels: Path | None = None


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


@dataclass(frozen=True)
class SyntheticLabels:
    """Ground-truth poses for one synthetic official split, in label-file order.

    Attributes:
        filenames: (n,) image filenames.
        q: (n, 4) `q_vbs2tango_true`, in the file's element order (P1's convention:
            `pose_quaternion_order()`).
        t: (n, 3) `r_Vo2To_vbs_true`, metres, camera frame.
    """

    filenames: NDArray[np.str_]
    q: NDArray[np.float64]
    t: NDArray[np.float64]

    def __len__(self) -> int:
        """Number of labelled frames."""
        return len(self.filenames)


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
        keypoint_labels=absolute(loaded[_LABELS_KEY]) if _LABELS_KEY in loaded else None,
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
        # The crop cache is never read here (the dump crops through the pipeline). Keypoint labels
        # are set only when the local paths name them; otherwise P1 code needing them fails loudly.
        block.pop("crop_cache", None)
        if paths.keypoint_labels is None:
            block.pop("keypoint_labels", None)
        else:
            block["keypoint_labels"] = str(paths.keypoint_labels)
    return config


class HILLabelAccessError(PermissionError):
    """A HIL label was requested outside poolB evaluation and outside an `oracle_*` arm.

    CLAUDE.md invariant 4 calls this test-set leakage. It is raised, never worked around.
    """


def hil_label_access_allowed(domain: str, frames: Sequence[str] | None, tag: str | None) -> bool:
    """Whether CLAUDE.md invariant 4 permits reading labels of `frames` in `domain`.

    Args:
        domain: One of `DOMAINS`.
        frames: The filenames whose labels would be read, or None for "all of them".
        tag: The calling arm's tag, or None.

    Returns:
        True for synthetic; for HIL, True iff the tag starts with `oracle_` or every frame is in
        the committed `<domain>_poolB` manifest.
    """
    if domain not in HIL_DOMAINS:
        return True
    if tag is not None and tag.startswith(_ORACLE_PREFIX):
        return True
    if frames is None:
        return False
    pool_b = set(load_split_manifest(f"{domain}_poolB"))
    return all(str(name) in pool_b for name in frames)


def load_sidecar(
    run: str,
    domain: str,
    crop_source: str,
    *,
    columns: Sequence[str] | None = None,
    frames: Sequence[str] | None = None,
    tag: str | None = None,
) -> Sidecar:
    """Load a P1 committed per-sample sidecar `results/<run>/<domain>[_<arm>]_samples.npz`.

    No dataset or GPU is needed: sidecars are committed in the submodule. `NpzFile` is lazy, so a
    column not requested is never decompressed.

    On a HIL domain, any column outside `SIDECAR_LABEL_FREE_COLUMNS` is label-derived. It is
    returned only when `hil_label_access_allowed(domain, frames, tag)` holds; otherwise
    `HILLabelAccessError` is raised. Omitting `columns` on a HIL domain requests every column.

    Args:
        run: P1 run, e.g. `"keypoint_a2"`.
        domain: One of `DOMAINS`.
        crop_source: One of `CROP_SOURCES`.
        columns: Columns to load (`filename` is always included), or None for all.
        frames: Restrict the returned rows to these filenames (sidecar order is kept). Required to
            read label-derived HIL columns without an `oracle_*` tag, and then must be poolB.
        tag: The calling arm's tag.

    Returns:
        The loaded sidecar.

    Raises:
        ValueError: On an unknown domain, crop source or column, ragged columns, a `meta` record
            that disagrees with the request, or a requested frame the sidecar lacks.
        FileNotFoundError: If the sidecar does not exist (lists what the run has).
        HILLabelAccessError: On a forbidden HIL label read.
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
        present = [key for key in archive.files if key != "meta"]
        wanted = present if columns is None else list(dict.fromkeys(["filename", *columns]))
        unknown = sorted(set(wanted) - set(present))
        if unknown:
            raise ValueError(f"{path} has no column(s) {unknown}; it has {sorted(present)}")
        label_derived = sorted(set(wanted) - set(SIDECAR_LABEL_FREE_COLUMNS))
        if label_derived and not hil_label_access_allowed(domain, frames, tag):
            raise HILLabelAccessError(
                f"{domain} sidecar columns {label_derived} are label-derived; invariant 4 allows "
                "them only for poolB frames or an oracle_* arm (got "
                f"{'all frames' if frames is None else f'{len(frames)} frames'}, tag={tag!r})"
            )
        fields = {key: archive[key] for key in wanted}
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
    if frames is not None:
        requested = {str(name) for name in frames}
        keep = np.isin(fields["filename"], list(requested))
        if int(keep.sum()) != len(requested):
            raise ValueError(f"{path} lacks {len(requested) - int(keep.sum())} requested frames")
        fields = {key: value[keep] for key, value in fields.items()}
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


def load_synthetic_labels(split: str, *, paths: P1Paths) -> SyntheticLabels:
    """Ground-truth poses for a **synthetic** split, through P1's own label loader.

    There is deliberately no domain argument: HIL labels are never loaded by this function
    (CLAUDE.md invariant 4). Phase 6 adds its own, tagged path for poolB evaluation.

    Args:
        split: `"train"` or `"validation"`.
        paths: Local paths from `p1_paths()`.

    Returns:
        The labels, in label-file order.

    Raises:
        ValueError: On any other split.
        FileNotFoundError: If the label file or image directory is missing.
    """
    from speedpose.data.speedplus import load_split

    if split not in _SYNTHETIC_SPLITS:
        raise ValueError(f"synthetic split must be one of {_SYNTHETIC_SPLITS}, got {split!r}")
    index = load_split(paths.speedplus_root, "synthetic", split)
    return SyntheticLabels(
        filenames=np.array([sample.filename for sample in index.samples]),
        q=np.stack([sample.quaternion for sample in index.samples]).astype(np.float64),
        t=np.stack([sample.translation for sample in index.samples]).astype(np.float64),
    )


def list_image_filenames(domain: str, *, paths: P1Paths) -> list[str]:
    """Sorted image filenames of a domain, from a directory listing only.

    No label file, sidecar or image content is opened, so this is safe for HIL domains before
    any HIL evaluation (it is how the HIL poolA/poolB manifests are drawn).

    Args:
        domain: One of `DOMAINS`.
        paths: Local paths from `p1_paths()`.

    Returns:
        The sorted `*.jpg` names in `<speedplus_root>/<domain>/images/`.

    Raises:
        ValueError: On an unknown domain.
        FileNotFoundError: If the image directory is missing or empty.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    image_dir = paths.speedplus_root / domain / _IMAGE_DIRNAME
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory {image_dir} not found; do not re-download it")
    names = sorted(p.name for p in image_dir.glob(_IMAGE_GLOB))
    if not names:
        raise FileNotFoundError(f"no {_IMAGE_GLOB} files in {image_dir}")
    return names


def dataset_audit_counts() -> dict[str, int]:
    """Frame counts per `<domain>/<split>` from P1's committed `results/dataset_audit.json`."""
    path = P1_ROOT / "results" / "dataset_audit.json"
    return {key: int(value) for key, value in json.loads(path.read_text("utf-8"))["counts"].items()}


# ---------------------------------------------------------------------------------------------
# Phase 3: evaluation frames and labels, keypoint-network hooks, P1-faithful PnP.
# ---------------------------------------------------------------------------------------------


def eval_split(domain: str) -> str:
    """The official split P1 evaluates `domain` on (`validation` for synthetic, else `test`).

    Raises:
        ValueError: On an unknown domain.
    """
    if domain not in _EVAL_SPLITS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    return _EVAL_SPLITS[domain]


def keypoint_labels_file(domain: str, *, paths: P1Paths) -> Path:
    """P1's keypoint-label product for a domain's evaluation split.

    Raises:
        KeyError: If `p1_keypoint_labels` is not set in the local paths.
        FileNotFoundError: If the file is missing (it is never regenerated here).
    """
    if paths.keypoint_labels is None:
        raise KeyError(f"configs/paths.local.yaml has no {_LABELS_KEY!r}; Phase 3 needs it")
    path = paths.keypoint_labels / f"{domain}_{eval_split(domain)}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"P1 keypoint labels {path} not found; do not regenerate them here")
    return path


@dataclass(frozen=True)
class EvalFrames:
    """One domain's evaluation frames in P1's official order, by name only.

    Attributes:
        domain: The domain.
        filenames: (n,) image filenames, in the order P1's evaluation and sidecars use.
        image_paths: Absolute image paths, aligned with `filenames`.
    """

    domain: str
    filenames: NDArray[np.str_]
    image_paths: list[str]

    def __len__(self) -> int:
        """Number of frames."""
        return len(self.filenames)


def eval_frames(domain: str, *, paths: P1Paths) -> EvalFrames:
    """Frames P1 evaluates on, in its order, read from the label product's `filenames` member only.

    P1's `_load_labels` asserts that this member equals the official split JSON row for row, and
    no row was unusable in any committed `keypoint_a2` result, so this is exactly P1's frame list.
    Only the name column is decompressed: nothing label-valued is read, so it is safe on HIL.

    Args:
        domain: One of `DOMAINS`.
        paths: Local paths from `p1_paths()`.

    Returns:
        The frames.

    Raises:
        FileNotFoundError: If the label product or the image directory is missing.
    """
    with np.load(keypoint_labels_file(domain, paths=paths), allow_pickle=False) as archive:
        filenames = np.asarray(archive["filenames"]).astype(np.str_)
    image_dir = paths.speedplus_root / domain / _IMAGE_DIRNAME
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory {image_dir} not found; do not re-download it")
    return EvalFrames(
        domain=domain,
        filenames=filenames,
        image_paths=[str(image_dir / name) for name in filenames],
    )


@dataclass(frozen=True)
class EvalLabels:
    """Ground truth for a set of evaluation frames, in the requested order.

    Attributes:
        filenames: (n,) filenames.
        q: (n, 4) GT quaternions, P1's label order.
        t: (n, 3) GT translations, metres.
        keypoints_2d: (n, K, 2) GT keypoint projections, full-frame px (NaN where invalid).
        in_frame: (n, K) P1's `visible`: positive depth and inside the full frame — the §1.3
            keypoint-inclusion rule.
        bbox_tight: (n, 4) tight GT box, full-frame px.
    """

    filenames: NDArray[np.str_]
    q: NDArray[np.float64]
    t: NDArray[np.float64]
    keypoints_2d: NDArray[np.float64]
    in_frame: NDArray[np.bool_]
    bbox_tight: NDArray[np.float64]

    def __len__(self) -> int:
        """Number of frames."""
        return len(self.filenames)


def load_eval_labels(
    domain: str, *, paths: P1Paths, frames: Sequence[str], tag: str | None
) -> EvalLabels:
    """Ground truth for `frames`, guarded by CLAUDE.md invariant 4 on HIL domains.

    Args:
        domain: One of `DOMAINS`.
        paths: Local paths from `p1_paths()`.
        frames: Filenames to return, in this order.
        tag: The calling arm's tag. On HIL, frames outside poolB need an `oracle_*` tag.

    Returns:
        The labels.

    Raises:
        HILLabelAccessError: On a forbidden HIL label read.
        ValueError: If the label product and the official split disagree, or a frame is unknown.
    """
    from speedpose.data.speedplus import load_split

    if not hil_label_access_allowed(domain, frames, tag):
        raise HILLabelAccessError(
            f"{domain} labels requested for frames outside poolB with tag={tag!r}; invariant 4 "
            "allows them only for poolB evaluation or an oracle_* arm"
        )
    index = load_split(paths.speedplus_root, domain, eval_split(domain))
    with np.load(keypoint_labels_file(domain, paths=paths), allow_pickle=False) as archive:
        label_names = [str(name) for name in archive["filenames"]]
        if [sample.filename for sample in index.samples] != label_names:
            raise ValueError(f"{domain} label product and official split disagree row for row")
        position = {name: row for row, name in enumerate(label_names)}
        missing = [name for name in frames if str(name) not in position]
        if missing:
            raise ValueError(f"{len(missing)} requested {domain} frames are not in the split")
        rows = np.array([position[str(name)] for name in frames], dtype=np.int64)
        keypoints = np.asarray(archive["keypoints_2d"], dtype=np.float64)[rows]
        visible = np.asarray(archive["visible"], dtype=bool)[rows]
        boxes = np.asarray(archive["bbox_tight"], dtype=np.float64)[rows]
    return EvalLabels(
        filenames=np.asarray(frames).astype(np.str_),
        q=np.stack([index.samples[row].quaternion for row in rows]).astype(np.float64),
        t=np.stack([index.samples[row].translation for row in rows]).astype(np.float64),
        keypoints_2d=keypoints,
        in_frame=visible,
        bbox_tight=boxes,
    )


@dataclass
class HookCapture:
    """Tensors captured by `keypoint_hooks` during the last `pipeline.keypoints()` call.

    Attributes:
        heatmaps: (B, K, H, W) raw heatmaps, the third output of `KeypointNet.forward`.
        encoder_pooled: (B, C_enc) global mean of the encoder's last feature map, fp32.
        decoder: (B, C_dec, H, W) input to the final 1x1 conv, or None when `keep_decoder` is off.
        keep_decoder: Whether the next forward should keep the decoder features.
    """

    heatmaps: Any = None
    encoder_pooled: Any = None
    decoder: Any = None
    keep_decoder: bool = False


@contextmanager
def keypoint_hooks(pipeline: Any, *, keep_decoder: bool = False) -> Iterator[HookCapture]:
    """Observe P1's keypoint network without changing it.

    Three hooks, all returning None so no output is replaced: a forward hook on the network for
    its heatmaps, one on the encoder for the pooled last feature map, and a forward *pre*-hook on
    the final 1x1 conv for the decoder features feeding it. The pose path is untouched, which
    `tests/test_dump.py` checks bit for bit.

    Args:
        pipeline: A `PosePipeline` (from `load_pipeline`).
        keep_decoder: Initial value of `HookCapture.keep_decoder`; may be toggled per batch.

    Yields:
        The capture, overwritten on every forward.

    Raises:
        TypeError: If the network does not end in the expected 1x1 conv.
    """
    import torch

    model = pipeline.keypoint_model
    final = model.decoder[-1]
    if not isinstance(final, torch.nn.Conv2d):
        raise TypeError(f"expected KeypointNet.decoder to end in a Conv2d, got {type(final)}")
    capture = HookCapture(keep_decoder=keep_decoder)

    def on_model(_module: Any, _inputs: Any, output: Any) -> None:
        capture.heatmaps = output[2].detach()

    def on_encoder(_module: Any, _inputs: Any, output: Any) -> None:
        capture.encoder_pooled = output[-1].detach().float().mean(dim=(2, 3))

    def on_final_conv(_module: Any, inputs: Any) -> None:
        capture.decoder = inputs[0].detach() if capture.keep_decoder else None

    handles = [
        model.register_forward_hook(on_model),
        model.encoder.register_forward_hook(on_encoder),
        final.register_forward_pre_hook(on_final_conv),
    ]
    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()


def crop_to_full(coords: NDArray[np.float64], affine: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map (K, 2) crop-pixel points to full-frame pixels with P1's own inversion.

    Args:
        coords: (K, 2) crop-pixel points.
        affine: (2, 3) full-frame-to-crop affine from `PosePipeline.crop_one`.

    Returns:
        (K, 2) full-frame points.
    """
    from speedpose.geometry.projection import invert_affine as p1_invert_affine
    from speedpose.geometry.projection import transform_points

    return transform_points(coords, p1_invert_affine(affine))


def invert_affine(affine: NDArray[np.float64]) -> NDArray[np.float64]:
    """P1's `invert_affine` (crop-to-full-frame), for mapping covariances."""
    from speedpose.geometry.projection import invert_affine as p1_invert_affine

    return p1_invert_affine(affine)


@dataclass(frozen=True)
class SolveResults:
    """Per-frame PnP outcomes, aligned with the input frames.

    Attributes:
        success: (n,) whether a usable pose was recovered (P1's `solve_batch` rule).
        q: (n, 4) predicted quaternion in P1's label order (canonical sign), NaN on failure.
        t: (n, 3) predicted translation, metres, NaN on failure.
        n_inliers: (n,) RANSAC inliers (P1 keeps the solver's count on failure).
        reprojection_rmse: (n,) full-frame px over inliers, NaN where P1 has None.
        failure_reason: (n,) reason string, empty on success.
        chunks: The `(start, stop, seed)` chunks solved, for provenance.
    """

    success: NDArray[np.bool_]
    q: NDArray[np.float64]
    t: NDArray[np.float64]
    n_inliers: NDArray[np.int32]
    reprojection_rmse: NDArray[np.float64]
    failure_reason: NDArray[np.str_]
    chunks: list[tuple[int, int, int]] = field(default_factory=list)


def solve_chunks(total: int, *, workers: int, seed: int) -> list[tuple[int, int, int]]:
    """The `(start, stop, rng_seed)` chunks P1's `engine.pose.solve_many` would use.

    `solvePnPRansac` draws from OpenCV's process-global RNG, seeded once per chunk, so a frame's
    solve depends on which chunk it lands in. Reproducing P1's sidecars therefore needs P1's
    chunking: `effective = min(max(workers, 1), os.cpu_count())`; below 2 workers or 256 frames
    one sequential chunk with `seed`; otherwise `np.linspace` bounds, chunk i seeded `seed + i`
    (empty chunks skipped but still counted). `tests/test_dump.py` checks the outcome against
    `solve_many` itself.

    Args:
        total: Number of frames.
        workers: P1's `runtime.num_workers`.
        seed: P1's `runtime.seed`.

    Returns:
        The chunks, in order.
    """
    if total == 0:
        return []
    effective = min(max(workers, 1), os.cpu_count() or 1)
    if effective <= 1 or total < _P1_PARALLEL_MIN_FRAMES:
        return [(0, total, seed)]
    bounds = np.linspace(0, total, effective + 1).astype(int)
    return [
        (int(start), int(stop), seed + index)
        for index, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
        if stop > start
    ]


def _solve_chunk(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Solve one chunk with P1's `solve_single`, seeded as `solve_batch` seeds it.

    Args:
        payload: Arrays for the chunk plus the shared geometry and config.

    Returns:
        One `(success, q, t, n_inliers, rmse, reason)` tuple per frame.
    """
    import cv2
    from speedpose.engine.pose import solve_single
    from speedpose.geometry.conventions import rotation_matrix_to_quat

    if payload["in_worker"]:
        cv2.setNumThreads(0)
    cv2.setRNGSeed(int(payload["seed"]))
    rows: list[tuple[Any, ...]] = []
    for coords, confidence, affine in zip(
        payload["coords"], payload["confidence"], payload["affines"], strict=True
    ):
        result = solve_single(
            coords=coords,
            confidence=confidence,
            affine=affine,
            wireframe=payload["wireframe"],
            intrinsics=payload["intrinsics"],
            convention=payload["convention"],
            config=payload["config"],
        )
        if not result.success or result.rotation is None or result.translation is None:
            rows.append(
                (
                    False,
                    None,
                    None,
                    result.n_inliers,
                    result.reprojection_rmse,
                    result.failure_reason,
                )
            )
            continue
        rows.append(
            (
                True,
                rotation_matrix_to_quat(result.rotation, payload["convention"]),
                np.asarray(result.translation, dtype=np.float64),
                result.n_inliers,
                result.reprojection_rmse,
                None,
            )
        )
    return rows


def solve_frames(
    pipeline: Any,
    coords: NDArray[np.float64],
    confidence: NDArray[np.float64],
    affines: NDArray[np.float64],
    *,
    workers: int,
    seed: int,
) -> SolveResults:
    """Solve PnP for a whole pass exactly as P1's evaluation does, keeping every solver output.

    Calls `speedpose.engine.pose.solve_single` per frame with the pipeline's wireframe,
    intrinsics, convention and config, in P1's chunks (`solve_chunks`), across processes when P1
    would. Unlike `solve_many` it needs no ground truth, so it is label-free.

    Args:
        pipeline: The `PosePipeline` whose geometry and config to use.
        coords: (n, K, 2) predicted keypoints, crop px.
        confidence: (n, K) per-keypoint confidence.
        affines: (n, 2, 3) full-frame-to-crop affines.
        workers: P1's `runtime.num_workers`.
        seed: P1's `runtime.seed`.

    Returns:
        The per-frame results.
    """
    total = len(coords)
    chunks = solve_chunks(total, workers=workers, seed=seed)
    parallel = len(chunks) > 1
    shared = {
        "wireframe": pipeline.wireframe,
        "intrinsics": pipeline.intrinsics,
        "convention": pipeline.convention,
        "config": pipeline.config,
        "in_worker": bool(parallel),
    }
    payloads = [
        {
            "coords": coords[start:stop],
            "confidence": confidence[start:stop],
            "affines": affines[start:stop],
            "seed": chunk_seed,
            **shared,
        }
        for start, stop, chunk_seed in chunks
    ]
    rows: list[tuple[Any, ...]] = []
    if parallel:
        with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
            for chunk_rows in pool.map(_solve_chunk, payloads):
                rows.extend(chunk_rows)
    else:
        for payload in payloads:
            rows.extend(_solve_chunk(payload))

    q = np.full((total, 4), np.nan)
    t = np.full((total, 3), np.nan)
    for index, row in enumerate(rows):
        if row[0]:
            q[index], t[index] = row[1], row[2]
    return SolveResults(
        success=np.array([row[0] for row in rows], dtype=bool),
        q=q,
        t=t,
        n_inliers=np.array([row[3] for row in rows], dtype=np.int32),
        reprojection_rmse=np.array(
            [np.nan if row[4] is None else row[4] for row in rows], dtype=np.float64
        ),
        failure_reason=np.array([row[5] or "" for row in rows], dtype=np.str_),
        chunks=chunks,
    )


def pose_errors(
    q_gt: NDArray[np.float64],
    t_gt: NDArray[np.float64],
    q_pred: NDArray[np.float64],
    t_pred: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """P1's `e_r` (rad) and `e_t` (m) per frame, NaN where the prediction is NaN.

    Args:
        q_gt: (n, 4) GT quaternions.
        t_gt: (n, 3) GT translations.
        q_pred: (n, 4) predicted quaternions (NaN rows = failures).
        t_pred: (n, 3) predicted translations.

    Returns:
        `(e_r, e_t)`, each (n,).
    """
    from speedpose.geometry.metrics import pose_errors as p1_pose_errors

    n = len(q_gt)
    e_r = np.full(n, np.nan)
    e_t = np.full(n, np.nan)
    for index in range(n):
        if not (np.isfinite(q_pred[index]).all() and np.isfinite(t_pred[index]).all()):
            continue
        errors = p1_pose_errors(
            q_gt[index], t_gt[index], q_pred[index], t_pred[index], apply_hil_thresholds=False
        )
        e_r[index], e_t[index] = errors.e_r, errors.e_t
    return e_r, e_t


def project_keypoints(
    q: NDArray[np.float64], t: NDArray[np.float64], *, run: str = "keypoint_a2", camera_json: Path
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Project P1's wireframe with P1's convention and projection (for test fixtures).

    Args:
        q: (4,) quaternion, P1's label order.
        t: (3,) translation, metres.
        run: P1 config naming the wireframe and convention.
        camera_json: A SPEED+-format `camera.json`.

    Returns:
        `(points, valid)`: (K, 2) full-frame px and (K,) positive-depth mask.
    """
    from speedpose.geometry.camera import load_camera
    from speedpose.geometry.conventions import load_convention, quat_to_rotation_matrix
    from speedpose.geometry.projection import load_wireframe, project_points

    block = load_p1_config(run)["paths"]
    convention = load_convention(block["pose_convention"])
    return project_points(
        load_wireframe(block["wireframe"]),
        quat_to_rotation_matrix(np.asarray(q, dtype=np.float64), convention),
        np.asarray(t, dtype=np.float64),
        load_camera(camera_json),
        apply_distortion=convention.apply_distortion,
    )


def solve_many_reference(
    pipeline: Any,
    coords: NDArray[np.float64],
    confidence: NDArray[np.float64],
    affines: NDArray[np.float64],
    *,
    workers: int,
    seed: int,
) -> list[Any]:
    """P1's own `solve_many`, with dummy GT poses (tests compare it with `solve_frames`).

    Returns:
        P1 `PoseEstimate`s.
    """
    from speedpose.engine.pose import solve_many

    total = len(coords)
    identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (total, 1))
    return solve_many(
        coords=coords,
        confidence=confidence,
        affines=affines,
        quaternions=identity,
        translations=np.tile(np.array([0.0, 0.0, 10.0]), (total, 1)),
        filenames=[f"f{index}" for index in range(total)],
        wireframe=pipeline.wireframe,
        intrinsics=pipeline.intrinsics,
        convention=pipeline.convention,
        config=pipeline.config,
        apply_hil_thresholds=False,
        workers=workers,
        rng_seed=seed,
    )


def write_random_init_checkpoints(
    directory: Path, *, run: str = "keypoint_a2", paths: P1Paths | None = None, seed: int = 0
) -> tuple[Path, Path]:
    """Write random-init detector and keypoint checkpoints in P1's format (test fixtures only).

    Args:
        directory: Destination; `detector/best.pt` and `<run>/best.pt` are created under it.
        run: Keypoint run whose config to embed.
        paths: Local paths to resolve the configs with (a fixture tree), or None.
        seed: Torch seed for the initialisation.

    Returns:
        `(detector_checkpoint, keypoint_checkpoint)`.
    """
    import torch
    from speedpose.engine.train import build_model

    torch.manual_seed(seed)
    written = []
    for name in ("detector", run):
        config = load_p1_config(name, paths=paths)
        model = build_model(config, pretrained=False)
        destination = Path(directory) / name / "best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = {"config": config, "ema": {"shadow": model.state_dict()}, "git_sha": "random-init"}
        torch.save(state, destination)
        written.append(destination)
    return written[0], written[1]


#: Label-free aggregate keys of a P1 results JSON (counts of solves, not errors).
P1_RESULT_LABEL_FREE_KEYS = ("count", "solved_count", "solved_rate", "pnp_failures", "split")


def p1_result_counts(run: str, domain: str, crop_source: str) -> dict[str, Any]:
    """Label-free solve counts from P1's committed `results/<run>/<domain>[_<arm>].json`.

    Only `P1_RESULT_LABEL_FREE_KEYS` are returned; error aggregates in the same file are dropped
    here, so a HIL caller never receives a label-derived number through this function.

    Args:
        run: P1 run.
        domain: One of `DOMAINS`.
        crop_source: One of `CROP_SOURCES`.

    Returns:
        The selected keys.

    Raises:
        FileNotFoundError: If the results file is missing.
        KeyError: If a key is absent.
    """
    if crop_source not in CROP_SOURCES:
        raise ValueError(f"unknown crop_source {crop_source!r}")
    path = P1_ROOT / "results" / run / f"{domain}{CROP_SOURCES[crop_source]}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no P1 results file {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    return {key: record[key] for key in P1_RESULT_LABEL_FREE_KEYS}
