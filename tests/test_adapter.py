"""`poseconf.p1_adapter`: the single P1 boundary. No dataset, no GPU — sidecars are committed in P1.

HIL sidecars are deliberately never loaded here: they carry labels (CLAUDE.md invariant 4).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from poseconf import p1_adapter
from poseconf.p1_adapter import (
    CROP_SOURCES,
    P1_ROOT,
    load_p1_config,
    load_sidecar,
    p1_commit,
    p1_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Columns present in every keypoint sidecar. SOURCES.md lists the box columns as universal, but P1
#: writes them only where a box is used: `gt_crop` has none, `pipeline_gt_crop` has no IoU (the box
#: is the GT box), `predicted_crop` has all three. Measured 2026-10-06 on keypoint_a2/synthetic.
SIDECAR_COLUMNS = {
    "filename",
    "success",
    "is_outlier",
    "n_inliers",
    "failure_reason",
    "range_m",
    "reprojection_rmse",
    "e_t",
    "e_t_normalised",
    "e_r",
    "e_pose",
    "q_pred",
    "t_pred",
    "confidence_mean",
    "pixel_error_mean",
}
BOX_COLUMNS = {
    "gt_crop": set(),
    "pipeline_gt_crop": {"bbox_used", "bbox_gt"},
    "predicted_crop": {"bbox_used", "bbox_gt", "bbox_iou"},
}


def _p1_conformal_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "conformal.yaml").read_text(encoding="utf-8"))


# --- the submodule --------------------------------------------------------------------------------


def test_speedpose_resolves_to_pinned_submodule() -> None:
    assert (REPO_ROOT / "external" / "spacecraft-pose-baseline").resolve() == P1_ROOT


def test_submodule_at_pinned_commit() -> None:
    assert p1_commit() == _p1_conformal_config()["p1"]["commit"]


def test_only_adapter_imports_speedpose() -> None:
    offenders = []
    for top in ("src", "scripts", "tests"):
        for path in sorted((REPO_ROOT / top).rglob("*.py")):
            if path == Path(p1_adapter.__file__).resolve():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names = [node.module]
                if any(name.split(".")[0] == "speedpose" for name in names):
                    offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"speedpose imported outside p1_adapter: {offenders}"


# --- sidecars -------------------------------------------------------------------------------------


@pytest.mark.parametrize("crop_source", sorted(CROP_SOURCES))
def test_synthetic_sidecar_loads(crop_source: str) -> None:
    sidecar = load_sidecar("keypoint_a2", "synthetic", crop_source)
    assert set(sidecar.fields) >= SIDECAR_COLUMNS | BOX_COLUMNS[crop_source]
    assert {len(v) for v in sidecar.fields.values()} == {len(sidecar)}
    assert sidecar.meta["run"] == "keypoint_a2"
    assert sidecar.meta["domain"] == "synthetic"
    assert sidecar.meta["crop_source"] == crop_source
    assert sidecar.fields["success"].dtype == np.bool_
    assert sidecar.fields["q_pred"].shape == (len(sidecar), 4)
    assert sidecar.fields["t_pred"].shape == (len(sidecar), 3)


def test_box_columns_table_covers_every_crop_source() -> None:
    assert set(BOX_COLUMNS) == set(CROP_SOURCES)


def test_synthetic_sidecar_covers_all_of_validation() -> None:
    """IMPLEMENTATION_PLAN §1.2 asks Phase 0 to confirm the split sizes against P1's audit."""
    audit = json.loads((P1_ROOT / "results" / "dataset_audit.json").read_text(encoding="utf-8"))
    sidecar = load_sidecar("keypoint_a2", "synthetic", "predicted_crop")
    filenames = sidecar.fields["filename"]
    assert len(sidecar) == audit["counts"]["synthetic/validation"]
    assert len(np.unique(filenames)) == len(filenames)


def test_unknown_domain_rejected() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        load_sidecar("keypoint_a2", "validation", "predicted_crop")


def test_unknown_crop_source_rejected() -> None:
    with pytest.raises(ValueError, match="unknown crop_source"):
        load_sidecar("keypoint_a2", "synthetic", "oracle_crop")


def test_missing_run_rejected() -> None:
    with pytest.raises(FileNotFoundError, match="no P1 sidecar"):
        load_sidecar("keypoint_a9", "synthetic", "predicted_crop")


def test_detector_has_no_sidecar() -> None:
    with pytest.raises(FileNotFoundError, match="no sidecars"):
        load_sidecar("detector", "synthetic", "gt_crop")


# --- configs --------------------------------------------------------------------------------------


def test_config_without_local_paths_has_no_dataset_keys() -> None:
    config = load_p1_config("keypoint_a2")
    paths = config["paths"]
    for key in ("wireframe", "pose_convention", "results_root"):
        assert Path(paths[key]).is_absolute()
        assert Path(paths[key]).exists()
        assert Path(paths[key]).is_relative_to(P1_ROOT)
    for key in ("dataset_root", "camera_json", "crop_cache", "keypoint_labels", "output_root"):
        assert key not in paths
    assert config["dataset"]["num_keypoints"] == 11  # inherits: from base.yaml resolved


def test_config_with_local_paths_points_at_them(tmp_path: Path) -> None:
    dataset = tmp_path / "speedplus"
    runs = tmp_path / "runs"
    dataset.mkdir()
    runs.mkdir()
    (dataset / "camera.json").write_text("{}", encoding="utf-8")
    local = p1_adapter.P1Paths(
        speedplus_root=dataset, p1_runs=runs, dumps_root=tmp_path, release_sums=tmp_path / "s"
    )
    paths = load_p1_config("keypoint_a2", paths=local)["paths"]
    assert paths["dataset_root"] == str(dataset)
    assert paths["camera_json"] == str(dataset / "camera.json")
    assert paths["output_root"] == str(runs)
    assert "crop_cache" not in paths  # only when the optional p1_crop_cache key names it


def test_crop_cache_is_set_only_when_named(tmp_path: Path) -> None:
    dataset = tmp_path / "speedplus"
    dataset.mkdir()
    (dataset / "camera.json").write_text("{}", encoding="utf-8")
    local = p1_adapter.P1Paths(
        speedplus_root=dataset,
        p1_runs=tmp_path,
        dumps_root=tmp_path,
        release_sums=tmp_path / "s",
        crop_cache=tmp_path / "crops_320",
    )
    assert load_p1_config("keypoint_a2", paths=local)["paths"]["crop_cache"] == str(
        tmp_path / "crops_320"
    )
    with pytest.raises(FileNotFoundError, match="p1_crop_cache"):
        p1_adapter.keypoint_crop_dataset(
            "keypoint_a2", "train", paths=local, augmentation="a2", seed=0
        )
    with pytest.raises(ValueError, match="split"):
        p1_adapter.keypoint_crop_dataset(
            "keypoint_a2", "test", paths=local, augmentation="a0", seed=0
        )


def test_config_returned_is_a_copy() -> None:
    first = load_p1_config("keypoint_a2")
    first["paths"]["wireframe"] = "mutated"
    assert load_p1_config("keypoint_a2")["paths"]["wireframe"] != "mutated"


def test_unknown_config_rejected() -> None:
    with pytest.raises(FileNotFoundError, match="no P1 config"):
        load_p1_config("keypoint_a9")


# --- paths.local.yaml -----------------------------------------------------------------------------


def _write_local(tmp_path: Path, **overrides: str) -> Path:
    dataset = tmp_path / "speedplus"
    runs = tmp_path / "runs"
    dataset.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)
    (dataset / "camera.json").write_text("{}", encoding="utf-8")
    values = {
        "speedplus_root": str(dataset),
        "p1_runs": str(runs),
        "dumps_root": "dumps",
        "p1_release_sums": str(tmp_path / "SHA256SUMS.txt"),
    }
    values.update(overrides)
    path = tmp_path / "paths.local.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in values.items() if v}), encoding="utf-8")
    return path


def test_p1_paths_valid(tmp_path: Path) -> None:
    resolved = p1_paths(_write_local(tmp_path))
    assert resolved.speedplus_root == tmp_path / "speedplus"
    assert resolved.dumps_root == REPO_ROOT / "dumps"  # relative -> anchored at the repo
    assert resolved.release_sums == tmp_path / "SHA256SUMS.txt"  # need not exist yet


def test_p1_paths_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="paths.local.example.yaml"):
        p1_paths(tmp_path / "absent.yaml")


def test_p1_paths_missing_key(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="p1_release_sums"):
        p1_paths(_write_local(tmp_path, p1_release_sums=""))


def test_p1_paths_placeholder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="placeholder"):
        p1_paths(_write_local(tmp_path, p1_runs="/home/YOU/code/runs"))


def test_p1_paths_example_file_is_rejected() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        p1_paths(REPO_ROOT / "configs" / "paths.local.example.yaml")


def test_p1_paths_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="camera.json"):
        p1_paths(_write_local(tmp_path, speedplus_root=str(tmp_path / "nowhere")))


def test_p1_paths_missing_runs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="p1_runs"):
        p1_paths(_write_local(tmp_path, p1_runs=str(tmp_path / "nowhere")))


def test_local_paths_file_is_gitignored(repo_root: Path) -> None:
    ignore = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "configs/paths.local.yaml" in ignore


# --- the real machine -----------------------------------------------------------------------------


@pytest.mark.dataset
def test_real_local_paths(local_paths) -> None:
    assert (local_paths.speedplus_root / "camera.json").is_file()
    assert local_paths.p1_runs.is_dir()


@pytest.mark.dataset
@pytest.mark.slow
def test_load_pipeline_on_cpu(local_paths) -> None:
    import torch

    pipeline = p1_adapter.load_pipeline(
        "keypoint_a2",
        crop_source="predicted_crop",
        paths=local_paths,
        device=torch.device("cpu"),
    )
    assert pipeline.detector_model is not None
    assert pipeline.wireframe.shape == (11, 3)


# --- pose convention ------------------------------------------------------------------------------


def test_pose_quaternion_order_comes_from_p1() -> None:
    expected = yaml.safe_load(
        (P1_ROOT / "configs" / "pose_convention.yaml").read_text(encoding="utf-8")
    )["quaternion_order"]
    assert p1_adapter.pose_quaternion_order() == expected == "scalar_first"


# --- Phase 2: synthetic labels and filename listings ---------------------------------------------


def test_dataset_audit_counts() -> None:
    counts = p1_adapter.dataset_audit_counts()
    assert counts["synthetic/validation"] == 11994
    assert counts["lightbox/test"] == 6740
    assert counts["sunlamp/test"] == 2791


def test_synthetic_label_loader_has_no_domain_argument() -> None:
    import inspect

    params = inspect.signature(p1_adapter.load_synthetic_labels).parameters
    assert "domain" not in params


def test_synthetic_label_loader_refuses_other_splits(tmp_path: Path) -> None:
    fake = p1_adapter.P1Paths(tmp_path, tmp_path, tmp_path, tmp_path)
    for split in ("test", "lightbox", "val_cal"):
        with pytest.raises(ValueError, match="synthetic split"):
            p1_adapter.load_synthetic_labels(split, paths=fake)


def test_list_image_filenames_rejects_unknown_domain(tmp_path: Path) -> None:
    fake = p1_adapter.P1Paths(tmp_path, tmp_path, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="unknown domain"):
        p1_adapter.list_image_filenames("hil", paths=fake)
    with pytest.raises(FileNotFoundError):
        p1_adapter.list_image_filenames("lightbox", paths=fake)


@pytest.mark.dataset
def test_synthetic_validation_labels_match_sidecar(local_paths) -> None:
    labels = p1_adapter.load_synthetic_labels("validation", paths=local_paths)
    sidecar = load_sidecar("keypoint_a2", "synthetic", "predicted_crop")
    assert len(labels) == len(sidecar) == 11994
    assert sorted(labels.filenames.tolist()) == sidecar.fields["filename"].tolist()
    assert labels.q.shape == (11994, 4) and labels.t.shape == (11994, 3)
    assert np.allclose(np.linalg.norm(labels.q, axis=1), 1.0, atol=1e-5)


@pytest.mark.dataset
@pytest.mark.slow
def test_training_crops_are_p1_samples(local_paths) -> None:
    if local_paths.crop_cache is None:
        pytest.skip("p1_crop_cache not set")
    train = p1_adapter.keypoint_crop_dataset(
        "keypoint_a2", "train", paths=local_paths, augmentation="a2", seed=1337
    )
    assert train.augment and len(train) > 47_000
    sample = train[0]
    assert tuple(sample["image"].shape) == (1, 256, 256)
    assert (
        tuple(sample["keypoints"].shape) == (11, 2)
        and sample["mask"].dtype.is_floating_point is False
    )
    names = p1_adapter.crop_dataset_filenames(train)
    assert names[0] == sample["filename"]
    with pytest.raises(ValueError, match="augmentation"):
        p1_adapter.keypoint_crop_dataset(
            "keypoint_a2", "validation", paths=local_paths, augmentation="a2", seed=1337
        )


def test_wireframe_edges_index_the_11_keypoints() -> None:
    from poseconf.p1_adapter import wireframe_edges

    edges = wireframe_edges()
    assert len(edges) == len(set(edges)) == 15
    assert all(0 <= a < 11 and 0 <= b < 11 and a != b for a, b in edges)
