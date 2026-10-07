"""Level B engine: dump/label join, label-free sets, PURSE agreement, propagation runners."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from poseconf.conformal import propagate as pr
from poseconf.conformal.so3 import quat_to_matrix
from poseconf.engine import level_b as lb

REPO_ROOT = Path(__file__).resolve().parents[1]
K = 11
GEOM = pr.CameraGeometry(
    wireframe=np.random.default_rng(0).uniform(-0.6, 0.6, size=(K, 3)),
    camera_matrix=np.array([[2988.58, 0.0, 960.0], [0.0, 2988.34, 600.0], [0.0, 0.0, 1.0]]),
    distortion=np.array([-0.2238, 0.5141, -0.000665, -0.000214, -0.1312]),
    width=1920,
    height=1200,
    min_depth=1e-9,
    max_normalised_radius=1.6,
)
ORDER = "scalar_first"


def _fixture(n: int = 60, seed: int = 1) -> tuple[dict, dict, list[str]]:
    """A prediction dump and label file in P1's layout, rows in different orders."""
    rng = np.random.default_rng(seed)
    q_gt = np.roll(Rotation.random(n, random_state=seed).as_quat(), 1, axis=1)
    z = rng.uniform(4.0, 15.0, n)
    t_gt = np.column_stack([rng.uniform(-0.15, 0.15, (n, 2)) * z[:, None], z])
    y_gt, visible = pr.project(quat_to_matrix(q_gt, ORDER), t_gt, GEOM)
    names = [f"img{i:05d}.jpg" for i in range(n)]
    labels = {
        "filename": np.asarray(names),
        "q_gt": q_gt,
        "t_gt": t_gt,
        "kp_gt_full": y_gt + 1e-4,  # P1's stored product differs from the projection by ~1e-3 px
        "in_frame": visible,
        "bbox_gt": np.tile([100.0, 100.0, 600.0, 500.0], (n, 1)),
    }
    order = rng.permutation(n)
    noise = Rotation.from_rotvec(rng.normal(scale=0.01, size=(n, 3)))
    q_hat = np.roll((noise * Rotation.from_quat(np.roll(q_gt, -1, axis=1))).as_quat(), 1, axis=1)
    t_hat = t_gt + rng.normal(scale=0.01, size=(n, 3)) * z[:, None]
    success = rng.uniform(size=n) > 0.15
    q_hat[~success], t_hat[~success] = np.nan, np.nan
    y_hat = np.nan_to_num(y_gt, nan=300.0) + rng.normal(scale=3.0, size=(n, K, 2))
    lin = rng.uniform(0.3, 0.6, n)[:, None, None] * np.eye(2)
    affine = np.concatenate([lin, rng.normal(scale=50, size=(n, 2, 1))], axis=2)
    a = rng.normal(size=(n, K, 2, 2))
    cov_crop = a @ a.transpose(0, 1, 3, 2) + np.eye(2)
    empty = np.zeros((n, K), dtype=bool)
    empty[3, 4] = True
    cov_crop[empty] = 0.0
    dump = {
        "filename": np.asarray(names)[order],
        "kp_pred_full": y_hat[order],
        "heatmap_cov_crop": cov_crop[order],
        "affine": affine[order],
        "heatmap_cov_full": pr.crop_cov_to_full(cov_crop, affine)[order],
        "heatmap_empty": empty[order],
        "bbox_used": np.tile([100.0, 100.0, 400.0, 500.0], (n, 1))[order],
        "success": success[order],
        "q_pred": q_hat[order],
        "t_pred": t_hat[order],
    }
    return dump, labels, names


def test_join_dump_aligns_rows_and_projects_labels() -> None:
    dump, labels, names = _fixture()
    pick = names[10:40]
    frames, check = lb.join_dump(dump, labels, pick, GEOM, ORDER)
    assert list(frames.filenames) == pick
    row = {name: i for i, name in enumerate(dump["filename"])}
    for j, name in enumerate(pick):
        np.testing.assert_array_equal(frames.y_hat[j], dump["kp_pred_full"][row[name]])
    assert frames.d_hat == pytest.approx(np.full(len(pick), 500.0))
    assert check["visibility_equal"] and check["max_abs_diff_px"] == pytest.approx(1e-4)
    assert check["n_unconstrained_keypoints"] == int(frames.unconstrained.sum())


def test_join_dump_refuses_a_visibility_mismatch_and_duplicates() -> None:
    dump, labels, names = _fixture()
    labels["in_frame"] = labels["in_frame"].copy()
    labels["in_frame"][0, 0] = ~labels["in_frame"][0, 0]
    with pytest.raises(ValueError, match="visibility"):
        lb.join_dump(dump, labels, names, GEOM, ORDER)
    dump, labels, names = _fixture()
    with pytest.raises(ValueError, match="unique"):
        lb.join_dump(dump, labels, [names[0], names[0]], GEOM, ORDER)


def test_check_cov_mapping() -> None:
    dump, _, _ = _fixture()
    assert lb.check_cov_mapping(dump) < 1e-12
    dump["heatmap_cov_full"] = dump["heatmap_cov_full"] * 1.001
    assert lb.check_cov_mapping(dump) > 1e-4


def test_sets_need_no_labels_and_purse_agrees() -> None:
    dump, labels, names = _fixture()
    frames, _ = lb.join_dump(dump, labels, names, GEOM, ORDER)
    for score_id in lb.LEVEL_B_SCORES:
        s = lb.score_frames(score_id, frames, "abstain_allowed")
        q = float(np.quantile(s[frames.valid], 0.8))
        blind = frames.subset(np.arange(len(frames)))
        object.__setattr__(blind, "y_gt", np.full_like(frames.y_gt, np.nan))
        object.__setattr__(blind, "q_gt", np.full_like(frames.q_gt, np.nan))
        kset = lb.keypoint_set(score_id, blind, q)  # labels poisoned: the set must not care
        agreement = lb.purse_agreement(kset, frames, s, ORDER, GEOM)
        assert agreement["n_agree"] == agreement["n_checked"] == int(frames.valid.sum())
        assert 0 < agreement["n_member"] < agreement["n_checked"]


def test_unit_radius_and_set_size() -> None:
    dump, labels, names = _fixture()
    frames, _ = lb.join_dump(dump, labels, names, GEOM, ORDER)
    radius = lb.unit_radius_px("B1", frames)
    assert np.all(np.isnan(radius[~frames.valid]))
    bounded = frames.valid & ~frames.unconstrained.any(axis=1)
    assert radius[bounded] == pytest.approx(frames.d_hat[bounded])
    assert np.all(np.isinf(radius[frames.valid & frames.unconstrained.any(axis=1)]))
    kset = lb.keypoint_set("B2", frames, 1.0)
    report = lb.set_size_report(kset, frames.valid)
    assert report["n_frames_with_unconstrained_keypoint"] == int(
        frames.unconstrained[frames.valid].any(axis=1).sum()
    )
    assert lb.set_size_report(kset, np.zeros(len(frames), dtype=bool)) is None


def test_linearised_and_sampled_runners() -> None:
    dump, labels, names = _fixture()
    frames, _ = lb.join_dump(dump, labels, names, GEOM, ORDER)
    extents, seconds = lb.linearised_frames("B2", frames, GEOM, ORDER)
    assert all((e is None) == (not v) for e, v in zip(extents, frames.valid, strict=True))
    assert np.all(np.isnan(seconds[~frames.valid])) and np.all(seconds[frames.valid] > 0)
    rows = np.flatnonzero(frames.valid)
    small = lb.linearised_radii(extents, rows, 2.0)  # residuals exceed the sets: inner empty
    assert np.isnan(small["rot_inner"]).any()
    radii = lb.linearised_radii(extents, rows, 200.0)
    assert np.all(np.isfinite(radii["rot_inner"]))
    assert np.all(radii["rot_outer"] >= radii["rot_inner"])
    with pytest.raises(ValueError, match="failed frame"):
        lb.linearised_radii(extents, np.flatnonzero(~frames.valid), 2.0)

    kset = lb.keypoint_set("B2", frames, 2.0)
    one, _ = lb.sampled_frames(
        kset, frames, rows[:8], GEOM, ORDER, samples=16, seed_key=[1, 2], workers=1
    )
    three, _ = lb.sampled_frames(
        kset, frames, rows[:8], GEOM, ORDER, samples=16, seed_key=[1, 2], workers=3
    )
    for row in rows[:8]:
        a, b = one[int(row)], three[int(row)]
        assert (a.n_accepted, a.n_solved) == (b.n_accepted, b.n_solved)
        assert (a.rot == b.rot or (math.isnan(a.rot) and math.isnan(b.rot))) and (
            a.trans == b.trans or (math.isnan(a.trans) and math.isnan(b.trans))
        )


def test_propagation_report_measures_ball_coverage() -> None:
    dump, labels, names = _fixture()
    frames, _ = lb.join_dump(dump, labels, names, GEOM, ORDER)
    rows = np.flatnonzero(frames.valid)
    huge = np.full(rows.size, 10.0)
    report = lb.propagation_report(frames, rows, huge, np.full(rows.size, 1e3))
    assert report["measured_ball_coverage"]["value"] == 1.0
    tiny = lb.propagation_report(frames, rows, np.full(rows.size, 1e-9), np.full(rows.size, 1e3))
    assert tiny["measured_ball_coverage"]["value"] == 0.0
    nan_rot = huge.copy()
    nan_rot[:3] = np.nan
    partial = lb.propagation_report(frames, rows, nan_rot, np.full(rows.size, 1e3))
    measured = partial["measured_ball_coverage"]
    assert partial["n_nan"] == 3 and measured["n"] == rows.size - 3
    assert measured["value_nan_as_uncovered"]["n"] == rows.size
    assert measured["value_nan_as_uncovered"]["value"] == pytest.approx((rows.size - 3) / rows.size)
    lo, hi = measured["coverage_ci95"]
    assert lo < measured["value"] <= hi == 1.0
    assert partial["answer_rate"] == pytest.approx(rows.size / len(frames))
    inf_rot = huge.copy()
    inf_rot[0] = np.inf
    assert lb.propagation_report(frames, rows, inf_rot, np.full(rows.size, 1e3))["n_inf"] == 1


def test_runtime_summary() -> None:
    out = lb.runtime_summary(np.array([0.001, 0.002, np.nan, 0.004]))
    assert out["n_frames"] == 3 and out["max_ms"] == pytest.approx(4.0)
    assert out["total_s"] == pytest.approx(0.007)


def _script():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import run_level_b
    finally:
        sys.path.pop(0)
    return run_level_b


def test_script_reads_synthetic_splits_only() -> None:
    script = _script()
    with pytest.raises(ValueError, match="synthetic splits only"):
        script._synthetic_split("lightbox_poolB")
    source = (REPO_ROOT / "scripts" / "run_level_b.py").read_text("utf-8")
    for token in ("lightbox", "sunlamp", "poolA", "poolB", "load_eval_labels", "load_sidecar"):
        assert token not in source, f"run_level_b.py mentions {token!r}"


# --------------------------------------------------------------------------------------------------
# dataset: the real synthetic dump and labels
# --------------------------------------------------------------------------------------------------


@pytest.mark.dataset
def test_real_projection_and_covariance_mapping(local_paths) -> None:
    from poseconf import p1_adapter
    from poseconf.data.dump import dump_file, load_dump, load_dump_labels
    from poseconf.data.splits import load_split

    path = dump_file(
        local_paths.dumps_root, "keypoint_a2", "synthetic", "predicted_crop", subset=False
    )
    if not path.is_file():
        pytest.skip(f"no dump at {path}")
    dump, _ = load_dump(path)
    labels, _ = load_dump_labels(
        local_paths.dumps_root, "keypoint_a2", "synthetic", "all", tag="split"
    )
    geometry = p1_adapter.projection_geometry("keypoint_a2", paths=local_paths)
    order = p1_adapter.pose_quaternion_order("keypoint_a2")
    assert lb.check_cov_mapping(dump) <= 1e-9
    frames, check = lb.join_dump(dump, labels, load_split("synthetic_val_test"), geometry, order)
    assert check["visibility_equal"] and check["max_abs_diff_px"] <= 0.01
    # P1's own projection of a label pose equals ours.
    i = int(np.flatnonzero(frames.include.any(axis=1))[0])
    ref, vis = p1_adapter.p1_project(
        geometry, quat_to_matrix(frames.q_gt[i], order), frames.t_gt[i]
    )
    np.testing.assert_array_equal(vis, frames.include[i])
    np.testing.assert_allclose(ref[vis], frames.y_gt[i][vis], atol=1e-9)
