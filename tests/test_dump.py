"""Phase 3: prediction dump, P1-faithful PnP, heatmap moments, parity logic and HIL label guards.

Everything except the last test runs without the dataset or a GPU: the dump runs on a 6-frame fake
SPEED+ tree with random-init P1 checkpoints (`tests/fixtures/tiny_speedplus.py`).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fixtures.tiny_speedplus import build_tiny_speedplus
from poseconf import p1_adapter
from poseconf.data import dump as dump_module
from poseconf.data.splits import load_split
from poseconf.engine.parity import ErrorDeltas, parity_cell, quaternion_angle

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "conformal.yaml"
RUN = "keypoint_a2"
GATE = {"flag_agreement_min": 0.999, "median_abs_de_r_max_rad": 1e-6, "solved_count_exact": True}


def _script(name: str):
    """Import a script module from `scripts/` (the repo's convention, see test_level_a.py)."""
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


PREDICTION_FIELDS = {
    "filename",
    "bbox_used",
    "affine",
    "croppable",
    "kp_pred_crop",
    "kp_pred_full",
    "confidence",
    "heatmap_cov_crop",
    "heatmap_cov_full",
    "heatmap_peak",
    "heatmap_entropy",
    "heatmap_empty",
    "enc_feat",
    "success",
    "failure_reason",
    "q_pred",
    "t_pred",
    "n_inliers",
    "reprojection_rmse",
    "subset_mask",
}


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """The fake SPEED+ tree with random-init checkpoints."""
    return build_tiny_speedplus(tmp_path_factory.mktemp("tiny_speedplus"))


@pytest.fixture(scope="session")
def fixture_dump(tiny):
    """Run the dump script once on the fixture, both arms, on the CPU."""
    script = _script("dump_predictions")

    code = script.main(
        [
            "--config", str(CONFIG), "--paths", str(tiny.paths_yaml),
            "--domain", "synthetic", "--all-arms", "--device", "cpu",
        ]
    )  # fmt: skip
    assert code == 0
    return tiny.paths.dumps_root / RUN


@pytest.fixture(scope="session")
def cpu_pipeline(tiny):
    """A GT-crop P1 pipeline on the fixture's random-init checkpoint."""
    return p1_adapter.load_pipeline(
        RUN, crop_source="gt_crop", paths=tiny.paths, device=torch.device("cpu")
    )


# --- end-to-end dump on the fixture ------------------------------------------------------------


@pytest.mark.parametrize("arm", dump_module.CROP_ARMS)
def test_fixture_dump_schema_and_invariants(fixture_dump, tiny, arm):
    arrays, meta = dump_module.load_dump(fixture_dump / f"synthetic_{arm}.npz")
    n = len(tiny.filenames)
    assert set(arrays) == PREDICTION_FIELDS
    assert all(len(value) == n for value in arrays.values())  # every frame has a row
    assert arrays["filename"].tolist() == tiny.filenames
    assert arrays["kp_pred_crop"].shape == (n, 11, 2)
    assert arrays["heatmap_cov_full"].shape == (n, 11, 2, 2)
    assert arrays["enc_feat"].shape == (n, 512) and arrays["enc_feat"].dtype == np.float16
    assert arrays["q_pred"].dtype == np.float64
    solved = arrays["success"]
    assert np.isnan(arrays["q_pred"][~solved]).all()
    assert all(reason for reason in arrays["failure_reason"][~solved])
    assert np.allclose(np.linalg.norm(arrays["q_pred"][solved], axis=1), 1.0)
    for row in range(n):
        expected = p1_adapter.crop_to_full(arrays["kp_pred_crop"][row], arrays["affine"][row])
        np.testing.assert_array_equal(arrays["kp_pred_full"][row], expected)
    cov = arrays["heatmap_cov_full"]
    np.testing.assert_allclose(cov, np.swapaxes(cov, -1, -2), atol=1e-9)
    assert (np.linalg.eigvalsh(cov) >= -1e-9).all()
    np.testing.assert_array_equal(arrays["heatmap_peak"], arrays["confidence"].astype(np.float32))

    assert meta["kind"] == "predictions" and meta["crop_source"] == arm
    assert meta["tag"] == arm and meta["oracle"] is False  # synthetic: no oracle
    assert meta["p1_reference_arm"] == dump_module.p1_reference_arm(arm)
    for key in ("poseconf_git_sha", "p1_commit", "p1_checkpoint_sha256", "pnp_config"):
        assert key in meta["provenance"]
    assert meta["runtime"]["cudnn_benchmark"] is False
    assert meta["info"]["max_moment_mean_dev_crop_px"] <= 1e-3

    subset, _ = dump_module.load_dump(fixture_dump / f"synthetic_{arm}_subset.npz")
    assert subset["heatmaps"].shape == (n, 11, 64, 64)
    assert subset["decoder_feat"].shape == (n, 256, 64, 64)
    assert subset["filename"].tolist() == tiny.filenames  # n < 256: every frame


def test_fixture_labels_file(fixture_dump, tiny):
    labels, meta = dump_module.load_dump_labels(
        fixture_dump.parent, RUN, "synthetic", "all", tag=None
    )
    assert labels["filename"].tolist() == tiny.filenames
    assert labels["kp_gt_full"].shape == (len(tiny.filenames), 11, 2)
    assert labels["in_frame"].dtype == bool
    assert meta["kind"] == "labels"


def test_dump_is_deterministic(fixture_dump, tiny, tmp_path):
    script = _script("dump_predictions")

    script.main(
        [
            "--config", str(CONFIG), "--paths", str(tiny.paths_yaml), "--domain", "synthetic",
            "--crop-source", "predicted_crop", "--device", "cpu", "--out", str(tmp_path),
        ]
    )  # fmt: skip
    first, _ = dump_module.load_dump(fixture_dump / "synthetic_predicted_crop.npz")
    second, _ = dump_module.load_dump(tmp_path / RUN / "synthetic_predicted_crop.npz")
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def test_limit_refused_inside_dumps_root(tiny):
    script = _script("dump_predictions")

    base = ["--config", str(CONFIG), "--paths", str(tiny.paths_yaml), "--domain", "synthetic"]
    with pytest.raises(SystemExit, match="outside dumps_root"):
        script.main([*base, "--all-arms", "--device", "cpu", "--limit", "2"])
    with pytest.raises(SystemExit, match="outside dumps_root"):
        script.main(
            [*base, "--all-arms", "--device", "cpu", "--limit", "2",
             "--out", str(tiny.paths.dumps_root / "x")]
        )  # fmt: skip


# --- hooks and moments ---------------------------------------------------------------------------


def test_hooks_leave_p1_outputs_bit_identical(cpu_pipeline):
    torch.manual_seed(0)
    crops = torch.randn(3, 1, 256, 256)
    coords_plain, conf_plain = cpu_pipeline.keypoints(crops)
    with p1_adapter.keypoint_hooks(cpu_pipeline, keep_decoder=True) as capture:
        coords_hooked, conf_hooked = cpu_pipeline.keypoints(crops)
    np.testing.assert_array_equal(coords_plain, coords_hooked)
    np.testing.assert_array_equal(conf_plain, conf_hooked)
    assert capture.heatmaps.shape == (3, 11, 64, 64)
    assert capture.encoder_pooled.shape == (3, 512)
    assert capture.decoder.shape == (3, 256, 64, 64)
    model = cpu_pipeline.keypoint_model
    assert not model._forward_hooks and not model.encoder._forward_hooks
    assert not model.decoder[-1]._forward_pre_hooks


def test_moment_mean_is_p1_coordinate(cpu_pipeline):
    torch.manual_seed(1)
    crops = torch.randn(4, 1, 256, 256)
    with p1_adapter.keypoint_hooks(cpu_pipeline) as capture:
        coords, confidence = cpu_pipeline.keypoints(crops)
    model = cpu_pipeline.keypoint_model
    moments = dump_module.heatmap_moments(capture.heatmaps, model.head, float(model.stride))
    hit = ~moments.empty.numpy()
    np.testing.assert_allclose(moments.mean_crop.double().numpy()[hit], coords[hit], atol=1e-5)
    np.testing.assert_array_equal(moments.peak.double().numpy(), confidence)
    cov = moments.cov_crop.double().numpy()
    assert (np.linalg.eigvalsh(cov) >= -1e-9).all()


def test_moment_covariance_of_a_known_gaussian(cpu_pipeline):
    """Windowed relu(h)^2 of a unit Gaussian (σ = 1 hm px): variance 1 / (1/0.5 + 1/σ_w²)."""
    model = cpu_pipeline.keypoint_model
    ys, xs = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    centre = (30.3, 20.7)
    gaussian = torch.exp(-((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) / 2.0)
    maps = gaussian.expand(1, 11, 64, 64).clone()
    moments = dump_module.heatmap_moments(maps, model.head, float(model.stride))
    stride = float(model.stride)
    cov = moments.cov_crop[0, 0].double().numpy()
    # The refine weights are relu(h)^2 (variance 0.5) times the Gaussian window (variance σ_w²):
    # precisions add.
    expected = 1.0 / (1.0 / 0.5 + 1.0 / model.head.window_sigma**2)
    np.testing.assert_allclose(cov, np.diag([expected, expected]) * stride**2, rtol=0.01, atol=1e-5)
    np.testing.assert_allclose(
        moments.mean_crop[0, 0].double().numpy(),
        (np.array(centre) + 0.5) * stride - 0.5,
        atol=0.02,
    )
    assert not moments.empty.any()


def test_empty_channel_is_flagged_and_decodes_to_origin(cpu_pipeline):
    model = cpu_pipeline.keypoint_model
    maps = -torch.ones(1, 11, 64, 64)
    moments = dump_module.heatmap_moments(maps, model.head, float(model.stride))
    assert moments.empty.all()
    assert torch.equal(moments.cov_crop, torch.zeros_like(moments.cov_crop))
    assert torch.allclose(moments.mean_crop, torch.full_like(moments.mean_crop, 1.5))


def test_crop_cov_to_full_under_scale_and_shift():
    scale = 0.37
    affine = np.array([[scale, 0.0, -120.0], [0.0, scale, -40.0]])
    cov = np.tile(np.array([[4.0, 1.0], [1.0, 9.0]]), (1, 11, 1, 1))
    full = dump_module.crop_cov_to_full(cov, affine[None])
    np.testing.assert_allclose(full, cov / scale**2)


def test_subset_rows_are_fixed_and_sorted():
    rows = dump_module.subset_rows(11994, 256, 1337)
    assert len(rows) == 256 and np.all(np.diff(rows) > 0)
    np.testing.assert_array_equal(rows, dump_module.subset_rows(11994, 256, 1337))
    np.testing.assert_array_equal(dump_module.subset_rows(6, 256, 1337), np.arange(6))


# --- P1-faithful PnP -----------------------------------------------------------------------------


def test_solve_chunks_mirror_solve_many(monkeypatch):
    monkeypatch.setattr(p1_adapter.os, "cpu_count", lambda: 36)
    assert p1_adapter.solve_chunks(255, workers=16, seed=7) == [(0, 255, 7)]
    assert p1_adapter.solve_chunks(1000, workers=1, seed=7) == [(0, 1000, 7)]
    chunks = p1_adapter.solve_chunks(11994, workers=16, seed=1337)
    bounds = np.linspace(0, 11994, 17).astype(int)
    assert chunks == [
        (int(a), int(b), 1337 + i)
        for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
    ]
    monkeypatch.setattr(p1_adapter.os, "cpu_count", lambda: 4)
    assert len(p1_adapter.solve_chunks(11994, workers=16, seed=0)) == 4


def _projected_frames(tiny, n, seed):
    """Noisy crop-pixel keypoints of random poses, with some hopeless frames for failures."""
    rng = np.random.default_rng(seed)
    camera = tiny.paths.speedplus_root / "camera.json"
    coords, confidence, affines = [], [], []
    for index in range(n):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        t = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2), rng.uniform(6.0, 10.0)])
        points, _ = p1_adapter.project_keypoints(q, t, camera_json=camera)
        points = np.nan_to_num(points, nan=960.0)
        scale = 256.0 / 900.0
        affine = np.array([[scale, 0.0, -scale * 510.0], [0.0, scale, -scale * 150.0]])
        crop = points @ affine[:, :2].T + affine[:, 2]
        noise = 40.0 if index % 7 == 0 else 0.4
        coords.append(crop + rng.normal(scale=noise, size=crop.shape))
        confidence.append(rng.uniform(0.05, 1.0, size=11))
        affines.append(affine)
    return np.array(coords), np.array(confidence), np.array(affines)


def test_solve_frames_equals_p1_solve_many(tiny, cpu_pipeline):
    coords, confidence, affines = _projected_frames(tiny, 320, seed=3)
    ours = p1_adapter.solve_frames(cpu_pipeline, coords, confidence, affines, workers=4, seed=1337)
    reference = p1_adapter.solve_many_reference(
        cpu_pipeline, coords, confidence, affines, workers=4, seed=1337
    )
    assert len(ours.chunks) == min(4, p1_adapter.os.cpu_count() or 1) or len(ours.chunks) == 1
    success = np.array([item.success for item in reference])
    np.testing.assert_array_equal(ours.success, success)
    assert 0 < success.sum() < len(success)  # both branches exercised
    np.testing.assert_array_equal(ours.n_inliers, [item.n_inliers for item in reference])
    for row, item in enumerate(reference):
        if item.success:
            np.testing.assert_array_equal(ours.q[row], item.quaternion)
            np.testing.assert_array_equal(ours.t[row], item.translation)
        else:
            assert np.isnan(ours.q[row]).all() and ours.failure_reason[row] == item.failure_reason


# --- parity logic --------------------------------------------------------------------------------


def _fake_arm(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    success = rng.uniform(size=n) > 0.1
    q[~success] = np.nan
    t = rng.normal(size=(n, 3))
    t[~success] = np.nan
    return {
        "filename": np.array([f"img{i:06d}.jpg" for i in range(n)]),
        "success": success,
        "q_pred": q,
        "t_pred": t,
        "n_inliers": np.where(success, 11, 0).astype(np.int32),
        "failure_reason": np.where(success, "", "no consensus"),
    }


def test_parity_identical_meets_gate():
    arm = _fake_arm(2000, 0)
    cell = parity_cell(arm, arm, errors=None, p1_solved_count=int(arm["success"].sum()),
                       gate=GATE, max_listed=10)  # fmt: skip
    assert cell["success_flag_agreement"] == 1.0 and cell["gate"]["met"]
    assert cell["label_free_bound"]["rotation_delta_rad"]["median"] == 0.0


def test_parity_flag_and_count_misses():
    p1_arm = _fake_arm(2000, 0)
    dump = {key: value.copy() for key, value in p1_arm.items()}
    flipped = np.flatnonzero(dump["success"])[:3]
    dump["success"][flipped] = False
    cell = parity_cell(dump, p1_arm, errors=None, p1_solved_count=int(p1_arm["success"].sum()),
                       gate=GATE, max_listed=2)  # fmt: skip
    assert cell["success_flag_agreement"] == pytest.approx(1 - 3 / 2000)
    assert cell["gate"]["criteria"]["flag_agreement"] is False
    assert cell["gate"]["criteria"]["solved_count_vs_p1_json"] is False
    assert len(cell["disagreements"]) == 2 and not cell["gate"]["met"]


def test_parity_rotation_bound_catches_pose_drift():
    p1_arm = _fake_arm(500, 1)
    dump = {key: value.copy() for key, value in p1_arm.items()}
    dump["q_pred"] = dump["q_pred"] + 1e-5
    cell = parity_cell(dump, p1_arm, errors=None, p1_solved_count=None, gate=GATE, max_listed=0)
    assert cell["gate"]["criteria"]["median_rotation_bound_unlabelled"] is False


def test_parity_error_deltas_and_scope():
    arm = _fake_arm(100, 2)
    names = arm["filename"][:40]
    e = np.where(arm["success"][:40], 0.05, np.nan)
    errors = ErrorDeltas("poolB", names, e, e, e + 2e-6, e)
    cell = parity_cell(arm, arm, errors=errors, p1_solved_count=None, gate=GATE, max_listed=0)
    assert cell["errors"]["scope"] == "poolB"
    assert cell["errors"]["abs_de_r_rad"]["median"] == pytest.approx(2e-6)
    assert cell["gate"]["criteria"]["median_abs_de_r"] is False
    assert cell["label_free_bound"]["rotation_delta_rad"]["n"] == int(arm["success"][40:].sum())


def test_quaternion_angle_is_accurate_for_tiny_rotations():
    angle = 3e-9
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    assert quaternion_angle(q1, q2) == pytest.approx(angle, rel=1e-6)
    assert quaternion_angle(q1, -q2) == pytest.approx(angle, rel=1e-6)


# --- HIL label guards (CLAUDE.md invariant 4) ----------------------------------------------------


def test_hil_access_rule():
    pool_a, pool_b = load_split("lightbox_poolA"), load_split("lightbox_poolB")
    assert p1_adapter.hil_label_access_allowed("lightbox", pool_b[:5], None)
    assert not p1_adapter.hil_label_access_allowed("lightbox", [*pool_b[:5], pool_a[0]], None)
    assert not p1_adapter.hil_label_access_allowed("lightbox", None, None)
    assert p1_adapter.hil_label_access_allowed("lightbox", pool_a[:5], "oracle_gt_box")
    assert p1_adapter.hil_label_access_allowed("synthetic", None, None)


def test_hil_sidecar_label_columns_are_guarded():
    with pytest.raises(p1_adapter.HILLabelAccessError):
        p1_adapter.load_sidecar(RUN, "sunlamp", "predicted_crop")
    with pytest.raises(p1_adapter.HILLabelAccessError):
        p1_adapter.load_sidecar(
            RUN, "sunlamp", "predicted_crop", columns=["e_r"], frames=load_split("sunlamp_poolA")
        )
    free = p1_adapter.load_sidecar(
        RUN, "sunlamp", "predicted_crop", columns=list(p1_adapter.SIDECAR_LABEL_FREE_COLUMNS)
    )
    assert "e_r" not in free.fields and len(free) == 2791
    pool_b = load_split("sunlamp_poolB")
    labelled = p1_adapter.load_sidecar(
        RUN, "sunlamp", "predicted_crop", columns=["e_r"], frames=pool_b
    )
    assert len(labelled) == len(pool_b) and set(labelled.fields) == {"filename", "e_r"}


def test_dump_labels_pool_a_needs_oracle_tag(tmp_path):
    with pytest.raises(p1_adapter.HILLabelAccessError):
        dump_module.load_dump_labels(tmp_path, RUN, "lightbox", "poolA", tag=None)
    with pytest.raises(p1_adapter.HILLabelAccessError):
        dump_module.load_dump_labels(tmp_path, RUN, "lightbox", "poolA", tag="split")
    with pytest.raises(FileNotFoundError):  # allowed, so it gets as far as the (absent) file
        dump_module.load_dump_labels(tmp_path, RUN, "lightbox", "poolA", tag="oracle_x")
    assert dump_module.arm_tag("lightbox", "gt_crop") == "oracle_gt_box"
    assert dump_module.arm_tag("synthetic", "gt_crop") == "gt_crop"
    assert dump_module.arm_tag("sunlamp", "predicted_crop") == "predicted_crop"


def test_parity_script_reads_hil_labels_for_pool_b_only(tiny, tmp_path, monkeypatch):
    """End to end on P1's committed sunlamp sidecar: poolA is gated by the label-free bound."""
    script = _script("check_p1_parity")

    side = p1_adapter.load_sidecar(
        RUN, "sunlamp", "predicted_crop", columns=list(p1_adapter.SIDECAR_LABEL_FREE_COLUMNS)
    ).fields
    n = len(side["filename"])
    arrays = {
        "filename": side["filename"],
        "success": side["success"],
        "q_pred": side["q_pred"].astype(np.float64),
        "t_pred": side["t_pred"].astype(np.float64),
        "n_inliers": side["n_inliers"],
        "failure_reason": side["failure_reason"],
    }
    meta = {
        "tag": "predicted_crop", "oracle": False, "split": "test", "limit": None,
        "provenance": {"poseconf_git_sha": "test", "created_at": "now",
                       "p1_checkpoint_sha256": {"detector": "x", RUN: "y"}},
        "keypoint_labels_sha256": "z", "runtime": {}, "info": {},
    }  # fmt: skip
    dumps = tmp_path / "dumps"
    dump_module.write_dump(
        dump_module.dump_file(dumps, RUN, "sunlamp", "predicted_crop", subset=False),
        arrays,
        meta,
        compress=True,
    )
    pool_b = set(load_split("sunlamp_poolB"))
    rows = np.array([str(name) in pool_b for name in side["filename"]])
    fake = p1_adapter.EvalLabels(
        filenames=side["filename"][rows].astype(str),
        q=np.tile([1.0, 0.0, 0.0, 0.0], (rows.sum(), 1)),
        t=np.tile([0.0, 0.0, 10.0], (rows.sum(), 1)),
        keypoints_2d=np.zeros((rows.sum(), 11, 2)),
        in_frame=np.ones((rows.sum(), 11), dtype=bool),
        bbox_tight=np.zeros((rows.sum(), 4)),
    )
    dump_module.write_labels(dump_module.labels_file(dumps, RUN, "sunlamp", "poolB"), fake, {})

    requested = []
    original = script.load_dump_labels

    def spy(root, run, domain, pool, *, tag):
        requested.append(pool)
        return original(root, run, domain, pool, tag=tag)

    monkeypatch.setattr(script, "load_dump_labels", spy)
    out = tmp_path / "summary.json"
    code = script.main(
        ["--config", str(CONFIG), "--paths", str(tiny.paths_yaml), "--dumps-root", str(dumps),
         "--domain", "sunlamp", "--out", str(out)]
    )  # fmt: skip
    assert requested == ["poolB"]
    summary = json.loads(out.read_text())
    cell = summary["cells"][0]
    assert summary["kind"] == "dump_summary" and cell["n_total"] == n
    assert cell["success_flag_agreement"] == 1.0
    assert cell["errors"]["scope"] == "poolB"
    assert cell["label_free_bound"]["rotation_delta_rad"]["median"] == 0.0
    # The fake GT makes |de_r| large, so the gate honestly misses and the script says so.
    assert code == 1 and cell["gate"]["criteria"]["median_abs_de_r"] is False


# --- real checkpoints (local only) ----------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.dataset
@pytest.mark.slow
def test_first_batch_reproduces_p1(local_paths, tmp_path):
    """First 48 synthetic frames: chunk 0 is seeded identically, so rows must match P1 exactly."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    parity = _script("check_p1_parity")
    script = _script("dump_predictions")

    assert script.main(
        ["--config", str(CONFIG), "--domain", "synthetic", "--all-arms", "--device", "cuda",
         "--limit", "48", "--out", str(tmp_path)]
    ) == 0  # fmt: skip
    assert parity.main(
        ["--config", str(CONFIG), "--dumps-root", str(tmp_path), "--domain", "synthetic",
         "--out", str(tmp_path / "summary.json")]
    ) == 0  # fmt: skip
