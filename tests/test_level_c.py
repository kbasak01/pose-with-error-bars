"""Level C (Phase 5): variance sidecar dump, C1/C2 plumbing, C2's σ̂, head-quality report."""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fixtures.tiny_speedplus import build_tiny_speedplus
from poseconf.conformal.propagate import LinearisedExtent
from poseconf.conformal.scores import SCORES
from poseconf.data import dump as dump_module
from poseconf.engine import level_b, level_c
from poseconf.models.variance_head import HeadConfig, VarianceHead, p1_state_hash
from poseconf.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "conformal.yaml"
RUN = "keypoint_a2"
HEAD = HeadConfig(256, 16, 11, 0.05, 64.0, 0.99, 2.0)


def _script(name: str):
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


# --- variance sidecar on the fixture tree (CPU) --------------------------------------------------


@pytest.fixture(scope="module")
def tiny_with_dump(tmp_path_factory):
    """Fixture tree + its Phase 3 dump (predicted_crop) + a random variance-head checkpoint."""
    tiny = build_tiny_speedplus(tmp_path_factory.mktemp("tiny_vhead"))
    code = _script("dump_predictions").main(
        [
            "--config", str(CONFIG), "--paths", str(tiny.paths_yaml),
            "--domain", "synthetic", "--crop-source", "predicted_crop", "--device", "cpu",
        ]
    )  # fmt: skip
    assert code == 0
    from poseconf import p1_adapter

    model, _ = p1_adapter.load_keypoint_model(RUN, paths=tiny.paths)
    torch.manual_seed(0)
    head = VarianceHead(HEAD)
    torch.nn.init.normal_(head.out.weight, std=0.01)
    checkpoint = tiny.root / "vhead" / "best.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "head": head.state_dict(),
            "head_config": HEAD.__dict__,
            "config": {"run_name": "fixture"},
            "p1_run": RUN,
            "p1_checkpoint_sha256": sha256_file(tiny.paths.p1_runs / RUN / "best.pt"),
            "p1_state_hash": p1_state_hash(model),
            "git_sha": "fixture",
            "epoch": 1,
            "val_tune_nll": 0.0,
        },
        checkpoint,
    )
    return tiny, checkpoint


def test_variance_sidecar_matches_the_dump(tiny_with_dump):
    tiny, checkpoint = tiny_with_dump
    code = _script("dump_variance").main(
        [
            "--config", str(CONFIG), "--paths", str(tiny.paths_yaml),
            "--variance-head", str(checkpoint), "--domain", "synthetic",
            "--crop-source", "predicted_crop", "--device", "cpu",
        ]
    )  # fmt: skip
    assert code == 0
    root = tiny.paths.dumps_root
    dump, _ = dump_module.load_dump(
        dump_module.dump_file(root, RUN, "synthetic", "predicted_crop", subset=False)
    )
    side, meta = dump_module.load_dump(
        dump_module.variance_dump_file(root, RUN, "synthetic", "predicted_crop")
    )
    np.testing.assert_array_equal(side["kp_pred_crop"], dump["kp_pred_crop"])
    assert meta["kp_pred_crop_bit_equal_to_base"] and meta["oracle"] is False
    assert (np.linalg.eigvalsh(side["vhead_cov_crop"]) > 0).all()
    merged = level_c.attach_variance(dump, side)
    assert level_c.check_variance_mapping(merged) < 1e-12


def test_variance_dump_refuses_a_different_dump(tiny_with_dump):
    tiny, checkpoint = tiny_with_dump
    from poseconf import p1_adapter
    from poseconf.engine.train_variance import load_variance_net

    root = tiny.paths.dumps_root
    dump, _ = dump_module.load_dump(
        dump_module.dump_file(root, RUN, "synthetic", "predicted_crop", subset=False)
    )
    tampered = dict(dump)
    tampered["kp_pred_crop"] = dump["kp_pred_crop"] + 1e-9
    net, _ = load_variance_net(checkpoint, paths=tiny.paths, device=torch.device("cpu"))
    pipeline = p1_adapter.load_pipeline(
        RUN, crop_source="predicted_crop", paths=tiny.paths, device=torch.device("cpu")
    )
    frames = p1_adapter.eval_frames("synthetic", paths=tiny.paths)
    with pytest.raises(RuntimeError, match="differ from the Phase 3 dump"):
        dump_module.run_variance_dump(
            pipeline,
            frames,
            net.head,
            tampered,
            batch_size=48,
            decode_threads=1,
            log=lambda _: None,
        )


def test_attach_variance_refuses_mismatch():
    dump = {"filename": np.array(["a", "b"]), "kp_pred_crop": np.zeros((2, 1, 2))}
    side = {
        "filename": np.array(["a", "b"]),
        "kp_pred_crop": np.ones((2, 1, 2)),
        "vhead_chol_crop": np.zeros((2, 1, 3)),
        "vhead_cov_crop": np.zeros((2, 1, 2, 2)),
        "vhead_cov_full": np.zeros((2, 1, 2, 2)),
    }
    with pytest.raises(ValueError, match="coordinates"):
        level_c.attach_variance(dump, side)
    side["kp_pred_crop"] = dump["kp_pred_crop"]
    side["filename"] = np.array(["b", "a"])
    with pytest.raises(ValueError, match="order"):
        level_c.attach_variance(dump, side)


# --- C2's sigma and scores -----------------------------------------------------------------------


def _extent(rot_var: float, trans_var: float, n_used: int = 6) -> LinearisedExtent:
    return LinearisedExtent(rot_var, trans_var, 0.0, 0.0, 0.0, n_used, 0.0)


def test_pose_sigma_marks_undefined_frames():
    extents = [_extent(4e-4, 9e-2), None, _extent(math.inf, math.inf, 2), _extent(1e-6, 1e-4)]
    sigma = level_c.pose_sigma(extents)
    np.testing.assert_allclose(sigma.sigma_R[[0, 3]], [2e-2, 1e-3])
    np.testing.assert_allclose(sigma.sigma_t[[0, 3]], [0.3, 1e-2])
    assert sigma.ok.tolist() == [True, False, False, True]
    assert sigma.n_used.tolist() == [6, 0, 2, 6]


class _Frames:
    def __init__(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        q = rng.normal(size=(n, 4))
        self.q_gt = q / np.linalg.norm(q, axis=1, keepdims=True)
        noise = self.q_gt + 0.01 * rng.normal(size=(n, 4))
        self.q_hat = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        self.t_gt = rng.normal(size=(n, 3)) + [0, 0, 10]
        self.t_hat = self.t_gt + 0.05 * rng.normal(size=(n, 3))
        self.valid = np.ones(n, dtype=bool)
        self.valid[0] = False
        self.q_hat[0] = np.nan
        self.t_hat[0] = np.nan


@pytest.mark.parametrize("convention", ["abstain_allowed", "answer_required"])
def test_c2_undefined_sigma_follows_the_failure_convention(convention):
    frames = _Frames(5)
    extents = [None, _extent(1e-4, 1e-2), _extent(math.inf, math.inf, 2)] + [
        _extent(4e-4, 4e-2)
    ] * 2
    sigma = level_c.pose_sigma(extents)
    scores = level_c.c2_scores(frames, sigma, convention)
    fail = -math.inf if convention == "abstain_allowed" else math.inf
    assert scores[0] == fail and scores[2] == fail  # PnP failure; σ undefined
    ok = np.array([False, True, False, True, True])
    direct = SCORES["C2"].score_fn(
        frames.q_hat[ok], frames.t_hat[ok], frames.q_gt[ok], frames.t_gt[ok], np.ones(3, bool),
        sigma_R=sigma.sigma_R[ok], sigma_t=sigma.sigma_t[ok], convention=convention,
    )  # fmt: skip
    np.testing.assert_array_equal(scores[ok], direct)
    pset = level_c.c2_set(frames, sigma, 2.0)
    np.testing.assert_allclose(pset.rot_radius[ok], 2.0 * sigma.sigma_R[ok])
    assert not pset.valid[2]


def test_c1_is_b2_code_on_another_covariance():
    assert SCORES["C1"].score_fn is SCORES["B2"].score_fn
    assert "C1" in level_b.KEYPOINT_SCORES and "C1" not in level_b.LEVEL_B_SCORES


# --- head-quality report -------------------------------------------------------------------------


def test_head_quality_report_fits_scale_on_tune_only():
    rng = np.random.default_rng(1)
    n, k = 300, 2
    names = [f"f{i}" for i in range(n)]
    affine = np.tile(np.array([[1.0, 0.0, -10.0], [0.0, 1.0, -20.0]]), (n, 1, 1))
    gt_full = rng.uniform(50, 100, size=(n, k, 2))
    gt_crop = gt_full - [10.0, 20.0]
    learned = np.tile(np.eye(2) * 4.0, (n, k, 1, 1))
    moment = np.tile(np.eye(2) * 1.0, (n, k, 1, 1))
    residual = rng.normal(scale=2.0, size=(n, k, 2))
    dump = {
        "filename": np.array(names),
        "affine": affine,
        "kp_pred_crop": gt_crop + residual,
        "heatmap_empty": np.zeros((n, k), dtype=bool),
        "confidence": rng.uniform(size=(n, k)),
        "vhead_cov_crop": learned,
        "heatmap_cov_crop": moment,
    }
    labels = {
        "filename": np.array(names),
        "kp_gt_full": gt_full,
        "in_frame": np.ones((n, k), dtype=bool),
    }
    residual_got, _, _ = level_c.crop_residuals(dump, labels, names)
    np.testing.assert_allclose(residual_got, residual, atol=1e-12)
    report = level_c.head_quality_report(
        dump, labels, fit_names=names[:150], test_names=names[150:]
    )
    tune, test = report["val_tune"], report["val_test"]
    for source in ("learned", "heatmap_moment"):
        assert tune[source]["global_scale_fitted_on_this_split"]
        assert not test[source]["global_scale_fitted_on_this_split"]
        assert test[source]["global_scale"] == tune[source]["global_scale"]
    assert tune["learned"]["nll_mean"] < tune["heatmap_moment"]["nll_mean"]  # σ² = 4 is right
    assert "spearman_neg_confidence_vs_error" in test["learned"]
    assert json.dumps(report)  # JSON-ready
