"""Phase 7 latency harness: methodology minimums, the uncertainty stage wiring. CPU, no dataset."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from poseconf.conformal.propagate import crop_cov_to_full
from poseconf.export import bench
from poseconf.export.parity import chol_to_cov

MINIMUMS = (50, 500)


def test_time_calls_enforces_the_methodology_minimums():
    with pytest.raises(ValueError, match="minimums"):
        bench.time_calls(lambda i: None, n_inputs=1, warmup=49, iters=500, minimums=MINIMUMS)
    with pytest.raises(ValueError, match="minimums"):
        bench.time_calls(lambda i: None, n_inputs=1, warmup=50, iters=499, minimums=MINIMUMS)


def test_time_calls_cycles_inputs_and_reports_the_tail():
    seen: list[int] = []
    stats = bench.time_calls(seen.append, n_inputs=7, warmup=50, iters=500, minimums=MINIMUMS)
    assert seen[:8] == [0, 1, 2, 3, 4, 5, 6, 0] and len(seen) == 550
    assert stats["p50_ms"] <= stats["p90_ms"] <= stats["p99_ms"] <= stats["max_ms"]
    assert stats["timed_iters"] == 500 and stats["warmup_iters"] == 50


def test_uncertainty_inputs_maps_covariances_as_the_dump_does():
    rng = np.random.default_rng(0)
    chol = np.stack([rng.uniform(0.5, 3, 11), rng.normal(size=11), rng.uniform(0.5, 3, 11)], -1)
    affine = np.array([[0.4, 0.0, -100.0], [0.0, 0.4, -50.0]])
    estimate, cov = bench.uncertainty_inputs(
        valid=True,
        rotation_quat=np.array([1.0, 0, 0, 0]),
        translation=np.array([0, 0, 10.0]),
        keypoints_full=rng.uniform(0, 1000, (11, 2)),
        confidence=np.full(11, 0.5),
        bbox=np.array([0, 0, 100, 100.0]),
        cov_chol_crop=chol,
        keypoint_empty=np.zeros(11, bool),
        affine=affine,
    )
    np.testing.assert_array_equal(cov, crop_cov_to_full(chol_to_cov(chol)[None], affine[None])[0])
    assert estimate.confidence_mean == pytest.approx(0.5)


class _FakeHead:
    score_id = "C1"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, estimate, cov):
        self.calls += 1
        assert cov.shape == (11, 2, 2)
        return object()


def test_uncertainty_frame_adds_one_stage_and_keeps_p1s():
    rng = np.random.default_rng(1)
    pnp = SimpleNamespace(success=True, rotation=np.eye(3), translation=np.array([0, 0, 10.0]))
    output = SimpleNamespace(
        pnp=pnp,
        keypoints_2d=rng.uniform(0, 1000, (11, 2)),
        confidence=np.full(11, 0.6),
        bbox=np.array([0, 0, 100, 100.0]),
        stage_times_ms={"detect": 1.0, "keypoints": 2.0, "pnp": 0.5},
    )
    chol = np.stack([np.ones(11), np.zeros(11), np.ones(11)], -1)[None]
    pipeline = SimpleNamespace(
        last_cov_chol=chol,
        last_keypoint_empty=np.zeros((1, 11), bool),
        last_affine=np.array([[0.5, 0, 0], [0, 0.5, 0.0]]),
        pose_quaternion=lambda r: np.array([1.0, 0, 0, 0]),
    )

    class _Pipeline:
        def __getattr__(self, name):
            return getattr(pipeline, name)

        def __call__(self, frame):
            return output

    head = _FakeHead()
    out = bench.UncertaintyFrame(_Pipeline(), [head])(np.zeros((4, 4), np.uint8))
    assert set(out.stage_times_ms) == {"detect", "keypoints", "pnp", "uncertainty"}
    assert out.stage_times_ms["uncertainty"] >= 0 and head.calls == 1
    with pytest.raises(ValueError, match="no bbox"):
        bench.UncertaintyFrame(_Pipeline(), [head])(np.zeros((4, 4), np.uint8), bbox=[0, 0, 1, 1])
