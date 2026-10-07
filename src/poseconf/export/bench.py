"""Latency of the uncertainty post-process and of the full frame, P1's methodology. Phase 7.

Network stages are timed by P1's own harness (`speedpose.export.bench`, through `p1_adapter`):
provider verified by constructing a session, IOBinding, compute separate from host<->device copies,
>= 50 warmup and >= 500 timed iterations, p50/p90/p99/max. This module adds the CPU-side pieces P1
has no counterpart for (the conformal head, linearised and sampled propagation), timed by
`time_calls` under the same minimums, and `UncertaintyFrame`, which runs P1's ORT pipeline and the
head as one extra `uncertainty` stage, so P1's `end_to_end_frame_budget` measures a frame with
uncertainty exactly as it measures one without.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from poseconf.conformal.head import Answer, ConformalPoseHead, FrameEstimate
from poseconf.conformal.propagate import crop_cov_to_full
from poseconf.export.parity import chol_to_cov

__all__ = [
    "HARDWARE_CAVEAT",
    "UncertaintyFrame",
    "latency_stats",
    "time_calls",
    "uncertainty_inputs",
]

#: Every latency file carries this (checklist G6). Same statement as P1's `bench_onnx.py`.
HARDWARE_CAVEAT = (
    "Measured on an RTX A4000 (16 GB, Ampere, 140 W desktop workstation card) and its host CPU. "
    "The A4000 is a stand-in for a Jetson-class embedded target, not a proxy for one: it has "
    "roughly an order of magnitude more memory bandwidth and power budget. These absolute "
    "latencies will NOT transfer to a Jetson. The ratios (head on vs off, post-process vs network) "
    "and the shape of the frame budget are the transferable parts."
)


def latency_stats(samples_ms: NDArray[np.float64]) -> dict[str, float]:
    """p50 / p90 / p99 / max / mean of per-call latencies, milliseconds."""
    s = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(s, 50.0)),
        "p90_ms": float(np.percentile(s, 90.0)),
        "p99_ms": float(np.percentile(s, 99.0)),
        "max_ms": float(s.max()),
        "mean_ms": float(s.mean()),
    }


def time_calls(
    fn: Callable[[int], Any],
    *,
    n_inputs: int,
    warmup: int,
    iters: int,
    minimums: tuple[int, int],
) -> dict[str, Any]:
    """Time `fn(i)` per call, cycling `i` over `n_inputs` real inputs.

    Args:
        fn: The call; receives the input index.
        n_inputs: Number of distinct inputs to cycle.
        warmup: Untimed calls first (>= minimums[0]).
        iters: Timed calls (>= minimums[1]).
        minimums: `(warmup, timed)` methodology minimums (P1's).

    Returns:
        `latency_stats` plus the iteration counts.

    Raises:
        ValueError: Below the methodology minimums.
    """
    if warmup < minimums[0] or iters < minimums[1]:
        raise ValueError(
            f"methodology minimums are warmup >= {minimums[0]}, iters >= {minimums[1]}"
        )
    for i in range(warmup):
        fn(i % n_inputs)
    samples = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        started = time.perf_counter()
        fn(i % n_inputs)
        samples[i] = (time.perf_counter() - started) * 1e3
    return {**latency_stats(samples), "warmup_iters": warmup, "timed_iters": iters}


def uncertainty_inputs(
    *,
    valid: bool,
    rotation_quat: NDArray[np.float64],
    translation: NDArray[np.float64],
    keypoints_full: NDArray[np.float64],
    confidence: NDArray[np.float64],
    bbox: NDArray[np.float64],
    cov_chol_crop: NDArray[np.float64],
    keypoint_empty: NDArray[np.bool_],
    affine: NDArray[np.float64],
) -> tuple[FrameEstimate, NDArray[np.float64]]:
    """The deployment path from graph outputs to the head's inputs (part of the timed stage).

    `cov_chol` (crop px) -> Σ̂ crop -> Σ̂ full frame (`crop_cov_to_full`); confidence mean as P1's
    evaluator stores it (float32).
    """
    cov_full = crop_cov_to_full(chol_to_cov(cov_chol_crop)[None], affine[None])[0]
    estimate = FrameEstimate(
        valid=valid,
        q_hat=rotation_quat,
        t_hat=translation,
        keypoints=keypoints_full,
        confidence_mean=float(np.asarray(confidence, np.float32).mean(dtype=np.float32)),
        bbox=bbox,
        unconstrained=np.asarray(keypoint_empty, dtype=bool),
    )
    return estimate, cov_full


@dataclass
class _Output:
    stage_times_ms: dict[str, float]
    answers: dict[str, bool]


class UncertaintyFrame:
    """P1's ORT pipeline (variance graph) plus the conformal heads as one `uncertainty` stage.

    Callable like `PosePipeline` (`frame -> object with stage_times_ms`), so P1's
    `end_to_end_frame_budget` times it unchanged. The `uncertainty` stage covers the covariance
    mapping and every head's `predict` (C2 includes its linearisation).
    """

    def __init__(self, pipeline: Any, heads: Sequence[ConformalPoseHead]) -> None:
        """Wrap a pipeline built by `p1_adapter.ort_pose_pipeline(..., with_uncertainty=True)`.

        Args:
            pipeline: The ORT pipeline.
            heads: The heads to run per frame.
        """
        self.pipeline = pipeline
        self.heads = list(heads)

    def __call__(self, frame: NDArray[np.uint8], bbox: Any = None) -> _Output:
        """Run one frame; `bbox` is accepted for signature parity and must be None."""
        if bbox is not None:
            raise ValueError("UncertaintyFrame runs end to end (predicted crop); no bbox")
        out = self.pipeline(frame)
        started = time.perf_counter()
        solved = bool(out.pnp.success)
        estimate, cov_full = uncertainty_inputs(
            valid=solved,
            rotation_quat=(
                self.pipeline.pose_quaternion(out.pnp.rotation) if solved else np.full(4, np.nan)
            ),
            translation=(
                np.asarray(out.pnp.translation, np.float64) if solved else np.full(3, np.nan)
            ),
            keypoints_full=np.asarray(out.keypoints_2d, dtype=np.float64),
            confidence=out.confidence,
            bbox=np.asarray(out.bbox, dtype=np.float64),
            cov_chol_crop=self.pipeline.last_cov_chol[0],
            keypoint_empty=self.pipeline.last_keypoint_empty[0],
            affine=np.asarray(self.pipeline.last_affine, dtype=np.float64),
        )
        answers = {
            head.score_id: isinstance(head.predict(estimate, cov=cov_full), Answer)
            for head in self.heads
        }
        times = dict(out.stage_times_ms)
        times["uncertainty"] = (time.perf_counter() - started) * 1e3
        return _Output(stage_times_ms=times, answers=answers)
