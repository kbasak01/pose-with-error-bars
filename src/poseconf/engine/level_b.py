"""Level B: keypoint-set conformal scores (B1, B2) from the Phase 3 dump, PURSE, propagation.

Data roles (CLAUDE.md invariants 3-5):
* B1 and B2 fit nothing: d_hat is the detector box diagonal and Sigma_k the heatmap moment, both
  predictions. `val_tune` is therefore not read by Level B.
* Scores use labels (the GT pose, projected with P1's model, and P1's visibility mask); sets
  (`keypoint_set`) and their PURSE use predictions and q only.
* Synthetic only in Phase 4: the dump labels come from `synthetic_labels.npz`.

The valid mask is P1's PnP success, as for Level A, so Levels A and B share one answer rate
(docs/DECISIONS.md 2026-10-06). An empty heatmap channel is an *unconstrained* keypoint (its set
is the whole image; docs/DECISIONS.md 2026-10-07).
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from poseconf.conformal.metrics import clopper_pearson, set_size_summary
from poseconf.conformal.propagate import (
    CameraGeometry,
    LinearisedExtent,
    PurseSet,
    crop_cov_to_full,
    linearised_extent,
    project,
    projection_jacobian,
    sampled_extent,
    scale_linearised,
    unit_shapes,
)
from poseconf.conformal.scores import SCORES, KeypointSet
from poseconf.conformal.so3 import geodesic_distance, quat_to_matrix

__all__ = [
    "LEVEL_B_SCORES",
    "KeypointFrames",
    "check_cov_mapping",
    "join_dump",
    "keypoint_set",
    "linearised_frames",
    "linearised_radii",
    "estimate_in_purse",
    "propagation_report",
    "purse_agreement",
    "runtime_summary",
    "sampled_frames",
    "score_frames",
    "set_size_report",
    "unit_radius_px",
    "worker_count",
]

#: The keypoint scores this module handles.
LEVEL_B_SCORES = ("B1", "B2")

_RAD_TO_DEG = 180.0 / math.pi


@dataclass(frozen=True)
class KeypointFrames:
    """Per-frame dump predictions joined to synthetic labels, in split-manifest order.

    Attributes:
        filenames: (n,) image filenames.
        y_hat: (n, K, 2) predicted keypoints, full-frame px.
        cov: (n, K, 2, 2) heatmap-moment covariances, full-frame px^2.
        unconstrained: (n, K) empty heatmap channels (prediction-side).
        d_hat: (n,) predicted (detector) box diagonal, px.
        valid: (n,) PnP solved.
        q_hat: (n, 4) PnP quaternion (P1 order), NaN on failure.
        t_hat: (n, 3) PnP translation, metres, NaN on failure.
        q_gt: (n, 4) label quaternion. Label.
        t_gt: (n, 3) label translation. Label.
        y_gt: (n, K, 2) the label pose projected with P1's model (NaN where not visible). Label.
        include: (n, K) P1's visibility of the label keypoints. Label-derived.
    """

    filenames: NDArray[np.str_]
    y_hat: NDArray[np.float64]
    cov: NDArray[np.float64]
    unconstrained: NDArray[np.bool_]
    d_hat: NDArray[np.float64]
    valid: NDArray[np.bool_]
    q_hat: NDArray[np.float64]
    t_hat: NDArray[np.float64]
    q_gt: NDArray[np.float64]
    t_gt: NDArray[np.float64]
    y_gt: NDArray[np.float64]
    include: NDArray[np.bool_]

    def __len__(self) -> int:
        """Number of frames."""
        return len(self.filenames)

    def subset(self, index: NDArray[np.int64]) -> KeypointFrames:
        """Rows `index`, in that order."""
        return KeypointFrames(
            **{name: getattr(self, name)[index] for name in self.__dataclass_fields__}
        )

    def concat(self, other: KeypointFrames) -> KeypointFrames:
        """These frames followed by `other`."""
        return KeypointFrames(
            **{
                name: np.concatenate([getattr(self, name), getattr(other, name)])
                for name in self.__dataclass_fields__
            }
        )


def _index_of(column: NDArray[np.str_], names: Sequence[str], source: str) -> NDArray[np.int64]:
    position = {str(name): i for i, name in enumerate(column)}
    if len(position) != len(column):
        raise ValueError(f"{source} has duplicate filenames")
    missing = [name for name in names if name not in position]
    if missing:
        raise ValueError(f"{len(missing)} names missing from {source}, e.g. {missing[:3]}")
    return np.array([position[name] for name in names], dtype=np.int64)


def check_cov_mapping(dump: dict[str, NDArray[Any]]) -> float:
    """Max relative difference between the dump's full-frame covariances and
    `propagate.crop_cov_to_full(heatmap_cov_crop, affine)` (an independent inversion).

    Non-empty channels only (empty ones are zero in both).
    """
    mapped = crop_cov_to_full(dump["heatmap_cov_crop"], dump["affine"])
    stored = np.asarray(dump["heatmap_cov_full"], dtype=np.float64)
    keep = ~np.asarray(dump["heatmap_empty"], dtype=bool)
    scale = np.abs(stored[keep]).max(axis=(-2, -1), keepdims=True)
    return float(np.max(np.abs(mapped[keep] - stored[keep]) / scale)) if keep.any() else 0.0


def join_dump(
    dump: dict[str, NDArray[Any]],
    labels: dict[str, NDArray[Any]],
    names: Sequence[str],
    geometry: CameraGeometry,
    order: str,
) -> tuple[KeypointFrames, dict[str, Any]]:
    """Select `names` from a prediction dump and its label file; project the label poses.

    Args:
        dump: Prediction-dump arrays (`load_dump`).
        labels: Label-file arrays (`load_dump_labels`).
        names: Filenames to select (a split manifest).
        geometry: P1's camera and wireframe.
        order: P1's quaternion order.

    Returns:
        `(frames, check)`. `check` records the label-product cross-check: P1's visibility mask
        must equal the projection's exactly; `max_abs_diff_px` is the largest distance between the
        projected label keypoints and P1's stored label product on visible keypoints.

    Raises:
        ValueError: On a join failure, a visibility mismatch, or a degenerate predicted box.
    """
    if len(set(names)) != len(names):
        raise ValueError("split names must be unique")
    d = _index_of(dump["filename"], names, "dump")
    lab = _index_of(labels["filename"], names, "labels")
    q_gt = np.asarray(labels["q_gt"], dtype=np.float64)[lab]
    t_gt = np.asarray(labels["t_gt"], dtype=np.float64)[lab]
    y_gt, visible = project(quat_to_matrix(q_gt, order), t_gt, geometry)
    include = np.asarray(labels["in_frame"], dtype=bool)[lab]
    if not np.array_equal(visible, include):
        raise ValueError("projected label visibility disagrees with P1's stored in_frame mask")
    stored = np.asarray(labels["kp_gt_full"], dtype=np.float64)[lab]
    box = np.asarray(dump["bbox_used"], dtype=np.float64)[d]
    d_hat = np.hypot(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    valid = np.asarray(dump["success"], dtype=bool)[d]
    if not np.all(np.isfinite(d_hat[valid]) & (d_hat[valid] > 0)):
        raise ValueError("a solved frame has a degenerate predicted box")
    frames = KeypointFrames(
        filenames=np.asarray(names),
        y_hat=np.asarray(dump["kp_pred_full"], dtype=np.float64)[d],
        cov=np.asarray(dump["heatmap_cov_full"], dtype=np.float64)[d],
        unconstrained=np.asarray(dump["heatmap_empty"], dtype=bool)[d],
        d_hat=d_hat,
        valid=valid,
        q_hat=np.asarray(dump["q_pred"], dtype=np.float64)[d],
        t_hat=np.asarray(dump["t_pred"], dtype=np.float64)[d],
        q_gt=q_gt,
        t_gt=t_gt,
        y_gt=y_gt,
        include=include,
    )
    check = {
        "visibility_equal": True,
        "max_abs_diff_px": float(np.max(np.abs(y_gt[include] - stored[include]))),
        "n_keypoints_checked": int(include.sum()),
        "n_frames_no_included_keypoint": int((~include.any(axis=1)).sum()),
        "n_unconstrained_keypoints": int(frames.unconstrained.sum()),
    }
    return frames, check


def _kwargs(score_id: str, frames: KeypointFrames) -> dict[str, Any]:
    if score_id == "B1":
        return {"d_hat": frames.d_hat}
    if score_id == "B2":
        return {"cov": frames.cov, "unconstrained": frames.unconstrained}
    raise ValueError(f"not a Level B score: {score_id!r}; expected one of {LEVEL_B_SCORES}")


def score_frames(score_id: str, frames: KeypointFrames, convention: str) -> NDArray[np.float64]:
    """Joint keypoint scores (uses labels), +/-inf on PnP failure per `convention`."""
    return SCORES[score_id].score_fn(
        frames.y_hat,
        frames.y_gt,
        frames.include,
        frames.valid,
        convention=convention,
        **_kwargs(score_id, frames),
    )


def keypoint_set(score_id: str, frames: KeypointFrames, q: float) -> KeypointSet:
    """The test-time keypoint sets: predictions and q only — no label is passed."""
    return SCORES[score_id].set_fn(frames.y_hat, frames.valid, q, **_kwargs(score_id, frames))


def unit_radius_px(score_id: str, frames: KeypointFrames) -> NDArray[np.float64]:
    """(n,) largest keypoint radius per unit q, px (NaN on failed frames); for re-split sizes."""
    radius = keypoint_set(score_id, frames, 1.0).radius_px
    return np.where(frames.valid, radius, np.nan)


def set_size_report(kset: KeypointSet, answered: NDArray[np.bool_]) -> dict[str, Any] | None:
    """Median / p90 of the largest keypoint radius (px) over answered frames; None if none."""
    if not answered.any():
        return None
    return {
        "kp_px": set_size_summary(kset.radius_px[answered]),
        "n_frames_with_unconstrained_keypoint": int(
            (kset.unconstrained[answered].any(axis=1)).sum()
        ),
    }


def purse_agreement(
    kset: KeypointSet,
    frames: KeypointFrames,
    scores: NDArray[np.float64],
    order: str,
    geometry: CameraGeometry,
) -> dict[str, int]:
    """Count frames where PURSE membership of the label pose equals the joint keypoint coverage.

    Valid frames only (on failed frames both follow the convention, not the geometry). Evaluation
    only: the label pose is the candidate.
    """
    ok = frames.valid
    member = PurseSet(kset, geometry).contains(frames.q_gt, frames.t_gt, order)
    covered = scores <= kset.q
    return {
        "n_checked": int(ok.sum()),
        "n_agree": int((member[ok] == covered[ok]).sum()),
        "n_member": int(member[ok].sum()),
    }


# --------------------------------------------------------------------------------------------------
# propagation
# --------------------------------------------------------------------------------------------------


def _pose_matrices(frames: KeypointFrames, order: str) -> NDArray[np.float64]:
    q = frames.q_hat.copy()
    q[~frames.valid] = (1.0, 0.0, 0.0, 0.0)
    return quat_to_matrix(q, order)


def linearised_frames(
    score_id: str, frames: KeypointFrames, geometry: CameraGeometry, order: str
) -> tuple[list[LinearisedExtent | None], NDArray[np.float64]]:
    """Linearised PURSE geometry (q-free) of every valid frame, and the per-frame wall time (s).

    U = keypoints visible under the PnP estimate and constrained (label-free); the residual is
    the estimate's projection minus the predicted keypoints. Failed frames get None and NaN time.
    The time covers the projection, the Jacobian and the 6x6 solve.
    """
    kset = keypoint_set(score_id, frames, 1.0)
    shapes = unit_shapes(kset)
    rotations = _pose_matrices(frames, order)
    out: list[LinearisedExtent | None] = [None] * len(frames)
    seconds = np.full(len(frames), np.nan)
    for i in np.flatnonzero(frames.valid):
        start = time.perf_counter()
        points, visible = project(rotations[i][None], frames.t_hat[i][None], geometry)
        use = visible[0] & kset.constrained[i]
        jac = projection_jacobian(rotations[i], frames.t_hat[i], geometry)
        out[i] = linearised_extent(jac, shapes[i], use, points[0] - frames.y_hat[i])
        seconds[i] = time.perf_counter() - start
    return out, seconds


def _sampled_chunk(payload: dict[str, Any]) -> list[tuple[int, Any]]:
    cv2.setNumThreads(0)
    kset, rotations, translations = payload["kset"], payload["rotations"], payload["translations"]
    rows: list[tuple[int, Any]] = []
    for local, row in enumerate(payload["rows"]):
        rng = np.random.default_rng(np.random.SeedSequence([*payload["seed_key"], int(row)]))
        rows.append(
            (
                int(row),
                sampled_extent(
                    kset,
                    local,
                    rotations[local],
                    translations[local],
                    payload["geometry"],
                    samples=payload["samples"],
                    rng=rng,
                ),
            )
        )
    return rows


def sampled_frames(
    kset: KeypointSet,
    frames: KeypointFrames,
    rows: NDArray[np.int64],
    geometry: CameraGeometry,
    order: str,
    *,
    samples: int,
    seed_key: Sequence[int],
    workers: int,
) -> tuple[dict[int, Any], float]:
    """`sampled_extent` for frames `rows` (valid, finite q), across processes.

    Each frame's generator is seeded by `SeedSequence([*seed_key, row])`, so results do not depend
    on the chunking. Returns `({row: SampledExtent}, wall seconds)`.
    """
    rotations = _pose_matrices(frames, order)
    chunks = [c for c in np.array_split(np.asarray(rows), max(workers, 1)) if c.size]

    def sub(chunk: NDArray[np.int64]) -> KeypointSet:
        return KeypointSet(
            kset.q,
            kset.centers[chunk],
            None if kset.d_hat is None else kset.d_hat[chunk],
            None if kset.cov is None else kset.cov[chunk],
            kset.valid[chunk],
            kset.unconstrained[chunk],
        )

    payloads = [
        {
            "kset": sub(chunk),
            "rotations": rotations[chunk],
            "translations": frames.t_hat[chunk],
            "rows": chunk,
            "geometry": geometry,
            "samples": samples,
            "seed_key": list(seed_key),
        }
        for chunk in chunks
    ]
    start = time.perf_counter()
    results: dict[int, Any] = {}
    if len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
            for chunk_rows in pool.map(_sampled_chunk, payloads):
                results.update(chunk_rows)
    else:
        for payload in payloads:
            results.update(_sampled_chunk(payload))
    return results, time.perf_counter() - start


def runtime_summary(seconds: NDArray[np.float64]) -> dict[str, float]:
    """p50 / p90 / p99 / max / total of per-frame wall times, milliseconds (total in seconds)."""
    s = np.asarray(seconds, dtype=np.float64)
    s = s[np.isfinite(s)]
    p50, p90, p99 = np.percentile(s, [50, 90, 99]) * 1e3
    return {
        "n_frames": int(s.size),
        "p50_ms": float(p50),
        "p90_ms": float(p90),
        "p99_ms": float(p99),
        "max_ms": float(s.max() * 1e3),
        "total_s": float(s.sum()),
    }


def _radius_summary(
    rot: NDArray[np.float64], trans: NDArray[np.float64], pred_range: NDArray[np.float64]
) -> dict[str, Any]:
    return {
        "rot_deg": set_size_summary(rot * _RAD_TO_DEG),
        "trans_m": set_size_summary(trans),
        "trans_frac": set_size_summary(trans / pred_range),
    }


def _measured(inside: NDArray[np.bool_]) -> dict[str, Any]:
    k, m = int(inside.sum()), int(inside.size)
    return {"value": k / m, "n": m, "coverage_ci95": list(clopper_pearson(k, m))}


def propagation_report(
    frames: KeypointFrames,
    rows: NDArray[np.int64],
    rot: NDArray[np.float64],
    trans: NDArray[np.float64],
) -> dict[str, Any]:
    """Radius summaries over `rows` (answered frames), plus the *measured* pose-ball coverage.

    `measured_ball_coverage` is the fraction of answered frames whose true pose has E_R <= rot and
    ||t - t_hat|| <= trans, with a Clopper-Pearson interval: an empirical check of the radius,
    using labels for evaluation only, not a guarantee. Rows with a NaN radius (sampled: nothing
    accepted; linearised: empty ellipsoid) are excluded from `value` and the radius summaries
    (`n_nan`) and counted as uncovered in `value_nan_as_uncovered`. An infinite radius covers
    (`n_inf`). Answer rate = answered / all frames of the evaluated split.
    """
    rot, trans = np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64)
    finite = ~(np.isnan(rot) | np.isnan(trans))
    keep = rows[finite]
    out: dict[str, Any] = {
        "n_total": len(frames),
        "n_answered": int(rows.size),
        "answer_rate": rows.size / len(frames),
        "n_nan": int((~finite).sum()),
        "n_inf": int((np.isinf(rot) | np.isinf(trans)).sum()),
    }
    if keep.size == 0:
        return out
    pred_range = np.linalg.norm(frames.t_hat[keep], axis=-1)
    e_r = geodesic_distance(frames.q_hat[keep], frames.q_gt[keep])
    e_t = np.linalg.norm(frames.t_hat[keep] - frames.t_gt[keep], axis=-1)
    inside = (e_r <= rot[finite]) & (e_t <= trans[finite])
    out.update(_radius_summary(rot[finite], trans[finite], pred_range))
    all_rows = np.zeros(rows.size, dtype=bool)
    all_rows[finite] = inside
    out["measured_ball_coverage"] = {
        **_measured(inside),
        "value_nan_as_uncovered": _measured(all_rows),
        "convention_note": "answered frames only; see answer_rate",
        "note": "measured on val_test; not a guarantee",
    }
    return out


def estimate_in_purse(
    kset: KeypointSet, frames: KeypointFrames, geometry: CameraGeometry, order: str
) -> NDArray[np.bool_]:
    """(n,) whether the PnP estimate lies in its own PURSE (label-free; failed frames False).

    When it does not, the PnP residual exceeds the set on some keypoint: the linearised ellipsoids
    are then centred away from the estimate and the sampled estimator may accept nothing.
    """
    member = PurseSet(kset, geometry).contains_pose(
        _pose_matrices(frames, order), np.where(frames.valid[:, None], frames.t_hat, 0.0)
    )
    return member & frames.valid


def linearised_radii(
    extents: Sequence[LinearisedExtent | None], rows: NDArray[np.int64], q: float
) -> dict[str, NDArray[np.float64]]:
    """Per-row inner/outer rotation (rad) and translation (m) radii at quantile q."""
    keys = ("rot_inner", "rot_outer", "trans_inner", "trans_outer")
    out = {key: np.empty(rows.size) for key in keys}
    for j, row in enumerate(rows):
        extent = extents[int(row)]
        if extent is None:
            raise ValueError(f"row {row} has no linearised extent (failed frame)")
        radii = scale_linearised(extent, q)
        for key in keys:
            out[key][j] = radii[key]
    return out


def worker_count(requested: int | None) -> int:
    """`requested`, or the CPU count when None."""
    return int(requested) if requested else int(os.cpu_count() or 1)
