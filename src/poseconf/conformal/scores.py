"""Nonconformity scores A1-A3, B1-B2, C1-C2 (IMPLEMENTATION_PLAN.md section 1.3; docs/SCORES.md).

Each score is a pure `score_*(predictions, labels, valid, ..., convention) -> s` plus a set
constructor `set_*(predictions, valid, q, ...) -> set` that never takes labels (enforced by a
signature test). Sets are evaluated against labels only through their `contains` method.

Exactness: a set stores the same per-frame scales the score divides by, and `contains` recomputes
the normalised error with the same helper the score uses, so on valid frames `truth in set(q)` is
*bit-for-bit* `score <= q` rather than equal up to rounding at the boundary. On failed frames
`contains` is True only when q = +inf (whole space). That matches `score <= q` under
`answer_required` but NOT under `abstain_allowed`, where a failure is covered by abstaining.
Compute coverage with `metrics.outcomes`, never with `contains` alone.

Failures: a frame without a point estimate (`valid = False`) scores `+inf` under `answer_required`
and `-inf` under `abstain_allowed` (CLAUDE.md invariant 6). Its set is the whole space when
q = +inf and empty otherwise; `abstain` is True for it, and for every frame when q = -inf.

Normalisers (`c_R`, `c_t`, `c_z`, `c_xy`), the A3 difficulty `g(conf)`, the B2/C1 covariances and
the C2 pose standard deviations are *inputs*: they are fitted or predicted elsewhere (normalisers
and `g` on `val_tune` only) and must be computable without the frame's label.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from poseconf.conformal.so3 import geodesic_distance
from poseconf.conformal.split import failure_score

__all__ = [
    "SCORES",
    "KeypointSet",
    "PoseSet",
    "ScoreSpec",
    "median_normaliser",
    "pose_error_components",
    "score_a1",
    "score_a2",
    "score_a3",
    "score_b1",
    "score_b2",
    "score_c2",
    "score_mahalanobis",
    "set_a1",
    "set_a2",
    "set_a3",
    "set_b1",
    "set_b2",
    "set_c2",
    "set_mahalanobis",
]

Translation = Literal["ball", "cylinder"]

#: Relative tolerance for covariance symmetry.
_SYMMETRY_RTOL = 1e-9

# --------------------------------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------------------------------


def _as_valid(valid: ArrayLike) -> NDArray[np.bool_]:
    ok = np.asarray(valid)
    if ok.dtype != np.bool_ or ok.ndim != 1:
        raise ValueError(f"valid must be a 1-D bool array, got dtype {ok.dtype}, shape {ok.shape}")
    return ok


def _array(x: ArrayLike, name: str, shape: tuple[int, ...], ok: NDArray[np.bool_]) -> NDArray:
    """float64 array of the given shape, finite on valid rows (invalid rows may hold NaN)."""
    a = np.asarray(x, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if not np.all(np.isfinite(a[ok])):
        raise ValueError(f"{name} has NaN or inf on a valid frame")
    return a


def _positive_scalar(x: float, name: str) -> float:
    value = float(x)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"normaliser {name} must be finite and > 0, got {value}")
    return value


def _positive_per_frame(x: ArrayLike, name: str, ok: NDArray[np.bool_]) -> NDArray[np.float64]:
    a = _array(x, name, ok.shape, ok)
    if np.any(a[ok] <= 0.0):
        raise ValueError(f"{name} must be > 0 on every valid frame")
    return a


def _check_q(q: float) -> float:
    value = float(q)
    if math.isnan(value):
        raise ValueError("q is NaN")
    return value


def _frame_radius(scale: NDArray, q: float, ok: NDArray[np.bool_]) -> NDArray[np.float64]:
    """q * scale on valid frames; +inf (whole space) / -inf (no set) on invalid ones."""
    out = np.full(scale.shape, math.inf if q == math.inf else -math.inf)
    out[ok] = q * scale[ok]
    return out


# --------------------------------------------------------------------------------------------------
# pose scores (A1, A2, A3, C2)
# --------------------------------------------------------------------------------------------------


def _pose_errors(
    q_hat: NDArray, t_hat: NDArray, q_gt: NDArray, t_gt: NDArray, translation: Translation
) -> tuple[NDArray, NDArray]:
    """(E_R (n,), translation errors (n, m)) in radians / metres; m = 1 ball, 2 cylinder."""
    rot = geodesic_distance(q_hat, q_gt)
    dt = t_hat - t_gt
    if translation == "ball":
        trans = np.linalg.norm(dt, axis=-1)[:, None]
    else:
        trans = np.stack([np.abs(dt[:, 2]), np.linalg.norm(dt[:, :2], axis=-1)], axis=-1)
    return rot, trans


def _normalised_pose(
    rot: NDArray, trans: NDArray, rot_scale: NDArray, trans_scale: NDArray
) -> NDArray[np.float64]:
    return np.maximum(rot / rot_scale, np.max(trans / trans_scale, axis=-1))


@dataclass(frozen=True)
class PoseSet:
    """Rotation geodesic ball x translation ball (A1, A3, C2) or cylinder (A2), per frame.

    Attributes:
        q: The conformal quantile.
        q_center: (n, 4) predicted quaternions (any order, consistent with the labels').
        t_center: (n, 3) predicted translations, camera frame, metres.
        rot_scale: (n,) radians per unit score.
        trans_scale: (n, m) metres per unit score; m = 1 (ball) or 2 (boresight |dz|, lateral |dxy|).
        translation: `"ball"` or `"cylinder"`.
        valid: (n,) frames with a point estimate.
    """

    q: float
    q_center: NDArray[np.float64]
    t_center: NDArray[np.float64]
    rot_scale: NDArray[np.float64]
    trans_scale: NDArray[np.float64]
    translation: Translation
    valid: NDArray[np.bool_]

    @property
    def abstain(self) -> NDArray[np.bool_]:
        """No set issued: no point estimate, or q = -inf (an empty set is an abstention)."""
        return ~self.valid | (self.q == -math.inf)

    @property
    def rot_radius(self) -> NDArray[np.float64]:
        """(n,) geodesic-ball radius in radians."""
        return _frame_radius(self.rot_scale, self.q, self.valid)

    @property
    def trans_radius(self) -> NDArray[np.float64]:
        """(n, m) translation radii in metres (ball radius, or cylinder half-length and radius)."""
        out = np.full(self.trans_scale.shape, math.inf if self.q == math.inf else -math.inf)
        out[self.valid] = self.q * self.trans_scale[self.valid]
        return out

    def contains(self, q_gt: ArrayLike, t_gt: ArrayLike) -> NDArray[np.bool_]:
        """Evaluation only: whether each true pose lies in its set.

        Valid frames: exactly `score <= q`. Failed frames: True only when q = +inf. This is not the
        `abstain_allowed` coverage indicator; use `metrics.outcomes` (or `abstain | contains`).
        """
        ok = self.valid
        n = ok.shape[0]
        qg = _array(q_gt, "q_gt", (n, 4), ok)
        tg = _array(t_gt, "t_gt", (n, 3), ok)
        inside = np.full(n, self.q == math.inf)
        if ok.any():
            rot, trans = _pose_errors(
                self.q_center[ok], self.t_center[ok], qg[ok], tg[ok], self.translation
            )
            s = _normalised_pose(rot, trans, self.rot_scale[ok], self.trans_scale[ok])
            inside[ok] = s <= self.q
        return inside


def _pose_inputs(
    q_hat: ArrayLike, t_hat: ArrayLike, valid: ArrayLike
) -> tuple[NDArray, NDArray, NDArray[np.bool_], NDArray[np.float64]]:
    ok = _as_valid(valid)
    n = ok.shape[0]
    qh = _array(q_hat, "q_hat", (n, 4), ok)
    th = _array(t_hat, "t_hat", (n, 3), ok)
    rng = np.ones(n)
    rng[ok] = np.linalg.norm(th[ok], axis=-1)
    if np.any(rng[ok] <= 0.0):
        raise ValueError("predicted range ||t_hat|| must be > 0 on every valid frame")
    return qh, th, ok, rng


def _pose_score(
    q_hat: ArrayLike,
    t_hat: ArrayLike,
    q_gt: ArrayLike,
    t_gt: ArrayLike,
    valid: ArrayLike,
    scales: Callable[[NDArray[np.bool_], NDArray[np.float64]], tuple[NDArray, NDArray]],
    translation: Translation,
    convention: str,
) -> NDArray[np.float64]:
    fail = failure_score(convention)
    qh, th, ok, pred_range = _pose_inputs(q_hat, t_hat, valid)
    n = ok.shape[0]
    qg = _array(q_gt, "q_gt", (n, 4), ok)
    tg = _array(t_gt, "t_gt", (n, 3), ok)
    rot_scale, trans_scale = scales(ok, pred_range)
    out = np.full(n, fail)
    if ok.any():
        rot, trans = _pose_errors(qh[ok], th[ok], qg[ok], tg[ok], translation)
        out[ok] = _normalised_pose(rot, trans, rot_scale[ok], trans_scale[ok])
    return out


def _pose_set(
    q_hat: ArrayLike,
    t_hat: ArrayLike,
    valid: ArrayLike,
    q: float,
    scales: Callable[[NDArray[np.bool_], NDArray[np.float64]], tuple[NDArray, NDArray]],
    translation: Translation,
) -> PoseSet:
    q = _check_q(q)
    qh, th, ok, pred_range = _pose_inputs(q_hat, t_hat, valid)
    rot_scale, trans_scale = scales(ok, pred_range)
    return PoseSet(q, qh, th, rot_scale, trans_scale, translation, ok)


def _a1_scales(c_R: float, c_t: float, difficulty: ArrayLike | None = None):
    c_r = _positive_scalar(c_R, "c_R")
    c_tr = _positive_scalar(c_t, "c_t")

    def scales(ok: NDArray[np.bool_], pred_range: NDArray) -> tuple[NDArray, NDArray]:
        g = (
            np.ones(ok.shape[0])
            if difficulty is None
            else _positive_per_frame(difficulty, "difficulty", ok)
        )
        g = np.where(ok, g, 1.0)
        return c_r * g, (c_tr * pred_range * g)[:, None]

    return scales


def score_a1(
    q_hat: ArrayLike,
    t_hat: ArrayLike,
    q_gt: ArrayLike,
    t_gt: ArrayLike,
    valid: ArrayLike,
    *,
    c_R: float,
    c_t: float,
    convention: str,
) -> NDArray[np.float64]:
    """A1 = max(E_R / c_R, ||t_hat - t|| / (||t_hat|| c_t)).

    Args:
        q_hat: (n, 4) predicted quaternions. NaN allowed on invalid frames.
        t_hat: (n, 3) predicted translations, metres.
        q_gt: (n, 4) true quaternions (same order). Label.
        t_gt: (n, 3) true translations. Label.
        valid: (n,) bool, frame has a point estimate (PnP solved).
        c_R: Rotation normaliser, radians (median E_R on val_tune solved frames).
        c_t: Translation normaliser, dimensionless (median ||dt|| / ||t_hat|| on val_tune solved).
        convention: PnP-failure convention.

    Returns:
        (n,) scores, +/-inf on invalid frames.
    """
    return _pose_score(q_hat, t_hat, q_gt, t_gt, valid, _a1_scales(c_R, c_t), "ball", convention)


def set_a1(
    q_hat: ArrayLike, t_hat: ArrayLike, valid: ArrayLike, q: float, *, c_R: float, c_t: float
) -> PoseSet:
    """A1 set: geodesic ball radius q c_R x translation ball radius q c_t ||t_hat||."""
    return _pose_set(q_hat, t_hat, valid, q, _a1_scales(c_R, c_t), "ball")


def score_a3(
    q_hat: ArrayLike,
    t_hat: ArrayLike,
    q_gt: ArrayLike,
    t_gt: ArrayLike,
    valid: ArrayLike,
    *,
    c_R: float,
    c_t: float,
    difficulty: ArrayLike,
    convention: str,
) -> NDArray[np.float64]:
    """A3 = A1 / g(conf), g > 0 a monotone fit of error on mean keypoint confidence (val_tune).

    Args:
        difficulty: (n,) g evaluated at each frame's *predicted* mean keypoint confidence.
        (others as `score_a1`)
    """
    scales = _a1_scales(c_R, c_t, difficulty)
    return _pose_score(q_hat, t_hat, q_gt, t_gt, valid, scales, "ball", convention)


def set_a3(
    q_hat: ArrayLike,
    t_hat: ArrayLike,
    valid: ArrayLike,
    q: float,
    *,
    c_R: float,
    c_t: float,
    difficulty: ArrayLike,
) -> PoseSet:
    """A3 set: A1's radii scaled per frame by g(conf)."""
    return _pose_set(q_hat, t_hat, valid, q, _a1_scales(c_R, c_t, difficulty), "ball")


def _a2_scales(c_R: float, c_z: float, c_xy: float):
    c_r = _positive_scalar(c_R, "c_R")
    cz = _positive_scalar(c_z, "c_z")
    cxy = _positive_scalar(c_xy, "c_xy")

    def scales(ok: NDArray[np.bool_], pred_range: NDArray) -> tuple[NDArray, NDArray]:
        return np.full(ok.shape[0], c_r), np.stack([cz * pred_range, cxy * pred_range], axis=-1)

    return scales


def score_a2(
    q_hat: ArrayLike,
    t_hat: ArrayLike,
    q_gt: ArrayLike,
    t_gt: ArrayLike,
    valid: ArrayLike,
    *,
    c_R: float,
    c_z: float,
    c_xy: float,
    convention: str,
) -> NDArray[np.float64]:
    """A2 = max(E_R / c_R, |dz| / (||t_hat|| c_z), ||dxy|| / (||t_hat|| c_xy)).

    z is the camera optical axis (boresight); xy is lateral. See `score_a1` for the arguments.
    """
    scales = _a2_scales(c_R, c_z, c_xy)
    return _pose_score(q_hat, t_hat, q_gt, t_gt, valid, scales, "cylinder", convention)


def set_a2(
    q_hat: ArrayLike,
    t_hat: ArrayLike,
    valid: ArrayLike,
    q: float,
    *,
    c_R: float,
    c_z: float,
    c_xy: float,
) -> PoseSet:
    """A2 set: geodesic ball x cylinder (half-length q c_z ||t_hat|| along z, radius q c_xy ||t_hat||)."""
    return _pose_set(q_hat, t_hat, valid, q, _a2_scales(c_R, c_z, c_xy), "cylinder")


def _c2_scales(sigma_R: ArrayLike, sigma_t: ArrayLike):
    def scales(ok: NDArray[np.bool_], pred_range: NDArray) -> tuple[NDArray, NDArray]:
        s_r = np.where(ok, _positive_per_frame(sigma_R, "sigma_R", ok), 1.0)
        s_t = np.where(ok, _positive_per_frame(sigma_t, "sigma_t", ok), 1.0)
        return s_r, s_t[:, None]

    return scales


def score_c2(
    q_hat: ArrayLike,
    t_hat: ArrayLike,
    q_gt: ArrayLike,
    t_gt: ArrayLike,
    valid: ArrayLike,
    *,
    sigma_R: ArrayLike,
    sigma_t: ArrayLike,
    convention: str,
) -> NDArray[np.float64]:
    """C2 = max(E_R / sigma_R, ||t_hat - t|| / sigma_t).

    Args:
        sigma_R: (n,) predicted rotation std (radians), linearised from C1's keypoint covariances.
        sigma_t: (n,) predicted translation std (metres), same source.
        (others as `score_a1`)
    """
    scales = _c2_scales(sigma_R, sigma_t)
    return _pose_score(q_hat, t_hat, q_gt, t_gt, valid, scales, "ball", convention)


def set_c2(
    q_hat: ArrayLike,
    t_hat: ArrayLike,
    valid: ArrayLike,
    q: float,
    *,
    sigma_R: ArrayLike,
    sigma_t: ArrayLike,
) -> PoseSet:
    """C2 set: geodesic ball radius q sigma_R x translation ball radius q sigma_t."""
    return _pose_set(q_hat, t_hat, valid, q, _c2_scales(sigma_R, sigma_t), "ball")


def pose_error_components(
    q_hat: ArrayLike, t_hat: ArrayLike, q_gt: ArrayLike, t_gt: ArrayLike
) -> dict[str, NDArray[np.float64]]:
    """Per-frame error components the pose normalisers are medians of (pass solved frames only).

    Returns:
        `rot` (E_R, rad), `trans_rel` (||dt|| / ||t_hat||), `boresight_rel` (|dz| / ||t_hat||),
        `lateral_rel` (||dxy|| / ||t_hat||).
    """
    qh = np.asarray(q_hat, dtype=np.float64)
    ok = np.ones(qh.shape[0], dtype=bool)
    qh, th, ok, pred_range = _pose_inputs(qh, t_hat, ok)
    qg = _array(q_gt, "q_gt", qh.shape, ok)
    tg = _array(t_gt, "t_gt", th.shape, ok)
    rot, ball = _pose_errors(qh, th, qg, tg, "ball")
    _, cyl = _pose_errors(qh, th, qg, tg, "cylinder")
    return {
        "rot": rot,
        "trans_rel": ball[:, 0] / pred_range,
        "boresight_rel": cyl[:, 0] / pred_range,
        "lateral_rel": cyl[:, 1] / pred_range,
    }


def median_normaliser(errors: ArrayLike, valid: ArrayLike) -> float:
    """Median of an error component over valid frames. Call on `val_tune` only.

    Raises:
        ValueError: If no valid frame, a non-finite valid error, or a median <= 0.
    """
    ok = _as_valid(valid)
    e = np.asarray(errors, dtype=np.float64)[ok]
    if e.size == 0 or not np.all(np.isfinite(e)):
        raise ValueError("median_normaliser needs at least one finite error on a valid frame")
    med = float(np.median(e))
    if med <= 0.0:
        raise ValueError(f"normaliser median must be > 0, got {med}")
    return med


# --------------------------------------------------------------------------------------------------
# keypoint scores (B1, B2, C1)
# --------------------------------------------------------------------------------------------------


def _check_cov(cov: ArrayLike, n: int, k: int, ok: NDArray[np.bool_]) -> NDArray[np.float64]:
    c = _array(cov, "cov", (n, k, 2, 2), ok)
    a, b, b2, d = c[ok, :, 0, 0], c[ok, :, 0, 1], c[ok, :, 1, 0], c[ok, :, 1, 1]
    if np.any(np.abs(b - b2) > _SYMMETRY_RTOL * (np.abs(a) + np.abs(d))):
        raise ValueError("cov must be symmetric")
    if np.any(a <= 0) or np.any(a * d - b * b <= 0):
        raise ValueError("cov must be positive-definite on every keypoint of every valid frame")
    return c


def _keypoint_errors(
    y_hat: NDArray, y_gt: NDArray, d_hat: NDArray | None, cov: NDArray | None
) -> NDArray[np.float64]:
    """(n, K) normalised per-keypoint errors: ||r|| / d_hat (B1) or sqrt(r^T cov^-1 r) (B2/C1)."""
    r = y_hat - y_gt
    if cov is None:
        return np.linalg.norm(r, axis=-1) / d_hat[:, None]
    a, b, d = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    rx, ry = r[..., 0], r[..., 1]
    m2 = (d * rx * rx - 2.0 * b * rx * ry + a * ry * ry) / (a * d - b * b)
    return np.sqrt(np.maximum(m2, 0.0))


def _joint(errors: NDArray, include: NDArray[np.bool_]) -> NDArray[np.float64]:
    """max over included keypoints; 0 for a frame with none (vacuously covered for q >= 0)."""
    worst = np.max(np.where(include, errors, -math.inf), axis=-1)
    return np.where(include.any(axis=-1), worst, 0.0)


@dataclass(frozen=True)
class KeypointSet:
    """K discs (B1) or K Mahalanobis ellipses (B2, C1) per frame, full-frame pixels.

    Attributes:
        q: The conformal quantile.
        centers: (n, K, 2) predicted keypoints.
        d_hat: (n,) predicted box diagonal, px (discs), or None.
        cov: (n, K, 2, 2) predicted covariances, px^2 (ellipses), or None.
        valid: (n,) frames with a point estimate.
    """

    q: float
    centers: NDArray[np.float64]
    d_hat: NDArray[np.float64] | None
    cov: NDArray[np.float64] | None
    valid: NDArray[np.bool_]

    @property
    def abstain(self) -> NDArray[np.bool_]:
        """No set issued: no point estimate, or q = -inf."""
        return ~self.valid | (self.q == -math.inf)

    @property
    def radius_px(self) -> NDArray[np.float64]:
        """(n,) largest keypoint radius in the frame: q d_hat (disc) or q sqrt(lambda_max) (ellipse)."""
        if self.cov is None:
            scale = self.d_hat
        else:
            scale = np.ones(self.valid.shape[0])
            eig = np.linalg.eigvalsh(self.cov[self.valid])
            scale[self.valid] = np.sqrt(eig[..., -1].max(axis=-1))
        return _frame_radius(scale, self.q, self.valid)

    def contains(self, y_gt: ArrayLike, include: ArrayLike) -> NDArray[np.bool_]:
        """Evaluation only: every included true keypoint inside its set.

        Valid frames: exactly `score <= q`. Failed frames: True only when q = +inf. This is not the
        `abstain_allowed` coverage indicator; use `metrics.outcomes` (or `abstain | contains`).

        Args:
            y_gt: (n, K, 2) true keypoint projections, full-frame px.
            include: (n, K) bool, GT projection inside the full frame — same rule as calibration.
        """
        ok = self.valid
        n, k = self.centers.shape[:2]
        inc = _include(include, n, k)
        yg = _array(y_gt, "y_gt", (n, k, 2), np.zeros(n, bool))
        _check_finite_included(yg, inc, ok)
        inside = np.full(n, self.q == math.inf)
        if ok.any():
            d = None if self.d_hat is None else self.d_hat[ok]
            c = None if self.cov is None else self.cov[ok]
            s = _joint(_keypoint_errors(self.centers[ok], yg[ok], d, c), inc[ok])
            inside[ok] = s <= self.q
        return inside


def _include(include: ArrayLike, n: int, k: int) -> NDArray[np.bool_]:
    inc = np.asarray(include)
    if inc.dtype != np.bool_ or inc.shape != (n, k):
        raise ValueError(f"include must be bool of shape {(n, k)}, got {inc.dtype} {inc.shape}")
    return inc


def _check_finite_included(y_gt: NDArray, inc: NDArray[np.bool_], ok: NDArray[np.bool_]) -> None:
    if not np.all(np.isfinite(y_gt[ok][inc[ok]])):
        raise ValueError("y_gt has NaN or inf on an included keypoint of a valid frame")


def _keypoint_score(
    y_hat: ArrayLike,
    y_gt: ArrayLike,
    include: ArrayLike,
    valid: ArrayLike,
    d_hat: ArrayLike | None,
    cov: ArrayLike | None,
    convention: str,
) -> NDArray[np.float64]:
    fail = failure_score(convention)
    st = _keypoint_set(y_hat, valid, 0.0, d_hat, cov)
    ok = st.valid
    n, k = st.centers.shape[:2]
    inc = _include(include, n, k)
    yg = np.asarray(y_gt, dtype=np.float64)
    if yg.shape != (n, k, 2):
        raise ValueError(f"y_gt must have shape {(n, k, 2)}, got {yg.shape}")
    _check_finite_included(yg, inc, ok)
    out = np.full(n, fail)
    if ok.any():
        d = None if st.d_hat is None else st.d_hat[ok]
        c = None if st.cov is None else st.cov[ok]
        out[ok] = _joint(_keypoint_errors(st.centers[ok], yg[ok], d, c), inc[ok])
    return out


def _keypoint_set(
    y_hat: ArrayLike, valid: ArrayLike, q: float, d_hat: ArrayLike | None, cov: ArrayLike | None
) -> KeypointSet:
    q = _check_q(q)
    ok = _as_valid(valid)
    yh = np.asarray(y_hat, dtype=np.float64)
    if yh.ndim != 3 or yh.shape[0] != ok.shape[0] or yh.shape[2] != 2:
        raise ValueError(f"y_hat must have shape (n, K, 2) with n = {ok.shape[0]}, got {yh.shape}")
    n, k = yh.shape[:2]
    yh = _array(yh, "y_hat", (n, k, 2), ok)
    d = None if d_hat is None else _positive_per_frame(d_hat, "d_hat", ok)
    c = None if cov is None else _check_cov(cov, n, k, ok)
    return KeypointSet(q, yh, d, c, ok)


def score_b1(
    y_hat: ArrayLike,
    y_gt: ArrayLike,
    include: ArrayLike,
    valid: ArrayLike,
    *,
    d_hat: ArrayLike,
    convention: str,
) -> NDArray[np.float64]:
    """B1 = max_k ||y_hat_k - y_k|| / d_hat over included keypoints.

    Args:
        y_hat: (n, K, 2) predicted keypoints, full-frame px.
        y_gt: (n, K, 2) true projections, full-frame px. Label.
        include: (n, K) bool, GT projection inside the full frame. Label-derived; same rule at
            calibration and evaluation. A frame with no included keypoint scores 0.
        valid: (n,) bool, frame has a point estimate.
        d_hat: (n,) predicted (detector) box diagonal, px.
        convention: PnP-failure convention.
    """
    return _keypoint_score(y_hat, y_gt, include, valid, d_hat, None, convention)


def set_b1(y_hat: ArrayLike, valid: ArrayLike, q: float, *, d_hat: ArrayLike) -> KeypointSet:
    """B1 set: K discs centred on y_hat_k, radius q d_hat."""
    return _keypoint_set(y_hat, valid, q, d_hat, None)


def score_mahalanobis(
    y_hat: ArrayLike,
    y_gt: ArrayLike,
    include: ArrayLike,
    valid: ArrayLike,
    *,
    cov: ArrayLike,
    convention: str,
) -> NDArray[np.float64]:
    """B2 / C1 = max_k sqrt(r_k^T cov_k^-1 r_k) over included keypoints.

    Args:
        cov: (n, K, 2, 2) predicted covariances in full-frame px^2 (B2: heatmap second moment;
            C1: variance head), positive-definite on every valid frame.
        (others as `score_b1`)
    """
    return _keypoint_score(y_hat, y_gt, include, valid, None, cov, convention)


def set_mahalanobis(y_hat: ArrayLike, valid: ArrayLike, q: float, *, cov: ArrayLike) -> KeypointSet:
    """B2 / C1 set: K ellipses {y : (y - y_hat_k)^T cov_k^-1 (y - y_hat_k) <= q^2}."""
    return _keypoint_set(y_hat, valid, q, None, cov)


score_b2 = score_mahalanobis
set_b2 = set_mahalanobis


# --------------------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreSpec:
    """One row of IMPLEMENTATION_PLAN.md section 1.3.

    Attributes:
        score_id: `"A1"` ... `"C2"`.
        level: `"pose"` or `"keypoint"`.
        score_fn: (predictions, labels, valid, ..., convention) -> scores.
        set_fn: (predictions, valid, q, ...) -> set. Never takes labels.
        inputs: What the score needs beyond P1's point predictions.
        covariance_source: For Mahalanobis scores, where cov comes from; else None.
    """

    score_id: str
    level: Literal["pose", "keypoint"]
    score_fn: Callable[..., NDArray[np.float64]]
    set_fn: Callable[..., Any]
    inputs: str
    covariance_source: str | None = None


SCORES: dict[str, ScoreSpec] = {
    "A1": ScoreSpec("A1", "pose", score_a1, set_a1, "sidecar; c_R, c_t from val_tune"),
    "A2": ScoreSpec("A2", "pose", score_a2, set_a2, "sidecar; c_R, c_z, c_xy from val_tune"),
    "A3": ScoreSpec("A3", "pose", score_a3, set_a3, "sidecar; c_R, c_t, g(conf) from val_tune"),
    "B1": ScoreSpec("B1", "keypoint", score_b1, set_b1, "dump; predicted box diagonal"),
    "B2": ScoreSpec(
        "B2",
        "keypoint",
        score_mahalanobis,
        set_mahalanobis,
        "dump; heatmap second moment mapped to full frame",
        covariance_source="heatmap_second_moment",
    ),
    "C1": ScoreSpec(
        "C1",
        "keypoint",
        score_mahalanobis,
        set_mahalanobis,
        "Phase 5 variance head, mapped to full frame",
        covariance_source="variance_head",
    ),
    "C2": ScoreSpec("C2", "pose", score_c2, set_c2, "Phase 4/5 linearised pose std from C1"),
}
