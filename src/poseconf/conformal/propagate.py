"""Keypoint sets to pose: PURSE membership (exact), linearised and sampled pose-space extent.

The PURSE (Yang & Pavone 2023) of a frame is the set of poses whose projected keypoints all lie in
their keypoint sets. Here "all" means every keypoint the pose projects *into the full frame*
(P1's `visible` rule: positive depth, inside the distortion model's monotonic range, inside the
image), excluding unconstrained keypoints, whose set is the whole image. That is exactly the
keypoint-inclusion rule of B1/B2 applied to the candidate pose, so for the true pose

    true pose in PURSE  <=>  every included true keypoint inside its set  <=>  joint score <= q,

the joint keypoint coverage event (`PurseSet.contains` uses the score's own error helper, so the
equivalence holds bit-for-bit, not up to rounding).

The PURSE is unbounded: a pose that projects no constrained keypoint into the frame (behind the
camera, far off-axis) meets no constraint and is a member at every q >= 0, so its global extent is
pi in rotation and infinite in translation. The estimators below describe the PURSE *near the PnP
estimate*, which is what a pose set around a point estimate means.

Pose-space extent near the PnP estimate, two estimators, neither a bound:

* `linearised_extent` / `scale_linearised`: first order in a tangent perturbation
  delta = (omega, dt), where R = Exp(omega) R_hat (so ||omega|| is the geodesic angle) and
  t = t_hat + dt, with the projection Jacobian from `cv2.projectPoints`, over the keypoints U
  visible under the estimate. With unit shapes S_k (B2: Sigma_k; B1: d_hat^2 I) and residuals
  r_k = pi_k(theta_hat) - y_hat_k (PnP does not reproduce the predicted keypoints exactly), let
  H = sum J_k^T S_k^-1 J_k, g = sum J_k^T S_k^-1 r_k, delta* = -H^-1 g and
  c = sum r_k^T S_k^-1 r_k - g^T H^-1 g >= 0. Then

      sum_k ||S_k^-1/2 (r_k + J_k delta)||^2 = (delta - delta*)^T H (delta - delta*) + c,

  so the linearised PURSE L = {delta : ||S_k^-1/2 (r_k + J_k delta)|| <= q for all k} contains
  E_in = {sum <= q^2} and lies inside E_out = {sum <= |U| q^2}: ellipsoids centred at delta* with
  squared H-radii q^2 - c and |U| q^2 - c. With lambda the largest eigenvalue of the (H^-1)
  rotation (translation) block and o that block of delta*:
  - *inner* radius sqrt(||o||^2 + (q^2 - c) lambda): reached or exceeded by a point of E_in, so
    it is at most the extent of L about the estimate (a loose lower estimate);
  - *outer* radius ||o|| + sqrt((|U| q^2 - c) lambda): at least the extent of E_out, hence of L.
  E_in is empty when c > q^2 (inner NaN); L is empty when c > |U| q^2 (both NaN), and may also be
  empty with a finite outer radius. With zero
  residual they reduce to q sqrt(lambda) and sqrt(|U|) q sqrt(lambda). Linearisation and a fixed U
  make both an approximation, not a bound.
* `sampled_extent`: M keypoint configurations drawn uniformly inside the sets, iterative PnP
  warm-started at the estimate, keeping only solutions that are PURSE members; the largest
  geodesic / translation deviation among them. Every kept pose is in the PURSE, so this is an
  *inner* approximation of the PURSE's extent near the estimate: a lower bound on how far the
  members that local PnP reaches from the sets lie from it. Offline only.

Geometry (wireframe, intrinsics, distortion, image size, P1's projection constants) comes in as a
`CameraGeometry`; `poseconf.p1_adapter.projection_geometry` builds it from P1. This module imports
OpenCV (`cv2.projectPoints`, `cv2.solvePnP`, `cv2.Rodrigues`) and is the only one in
`poseconf.conformal` that may (docs/DECISIONS.md, 2026-10-07); it never imports torch or speedpose.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from poseconf.conformal.scores import KeypointSet, _joint
from poseconf.conformal.so3 import quat_to_matrix

__all__ = [
    "CameraGeometry",
    "LinearisedExtent",
    "PurseSet",
    "SampledExtent",
    "crop_cov_to_full",
    "left_jacobian",
    "linearised_extent",
    "project",
    "projection_jacobian",
    "sampled_extent",
    "scale_linearised",
    "unit_shapes",
]

#: Fewest used keypoints for which the 6-DoF linearised information matrix can be full rank.
MIN_LINEARISED_KEYPOINTS = 3

#: Fewest used keypoints `cv2.solvePnP(SOLVEPNP_ITERATIVE)` accepts.
MIN_SAMPLED_KEYPOINTS = 4

#: Below this rotation angle the SO(3) left Jacobian uses its Taylor expansion.
_SMALL_ANGLE = 1e-6


@dataclass(frozen=True)
class CameraGeometry:
    """Everything the projection needs, P1's model and constants (no speedpose import).

    Attributes:
        wireframe: (K, 3) body-frame keypoints, metres.
        camera_matrix: (3, 3) intrinsics K (no skew).
        distortion: (5,) OpenCV (k1, k2, p1, p2, k3), or None when the convention applies none.
        width: Image width, px.
        height: Image height, px.
        min_depth: P1's `_MIN_DEPTH_M`: points at or nearer than this are not visible.
        max_normalised_radius: P1's `_MAX_MONOTONIC_NORMALISED_RADIUS`: beyond it the distortion
            polynomial folds back and the point is not visible.
    """

    wireframe: NDArray[np.float64]
    camera_matrix: NDArray[np.float64]
    distortion: NDArray[np.float64] | None
    width: int
    height: int
    min_depth: float
    max_normalised_radius: float

    @property
    def dist_coeffs(self) -> NDArray[np.float64]:
        """(5, 1) coefficients for OpenCV; zeros when no distortion is applied."""
        if self.distortion is None:
            return np.zeros((5, 1))
        return np.asarray(self.distortion, dtype=np.float64).reshape(-1)[:5].reshape(5, 1)


# --------------------------------------------------------------------------------------------------
# covariance frames
# --------------------------------------------------------------------------------------------------


def crop_cov_to_full(cov_crop: ArrayLike, affines: ArrayLike) -> NDArray[np.float64]:
    """Map crop-px^2 covariances to full-frame px^2: Sigma_full = A^-1 Sigma_crop A^-T.

    Args:
        cov_crop: (n, K, 2, 2) covariances in crop px^2.
        affines: (n, 2, 3) full-frame -> crop affines; A is their 2x2 linear part.

    Returns:
        (n, K, 2, 2) covariances in full-frame px^2. A Mahalanobis distance is invariant under the
        map: a crop residual r and its full-frame image A^-1 r have the same distance.
    """
    cov = np.asarray(cov_crop, dtype=np.float64)
    aff = np.asarray(affines, dtype=np.float64)
    if cov.ndim != 4 or cov.shape[-2:] != (2, 2) or aff.shape != (cov.shape[0], 2, 3):
        raise ValueError(
            f"need cov (n, K, 2, 2) and affines (n, 2, 3); got {cov.shape}, {aff.shape}"
        )
    a_inv = np.linalg.inv(aff[:, :, :2])
    return np.einsum("nij,nkjl,nml->nkim", a_inv, cov, a_inv)


# --------------------------------------------------------------------------------------------------
# projection (numpy port of P1's `speedpose.geometry.projection.project_points`)
# --------------------------------------------------------------------------------------------------


def project(
    rotation: ArrayLike, translation: ArrayLike, geometry: CameraGeometry
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Project the wireframe for n poses with P1's camera model and visibility rule.

    Args:
        rotation: (n, 3, 3) body -> camera rotations.
        translation: (n, 3) camera-frame translations, metres.
        geometry: The camera and wireframe.

    Returns:
        `(points, in_frame)`: (n, K, 2) full-frame px (NaN where not modellable) and (n, K) P1's
        `visible`: depth > min_depth, within the monotonic distortion range, and inside
        [0, W-1] x [0, H-1].
    """
    rot = np.asarray(rotation, dtype=np.float64)
    off = np.asarray(translation, dtype=np.float64)
    if rot.ndim != 3 or rot.shape[1:] != (3, 3) or off.shape != (rot.shape[0], 3):
        raise ValueError(
            f"need rotation (n, 3, 3), translation (n, 3); got {rot.shape}, {off.shape}"
        )
    cam = geometry.wireframe[None] @ rot.transpose(0, 2, 1) + off[:, None, :]
    depth = cam[..., 2]
    in_front = depth > geometry.min_depth
    modellable = in_front.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(in_front, cam[..., 0] / depth, np.nan)
        y = np.where(in_front, cam[..., 1] / depth, np.nan)
    if geometry.distortion is not None:
        k1, k2, p1, p2, k3 = geometry.dist_coeffs.reshape(-1).tolist()
        r2 = x * x + y * y
        with np.errstate(invalid="ignore"):
            within = r2 <= geometry.max_normalised_radius**2
        modellable &= within
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x, y = (
            x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        )
        x = np.where(within, x, np.nan)
        y = np.where(within, y, np.nan)
    k = geometry.camera_matrix
    points = np.stack([k[0, 0] * x + k[0, 2], k[1, 1] * y + k[1, 2]], axis=-1)
    with np.errstate(invalid="ignore"):
        inside = (
            (points[..., 0] >= 0.0)
            & (points[..., 0] <= geometry.width - 1)
            & (points[..., 1] >= 0.0)
            & (points[..., 1] <= geometry.height - 1)
        )
    return points, modellable & inside


# --------------------------------------------------------------------------------------------------
# PURSE membership
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PurseSet:
    """The PURSE of each frame: poses whose visible, constrained keypoints all lie in their sets.

    Built from a `KeypointSet` (predictions and q only) and the camera; no label enters.

    Attributes:
        keypoints: The per-frame keypoint sets (B1 discs or B2/C1 ellipses) at quantile q.
        geometry: Camera and wireframe.
    """

    keypoints: KeypointSet
    geometry: CameraGeometry

    def contains_pose(self, rotation: ArrayLike, translation: ArrayLike) -> NDArray[np.bool_]:
        """Membership of one candidate pose per frame (rotation matrices).

        Valid frames: every keypoint visible under the candidate and constrained lies in its set
        (vacuously true when there is none). Failed frames: True only when q = +inf, the
        `KeypointSet.contains` semantics; this is not the `abstain_allowed` coverage indicator.

        Args:
            rotation: (n, 3, 3) candidate rotations (rows of failed frames are ignored).
            translation: (n, 3) candidate translations.
        """
        ks = self.keypoints
        ok = ks.valid
        rot = np.asarray(rotation, dtype=np.float64)
        off = np.asarray(translation, dtype=np.float64)
        n = ok.shape[0]
        if rot.shape != (n, 3, 3) or off.shape != (n, 3):
            raise ValueError(f"need rotation ({n}, 3, 3), translation ({n}, 3)")
        if not (np.all(np.isfinite(rot[ok])) and np.all(np.isfinite(off[ok]))):
            raise ValueError("candidate pose has NaN or inf on a valid frame")
        inside = np.full(n, ks.q == math.inf)
        if ok.any():
            points, visible = project(rot[ok], off[ok], self.geometry)
            errors = ks.keypoint_errors(points, ok)
            inside[ok] = _joint(errors, visible & ks.constrained[ok]) <= ks.q
        return inside

    def contains(self, q_pose: ArrayLike, t_pose: ArrayLike, order: str) -> NDArray[np.bool_]:
        """`contains_pose` for quaternions in `order` (`"scalar_first"` / `"scalar_last"`)."""
        qp = np.array(q_pose, dtype=np.float64)
        qp[~self.keypoints.valid] = (1.0, 0.0, 0.0, 0.0)  # ignored rows; keep the converter happy
        return self.contains_pose(quat_to_matrix(qp, order), t_pose)


# --------------------------------------------------------------------------------------------------
# linearised extent
# --------------------------------------------------------------------------------------------------


def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def left_jacobian(rvec: ArrayLike) -> NDArray[np.float64]:
    """SO(3) left Jacobian J_l(r): Exp(r + d) = Exp(J_l(r) d) Exp(r) to first order.

    J_l = I + (1 - cos a)/a^2 [r]x + (a - sin a)/a^3 [r]x^2, a = ||r|| (Taylor below 1e-6).
    It is invertible for a in [0, pi] (det = 2(1 - cos a)/a^2 > 0), the range `cv2.Rodrigues` returns.
    """
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    a = float(np.linalg.norm(r))
    w = _skew(r)
    if a < _SMALL_ANGLE:
        return np.eye(3) + 0.5 * w + w @ w / 6.0
    return np.eye(3) + (1.0 - math.cos(a)) / a**2 * w + (a - math.sin(a)) / a**3 * (w @ w)


def projection_jacobian(
    rotation: ArrayLike, translation: ArrayLike, geometry: CameraGeometry
) -> NDArray[np.float64]:
    """d(projected keypoints)/d(omega, t) at one pose, (K, 2, 6), from `cv2.projectPoints`.

    OpenCV returns the derivative with respect to the Rodrigues vector r; it is mapped to the left
    tangent perturbation omega (R = Exp(omega) R_hat, ||omega|| = geodesic angle) by
    d/d omega = d/dr J_l(r)^-1. Translation columns are d/dt as returned.

    Args:
        rotation: (3, 3) body -> camera rotation.
        translation: (3,) translation, metres.
        geometry: Camera and wireframe (distortion included when the convention applies it).
    """
    rvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    tvec = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    _, jac = cv2.projectPoints(
        geometry.wireframe.reshape(-1, 1, 3),
        rvec,
        tvec,
        geometry.camera_matrix,
        geometry.dist_coeffs,
    )
    k = geometry.wireframe.shape[0]
    jac = jac.reshape(k, 2, -1)
    d_rot = jac[..., :3] @ np.linalg.inv(left_jacobian(rvec))
    return np.concatenate([d_rot, jac[..., 3:6]], axis=-1)


def unit_shapes(keypoints: KeypointSet) -> NDArray[np.float64]:
    """(n, K, 2, 2) set shapes at q = 1: Sigma_k (ellipses) or d_hat^2 I (discs)."""
    if keypoints.cov is not None:
        return keypoints.cov
    d2 = np.where(keypoints.valid, np.asarray(keypoints.d_hat, dtype=np.float64), 1.0) ** 2
    return d2[:, None, None, None] * np.broadcast_to(
        np.eye(2), (*keypoints.centers.shape[:2], 2, 2)
    )


@dataclass(frozen=True)
class LinearisedExtent:
    """Residual-aware linearised PURSE geometry of one frame; `scale_linearised` applies q.

    Attributes:
        rot_var: lambda_max((H^-1)_omega omega), rad^2 per unit q^2 (inf if H is singular).
        trans_var: lambda_max((H^-1)_tt), m^2 per unit q^2.
        rot_offset: ||delta*_omega||, radians: where the linearised PURSE is centred.
        trans_offset: ||delta*_t||, metres.
        c: Residual floor sum r^T S^-1 r - g^T H^-1 g (>= 0), unit-q^2 Mahalanobis units.
        n_used: |U|, the keypoints in the information matrix.
        residual_max: max_k sqrt(r_k^T S_k^-1 r_k) over U: with U fixed at the estimate's
            visibility, the estimate is inside its own PURSE iff this is <= q.
    """

    rot_var: float
    trans_var: float
    rot_offset: float
    trans_offset: float
    c: float
    n_used: int
    residual_max: float


def linearised_extent(
    jacobian: ArrayLike, shapes: ArrayLike, use: ArrayLike, residual: ArrayLike
) -> LinearisedExtent:
    """Linearised PURSE geometry about the estimate (module docstring).

    Args:
        jacobian: (K, 2, 6) d(pixels)/d(omega, t) at the estimate.
        shapes: (K, 2, 2) unit set shapes (positive-definite on used keypoints).
        use: (K,) bool, the keypoints U (visible under the estimate and constrained).
        residual: (K, 2) pi_k(estimate) - y_hat_k, px (finite on used keypoints).

    Returns:
        The geometry; infinite variances when |U| < 3 or H is not positive-definite.
    """
    jac = np.asarray(jacobian, dtype=np.float64)
    sel = np.asarray(use, dtype=bool)
    n_used = int(sel.sum())
    if n_used == 0:
        return LinearisedExtent(math.inf, math.inf, 0.0, 0.0, 0.0, 0, 0.0)
    j = jac[sel]
    inv = np.linalg.inv(np.asarray(shapes, dtype=np.float64)[sel])
    r = np.asarray(residual, dtype=np.float64)[sel]
    per_keypoint = np.einsum("ka,kab,kb->k", r, inv, r)
    residual_max = float(np.sqrt(per_keypoint.max()))
    singular = LinearisedExtent(math.inf, math.inf, 0.0, 0.0, 0.0, n_used, residual_max)
    if n_used < MIN_LINEARISED_KEYPOINTS:
        return singular
    info = np.einsum("kai,kab,kbj->ij", j, inv, j)
    info = 0.5 * (info + info.T)
    eig, vec = np.linalg.eigh(info)
    if not (np.all(np.isfinite(eig)) and eig[0] > eig[-1] * np.finfo(np.float64).eps * 6):
        return singular
    cov = (vec / eig) @ vec.T
    g = np.einsum("kai,kab,kb->i", j, inv, r)
    delta = -cov @ g
    return LinearisedExtent(
        rot_var=float(np.linalg.eigvalsh(cov[:3, :3])[-1]),
        trans_var=float(np.linalg.eigvalsh(cov[3:, 3:])[-1]),
        rot_offset=float(np.linalg.norm(delta[:3])),
        trans_offset=float(np.linalg.norm(delta[3:])),
        c=max(float(per_keypoint.sum() - g @ cov @ g), 0.0),
        n_used=n_used,
        residual_max=residual_max,
    )


def scale_linearised(extent: LinearisedExtent, q: float) -> dict[str, float]:
    """Inner and outer linearised radii at quantile q (module docstring).

    q = +inf: inf. q = -inf (no set): NaN. Infinite variance (|U| < 3, singular H): inf for any
    q >= 0. Inner is NaN when c > q^2 (the inner ellipsoid is empty); both are NaN when
    c > |U| q^2 (the linearised PURSE is empty).

    Returns:
        `rot_inner`, `rot_outer` (radians) and `trans_inner`, `trans_outer` (metres).
    """
    keys = ("rot_inner", "rot_outer", "trans_inner", "trans_outer")
    if q == -math.inf:
        return dict.fromkeys(keys, math.nan)
    if q < 0:
        raise ValueError(f"a keypoint-set quantile is >= 0 or -inf, got {q}")
    if math.isinf(q) or math.isinf(extent.rot_var):
        return dict.fromkeys(keys, math.inf)
    inner_sq = q * q - extent.c
    outer_sq = extent.n_used * q * q - extent.c
    if outer_sq < 0:
        return dict.fromkeys(keys, math.nan)
    out: dict[str, float] = {}
    for name, var, offset in (
        ("rot", extent.rot_var, extent.rot_offset),
        ("trans", extent.trans_var, extent.trans_offset),
    ):
        out[f"{name}_inner"] = (
            math.sqrt(offset * offset + inner_sq * var) if inner_sq >= 0 else math.nan
        )
        out[f"{name}_outer"] = offset + math.sqrt(outer_sq * var)
    return out


# --------------------------------------------------------------------------------------------------
# sampled extent
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampledExtent:
    """Sampled (inner) pose extent for one frame.

    Attributes:
        rot: Largest geodesic angle to the estimate among accepted poses, radians (NaN if none).
        trans: Largest ||t - t_hat|| among accepted poses, metres (NaN if none).
        n_samples: Keypoint configurations drawn (M).
        n_solved: Configurations for which iterative PnP returned a finite pose.
        n_accepted: Solved poses inside the PURSE.
        seconds: Wall time for this frame.
    """

    rot: float
    trans: float
    n_samples: int
    n_solved: int
    n_accepted: int
    seconds: float


def _single_frame(keypoints: KeypointSet, row: int, copies: int) -> KeypointSet:
    """Frame `row`'s set repeated `copies` times (one candidate pose per copy)."""

    def rep(a: NDArray | None) -> NDArray | None:
        return None if a is None else np.repeat(a[row : row + 1], copies, axis=0)

    return KeypointSet(
        keypoints.q,
        rep(keypoints.centers),
        rep(keypoints.d_hat),
        rep(keypoints.cov),
        np.ones(copies, dtype=bool),
        rep(keypoints.unconstrained),
    )


def sampled_extent(
    keypoints: KeypointSet,
    row: int,
    rotation: ArrayLike,
    translation: ArrayLike,
    geometry: CameraGeometry,
    *,
    samples: int,
    rng: np.random.Generator,
) -> SampledExtent:
    """Inner approximation of the PURSE extent about the estimate, for frame `row`.

    Draws `samples` configurations y_k = y_hat_k + q L_k u_k (L_k L_k^T = unit shape, u_k uniform in
    the unit disc, i.e. uniform in each set) for the keypoints U visible under the estimate and
    constrained, solves `cv2.solvePnP(SOLVEPNP_ITERATIVE, useExtrinsicGuess=True)` from the estimate,
    and keeps the poses `PurseSet.contains_pose` accepts.

    Args:
        keypoints: The sets (any number of frames).
        row: The frame; must be valid, with finite q >= 0.
        rotation: (3, 3) the frame's PnP rotation.
        translation: (3,) the frame's PnP translation.
        geometry: Camera and wireframe.
        samples: M.
        rng: Random generator (seed per frame for reproducibility).

    Returns:
        The extent; rot/trans NaN when |U| < 4 or no pose is accepted.
    """
    start = time.perf_counter()
    q = keypoints.q
    if not keypoints.valid[row] or not (math.isfinite(q) and q >= 0):
        raise ValueError("sampled_extent needs a valid frame and a finite q >= 0")
    rot_hat = np.asarray(rotation, dtype=np.float64)
    t_hat = np.asarray(translation, dtype=np.float64).reshape(3)
    _, visible = project(rot_hat[None], t_hat[None], geometry)
    use = visible[0] & keypoints.constrained[row]
    k = int(use.sum())
    if k < MIN_SAMPLED_KEYPOINTS:
        return SampledExtent(math.nan, math.nan, samples, 0, 0, time.perf_counter() - start)
    chol = np.linalg.cholesky(unit_shapes(keypoints)[row][use])
    radius = np.sqrt(rng.uniform(size=(samples, k)))
    angle = rng.uniform(0.0, 2.0 * math.pi, size=(samples, k))
    disc = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)
    draws = keypoints.centers[row][use][None] + q * np.einsum("kij,mkj->mki", chol, disc)

    object_points = geometry.wireframe[use].reshape(-1, 1, 3)
    rvec_hat, _ = cv2.Rodrigues(rot_hat)
    rotations = np.empty((samples, 3, 3))
    translations = np.empty((samples, 3))
    solved = np.zeros(samples, dtype=bool)
    for m in range(samples):
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            draws[m].reshape(-1, 1, 2),
            geometry.camera_matrix,
            geometry.dist_coeffs,
            rvec=rvec_hat.copy(),
            tvec=t_hat.reshape(3, 1).copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok and np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)):
            rotations[m] = cv2.Rodrigues(rvec)[0]
            translations[m] = tvec.reshape(3)
            solved[m] = True
    n_solved = int(solved.sum())
    if n_solved == 0:
        return SampledExtent(math.nan, math.nan, samples, 0, 0, time.perf_counter() - start)
    purse = PurseSet(_single_frame(keypoints, row, n_solved), geometry)
    accepted = purse.contains_pose(rotations[solved], translations[solved])
    n_accepted = int(accepted.sum())
    if n_accepted == 0:
        return SampledExtent(math.nan, math.nan, samples, n_solved, 0, time.perf_counter() - start)
    kept_r = rotations[solved][accepted]
    angles = [float(np.linalg.norm(cv2.Rodrigues(r @ rot_hat.T)[0])) for r in kept_r]
    shifts = np.linalg.norm(translations[solved][accepted] - t_hat, axis=-1)
    return SampledExtent(
        max(angles),
        float(shifts.max()),
        samples,
        n_solved,
        n_accepted,
        time.perf_counter() - start,
    )
