"""numpy SO(3) helpers: geodesic distance 2*arccos(|<q1,q2>|) with clamping, ball membership.

The geodesic distance is the same formula as P1's `E_R` (speedpose.geometry.metrics), so Phase 3
parity compares like with like. It is invariant to the quaternion sign and, because it only uses an
inner product, to the component order — as long as both arguments use the same order. Conversions
to matrices need the order explicitly; P1's comes from its `pose_convention.yaml` via
`poseconf.p1_adapter.pose_quaternion_order()` and is passed in here as a string, since this package
may not import the adapter.

Inputs are renormalised after a tolerance check: a label quaternion of norm 1 - d otherwise
manufactures 2*arccos(1 - d) of rotation error that does not exist (P1, `_unit_quaternion`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "QUATERNION_ORDERS",
    "geodesic_distance",
    "in_rotation_ball",
    "matrix_geodesic_distance",
    "normalise_quaternions",
    "quat_to_matrix",
    "to_scalar_first",
]

#: Supported component orders: (w, x, y, z) and (x, y, z, w).
QUATERNION_ORDERS = ("scalar_first", "scalar_last")

#: Largest accepted deviation of an input quaternion's norm from 1 before renormalising.
UNIT_NORM_TOL = 1e-3


def normalise_quaternions(quats: ArrayLike) -> NDArray[np.float64]:
    """Check (..., 4) quaternions are unit within `UNIT_NORM_TOL`, then renormalise exactly.

    Raises:
        ValueError: On a wrong last dimension, NaN, or a norm too far from 1.
    """
    q = np.asarray(quats, dtype=np.float64)
    if q.shape[-1:] != (4,):
        raise ValueError(f"quaternions must have last dimension 4, got shape {q.shape}")
    if np.isnan(q).any():
        raise ValueError("quaternions contain NaN")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(np.abs(norm - 1.0) > UNIT_NORM_TOL):
        worst = float(np.max(np.abs(norm - 1.0)))
        raise ValueError(f"quaternions are not unit (worst |norm - 1| = {worst:.3g})")
    return q / norm


def geodesic_distance(q1: ArrayLike, q2: ArrayLike) -> NDArray[np.float64]:
    """Rotation angle between orientations, 2*arccos(clip(|<q1, q2>|, 0, 1)), in radians [0, pi].

    Args:
        q1: (..., 4) unit quaternions.
        q2: (..., 4) unit quaternions in the same component order.

    Returns:
        (...,) angles.
    """
    a, b = normalise_quaternions(q1), normalise_quaternions(q2)
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def in_rotation_ball(q: ArrayLike, q_center: ArrayLike, radius: ArrayLike) -> NDArray[np.bool_]:
    """Membership in the geodesic ball {R : d(R, R_center) <= radius}. radius may be +/-inf."""
    return geodesic_distance(q, q_center) <= np.asarray(radius, dtype=np.float64)


def to_scalar_first(quats: ArrayLike, order: str) -> NDArray[np.float64]:
    """Reorder (..., 4) quaternions to (w, x, y, z)."""
    q = np.asarray(quats, dtype=np.float64)
    if order == "scalar_first":
        return q
    if order == "scalar_last":
        return np.concatenate([q[..., 3:], q[..., :3]], axis=-1)
    raise ValueError(f"unknown quaternion order {order!r}; expected one of {QUATERNION_ORDERS}")


def quat_to_matrix(quats: ArrayLike, order: str) -> NDArray[np.float64]:
    """(..., 4) unit quaternions to (..., 3, 3) rotation matrices (active, Hamilton convention).

    Args:
        quats: Quaternions.
        order: `"scalar_first"` or `"scalar_last"`.
    """
    w, x, y, z = np.moveaxis(normalise_quaternions(to_scalar_first(quats, order)), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def matrix_geodesic_distance(r1: ArrayLike, r2: ArrayLike) -> NDArray[np.float64]:
    """arccos((trace(R1^T R2) - 1) / 2), clamped; a cross-check of `geodesic_distance`."""
    a, b = np.asarray(r1, dtype=np.float64), np.asarray(r2, dtype=np.float64)
    cos = (np.einsum("...ji,...ji->...", a, b) - 1.0) / 2.0
    return np.arccos(np.clip(cos, -1.0, 1.0))
