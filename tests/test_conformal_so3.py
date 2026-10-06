"""numpy SO(3) helpers against scipy's Rotation as an independent oracle."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from poseconf.conformal.so3 import (
    QUATERNION_ORDERS,
    geodesic_distance,
    in_rotation_ball,
    matrix_geodesic_distance,
    normalise_quaternions,
    quat_to_matrix,
    to_scalar_first,
)


def _random_quats(n: int, seed: int) -> np.ndarray:
    """Scalar-first unit quaternions."""
    xyzw = Rotation.random(n, random_state=seed).as_quat()
    return np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1)


def test_geodesic_matches_scipy_magnitude() -> None:
    q1, q2 = _random_quats(500, 1), _random_quats(500, 2)
    r1 = Rotation.from_quat(np.roll(q1, -1, axis=1))
    r2 = Rotation.from_quat(np.roll(q2, -1, axis=1))
    expected = (r1.inv() * r2).magnitude()
    assert geodesic_distance(q1, q2) == pytest.approx(expected, abs=1e-7)


def test_geodesic_matches_matrix_formula() -> None:
    q1, q2 = _random_quats(200, 3), _random_quats(200, 4)
    d_q = geodesic_distance(q1, q2)
    d_m = matrix_geodesic_distance(
        quat_to_matrix(q1, "scalar_first"), quat_to_matrix(q2, "scalar_first")
    )
    assert d_q == pytest.approx(d_m, abs=1e-6)


def test_sign_invariance_and_symmetry() -> None:
    q1, q2 = _random_quats(100, 5), _random_quats(100, 6)
    assert geodesic_distance(q1, -q2) == pytest.approx(geodesic_distance(q1, q2))
    assert geodesic_distance(q2, q1) == pytest.approx(geodesic_distance(q1, q2))


def test_identity_is_zero_and_never_nan() -> None:
    q = _random_quats(100, 7)
    d = geodesic_distance(q, q)
    assert np.all(np.isfinite(d))
    assert np.all(d >= 0)
    assert np.max(d) < 1e-6  # 2*arccos(1 - eps) floor: ~ sqrt(8 eps)


def test_range_is_zero_to_pi() -> None:
    d = geodesic_distance(_random_quats(1000, 8), _random_quats(1000, 9))
    assert d.min() >= 0 and d.max() <= np.pi


def test_quat_to_matrix_orders_agree_with_scipy() -> None:
    q_wxyz = _random_quats(50, 10)
    q_xyzw = np.roll(q_wxyz, -1, axis=1)
    expected = Rotation.from_quat(q_xyzw).as_matrix()
    assert quat_to_matrix(q_wxyz, "scalar_first") == pytest.approx(expected, abs=1e-12)
    assert quat_to_matrix(q_xyzw, "scalar_last") == pytest.approx(expected, abs=1e-12)
    assert to_scalar_first(q_xyzw, "scalar_last") == pytest.approx(q_wxyz)
    assert set(QUATERNION_ORDERS) == {"scalar_first", "scalar_last"}
    with pytest.raises(ValueError, match="order"):
        quat_to_matrix(q_wxyz, "wxyz")


def test_normalise_quaternions() -> None:
    q = _random_quats(10, 11) * (1 + 1e-7)
    out = normalise_quaternions(q)
    assert np.linalg.norm(out, axis=1) == pytest.approx(np.ones(10), abs=1e-15)
    with pytest.raises(ValueError, match="unit"):
        normalise_quaternions(q * 1.5)
    with pytest.raises(ValueError, match="NaN"):
        normalise_quaternions(np.array([[np.nan, 0, 0, 0]]))
    with pytest.raises(ValueError, match="4"):
        normalise_quaternions(np.zeros((3, 3)))


def test_geodesic_normalises_inputs() -> None:
    """A slightly non-unit label must not manufacture rotation error (P1's 8.7e-7 lesson)."""
    q = _random_quats(20, 12)
    assert geodesic_distance(q, q * (1 - 8.7e-7)).max() < 1e-6


def test_ball_membership() -> None:
    q_c = _random_quats(300, 13)
    q = _random_quats(300, 14)
    d = geodesic_distance(q, q_c)
    radius = np.median(d)
    inside = in_rotation_ball(q, q_c, radius)
    assert inside.tolist() == (d <= radius).tolist()
    assert in_rotation_ball(q, q_c, np.inf).all()
    assert not in_rotation_ball(q, q_c, -np.inf).any()
