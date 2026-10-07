"""propagate: covariance frames, P1-identical projection, Jacobians, PURSE, extent estimators.

No dataset: a SPEED+-like camera (intrinsics and distortion of SPEED+'s camera.json) and a random
11-point wireframe. The PURSE tests are the Phase 4 equivalence gate on fixtures: membership of the
true pose equals joint keypoint coverage, bit-for-bit, for every q.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from poseconf import p1_adapter
from poseconf.conformal import propagate as pr
from poseconf.conformal.scores import score_b1, score_mahalanobis, set_b1, set_mahalanobis
from poseconf.conformal.so3 import quat_to_matrix
from poseconf.data import dump as dump_module

K = 11
CAMERA = np.array([[2988.58, 0.0, 960.0], [0.0, 2988.34, 600.0], [0.0, 0.0, 1.0]])
DIST = np.array([-0.2238, 0.5141, -0.000665, -0.000214, -0.1312])
GEOM = pr.CameraGeometry(
    wireframe=np.random.default_rng(0).uniform(-0.6, 0.6, size=(K, 3)),
    camera_matrix=CAMERA,
    distortion=DIST,
    width=1920,
    height=1200,
    min_depth=1e-9,
    max_normalised_radius=1.6,
)
CONVENTIONS = ("answer_required", "abstain_allowed")


def _poses(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(quaternions scalar-first, rotation matrices, translations); some near the image edge."""
    rng = np.random.default_rng(seed)
    rot = Rotation.random(n, random_state=seed)
    z = rng.uniform(3.0, 20.0, n)
    edge = rng.uniform(size=n) < 0.3  # push 30 % towards the border so keypoints leave the frame
    lateral = np.where(edge, 0.30, 0.08)[:, None] * z[:, None]
    t = np.column_stack([rng.uniform(-1, 1, (n, 2)) * lateral, z])
    quats = np.roll(rot.as_quat(), 1, axis=1)
    # The label keypoints and PURSE membership must project through the same quaternion ->
    # matrix conversion (as Level B does), or a q placed exactly on a score can flip by an ulp.
    return quats, quat_to_matrix(quats, "scalar_first"), t


def _frames(n: int, seed: int) -> dict[str, np.ndarray]:
    """Predictions around the true projection, B1/B2 inputs, failures and unconstrained channels."""
    rng = np.random.default_rng(seed)
    q_gt, r_gt, t_gt = _poses(n, seed)
    y_gt, include = pr.project(r_gt, t_gt, GEOM)
    sd = rng.uniform(1.0, 8.0, size=(n, K, 1))
    base = np.nan_to_num(y_gt, nan=500.0)
    y_hat = base + rng.standard_t(4, size=(n, K, 2)) * sd
    a = rng.normal(size=(n, K, 2, 2))
    cov = a @ a.transpose(0, 1, 3, 2) * sd[..., None] ** 2 + 0.5 * np.eye(2)
    unconstrained = rng.uniform(size=(n, K)) < 0.05
    cov[unconstrained] = 0.0  # an empty heatmap channel: zero covariance
    return {
        "q_gt": q_gt,
        "r_gt": r_gt,
        "t_gt": t_gt,
        "y_gt": y_gt,
        "include": include,
        "y_hat": y_hat,
        "cov": cov,
        "unconstrained": unconstrained,
        "d_hat": rng.uniform(200.0, 900.0, n),
        "valid": rng.uniform(size=n) > 0.1,
    }


# --------------------------------------------------------------------------------------------------
# covariance frames
# --------------------------------------------------------------------------------------------------


def test_crop_cov_to_full_matches_the_dump_and_preserves_mahalanobis() -> None:
    rng = np.random.default_rng(1)
    n = 50
    lin = rng.normal(size=(n, 2, 2)) + 2.0 * np.eye(2)
    affines = np.concatenate([lin, rng.normal(scale=100.0, size=(n, 2, 1))], axis=2)
    a = rng.normal(size=(n, K, 2, 2))
    cov_crop = a @ a.transpose(0, 1, 3, 2) + 0.1 * np.eye(2)
    mapped = pr.crop_cov_to_full(cov_crop, affines)
    np.testing.assert_allclose(mapped, dump_module.crop_cov_to_full(cov_crop, affines), rtol=1e-10)
    r_crop = rng.normal(size=(n, K, 2))
    r_full = np.einsum("nij,nkj->nki", np.linalg.inv(lin), r_crop)
    m_crop = np.einsum("nki,nkij,nkj->nk", r_crop, np.linalg.inv(cov_crop), r_crop)
    m_full = np.einsum("nki,nkij,nkj->nk", r_full, np.linalg.inv(mapped), r_full)
    np.testing.assert_allclose(m_full, m_crop, rtol=1e-9)


def test_crop_cov_to_full_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="affines"):
        pr.crop_cov_to_full(np.zeros((3, K, 2, 2)), np.zeros((2, 2, 3)))


# --------------------------------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------------------------------


def test_project_is_p1_project_points() -> None:
    _, rot, t = _poses(200, 2)
    t[0] = (0.0, 0.0, -5.0)  # behind the camera
    t[1] = (6.0, 0.0, 1.0)  # beyond the distortion model's monotonic range
    points, visible = pr.project(rot, t, GEOM)
    for i in range(len(t)):
        ref_points, ref_visible = p1_adapter.p1_project(GEOM, rot[i], t[i])
        np.testing.assert_array_equal(visible[i], ref_visible)
        np.testing.assert_array_equal(np.isnan(points[i]), np.isnan(ref_points))
        np.testing.assert_array_equal(points[i], ref_points)  # bit-identical, NaN in place
    assert not visible[0].any() and np.isnan(points[0]).all()
    assert not visible[1].any()
    assert visible.mean() > 0.6 and not visible.all()  # the fixture exercises both sides


def test_unbounded_purse_far_poses_are_members() -> None:
    """Documented property: a pose projecting no keypoint into the frame meets no constraint."""
    f = _frames(5, 22)
    f["valid"][:] = True
    kset, _ = _sets_and_scores(f, "B2", 0.0, "abstain_allowed")
    behind = f["t_gt"] * np.array([1.0, 1.0, -1.0])
    assert pr.PurseSet(kset, GEOM).contains_pose(f["r_gt"], behind).all()


def test_project_matches_cv2_where_modellable() -> None:
    _, rot, t = _poses(100, 3)
    points, _ = pr.project(rot, t, GEOM)
    for i in range(len(t)):
        rvec, _ = cv2.Rodrigues(rot[i])
        ref, _ = cv2.projectPoints(GEOM.wireframe, rvec, t[i], CAMERA, GEOM.dist_coeffs)
        ok = np.isfinite(points[i]).all(axis=1)
        np.testing.assert_allclose(points[i][ok], ref.reshape(-1, 2)[ok], atol=1e-6)


# --------------------------------------------------------------------------------------------------
# Jacobians
# --------------------------------------------------------------------------------------------------


def _exp(w: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(w, dtype=np.float64).reshape(3, 1))[0]


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.3, 1.5, 3.0, math.pi - 1e-6])
def test_left_jacobian_first_order(angle: float) -> None:
    rng = np.random.default_rng(4)
    axis = rng.normal(size=3)
    r = angle * axis / np.linalg.norm(axis)
    d = rng.normal(size=3)
    jl = pr.left_jacobian(r)
    errors = []
    for eps in (1e-3, 1e-4):
        lhs = _exp(r + eps * d)
        rhs = _exp(jl @ (eps * d)) @ _exp(r)
        errors.append(np.abs(lhs - rhs).max())
    assert errors[1] < 2e-7  # second order: shrinks ~100x when eps shrinks 10x
    assert errors[1] < errors[0] / 30 or errors[0] < 1e-12


def _fd_jacobian(rot: np.ndarray, t: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central differences of the projection w.r.t. (omega, t), R' = Exp(omega) R."""
    cols = []
    for i in range(6):
        e = np.zeros(6)
        e[i] = h
        plus = (_exp(e[:3]) @ rot)[None], (t + e[3:])[None]
        minus = (_exp(-e[:3]) @ rot)[None], (t - e[3:])[None]
        p, _ = pr.project(*plus, GEOM)
        m, _ = pr.project(*minus, GEOM)
        cols.append((p[0] - m[0]) / (2 * h))
    return np.stack(cols, axis=-1)


def test_projection_jacobian_matches_finite_differences() -> None:
    _, rot, t = _poses(30, 5)
    for i in range(len(t)):
        jac = pr.projection_jacobian(rot[i], t[i], GEOM)
        fd = _fd_jacobian(rot[i], t[i])
        ok = np.isfinite(fd).all(axis=(1, 2))
        scale = np.abs(jac[ok]).max()
        assert np.abs(jac[ok] - fd[ok]).max() <= 1e-6 * scale


def test_cv2_rvec_columns_match_finite_differences() -> None:
    """The raw OpenCV Jacobian (before the J_l^-1 map) is d/d(Rodrigues vector)."""
    _, rot, t = _poses(10, 6)
    h = 1e-7
    for i in range(len(t)):
        rvec, _ = cv2.Rodrigues(rot[i])
        _, jac = cv2.projectPoints(GEOM.wireframe, rvec, t[i], CAMERA, GEOM.dist_coeffs)
        jac = jac.reshape(K, 2, -1)[..., :3]
        for c in range(3):
            e = np.zeros((3, 1))
            e[c] = h
            p, _ = cv2.projectPoints(GEOM.wireframe, rvec + e, t[i], CAMERA, GEOM.dist_coeffs)
            m, _ = cv2.projectPoints(GEOM.wireframe, rvec - e, t[i], CAMERA, GEOM.dist_coeffs)
            fd = (p - m).reshape(K, 2) / (2 * h)
            assert np.abs(jac[..., c] - fd).max() <= 1e-5 * np.abs(jac).max()


# --------------------------------------------------------------------------------------------------
# PURSE membership == joint keypoint coverage
# --------------------------------------------------------------------------------------------------


def _sets_and_scores(f: dict, score_id: str, q: float, convention: str):
    if score_id == "B1":
        s = score_b1(
            f["y_hat"], f["y_gt"], f["include"], f["valid"], d_hat=f["d_hat"], convention=convention
        )
        return set_b1(f["y_hat"], f["valid"], q, d_hat=f["d_hat"]), s
    s = score_mahalanobis(
        f["y_hat"],
        f["y_gt"],
        f["include"],
        f["valid"],
        cov=f["cov"],
        unconstrained=f["unconstrained"],
        convention=convention,
    )
    kset = set_mahalanobis(
        f["y_hat"], f["valid"], q, cov=f["cov"], unconstrained=f["unconstrained"]
    )
    return kset, s


@pytest.mark.parametrize("score_id", ["B1", "B2"])
@pytest.mark.parametrize("convention", CONVENTIONS)
def test_purse_membership_is_joint_keypoint_coverage(score_id: str, convention: str) -> None:
    f = _frames(400, 7)
    assert (~f["include"]).any() and f["include"].any(axis=1).mean() > 0.9
    _, scores = _sets_and_scores(f, score_id, 0.0, convention)
    finite = np.unique(scores[np.isfinite(scores)])
    # Every boundary (q equal to a frame's own score), between scores, and the extremes.
    qs = [*finite[:: max(1, finite.size // 60)], *(finite[:-1:7] + np.diff(finite)[::7] / 2)]
    qs += [0.0, math.inf, -math.inf]
    for q in qs:
        kset, s = _sets_and_scores(f, score_id, float(q), convention)
        member = pr.PurseSet(kset, GEOM).contains(f["q_gt"], f["t_gt"], "scalar_first")
        ok = f["valid"]
        np.testing.assert_array_equal(member[ok], s[ok] <= q)
        np.testing.assert_array_equal(member[~ok], np.full((~ok).sum(), q == math.inf))
        np.testing.assert_array_equal(member, kset.contains(f["y_gt"], f["include"]))


def test_purse_uses_visibility_under_the_candidate() -> None:
    """A keypoint the candidate projects out of frame is unconstrained for that candidate."""
    q_gt, r_gt, t_gt = _poses(1, 8)
    y, vis = pr.project(r_gt, t_gt, GEOM)
    assert vis.all()
    y_hat = y.copy()
    y_hat[0, 0] += 5000.0  # keypoint 0's set is far from where the true pose puts it
    kset = set_b1(y_hat, np.array([True]), 0.01, d_hat=np.array([500.0]))
    purse = pr.PurseSet(kset, GEOM)
    assert not purse.contains_pose(r_gt, t_gt)[0]
    # Move the camera so keypoint 0 leaves the frame while the rest stay put: now a member is
    # possible only through the visibility rule. Emulate by marking keypoint 0 unconstrained.
    kset_unc = set_mahalanobis(
        y_hat,
        np.array([True]),
        0.01,
        cov=np.broadcast_to(np.eye(2) * 1e4, (1, K, 2, 2)),
        unconstrained=np.eye(1, K, 0, dtype=bool),
    )
    assert pr.PurseSet(kset_unc, GEOM).contains_pose(r_gt, t_gt)[0]


def test_purse_rejects_nan_candidate_on_valid_frame() -> None:
    f = _frames(5, 9)
    kset, _ = _sets_and_scores(f, "B1", 0.1, "answer_required")
    rot = f["r_gt"].copy()
    rot[np.flatnonzero(f["valid"])[0]] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pr.PurseSet(kset, GEOM).contains_pose(rot, f["t_gt"])


# --------------------------------------------------------------------------------------------------
# linearised extent
# --------------------------------------------------------------------------------------------------


def _exact_frame(seed: int, sd: float):
    """One frame whose predictions are exact projections of the estimate (no PnP residual)."""
    rng = np.random.default_rng(seed)
    _, rot, t = _poses(1, seed)
    t[0, :2] = 0.0
    y, vis = pr.project(rot, t, GEOM)
    a = rng.normal(size=(1, K, 2, 2))
    cov = (a @ a.transpose(0, 1, 3, 2) + np.eye(2)) * sd**2
    return rot[0], t[0], y, vis[0], cov


def _linearised(rot, t, y_hat, vis, cov):
    jac = pr.projection_jacobian(rot, t, GEOM)
    points, _ = pr.project(rot[None], t[None], GEOM)
    return jac, pr.linearised_extent(jac, cov[0], vis, points[0] - y_hat[0])


def test_linearised_zero_residual_scales_with_q() -> None:
    rot, t, y, vis, cov = _exact_frame(10, 2.0)
    jac, ext = _linearised(rot, t, y, vis, cov)
    assert ext.n_used == int(vis.sum()) and math.isfinite(ext.rot_var)
    assert ext.c == pytest.approx(0.0, abs=1e-12) and ext.rot_offset < 1e-9
    one, two = pr.scale_linearised(ext, 1.5), pr.scale_linearised(ext, 3.0)
    for key in one:
        assert two[key] == pytest.approx(2.0 * one[key], rel=1e-6)
    assert one["rot_outer"] == pytest.approx(math.sqrt(ext.n_used) * one["rot_inner"], rel=1e-6)
    assert one["trans_outer"] >= one["trans_inner"]
    assert all(math.isnan(v) for v in pr.scale_linearised(ext, -math.inf).values())
    assert all(math.isinf(v) for v in pr.scale_linearised(ext, math.inf).values())
    few = np.zeros(K, dtype=bool)
    few[:2] = True
    singular = pr.linearised_extent(jac, cov[0], few, np.zeros((K, 2)))
    assert math.isinf(singular.rot_var)
    assert all(math.isinf(v) for v in pr.scale_linearised(singular, 0.0).values())


def test_linearised_residual_decomposition() -> None:
    """sum_k Mahalanobis^2(r_k + J_k d) = (d - d*)^T H (d - d*) + c, and the radii follow it."""
    rot, t, y, vis, cov = _exact_frame(19, 2.0)
    rng = np.random.default_rng(20)
    y_hat = y + rng.normal(scale=3.0, size=y.shape)  # PnP does not reproduce the predictions
    jac, ext = _linearised(rot, t, y_hat, vis, cov)
    points, _ = pr.project(rot[None], t[None], GEOM)
    r = (points[0] - y_hat[0])[vis]
    j = jac[vis]
    inv = np.linalg.inv(cov[0][vis])
    info = np.einsum("kai,kab,kbj->ij", j, inv, j)
    star = -np.linalg.solve(info, np.einsum("kai,kab,kb->i", j, inv, r))
    assert np.linalg.norm(star[:3]) == pytest.approx(ext.rot_offset, rel=1e-9)
    for d in rng.normal(scale=0.01, size=(5, 6)):
        e = r + np.einsum("kai,i->ka", j, d)
        lhs = np.einsum("ka,kab,kb->", e, inv, e)
        assert lhs == pytest.approx((d - star) @ info @ (d - star) + ext.c, rel=1e-9)
    per_kp = np.einsum("ka,kab,kb->k", r, inv, r)
    assert ext.residual_max == pytest.approx(np.sqrt(per_kp.max()))
    q = 2.0 * math.sqrt(ext.c)
    radii = pr.scale_linearised(ext, q)
    lam = ext.rot_var
    assert radii["rot_inner"] == pytest.approx(math.sqrt(ext.rot_offset**2 + (q * q - ext.c) * lam))
    assert radii["rot_outer"] == pytest.approx(
        ext.rot_offset + math.sqrt((ext.n_used * q * q - ext.c) * lam)
    )
    small = pr.scale_linearised(ext, 0.9 * math.sqrt(ext.c))  # inner ellipsoid empty
    assert math.isnan(small["rot_inner"]) and math.isfinite(small["rot_outer"])
    empty = pr.scale_linearised(ext, 0.9 * math.sqrt(ext.c / ext.n_used))
    assert all(math.isnan(v) for v in empty.values())


@pytest.mark.parametrize("noise", [0.0, 0.15])
def test_linearised_brackets_the_small_set_purse(noise: float) -> None:
    """For a small set, the true PURSE near the estimate lies between inner and outer."""
    rot, t, y, vis, cov = _exact_frame(11, 0.5)
    rng = np.random.default_rng(21)
    y_hat = y + noise * rng.normal(size=y.shape)
    q = 1.0
    kset = set_mahalanobis(y_hat, np.array([True]), q, cov=cov)
    jac, ext = _linearised(rot, t, y_hat, vis, cov)
    radii = pr.scale_linearised(ext, q)
    assert ext.c < q * q  # the inner ellipsoid exists for this fixture

    def members(deltas: np.ndarray) -> np.ndarray:
        n = len(deltas)
        rots = np.stack([_exp(d[:3]) @ rot for d in deltas])
        ks = pr._single_frame(kset, 0, n)
        return pr.PurseSet(ks, GEOM).contains_pose(rots, t + deltas[:, 3:])

    points, _ = pr.project(rot[None], t[None], GEOM)
    r = (points[0] - y_hat[0])[vis]
    j = jac[vis]
    inv = np.linalg.inv(cov[0][vis])
    info = np.einsum("kai,kab,kbj->ij", j, inv, j)
    star = -np.linalg.solve(info, np.einsum("kai,kab,kb->i", j, inv, r))
    # (i) The inner ellipsoid's point that attains the inner radius (pulled 2 % towards its
    # centre) is a PURSE member.
    w, v = np.linalg.eigh(info)
    half = (v / np.sqrt(w)) @ v.T  # H^-1/2
    u, _, _ = np.linalg.svd(half[:3, :])
    direction = half @ (half[:3, :].T @ u[:, 0])
    direction /= np.sqrt(direction @ info @ direction)
    if star[:3] @ direction[:3] < 0:
        direction = -direction
    extreme = star + math.sqrt(q * q - ext.c) * direction
    assert np.linalg.norm(extreme[:3]) >= radii["rot_inner"] * (1 - 1e-9)
    assert members((star + 0.98 * (extreme - star))[None])[0]
    # (ii) No PURSE point on rays from the linearised centre goes beyond the outer radius.
    assert members(star[None])[0]
    dirs = rng.normal(size=(1500, 6))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs[:, 3:] *= radii["trans_outer"] / radii["rot_outer"]  # comparable units on both blocks
    lo, hi = np.zeros(len(dirs)), np.full(len(dirs), 10.0 * radii["rot_outer"])
    assert not members(star + hi[:, None] * dirs).any()
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        inside = members(star + mid[:, None] * dirs)
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    reach = star + lo[:, None] * dirs
    assert np.linalg.norm(reach[:, :3], axis=1).max() <= 1.02 * radii["rot_outer"]
    assert np.linalg.norm(reach[:, 3:], axis=1).max() <= 1.02 * radii["trans_outer"]


# --------------------------------------------------------------------------------------------------
# sampled extent
# --------------------------------------------------------------------------------------------------


def test_sampled_keeps_only_purse_members_and_is_seeded(monkeypatch: pytest.MonkeyPatch) -> None:
    rot, t, y, vis, cov = _exact_frame(13, 1.0)
    kset = set_mahalanobis(y, np.array([True]), 2.0, cov=cov)
    seen: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    original = pr.PurseSet.contains_pose

    def spy(self, rotation, translation):
        out = original(self, rotation, translation)
        seen.append((np.asarray(rotation), np.asarray(translation), out))
        return out

    monkeypatch.setattr(pr.PurseSet, "contains_pose", spy)
    ext = pr.sampled_extent(kset, 0, rot, t, GEOM, samples=128, rng=np.random.default_rng(14))
    assert 0 < ext.n_accepted <= ext.n_solved <= ext.n_samples == 128
    rots, trans, accepted = seen[-1]
    assert accepted.sum() == ext.n_accepted
    angles = [np.linalg.norm(cv2.Rodrigues(r @ rot.T)[0]) for r in rots[accepted]]
    assert ext.rot == pytest.approx(max(angles), rel=1e-12)
    assert ext.trans == pytest.approx(np.linalg.norm(trans[accepted] - t, axis=1).max())
    monkeypatch.undo()
    again = pr.sampled_extent(kset, 0, rot, t, GEOM, samples=128, rng=np.random.default_rng(14))
    assert (again.rot, again.trans, again.n_accepted) == (ext.rot, ext.trans, ext.n_accepted)


def test_sampled_is_an_inner_approximation_of_a_small_set() -> None:
    """Every accepted pose is a PURSE member, so for a small set it sits inside the outer radius."""
    rot, t, y, vis, cov = _exact_frame(15, 0.5)
    q = 1.0
    kset = set_mahalanobis(y, np.array([True]), q, cov=cov)
    _, lin = _linearised(rot, t, y, vis, cov)
    radii = pr.scale_linearised(lin, q)
    ext = pr.sampled_extent(kset, 0, rot, t, GEOM, samples=256, rng=np.random.default_rng(16))
    assert ext.n_accepted > 0
    assert 0.0 < ext.rot <= 1.02 * radii["rot_outer"]
    assert 0.0 < ext.trans <= 1.02 * radii["trans_outer"]


def test_sampled_needs_four_keypoints_and_a_finite_q() -> None:
    rot, t, y, vis, cov = _exact_frame(17, 1.0)
    unc = np.ones((1, K), dtype=bool)
    unc[0, :3] = False
    kset = set_mahalanobis(y, np.array([True]), 1.0, cov=cov, unconstrained=unc)
    ext = pr.sampled_extent(kset, 0, rot, t, GEOM, samples=8, rng=np.random.default_rng(0))
    assert math.isnan(ext.rot) and ext.n_accepted == 0
    with pytest.raises(ValueError, match="finite q"):
        pr.sampled_extent(
            set_mahalanobis(y, np.array([True]), math.inf, cov=cov),
            0,
            rot,
            t,
            GEOM,
            samples=8,
            rng=np.random.default_rng(0),
        )


def test_unit_shapes() -> None:
    f = _frames(6, 18)
    disc = set_b1(f["y_hat"], f["valid"], 1.0, d_hat=f["d_hat"])
    shapes = pr.unit_shapes(disc)
    ok = f["valid"]
    np.testing.assert_allclose(shapes[ok, 0], f["d_hat"][ok, None, None] ** 2 * np.eye(2))
