"""Scores A1-C2: set inversion is exact, sets take no labels, failures follow the convention."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from poseconf.conformal.metrics import outcomes
from poseconf.conformal.scores import (
    SCORES,
    KeypointSet,
    PoseSet,
    median_normaliser,
    pose_error_components,
    score_a1,
    score_a3,
    score_b1,
    score_b2,
    set_a1,
)
from poseconf.conformal.split import conformal_quantile

CONVENTIONS = ("answer_required", "abstain_allowed")

#: Every argument a set function may take: predictions, the quantile, fitted normalisers.
PREDICTION_SIDE = {
    "q_hat",
    "t_hat",
    "y_hat",
    "valid",
    "q",
    "c_R",
    "c_t",
    "c_z",
    "c_xy",
    "difficulty",
    "d_hat",
    "cov",
    "sigma_R",
    "sigma_t",
    "unconstrained",
}
LABEL_MARKERS = ("gt", "label", "true", "incl", "range", "iou", "pixel_error")

N, K = 400, 11


def _pose_frame(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    q_gt = np.roll(Rotation.random(N, random_state=seed).as_quat(), 1, axis=1)
    noise = Rotation.from_rotvec(rng.normal(scale=0.05, size=(N, 3)))
    q_hat = np.roll((Rotation.from_quat(np.roll(q_gt, -1, axis=1)) * noise).as_quat(), 1, axis=1)
    t_gt = np.column_stack([rng.normal(0, 0.5, N), rng.normal(0, 0.5, N), rng.uniform(3, 30, N)])
    t_hat = t_gt + rng.normal(scale=0.05, size=(N, 3)) * t_gt[:, 2:3] / 10
    valid = rng.uniform(size=N) > 0.1
    q_hat[~valid] = np.nan  # failures carry no estimate
    t_hat[~valid] = np.nan
    return {"q_hat": q_hat, "t_hat": t_hat, "q_gt": q_gt, "t_gt": t_gt, "valid": valid}


def _keypoint_frame(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_gt = rng.uniform(0, 1920, size=(N, K, 2))
    y_hat = y_gt + rng.normal(scale=5, size=(N, K, 2))
    include = rng.uniform(size=(N, K)) > 0.1
    include[0] = False  # a frame with no keypoint in the image
    valid = rng.uniform(size=N) > 0.1
    valid[0] = True
    d_hat = rng.uniform(200, 1500, N)
    a = rng.uniform(1, 30, size=(N, K))
    c = rng.uniform(1, 30, size=(N, K))
    b = rng.uniform(-0.9, 0.9, size=(N, K)) * np.sqrt(a * c)
    cov = np.stack([np.stack([a, b], -1), np.stack([b, c], -1)], -2)
    y_hat[~valid] = np.nan
    return {
        "y_hat": y_hat,
        "y_gt": y_gt,
        "include": include,
        "valid": valid,
        "d_hat": d_hat,
        "cov": cov,
    }


def _score_and_set(score_id: str, convention: str, seed: int):
    """(scores, set builder taking q, contains taking the set) for a registry entry."""
    spec = SCORES[score_id]
    if spec.level == "pose":
        f = _pose_frame(seed)
        rng = np.random.default_rng(seed + 1)
        extra = {
            "A1": {"c_R": 0.04, "c_t": 0.01},
            "A2": {"c_R": 0.04, "c_z": 0.008, "c_xy": 0.004},
            "A3": {"c_R": 0.04, "c_t": 0.01, "difficulty": rng.uniform(0.3, 3, N)},
            "C2": {"sigma_R": rng.uniform(0.01, 0.1, N), "sigma_t": rng.uniform(0.01, 0.5, N)},
        }[score_id]
        s = spec.score_fn(
            f["q_hat"],
            f["t_hat"],
            f["q_gt"],
            f["t_gt"],
            f["valid"],
            **extra,
            convention=convention,
        )
        build = lambda q: spec.set_fn(f["q_hat"], f["t_hat"], f["valid"], q, **extra)  # noqa: E731
        contains = lambda st: st.contains(f["q_gt"], f["t_gt"])  # noqa: E731
    else:
        f = _keypoint_frame(seed)
        extra = {"d_hat": f["d_hat"]} if score_id == "B1" else {"cov": f["cov"]}
        s = spec.score_fn(
            f["y_hat"], f["y_gt"], f["include"], f["valid"], **extra, convention=convention
        )
        build = lambda q: spec.set_fn(f["y_hat"], f["valid"], q, **extra)  # noqa: E731
        contains = lambda st: st.contains(f["y_gt"], f["include"])  # noqa: E731
    return s, build, contains, f["valid"]


def test_registry_is_exactly_the_seven_scores() -> None:
    assert sorted(SCORES) == ["A1", "A2", "A3", "B1", "B2", "C1", "C2"]
    assert {SCORES[k].level for k in ("A1", "A2", "A3", "C2")} == {"pose"}
    assert {SCORES[k].level for k in ("B1", "B2", "C1")} == {"keypoint"}
    assert SCORES["B2"].score_fn is SCORES["C1"].score_fn  # same math, different covariance source
    assert SCORES["B2"].covariance_source != SCORES["C1"].covariance_source


@pytest.mark.parametrize("score_id", ["A1", "A2", "A3", "B1", "B2", "C1", "C2"])
def test_set_functions_take_no_labels(score_id: str) -> None:
    params = inspect.signature(SCORES[score_id].set_fn).parameters
    assert set(params) <= PREDICTION_SIDE, set(params) - PREDICTION_SIDE
    for name in params:
        assert not any(marker in name.lower() for marker in LABEL_MARKERS), name


@pytest.mark.parametrize("score_id", ["A1", "A2", "A3", "B1", "B2", "C1", "C2"])
@pytest.mark.parametrize("convention", CONVENTIONS)
def test_set_inverts_score_exactly(score_id: str, convention: str) -> None:
    """truth in set(pred, q)  <=>  score(pred, label) <= q, for every frame and many q."""
    s, build, contains, valid = _score_and_set(score_id, convention, seed=41)
    finite = s[np.isfinite(s)]
    qs = [*np.quantile(finite, [0.0, 0.1, 0.5, 0.9, 1.0]), *finite[:20], 0.0, np.inf, -np.inf]
    qs.append(conformal_quantile(s, 0.1))
    for q in qs:
        st = build(q)
        inside = contains(st)
        # Answer-required semantics of "truth in set": the score test itself.
        expected = np.where(valid, s <= q, q == np.inf)
        assert inside.tolist() == expected.tolist(), (score_id, convention, q)
        # The set's abstention and the metrics module agree on what was covered.
        covered, answered = outcomes(s, valid, q, convention)
        assert answered.tolist() == (~st.abstain).tolist()
        if convention == "abstain_allowed":
            assert covered.tolist() == (st.abstain | inside).tolist()
        else:
            assert covered.tolist() == inside.tolist()


@pytest.mark.parametrize("score_id", ["A1", "A2", "A3", "B1", "B2", "C1", "C2"])
def test_failures_follow_convention(score_id: str) -> None:
    s_ar, *_, valid = _score_and_set(score_id, "answer_required", seed=42)
    s_aa, *_ = _score_and_set(score_id, "abstain_allowed", seed=42)
    assert np.all(s_ar[~valid] == np.inf)
    assert np.all(s_aa[~valid] == -np.inf)
    assert np.all(np.isfinite(s_ar[valid]))
    assert s_ar[valid].tolist() == s_aa[valid].tolist()


def test_bad_convention_raises() -> None:
    f = _pose_frame(43)
    with pytest.raises(ValueError, match="convention"):
        score_a1(
            f["q_hat"],
            f["t_hat"],
            f["q_gt"],
            f["t_gt"],
            f["valid"],
            c_R=0.1,
            c_t=0.1,
            convention="solved_only",
        )


def test_translation_is_normalised_by_predicted_range() -> None:
    """Same t_hat and same |dt|, different GT range: the A1 score must not change."""
    q = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    t_hat = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
    t_gt = np.array([[0.0, 0.0, 9.0], [0.0, 0.0, 11.0]])  # |t| = 9 vs 11, |dt| = 1 for both
    s = score_a1(
        q, t_hat, q, t_gt, np.ones(2, bool), c_R=1.0, c_t=1.0, convention="answer_required"
    )
    assert s[0] == s[1] == pytest.approx(0.1)


def test_a1_formula() -> None:
    f = _pose_frame(44)
    v = f["valid"]
    comp = pose_error_components(f["q_hat"][v], f["t_hat"][v], f["q_gt"][v], f["t_gt"][v])
    s = score_a1(
        f["q_hat"],
        f["t_hat"],
        f["q_gt"],
        f["t_gt"],
        v,
        c_R=0.04,
        c_t=0.01,
        convention="answer_required",
    )
    expected = np.maximum(comp["rot"] / 0.04, comp["trans_rel"] / 0.01)
    assert s[v] == pytest.approx(expected, rel=1e-12)
    assert comp["trans_rel"] == pytest.approx(
        np.linalg.norm(f["t_hat"][v] - f["t_gt"][v], axis=1) / np.linalg.norm(f["t_hat"][v], axis=1)
    )
    st = set_a1(f["q_hat"], f["t_hat"], v, 2.0, c_R=0.04, c_t=0.01)
    assert isinstance(st, PoseSet)
    assert st.rot_radius[v] == pytest.approx(np.full(v.sum(), 0.08))
    assert st.trans_radius[v, 0] == pytest.approx(0.02 * np.linalg.norm(f["t_hat"][v], axis=1))


def test_a3_is_a1_over_difficulty() -> None:
    f = _pose_frame(45)
    g = np.random.default_rng(0).uniform(0.5, 2.0, N)
    kw = {"c_R": 0.04, "c_t": 0.01, "convention": "answer_required"}
    a1 = score_a1(f["q_hat"], f["t_hat"], f["q_gt"], f["t_gt"], f["valid"], **kw)
    a3 = score_a3(f["q_hat"], f["t_hat"], f["q_gt"], f["t_gt"], f["valid"], difficulty=g, **kw)
    v = f["valid"]
    assert a3[v] == pytest.approx(a1[v] / g[v], rel=1e-12)
    bad = g.copy()
    bad[np.flatnonzero(v)[0]] = 0.0
    with pytest.raises(ValueError, match="difficulty"):
        score_a3(f["q_hat"], f["t_hat"], f["q_gt"], f["t_gt"], v, difficulty=bad, **kw)


def test_keypoint_scores() -> None:
    f = _keypoint_frame(46)
    v, inc = f["valid"], f["include"]
    s = score_b1(f["y_hat"], f["y_gt"], inc, v, d_hat=f["d_hat"], convention="answer_required")
    err = np.linalg.norm(f["y_hat"] - f["y_gt"], axis=-1) / f["d_hat"][:, None]
    expected = np.where(inc, err, -np.inf).max(axis=1)
    rows = v & inc.any(axis=1)
    assert s[rows] == pytest.approx(expected[rows])
    assert s[0] == 0.0  # no included keypoint: vacuously covered for any q >= 0
    st = SCORES["B1"].set_fn(f["y_hat"], v, 0.01, d_hat=f["d_hat"])
    assert isinstance(st, KeypointSet)
    assert st.radius_px[v] == pytest.approx(0.01 * f["d_hat"][v])

    s2 = score_b2(f["y_hat"], f["y_gt"], inc, v, cov=f["cov"], convention="answer_required")
    r = f["y_hat"] - f["y_gt"]
    m = np.sqrt(np.einsum("nki,nkij,nkj->nk", r, np.linalg.inv(f["cov"]), r))
    expected2 = np.where(inc, m, -np.inf).max(axis=1)
    assert s2[rows] == pytest.approx(expected2[rows], rel=1e-9)


def test_non_pd_covariance_raises() -> None:
    f = _keypoint_frame(47)
    cov = f["cov"].copy()
    i = int(np.flatnonzero(f["valid"])[1])
    cov[i, 3] = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="positive-definite"):
        score_b2(
            f["y_hat"], f["y_gt"], f["include"], f["valid"], cov=cov, convention="answer_required"
        )
    with pytest.raises(ValueError, match="positive-definite"):
        SCORES["B2"].set_fn(f["y_hat"], f["valid"], 1.0, cov=cov)


def test_nan_in_valid_prediction_raises() -> None:
    f = _pose_frame(48)
    i = int(np.flatnonzero(f["valid"])[0])
    f["t_hat"][i, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        score_a1(
            f["q_hat"],
            f["t_hat"],
            f["q_gt"],
            f["t_gt"],
            f["valid"],
            c_R=0.1,
            c_t=0.1,
            convention="answer_required",
        )
    with pytest.raises(ValueError, match="NaN"):
        set_a1(f["q_hat"], f["t_hat"], f["valid"], 1.0, c_R=0.1, c_t=0.1)


def test_bad_normaliser_raises() -> None:
    f = _pose_frame(49)
    with pytest.raises(ValueError, match="c_t"):
        set_a1(f["q_hat"], f["t_hat"], f["valid"], 1.0, c_R=0.1, c_t=0.0)
    with pytest.raises(ValueError, match="q"):
        set_a1(f["q_hat"], f["t_hat"], f["valid"], np.nan, c_R=0.1, c_t=0.1)


def test_median_normaliser() -> None:
    errors = np.array([1.0, 2.0, 3.0, np.nan, 100.0])
    valid = np.array([True, True, True, False, False])
    assert median_normaliser(errors, valid) == 2.0
    with pytest.raises(ValueError):
        median_normaliser(errors, np.zeros(5, bool))
    with pytest.raises(ValueError):
        median_normaliser(np.zeros(3), np.ones(3, bool))


def test_contains_alone_is_not_abstain_allowed_coverage() -> None:
    """Pins the documented semantics: a failed frame is never 'in' a finite set, but under
    abstain_allowed it is covered (by abstention) — so coverage must go through `outcomes`."""
    q_id = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    t_hat = np.array([[0.0, 0.0, 10.0], [np.nan, np.nan, np.nan]])
    q_hat = np.array([[1.0, 0.0, 0.0, 0.0], [np.nan] * 4])
    t_gt = np.array([[0.0, 0.0, 10.1], [0.0, 0.0, 10.0]])
    valid = np.array([True, False])
    kw = {"c_R": 1.0, "c_t": 1.0}
    s = score_a1(q_hat, t_hat, q_id, t_gt, valid, **kw, convention="abstain_allowed")
    st = set_a1(q_hat, t_hat, valid, 1.0, **kw)
    assert (s <= 1.0).tolist() == [True, True]
    assert st.contains(q_id, t_gt).tolist() == [True, False]
    covered, _ = outcomes(s, valid, 1.0, "abstain_allowed")
    assert covered.tolist() == [True, True] == (st.abstain | st.contains(q_id, t_gt)).tolist()


def test_unconstrained_keypoints_leave_the_joint_max() -> None:
    """B2's empty-heatmap rule: an unconstrained keypoint's set is the whole image."""
    f = _keypoint_frame(49)
    v, inc = f["valid"], f["include"]
    rng = np.random.default_rng(50)
    unc = rng.uniform(size=inc.shape) < 0.2
    cov = f["cov"].copy()
    cov[unc] = 0.0  # zero covariance (empty channel) is accepted where unconstrained
    y_gt = f["y_gt"].copy()
    y_gt[unc] = np.nan  # never read for an unconstrained keypoint
    for convention in CONVENTIONS:
        s = score_b2(f["y_hat"], y_gt, inc, v, cov=cov, unconstrained=unc, convention=convention)
        ref = score_b2(f["y_hat"], f["y_gt"], inc & ~unc, v, cov=f["cov"], convention=convention)
        np.testing.assert_array_equal(s, ref)
    q = conformal_quantile(s[v], 0.2)
    st = SCORES["B2"].set_fn(f["y_hat"], v, q, cov=cov, unconstrained=unc)
    np.testing.assert_array_equal(st.contains(y_gt, inc)[v], s[v] <= q)
    radius = st.radius_px
    assert np.all(np.isinf(radius[v & unc.any(axis=1)]))
    assert np.all(np.isfinite(radius[v & ~unc.any(axis=1)]))
    # A constrained keypoint still needs a positive-definite covariance.
    bad = cov.copy()
    i = int(np.flatnonzero(v & ~unc[:, 2])[0])
    bad[i, 2] = 0.0
    with pytest.raises(ValueError, match="positive-definite"):
        SCORES["B2"].set_fn(f["y_hat"], v, q, cov=bad, unconstrained=unc)
    with pytest.raises(ValueError, match="unconstrained"):
        SCORES["B2"].set_fn(f["y_hat"], v, q, cov=cov, unconstrained=unc[:, :3])


def test_b1_shares_the_unconstrained_rule() -> None:
    """B1 uses the same empty-channel rule as B2 (user decision before Phase 6)."""
    f = _keypoint_frame(51)
    v, inc = f["valid"], f["include"]
    unc = np.random.default_rng(52).uniform(size=inc.shape) < 0.2
    y_gt = f["y_gt"].copy()
    y_gt[unc] = np.nan
    for convention in CONVENTIONS:
        s = score_b1(
            f["y_hat"], y_gt, inc, v, d_hat=f["d_hat"], unconstrained=unc, convention=convention
        )
        ref = score_b1(
            f["y_hat"], f["y_gt"], inc & ~unc, v, d_hat=f["d_hat"], convention=convention
        )
        np.testing.assert_array_equal(s, ref)
    q = conformal_quantile(s[v], 0.2)
    st = SCORES["B1"].set_fn(f["y_hat"], v, q, d_hat=f["d_hat"], unconstrained=unc)
    np.testing.assert_array_equal(st.contains(y_gt, inc)[v], s[v] <= q)
    assert np.all(np.isinf(st.radius_px[v & unc.any(axis=1)]))
