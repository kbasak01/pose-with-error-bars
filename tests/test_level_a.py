"""Level A engine: joins, val_tune fitting, label-free sets, vectorised re-splits, verdicts."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from poseconf.conformal.metrics import outcomes
from poseconf.conformal.split import CONVENTIONS, conformal_quantile
from poseconf.engine.level_a import (
    Frames,
    check_sidecar_errors,
    fit_level_a,
    join_frames,
    pose_set,
    resplit_draws,
    score_frames,
    set_size_report,
    summarise_draws,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.05, 0.1, 0.3)


def _random_quats(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _perturb(rng: np.random.Generator, q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Right-multiply by a small random rotation (scalar-first)."""
    axis = rng.normal(size=(len(q), 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    half = 0.5 * np.abs(rng.normal(size=len(q))) * scale
    d = np.concatenate([np.cos(half)[:, None], np.sin(half)[:, None] * axis], axis=1)
    w1, v1 = q[:, :1], q[:, 1:]
    w2, v2 = d[:, :1], d[:, 1:]
    return np.concatenate(
        [w1 * w2 - np.sum(v1 * v2, axis=1, keepdims=True), w1 * v2 + w2 * v1 + np.cross(v1, v2)],
        axis=1,
    )


def make_frames(n: int, seed: int, fail_rate: float = 0.07) -> Frames:
    """Synthetic frames: error scale grows as confidence falls (so g has something to fit)."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.3, 1.0, size=n)
    scale = 0.02 / conf
    q_gt = _random_quats(rng, n)
    t_gt = np.column_stack([rng.normal(0, 0.5, n), rng.normal(0, 0.5, n), rng.uniform(4, 12, n)])
    q_hat = _perturb(rng, q_gt, scale)
    t_hat = t_gt + rng.normal(size=(n, 3)) * scale[:, None] * t_gt[:, 2:3] * 0.2
    valid = rng.uniform(size=n) >= fail_rate
    q_hat[~valid], t_hat[~valid] = np.nan, np.nan
    e_r = np.full(n, np.nan)
    e_t = np.full(n, np.nan)
    from poseconf.conformal.so3 import geodesic_distance

    e_r[valid] = geodesic_distance(q_hat[valid], q_gt[valid])
    e_t[valid] = np.linalg.norm(t_hat[valid] - t_gt[valid], axis=1)
    return Frames(
        filenames=np.array([f"img{i:06d}.jpg" for i in range(n)]),
        q_hat=q_hat,
        t_hat=t_hat,
        valid=valid,
        conf=conf,
        q_gt=q_gt,
        t_gt=t_gt,
        e_r=e_r,
        e_t=e_t,
    )


def _sidecar(frames: Frames) -> dict[str, np.ndarray]:
    return {
        "filename": frames.filenames,
        "success": frames.valid,
        "q_pred": frames.q_hat.astype(np.float32),
        "t_pred": frames.t_hat.astype(np.float32),
        "confidence_mean": frames.conf.astype(np.float32),
        "e_r": frames.e_r.astype(np.float32),
        "e_t": frames.e_t.astype(np.float32),
    }


# --- join and join check ------------------------------------------------------------------------


def test_join_selects_by_name_and_checks_errors() -> None:
    frames = make_frames(200, seed=1)
    names = list(frames.filenames[::-1][:50])
    joined = join_frames(_sidecar(frames), frames.filenames, frames.q_gt, frames.t_gt, names)
    assert joined.filenames.tolist() == names
    report = check_sidecar_errors(joined, tol_rad=1e-4, tol_m=1e-4)
    assert report["n_checked"] == int(joined.valid.sum())


def test_join_rejects_missing_and_duplicates() -> None:
    frames = make_frames(20, seed=2)
    side = _sidecar(frames)
    with pytest.raises(ValueError, match="missing"):
        join_frames(side, frames.filenames, frames.q_gt, frames.t_gt, ["nope.jpg"])
    with pytest.raises(ValueError, match="unique"):
        join_frames(side, frames.filenames, frames.q_gt, frames.t_gt, ["img000001.jpg"] * 2)


def test_misaligned_labels_fail_the_join_check() -> None:
    frames = make_frames(100, seed=3)
    shuffled = np.roll(np.arange(100), 1)
    joined = join_frames(
        _sidecar(frames),
        frames.filenames,
        frames.q_gt[shuffled],
        frames.t_gt[shuffled],
        list(frames.filenames),
    )
    with pytest.raises(ValueError, match="not reproduced"):
        check_sidecar_errors(joined, tol_rad=1e-4, tol_m=1e-4)


# --- fitting and sets ---------------------------------------------------------------------------


def test_fit_is_positive_and_g_decreasing() -> None:
    fit = fit_level_a(make_frames(2000, seed=4))
    assert min(fit.c_R, fit.c_t, fit.c_z, fit.c_xy) > 0
    assert np.all(np.diff(fit.g_x) > 0) and np.all(np.diff(fit.g_y) <= 0)
    assert np.all(fit.g_y > 0)
    # Low confidence -> larger difficulty; clipped outside the fitted range.
    assert fit.difficulty(np.array([0.0]))[0] == fit.g_y[0]
    assert fit.difficulty(np.array([2.0]))[0] == fit.g_y[-1]


@pytest.mark.parametrize("score_id", ["A1", "A2", "A3"])
def test_sets_never_see_labels(score_id: str) -> None:
    frames = make_frames(300, seed=5)
    fit = fit_level_a(make_frames(1000, seed=6))
    blind = Frames(
        **{
            **{k: getattr(frames, k) for k in frames.__dataclass_fields__},
            "q_gt": np.full_like(frames.q_gt, np.nan),
            "t_gt": np.full_like(frames.t_gt, np.nan),
            "e_r": np.full_like(frames.e_r, np.nan),
            "e_t": np.full_like(frames.e_t, np.nan),
        }
    )
    a, b = pose_set(score_id, frames, fit, 1.7), pose_set(score_id, blind, fit, 1.7)
    np.testing.assert_array_equal(a.rot_radius, b.rot_radius)
    np.testing.assert_array_equal(a.trans_radius, b.trans_radius)


@pytest.mark.parametrize("score_id", ["A1", "A2", "A3"])
def test_set_contains_iff_score_le_q(score_id: str) -> None:
    frames = make_frames(500, seed=7)
    fit = fit_level_a(make_frames(1000, seed=8))
    s = score_frames(score_id, frames, fit, "answer_required")
    q = float(np.median(s[frames.valid]))
    inside = pose_set(score_id, frames, fit, q).contains(frames.q_gt, frames.t_gt)
    np.testing.assert_array_equal(inside, s <= q)


def test_set_size_report_keeps_inf_and_handles_no_answer() -> None:
    frames = make_frames(100, seed=9)
    fit = fit_level_a(make_frames(500, seed=10))
    report = set_size_report(pose_set("A2", frames, fit, math.inf), frames.valid)
    assert report["rot_deg"]["median"] == math.inf
    assert set(report) == {"rot_deg", "boresight_frac", "boresight_m", "lateral_frac", "lateral_m"}
    assert set_size_report(pose_set("A1", frames, fit, -math.inf), np.zeros(100, bool)) is None


# --- vectorised re-splits -----------------------------------------------------------------------


@pytest.mark.parametrize("convention", list(CONVENTIONS))
def test_resplit_draws_match_reference_implementation(convention: str) -> None:
    frames = make_frames(400, seed=11, fail_rate=0.08)
    fit = fit_level_a(make_frames(800, seed=12))
    s = score_frames("A3", frames, fit, convention)
    rot_scale = pose_set("A3", frames, fit, 1.0).rot_scale
    rng = np.random.default_rng(13)
    perms = np.stack([rng.permutation(400) for _ in range(25)])
    alphas = (0.01, 0.05, 0.1, 0.5)
    draws = resplit_draws(s, frames.valid, rot_scale, perms, 200, alphas, convention)
    for r, perm in enumerate(perms):
        cal, test = perm[:200], perm[200:]
        for i, alpha in enumerate(alphas):
            q = conformal_quantile(s[cal], alpha)
            covered, answered = outcomes(s[test], frames.valid[test], q, convention)
            assert draws.q[i, r] == q
            assert draws.n_covered[i, r] == covered.sum()
            assert draws.answer_rate[i, r] == pytest.approx(answered.mean())
            assert draws.silent_failure_rate[i, r] == pytest.approx((answered & ~covered).mean())


def test_resplit_draws_reject_mixed_conventions() -> None:
    frames = make_frames(50, seed=14)
    fit = fit_level_a(make_frames(300, seed=15))
    s = score_frames("A1", frames, fit, "answer_required")
    perms = np.arange(50)[None, :]
    with pytest.raises(ValueError, match="inconsistent"):
        resplit_draws(s, frames.valid, np.ones(50), perms, 25, (0.1,), "abstain_allowed")


@pytest.mark.slow
@pytest.mark.parametrize("convention", list(CONVENTIONS))
def test_summary_valid_on_exchangeable_pool_and_degenerate_on_atom(convention: str) -> None:
    rng = np.random.default_rng(16)
    n_pool, fail = 1000, 120
    valid = np.ones(n_pool, bool)
    valid[:fail] = False
    s = rng.exponential(size=n_pool)
    s[~valid] = CONVENTIONS[convention]
    perms = np.stack([rng.permutation(n_pool) for _ in range(1000)])
    draws = resplit_draws(s, valid, np.ones(n_pool), perms, 500, (0.05, 0.3), convention)
    atoms = {
        "n_top_atom": fail if convention == "answer_required" else 0,
        "n_bottom_atom": fail if convention == "abstain_allowed" else 0,
    }
    kw = {"n_finite_ties": 0, "ks_min_p": 0.01, "mean_band_se": 3.0, **atoms}
    high = summarise_draws(draws, 1, **kw)
    assert high["verdict"] == "VALID"
    low = summarise_draws(draws, 0, **kw)
    expected = "DEGENERATE" if convention == "answer_required" else "VALID"
    assert low["verdict"] == expected


# --- end to end on the real sidecar (local only) ------------------------------------------------


@pytest.mark.dataset
def test_run_level_a_end_to_end_reads_synthetic_only(
    local_paths, tmp_path: Path, monkeypatch
) -> None:
    import json

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_level_a

    seen: list[str] = []
    original = run_level_a.load_sidecar

    def spy(run: str, domain: str, crop_source: str):
        seen.append(domain)
        return original(run, domain, crop_source)

    monkeypatch.setattr(run_level_a, "load_sidecar", spy)
    code = run_level_a.main(["--out-dir", str(tmp_path)])
    assert seen == ["synthetic"]
    result = json.loads((tmp_path / "keypoint_a2_predicted_crop_synthetic.json").read_text())
    assert len(result["rows"]) == 3 * 2 * 8
    assert all(row["domain"] == "synthetic" and not row["oracle"] for row in result["rows"])
    assert code == 0
