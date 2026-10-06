"""Weighted split CP (Tibshirani et al. 2019): definition, reduction to split CP, shift validity."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from poseconf.conformal.split import conformal_quantile
from poseconf.conformal.weighted import (
    clip_weights,
    effective_sample_size,
    max_normalised_weight,
    weighted_conformal_quantile,
    weighted_conformal_quantiles,
    weights_from_probabilities,
)
from poseconf.synthetic_validity import case_rng, covariate_shift_draws

N_RESAMPLES = 2000


def _brute_force_weighted(scores: list[float], weights: list[int], w_test: int, alpha: float):
    """Smallest v in scores U {+inf} with sum_{s_i <= v} p_i (+ p_test if v = inf) >= 1 - alpha."""
    total = sum(weights) + w_test
    target = 1 - Fraction(str(alpha))
    for v in sorted(set(scores)):
        mass = Fraction(sum(w for s, w in zip(scores, weights, strict=True) if s <= v), total)
        if mass >= target:
            return v
    return float("inf")


def test_matches_brute_force_with_integer_weights() -> None:
    rng = np.random.default_rng(11)
    for _ in range(300):
        n = int(rng.integers(1, 15))
        scores = rng.integers(0, 6, n).astype(float)  # ties on purpose
        weights = rng.integers(1, 5, n)
        w_test = int(rng.integers(1, 5))
        alpha = float(rng.choice([0.05, 0.1, 0.2, 0.3, 0.5]))
        expected = _brute_force_weighted(scores.tolist(), weights.tolist(), w_test, alpha)
        got = weighted_conformal_quantile(scores, weights.astype(float), float(w_test), alpha)
        assert got == expected, (scores, weights, w_test, alpha)


@pytest.mark.parametrize("n", [1, 2, 9, 10, 99, 4798])
@pytest.mark.parametrize("alpha", [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
def test_equal_weights_reduce_to_split(n: int, alpha: float) -> None:
    rng = np.random.default_rng(n)
    scores = rng.standard_exponential(n)
    for c in (1.0, 0.37, 1e6):
        q = weighted_conformal_quantiles(scores, np.full(n, c), np.array([c, c]), alpha)
        assert q.tolist() == [conformal_quantile(scores, alpha)] * 2


def test_test_point_mass_sits_at_inf() -> None:
    scores = np.arange(10.0)
    # A test weight heavier than alpha / (1 - alpha) of the total pushes q to +inf.
    assert weighted_conformal_quantile(scores, np.ones(10), 100.0, 0.1) == np.inf
    # Dropping the test mass from the normaliser would give a finite answer here.
    assert weighted_conformal_quantile(scores, np.ones(10), 1e-9, 0.1) == 9.0


def test_vectorised_matches_scalar() -> None:
    rng = np.random.default_rng(12)
    scores = rng.normal(size=200)
    w = rng.uniform(0.1, 3, size=200)
    w_test = rng.uniform(0.1, 30, size=50)
    vec = weighted_conformal_quantiles(scores, w, w_test, 0.1)
    assert vec.tolist() == [weighted_conformal_quantile(scores, w, t, 0.1) for t in w_test]


def test_infinite_scores_are_kept() -> None:
    scores = np.array([0.1, 0.2, np.inf, -np.inf, 0.3])
    assert weighted_conformal_quantile(scores, np.ones(5), 1.0, 0.5) == 0.2


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="NaN"):
        weighted_conformal_quantile(np.array([np.nan]), np.ones(1), 1.0, 0.1)
    with pytest.raises(ValueError, match="weight"):
        weighted_conformal_quantile(np.array([1.0]), np.array([-1.0]), 1.0, 0.1)
    with pytest.raises(ValueError, match="weight"):
        weighted_conformal_quantile(np.array([1.0]), np.array([np.inf]), 1.0, 0.1)
    with pytest.raises(ValueError, match="weight"):
        weighted_conformal_quantile(np.array([1.0]), np.ones(1), -1.0, 0.1)
    with pytest.raises(ValueError, match="length"):
        weighted_conformal_quantile(np.array([1.0, 2.0]), np.ones(1), 1.0, 0.1)
    with pytest.raises(ValueError, match="alpha"):
        weighted_conformal_quantile(np.array([1.0]), np.ones(1), 1.0, 1.0)


def test_ess_and_max_weight() -> None:
    assert effective_sample_size(np.ones(50)) == pytest.approx(50.0)
    assert effective_sample_size(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert max_normalised_weight(np.array([1.0, 1.0, 2.0])) == pytest.approx(0.5)


def test_weights_from_probabilities() -> None:
    p = np.array([0.5, 0.2, 0.8])
    w = weights_from_probabilities(p, n_source=200, n_target=100)
    assert w == pytest.approx(p / (1 - p) * 2.0)
    with pytest.raises(ValueError, match="probabilit"):
        weights_from_probabilities(np.array([0.0, 0.5]), 10, 10)
    with pytest.raises(ValueError, match="probabilit"):
        weights_from_probabilities(np.array([1.0]), 10, 10)


def test_clip_weights() -> None:
    w = np.array([0.5, 2.0, 10.0])
    assert clip_weights(w, 3.0).tolist() == [0.5, 2.0, 3.0]
    with pytest.raises(ValueError, match="clip"):
        clip_weights(w, 0.0)


@pytest.mark.parametrize("alpha", [0.10, 0.20])
def test_weighted_cp_restores_coverage_under_known_shift(alpha: float) -> None:
    n, n_test = 500, 200
    weighted = covariate_shift_draws(
        n,
        alpha,
        N_RESAMPLES,
        n_test,
        case_rng(21, round(alpha * 100)),
        weighted=True,
    )
    plain = covariate_shift_draws(
        n,
        alpha,
        N_RESAMPLES,
        n_test,
        case_rng(22, round(alpha * 100)),
        weighted=False,
    )
    cov_w, cov_p = weighted["coverage"], plain["coverage"]
    se_w = np.std(cov_w, ddof=1) / np.sqrt(N_RESAMPLES)
    se_p = np.std(cov_p, ddof=1) / np.sqrt(N_RESAMPLES)
    # The shift matters: plain split CP under-covers by many standard errors.
    assert cov_p.mean() < 1 - alpha - 10 * se_p, cov_p.mean()
    # Weighted CP with the true weights is valid.
    assert cov_w.mean() >= 1 - alpha - 3 * se_w, (cov_w.mean(), se_w)
    # ESS is reported and below n (the weights are not uniform).
    assert 1.0 < weighted["ess"].mean() < n
