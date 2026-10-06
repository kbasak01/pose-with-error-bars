"""Split conformal quantile: index, +inf past n, +/-inf scores, NaN refusal.

The brute force here is the definition, enumerated in exact rationals: q is the smallest order
statistic s_(j) with j / (n + 1) >= 1 - alpha; if no j <= n qualifies, q = +inf.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from poseconf.conformal.split import conformal_quantile, quantile_index, rational_alpha

N_VALUES = (1, 2, 9, 10, 99, 4798)
ALPHA_GRID = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


def _brute_force_index(n: int, alpha: float) -> int | None:
    """Smallest 1-indexed j in 1..n with j/(n+1) >= 1 - alpha, exactly; None if none."""
    target = 1 - Fraction(str(alpha))
    for j in range(1, n + 1):
        if Fraction(j, n + 1) >= target:
            return j
    return None


def _brute_force_quantile(scores: np.ndarray, alpha: float) -> float:
    j = _brute_force_index(len(scores), alpha)
    return float("inf") if j is None else float(np.sort(scores)[j - 1])


def _alphas() -> list[float]:
    rng = np.random.default_rng(20261006)
    return [*ALPHA_GRID, *np.round(rng.uniform(0.001, 0.999, size=12), 3).tolist()]


@pytest.mark.parametrize("n", N_VALUES)
def test_quantile_index_matches_brute_force(n: int) -> None:
    for alpha in _alphas():
        j = _brute_force_index(n, alpha)
        expected = n + 1 if j is None else j  # index past n means +inf
        k = quantile_index(n, alpha)
        assert k == expected or (j is None and k > n), (n, alpha, k, j)


@pytest.mark.parametrize("n", N_VALUES)
def test_conformal_quantile_matches_brute_force(n: int) -> None:
    rng = np.random.default_rng(n)
    for alpha in _alphas():
        scores = rng.standard_exponential(n)
        assert conformal_quantile(scores, alpha) == _brute_force_quantile(scores, alpha)


@pytest.mark.parametrize(
    ("n", "alpha", "k"),
    [
        (9, 0.1, 9),  # ceil(10 * 0.9) in floats is ceil(9.000000000000002) = 10
        (99, 0.01, 99),
        (99, 0.05, 95),
        (19, 0.05, 19),
        (4798, 0.1, 4320),
    ],
)
def test_float_traps_are_pinned(n: int, alpha: float, k: int) -> None:
    assert quantile_index(n, alpha) == k


def test_rational_alpha_is_exact_for_decimal_grid() -> None:
    for alpha in ALPHA_GRID:
        assert rational_alpha(alpha) == Fraction(str(alpha))


def test_index_past_n_gives_inf_not_the_max() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    # n = 3, alpha = 0.1: k = ceil(4 * 0.9) = 4 > 3.
    assert quantile_index(3, 0.1) == 4
    assert conformal_quantile(scores, 0.1) == np.inf


def test_empty_calibration_gives_inf() -> None:
    assert conformal_quantile(np.array([]), 0.1) == np.inf


def test_infinite_scores_sort_and_are_not_dropped() -> None:
    # answer_required: failures at +inf. 3 failures out of 19 at alpha = 0.1 -> k = 18 -> +inf.
    finite = np.arange(16, dtype=float)
    scores = np.concatenate([finite, np.full(3, np.inf)])
    assert quantile_index(19, 0.1) == 18
    assert conformal_quantile(scores, 0.1) == np.inf
    # abstain_allowed: the same failures at -inf shift the order statistic down, never drop.
    scores = np.concatenate([finite, np.full(3, -np.inf)])
    assert conformal_quantile(scores, 0.1) == np.sort(scores)[17] == 14.0


def test_all_failure_conventions() -> None:
    assert conformal_quantile(np.full(50, np.inf), 0.1) == np.inf
    assert conformal_quantile(np.full(50, -np.inf), 0.1) == -np.inf


def test_ties_resolve_to_the_tied_value() -> None:
    scores = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    assert conformal_quantile(scores, 0.5) == 2.0


def test_order_of_input_does_not_matter() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=101)
    assert conformal_quantile(scores, 0.1) == conformal_quantile(rng.permutation(scores), 0.1)


def test_nan_raises() -> None:
    with pytest.raises(ValueError, match="NaN"):
        conformal_quantile(np.array([0.1, np.nan, 0.3]), 0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_bad_alpha_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile(np.array([0.1, 0.2]), alpha)


def test_non_1d_scores_raise() -> None:
    with pytest.raises(ValueError, match="1-D"):
        conformal_quantile(np.zeros((3, 2)), 0.1)
