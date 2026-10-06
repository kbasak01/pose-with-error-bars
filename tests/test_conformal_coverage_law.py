"""Split CP reproduces the Beta coverage law on synthetic scores with known distributions.

For continuous exchangeable scores, coverage given the calibration set is Beta(n + 1 - l, l),
l = floor((n + 1) alpha). Each case draws R = 2,000 independent calibration sets and computes the
exact conditional coverage of each; the draws must match the law: KS p > 0.01, the mean inside the
law's 99 % band for a mean of R draws, and ~99 % of single draws inside the per-draw 99 % band.

Seeds are fixed in `_rng` and were set before the first run. A failing case is reported, never
reseeded (CLAUDE.md invariant 11).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from poseconf.synthetic_validity import (
    DISTRIBUTIONS,
    case_rng,
    split_case_key,
    split_coverage_draws,
    summarise_split_draws,
)

N_RESAMPLES = 2000
N_CAL = (100, 1000)
ALPHAS = (0.05, 0.10)

#: Lower bound on the fraction of single draws inside the per-draw 99 % band: the binomial
#: 0.1 % quantile of Binomial(2000, 0.99) / 2000.
_MIN_FRAC_IN_BAND = float(stats.binom.ppf(0.001, N_RESAMPLES, 0.99) / N_RESAMPLES)


def _rng(*key: int) -> np.random.Generator:
    return case_rng(*key)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
@pytest.mark.parametrize("n", N_CAL)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_split_coverage_follows_beta_law(dist: str, n: int, alpha: float) -> None:
    rng = _rng(*split_case_key(dist, n, alpha))
    draws = split_coverage_draws(dist, n, alpha, N_RESAMPLES, rng)
    s = summarise_split_draws(draws, n, alpha, level=0.99)
    assert s["ks_pvalue"] > 0.01, s
    assert s["mean_in_band"], s
    assert s["frac_draws_in_band"] >= _MIN_FRAC_IN_BAND, s
    # The textbook bracket, within 3 MC standard errors.
    lo, hi = s["theory_interval"]
    assert lo - 3 * s["mc_se"] <= s["mean_coverage"] <= hi + 3 * s["mc_se"], s


@pytest.mark.parametrize("convention", ["answer_required", "abstain_allowed"])
def test_failure_mixture_stays_valid(convention: str) -> None:
    """5 % failures at alpha = 0.1: atoms at +/-inf make the law conservative, never invalid."""
    n, alpha, rate = 1000, 0.10, 0.05
    rng = _rng(7, n, 0 if convention == "answer_required" else 1)
    draws = split_coverage_draws(
        "gaussian", n, alpha, N_RESAMPLES, rng, failure_rate=rate, convention=convention
    )
    s = summarise_split_draws(draws, n, alpha)
    assert s["mean_coverage"] >= 1 - alpha - 2.5758293035489 * s["mc_se"], s


def test_answer_required_failure_rate_above_alpha_gives_infinite_sets() -> None:
    """Failures above alpha: under answer_required, q = +inf in essentially every resample."""
    n, alpha, rate = 1000, 0.05, 0.10
    draws = split_coverage_draws(
        "gaussian", n, alpha, 200, _rng(8), failure_rate=rate, convention="answer_required"
    )
    assert np.all(draws == 1.0)  # covered only because the set is the whole space


def test_wrong_quantile_would_be_detected() -> None:
    """Power check: an off-by-one index (k - 1) at n = 100 must fail the KS test."""
    from poseconf.conformal.split import quantile_index
    from poseconf.synthetic_validity import sample_scores, score_cdf

    n, alpha = 100, 0.10
    rng = _rng(9)
    k = quantile_index(n, alpha) - 1
    draws = np.array(
        [
            score_cdf("gaussian", np.sort(sample_scores("gaussian", n, rng))[k - 1])
            for _ in range(2000)
        ]
    )
    s = summarise_split_draws(draws, n, alpha)
    assert s["ks_pvalue"] < 0.01, s
