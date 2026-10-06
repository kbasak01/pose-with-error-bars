"""Synthetic coverage studies with known score laws, for validating the conformal core.

Shared by `tests/test_conformal_coverage_law.py`, `tests/test_conformal_weighted.py` and
`scripts/run_conformal_synthetic_validity.py`, so the committed results JSON and the tests measure
the same thing. No SPEED+ data, no torch.

Coverage *given the calibration set* is computed exactly from the score CDF (no test-sample noise),
so each resample yields one draw from the conditional-coverage law, which for continuous scores is
Beta(n + 1 - l, l), l = floor((n + 1) alpha) (Angelopoulos & Bates, section 3.2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from poseconf.conformal.metrics import (
    beta_band,
    beta_mean_band,
    beta_parameters,
    ks_against_beta,
    mc_standard_error,
)
from poseconf.conformal.split import conformal_quantile, failure_score
from poseconf.conformal.weighted import (
    effective_sample_size,
    max_normalised_weight,
    weighted_conformal_quantiles,
)

__all__ = [
    "DISTRIBUTIONS",
    "SEED",
    "case_rng",
    "covariate_shift_draws",
    "score_cdf",
    "sample_scores",
    "split_case_key",
    "split_coverage_draws",
    "summarise_split_draws",
]

#: Root seed of every synthetic study (CLAUDE.md: seed 1337).
SEED = 1337

#: Score laws with a closed-form (or quadrature-exact) CDF.
DISTRIBUTIONS = ("gaussian", "student_t3", "heteroscedastic")

#: Student-t degrees of freedom for the heavy-tailed case.
_T_DF = 3

#: Heteroscedastic model: x ~ U(0, 1), y = x + (a + b x) eps, score |y - x| (unnormalised).
_HET_A, _HET_B = 0.1, 2.0

#: Gauss-Legendre nodes for integrating the heteroscedastic CDF over x in [0, 1].
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(96)
_GL_X = 0.5 * (_GL_NODES + 1.0)
_GL_W = 0.5 * _GL_WEIGHTS

#: Covariate shift: source x ~ N(0, 1), target x ~ N(SHIFT, 1); y = x + exp(x / 2) eps.
SHIFT_MEAN = 1.0


def case_rng(*key: int) -> np.random.Generator:
    """The generator for one study case, keyed by integers; tests and the results script share it."""
    return np.random.default_rng([SEED, *key])


def split_case_key(dist: str, n: int, alpha: float) -> tuple[int, int, int]:
    """Seed key of a split Beta-law case."""
    return DISTRIBUTIONS.index(dist), n, round(alpha * 1000)


def sample_scores(dist: str, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray:
    """Draw nonconformity scores from a named law.

    Args:
        dist: One of `DISTRIBUTIONS`.
        size: Output shape.
        rng: Generator.

    Returns:
        Non-negative scores.
    """
    if dist == "gaussian":
        return np.abs(rng.standard_normal(size))
    if dist == "student_t3":
        return np.abs(rng.standard_t(_T_DF, size))
    if dist == "heteroscedastic":
        x = rng.uniform(0.0, 1.0, size)
        return np.abs((_HET_A + _HET_B * x) * rng.standard_normal(size))
    raise ValueError(f"unknown distribution {dist!r}; expected one of {DISTRIBUTIONS}")


def score_cdf(dist: str, q: float) -> float:
    """Exact P(score <= q) for a fresh draw of the named law.

    Args:
        dist: One of `DISTRIBUTIONS`.
        q: Threshold; may be +/-inf.

    Returns:
        The probability.
    """
    if q == np.inf:
        return 1.0
    if q < 0.0:
        return 0.0
    if dist == "gaussian":
        return float(2.0 * stats.norm.cdf(q) - 1.0)
    if dist == "student_t3":
        return float(2.0 * stats.t.cdf(q, _T_DF) - 1.0)
    if dist == "heteroscedastic":
        sigma = _HET_A + _HET_B * _GL_X
        return float(np.sum(_GL_W * (2.0 * stats.norm.cdf(q / sigma) - 1.0)))
    raise ValueError(f"unknown distribution {dist!r}; expected one of {DISTRIBUTIONS}")


def split_coverage_draws(
    dist: str,
    n: int,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
    *,
    failure_rate: float = 0.0,
    convention: str = "answer_required",
) -> NDArray:
    """Conditional coverage of split CP over independent calibration sets.

    A frame "fails" (no point estimate) with probability `failure_rate`; its score is
    `failure_score(convention)`. Coverage of a fresh frame given q is then
    `(1 - p) F(q) + p * [failure covered]`, where a failure is covered under `abstain_allowed`
    always, and under `answer_required` only when q = +inf.

    Args:
        dist: Score law.
        n: Calibration size.
        alpha: Miscoverage level.
        n_resamples: Number of independent calibration sets.
        rng: Generator.
        failure_rate: Probability a frame has no point estimate.
        convention: PnP-failure convention for failed frames.

    Returns:
        (n_resamples,) exact conditional coverages.
    """
    fail_value = failure_score(convention)
    draws = np.empty(n_resamples)
    for r in range(n_resamples):
        scores = sample_scores(dist, n, rng)
        if failure_rate > 0.0:
            scores[rng.uniform(size=n) < failure_rate] = fail_value
        q = conformal_quantile(scores, alpha)
        solved_part = score_cdf(dist, q)
        if convention == "abstain_allowed":
            # q = -inf: every frame abstains, so every frame is covered.
            failed_part = 1.0
            if q == -np.inf:
                solved_part = 1.0
        else:
            failed_part = 1.0 if q == np.inf else 0.0
        draws[r] = (1.0 - failure_rate) * solved_part + failure_rate * failed_part
    return draws


def summarise_split_draws(
    draws: NDArray, n: int, alpha: float, *, level: float = 0.99
) -> dict[str, Any]:
    """Compare conditional-coverage draws with the Beta law.

    Args:
        draws: Coverage draws from `split_coverage_draws`.
        n: Calibration size.
        alpha: Miscoverage level.
        level: Band level for the mean and for single draws.

    Returns:
        Mean, MC standard error, Beta parameters, mean band, KS statistic and p-value, the fraction
        of single draws inside the per-draw band, and the theoretical interval
        [1 - alpha, 1 - alpha + 1/(n + 1)].
    """
    a, b = beta_parameters(n, alpha)
    mean_lo, mean_hi = beta_mean_band(n, alpha, n_resamples=len(draws), level=level)
    draw_lo, draw_hi = beta_band(n, alpha, level=level)
    ks_stat, ks_p = ks_against_beta(draws, n, alpha)
    mean = float(np.mean(draws))
    return {
        "n_cal": n,
        "alpha": alpha,
        "n_resamples": len(draws),
        "beta_a": a,
        "beta_b": b,
        "beta_mean": a / (a + b),
        "mean_coverage": mean,
        "mc_se": mc_standard_error(draws),
        "mean_band": [mean_lo, mean_hi],
        "mean_in_band": bool(mean_lo <= mean <= mean_hi),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "frac_draws_in_band": float(np.mean((draws >= draw_lo) & (draws <= draw_hi))),
        "band_level": level,
        "theory_interval": [1.0 - alpha, 1.0 - alpha + 1.0 / (n + 1)],
    }


def covariate_shift_draws(
    n: int,
    alpha: float,
    n_resamples: int,
    n_test: int,
    rng: np.random.Generator,
    *,
    weighted: bool,
) -> dict[str, NDArray]:
    """Coverage on a shifted target, with or without weighted CP using the true weights.

    Source x ~ N(0, 1), target x ~ N(SHIFT_MEAN, 1); y = x + exp(x / 2) eps, eps ~ N(0, 1);
    predictor f(x) = x; score |y - f(x)| = exp(x / 2) |eps|. The true likelihood ratio is
    w(x) = exp(SHIFT_MEAN x - SHIFT_MEAN^2 / 2). For each resample, coverage is averaged over
    `n_test` target x of the exact P(|eps| exp(x/2) <= q(x)), so only x is Monte-Carlo.

    Args:
        n: Source calibration size.
        alpha: Miscoverage level.
        n_resamples: Independent calibration sets.
        n_test: Target x draws per resample.
        rng: Generator.
        weighted: Weighted CP with true weights if True, plain split CP if False.

    Returns:
        `coverage` (n_resamples,), `ess` (n_resamples,), `max_weight` (n_resamples,),
        `inf_fraction` (n_resamples,) — the fraction of test points whose quantile is +inf.
    """
    coverage = np.empty(n_resamples)
    ess = np.empty(n_resamples)
    max_w = np.empty(n_resamples)
    inf_frac = np.empty(n_resamples)
    for r in range(n_resamples):
        x_cal = rng.standard_normal(n)
        s_cal = np.exp(x_cal / 2.0) * np.abs(rng.standard_normal(n))
        x_test = SHIFT_MEAN + rng.standard_normal(n_test)
        w_cal = np.exp(SHIFT_MEAN * x_cal - SHIFT_MEAN**2 / 2.0)
        if weighted:
            w_test = np.exp(SHIFT_MEAN * x_test - SHIFT_MEAN**2 / 2.0)
            q = weighted_conformal_quantiles(s_cal, w_cal, w_test, alpha)
        else:
            q = np.full(n_test, conformal_quantile(s_cal, alpha))
        p = np.where(np.isinf(q), 1.0, 2.0 * stats.norm.cdf(q / np.exp(x_test / 2.0)) - 1.0)
        coverage[r] = float(np.mean(p))
        ess[r] = effective_sample_size(w_cal)
        max_w[r] = max_normalised_weight(w_cal)
        inf_frac[r] = float(np.mean(np.isinf(q)))
    return {"coverage": coverage, "ess": ess, "max_weight": max_w, "inf_fraction": inf_frac}
