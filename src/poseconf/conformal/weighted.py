"""Weighted split conformal under covariate shift (Tibshirani et al. 2019).

For a test point with weight w and calibration weights w_i (likelihood ratios dP_target/dP_source),
the normalised masses are p_i = w_i / (sum_j w_j + w) and p_test = w / (same sum). The quantile is
the smallest calibration score s with sum_{s_i <= s} p_i >= 1 - alpha, the test mass p_test sitting
at +inf; if the calibration mass alone never reaches 1 - alpha, q = +inf. The quantile is per test
point. With equal weights it reduces exactly to `split.conformal_quantile`.

Weight clipping trades validity for set size. If used, the clip value is chosen on `val_tune` only
and recorded with the result (CLAUDE.md invariant 3).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from poseconf.conformal.split import as_scores, rational_alpha

__all__ = [
    "clip_weights",
    "effective_sample_size",
    "max_normalised_weight",
    "weighted_conformal_quantile",
    "weighted_conformal_quantiles",
    "weights_from_probabilities",
]

#: Relative tolerance on the cumulative-mass comparison. Float cumulative sums of equal weights hit
#: (1 - alpha)(n + 1) exactly in exact arithmetic but can land one ulp below it; without this the
#: equal-weight case would be one order statistic too high (tests pin the reduction to split CP).
_MASS_REL_TOL = 1e-12


def _as_weights(weights: ArrayLike, name: str) -> NDArray[np.float64]:
    w = np.asarray(weights, dtype=np.float64)
    if np.isnan(w).any() or np.isinf(w).any() or (w < 0).any():
        raise ValueError(f"{name}: every weight must be finite and >= 0")
    return w


def weighted_conformal_quantiles(
    cal_scores: ArrayLike, cal_weights: ArrayLike, test_weights: ArrayLike, alpha: float
) -> NDArray[np.float64]:
    """Per-test-point weighted conformal quantiles.

    Args:
        cal_scores: (n,) calibration scores; +/-inf kept, NaN refused.
        cal_weights: (n,) non-negative finite weights w(x_i).
        test_weights: (m,) non-negative finite weights w(x) of the test points.
        alpha: Miscoverage level in (0, 1).

    Returns:
        (m,) quantiles, +inf where the calibration mass cannot reach 1 - alpha.
    """
    s = as_scores(cal_scores, "cal_scores")
    w = _as_weights(cal_weights, "cal_weights")
    w_test = _as_weights(test_weights, "test_weights").reshape(-1)
    if w.shape != s.shape:
        raise ValueError(f"cal_weights length {w.shape} does not match cal_scores {s.shape}")
    level = float(1 - rational_alpha(alpha))

    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    cum = np.cumsum(w[order])
    total = w.sum() + w_test  # (m,), includes the test point's own mass
    threshold = level * total * (1.0 - _MASS_REL_TOL)
    # First index whose cumulative mass reaches the threshold; == n means none does.
    idx = np.searchsorted(cum, threshold, side="left")
    out = np.full(w_test.shape, math.inf)
    hit = (idx < s.size) & (total > 0)
    out[hit] = s_sorted[idx[hit]]
    # Ties: the mass at s_sorted[j] is the cumulative sum through the last tie, which is >= cum[j],
    # so the first index reaching the threshold already returns the correct tied value.
    return out


def weighted_conformal_quantile(
    cal_scores: ArrayLike, cal_weights: ArrayLike, test_weight: float, alpha: float
) -> float:
    """Weighted conformal quantile for a single test point. See `weighted_conformal_quantiles`."""
    return float(weighted_conformal_quantiles(cal_scores, cal_weights, [test_weight], alpha)[0])


def effective_sample_size(weights: ArrayLike) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2."""
    w = _as_weights(weights, "weights").reshape(-1)
    denom = float(np.sum(w**2))
    if denom == 0.0:
        raise ValueError("all weights are zero")
    return float(np.sum(w) ** 2 / denom)


def max_normalised_weight(weights: ArrayLike) -> float:
    """max_i w_i / sum_j w_j."""
    w = _as_weights(weights, "weights").reshape(-1)
    total = float(np.sum(w))
    if total == 0.0:
        raise ValueError("all weights are zero")
    return float(np.max(w) / total)


def weights_from_probabilities(
    p_target: ArrayLike, n_source: int, n_target: int
) -> NDArray[np.float64]:
    """Likelihood-ratio weights from a domain classifier: c / (1 - c) * n_source / n_target.

    Args:
        p_target: (n,) classifier probabilities P(target | x), strictly inside (0, 1).
        n_source: Source points the classifier was trained on.
        n_target: Target points the classifier was trained on.

    Returns:
        (n,) weights.
    """
    p = np.asarray(p_target, dtype=np.float64)
    if np.isnan(p).any() or (p <= 0).any() or (p >= 1).any():
        raise ValueError("classifier probabilities must lie strictly inside (0, 1)")
    if n_source < 1 or n_target < 1:
        raise ValueError(f"n_source and n_target must be >= 1, got {n_source}, {n_target}")
    return p / (1.0 - p) * (n_source / n_target)


def clip_weights(weights: ArrayLike, clip: float) -> NDArray[np.float64]:
    """Clip weights at `clip` (chosen on val_tune only, recorded with the result).

    Clipping invalidates the exact weighted-CP guarantee; report it beside every number it touches.
    """
    if not clip > 0:
        raise ValueError(f"clip must be > 0, got {clip}")
    return np.minimum(_as_weights(weights, "weights"), clip)
