"""Split conformal quantile: the ceil((n+1)(1-alpha))-th order statistic, +inf past n.

Handles +/-inf scores (PnP-failure conventions) and refuses NaN loudly.

The index is computed in exact rational arithmetic. In floats, `ceil((9 + 1) * (1 - 0.1))` is
`ceil(9.000000000000002) = 10`, one order statistic too high — conservative by 1/(n+1), and invisible
in any plot. `rational_alpha` maps alpha to the nearest fraction with denominator <= 10**6, which is
exact for every decimal alpha in the config grid.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CONVENTIONS",
    "as_scores",
    "check_alpha",
    "conformal_quantile",
    "failure_score",
    "quantile_index",
    "rational_alpha",
]

#: PnP-failure conventions (CLAUDE.md invariant 6) and the score a failed frame gets under each.
CONVENTIONS: dict[str, float] = {"answer_required": math.inf, "abstain_allowed": -math.inf}

#: Largest denominator considered when rationalising alpha.
_ALPHA_MAX_DENOMINATOR = 10**6


def failure_score(convention: str) -> float:
    """The score of a frame with no point estimate under a PnP-failure convention.

    Args:
        convention: `"answer_required"` (failure -> +inf) or `"abstain_allowed"` (failure -> -inf).

    Returns:
        +inf or -inf.

    Raises:
        ValueError: On any other convention.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; expected one of {list(CONVENTIONS)}")
    return CONVENTIONS[convention]


def check_alpha(alpha: float) -> float:
    """Validate a miscoverage level.

    Raises:
        ValueError: Unless 0 < alpha < 1.
    """
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:  # also rejects NaN
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


def rational_alpha(alpha: float) -> Fraction:
    """alpha as an exact fraction (nearest with denominator <= 10**6).

    Args:
        alpha: Miscoverage level in (0, 1).

    Returns:
        The fraction; e.g. 0.1 -> 1/10, not 3602879701896397/36028797018963968.
    """
    return Fraction(check_alpha(alpha)).limit_denominator(_ALPHA_MAX_DENOMINATOR)


def quantile_index(n: int, alpha: float) -> int:
    """k = ceil((n + 1)(1 - alpha)), computed exactly. 1-indexed; k > n means q = +inf.

    Args:
        n: Number of calibration scores (>= 0).
        alpha: Miscoverage level in (0, 1).

    Returns:
        The 1-indexed order statistic.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return math.ceil((n + 1) * (1 - rational_alpha(alpha)))


def as_scores(scores: ArrayLike, name: str = "scores") -> NDArray[np.float64]:
    """Validate a 1-D float score array: NaN is refused, +/-inf is kept.

    Raises:
        ValueError: If not 1-D or any entry is NaN.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {values.shape}")
    if np.isnan(values).any():
        raise ValueError(f"{name} contain NaN; a NaN score is a bug upstream, not a value")
    return values


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Split conformal quantile of calibration scores.

    q = the k-th smallest score, k = ceil((n + 1)(1 - alpha)); q = +inf if k > n. For a test score
    exchangeable with the calibration scores, P(s_test <= q) >= 1 - alpha.

    Infinite scores (PnP failures: +inf under answer_required, -inf under abstain_allowed) sort
    like any other value and are never dropped. NaN raises.

    Args:
        scores: (n,) calibration scores; n = 0 is allowed and gives +inf.
        alpha: Miscoverage level in (0, 1).

    Returns:
        The quantile, possibly +/-inf.
    """
    values = as_scores(scores)
    k = quantile_index(values.size, alpha)
    if k > values.size:
        return math.inf
    return float(np.partition(values, k - 1)[k - 1])
