"""Coverage with Clopper-Pearson intervals, set size, answer rate, silent-failure rate, Beta law.

Outcome semantics (docs/SCORES.md, "Coverage event"):

* A frame is **answered** when it has a point estimate (`valid`) and q > -inf. A set at q = -inf is
  empty, and an empty set is an abstention.
* Under `answer_required`, a frame is **covered** iff s <= q (a failure, s = +inf, is covered only by
  the whole-space set q = +inf).
* Under `abstain_allowed`, a frame is covered iff it is not answered or s <= q.
* **Silent failure** = answered and not covered.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from poseconf.conformal.split import as_scores, failure_score, quantile_index, rational_alpha

__all__ = [
    "beta_band",
    "beta_mean_band",
    "beta_parameters",
    "clopper_pearson",
    "coverage_by_slice",
    "coverage_summary",
    "ks_against_beta",
    "ks_against_pmf",
    "mc_standard_error",
    "outcomes",
    "pmf_mean",
    "resplit_coverage_law",
    "set_size_summary",
]

#: Default two-sided confidence level for coverage intervals.
CI_LEVEL = 0.95


def outcomes(
    scores: ArrayLike, valid: ArrayLike, q: float, convention: str
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Per-frame (covered, answered) for test scores against a quantile.

    Args:
        scores: (m,) test scores computed under `convention`.
        valid: (m,) True where the frame has a point estimate.
        q: Conformal quantile (may be +/-inf, not NaN).
        convention: `"answer_required"` or `"abstain_allowed"`.

    Returns:
        `covered`, `answered`, both (m,) bool.

    Raises:
        ValueError: On NaN, or if the scores are inconsistent with the convention (a failed frame
            whose score is not the convention's +/-inf, or a valid frame with an infinite score).
            This is what stops two conventions being mixed in one evaluation.
    """
    s = as_scores(scores)
    ok = np.asarray(valid, dtype=bool)
    if ok.shape != s.shape:
        raise ValueError(f"valid has shape {ok.shape}, scores {s.shape}")
    if math.isnan(q):
        raise ValueError("q is NaN")
    fail = failure_score(convention)
    if np.any(s[~ok] != fail):
        raise ValueError(
            f"scores inconsistent with convention {convention!r}: failed frames must score {fail}"
        )
    if not np.all(np.isfinite(s[ok])):
        raise ValueError(
            f"scores inconsistent with convention {convention!r}: a valid frame has an infinite "
            "score"
        )
    answered = ok & (q > -math.inf)
    inside = s <= q
    covered = (~answered | inside) if convention == "abstain_allowed" else inside
    return covered, answered


def clopper_pearson(k: int, m: int, level: float = CI_LEVEL) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial interval for k successes in m trials.

    Args:
        k: Successes, 0 <= k <= m.
        m: Trials, m >= 1.
        level: Two-sided confidence level.

    Returns:
        (lower, upper); 0 at k = 0 and 1 at k = m.
    """
    if m < 1 or not 0 <= k <= m:
        raise ValueError(f"need 0 <= k <= m and m >= 1, got k={k}, m={m}")
    tail = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(stats.beta.ppf(tail, k, m - k + 1))
    hi = 1.0 if k == m else float(stats.beta.ppf(1.0 - tail, k + 1, m - k))
    return lo, hi


def coverage_summary(
    covered: ArrayLike, answered: ArrayLike, level: float = CI_LEVEL
) -> dict[str, Any]:
    """Coverage, its Clopper-Pearson interval, answer rate and silent-failure rate.

    Args:
        covered: (m,) bool from `outcomes`.
        answered: (m,) bool from `outcomes`.
        level: Interval level.

    Returns:
        `n_total`, `n_covered`, `n_answered`, `coverage`, `coverage_ci95` (as [lo, hi]),
        `answer_rate`, `silent_failure_rate`.
    """
    c = np.asarray(covered, dtype=bool)
    a = np.asarray(answered, dtype=bool)
    if c.shape != a.shape or c.ndim != 1 or c.size == 0:
        raise ValueError(f"covered/answered must be equal-length non-empty 1-D, got {c.shape}")
    m, k = int(c.size), int(c.sum())
    return {
        "n_total": m,
        "n_covered": k,
        "n_answered": int(a.sum()),
        "coverage": k / m,
        "coverage_ci95": list(clopper_pearson(k, m, level)),
        "answer_rate": float(a.mean()),
        "silent_failure_rate": float((a & ~c).mean()),
    }


def coverage_by_slice(
    covered: ArrayLike, answered: ArrayLike, slices: ArrayLike, level: float = CI_LEVEL
) -> dict[Any, dict[str, Any]]:
    """`coverage_summary` per slice label. Slices may use GT quantities; scores may not.

    Args:
        covered: (m,) bool.
        answered: (m,) bool.
        slices: (m,) slice labels.
        level: Interval level.

    Returns:
        Slice label -> summary, in sorted label order.
    """
    c = np.asarray(covered, dtype=bool)
    a = np.asarray(answered, dtype=bool)
    labels = np.asarray(slices)
    if labels.shape != c.shape:
        raise ValueError(f"slices has shape {labels.shape}, covered {c.shape}")
    return {
        label.item(): coverage_summary(c[labels == label], a[labels == label], level)
        for label in np.unique(labels)
    }


def set_size_summary(radii: ArrayLike) -> dict[str, float]:
    """Median and p90 of per-frame set radii, keeping +inf (never interpolated, never dropped).

    Uses the inverted-CDF (order-statistic) definition so an infinite radius is reported as inf
    rather than producing NaN by interpolation. Pass the answered frames' radii.

    Args:
        radii: (m,) radii, m >= 1.

    Returns:
        `median`, `p90`.
    """
    r = np.asarray(radii, dtype=np.float64).reshape(-1)
    if r.size == 0:
        raise ValueError("no radii to summarise")
    if np.isnan(r).any():
        raise ValueError("radii contain NaN")
    med, p90 = np.quantile(r, [0.5, 0.9], method="inverted_cdf")
    return {"median": float(med), "p90": float(p90)}


def beta_parameters(n: int, alpha: float) -> tuple[int, int]:
    """Parameters (n + 1 - l, l), l = floor((n + 1) alpha), of the conditional-coverage law.

    For continuous exchangeable scores, coverage given the calibration set is Beta(n + 1 - l, l)
    (Angelopoulos & Bates, section 3.2). l = 0 means q = +inf and coverage identically 1.

    Args:
        n: Calibration size.
        alpha: Miscoverage level.

    Returns:
        (a, b) with b = l, possibly 0.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    l_index = math.floor((n + 1) * rational_alpha(alpha))
    return n + 1 - l_index, l_index


def beta_band(n: int, alpha: float, level: float = 0.99) -> tuple[float, float]:
    """Central `level` interval of a single conditional-coverage draw.

    Returns:
        (lo, hi); (1, 1) when the law is degenerate at 1 (l = 0).
    """
    a, b = beta_parameters(n, alpha)
    if b == 0:
        return 1.0, 1.0
    tail = (1.0 - level) / 2.0
    return float(stats.beta.ppf(tail, a, b)), float(stats.beta.ppf(1.0 - tail, a, b))


def beta_mean_band(
    n: int, alpha: float, *, n_resamples: int, level: float = 0.99
) -> tuple[float, float]:
    """Normal-approximation `level` band for the mean of `n_resamples` independent draws.

    Returns:
        (mean - z sd / sqrt(R), mean + z sd / sqrt(R)) of the Beta law.
    """
    a, b = beta_parameters(n, alpha)
    if b == 0:
        return 1.0, 1.0
    mean = a / (a + b)
    sd = float(stats.beta.std(a, b)) / math.sqrt(n_resamples)
    z = float(stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    return mean - z * sd, mean + z * sd


def ks_against_beta(coverages: ArrayLike, n: int, alpha: float) -> tuple[float, float]:
    """One-sample KS test of coverage draws against Beta(n + 1 - l, l).

    Returns:
        (statistic, p-value). For the degenerate law (l = 0), (0, 1) if every draw is 1, else (1, 0).
    """
    x = np.asarray(coverages, dtype=np.float64).reshape(-1)
    a, b = beta_parameters(n, alpha)
    if b == 0:
        return (0.0, 1.0) if np.all(x == 1.0) else (1.0, 0.0)
    result = stats.kstest(x, stats.beta(a, b).cdf)
    return float(result.statistic), float(result.pvalue)


def mc_standard_error(values: ArrayLike) -> float:
    """Monte-Carlo standard error of a mean: sd(values, ddof=1) / sqrt(R)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 2:
        raise ValueError("need at least two values for a standard error")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def resplit_coverage_law(
    n: int, m: int, alpha: float, *, n_top_atom: int = 0, n_bottom_atom: int = 0
) -> NDArray[np.float64]:
    """Exact law of the covered count when a fixed pool is re-split at random (cal n, test m).

    Conditional on a pool of N = n + m scores whose finite values are distinct, a uniformly random
    partition makes every rank pattern equally likely. With k = ceil((n+1)(1-alpha)), the number
    X_tb of test scores ranked below the k-th calibration order statistic (ties inside an atom
    broken at random) is negative-hypergeometric, i.e. Beta-Binomial(m; k, n + 1 - k). Its mean is
    m k / (n + 1); its variance exceeds that of m * Beta(k, n + 1 - k) because the test set is
    finite.

    PnP failures form atoms (`split.CONVENTIONS`), and `s <= q` counts a whole tie:
    * `n_top_atom` scores at +inf (answer_required): q = +inf iff the k-th calibration order
      statistic falls in the atom, i.e. iff k + X_tb > N - n_top_atom; then every test frame is
      covered (X = m). Otherwise q is finite and X = X_tb.
    * `n_bottom_atom` scores at -inf (abstain_allowed): q = -inf iff k + X_tb <= n_bottom_atom;
      then every frame abstains and is covered (X = m). Otherwise X = X_tb (failures sit below q
      and are covered by abstaining).
    If k > n, q = +inf and X = m surely. Ties among finite scores (not modelled) can only raise X.

    Args:
        n: Calibration size per re-split.
        m: Test size per re-split.
        alpha: Miscoverage level.
        n_top_atom: Pool frames scoring +inf.
        n_bottom_atom: Pool frames scoring -inf.

    Returns:
        (m + 1,) pmf of the covered count X; coverage is X / m.

    Raises:
        ValueError: On non-positive sizes or atoms that do not fit in the pool.
    """
    if n < 1 or m < 1:
        raise ValueError(f"need n >= 1 and m >= 1, got n={n}, m={m}")
    total = n + m
    if n_top_atom < 0 or n_bottom_atom < 0 or n_top_atom + n_bottom_atom > total:
        raise ValueError(f"atoms ({n_bottom_atom}, {n_top_atom}) do not fit a pool of {total}")
    pmf = np.zeros(m + 1)
    k = quantile_index(n, alpha)
    if k > n:
        pmf[m] = 1.0
        return pmf
    j = np.arange(m + 1)
    tie_broken = stats.betabinom.pmf(j, m, k, n + 1 - k)
    to_full = (k + j > total - n_top_atom) | (k + j <= n_bottom_atom)
    pmf[~to_full] = tie_broken[~to_full]
    pmf[m] += float(tie_broken[to_full].sum())
    return pmf / pmf.sum()


def pmf_mean(pmf: ArrayLike) -> float:
    """Mean coverage X / m of a covered-count pmf over 0..m."""
    p = np.asarray(pmf, dtype=np.float64)
    m = p.size - 1
    return float(np.dot(np.arange(m + 1), p) / m)


def ks_against_pmf(counts: ArrayLike, pmf: ArrayLike) -> tuple[float, float]:
    """KS test of integer covered counts (0..m) against a discrete pmf.

    D = max_j |F_emp(j) - F(j)| over the support, which is the exact sup for two step functions on
    the integers. The p-value uses the continuous Kolmogorov law (`scipy.stats.kstwo`), which is
    conservative for a discrete null: the true p-value is at least the one returned.

    Args:
        counts: (R,) covered counts.
        pmf: (m + 1,) reference pmf.

    Returns:
        (D, p-value).
    """
    x = np.asarray(counts).reshape(-1)
    p = np.asarray(pmf, dtype=np.float64)
    m = p.size - 1
    if x.size < 1 or np.any(x != np.round(x)) or np.any((x < 0) | (x > m)):
        raise ValueError(f"counts must be integers in [0, {m}]")
    empirical = np.cumsum(np.bincount(x.astype(np.int64), minlength=m + 1)) / x.size
    statistic = float(np.max(np.abs(empirical - np.cumsum(p))))
    return statistic, float(stats.kstwo.sf(statistic, x.size))
