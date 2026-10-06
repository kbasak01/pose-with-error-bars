"""Coverage outcomes, Clopper-Pearson, set-size summaries and the Beta coverage law."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from poseconf.conformal.metrics import (
    beta_band,
    beta_mean_band,
    beta_parameters,
    clopper_pearson,
    coverage_by_slice,
    coverage_summary,
    ks_against_beta,
    mc_standard_error,
    outcomes,
    set_size_summary,
)
from poseconf.conformal.split import conformal_quantile, failure_score


@pytest.mark.parametrize(("k", "m"), [(0, 10), (10, 10), (3, 10), (450, 500), (1, 1), (0, 1)])
def test_clopper_pearson_matches_statsmodels(k: int, m: int) -> None:
    proportion = pytest.importorskip("statsmodels.stats.proportion")
    lo, hi = clopper_pearson(k, m)
    ref_lo, ref_hi = proportion.proportion_confint(k, m, alpha=0.05, method="beta")
    assert lo == pytest.approx(ref_lo, abs=1e-12)
    assert hi == pytest.approx(ref_hi, abs=1e-12)


def test_clopper_pearson_edges() -> None:
    assert clopper_pearson(0, 20)[0] == 0.0
    assert clopper_pearson(20, 20)[1] == 1.0
    with pytest.raises(ValueError):
        clopper_pearson(0, 0)
    with pytest.raises(ValueError):
        clopper_pearson(5, 4)


def test_failure_score_conventions() -> None:
    assert failure_score("answer_required") == np.inf
    assert failure_score("abstain_allowed") == -np.inf
    with pytest.raises(ValueError, match="convention"):
        failure_score("solved_only")


def test_outcomes_truth_table() -> None:
    # Two solved frames (scores 0.5 and 2.0) and one failure.
    valid = np.array([True, True, False])
    ar = np.array([0.5, 2.0, np.inf])
    aa = np.array([0.5, 2.0, -np.inf])

    covered, answered = outcomes(ar, valid, 1.0, "answer_required")
    assert covered.tolist() == [True, False, False]
    assert answered.tolist() == [True, True, False]

    covered, answered = outcomes(aa, valid, 1.0, "abstain_allowed")
    assert covered.tolist() == [True, False, True]  # the failure abstains
    assert answered.tolist() == [True, True, False]

    # q = +inf under answer_required: the whole space, everything covered.
    covered, answered = outcomes(ar, valid, np.inf, "answer_required")
    assert covered.all()
    assert answered.tolist() == [True, True, False]


def test_all_failure_calibration_answer_required_gives_infinite_set() -> None:
    q = conformal_quantile(np.full(40, failure_score("answer_required")), 0.1)
    assert q == np.inf
    covered, answered = outcomes(
        np.array([0.3, 7.0, np.inf]), np.array([True, True, False]), q, "answer_required"
    )
    assert covered.all()


def test_all_failure_calibration_abstain_allowed_gives_always_abstain() -> None:
    q = conformal_quantile(np.full(40, failure_score("abstain_allowed")), 0.1)
    assert q == -np.inf
    covered, answered = outcomes(
        np.array([0.3, 7.0, -np.inf]), np.array([True, True, False]), q, "abstain_allowed"
    )
    assert not answered.any()  # an empty set is an abstention
    assert covered.all()
    summary = coverage_summary(covered, answered)
    assert summary["answer_rate"] == 0.0
    assert summary["silent_failure_rate"] == 0.0


@pytest.mark.parametrize(
    ("scores", "convention"),
    [
        (np.array([0.5, -np.inf]), "answer_required"),  # failure with the other convention's sign
        (np.array([0.5, np.inf]), "abstain_allowed"),
        (np.array([0.5, 3.0]), "abstain_allowed"),  # failure with a finite score
        (np.array([np.inf, -np.inf]), "abstain_allowed"),  # solved frame with an infinite score
    ],
)
def test_outcomes_refuse_mixed_conventions(scores: np.ndarray, convention: str) -> None:
    with pytest.raises(ValueError, match="convention"):
        outcomes(scores, np.array([True, False]), 1.0, convention)


def test_outcomes_refuse_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        outcomes(np.array([np.nan]), np.array([True]), 1.0, "answer_required")
    with pytest.raises(ValueError, match="NaN"):
        outcomes(np.array([1.0]), np.array([True]), np.nan, "answer_required")


def test_coverage_summary_fields() -> None:
    covered = np.array([True, True, False, True, False])
    answered = np.array([True, True, True, False, False])
    s = coverage_summary(covered, answered)
    assert s["n_total"] == 5
    assert s["n_covered"] == 3
    assert s["n_answered"] == 3
    assert s["coverage"] == pytest.approx(0.6)
    assert s["answer_rate"] == pytest.approx(0.6)
    assert s["silent_failure_rate"] == pytest.approx(0.2)  # answered and not covered: frame 2
    assert s["coverage_ci95"] == list(clopper_pearson(3, 5))


def test_coverage_by_slice() -> None:
    covered = np.array([True, False, True, True])
    answered = np.ones(4, dtype=bool)
    out = coverage_by_slice(covered, answered, np.array(["a", "a", "b", "b"]))
    assert out["a"]["coverage"] == 0.5
    assert out["b"]["coverage"] == 1.0


def test_set_size_summary_keeps_inf() -> None:
    radii = np.array([1.0, 2.0, 3.0, np.inf, np.inf])
    s = set_size_summary(radii)
    assert s["median"] == 3.0
    assert s["p90"] == np.inf
    assert set_size_summary(np.array([1.0, 2.0, 3.0, 4.0]))["median"] in (2.0, 3.0)
    with pytest.raises(ValueError, match="NaN"):
        set_size_summary(np.array([1.0, np.nan]))


def test_beta_parameters() -> None:
    # n = 99, alpha = 0.1: l = floor(100 * 0.1) = 10 -> Beta(90, 10), mean 0.9.
    assert beta_parameters(99, 0.1) == (90, 10)
    # n = 9, alpha = 0.1: l = floor(10 * 0.1) = 1 exactly (the float product is 1.0000000000000002).
    assert beta_parameters(9, 0.1) == (9, 1)
    # l = 0: q = +inf, coverage identically 1.
    assert beta_parameters(5, 0.1) == (6, 0)
    assert beta_band(5, 0.1) == (1.0, 1.0)


def test_beta_band_and_ks() -> None:
    a, b = beta_parameters(999, 0.1)
    lo, hi = beta_band(999, 0.1, level=0.99)
    assert lo == pytest.approx(stats.beta.ppf(0.005, a, b))
    assert hi == pytest.approx(stats.beta.ppf(0.995, a, b))
    draws = stats.beta.rvs(a, b, size=2000, random_state=np.random.default_rng(3))
    _, pvalue = ks_against_beta(draws, 999, 0.1)
    assert pvalue > 0.01
    _, pvalue = ks_against_beta(draws - 0.01, 999, 0.1)
    assert pvalue < 1e-6


def test_beta_mean_band_contains_beta_mean() -> None:
    a, b = beta_parameters(999, 0.1)
    lo, hi = beta_mean_band(999, 0.1, n_resamples=2000, level=0.99)
    sd = stats.beta.std(a, b) / np.sqrt(2000)
    assert lo == pytest.approx(a / (a + b) - 2.5758293035489 * sd)
    assert hi == pytest.approx(a / (a + b) + 2.5758293035489 * sd)


def test_mc_standard_error() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert mc_standard_error(x) == pytest.approx(np.std(x, ddof=1) / 2.0)
