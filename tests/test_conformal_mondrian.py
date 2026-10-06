"""Mondrian (group-conditional) CP: each group is its own split-CP problem; small groups get inf."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from poseconf.conformal.mondrian import (
    assign_bins,
    bin_edges,
    mondrian_quantiles,
    product_groups,
)
from poseconf.conformal.split import conformal_quantile, quantile_index


def test_each_group_equals_split_on_its_subset() -> None:
    rng = np.random.default_rng(31)
    scores = rng.standard_exponential(600)
    groups = rng.integers(0, 3, 600)
    q = mondrian_quantiles(scores, groups, 0.1, group_ids=[0, 1, 2])
    for g in (0, 1, 2):
        assert q[g] == conformal_quantile(scores[groups == g], 0.1)


def test_missing_group_is_inf_not_borrowed() -> None:
    scores = np.arange(100.0)
    groups = np.zeros(100, dtype=int)
    q = mondrian_quantiles(scores, groups, 0.1, group_ids=[0, 7])
    assert q[7] == np.inf
    assert np.isfinite(q[0])


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_small_group_threshold(alpha: float) -> None:
    """A group is finite iff quantile_index(n_g) <= n_g; the smallest such n_g is ceil(1/a) - 1."""
    smallest = next(n for n in range(1, 1000) if quantile_index(n, alpha) <= n)
    assert smallest == int(np.ceil(1 / alpha)) - 1
    groups = np.array([0] * smallest + [1] * (smallest - 1) + [2] * smallest)
    scores = np.arange(groups.size, dtype=float)
    q = mondrian_quantiles(scores, groups, alpha, group_ids=[0, 1, 2])
    assert np.isfinite(q[0])
    assert q[1] == np.inf


def test_unrequested_groups_are_still_returned() -> None:
    q = mondrian_quantiles(np.arange(40.0), np.repeat([3, 5], 20), 0.1)
    assert set(q) == {3, 5}


def test_group_conditional_coverage_holds_per_group() -> None:
    """Two groups with 10x different noise: pooled CP misses one group, Mondrian holds both."""
    rng = np.random.default_rng(32)
    alpha, n_per, reps = 0.1, 400, 1000
    scale = {0: 1.0, 1: 10.0}
    cov_m = {0: [], 1: []}
    cov_pooled = {0: [], 1: []}
    for _ in range(reps):
        groups = np.repeat([0, 1], n_per)
        scores = np.abs(rng.standard_normal(2 * n_per)) * np.where(groups == 0, 1.0, 10.0)
        q = mondrian_quantiles(scores, groups, alpha, group_ids=[0, 1])
        q_pooled = conformal_quantile(scores, alpha)
        for g in (0, 1):
            cov_m[g].append(2 * stats.norm.cdf(q[g] / scale[g]) - 1)
            cov_pooled[g].append(2 * stats.norm.cdf(q_pooled / scale[g]) - 1)
    for g in (0, 1):
        se = np.std(cov_m[g], ddof=1) / np.sqrt(reps)
        assert np.mean(cov_m[g]) >= 1 - alpha - 3 * se
    assert np.mean(cov_pooled[1]) < 1 - alpha - 0.02  # the noisy group under-covers when pooled


def test_bins_and_product_groups() -> None:
    tune = np.arange(1.0, 10.0)  # fit edges on "val_tune"
    edges = bin_edges(tune, 3)
    assert len(edges) == 2
    bins = assign_bins(np.array([0.0, 4.0, 5.0, 100.0]), edges)
    assert bins.min() == 0 and bins.max() == 2
    assert bins[0] == 0 and bins[-1] == 2
    a = np.array([0, 1, 2, 0])
    b = np.array([1, 1, 0, 2])
    g = product_groups([a, b], [3, 3])
    assert g.tolist() == [1, 4, 6, 2]
    # Same tuple, same label, whatever else is in the batch.
    assert product_groups([a[:1], b[:1]], [3, 3]).tolist() == [1]
    with pytest.raises(ValueError, match="0, 3"):
        product_groups([a, b + 2], [3, 3])
    with pytest.raises(ValueError, match="NaN"):
        assign_bins(np.array([np.nan]), edges)
    with pytest.raises(ValueError, match="length"):
        product_groups([a, b[:2]], [3, 3])


def test_nan_raises() -> None:
    with pytest.raises(ValueError, match="NaN"):
        mondrian_quantiles(np.array([np.nan, 1.0]), np.array([0, 0]), 0.1)
    with pytest.raises(ValueError, match="length"):
        mondrian_quantiles(np.array([1.0, 2.0]), np.array([0]), 0.1)
