"""Group-conditional (Mondrian) conformal quantiles.

Each group is its own exchangeable split-CP problem, so coverage >= 1 - alpha holds within every
group. A group too small for its quantile (n_g < ceil(1/alpha) - 1) returns +inf rather than
borrowing from neighbours; a group with no calibration points does the same.

Groups must be computable at test time without labels (predicted range, predicted confidence, ...),
and bin edges are fitted on `val_tune` only. A PnP-failed frame has no predicted range, so the
caller gives failures their own group.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from poseconf.conformal.split import as_scores, conformal_quantile

__all__ = ["assign_bins", "bin_edges", "mondrian_quantiles", "product_groups"]


def mondrian_quantiles(
    scores: ArrayLike,
    groups: ArrayLike,
    alpha: float,
    group_ids: Iterable[Hashable] | None = None,
) -> dict[Hashable, float]:
    """Split conformal quantile within each group.

    Args:
        scores: (n,) calibration scores.
        groups: (n,) group label per calibration score.
        alpha: Miscoverage level.
        group_ids: Groups to return in addition to those present; absent ones get +inf.

    Returns:
        Group label -> quantile.
    """
    s = as_scores(scores)
    g = np.asarray(groups)
    if g.shape != s.shape:
        raise ValueError(f"groups length {g.shape} does not match scores {s.shape}")
    present = [label.item() for label in np.unique(g)]
    wanted = list(dict.fromkeys([*present, *(group_ids or [])]))
    return {label: conformal_quantile(s[g == label], alpha) for label in wanted}


def bin_edges(values: ArrayLike, n_bins: int) -> NDArray[np.float64]:
    """Interior equal-mass bin edges (n_bins - 1 of them). Fit on `val_tune` only.

    Args:
        values: Fitting values, finite.
        n_bins: Number of bins (3 for tertiles).

    Returns:
        (n_bins - 1,) increasing edges.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("bin_edges needs a non-empty array of finite values")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    return np.quantile(v, np.arange(1, n_bins) / n_bins)


def assign_bins(values: ArrayLike, edges: ArrayLike) -> NDArray[np.int64]:
    """Bin index 0..len(edges) for each value (right-closed lower bins: v <= edge goes below).

    Values outside the fitted range fall into the first or last bin.
    """
    v = np.asarray(values, dtype=np.float64)
    if np.isnan(v).any():
        raise ValueError("values contain NaN; give failed frames their own group explicitly")
    return np.searchsorted(np.asarray(edges, dtype=np.float64), v, side="left").astype(np.int64)


def product_groups(labels: Sequence[ArrayLike], n_levels: Sequence[int]) -> NDArray[np.int64]:
    """Combine integer group labels into one label per frame, e.g. range tertile x conf tertile.

    The radix is given explicitly (not inferred from the data) so calibration and test frames get the
    same label for the same tuple.

    Args:
        labels: Equal-length integer label arrays, label j in [0, n_levels[j]).
        n_levels: Number of levels of each label array.

    Returns:
        (n,) combined labels in [0, prod(n_levels)).
    """
    if not labels or len(labels) != len(n_levels):
        raise ValueError("need one n_levels entry per (non-empty list of) label arrays")
    arrays = [np.asarray(x, dtype=np.int64) for x in labels]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(f"label arrays differ in length: {[a.shape for a in arrays]}")
    combined = np.zeros(arrays[0].shape, dtype=np.int64)
    for a, levels in zip(arrays, n_levels, strict=True):
        if (a < 0).any() or (a >= levels).any():
            raise ValueError(f"group labels must lie in [0, {levels})")
        combined = combined * levels + a
    return combined
