"""Library oracle: MAPIE v1 split CP must give our quantile on identical residuals.

MAPIE is a test oracle only; nothing under `src/` imports it (`test_no_torch_in_conformal.py`).
TorchCP is not installable alongside P1's numpy pin (docs/DECISIONS.md, 2026-10-06); the second
oracle is the exact-rational brute force in `test_conformal_split.py`.

A `SplitConformalRegressor(prefit=True, conformity_score="absolute")` around an estimator that
always predicts 0 has interval half-width equal to its conformal quantile of |y|.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from poseconf.conformal.split import conformal_quantile, quantile_index

pytestmark = pytest.mark.slow

ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
N_VALUES = (99, 100, 250, 1000, 4798)


def _mapie_half_width(residuals: np.ndarray, alpha: float) -> float:
    regression = pytest.importorskip("mapie.regression")
    from sklearn.dummy import DummyRegressor

    estimator = DummyRegressor(strategy="constant", constant=0.0).fit(np.zeros((2, 1)), [0.0, 0.0])
    model = regression.SplitConformalRegressor(
        estimator=estimator,
        confidence_level=1 - alpha,
        conformity_score="absolute",
        prefit=True,
    )
    model.conformalize(np.zeros((len(residuals), 1)), residuals)
    _, intervals = model.predict_interval(np.zeros((1, 1)))
    return float(intervals[0, 1, 0])


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_matches_mapie_where_mapie_answers(n: int, alpha: float) -> None:
    rng = np.random.default_rng([n, round(alpha * 100)])
    y = rng.standard_normal(n)
    ours = conformal_quantile(np.abs(y), alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            theirs = _mapie_half_width(y, alpha)
        except ValueError as error:
            # MAPIE refuses small n rather than returning +inf; that must coincide with our k > n.
            assert "too low" in str(error)
            assert quantile_index(n, alpha) > n or n <= 1 / alpha, (n, alpha)
            return
    assert theirs == pytest.approx(ours, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(("n", "alpha"), [(2, 0.1), (9, 0.1), (10, 0.05), (18, 0.05)])
def test_small_n_mapie_refuses_we_return_order_statistic_or_inf(n: int, alpha: float) -> None:
    """Recorded behaviour: MAPIE raises for small n; we return the exact order statistic or inf."""
    y = np.random.default_rng(n).standard_normal(n)
    with warnings.catch_warnings(), pytest.raises(ValueError, match="too low"):
        warnings.simplefilter("ignore")
        _mapie_half_width(y, alpha)
    k = quantile_index(n, alpha)
    expected = np.inf if k > n else float(np.sort(np.abs(y))[k - 1])
    assert conformal_quantile(np.abs(y), alpha) == expected
