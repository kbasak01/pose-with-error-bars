"""Quality of a predicted keypoint covariance against observed residuals (numpy, Phase 5).

Used for the variance head's selection diagnostics (`val_tune`) and, once selection is frozen, for
the Level C report (`val_tune`, `val_test`). The same functions score the heatmap-moment baseline
(B2's Σ̂) so the comparison is like for like. Nothing here is a conformal guarantee: the 1σ/2σ
ellipse coverage is the *uncalibrated* reliability of a Gaussian reading of Σ̂ (≈ 0.393 / 0.865 if
the residuals were exactly that Gaussian).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

__all__ = [
    "GAUSSIAN_2D_COVERAGE",
    "gaussian_nll",
    "global_scale",
    "head_quality",
    "mahalanobis_sq",
    "sigma_scalar",
]

#: Probability that a 2-D Gaussian residual lies inside its 1σ and 2σ Mahalanobis ellipse.
GAUSSIAN_2D_COVERAGE = {"1sigma": 1.0 - math.exp(-0.5), "2sigma": 1.0 - math.exp(-2.0)}

_LOG_TWO_PI = math.log(2.0 * math.pi)


def mahalanobis_sq(cov: NDArray[np.float64], residual: NDArray[np.float64]) -> NDArray[np.float64]:
    """`rᵀ Σ⁻¹ r` for (n, 2, 2) covariances and (n, 2) residuals."""
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = a * c - b * b
    x, y = residual[:, 0], residual[:, 1]
    return (c * x * x - 2.0 * b * x * y + a * y * y) / det


def gaussian_nll(cov: NDArray[np.float64], residual: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-keypoint 2-D Gaussian NLL, `½(rᵀΣ⁻¹r + log det Σ) + log 2π` (nats)."""
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
    return 0.5 * (mahalanobis_sq(cov, residual) + np.log(det)) + _LOG_TWO_PI


def sigma_scalar(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scalar σ̂ of a 2-D covariance: `det(Σ)^{1/4}`, the geometric-mean axis length."""
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
    return det**0.25


def global_scale(cov: NDArray[np.float64], residual: NDArray[np.float64]) -> float:
    """The single factor s minimising the mean NLL of `s·Σ`: `s* = mean(rᵀΣ⁻¹r) / 2`."""
    return float(np.mean(mahalanobis_sq(cov, residual)) / 2.0)


def head_quality(
    cov: NDArray[np.float64],
    residual: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    scale: float | None = None,
    confidence: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """NLL, ellipse reliability and error ranking of one covariance source on one split.

    Args:
        cov: (n, K, 2, 2) covariances; must be positive definite where `mask`.
        residual: (n, K, 2) prediction − truth, same units as `cov`.
        mask: (n, K) keypoints that count (labelled, and not an empty heatmap channel).
        scale: A global factor fitted elsewhere (on `val_tune`) applied for the rescaled NLL; None
            fits it on this split (only legitimate on the split it is reported as fitted on).
        confidence: (n, K) P1 confidence, for its own Spearman against the error (higher
            confidence should mean lower error, so its rank correlation is reported negated).

    Returns:
        A JSON-ready mapping.

    Raises:
        ValueError: If a masked covariance is not positive definite or nothing is masked in.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("no keypoint is masked in")
    c, r = cov[mask], residual[mask]
    det = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] * c[:, 1, 0]
    if not (np.all(c[:, 0, 0] > 0) and np.all(det > 0)):
        raise ValueError("a masked covariance is not positive definite")
    m2 = mahalanobis_sq(c, r)
    nll = gaussian_nll(c, r)
    fitted_here = scale is None
    s = global_scale(c, r) if fitted_here else float(scale)
    error = np.linalg.norm(r, axis=1)
    sigma = sigma_scalar(c)
    keypoints = np.nonzero(mask)[1]
    per_keypoint = []
    for k in range(cov.shape[1]):
        chosen = keypoints == k
        rho = spearmanr(sigma[chosen], error[chosen]).statistic if chosen.sum() > 2 else math.nan
        per_keypoint.append(None if math.isnan(rho) else float(rho))
    out: dict[str, Any] = {
        "n_keypoints": int(mask.sum()),
        "n_frames": int(mask.any(axis=1).sum()),
        "nll_mean": float(nll.mean()),
        "nll_median": float(np.median(nll)),
        "global_scale": s,
        "global_scale_fitted_on_this_split": fitted_here,
        "nll_mean_rescaled": float(gaussian_nll(c * s, r).mean()),
        "coverage_1sigma": float(np.mean(m2 <= 1.0)),
        "coverage_2sigma": float(np.mean(m2 <= 4.0)),
        "coverage_1sigma_rescaled": float(np.mean(m2 / s <= 1.0)),
        "coverage_2sigma_rescaled": float(np.mean(m2 / s <= 4.0)),
        "gaussian_reference": GAUSSIAN_2D_COVERAGE,
        "sigma_px": {
            "p05": float(np.quantile(sigma, 0.05)),
            "median": float(np.median(sigma)),
            "p95": float(np.quantile(sigma, 0.95)),
        },
        "error_px_median": float(np.median(error)),
        "spearman_sigma_vs_error": float(spearmanr(sigma, error).statistic),
        "spearman_sigma_vs_error_per_keypoint": per_keypoint,
    }
    if confidence is not None:
        conf = np.asarray(confidence)[mask]
        out["spearman_neg_confidence_vs_error"] = float(spearmanr(-conf, error).statistic)
    return out
