"""Level C: scores from the learned keypoint covariance (C1, C2), and the head's quality report.

Data roles (CLAUDE.md invariants 3-5):
* The variance head was trained on synthetic `train` and selected on `val_tune` (Phase 5 training).
  Here `val_tune` is read once more, only to fit the single global scale of the head-quality
  report (an NLL diagnostic, never a score input). `val_cal` calibrates; `val_test` evaluates.
* C1 is B2's Mahalanobis code on the head's Σ̂ₖ (`vhead_cov_full`), with B2's empty-channel rule.
* C2 = max(E_R / σ̂_R, ‖t̂ − t‖ / σ̂_t), σ̂ from C1's Σ̂ₖ propagated through the PnP Jacobian at the
  estimate (Phase 4's linearisation, keypoints visible under the estimate and constrained):
  σ̂_R = √λ_max((H⁻¹)_ωω) rad, σ̂_t = √λ_max((H⁻¹)_tt) m. Predictions only. A solved frame whose
  σ̂ is undefined (|U| < 3, singular H) is *invalid* for C2: it abstains (`abstain_allowed`) or
  scores +∞ (`answer_required`), never a finite placeholder (docs/SCORES.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from poseconf.conformal.propagate import LinearisedExtent, crop_cov_to_full
from poseconf.conformal.scores import SCORES, PoseSet
from poseconf.engine.head_metrics import head_quality

__all__ = [
    "LEVEL_C_SCORES",
    "PoseSigma",
    "attach_variance",
    "c2_scores",
    "c2_set",
    "check_variance_mapping",
    "crop_residuals",
    "head_quality_report",
    "pose_sigma",
]

#: The Level C scores.
LEVEL_C_SCORES = ("C1", "C2")


def attach_variance(
    dump: dict[str, NDArray[Any]], sidecar: dict[str, NDArray[Any]]
) -> dict[str, NDArray[Any]]:
    """The dump with the variance sidecar's columns added, after checking they describe it.

    Raises:
        ValueError: If frame order or P1 coordinates differ between dump and sidecar.
    """
    if not np.array_equal(dump["filename"], sidecar["filename"]):
        raise ValueError("variance sidecar frame order differs from the dump")
    if not np.array_equal(dump["kp_pred_crop"], sidecar["kp_pred_crop"]):
        raise ValueError("variance sidecar coordinates differ from the dump (invariant 10)")
    return {
        **dump,
        "vhead_chol_crop": sidecar["vhead_chol_crop"],
        "vhead_cov_crop": sidecar["vhead_cov_crop"],
        "vhead_cov_full": sidecar["vhead_cov_full"],
    }


def check_variance_mapping(dump: dict[str, NDArray[Any]]) -> float:
    """Max relative difference of `vhead_cov_full` vs `propagate.crop_cov_to_full` (independent)."""
    mapped = crop_cov_to_full(dump["vhead_cov_crop"], dump["affine"])
    stored = np.asarray(dump["vhead_cov_full"], dtype=np.float64)
    scale = np.abs(stored).max(axis=(-2, -1), keepdims=True)
    return float(np.max(np.abs(mapped - stored) / scale))


def crop_residuals(
    dump: dict[str, NDArray[Any]], labels: dict[str, NDArray[Any]], names: list[str]
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.int64]]:
    """Prediction − label keypoints in crop px for `names` (labels mapped by the dump's affine).

    Returns:
        `(residual (n, K, 2), in_frame (n, K), dump rows (n,))`.
    """
    d_index = {str(f): i for i, f in enumerate(dump["filename"])}
    l_index = {str(f): i for i, f in enumerate(labels["filename"])}
    d = np.array([d_index[name] for name in names], dtype=np.int64)
    lab = np.array([l_index[name] for name in names], dtype=np.int64)
    affine = np.asarray(dump["affine"], dtype=np.float64)[d]
    gt_full = np.asarray(labels["kp_gt_full"], dtype=np.float64)[lab]
    gt_crop = np.einsum("nij,nkj->nki", affine[:, :, :2], gt_full) + affine[:, None, :, 2]
    residual = np.asarray(dump["kp_pred_crop"], dtype=np.float64)[d] - gt_crop
    return residual, np.asarray(labels["in_frame"], dtype=bool)[lab], d


def head_quality_report(
    dump: dict[str, NDArray[Any]],
    labels: dict[str, NDArray[Any]],
    *,
    fit_names: list[str],
    test_names: list[str],
) -> dict[str, Any]:
    """Learned Σ̂ vs heatmap-moment Σ̂ (B2's) on predicted crops: NLL, reliability, ranking.

    Keypoints counted: in frame (P1's label visibility) and not an empty heatmap channel, on every
    frame (solved or not). The global rescaling factor of each source is fitted on `fit_names`
    (`val_tune`) and applied unchanged on `test_names` (`val_test`).
    """
    out: dict[str, Any] = {
        "units": "crop px (predicted-box crops)",
        "keypoints_counted": "in_frame and not heatmap_empty, all frames",
    }
    scales: dict[str, float] = {}
    for split, names in (("val_tune", fit_names), ("val_test", test_names)):
        residual, in_frame, rows = crop_residuals(dump, labels, names)
        mask = in_frame & ~np.asarray(dump["heatmap_empty"], dtype=bool)[rows]
        confidence = np.asarray(dump["confidence"], dtype=np.float64)[rows]
        block = {}
        for source, key in (("learned", "vhead_cov_crop"), ("heatmap_moment", "heatmap_cov_crop")):
            cov = np.asarray(dump[key], dtype=np.float64)[rows]
            block[source] = head_quality(
                cov,
                residual,
                mask,
                scale=scales.get(source),
                confidence=confidence if source == "learned" else None,
            )
            scales.setdefault(source, block[source]["global_scale"])
        out[split] = block
    return out


@dataclass(frozen=True)
class PoseSigma:
    """C2's per-frame predicted pose standard deviations.

    Attributes:
        sigma_R: (n,) radians (NaN where undefined).
        sigma_t: (n,) metres (NaN where undefined).
        ok: (n,) solved and both finite and positive.
        n_used: (n,) keypoints in the information matrix (0 on failed frames).
    """

    sigma_R: NDArray[np.float64]
    sigma_t: NDArray[np.float64]
    ok: NDArray[np.bool_]
    n_used: NDArray[np.int64]


def pose_sigma(extents: list[LinearisedExtent | None]) -> PoseSigma:
    """σ̂_R = √rot_var, σ̂_t = √trans_var of each frame's q-free linearisation (C1's Σ̂ as shapes)."""
    n = len(extents)
    sigma_r, sigma_t = np.full(n, np.nan), np.full(n, np.nan)
    n_used = np.zeros(n, dtype=np.int64)
    for i, extent in enumerate(extents):
        if extent is None:
            continue
        n_used[i] = extent.n_used
        if math.isfinite(extent.rot_var) and math.isfinite(extent.trans_var):
            sigma_r[i], sigma_t[i] = math.sqrt(extent.rot_var), math.sqrt(extent.trans_var)
    ok = np.isfinite(sigma_r) & np.isfinite(sigma_t) & (sigma_r > 0) & (sigma_t > 0)
    return PoseSigma(sigma_R=sigma_r, sigma_t=sigma_t, ok=ok, n_used=n_used)


def c2_scores(frames: Any, sigma: PoseSigma, convention: str) -> NDArray[np.float64]:
    """C2 scores (labels used); frames without a defined σ̂ follow the failure convention."""
    valid = frames.valid & sigma.ok
    return SCORES["C2"].score_fn(
        frames.q_hat,
        frames.t_hat,
        frames.q_gt,
        frames.t_gt,
        valid,
        sigma_R=np.where(valid, sigma.sigma_R, 1.0),
        sigma_t=np.where(valid, sigma.sigma_t, 1.0),
        convention=convention,
    )


def c2_set(frames: Any, sigma: PoseSigma, q: float) -> PoseSet:
    """The test-time C2 set: predictions, σ̂ and q only — no label is passed."""
    valid = frames.valid & sigma.ok
    return SCORES["C2"].set_fn(
        frames.q_hat,
        frames.t_hat,
        valid,
        q,
        sigma_R=np.where(valid, sigma.sigma_R, 1.0),
        sigma_t=np.where(valid, sigma.sigma_t, 1.0),
    )
