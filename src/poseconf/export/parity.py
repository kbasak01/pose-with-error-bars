"""PyTorch-vs-ONNX Runtime parity of the variance graph: tensor, covariance and set level. Phase 7.

The variance outputs are gated on their own (IMPLEMENTATION_PLAN Phase 7 step 2; checklist G2):
`cov_chol` and the derived covariance Σ̂ = L Lᵀ to 1e-4 max abs in fp32 with TF32 off on *both*
sides, `keypoint_empty` exactly, and fp16 against P1's stated fp16 relaxation. `coords` and
`confidence` are P1's outputs; P1's own keypoint parity gate is unmet (bistable two-pass
soft-argmax), so their deltas are reported beside P1's committed record as an inherited miss (G3),
not gated here.

Because the head's readout uses P1's refine window, a soft-argmax flip on one keypoint can move its
covariance too. `cov_chol` is therefore also summarised on keypoints whose coordinates agree to
`COORD_STABLE_PX` (a diagnostic, not the gate). The set level is what ships: the C1 radius
q·max_k sqrt(λ_max Σ̂_k) per frame, in crop px. The crop affine is a similarity, so the relative
difference is the same in full-frame px.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "COORD_STABLE_PX",
    "FP32_VARIANCE_GATE",
    "c1_radius",
    "chol_to_cov",
    "difference_stats",
    "gate_status",
    "parity_record",
    "relative_stats",
]

#: Max abs gate for the variance outputs in fp32 (TF32 off on both sides), the plan's 1e-4.
FP32_VARIANCE_GATE = 1.0e-4

#: A keypoint whose coordinate differs by at most this many crop px counts as decode-stable.
COORD_STABLE_PX = 1.0e-3


def difference_stats(reference: NDArray[Any], other: NDArray[Any]) -> dict[str, float]:
    """max / mean / p99 of |other - reference| over every element."""
    diff = np.abs(np.asarray(other, np.float64) - np.asarray(reference, np.float64)).reshape(-1)
    if diff.size == 0:
        return {"max": float("nan"), "mean": float("nan"), "p99": float("nan"), "n": 0}
    return {
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "p99": float(np.percentile(diff, 99.0)),
        "n": int(diff.size),
    }


def relative_stats(reference: NDArray[Any], other: NDArray[Any]) -> dict[str, float]:
    """max / p99 of |other - reference| / |reference| over finite, non-zero references."""
    ref = np.asarray(reference, np.float64).reshape(-1)
    oth = np.asarray(other, np.float64).reshape(-1)
    ok = np.isfinite(ref) & np.isfinite(oth) & (ref != 0)
    rel = np.abs(oth[ok] - ref[ok]) / np.abs(ref[ok])
    if rel.size == 0:
        return {"max": float("nan"), "p99": float("nan"), "n": 0}
    return {"max": float(rel.max()), "p99": float(np.percentile(rel, 99.0)), "n": int(rel.size)}


def chol_to_cov(chol: NDArray[Any]) -> NDArray[np.float64]:
    """(..., 3) lower Cholesky factors `(l0, l1, l2)` -> (..., 2, 2) `L Lᵀ` (as the head builds Σ̂)."""
    c = np.asarray(chol, dtype=np.float64)
    l0, l1, l2 = c[..., 0], c[..., 1], c[..., 2]
    cxx, cxy, cyy = l0 * l0, l0 * l1, l1 * l1 + l2 * l2
    return np.stack([np.stack([cxx, cxy], -1), np.stack([cxy, cyy], -1)], -2)


def c1_radius(chol: NDArray[Any], empty: NDArray[np.bool_], q: float) -> NDArray[np.float64]:
    """(n,) C1 set radius in crop px: q · max over non-empty keypoints of sqrt(λ_max Σ̂_k).

    +inf when a frame has an empty keypoint (its set is the whole image), as `KeypointSet.radius_px`.
    """
    lam = np.linalg.eigvalsh(chol_to_cov(chol))[..., -1]
    lam = np.where(empty, -np.inf, lam)
    radius = q * np.sqrt(np.maximum(lam.max(axis=-1), 0.0))
    return np.where(np.asarray(empty).any(axis=-1), np.inf, radius)


def gate_status(record: dict[str, Any], fp16_tensor_max_abs: float) -> dict[str, Any]:
    """Gates on the variance outputs only; P1's coords gate is listed as inherited, not scored."""
    gates: dict[str, dict[str, Any]] = {}
    tensors = record["tensor_parity"]
    arms = [
        (tensors.get("ort_fp32"), "ort_fp32", FP32_VARIANCE_GATE),
        (tensors.get("ort_fp16"), "ort_fp16", fp16_tensor_max_abs),
        (record.get("cpu_reference", {}).get("tensor_parity"), "cpu_reference", FP32_VARIANCE_GATE),
    ]
    for block, backend, threshold in arms:
        if block is None:
            continue
        for key in ("cov_chol", "cov"):
            measured = block[key]["max"]
            gates[f"{backend}.tensor.{key}"] = {
                "measured": measured,
                "threshold": threshold,
                "met": bool(measured < threshold),
            }
        mismatches = block["keypoint_empty"]["mismatches"]
        gates[f"{backend}.keypoint_empty.mismatches"] = {
            "measured": mismatches,
            "threshold": 0,
            "met": mismatches == 0,
        }
    failed = sorted(name for name, gate in gates.items() if not gate["met"])
    return {
        "gates": gates,
        "failed": failed,
        "variance_outputs_fp32_met": not any(n.startswith("ort_fp32") for n in failed),
        "all_met": not failed,
        "inherited": {
            "p1_keypoint_parity": "unmet",
            "note": (
                "coords/confidence are P1's outputs; P1's keypoint parity gate is unmet "
                "(bistable two-pass soft-argmax). Their deltas here are reported beside P1's "
                "committed record, not gated."
            ),
        },
    }


def parity_record(
    stacked: dict[str, dict[str, NDArray[Any]]], *, c1_quantile: float
) -> dict[str, Any]:
    """Tensor, covariance, decode-stable and set-level parity of each ORT backend vs `torch_fp32`.

    Args:
        stacked: backend -> output name -> stacked array; must hold `torch_fp32`.
        c1_quantile: The C1 artifact's quantile, for the set-level comparison.

    Returns:
        `tensor_parity`, `decode_stable`, `set_level` blocks keyed by backend.
    """
    ref = stacked["torch_fp32"]
    ref_cov = chol_to_cov(ref["cov_chol"])
    ref_radius = c1_radius(ref["cov_chol"], ref["keypoint_empty"], c1_quantile)
    ref_sigma = np.sqrt(np.linalg.eigvalsh(ref_cov)[..., -1])
    tensor, stable, sets = {}, {}, {}
    for backend, out in stacked.items():
        if backend == "torch_fp32":
            continue
        cov = chol_to_cov(out["cov_chol"])
        empty_equal = out["keypoint_empty"].astype(bool) == ref["keypoint_empty"].astype(bool)
        tensor[backend] = {
            "coords": difference_stats(ref["coords"], out["coords"]),
            "confidence": difference_stats(ref["confidence"], out["confidence"]),
            "cov_chol": difference_stats(ref["cov_chol"], out["cov_chol"]),
            "cov_chol_relative": relative_stats(ref["cov_chol"], out["cov_chol"]),
            "cov": difference_stats(ref_cov, cov),
            "sigma_max_relative": relative_stats(
                ref_sigma, np.sqrt(np.linalg.eigvalsh(cov)[..., -1])
            ),
            "keypoint_empty": {
                "mismatches": int((~empty_equal).sum()),
                "n": int(empty_equal.size),
                "n_empty_reference": int(ref["keypoint_empty"].astype(bool).sum()),
            },
        }
        coord_diff = np.abs(out["coords"] - ref["coords"]).max(axis=-1)
        keep = coord_diff <= COORD_STABLE_PX
        stable[backend] = {
            "threshold_px": COORD_STABLE_PX,
            "keypoints_stable": int(keep.sum()),
            "keypoints_total": int(keep.size),
            "cov_chol_on_stable": difference_stats(ref["cov_chol"][keep], out["cov_chol"][keep]),
            "cov_chol_on_unstable": difference_stats(
                ref["cov_chol"][~keep], out["cov_chol"][~keep]
            ),
        }
        radius = c1_radius(out["cov_chol"], out["keypoint_empty"].astype(bool), c1_quantile)
        both_finite = np.isfinite(radius) & np.isfinite(ref_radius)
        sets[backend] = {
            "score": "C1",
            "quantile": c1_quantile,
            "units": "crop px (relative difference is unit-free: the crop affine is a similarity)",
            "radius_relative": relative_stats(ref_radius[both_finite], radius[both_finite]),
            "radius_abs_px": difference_stats(ref_radius[both_finite], radius[both_finite]),
            "frames_infinite_mismatch": int((np.isinf(radius) != np.isinf(ref_radius)).sum()),
        }
    return {"tensor_parity": tensor, "decode_stable": stable, "set_level": sets}
