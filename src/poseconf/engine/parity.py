"""P1 parity statistics for the Phase 3 gate (numpy only).

Compares one dump arm with the P1 sidecar arm it must reproduce, row for row by filename. Every
statistic here is label-free except the `e_r`/`e_t` deltas, which the caller computes only where
invariant 4 allows (synthetic, HIL poolB) and passes in with the frames they cover.

The label-free rotation delta `angle(q̂_dump, q̂_P1)` bounds the error delta on every frame:
by the triangle inequality on SO(3), `|e_r(q̂_dump) − e_r(q̂_P1)| ≤ angle(q̂_dump, q̂_P1)`, and likewise
`|e_t(t̂_dump) − e_t(t̂_P1)| ≤ ‖t̂_dump − t̂_P1‖`. On HIL poolA, where no label may be read, the gate
uses the bound.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["ErrorDeltas", "parity_cell", "quaternion_angle"]


def quaternion_angle(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation angle between unit quaternions, accurate near zero.

    `4·asin(min(‖q1 − q2‖, ‖q1 + q2‖) / 2)` after normalisation. Unlike `2·acos(|<q1, q2>|)`, it
    does not lose half the digits for nearly equal rotations, which is the regime parity lives in.

    Args:
        q1: (..., 4) quaternions.
        q2: (..., 4) quaternions, same component order.

    Returns:
        (...,) angles in radians.
    """
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    chord = np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))
    return 4.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


@dataclass(frozen=True)
class ErrorDeltas:
    """Recomputed vs P1-committed errors on the frames where labels may be read.

    Attributes:
        scope: Which frames these are, e.g. `"all"` (synthetic) or `"poolB"` (HIL).
        filenames: (m,) frames covered.
        e_r_dump: (m,) `e_r` of the dump's pose (rad), NaN where the dump failed.
        e_t_dump: (m,) `e_t` of the dump's pose (m), NaN where the dump failed.
        e_r_p1: (m,) P1's committed `e_r`, NaN where P1 failed.
        e_t_p1: (m,) P1's committed `e_t`, NaN where P1 failed.
    """

    scope: str
    filenames: NDArray[np.str_]
    e_r_dump: NDArray[np.float64]
    e_t_dump: NDArray[np.float64]
    e_r_p1: NDArray[np.float64]
    e_t_p1: NDArray[np.float64]


def _stats(values: NDArray[np.float64]) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0, "median": None, "max": None}
    return {"n": int(values.size), "median": float(np.median(values)), "max": float(values.max())}


def parity_cell(
    dump: Mapping[str, NDArray[Any]],
    p1: Mapping[str, NDArray[Any]],
    *,
    errors: ErrorDeltas | None,
    p1_solved_count: int | None,
    gate: Mapping[str, Any],
    max_listed: int,
) -> dict[str, Any]:
    """Parity statistics and gate verdict for one domain x arm.

    Args:
        dump: Dump arrays (`filename, success, q_pred, t_pred, n_inliers, reprojection_rmse,
            failure_reason`).
        p1: P1 sidecar arrays with the same names (label-free columns), any row order.
        errors: Error deltas where labels may be read, or None.
        p1_solved_count: P1's results-JSON `solved_count` when the dump covers P1's whole frame
            list, else None (a truncated debugging dump compares sidecar rows only).
        gate: `flag_agreement_min`, `median_abs_de_r_max_rad`, `solved_count_exact`.
        max_listed: How many success-flag disagreements to list.

    Returns:
        A JSON-ready dict with counts, agreements, deltas, the bound, and `gate`.

    Raises:
        ValueError: If a dump frame is missing from P1's sidecar, or filenames repeat.
    """
    names = np.asarray(dump["filename"]).astype(str)
    p1_names = np.asarray(p1["filename"]).astype(str)
    if len(set(names)) != len(names) or len(set(p1_names)) != len(p1_names):
        raise ValueError("duplicate filenames in a dump or sidecar")
    position = {name: row for row, name in enumerate(p1_names)}
    missing = [name for name in names if name not in position]
    if missing:
        raise ValueError(f"{len(missing)} dump frames are not in P1's sidecar, e.g. {missing[:3]}")
    rows = np.array([position[name] for name in names], dtype=np.int64)
    p1_success = np.asarray(p1["success"], dtype=bool)[rows]
    dump_success = np.asarray(dump["success"], dtype=bool)
    n = len(names)

    agree = dump_success == p1_success
    both = dump_success & p1_success
    q_dump = np.asarray(dump["q_pred"], dtype=np.float64)[both]
    q_p1 = np.asarray(p1["q_pred"], dtype=np.float64)[rows][both]
    rot_delta = quaternion_angle(q_dump, q_p1) if both.any() else np.zeros(0)
    t_delta = np.linalg.norm(
        np.asarray(dump["t_pred"], dtype=np.float64)[both]
        - np.asarray(p1["t_pred"], dtype=np.float64)[rows][both],
        axis=-1,
    )
    inliers_agree = (
        np.asarray(dump["n_inliers"])[both] == np.asarray(p1["n_inliers"])[rows][both]
    ).astype(bool)

    disagreements = []
    for row in np.flatnonzero(~agree)[:max_listed]:
        disagreements.append(
            {
                "filename": names[row],
                "dump": {
                    "success": bool(dump_success[row]),
                    "n_inliers": int(dump["n_inliers"][row]),
                    "failure_reason": str(dump["failure_reason"][row]),
                },
                "p1": {
                    "success": bool(p1_success[row]),
                    "n_inliers": int(p1["n_inliers"][rows[row]]),
                    "failure_reason": str(p1["failure_reason"][rows[row]]),
                },
            }
        )

    labelled = np.zeros(n, dtype=bool)
    error_block: dict[str, Any] = {"scope": None}
    de_r_median = None
    if errors is not None:
        covered = {name: index for index, name in enumerate(np.asarray(errors.filenames, str))}
        labelled = np.array([name in covered for name in names])
        both_err = np.isfinite(errors.e_r_dump) & np.isfinite(errors.e_r_p1)
        de_r = np.abs(errors.e_r_dump - errors.e_r_p1)[both_err]
        de_t = np.abs(errors.e_t_dump - errors.e_t_p1)[both_err]
        error_block = {
            "scope": errors.scope,
            "n_frames": len(errors.filenames),
            "abs_de_r_rad": _stats(de_r),
            "abs_de_t_m": _stats(de_t),
        }
        de_r_median = error_block["abs_de_r_rad"]["median"]

    # Frames both solved with no label-derived comparison: the label-free bound must carry them.
    unlabelled_both = both & ~labelled
    bound_rot = (
        quaternion_angle(
            np.asarray(dump["q_pred"], dtype=np.float64)[unlabelled_both],
            np.asarray(p1["q_pred"], dtype=np.float64)[rows][unlabelled_both],
        )
        if unlabelled_both.any()
        else np.zeros(0)
    )
    bound_block = {
        "frames": "both solved, no label read",
        "rotation_delta_rad": _stats(bound_rot),
    }

    dump_solved = int(dump_success.sum())
    sidecar_solved = int(p1_success.sum())
    flag_agreement = float(agree.mean()) if n else None
    threshold = float(gate["median_abs_de_r_max_rad"])
    criteria = {
        "flag_agreement": flag_agreement is not None
        and flag_agreement >= float(gate["flag_agreement_min"]),
        "median_abs_de_r": de_r_median is not None and de_r_median <= threshold
        if errors is not None
        else None,
        "median_rotation_bound_unlabelled": (
            bound_block["rotation_delta_rad"]["median"] <= threshold if bound_rot.size else None
        ),
        "solved_count_vs_p1_json": (
            dump_solved == p1_solved_count
            if (p1_solved_count is not None and bool(gate["solved_count_exact"]))
            else None
        ),
    }
    met = all(value for value in criteria.values() if value is not None) and any(
        value is not None
        for value in (criteria["median_abs_de_r"], criteria["median_rotation_bound_unlabelled"])
    )
    return {
        "n_total": n,
        "solved": {
            "dump": dump_solved,
            "p1_sidecar_same_frames": sidecar_solved,
            "p1_json": p1_solved_count,
            "dump_rate": dump_solved / n if n else None,
            "p1_json_rate": p1_solved_count / n if (p1_solved_count is not None and n) else None,
        },
        "success_flag_agreement": flag_agreement,
        "n_flag_disagreements": int((~agree).sum()),
        "disagreements": disagreements,
        "both_solved": int(both.sum()),
        "rotation_delta_rad": _stats(rot_delta),
        "translation_delta_m": _stats(t_delta),
        "n_inliers_agreement": float(inliers_agree.mean()) if inliers_agree.size else None,
        "errors": error_block,
        "label_free_bound": bound_block,
        "gate": {"criteria": criteria, "met": bool(met)},
    }
