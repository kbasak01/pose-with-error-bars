"""Level A: pose-score conformal sets (A1-A3) from P1's committed per-sample sidecars.

Data roles (CLAUDE.md invariants 3 and 5):
* `fit_level_a` sees `val_tune` only: normalisers `c_R, c_t, c_z, c_xy` (medians on solved frames)
  and the A3 difficulty `g(conf)` (decreasing isotonic fit of the A1 score on mean keypoint
  confidence).
* Scores use labels; sets (`pose_sets`) use predictions, the fitted inputs and q only.
* `resplit_draws` re-partitions `val_cal` + `val_test` only; `val_tune` never enters the pool.

This module is synthetic-only in Phase 2: frames come with labels from
`p1_adapter.load_synthetic_labels`, which cannot load a HIL domain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

from poseconf.conformal.metrics import (
    beta_parameters,
    ks_against_beta,
    ks_against_pmf,
    mc_standard_error,
    pmf_mean,
    resplit_coverage_law,
    set_size_summary,
)
from poseconf.conformal.scores import (
    SCORES,
    PoseSet,
    median_normaliser,
    pose_error_components,
    score_a1,
)
from poseconf.conformal.so3 import geodesic_distance
from poseconf.conformal.split import CONVENTIONS, quantile_index

__all__ = [
    "LEVEL_A_SCORES",
    "Frames",
    "LevelAFit",
    "ResplitDraws",
    "check_sidecar_errors",
    "join_frames",
    "fit_level_a",
    "pose_set",
    "resplit_draws",
    "score_frames",
    "set_size_report",
    "summarise_draws",
]

#: The pose scores this module handles.
LEVEL_A_SCORES = ("A1", "A2", "A3")

_RAD_TO_DEG = 180.0 / math.pi


@dataclass(frozen=True)
class Frames:
    """Per-frame predictions joined to synthetic labels, in a fixed filename order.

    Attributes:
        filenames: (n,) image filenames.
        q_hat: (n, 4) predicted quaternions (P1 order), NaN on PnP failure.
        t_hat: (n, 3) predicted translations, metres, NaN on PnP failure.
        valid: (n,) PnP solved (P1 `success`).
        conf: (n,) predicted mean keypoint confidence (finite on every frame).
        q_gt: (n, 4) label quaternions (same order). Label.
        t_gt: (n, 3) label translations. Label.
        e_r: (n,) P1's committed rotation error (rad), for the join check.
        e_t: (n,) P1's committed translation error (m), for the join check.
    """

    filenames: NDArray[np.str_]
    q_hat: NDArray[np.float64]
    t_hat: NDArray[np.float64]
    valid: NDArray[np.bool_]
    conf: NDArray[np.float64]
    q_gt: NDArray[np.float64]
    t_gt: NDArray[np.float64]
    e_r: NDArray[np.float64]
    e_t: NDArray[np.float64]

    def __len__(self) -> int:
        """Number of frames."""
        return len(self.filenames)

    def subset(self, index: NDArray[np.int64]) -> Frames:
        """Rows `index`, in that order."""
        return Frames(**{name: getattr(self, name)[index] for name in self.__dataclass_fields__})


def join_frames(
    sidecar: dict[str, NDArray[Any]],
    label_filenames: NDArray[np.str_],
    q_gt: NDArray[np.float64],
    t_gt: NDArray[np.float64],
    names: Sequence[str],
) -> Frames:
    """Select `names` from a sidecar and its labels; every name must appear exactly once in both.

    Args:
        sidecar: P1 sidecar columns (`filename`, `success`, `q_pred`, `t_pred`,
            `confidence_mean`, `e_r`, `e_t`).
        label_filenames: (L,) label filenames.
        q_gt: (L, 4) label quaternions.
        t_gt: (L, 3) label translations.
        names: Filenames to select (a split manifest).

    Returns:
        The joined frames, in `names` order.

    Raises:
        ValueError: On duplicates or a name missing from either source.
    """

    def index_of(column: NDArray[np.str_], source: str) -> NDArray[np.int64]:
        position = {str(name): i for i, name in enumerate(column)}
        if len(position) != len(column):
            raise ValueError(f"{source} has duplicate filenames")
        missing = [name for name in names if name not in position]
        if missing:
            raise ValueError(f"{len(missing)} names missing from {source}, e.g. {missing[:3]}")
        return np.array([position[name] for name in names], dtype=np.int64)

    if len(set(names)) != len(names):
        raise ValueError("split names must be unique")
    s = index_of(sidecar["filename"], "sidecar")
    lab = index_of(label_filenames, "labels")
    return Frames(
        filenames=np.asarray(names),
        q_hat=sidecar["q_pred"][s].astype(np.float64),
        t_hat=sidecar["t_pred"][s].astype(np.float64),
        valid=sidecar["success"][s].astype(bool),
        conf=sidecar["confidence_mean"][s].astype(np.float64),
        q_gt=np.asarray(q_gt, dtype=np.float64)[lab],
        t_gt=np.asarray(t_gt, dtype=np.float64)[lab],
        e_r=sidecar["e_r"][s].astype(np.float64),
        e_t=sidecar["e_t"][s].astype(np.float64),
    )


def check_sidecar_errors(frames: Frames, *, tol_rad: float, tol_m: float) -> dict[str, float]:
    """Recompute E_R and ||dt|| from predictions + labels and compare with P1's committed values.

    Agreement proves the filename join and the quaternion order. Solved frames only (P1 stores
    errors only where PnP succeeded).

    Returns:
        `max_abs_diff_e_r_rad`, `max_abs_diff_e_t_m`, `n_checked`.

    Raises:
        ValueError: If either difference exceeds its tolerance, or a frame has no point estimate
            but no failure flag.
    """
    ok = frames.valid
    if not np.all(np.isfinite(frames.q_hat[ok])) or not np.all(np.isfinite(frames.conf)):
        raise ValueError("a solved frame has a non-finite prediction, or conf is non-finite")
    e_r = geodesic_distance(frames.q_hat[ok], frames.q_gt[ok])
    e_t = np.linalg.norm(frames.t_hat[ok] - frames.t_gt[ok], axis=-1)
    report = {
        "max_abs_diff_e_r_rad": float(np.max(np.abs(e_r - frames.e_r[ok]))),
        "max_abs_diff_e_t_m": float(np.max(np.abs(e_t - frames.e_t[ok]))),
        "n_checked": int(ok.sum()),
    }
    if report["max_abs_diff_e_r_rad"] > tol_rad or report["max_abs_diff_e_t_m"] > tol_m:
        raise ValueError(f"sidecar errors not reproduced from labels: {report}")
    return report


@dataclass(frozen=True)
class LevelAFit:
    """Everything fitted on `val_tune`.

    Attributes:
        c_R: Median E_R (rad) on solved val_tune frames.
        c_t: Median ||dt|| / ||t_hat||.
        c_z: Median |dz| / ||t_hat||.
        c_xy: Median ||dxy|| / ||t_hat||.
        g_x: Isotonic knots, mean keypoint confidence (increasing).
        g_y: Isotonic knot values of g (> 0, non-increasing in confidence).
        n_fit: Solved val_tune frames used.
    """

    c_R: float
    c_t: float
    c_z: float
    c_xy: float
    g_x: NDArray[np.float64]
    g_y: NDArray[np.float64]
    n_fit: int

    def difficulty(self, conf: NDArray[np.float64]) -> NDArray[np.float64]:
        """g(conf) by linear interpolation between knots, flat outside (sklearn's `clip`)."""
        return np.interp(np.asarray(conf, dtype=np.float64), self.g_x, self.g_y)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "c_R_rad": self.c_R,
            "c_t": self.c_t,
            "c_z": self.c_z,
            "c_xy": self.c_xy,
            "g": {
                "kind": "isotonic_decreasing_A1_on_confidence_mean",
                "x": self.g_x.tolist(),
                "y": self.g_y.tolist(),
            },
            "n_fit_solved": self.n_fit,
        }


def fit_level_a(tune: Frames) -> LevelAFit:
    """Fit the Level A normalisers and g(conf) on `val_tune` solved frames.

    Args:
        tune: The `val_tune` frames. Never pass `val_cal` or `val_test`.

    Returns:
        The fit.
    """
    ok = tune.valid
    comp = pose_error_components(tune.q_hat[ok], tune.t_hat[ok], tune.q_gt[ok], tune.t_gt[ok])
    every = np.ones(int(ok.sum()), dtype=bool)
    c = {key: median_normaliser(comp[key], every) for key in comp}
    a1 = score_a1(
        tune.q_hat[ok],
        tune.t_hat[ok],
        tune.q_gt[ok],
        tune.t_gt[ok],
        every,
        c_R=c["rot"],
        c_t=c["trans_rel"],
        convention="answer_required",  # every frame is solved; the convention is moot here
    )
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(tune.conf[ok], a1)
    g_x = np.asarray(iso.X_thresholds_, dtype=np.float64)
    g_y = np.asarray(iso.y_thresholds_, dtype=np.float64)
    if not np.all(g_y > 0.0) or not np.all(np.isfinite(g_y)):
        raise ValueError("isotonic g(conf) must be finite and > 0 at every knot")
    return LevelAFit(
        c_R=c["rot"],
        c_t=c["trans_rel"],
        c_z=c["boresight_rel"],
        c_xy=c["lateral_rel"],
        g_x=g_x,
        g_y=g_y,
        n_fit=int(ok.sum()),
    )


def _score_kwargs(score_id: str, fit: LevelAFit, conf: NDArray[np.float64]) -> dict[str, Any]:
    if score_id == "A1":
        return {"c_R": fit.c_R, "c_t": fit.c_t}
    if score_id == "A2":
        return {"c_R": fit.c_R, "c_z": fit.c_z, "c_xy": fit.c_xy}
    if score_id == "A3":
        return {"c_R": fit.c_R, "c_t": fit.c_t, "difficulty": fit.difficulty(conf)}
    raise ValueError(f"not a Level A score: {score_id!r}; expected one of {LEVEL_A_SCORES}")


def score_frames(score_id: str, frames: Frames, fit: LevelAFit, convention: str) -> NDArray:
    """Nonconformity scores (uses labels), +/-inf on PnP failure per `convention`."""
    spec = SCORES[score_id]
    return spec.score_fn(
        frames.q_hat,
        frames.t_hat,
        frames.q_gt,
        frames.t_gt,
        frames.valid,
        convention=convention,
        **_score_kwargs(score_id, fit, frames.conf),
    )


def pose_set(score_id: str, frames: Frames, fit: LevelAFit, q: float) -> PoseSet:
    """The test-time set: predictions, fitted inputs and q only — no label is passed."""
    return SCORES[score_id].set_fn(
        frames.q_hat, frames.t_hat, frames.valid, q, **_score_kwargs(score_id, fit, frames.conf)
    )


def set_size_report(pset: PoseSet, answered: NDArray[np.bool_]) -> dict[str, Any] | None:
    """Median / p90 set radii over answered frames; None when no frame is answered.

    Rotation in degrees; translation in metres and as a fraction of the predicted range ||t_hat||
    (ball: `trans`; cylinder: `boresight` half-length along z and `lateral` radius).
    """
    if not answered.any():
        return None
    pred_range = np.linalg.norm(pset.t_center[answered], axis=-1)
    trans = pset.trans_radius[answered]
    out: dict[str, Any] = {"rot_deg": set_size_summary(pset.rot_radius[answered] * _RAD_TO_DEG)}
    parts = ("trans",) if pset.translation == "ball" else ("boresight", "lateral")
    for column, part in enumerate(parts):
        out[f"{part}_frac"] = set_size_summary(trans[:, column] / pred_range)
        out[f"{part}_m"] = set_size_summary(trans[:, column])
    return out


@dataclass(frozen=True)
class ResplitDraws:
    """Per-draw outcomes of R random cal/test re-partitions of one pool, one (score, convention).

    Arrays are (len(alphas), R).
    """

    alphas: tuple[float, ...]
    n_cal: int
    n_test: int
    q: NDArray[np.float64]
    n_covered: NDArray[np.int64]
    answer_rate: NDArray[np.float64]
    silent_failure_rate: NDArray[np.float64]
    median_rot_deg: NDArray[np.float64]

    @property
    def coverage(self) -> NDArray[np.float64]:
        """Covered fraction per draw."""
        return self.n_covered / self.n_test


def resplit_draws(
    scores: NDArray[np.float64],
    valid: NDArray[np.bool_],
    rot_scale: NDArray[np.float64],
    perms: NDArray[np.int64],
    n_cal: int,
    alphas: Sequence[float],
    convention: str,
) -> ResplitDraws:
    """Split CP on each re-partition: first `n_cal` of each permutation calibrate, the rest test.

    The quantile is the k-th smallest calibration score, k = `quantile_index(n_cal, alpha)`
    (+inf when k > n_cal), identical to `conformal_quantile`; outcomes follow `metrics.outcomes`.

    Args:
        scores: (N,) pool scores under `convention`.
        valid: (N,) PnP solved.
        rot_scale: (N,) radians per unit score (c_R, times g for A3); used for set size only.
        perms: (R, N) permutations of the pool (shared across scores: common random numbers).
        n_cal: Calibration size.
        alphas: Miscoverage levels.
        convention: PnP-failure convention the scores were computed under.

    Returns:
        The per-draw arrays.
    """
    fail = CONVENTIONS[convention]
    if np.any(scores[~valid] != fail) or not np.all(np.isfinite(scores[valid])):
        raise ValueError(f"scores inconsistent with convention {convention!r}")
    cal, test = perms[:, :n_cal], perms[:, n_cal:]
    cal_sorted = np.sort(scores[cal], axis=1)
    s_test, v_test, r_test = scores[test], valid[test], rot_scale[test]
    n_test = test.shape[1]
    shape = (len(alphas), perms.shape[0])
    q_all = np.empty(shape)
    covered_all = np.empty(shape, dtype=np.int64)
    answer_all, silent_all, rot_all = np.empty(shape), np.empty(shape), np.empty(shape)
    for i, alpha in enumerate(alphas):
        k = quantile_index(n_cal, alpha)
        q = cal_sorted[:, k - 1] if k <= n_cal else np.full(perms.shape[0], math.inf)
        answered = v_test & (q[:, None] > -math.inf)
        inside = s_test <= q[:, None]
        covered = (~answered | inside) if convention == "abstain_allowed" else inside
        q_all[i] = q
        covered_all[i] = covered.sum(axis=1)
        answer_all[i] = answered.mean(axis=1)
        silent_all[i] = (answered & ~covered).mean(axis=1)
        rows = answered.any(axis=1)
        med = np.full(len(q), np.nan)
        if rows.any():
            med[rows] = np.nanmedian(np.where(answered[rows], r_test[rows], np.nan), axis=1)
        rot_all[i] = np.where(rows, q * med * _RAD_TO_DEG, np.nan)
    return ResplitDraws(
        alphas=tuple(alphas),
        n_cal=n_cal,
        n_test=n_test,
        q=q_all,
        n_covered=covered_all,
        answer_rate=answer_all,
        silent_failure_rate=silent_all,
        median_rot_deg=rot_all,
    )


def summarise_draws(
    draws: ResplitDraws,
    index: int,
    *,
    n_top_atom: int,
    n_bottom_atom: int,
    n_finite_ties: int,
    ks_min_p: float,
    mean_band_se: float,
    fixed_n_covered: int | None = None,
) -> dict[str, Any]:
    """Compare one alpha's re-split coverages with the exact re-split law and the Beta law.

    Verdict:
    * `DEGENERATE` — q = +inf in every draw (answer_required: set is the whole space) or q = -inf
      in every draw (abstain_allowed: always abstain), *and* the law predicts it (KS not rejected).
      Coverage is 1 by construction; reported, not a validity test of the quantile.
    * `VALID` — KS against the atom-aware re-split law (`resplit_coverage_law`) not rejected at
      `ks_min_p`, and the mean within `mean_band_se` MC s.e. of that law's mean.
    * `DEVIATES` — otherwise.
    `in_plan_band` is IMPLEMENTATION_PLAN.md's literal criterion: the mean inside
    [1 - alpha, 1 - alpha + 1/(n+1)] widened by `mean_band_se` MC s.e. The plain Beta KS p is
    reported beside it; at finite m it is expected to reject (the re-split law is wider).
    `fixed_split_law_cdf` = P(X <= covered count of the committed val_cal -> val_test split) under
    the law: where that one split sits in the re-split distribution.
    """
    alpha = draws.alphas[index]
    n, m = draws.n_cal, draws.n_test
    cov = draws.coverage[index]
    q = draws.q[index]
    mean = float(cov.mean())
    se = mc_standard_error(cov)
    law = resplit_coverage_law(n, m, alpha, n_top_atom=n_top_atom, n_bottom_atom=n_bottom_atom)
    law_mean = pmf_mean(law)
    law_sd = float(np.sqrt(np.dot((np.arange(m + 1) / m - law_mean) ** 2, law)))
    ks_law = ks_against_pmf(draws.n_covered[index], law)
    ks_beta = ks_against_beta(cov, n, alpha)
    lo, hi = 1.0 - alpha, 1.0 - alpha + 1.0 / (n + 1)
    frac_pos, frac_neg = float(np.mean(q == math.inf)), float(np.mean(q == -math.inf))
    if (frac_pos == 1.0 or frac_neg == 1.0) and ks_law[1] >= ks_min_p:
        verdict = "DEGENERATE"
    elif ks_law[1] >= ks_min_p and abs(mean - law_mean) <= mean_band_se * se:
        verdict = "VALID"
    else:
        verdict = "DEVIATES"
    a, b = beta_parameters(n, alpha)
    return {
        "R": int(cov.size),
        "n_cal": n,
        "n_test": m,
        "k": quantile_index(n, alpha),
        "mean_coverage": mean,
        "mc_se": se,
        "sd_coverage": float(np.std(cov, ddof=1)),
        "plan_band": [lo - mean_band_se * se, hi + mean_band_se * se],
        "in_plan_band": bool(lo - mean_band_se * se <= mean <= hi + mean_band_se * se),
        "law": "beta_binomial_with_failure_atoms",
        "law_mean": law_mean,
        "law_sd": law_sd,
        "sd_ratio_observed_to_law": (
            None if verdict == "DEGENERATE" else float(np.std(cov, ddof=1)) / law_sd
        ),
        "fixed_split_law_cdf": (
            None if fixed_n_covered is None else float(law[: fixed_n_covered + 1].sum())
        ),
        "law_p_full_coverage_atom": float(law[m]),
        "ks_law": {
            "statistic": ks_law[0],
            "pvalue": ks_law[1],
            "pvalue_note": "upper bound on the exact p (discrete null): conservative",
        },
        "beta": {"a": a, "b": b, "ks_statistic": ks_beta[0], "ks_pvalue": ks_beta[1]},
        "frac_q_pos_inf": frac_pos,
        "frac_q_neg_inf": frac_neg,
        "pool_failure_atom": {"top_+inf": n_top_atom, "bottom_-inf": n_bottom_atom},
        "pool_finite_ties": n_finite_ties,
        "mean_answer_rate": float(draws.answer_rate[index].mean()),
        "mean_silent_failure_rate": float(draws.silent_failure_rate[index].mean()),
        "median_rot_deg_over_draws": (
            None
            if np.all(np.isnan(draws.median_rot_deg[index]))
            else float(np.nanmedian(draws.median_rot_deg[index]))
        ),
        "verdict": verdict,
    }
