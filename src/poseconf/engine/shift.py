"""The coverage-under-shift matrix (synthetic val_test, lightbox_poolB, sunlamp_poolB). Phase 6.

Every score (A1-C2) is calibrated with every arm of IMPLEMENTATION_PLAN.md §1.4 and evaluated on
synthetic `val_test` and on the HIL poolB halves. All frames come from the Phase 3 dumps (plus the
Phase 5 variance sidecar), so the A, B and C levels see one frame set.

Data roles (CLAUDE.md invariants 3-6):
* `val_tune` (synthetic) fits the Level A normalisers and g(conf), the Mondrian bin edges and the
  slice edges, and is the source half of the domain classifier. Nothing else fits anything.
* `val_cal` (synthetic) calibrates `split`, `mondrian` and `weighted_unlabeled_target`.
* `val_test` and `<hil>_poolB` are evaluation only. Their labels are read by
  `evaluation_labels` (poolB, or `all` on synthetic), and enter scores only to measure coverage.
* `<hil>_poolA`: the weighted arm reads its prediction-dump *features* only (`dump_features`);
  the oracle arm reads its *labels* through `oracle_target_labels`, the one function here that asks
  for poolA labels, under the `oracle_target_labels` tag.
* HIL `gt_crop` dumps are oracle-conditioned on every frame (`bbox_used` is the GT box), so their
  rows carry `oracle: true`, and the weighted arm refuses them (`require_label_free_dump`).
* Every arm returns one quantile per test frame and never sees a test label. PnP failures score
  +/-inf by convention everywhere, including inside oracle draws.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from poseconf import p1_adapter as p1
from poseconf.conformal.metrics import (
    coverage_by_slice,
    coverage_summary,
    outcomes,
    set_size_summary,
)
from poseconf.conformal.mondrian import (
    assign_bins,
    bin_edges,
    mondrian_quantiles,
    product_groups,
)
from poseconf.conformal.propagate import CameraGeometry
from poseconf.conformal.split import conformal_quantile
from poseconf.conformal.weighted import (
    effective_sample_size,
    max_normalised_weight,
)
from poseconf.data.dump import load_dump_labels
from poseconf.data.splits import DOMAIN_KEYS
from poseconf.engine import level_a, level_b
from poseconf.engine.level_c import PoseSigma, c2_scores, c2_set, pose_sigma

__all__ = [
    "ARM_FAMILIES",
    "MONDRIAN",
    "ORACLE",
    "SHIFT_SCORES",
    "SPLIT",
    "WEIGHTED",
    "DomainClassifier",
    "MondrianBins",
    "ShiftFrames",
    "build_frames",
    "dump_features",
    "evaluate",
    "evaluation_labels",
    "fit_domain_classifier",
    "fit_mondrian_bins",
    "likelihood_ratio_weights",
    "mondrian_frame_quantiles",
    "oracle_arm_tag",
    "oracle_draws",
    "oracle_target_labels",
    "pose_frames",
    "q_summary",
    "require_label_free_dump",
    "scores",
    "set_size_at",
    "slice_labels",
    "sliced_coverage",
    "split_frame_quantiles",
    "unit_radii",
    "valid_mask",
    "weight_report",
]

#: Every score in the matrix (IMPLEMENTATION_PLAN.md §1.3).
SHIFT_SCORES = ("A1", "A2", "A3", "B1", "B2", "C1", "C2")

SPLIT = "split"
MONDRIAN = "mondrian"
WEIGHTED = "weighted_unlabeled_target"
#: The oracle family's tag prefix; each n is its own arm, `oracle_target_labels_n<n>`.
ORACLE = "oracle_target_labels"
ARM_FAMILIES = (SPLIT, MONDRIAN, WEIGHTED, ORACLE)

_RAD_TO_DEG = 180.0 / math.pi
_POSE_SCORES = ("A1", "A2", "A3", "C2")
_RANGE_TERTILES = 3
_CONF_TERTILES = 3
_CONF_QUINTILES = 5
#: GT-bbox IoU slice edges, IMPLEMENTATION_PLAN.md §1.5.
IOU_BIN_EDGES = (0.5, 0.8)


def oracle_arm_tag(n: int) -> str:
    """`oracle_target_labels_n<n>`, the tag of one oracle arm."""
    return f"{ORACLE}_n{int(n)}"


# --------------------------------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------------------------------


def _take(obj: Any, index: NDArray[np.int64]) -> Any:
    return type(obj)(**{f.name: getattr(obj, f.name)[index] for f in dataclasses.fields(obj)})


@dataclass(frozen=True)
class ShiftFrames:
    """One domain x crop subset: keypoint frames, the vhead covariance, and what the arms need.

    Attributes:
        kp: Dump predictions joined to labels; `kp.cov` is the heatmap moment (B2's).
        cov_vhead: (n, K, 2, 2) the variance head's full-frame covariance (C1's).
        conf: (n,) mean keypoint confidence. Prediction.
        bbox_used: (n, 4) the box the crop came from (the GT box on `gt_crop`).
        bbox_gt: (n, 4) the tight GT box. Label: slicing only.
        sigma: C2's per-frame σ̂ (label-free).
    """

    kp: level_b.KeypointFrames
    cov_vhead: NDArray[np.float64]
    conf: NDArray[np.float64]
    bbox_used: NDArray[np.float64]
    bbox_gt: NDArray[np.float64]
    sigma: PoseSigma

    def __len__(self) -> int:
        """Number of frames."""
        return len(self.kp)

    def subset(self, index: NDArray[np.int64]) -> ShiftFrames:
        """Rows `index`, in that order."""
        return ShiftFrames(
            kp=self.kp.subset(index),
            cov_vhead=self.cov_vhead[index],
            conf=self.conf[index],
            bbox_used=self.bbox_used[index],
            bbox_gt=self.bbox_gt[index],
            sigma=_take(self.sigma, index),
        )

    @property
    def c1(self) -> level_b.KeypointFrames:
        """The keypoint frames with the variance head's covariance (C1)."""
        return dataclasses.replace(self.kp, cov=self.cov_vhead)


def build_frames(
    dump: dict[str, NDArray[Any]],
    labels: dict[str, NDArray[Any]],
    names: Sequence[str],
    geometry: CameraGeometry,
    order: str,
) -> tuple[ShiftFrames, dict[str, Any]]:
    """Join `names` of a dump (with the variance sidecar attached) to their labels.

    Args:
        dump: `load_dump` arrays after `level_c.attach_variance`.
        labels: A label file permitted for these frames (`evaluation_labels`,
            `oracle_target_labels`, or synthetic `all`).
        names: Filenames, in this order.
        geometry: P1's camera and wireframe.
        order: P1's quaternion order.

    Returns:
        `(frames, check)`; `check` is `level_b.join_dump`'s label-product cross-check.
    """
    kp, check = level_b.join_dump(dump, labels, names, geometry, order)
    position = {str(name): i for i, name in enumerate(dump["filename"])}
    rows = np.array([position[str(name)] for name in names], dtype=np.int64)
    l_position = {str(name): i for i, name in enumerate(labels["filename"])}
    l_rows = np.array([l_position[str(name)] for name in names], dtype=np.int64)
    cov_vhead = np.asarray(dump["vhead_cov_full"], dtype=np.float64)[rows]
    extents, _ = level_b.linearised_frames(
        "C1", dataclasses.replace(kp, cov=cov_vhead), geometry, order
    )
    frames = ShiftFrames(
        kp=kp,
        cov_vhead=cov_vhead,
        conf=np.asarray(dump["confidence"], dtype=np.float64)[rows].mean(axis=1),
        bbox_used=np.asarray(dump["bbox_used"], dtype=np.float64)[rows],
        bbox_gt=np.asarray(labels["bbox_gt"], dtype=np.float64)[l_rows],
        sigma=pose_sigma(extents),
    )
    return frames, check


def pose_frames(frames: ShiftFrames) -> level_a.Frames:
    """The Level A view of the frames. `e_r`/`e_t` (P1's sidecar join check) are not used: NaN."""
    kp = frames.kp
    nan = np.full(len(kp), np.nan)
    return level_a.Frames(
        filenames=kp.filenames,
        q_hat=kp.q_hat,
        t_hat=kp.t_hat,
        valid=kp.valid,
        conf=frames.conf,
        q_gt=kp.q_gt,
        t_gt=kp.t_gt,
        e_r=nan,
        e_t=nan,
    )


# --------------------------------------------------------------------------------------------------
# guarded label and feature access (CLAUDE.md invariant 4)
# --------------------------------------------------------------------------------------------------


def evaluation_labels(
    dumps_root: Path, run: str, domain: str, *, tag: str
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Labels of the evaluation subset: synthetic `all` (split manifests select rows) or HIL poolB.

    Never poolA: the oracle arm's labels come from `oracle_target_labels` only.
    """
    pool = "poolB" if domain in p1.HIL_DOMAINS else "all"
    return load_dump_labels(dumps_root, run, domain, pool, tag=tag)


def oracle_target_labels(
    dumps_root: Path, run: str, domain: str
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """HIL poolA labels, for the `oracle_target_labels_n*` arms only (tag `oracle_target_labels`).

    This is the only call in the shift study that requests poolA labels.

    Raises:
        ValueError: On a non-HIL domain.
    """
    if domain not in p1.HIL_DOMAINS:
        raise ValueError(f"the oracle arm calibrates on HIL poolA only, got {domain!r}")
    return load_dump_labels(dumps_root, run, domain, "poolA", tag=ORACLE)


def require_label_free_dump(meta: dict[str, Any], arm: str) -> None:
    """Refuse an oracle-conditioned dump (HIL `gt_crop`) for a non-oracle-tagged consumer (L2).

    Raises:
        p1_adapter.HILLabelAccessError: If the dump is oracle and `arm` is not an `oracle_*` tag.
    """
    if meta.get("oracle") and not arm.startswith("oracle_"):
        raise p1.HILLabelAccessError(
            f"dump tagged {meta.get('tag')!r} is GT-box-conditioned; arm {arm!r} may not read it"
        )


def dump_features(
    dump: dict[str, NDArray[Any]], meta: dict[str, Any], names: Sequence[str], *, arm: str
) -> NDArray[np.float64]:
    """The pooled 512-d encoder features of `names` from a prediction dump. Label-free.

    Raises:
        p1_adapter.HILLabelAccessError: On an oracle dump for a non-oracle arm.
        ValueError: On a name the dump lacks.
    """
    require_label_free_dump(meta, arm)
    position = {str(name): i for i, name in enumerate(dump["filename"])}
    missing = [name for name in names if str(name) not in position]
    if missing:
        raise ValueError(f"{len(missing)} names missing from the dump, e.g. {missing[:3]}")
    rows = np.array([position[str(name)] for name in names], dtype=np.int64)
    return np.asarray(dump["enc_feat"], dtype=np.float64)[rows]


# --------------------------------------------------------------------------------------------------
# scores and label-free set radii
# --------------------------------------------------------------------------------------------------


def valid_mask(score_id: str, frames: ShiftFrames) -> NDArray[np.bool_]:
    """Frames with a point estimate for `score_id` (C2 also needs a defined σ̂)."""
    if score_id == "C2":
        return frames.kp.valid & frames.sigma.ok
    return frames.kp.valid


def _keypoint_frames(score_id: str, frames: ShiftFrames) -> level_b.KeypointFrames:
    return frames.c1 if score_id == "C1" else frames.kp


def scores(
    score_id: str, frames: ShiftFrames, fit: level_a.LevelAFit, convention: str
) -> NDArray[np.float64]:
    """Nonconformity scores (uses labels), +/-inf on an invalid frame per `convention`."""
    if score_id in level_a.LEVEL_A_SCORES:
        return level_a.score_frames(score_id, pose_frames(frames), fit, convention)
    if score_id in level_b.KEYPOINT_SCORES:
        return level_b.score_frames(score_id, _keypoint_frames(score_id, frames), convention)
    if score_id == "C2":
        return c2_scores(frames.kp, frames.sigma, convention)
    raise ValueError(f"unknown score {score_id!r}; expected one of {SHIFT_SCORES}")


def unit_radii(
    score_id: str, frames: ShiftFrames, fit: level_a.LevelAFit
) -> dict[str, NDArray[np.float64]]:
    """Per-frame set radii at q = 1, from predictions and fitted inputs only (no label).

    Pose scores: `rot_deg`, and `trans_frac` / `trans_m` (ball) or `boresight_*` / `lateral_*`
    (A2's cylinder), as a fraction of ||t_hat|| and in metres. Keypoint scores: `kp_px`, the
    largest keypoint radius (+inf when a keypoint is unconstrained). Values on invalid frames are
    meaningless; `set_size_at` reads answered frames only.
    """
    if score_id in level_b.KEYPOINT_SCORES:
        kset = level_b.keypoint_set(score_id, _keypoint_frames(score_id, frames), 1.0)
        return {"kp_px": kset.radius_px}
    if score_id in level_a.LEVEL_A_SCORES:
        pset = level_a.pose_set(score_id, pose_frames(frames), fit, 1.0)
    elif score_id == "C2":
        pset = c2_set(frames.kp, frames.sigma, 1.0)
    else:
        raise ValueError(f"unknown score {score_id!r}; expected one of {SHIFT_SCORES}")
    pred_range = np.linalg.norm(np.nan_to_num(pset.t_center, nan=1.0), axis=-1)
    trans = pset.trans_radius
    out = {"rot_deg": pset.rot_radius * _RAD_TO_DEG}
    parts = ("trans",) if pset.translation == "ball" else ("boresight", "lateral")
    for column, part in enumerate(parts):
        out[f"{part}_frac"] = trans[:, column] / pred_range
        out[f"{part}_m"] = trans[:, column]
    return out


def _scaled(unit: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """q * unit with q = +inf or unit = +inf giving +inf (never NaN from 0 * inf)."""
    out = q * np.where(np.isinf(unit), 1.0, unit)
    return np.where(np.isinf(unit) | np.isinf(q), math.inf, out)


def set_size_at(
    unit: dict[str, NDArray[np.float64]], q: NDArray[np.float64], answered: NDArray[np.bool_]
) -> dict[str, Any] | None:
    """Median / p90 of each radius `q_i * unit_i` over answered frames, keeping +inf; None if none.

    Answered frames have q_i > -inf and finite, positive unit radii (or +inf, unconstrained).
    """
    if not answered.any():
        return None
    return {
        key: set_size_summary(_scaled(np.asarray(value)[answered], q[answered]))
        for key, value in unit.items()
    }


def q_summary(q: NDArray[np.float64]) -> dict[str, Any]:
    """Distribution of per-frame quantiles (order statistics, so +/-inf survive)."""
    lo, med, p90, hi = np.quantile(q, [0.0, 0.5, 0.9, 1.0], method="inverted_cdf")
    return {
        "min": float(lo),
        "median": float(med),
        "p90": float(p90),
        "max": float(hi),
        "frac_pos_inf": float(np.mean(q == math.inf)),
        "frac_neg_inf": float(np.mean(q == -math.inf)),
    }


# --------------------------------------------------------------------------------------------------
# arms: one quantile per test frame, never a test label
# --------------------------------------------------------------------------------------------------


def split_frame_quantiles(s_cal: NDArray[np.float64], alpha: float, m: int) -> NDArray[np.float64]:
    """Split CP: the one finite-sample quantile, broadcast to `m` test frames."""
    return np.full(m, conformal_quantile(s_cal, alpha))


@dataclass(frozen=True)
class MondrianBins:
    """Predicted-range tertiles x mean-confidence tertiles, edges fitted on `val_tune` solved frames.

    Invalid frames (no point estimate for the score) form their own group, `failure_group`.
    """

    range_edges: NDArray[np.float64]
    conf_edges: NDArray[np.float64]

    @property
    def n_levels(self) -> tuple[int, int]:
        """(range levels, confidence levels)."""
        return len(self.range_edges) + 1, len(self.conf_edges) + 1

    @property
    def failure_group(self) -> int:
        """The group of frames without a point estimate."""
        a, b = self.n_levels
        return a * b

    @property
    def group_ids(self) -> list[int]:
        """Every group, the failure group last."""
        return list(range(self.failure_group + 1))

    def groups(self, frames: ShiftFrames, valid: NDArray[np.bool_]) -> NDArray[np.int64]:
        """(n,) group per frame, from predictions only (||t_hat||, mean confidence)."""
        out = np.full(len(frames), self.failure_group, dtype=np.int64)
        if valid.any():
            rng = np.linalg.norm(frames.kp.t_hat[valid], axis=-1)
            out[valid] = product_groups(
                [
                    assign_bins(rng, self.range_edges),
                    assign_bins(frames.conf[valid], self.conf_edges),
                ],
                self.n_levels,
            )
        return out

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "groups": "predicted_range_tertile x mean_confidence_tertile; invalid frames own group",
            "range_edges_m": self.range_edges.tolist(),
            "conf_edges": self.conf_edges.tolist(),
            "failure_group": self.failure_group,
            "fit_on": "val_tune solved frames",
        }


def fit_mondrian_bins(tune: ShiftFrames) -> MondrianBins:
    """Tertile edges of ||t_hat|| and mean confidence on solved `val_tune` frames."""
    ok = tune.kp.valid
    return MondrianBins(
        range_edges=bin_edges(np.linalg.norm(tune.kp.t_hat[ok], axis=-1), _RANGE_TERTILES),
        conf_edges=bin_edges(tune.conf[ok], _CONF_TERTILES),
    )


def mondrian_frame_quantiles(
    s_cal: NDArray[np.float64],
    g_cal: NDArray[np.int64],
    g_test: NDArray[np.int64],
    alpha: float,
    group_ids: Sequence[int],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Per-group split CP; each test frame gets its group's quantile (+inf for a thin group).

    Returns:
        `(q per test frame, {group: {n_cal, q}})`.
    """
    per_group = mondrian_quantiles(s_cal, g_cal, alpha, group_ids=group_ids)
    q = np.array([per_group[int(g)] for g in g_test], dtype=np.float64)
    table = {
        str(g): {"n_cal": int((g_cal == g).sum()), "q": float(per_group[g])} for g in group_ids
    }
    return q, table


@dataclass(frozen=True)
class DomainClassifier:
    """Logistic source (synthetic `val_tune`) vs target (HIL poolA) classifier on encoder features.

    Attributes:
        model: The fitted sklearn pipeline (trained on every source and target row).
        n_source: Source rows.
        n_target: Target rows.
        cv_auc: Out-of-fold ROC AUC per fold.
    """

    model: Any
    n_source: int
    n_target: int
    cv_auc: list[float]

    def logits(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """log P(target | x) / P(source | x) for each row."""
        return np.asarray(self.model.decision_function(features), dtype=np.float64)


def _classifier(c: float, max_iter: int, standardise: bool) -> Any:
    logistic = LogisticRegression(C=c, max_iter=max_iter)
    return make_pipeline(StandardScaler(), logistic) if standardise else make_pipeline(logistic)


def fit_domain_classifier(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    c: float,
    max_iter: int,
    standardise: bool,
    cv_folds: int,
    cv_seed: int,
) -> DomainClassifier:
    """Fit the domain classifier and its `cv_folds`-fold stratified out-of-fold AUC.

    Args:
        source: (n_s, d) synthetic `val_tune` features.
        target: (n_t, d) HIL poolA features (unlabeled).
        c: Inverse L2 strength (fixed in config, never tuned on HIL).
        max_iter: Solver iterations.
        standardise: Standardise features (fitted inside each fold).
        cv_folds: Folds.
        cv_seed: Fold shuffling seed.
    """
    x = np.vstack([source, target])
    y = np.concatenate([np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)])
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
    auc = []
    for train, held in folds.split(x, y):
        model = _classifier(c, max_iter, standardise).fit(x[train], y[train])
        auc.append(float(roc_auc_score(y[held], model.decision_function(x[held]))))
    model = _classifier(c, max_iter, standardise).fit(x, y)
    return DomainClassifier(model, len(source), len(target), auc)


def likelihood_ratio_weights(
    logit_cal: NDArray[np.float64],
    logit_test: NDArray[np.float64],
    n_source: int,
    n_target: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """w = exp(logit - c) * n_s / n_t, one shared shift c = max logit over calibration and test.

    `exp(logit) * n_s / n_t` is the classifier odds ratio c/(1-c) * n_s/n_t of Tibshirani et al.;
    the common factor exp(-c) cancels in every normalised mass p_i and p_test, so the weighted
    quantiles are unchanged, and no weight overflows when the classifier separates the domains.

    Returns:
        `(w_cal, w_test, c)`.
    """
    shift = float(max(np.max(logit_cal), np.max(logit_test)))
    ratio = n_source / n_target
    return np.exp(logit_cal - shift) * ratio, np.exp(logit_test - shift) * ratio, shift


def weight_report(w_cal: NDArray[np.float64], w_test: NDArray[np.float64]) -> dict[str, Any]:
    """ESS and concentration of the calibration weights; spread of the test weights.

    Weights are reported relative to the median calibration weight, so the common shift drops out.
    """
    scale = float(np.median(w_cal)) if np.median(w_cal) > 0 else float(np.max(w_cal))
    rel = np.quantile(w_test / scale, [0.1, 0.5, 0.9], method="inverted_cdf") if scale > 0 else []
    return {
        "ess_cal": effective_sample_size(w_cal),
        "n_cal": int(w_cal.size),
        "max_normalised_weight_cal": max_normalised_weight(w_cal),
        "test_weight_over_median_cal_weight_p10_p50_p90": [float(v) for v in rel],
        "clip": None,
    }


def oracle_draws(
    n_pool: int, n: int, draws: int, seed: int, domain: str
) -> list[NDArray[np.int64]]:
    """`draws` sorted subsets of size n of a poolA index, without replacement, seeded per
    (seed, domain, n, draw). Failed frames are kept: they are part of the target distribution.

    Raises:
        ValueError: If n exceeds the pool.
    """
    if not 0 < n <= n_pool:
        raise ValueError(f"cannot draw n={n} from a pool of {n_pool}")
    out = []
    for draw in range(draws):
        rng = np.random.default_rng(np.random.SeedSequence([seed, DOMAIN_KEYS[domain], n, draw]))
        out.append(np.sort(rng.choice(n_pool, size=n, replace=False)))
    return out


# --------------------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------------------


def evaluate(
    score_id: str,
    frames: ShiftFrames,
    s_test: NDArray[np.float64],
    q: NDArray[np.float64],
    unit: dict[str, NDArray[np.float64]],
    convention: str,
) -> tuple[dict[str, Any], NDArray[np.bool_], NDArray[np.bool_]]:
    """Coverage, CI, answer and silent-failure rates, set size, and the per-score counts.

    Returns:
        `(row fields, covered, answered)`.
    """
    valid = valid_mask(score_id, frames)
    covered, answered = outcomes(s_test, valid, q, convention)
    row: dict[str, Any] = coverage_summary(covered, answered)
    # Coverage among answered frames: a slice by an observable event, reported beside (never instead
    # of) the marginal coverage, because abstentions count as covered under abstain_allowed.
    row["coverage_given_answered"] = (
        float((covered & answered).sum() / answered.sum()) if answered.any() else None
    )
    row["set_size"] = set_size_at(unit, q, answered)
    if score_id in level_b.KEYPOINT_SCORES:
        kp = _keypoint_frames(score_id, frames)
        bounded = kp.include & ~kp.unconstrained
        row["n_vacuous"] = int((kp.valid & ~bounded.any(axis=1)).sum())
        row["n_frames_with_unconstrained_keypoint"] = int(
            (answered & kp.unconstrained.any(axis=1)).sum()
        )
    if score_id == "C2":
        row["n_solved_sigma_undefined"] = int((frames.kp.valid & ~frames.sigma.ok).sum())
    return row, covered, answered


def _iou(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    x0, y0 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x1, y1 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda box: (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])  # noqa: E731
    union = area(a) + area(b) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def slice_labels(
    test: ShiftFrames, tune: ShiftFrames, *, with_iou: bool
) -> dict[str, NDArray[np.int64]]:
    """Evaluation slices (§1.5). GT quantities slice the evaluation; they never enter a score.

    Edges for GT range tertiles and confidence quintiles come from `val_tune`, so a slice means the
    same thing on every domain. IoU bins are fixed: <0.5, 0.5-0.8, >0.8 (`with_iou` only on
    `predicted_crop`; on `gt_crop` the box is the GT box).
    """
    gt_range_edges = bin_edges(np.linalg.norm(tune.kp.t_gt, axis=-1), _RANGE_TERTILES)
    conf_edges = bin_edges(tune.conf, _CONF_QUINTILES)
    out = {
        "gt_range_tertile": assign_bins(np.linalg.norm(test.kp.t_gt, axis=-1), gt_range_edges),
        "confidence_quintile": assign_bins(test.conf, conf_edges),
    }
    if with_iou:
        iou = _iou(test.bbox_used, test.bbox_gt)
        out["gt_bbox_iou_bin"] = np.searchsorted(np.asarray(IOU_BIN_EDGES), iou, side="right")
    return out


def sliced_coverage(
    covered: NDArray[np.bool_], answered: NDArray[np.bool_], slices: dict[str, NDArray[np.int64]]
) -> dict[str, Any]:
    """`coverage_by_slice` for each slicing, keys as strings for JSON."""
    return {
        name: {str(k): v for k, v in coverage_by_slice(covered, answered, labels).items()}
        for name, labels in slices.items()
    }
