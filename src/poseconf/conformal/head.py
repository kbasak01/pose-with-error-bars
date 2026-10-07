"""`ConformalPoseHead`: a calibration artifact, hash-locked to its checkpoints, applied per frame.

IMPLEMENTATION_PLAN §1.1 / Phase 7. The head is the deployable end of the conformal pipeline: it
takes one frame's *predictions* (`FrameEstimate`, no label field exists) and the quantile fitted on
`val_cal`, and returns `Answer(set)` or `Abstain(reason)`. Sets are built by the same
`scores.SCORES[score_id].set_fn` the evaluation used, so a set issued here is the set the committed
coverage numbers were measured on.

**Artifact.** JSON with a `payload` (everything that changes a set: score, version, alpha,
convention, quantile, normalisers, camera geometry for C2, quaternion order, the checkpoint locks,
and a `context` carrying the P1 commit and PnP configuration the scores were produced under) and a
`calibration_id`, the SHA-256 of the canonical payload. `provenance` sits outside the
payload so a rebuild from the same committed rows is byte-stable apart from it.

**Locks.** `from_json` hashes the checkpoint files the caller is running and refuses a P1 keypoint
checkpoint, a detector checkpoint (predicted-crop artifacts: the detector decides every crop) or,
for C1/C2, a variance-head checkpoint whose SHA-256 differs from the artifact's: a quantile is a
property of one pipeline's score distribution, and applying it to another's predictions has no
coverage meaning. It also refuses an artifact whose payload no longer hashes to its id, an unknown
schema or score version, and a payload that fails the same validation `build_artifact` applies.
`calibration_id` is an unkeyed hash: it detects edits, not a deliberate forgery.

**Abstention mirrors `metrics.outcomes`.** A frame is answered iff it has a point estimate (C2: and a
defined pose sigma) and q > -inf. Under `answer_required` an `Abstain` is a miss unless q = +inf
(the whole space); under `abstain_allowed` it is covered. `covered_if_abstained` encodes that.

B2 is not deployable: it needs heatmap second moments, which the exported graph does not output.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from poseconf.conformal.propagate import (
    CameraGeometry,
    LinearisedExtent,
    linearised_extent,
    project,
    projection_jacobian,
)
from poseconf.conformal.scores import SCORES, KeypointSet, PoseSet, set_mahalanobis
from poseconf.conformal.so3 import QUATERNION_ORDERS, quat_to_matrix
from poseconf.conformal.split import CONVENTIONS, check_alpha

__all__ = [
    "ARTIFACT_SCHEMA",
    "DEPLOYABLE_SCORES",
    "SCORE_VERSIONS",
    "Abstain",
    "Answer",
    "CalibrationMismatchError",
    "ConformalPoseHead",
    "FrameEstimate",
    "build_artifact",
    "calibration_id",
    "geometry_from_dict",
    "geometry_to_dict",
    "sha256_file",
]

ARTIFACT_SCHEMA = "poseconf.calibration.v1"

#: Bumped whenever a score's definition or its set construction changes (docs/DECISIONS.md). An
#: artifact carries the version it was calibrated under; the loader refuses any other.
SCORE_VERSIONS: dict[str, int] = {"A1": 1, "A2": 1, "A3": 1, "B1": 1, "C1": 1, "C2": 1}

#: Scores the head can issue sets for at test time. B2 is excluded: no heatmap moments in the graph.
DEPLOYABLE_SCORES = tuple(SCORE_VERSIONS)

#: Normaliser keys each score's artifact must carry.
_NORMALISERS: dict[str, tuple[str, ...]] = {
    "A1": ("c_R", "c_t"),
    "A2": ("c_R", "c_z", "c_xy"),
    "A3": ("c_R", "c_t", "g_x", "g_y"),
    "B1": (),
    "C1": (),
    "C2": (),
}

_COVARIANCE_SCORES = ("C1", "C2")
_CROP_SOURCES = ("predicted_crop", "gt_crop")
_HASH_CHUNK = 1 << 20


class CalibrationMismatchError(ValueError):
    """The artifact does not belong to the checkpoints, code or schema it is being loaded with."""


# --------------------------------------------------------------------------------------------------
# hashing and serialisation
# --------------------------------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode(value: Any) -> Any:
    """JSON-safe copy: +/-inf as the strings "inf" / "-inf" (results-provenance rule), no NaN."""
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, np.ndarray):
        return _encode(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            raise ValueError("NaN cannot enter a calibration artifact")
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    return value


def _decode_float(value: Any) -> float:
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return float(value)


def calibration_id(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of an (encoded) payload: sorted keys, no whitespace."""
    text = json.dumps(_encode(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def geometry_to_dict(geometry: CameraGeometry) -> dict[str, Any]:
    """`CameraGeometry` as plain JSON values (float64 round-trips exactly through `repr`)."""
    return {
        "wireframe": np.asarray(geometry.wireframe, dtype=np.float64).tolist(),
        "camera_matrix": np.asarray(geometry.camera_matrix, dtype=np.float64).tolist(),
        "distortion": (
            None
            if geometry.distortion is None
            else np.asarray(geometry.distortion, dtype=np.float64).reshape(-1).tolist()
        ),
        "width": int(geometry.width),
        "height": int(geometry.height),
        "min_depth": float(geometry.min_depth),
        "max_normalised_radius": float(geometry.max_normalised_radius),
    }


def geometry_from_dict(block: Mapping[str, Any]) -> CameraGeometry:
    """Inverse of `geometry_to_dict`."""
    return CameraGeometry(
        wireframe=np.asarray(block["wireframe"], dtype=np.float64),
        camera_matrix=np.asarray(block["camera_matrix"], dtype=np.float64),
        distortion=(
            None if block["distortion"] is None else np.asarray(block["distortion"], np.float64)
        ),
        width=int(block["width"]),
        height=int(block["height"]),
        min_depth=float(block["min_depth"]),
        max_normalised_radius=float(block["max_normalised_radius"]),
    )


def build_artifact(
    *,
    score_id: str,
    alpha: float,
    convention: str,
    quantile: float,
    n_cal: int,
    normalisers: Mapping[str, Any],
    quaternion_order: str,
    p1_checkpoint_sha256: str,
    context: Mapping[str, Any],
    detector_checkpoint_sha256: str | None = None,
    variance_head_sha256: str | None = None,
    geometry: CameraGeometry | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a calibration artifact (JSON-ready dict) and stamp its `calibration_id`.

    Args:
        score_id: One of `DEPLOYABLE_SCORES`.
        alpha: Miscoverage level.
        convention: PnP-failure convention the quantile was computed under.
        quantile: The split-conformal quantile from `val_cal` (may be +/-inf).
        n_cal: Calibration set size, failures included.
        normalisers: The score's fitted constants (`_NORMALISERS[score_id]`; empty for B1/C1/C2).
        quaternion_order: P1's quaternion order (`so3.QUATERNION_ORDERS`).
        p1_checkpoint_sha256: SHA-256 of the P1 keypoint checkpoint the scores came from.
        context: Locked description of how the scores were produced; must hold `crop_source`
            (`predicted_crop` or `gt_crop`). Also run, arm, calibration subset, P1 commit and
            PnP configuration, so a deployment can assert them.
        detector_checkpoint_sha256: SHA-256 of P1's detector (required for `predicted_crop` only).
        variance_head_sha256: SHA-256 of the variance-head checkpoint (required for C1/C2 only).
        geometry: Camera and wireframe (required for C2 only, which linearises PnP).
        provenance: Unlocked bookkeeping (source result file and SHA, git SHA, creation time).

    Returns:
        `{"schema", "calibration_id", "payload", "provenance"}`.

    Raises:
        ValueError: On an unsupported score, a missing or unexpected normaliser, or a missing lock.
    """
    if math.isnan(quantile):
        raise ValueError("quantile is NaN")
    payload: dict[str, Any] = {
        "score_id": score_id,
        "score_version": SCORE_VERSIONS.get(score_id),
        "alpha": float(alpha),
        "convention": convention,
        "quantile": float(quantile),
        "n_cal": int(n_cal),
        "normalisers": dict(normalisers),
        "quaternion_order": quaternion_order,
        "geometry": None if geometry is None else geometry_to_dict(geometry),
        "locks": {
            "p1_checkpoint_sha256": p1_checkpoint_sha256,
            "detector_checkpoint_sha256": detector_checkpoint_sha256,
            "variance_head_sha256": variance_head_sha256,
        },
        "context": dict(context),
    }
    _validate_payload(payload)
    encoded = _encode(payload)
    return {
        "schema": ARTIFACT_SCHEMA,
        "calibration_id": calibration_id(encoded),
        "payload": encoded,
        "provenance": _encode(dict(provenance or {})),
    }


def _validate_payload(payload: Mapping[str, Any]) -> None:
    """The structural rules every artifact obeys, checked at build and again at load.

    Raises:
        ValueError: On an unsupported score or version, convention, alpha, quaternion order,
            normaliser set, a NaN quantile, a non-positive n_cal, or a lock or geometry that is
            missing where required or present where not.
    """
    score_id = payload["score_id"]
    if score_id not in DEPLOYABLE_SCORES:
        raise ValueError(
            f"{score_id!r} is not deployable (supported: {DEPLOYABLE_SCORES}); B2 needs heatmap "
            "moments the exported graph does not produce"
        )
    if payload["score_version"] != SCORE_VERSIONS[score_id]:
        raise ValueError(
            f"{score_id} artifact is score version {payload['score_version']}, this code is "
            f"version {SCORE_VERSIONS[score_id]}; recalibrate"
        )
    if payload["convention"] not in CONVENTIONS:
        raise ValueError(f"unknown convention {payload['convention']!r}")
    check_alpha(payload["alpha"])
    if math.isnan(_decode_float(payload["quantile"])):
        raise ValueError("quantile is NaN")
    if int(payload["n_cal"]) < 1:
        raise ValueError("n_cal must be positive")
    if payload["quaternion_order"] not in QUATERNION_ORDERS:
        raise ValueError(f"unknown quaternion order {payload['quaternion_order']!r}")
    if set(payload["normalisers"]) != set(_NORMALISERS[score_id]):
        raise ValueError(
            f"{score_id} needs normalisers {_NORMALISERS[score_id]}, "
            f"got {sorted(payload['normalisers'])}"
        )
    locks, context = payload["locks"], payload["context"]
    if not locks.get("p1_checkpoint_sha256"):
        raise ValueError("p1_checkpoint_sha256 is required")
    crop_source = context.get("crop_source")
    if crop_source not in _CROP_SOURCES:
        raise ValueError(f"context.crop_source must be one of {_CROP_SOURCES}, got {crop_source!r}")
    if (crop_source == "predicted_crop") != bool(locks.get("detector_checkpoint_sha256")):
        raise ValueError("detector_checkpoint_sha256 is required for predicted_crop only")
    if (score_id in _COVARIANCE_SCORES) != bool(locks.get("variance_head_sha256")):
        raise ValueError(f"variance_head_sha256 is required for C1/C2 only (score {score_id})")
    if (score_id == "C2") != (payload["geometry"] is not None):
        raise ValueError(f"geometry is required for C2 only (score {score_id})")


# --------------------------------------------------------------------------------------------------
# per-frame input and output
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameEstimate:
    """One frame's predictions. There is deliberately no label field (CLAUDE.md invariant 5).

    Attributes:
        valid: PnP returned a pose.
        q_hat: (4,) quaternion, P1's order (NaN allowed when not valid).
        t_hat: (3,) translation, camera frame, metres.
        keypoints: (K, 2) predicted keypoints, full-frame px.
        confidence_mean: Mean keypoint confidence (A3), as P1's evaluator stores it (float32 mean).
        bbox: (4,) predicted tight box `[x0, y0, x1, y1]`, full-frame px (B1's diagonal).
        unconstrained: (K,) bool, keypoints whose set is the whole image (empty heatmap channel),
            or None for none.
    """

    valid: bool
    q_hat: NDArray[np.float64]
    t_hat: NDArray[np.float64]
    keypoints: NDArray[np.float64]
    confidence_mean: float
    bbox: NDArray[np.float64]
    unconstrained: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class Answer:
    """A set was issued.

    Attributes:
        set: A one-frame `PoseSet` (A1-A3, C2) or `KeypointSet` (B1, C1).
        score_id: The score.
        alpha: Miscoverage level of the calibration.
        convention: Failure convention of the calibration.
        calibration_id: The artifact's id.
        sigma: C2 only: `(sigma_R rad, sigma_t m)`, the linearised pose standard deviations.
    """

    set: PoseSet | KeypointSet
    score_id: str
    alpha: float
    convention: str
    calibration_id: str
    sigma: tuple[float, float] | None = None


@dataclass(frozen=True)
class Abstain:
    """No set was issued.

    Attributes:
        reason: `no_estimate` (PnP failed), `sigma_undefined` (C2: < 3 usable keypoints or a
            singular information matrix) or `q_neg_inf` (the quantile is -inf).
        calibration_id: The artifact's id.
        covered_if_abstained: How `metrics.outcomes` scores this frame: True under
            `abstain_allowed`, and under `answer_required` only when q = +inf (whole space).
    """

    reason: str
    calibration_id: str
    covered_if_abstained: bool


# --------------------------------------------------------------------------------------------------
# the head
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConformalPoseHead:
    """A loaded, lock-checked calibration. Build with `from_json` or `from_dict`, not directly."""

    payload: dict[str, Any]
    calibration_id: str
    geometry: CameraGeometry | None = field(default=None, repr=False)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        p1_checkpoint: str | Path,
        detector_checkpoint: str | Path | None = None,
        variance_head_checkpoint: str | Path | None = None,
    ) -> ConformalPoseHead:
        """Load an artifact, hashing the checkpoint files actually in use.

        Args:
            path: The artifact JSON.
            p1_checkpoint: The P1 keypoint checkpoint the predictions come from.
            detector_checkpoint: P1's detector checkpoint (predicted-crop artifacts).
            variance_head_checkpoint: The variance-head checkpoint (C1/C2).

        Raises:
            CalibrationMismatchError: See `from_dict`.
        """
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(
            artifact,
            p1_checkpoint_sha256=sha256_file(p1_checkpoint),
            detector_sha256=None
            if detector_checkpoint is None
            else sha256_file(detector_checkpoint),
            variance_head_sha256=(
                None if variance_head_checkpoint is None else sha256_file(variance_head_checkpoint)
            ),
        )

    @classmethod
    def from_dict(
        cls,
        artifact: Mapping[str, Any],
        *,
        p1_checkpoint_sha256: str,
        detector_sha256: str | None = None,
        variance_head_sha256: str | None = None,
    ) -> ConformalPoseHead:
        """Validate an artifact against the running checkpoints' hashes.

        Raises:
            CalibrationMismatchError: Unknown schema; payload not hashing to `calibration_id`;
                a payload failing `build_artifact`'s validation (unknown score or version,
                convention, alpha, order, normalisers, locks); P1 checkpoint hash differs;
                detector hash missing or different for a predicted-crop artifact; variance-head
                hash missing or different for C1/C2.
        """
        if artifact.get("schema") != ARTIFACT_SCHEMA:
            raise CalibrationMismatchError(
                f"unknown artifact schema {artifact.get('schema')!r}; expected {ARTIFACT_SCHEMA!r}"
            )
        payload = copy.deepcopy(dict(artifact["payload"]))
        if calibration_id(payload) != artifact.get("calibration_id"):
            raise CalibrationMismatchError(
                "artifact payload does not hash to its calibration_id: it was edited after it was "
                "built, so its quantile cannot be trusted"
            )
        try:
            _validate_payload(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise CalibrationMismatchError(f"invalid artifact payload: {error}") from error
        score_id = payload["score_id"]
        locks = payload["locks"]
        if p1_checkpoint_sha256 != locks["p1_checkpoint_sha256"]:
            raise CalibrationMismatchError(
                f"P1 checkpoint sha256 {p1_checkpoint_sha256} differs from the calibrated one "
                f"{locks['p1_checkpoint_sha256']}: the quantile does not apply to this model"
            )
        if locks["detector_checkpoint_sha256"] is not None:
            if detector_sha256 is None:
                raise CalibrationMismatchError(
                    "this predicted-crop artifact needs the detector checkpoint it was calibrated with"
                )
            if detector_sha256 != locks["detector_checkpoint_sha256"]:
                raise CalibrationMismatchError(
                    f"detector sha256 {detector_sha256} differs from the calibrated one "
                    f"{locks['detector_checkpoint_sha256']}"
                )
        if score_id in _COVARIANCE_SCORES:
            if variance_head_sha256 is None:
                raise CalibrationMismatchError(
                    f"{score_id} needs the variance-head checkpoint it was calibrated with"
                )
            if variance_head_sha256 != locks["variance_head_sha256"]:
                raise CalibrationMismatchError(
                    f"variance-head sha256 {variance_head_sha256} differs from the calibrated one "
                    f"{locks['variance_head_sha256']}"
                )
        geometry = None if payload["geometry"] is None else geometry_from_dict(payload["geometry"])
        return cls(payload=payload, calibration_id=artifact["calibration_id"], geometry=geometry)

    # ---- descriptors --------------------------------------------------------------------------

    @property
    def score_id(self) -> str:
        """The score this artifact calibrates."""
        return self.payload["score_id"]

    @property
    def alpha(self) -> float:
        """Miscoverage level."""
        return float(self.payload["alpha"])

    @property
    def convention(self) -> str:
        """PnP-failure convention."""
        return self.payload["convention"]

    @property
    def quantile(self) -> float:
        """The conformal quantile (may be +/-inf)."""
        return _decode_float(self.payload["quantile"])

    @property
    def n_cal(self) -> int:
        """Calibration set size."""
        return int(self.payload["n_cal"])

    # ---- prediction ---------------------------------------------------------------------------

    def pose_sigma(
        self, estimate: FrameEstimate, cov: ArrayLike
    ) -> tuple[float, float, LinearisedExtent] | None:
        """C2's (sigma_R, sigma_t) at the PnP estimate from full-frame keypoint covariances.

        Same construction as `engine.level_b.linearised_frames` + `engine.level_c.pose_sigma`: the
        covariances are validated and unconstrained entries replaced by the C1 set constructor at
        q = 1 (shape, finiteness, symmetry, positive-definiteness: a bad covariance raises rather
        than turning into an abstention); U = keypoints visible under the estimate and
        constrained; residual = projection - prediction.

        Raises:
            ValueError: If `cov` is malformed or not positive-definite on a constrained keypoint.

        Returns:
            `(sigma_R, sigma_t, extent)`, or None when either sigma is undefined (not finite or
            not positive) or the frame has no estimate.
        """
        if self.geometry is None:
            raise ValueError("pose_sigma needs the camera geometry of a C2 artifact")
        if not estimate.valid:
            return None
        order = self.payload["quaternion_order"]
        rotation = quat_to_matrix(np.asarray(estimate.q_hat, np.float64)[None], order)[0]
        t_hat = np.asarray(estimate.t_hat, dtype=np.float64)
        points, visible = project(rotation[None], t_hat[None], self.geometry)
        unconstrained = _unconstrained(estimate)
        use = visible[0] & ~unconstrained
        keypoints = np.asarray(estimate.keypoints, dtype=np.float64)
        shapes = set_mahalanobis(
            keypoints[None],
            np.array([True]),
            1.0,
            cov=np.asarray(cov, dtype=np.float64)[None],
            unconstrained=unconstrained[None],
        ).cov[0]
        jac = projection_jacobian(rotation, t_hat, self.geometry)
        extent = linearised_extent(jac, shapes, use, points[0] - keypoints)
        if not (math.isfinite(extent.rot_var) and math.isfinite(extent.trans_var)):
            return None
        sigma_r, sigma_t = math.sqrt(extent.rot_var), math.sqrt(extent.trans_var)
        if not (sigma_r > 0 and sigma_t > 0):
            return None
        return sigma_r, sigma_t, extent

    def predict(self, estimate: FrameEstimate, cov: ArrayLike | None = None) -> Answer | Abstain:
        """Issue this frame's set, or abstain.

        Args:
            estimate: The frame's predictions.
            cov: (K, 2, 2) full-frame keypoint covariances, px^2 (C1, C2; map crop covariances
                with `propagate.crop_cov_to_full`). Ignored by A1-A3 and B1.

        Returns:
            `Answer` or `Abstain` (see the module docstring for how each is scored).

        Raises:
            ValueError: If C1/C2 is called without `cov` on a valid frame.
        """
        sid, q = self.score_id, self.quantile
        valid = bool(estimate.valid)
        if sid in _COVARIANCE_SCORES and valid and cov is None:
            raise ValueError(f"{sid} needs the frame's keypoint covariances")

        sigma: tuple[float, float] | None = None
        reason = "no_estimate"
        if sid == "C2" and valid:
            got = self.pose_sigma(estimate, cov)
            if got is None:
                valid, reason = False, "sigma_undefined"
            else:
                sigma = (got[0], got[1])

        built = self._set(estimate, cov, valid, q, sigma)
        if not valid or q == -math.inf:
            if valid:
                reason = "q_neg_inf"
            covered = self.convention == "abstain_allowed" or q == math.inf
            return Abstain(reason, self.calibration_id, covered)
        return Answer(built, sid, self.alpha, self.convention, self.calibration_id, sigma)

    def _set(
        self,
        estimate: FrameEstimate,
        cov: ArrayLike | None,
        valid: bool,
        q: float,
        sigma: tuple[float, float] | None,
    ) -> PoseSet | KeypointSet:
        """The one-frame set, through the registry's `set_fn` (the evaluation's code path)."""
        sid = self.score_id
        spec = SCORES[sid]
        ok = np.array([valid])
        norm = self.payload["normalisers"]
        if spec.level == "pose":
            q_hat = np.asarray(estimate.q_hat, dtype=np.float64)[None]
            t_hat = np.asarray(estimate.t_hat, dtype=np.float64)[None]
            if sid == "A1":
                kwargs = {"c_R": norm["c_R"], "c_t": norm["c_t"]}
            elif sid == "A2":
                kwargs = {"c_R": norm["c_R"], "c_z": norm["c_z"], "c_xy": norm["c_xy"]}
            elif sid == "A3":
                g = np.interp(
                    np.array([estimate.confidence_mean], dtype=np.float64),
                    np.asarray(norm["g_x"], dtype=np.float64),
                    np.asarray(norm["g_y"], dtype=np.float64),
                )
                kwargs = {"c_R": norm["c_R"], "c_t": norm["c_t"], "difficulty": g}
            else:  # C2
                s_r, s_t = sigma if sigma is not None else (1.0, 1.0)
                kwargs = {"sigma_R": np.array([s_r]), "sigma_t": np.array([s_t])}
            return spec.set_fn(q_hat, t_hat, ok, q, **kwargs)

        y_hat = np.asarray(estimate.keypoints, dtype=np.float64)[None]
        unconstrained = _unconstrained(estimate)[None]
        if sid == "B1":
            box = np.asarray(estimate.bbox, dtype=np.float64)
            d_hat = np.array([math.hypot(box[2] - box[0], box[3] - box[1])])
            return spec.set_fn(y_hat, ok, q, d_hat=d_hat, unconstrained=unconstrained)
        k = y_hat.shape[1]
        c = np.broadcast_to(np.eye(2), (1, k, 2, 2)) if cov is None else np.asarray(cov)[None]
        return spec.set_fn(y_hat, ok, q, cov=c, unconstrained=unconstrained)


def _unconstrained(estimate: FrameEstimate) -> NDArray[np.bool_]:
    k = np.asarray(estimate.keypoints).shape[0]
    if estimate.unconstrained is None:
        return np.zeros(k, dtype=bool)
    unc = np.asarray(estimate.unconstrained)
    if unc.dtype != np.bool_ or unc.shape != (k,):
        raise ValueError(f"unconstrained must be bool of shape ({k},), got {unc.dtype} {unc.shape}")
    return unc
