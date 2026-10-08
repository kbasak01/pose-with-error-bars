"""Worked example: recalibrate `ConformalPoseHead` for a new target with a different wireframe.

Synthetic data only, CPU, no dataset and no SPEED+ constant (VALIDATION_CHECKLIST J2). It walks
the porting recipe in `docs/RECALIBRATION_EXAMPLE.md` end to end for a stand-in ship deck: 6
deck markers instead of the 11 Tango keypoints, a different camera, and a keypoint model whose
predicted covariances are deliberately too small and whose errors are heavy-tailed.

1. Simulate labelled calibration frames of the new target: predicted keypoints, predicted
   covariances, visibility, and PnP failures.
2. Score them with C1 (`score_mahalanobis`) and take the split-conformal quantile.
3. Package an artifact (`build_artifact`) locked to the new model's checkpoint hashes.
4. Load it (`ConformalPoseHead.from_dict`) and serve fresh test frames one at a time, label-free.
5. Measure coverage on the test frames, and compare it with trusting Σ̂ as a Gaussian.

    python scripts/recalibration_example.py --out results/validity/recalibration_example.json
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from poseconf.conformal import head as hd
from poseconf.conformal import propagate as pr
from poseconf.conformal.metrics import clopper_pearson
from poseconf.conformal.scores import score_mahalanobis
from poseconf.conformal.split import conformal_quantile
from poseconf.provenance import RESULT_SCHEMA, poseconf_git_sha, utc_now_iso, write_result_json

#: The stand-in target: 6 deck markers (metres, deck frame, z up out of the deck).
DECK_MARKERS = np.array(
    [
        [-10.0, -6.0, 0.0],
        [10.0, -6.0, 0.0],
        [10.0, 6.0, 0.0],
        [-10.0, 6.0, 0.0],
        [0.0, -6.0, 0.0],
        [0.0, 3.0, 1.5],  # a raised marker, so the target is not exactly planar
    ]
)
#: The stand-in camera: 1280 x 720, no distortion.
CAMERA = pr.CameraGeometry(
    wireframe=DECK_MARKERS,
    camera_matrix=np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),
    distortion=None,
    width=1280,
    height=720,
    min_depth=0.1,
    max_normalised_radius=10.0,
)
QUATERNION_ORDER = "scalar_first"
#: Simulation of the new keypoint model (all invented; this is a mechanics demonstration).
RANGE_M = (30.0, 80.0)
LATERAL_M = 6.0
TILT_DEG = 15.0
SIGMA_PX = (0.8, 4.0)
#: True errors are this much wider than the predicted Σ̂ and Student-t: Σ̂ is miscalibrated.
TRUE_SCALE = 1.6
STUDENT_DOF = 4
FAILURE_RATE = 0.05
#: Stand-ins for the new pipeline's checkpoint files. The head locks to their SHA-256.
CHECKPOINTS = {
    "keypoint_model": b"deck-keypoint-model-v0",
    "roi_detector": b"deck-roi-detector-v0",
    "variance_head": b"deck-variance-head-v0",
}


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def simulate(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Labelled frames of the deck: predictions, predicted covariances, labels, visibility."""
    rotvec = rng.normal(size=(n, 3))
    rotvec *= np.deg2rad(rng.uniform(0.0, TILT_DEG, n) / np.linalg.norm(rotvec, axis=1))[:, None]
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    translation = np.column_stack(
        [rng.uniform(-LATERAL_M, LATERAL_M, (n, 2)), rng.uniform(*RANGE_M, n)]
    )
    y_gt, visible = pr.project(rotation, translation, CAMERA)
    k = len(DECK_MARKERS)
    sx, sy = rng.uniform(*SIGMA_PX, (n, k)), rng.uniform(*SIGMA_PX, (n, k))
    rho = rng.uniform(-0.6, 0.6, (n, k))
    cov = np.empty((n, k, 2, 2))
    cov[..., 0, 0], cov[..., 1, 1] = sx**2, sy**2
    cov[..., 0, 1] = cov[..., 1, 0] = rho * sx * sy
    chol = np.linalg.cholesky(cov)
    t = rng.standard_t(STUDENT_DOF, (n, k, 2)) * math.sqrt((STUDENT_DOF - 2) / STUDENT_DOF)
    y_hat = y_gt + TRUE_SCALE * np.einsum("nkij,nkj->nki", chol, t)
    valid = rng.uniform(size=n) >= FAILURE_RATE
    # A pose estimate for the frame record; C1 never reads it.
    q_hat = np.roll(Rotation.from_matrix(rotation).as_quat(), 1, axis=1)
    return {
        "y_hat": y_hat,
        "cov": cov,
        "y_gt": y_gt,
        "include": visible,
        "valid": valid,
        "q_hat": q_hat,
        "t_hat": translation,
    }


def calibrate(frames: dict[str, np.ndarray], alpha: float, convention: str) -> dict[str, Any]:
    """Steps 2–3: C1 scores on the calibration frames, the quantile, and a locked artifact."""
    scores = score_mahalanobis(
        frames["y_hat"],
        frames["y_gt"],
        frames["include"],
        frames["valid"],
        cov=frames["cov"],
        convention=convention,
    )
    q = conformal_quantile(scores, alpha)
    return hd.build_artifact(
        score_id="C1",
        alpha=alpha,
        convention=convention,
        quantile=q,
        n_cal=len(scores),
        normalisers={},
        quaternion_order=QUATERNION_ORDER,
        # The P1-named slots hold the new pipeline's hashes (docs/LIMITATIONS.md, J1).
        p1_checkpoint_sha256=_sha(CHECKPOINTS["keypoint_model"]),
        detector_checkpoint_sha256=_sha(CHECKPOINTS["roi_detector"]),
        variance_head_sha256=_sha(CHECKPOINTS["variance_head"]),
        context={"crop_source": "predicted_crop", "target": "synthetic_ship_deck_6_markers"},
        provenance={"example": "scripts/recalibration_example.py"},
    )


def serve(head: hd.ConformalPoseHead, frames: dict[str, np.ndarray]) -> dict[str, Any]:
    """Steps 4–5: one `predict` per frame from predictions only, then score against the labels."""
    covered, answered, radius = [], [], []
    for i in range(len(frames["valid"])):
        estimate = hd.FrameEstimate(
            valid=bool(frames["valid"][i]),
            q_hat=frames["q_hat"][i],
            t_hat=frames["t_hat"][i],
            keypoints=frames["y_hat"][i],
            confidence_mean=float("nan"),
            bbox=np.r_[frames["y_hat"][i].min(0), frames["y_hat"][i].max(0)],
        )
        out = head.predict(estimate, cov=frames["cov"][i])
        if isinstance(out, hd.Answer):
            hit = out.set.contains(frames["y_gt"][i][None], frames["include"][i][None])[0]
            covered.append(bool(hit))
            answered.append(True)
            radius.append(float(out.set.radius_px[0]))
        else:
            covered.append(out.covered_if_abstained)
            answered.append(False)
    covered_arr, answered_arr = np.array(covered), np.array(answered)
    m, k = len(covered_arr), int(covered_arr.sum())
    return {
        "n_total": m,
        "n_answered": int(answered_arr.sum()),
        "n_covered": k,
        "coverage": k / m,
        "coverage_ci95": list(clopper_pearson(k, m)),
        "answer_rate": float(answered_arr.mean()),
        "silent_failure_rate": float((answered_arr & ~covered_arr).mean()),
        "radius_px_median": float(np.median(radius)),
    }


def repeated_draws(
    alpha: float, convention: str, n_cal: int, n_test: int, repeats: int, rng: np.random.Generator
) -> dict[str, Any]:
    """The guarantee is marginal over the calibration draw: mean coverage over fresh draws.

    Each draw simulates new calibration and test frames, recalibrates, and measures test coverage
    as `score <= q` (the head's decision rule, frame-for-frame equal to `predict` by
    `tests/test_conformal_head.py`).
    """
    coverage = np.empty(repeats)
    for r in range(repeats):
        cal, test = simulate(n_cal, rng), simulate(n_test, rng)
        scores = [
            score_mahalanobis(
                f["y_hat"], f["y_gt"], f["include"], f["valid"], cov=f["cov"], convention=convention
            )
            for f in (cal, test)
        ]
        coverage[r] = float(np.mean(scores[1] <= conformal_quantile(scores[0], alpha)))
    mean, sd = float(coverage.mean()), float(coverage.std(ddof=1))
    return {
        "repeats": repeats,
        "n_cal": n_cal,
        "n_test": n_test,
        "mean_coverage": mean,
        "sd_coverage": sd,
        "mc_se": sd / math.sqrt(repeats),
        "min_coverage": float(coverage.min()),
        "max_coverage": float(coverage.max()),
        "nominal": 1.0 - alpha,
        "upper_bound_continuous": 1.0 - alpha + 1.0 / (n_cal + 1),
    }


def naive_gaussian(frames: dict[str, np.ndarray], alpha: float) -> dict[str, Any]:
    """What trusting Σ̂ as an exact Gaussian gives: Bonferroni χ²₂ ellipses, no calibration."""
    k = len(DECK_MARKERS)
    q = math.sqrt(chi2.ppf(1.0 - alpha / k, df=2))
    ok = frames["valid"]
    s = score_mahalanobis(
        frames["y_hat"][ok],
        frames["y_gt"][ok],
        frames["include"][ok],
        np.ones(int(ok.sum()), dtype=bool),
        cov=frames["cov"][ok],
        convention="abstain_allowed",
    )
    hits = int((s <= q).sum())
    return {
        "q": q,
        "n_answered": int(ok.sum()),
        "coverage_given_answered": hits / int(ok.sum()),
        "coverage_given_answered_ci95": list(clopper_pearson(hits, int(ok.sum()))),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-cal", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--convention", default="abstain_allowed", choices=hd.CONVENTIONS)
    parser.add_argument("--repeats", type=int, default=200, help="fresh cal/test draws")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--out", type=Path, default=None, help="result JSON (default: print only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the example; return the process exit code."""
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    cal, test = simulate(args.n_cal, rng), simulate(args.n_test, rng)
    artifact = calibrate(cal, args.alpha, args.convention)
    head = hd.ConformalPoseHead.from_dict(
        artifact,
        p1_checkpoint_sha256=_sha(CHECKPOINTS["keypoint_model"]),
        detector_sha256=_sha(CHECKPOINTS["roi_detector"]),
        variance_head_sha256=_sha(CHECKPOINTS["variance_head"]),
    )
    try:  # a retrained keypoint model must not silently reuse the old quantile
        hd.ConformalPoseHead.from_dict(
            artifact,
            p1_checkpoint_sha256=_sha(b"deck-keypoint-model-v1"),
            detector_sha256=_sha(CHECKPOINTS["roi_detector"]),
            variance_head_sha256=_sha(CHECKPOINTS["variance_head"]),
        )
        refused = False
    except hd.CalibrationMismatchError:
        refused = True
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "validity",
        "stage": "recalibration_example",
        "score": "C1",
        "arm": "split",
        "convention": args.convention,
        "alpha": args.alpha,
        "domain": "synthetic_ship_deck_example",
        "oracle": False,
        "definitions": {
            "what": (
                "Mechanics demonstration on invented synthetic data: a 6-marker deck target and "
                "a 1280x720 camera, not SPEED+. Predicted covariances are 1.6x too small and the "
                "errors Student-t(4). Coverage is split-conformal on exchangeable simulated "
                "frames; it says nothing about a real deck."
            ),
            "repeated_draws": (
                "Fresh calibration and test draws, recalibrated each time. The split-conformal "
                "guarantee is on this mean (>= 1 - alpha; < 1 - alpha + 1/(n_cal + 1) for "
                "continuous scores, which the abstention atoms are not); one draw, like `test`, "
                "can sit below 1 - alpha."
            ),
            "naive_gaussian": (
                "Bonferroni chi-square(2) ellipses from the uncalibrated covariance, on answered "
                "frames: what trusting the variance head as a Gaussian would give."
            ),
        },
        "n_keypoints": len(DECK_MARKERS),
        "n_cal": args.n_cal,
        "quantile": head.quantile,
        "calibration_id": head.calibration_id,
        "test": serve(head, test),
        "naive_gaussian": naive_gaussian(test, args.alpha),
        "repeated_draws": repeated_draws(
            args.alpha, args.convention, args.n_cal, args.n_test, args.repeats, rng
        ),
        "refuses_retrained_model": refused,
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "created_at": utc_now_iso(),
            "seed": args.seed,
        },
    }
    t = result["test"]
    print(
        f"C1 recalibrated on {args.n_cal} deck frames: q = {head.quantile:.3f}; "
        f"test coverage {t['coverage']:.4f} {t['coverage_ci95']} (n = {t['n_total']}, "
        f"answer rate {t['answer_rate']:.3f}); naive Gaussian on answered frames "
        f"{result['naive_gaussian']['coverage_given_answered']:.4f}; "
        f"retrained model refused: {refused}"
    )
    rep = result["repeated_draws"]
    print(
        f"{rep['repeats']} fresh draws: mean coverage {rep['mean_coverage']:.4f} "
        f"(MC s.e. {rep['mc_se']:.4f}, nominal {rep['nominal']:.2f})"
    )
    if args.out is not None:
        write_result_json(args.out, result)
    return 0 if refused else 1


if __name__ == "__main__":
    sys.exit(main())
