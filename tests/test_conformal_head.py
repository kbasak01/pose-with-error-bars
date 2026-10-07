"""`ConformalPoseHead`: artifact round trip, hash locks, and per-frame equivalence with evaluation.

The equivalence tests are the point: for every deployable score and both conventions, the head's
per-frame `Answer`/`Abstain` reproduces `metrics.outcomes` on the batch scores the evaluation
computes, frame for frame, and its set radii equal the batch `set_fn`'s bit for bit. No dataset.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from poseconf.conformal import head as hd
from poseconf.conformal import propagate as pr
from poseconf.conformal.metrics import outcomes
from poseconf.conformal.scores import SCORES
from poseconf.conformal.so3 import quat_to_matrix
from poseconf.conformal.split import conformal_quantile
from poseconf.engine import level_b, level_c

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = REPO_ROOT / "results" / "calibration"

K = 11
ORDER = "scalar_first"
GEOM = pr.CameraGeometry(
    wireframe=np.random.default_rng(0).uniform(-0.6, 0.6, size=(K, 3)),
    camera_matrix=np.array([[2988.58, 0.0, 960.0], [0.0, 2988.34, 600.0], [0.0, 0.0, 1.0]]),
    distortion=np.array([-0.2238, 0.5141, -0.000665, -0.000214, -0.1312]),
    width=1920,
    height=1200,
    min_depth=1e-9,
    max_normalised_radius=1.6,
)
CONVENTIONS = ("answer_required", "abstain_allowed")
P1_SHA = "a" * 64
VHEAD_SHA = "b" * 64
DETECTOR_SHA = "e" * 64
NORMALISERS = {
    "A1": {"c_R": 0.012, "c_t": 0.0033},
    "A2": {"c_R": 0.012, "c_z": 0.0031, "c_xy": 0.00057},
    "A3": {"c_R": 0.012, "c_t": 0.0033, "g_x": [0.2, 0.5, 0.8], "g_y": [2.0, 1.2, 0.7]},
    "B1": {},
    "C1": {},
    "C2": {},
}


def _frames(n: int = 300, seed: int = 7) -> level_b.KeypointFrames:
    """Noisy predictions of random poses, with PnP failures and a few unconstrained keypoints."""
    rng = np.random.default_rng(seed)
    q_gt = np.roll(Rotation.random(n, random_state=seed).as_quat(), 1, axis=1)
    t_gt = np.column_stack([rng.uniform(-0.6, 0.6, (n, 2)), rng.uniform(4.0, 15.0, n)])
    y_gt, include = pr.project(quat_to_matrix(q_gt, ORDER), t_gt, GEOM)
    noise = rng.normal(scale=0.02, size=(n, 3))
    q_hat = np.roll(
        (Rotation.from_rotvec(noise) * Rotation.from_quat(np.roll(q_gt, -1, axis=1))).as_quat(),
        1,
        axis=1,
    )
    t_hat = t_gt * (1.0 + rng.normal(scale=0.01, size=(n, 1)))
    sd = rng.uniform(1.0, 6.0, size=(n, K, 1))
    y_hat = np.nan_to_num(y_gt, nan=500.0) + rng.normal(size=(n, K, 2)) * sd
    a = rng.normal(size=(n, K, 2, 2))
    cov = a @ a.transpose(0, 1, 3, 2) * sd[..., None] ** 2 + 0.5 * np.eye(2)
    unconstrained = rng.uniform(size=(n, K)) < 0.03
    valid = rng.uniform(size=n) > 0.1
    # A few solved frames with only two constrained keypoints: C2's sigma is undefined there.
    unconstrained[np.flatnonzero(valid)[:6], 2:] = True
    q_hat[~valid], t_hat[~valid] = np.nan, np.nan
    return level_b.KeypointFrames(
        filenames=np.array([f"img{i:06d}.jpg" for i in range(n)]),
        y_hat=y_hat,
        cov=cov,
        unconstrained=unconstrained,
        d_hat=np.zeros(n),  # replaced below from the boxes, as the head derives it
        valid=valid,
        q_hat=q_hat,
        t_hat=t_hat,
        q_gt=q_gt,
        t_gt=t_gt,
        y_gt=y_gt,
        include=include,
    )


FRAMES = _frames()
_RNG = np.random.default_rng(11)
_CORNER = np.column_stack([_RNG.uniform(0, 800, len(FRAMES)), _RNG.uniform(0, 500, len(FRAMES))])
BOXES = np.column_stack([_CORNER, _CORNER + _RNG.uniform(150, 700, (len(FRAMES), 2))])
D_HAT = np.hypot(BOXES[:, 2] - BOXES[:, 0], BOXES[:, 3] - BOXES[:, 1])
CONF = _RNG.uniform(0.1, 0.9, len(FRAMES))
FRAMES = level_b.KeypointFrames(**{**FRAMES.__dict__, "d_hat": D_HAT})
SIGMA = level_c.pose_sigma(level_b.linearised_frames("C1", FRAMES, GEOM, ORDER)[0])


def _batch_scores(score_id: str, convention: str) -> tuple[np.ndarray, np.ndarray]:
    """(scores, valid) exactly as the evaluation computes them."""
    f = FRAMES
    if score_id in ("B1", "C1"):
        return level_b.score_frames(score_id, f, convention), f.valid
    if score_id == "C2":
        return level_c.c2_scores(f, SIGMA, convention), f.valid & SIGMA.ok
    norm = dict(NORMALISERS[score_id])
    if score_id == "A3":
        norm = {
            "c_R": norm["c_R"],
            "c_t": norm["c_t"],
            "difficulty": np.interp(CONF, norm["g_x"], norm["g_y"]),
        }
    scores = SCORES[score_id].score_fn(
        f.q_hat, f.t_hat, f.q_gt, f.t_gt, f.valid, convention=convention, **norm
    )
    return scores, f.valid


def _estimate(i: int) -> hd.FrameEstimate:
    f = FRAMES
    return hd.FrameEstimate(
        valid=bool(f.valid[i]),
        q_hat=f.q_hat[i],
        t_hat=f.t_hat[i],
        keypoints=f.y_hat[i],
        confidence_mean=float(CONF[i]),
        bbox=BOXES[i],
        unconstrained=f.unconstrained[i],
    )


def _artifact(score_id: str, convention: str, q: float) -> dict:
    return hd.build_artifact(
        score_id=score_id,
        alpha=0.1,
        convention=convention,
        quantile=q,
        n_cal=1000,
        normalisers=NORMALISERS[score_id],
        quaternion_order=ORDER,
        p1_checkpoint_sha256=P1_SHA,
        detector_checkpoint_sha256=DETECTOR_SHA,
        variance_head_sha256=VHEAD_SHA if score_id in ("C1", "C2") else None,
        geometry=GEOM if score_id == "C2" else None,
        context={"run": "test", "crop_source": "predicted_crop"},
        provenance={"note": "unit test"},
    )


def _load(artifact: dict) -> hd.ConformalPoseHead:
    return hd.ConformalPoseHead.from_dict(
        json.loads(json.dumps(artifact)),
        p1_checkpoint_sha256=P1_SHA,
        detector_sha256=DETECTOR_SHA,
        variance_head_sha256=VHEAD_SHA,
    )


def _head_outcomes(head: hd.ConformalPoseHead) -> tuple[np.ndarray, np.ndarray, list]:
    """(covered, answered, results) from the head, frame by frame, scored against the labels."""
    covered, answered, results = [], [], []
    for i in range(len(FRAMES)):
        out = head.predict(_estimate(i), cov=FRAMES.cov[i])
        results.append(out)
        if isinstance(out, hd.Abstain):
            covered.append(out.covered_if_abstained)
            answered.append(False)
            continue
        if out.set.__class__.__name__ == "PoseSet":
            inside = out.set.contains(FRAMES.q_gt[i][None], FRAMES.t_gt[i][None])[0]
        else:
            inside = out.set.contains(FRAMES.y_gt[i][None], FRAMES.include[i][None])[0]
        covered.append(bool(inside))
        answered.append(True)
    return np.array(covered), np.array(answered), results


@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("score_id", hd.DEPLOYABLE_SCORES)
def test_head_reproduces_batch_outcomes_frame_by_frame(score_id: str, convention: str) -> None:
    scores, valid = _batch_scores(score_id, convention)
    q = conformal_quantile(scores[:150], 0.1)
    head = _load(_artifact(score_id, convention, q))
    covered, answered, results = _head_outcomes(head)
    ref_covered, ref_answered = outcomes(scores, valid, q, convention)
    np.testing.assert_array_equal(answered, ref_answered)
    np.testing.assert_array_equal(covered, ref_covered)
    assert 0 < answered.sum() < len(FRAMES)
    if score_id == "C2":
        reasons = {r.reason for r in results if isinstance(r, hd.Abstain)}
        assert reasons == {"no_estimate", "sigma_undefined"}


@pytest.mark.parametrize("score_id", hd.DEPLOYABLE_SCORES)
def test_head_radii_equal_the_batch_set(score_id: str) -> None:
    scores, valid = _batch_scores(score_id, "abstain_allowed")
    q = conformal_quantile(scores[:150], 0.1)
    head = _load(_artifact(score_id, "abstain_allowed", q))
    f = FRAMES
    if score_id in ("B1", "C1"):
        batch = level_b.keypoint_set(score_id, f, q).radius_px
    elif score_id == "C2":
        batch = level_c.c2_set(f, SIGMA, q).rot_radius
    else:
        kwargs = dict(NORMALISERS[score_id])
        if score_id == "A3":
            kwargs = {
                "c_R": kwargs["c_R"],
                "c_t": kwargs["c_t"],
                "difficulty": np.interp(CONF, kwargs["g_x"], kwargs["g_y"]),
            }
        batch = SCORES[score_id].set_fn(f.q_hat, f.t_hat, f.valid, q, **kwargs).rot_radius
    for i in np.flatnonzero(valid):
        out = head.predict(_estimate(i), cov=f.cov[i])
        assert isinstance(out, hd.Answer)
        got = out.set.radius_px if score_id in ("B1", "C1") else out.set.rot_radius
        assert got[0] == batch[i]  # bit for bit


def test_artifact_round_trips_through_a_file(tmp_path: Path) -> None:
    p1, vh = tmp_path / "p1.pt", tmp_path / "vhead.pt"
    p1.write_bytes(b"p1 weights")
    vh.write_bytes(b"variance head weights")
    for q in (2.5, math.inf, -math.inf):
        artifact = hd.build_artifact(
            score_id="C2",
            alpha=0.1,
            convention="answer_required",
            quantile=q,
            n_cal=10,
            normalisers={},
            quaternion_order=ORDER,
            p1_checkpoint_sha256=hd.sha256_file(p1),
            variance_head_sha256=hd.sha256_file(vh),
            geometry=GEOM,
            context={"crop_source": "gt_crop"},
        )
        path = tmp_path / "c2.json"
        path.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
        head = hd.ConformalPoseHead.from_json(path, p1_checkpoint=p1, variance_head_checkpoint=vh)
        assert head.quantile == q
        assert head.calibration_id == artifact["calibration_id"]
        np.testing.assert_array_equal(head.geometry.wireframe, GEOM.wireframe)
        np.testing.assert_array_equal(head.geometry.distortion, GEOM.distortion)


def test_infinite_quantiles_abstain_or_cover_as_outcomes_does() -> None:
    i = int(np.flatnonzero(FRAMES.valid)[0])
    j = int(np.flatnonzero(~FRAMES.valid)[0])
    neg = _load(_artifact("A1", "answer_required", -math.inf))
    out = neg.predict(_estimate(i))
    assert isinstance(out, hd.Abstain) and out.reason == "q_neg_inf"
    assert not out.covered_if_abstained
    pos = _load(_artifact("A1", "answer_required", math.inf))
    assert isinstance(pos.predict(_estimate(i)), hd.Answer)
    failed = pos.predict(_estimate(j))
    assert isinstance(failed, hd.Abstain) and failed.reason == "no_estimate"
    assert failed.covered_if_abstained  # whole space under answer_required
    finite = _load(_artifact("A1", "answer_required", 3.0)).predict(_estimate(j))
    assert isinstance(finite, hd.Abstain) and not finite.covered_if_abstained
    for q in (-math.inf, math.inf, 3.0):  # abstain_allowed: every abstention is covered
        head = _load(_artifact("A1", "abstain_allowed", q))
        assert head.predict(_estimate(j)).covered_if_abstained
    neg = _load(_artifact("A1", "abstain_allowed", -math.inf)).predict(_estimate(i))
    assert isinstance(neg, hd.Abstain) and neg.reason == "q_neg_inf" and neg.covered_if_abstained


def test_refuses_a_different_p1_checkpoint() -> None:
    artifact = _artifact("A1", "abstain_allowed", 2.0)
    with pytest.raises(hd.CalibrationMismatchError, match="P1 checkpoint"):
        hd.ConformalPoseHead.from_dict(
            artifact, p1_checkpoint_sha256="c" * 64, detector_sha256=DETECTOR_SHA
        )


def test_refuses_a_missing_or_different_detector() -> None:
    artifact = _artifact("B1", "abstain_allowed", 0.02)
    with pytest.raises(hd.CalibrationMismatchError, match="detector"):
        hd.ConformalPoseHead.from_dict(artifact, p1_checkpoint_sha256=P1_SHA)
    with pytest.raises(hd.CalibrationMismatchError, match="detector"):
        hd.ConformalPoseHead.from_dict(
            artifact, p1_checkpoint_sha256=P1_SHA, detector_sha256="f" * 64
        )


def test_refuses_a_rehashed_but_invalid_payload() -> None:
    artifact = _artifact("A1", "abstain_allowed", 2.0)
    artifact["payload"]["convention"] = "answer_if_convenient"
    artifact["calibration_id"] = hd.calibration_id(artifact["payload"])
    with pytest.raises(hd.CalibrationMismatchError, match="convention"):
        _load(artifact)


def test_c2_rejects_a_covariance_that_is_not_positive_definite() -> None:
    i = int(np.flatnonzero(FRAMES.valid & SIGMA.ok)[0])
    head = _load(_artifact("C2", "abstain_allowed", 2.0))
    bad = FRAMES.cov[i].copy()
    k = int(np.flatnonzero(~FRAMES.unconstrained[i])[0])
    bad[k] = -bad[k]
    with pytest.raises(ValueError, match="positive-definite"):
        head.predict(_estimate(i), cov=bad)
    with pytest.raises(ValueError):
        head.predict(_estimate(i), cov=FRAMES.cov[i][:3])


def test_refuses_a_different_checkpoint_file(tmp_path: Path) -> None:
    good, bad = tmp_path / "good.pt", tmp_path / "bad.pt"
    good.write_bytes(b"weights")
    bad.write_bytes(b"weights, retrained")
    artifact = hd.build_artifact(
        score_id="A1",
        alpha=0.1,
        convention="abstain_allowed",
        quantile=2.0,
        n_cal=10,
        normalisers=NORMALISERS["A1"],
        quaternion_order=ORDER,
        p1_checkpoint_sha256=hd.sha256_file(good),
        context={"crop_source": "gt_crop"},
    )
    path = tmp_path / "a1.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert hd.ConformalPoseHead.from_json(path, p1_checkpoint=good).quantile == 2.0
    with pytest.raises(hd.CalibrationMismatchError, match="P1 checkpoint"):
        hd.ConformalPoseHead.from_json(path, p1_checkpoint=bad)


@pytest.mark.parametrize("score_id", ["C1", "C2"])
def test_refuses_a_missing_or_different_variance_head(score_id: str) -> None:
    artifact = _artifact(score_id, "abstain_allowed", 2.0)
    with pytest.raises(hd.CalibrationMismatchError, match="variance-head"):
        hd.ConformalPoseHead.from_dict(
            artifact, p1_checkpoint_sha256=P1_SHA, detector_sha256=DETECTOR_SHA
        )
    with pytest.raises(hd.CalibrationMismatchError, match="variance-head"):
        hd.ConformalPoseHead.from_dict(
            artifact,
            p1_checkpoint_sha256=P1_SHA,
            detector_sha256=DETECTOR_SHA,
            variance_head_sha256="d" * 64,
        )


def test_refuses_an_edited_payload() -> None:
    artifact = _artifact("A1", "abstain_allowed", 2.0)
    artifact["payload"]["quantile"] = 9.0
    with pytest.raises(hd.CalibrationMismatchError, match="calibration_id"):
        _load(artifact)


def test_refuses_an_unknown_schema_or_score_version(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _artifact("A1", "abstain_allowed", 2.0)
    with pytest.raises(hd.CalibrationMismatchError, match="schema"):
        _load({**artifact, "schema": "poseconf.calibration.v0"})
    monkeypatch.setitem(hd.SCORE_VERSIONS, "A1", 2)
    with pytest.raises(hd.CalibrationMismatchError, match="score version"):
        _load(artifact)


def test_b2_is_not_deployable() -> None:
    with pytest.raises(ValueError, match="not deployable"):
        hd.build_artifact(
            score_id="B2",
            alpha=0.1,
            convention="abstain_allowed",
            quantile=1.0,
            n_cal=10,
            normalisers={},
            quaternion_order=ORDER,
            p1_checkpoint_sha256=P1_SHA,
            context={"crop_source": "gt_crop"},
        )


def test_build_refuses_wrong_normalisers_and_locks() -> None:
    with pytest.raises(ValueError, match="normalisers"):
        hd.build_artifact(
            score_id="A2",
            alpha=0.1,
            convention="abstain_allowed",
            quantile=1.0,
            n_cal=10,
            normalisers=NORMALISERS["A1"],
            quaternion_order=ORDER,
            p1_checkpoint_sha256=P1_SHA,
            context={"crop_source": "gt_crop"},
        )
    with pytest.raises(ValueError, match="variance_head_sha256"):
        hd.build_artifact(
            score_id="C1",
            alpha=0.1,
            convention="abstain_allowed",
            quantile=1.0,
            n_cal=10,
            normalisers={},
            quaternion_order=ORDER,
            p1_checkpoint_sha256=P1_SHA,
            context={"crop_source": "gt_crop"},
        )
    with pytest.raises(ValueError, match="detector_checkpoint_sha256"):
        hd.build_artifact(
            score_id="A1",
            alpha=0.1,
            convention="abstain_allowed",
            quantile=1.0,
            n_cal=10,
            normalisers=NORMALISERS["A1"],
            quaternion_order=ORDER,
            p1_checkpoint_sha256=P1_SHA,
            context={"crop_source": "predicted_crop"},
        )


def test_covariance_scores_need_cov_on_valid_frames() -> None:
    i = int(np.flatnonzero(FRAMES.valid)[0])
    head = _load(_artifact("C1", "abstain_allowed", 2.0))
    with pytest.raises(ValueError, match="covariances"):
        head.predict(_estimate(i))


def test_predict_takes_no_label() -> None:
    assert list(inspect.signature(hd.ConformalPoseHead.predict).parameters) == [
        "self",
        "estimate",
        "cov",
    ]
    names = {f.name for f in fields(hd.FrameEstimate)}
    assert not {n for n in names if "gt" in n or "label" in n or "true" in n}


def test_calibration_id_is_deterministic() -> None:
    a = _artifact("A3", "abstain_allowed", 2.0)
    b = _artifact("A3", "abstain_allowed", 2.0)
    assert a["calibration_id"] == b["calibration_id"]
    assert a["calibration_id"] != _artifact("A3", "abstain_allowed", 2.5)["calibration_id"]


# --------------------------------------------------------------------------------------------------
# committed artifacts
# --------------------------------------------------------------------------------------------------

COMMITTED = sorted(p for p in CALIBRATION_DIR.glob("*.json") if p.name != "index.json")


@pytest.mark.parametrize("path", COMMITTED, ids=[p.stem for p in COMMITTED])
def test_committed_artifact_matches_its_source_row(path: Path) -> None:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    payload, prov = artifact["payload"], artifact["provenance"]
    assert hd.calibration_id(payload) == artifact["calibration_id"]
    source = REPO_ROOT / prov["source_result"]
    assert hd.sha256_file(source) == prov["source_result_sha256"], "source result changed: rebuild"
    rows = json.loads(source.read_text(encoding="utf-8"))["rows"]
    (row,) = [
        r
        for r in rows
        if r["score"] == payload["score_id"]
        and r["convention"] == payload["convention"]
        and r["alpha"] == payload["alpha"]
    ]
    assert payload["quantile"] == row["quantile"]
    assert payload["n_cal"] == row["n_cal"]
    record = json.loads(source.read_text(encoding="utf-8"))
    assert record["splits"]["calibration"] == payload["context"]["calibration_subset"]
    assert payload["context"]["pnp_config"] == record["provenance"]["pnp_config"]
    assert payload["locks"]["p1_checkpoint_sha256"] == record["provenance"]["p1_checkpoint_sha256"]
    if payload["score_id"] in ("A1", "A2", "A3"):
        fit = record["fit"]
        assert record["splits"]["fit"] == "synthetic_val_tune"
        assert payload["normalisers"]["c_R"] == fit["c_R_rad"]
        for key in ("c_t", "c_z", "c_xy"):
            if key in payload["normalisers"]:
                assert payload["normalisers"][key] == fit[key]
        if payload["score_id"] == "A3":
            assert payload["normalisers"]["g_x"] == fit["g"]["x"]
            assert payload["normalisers"]["g_y"] == fit["g"]["y"]
    manifest = json.loads((REPO_ROOT / "results" / "p1_checkpoints.json").read_text("utf-8"))
    detector = {c["run"]: c["sha256_release"] for c in manifest["checkpoints"]}["detector"]
    assert payload["locks"]["detector_checkpoint_sha256"] == detector


def test_committed_artifacts_cover_every_deployable_score() -> None:
    if not COMMITTED:
        pytest.skip("results/calibration/ not built yet (make calibration)")
    seen = {
        (a["payload"]["score_id"], a["payload"]["convention"])
        for a in (json.loads(p.read_text(encoding="utf-8")) for p in COMMITTED)
    }
    assert seen == {(s, c) for s in hd.DEPLOYABLE_SCORES for c in CONVENTIONS}
