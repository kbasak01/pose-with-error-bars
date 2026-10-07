"""Phase 6 shift matrix: arms, weights, label-free sets, HIL label guards, result rows, tables."""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from scipy.special import expit

from poseconf import p1_adapter as p1
from poseconf.conformal import propagate as pr
from poseconf.conformal.metrics import outcomes
from poseconf.conformal.so3 import quat_to_matrix
from poseconf.conformal.split import conformal_quantile
from poseconf.conformal.weighted import weighted_conformal_quantiles, weights_from_probabilities
from poseconf.engine import level_a, level_b
from poseconf.engine import shift as sh
from poseconf.provenance import RESULT_SCHEMA, write_result_json

REPO_ROOT = Path(__file__).resolve().parents[1]
K = 11
GEOM = pr.CameraGeometry(
    wireframe=np.random.default_rng(0).uniform(-0.6, 0.6, size=(K, 3)),
    camera_matrix=np.array([[2988.58, 0.0, 960.0], [0.0, 2988.34, 600.0], [0.0, 0.0, 1.0]]),
    distortion=np.array([-0.2238, 0.5141, -0.000665, -0.000214, -0.1312]),
    width=1920,
    height=1200,
    min_depth=1e-9,
    max_normalised_radius=1.6,
)
ORDER = "scalar_first"


def _script(name: str):
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


def _dump_and_labels(n: int = 80, seed: int = 1) -> tuple[dict, dict, list[str]]:
    """A dump (with the variance-sidecar columns attached) and a label file, in P1's layout."""
    rng = np.random.default_rng(seed)
    q_gt = np.roll(Rotation.random(n, random_state=seed).as_quat(), 1, axis=1)
    z = rng.uniform(4.0, 15.0, n)
    t_gt = np.column_stack([rng.uniform(-0.1, 0.1, (n, 2)) * z[:, None], z])
    y_gt, visible = pr.project(quat_to_matrix(q_gt, ORDER), t_gt, GEOM)
    names = [f"img{i:05d}.jpg" for i in range(n)]
    labels = {
        "filename": np.asarray(names),
        "q_gt": q_gt,
        "t_gt": t_gt,
        "kp_gt_full": y_gt,
        "in_frame": visible,
        "bbox_gt": np.tile([100.0, 100.0, 600.0, 500.0], (n, 1)),
    }
    noise = Rotation.from_rotvec(rng.normal(scale=0.01, size=(n, 3)))
    q_hat = np.roll((noise * Rotation.from_quat(np.roll(q_gt, -1, axis=1))).as_quat(), 1, axis=1)
    t_hat = t_gt + rng.normal(scale=0.01, size=(n, 3)) * z[:, None]
    success = rng.uniform(size=n) > 0.15
    q_hat[~success], t_hat[~success] = np.nan, np.nan
    y_hat = np.nan_to_num(y_gt, nan=300.0) + rng.normal(scale=3.0, size=(n, K, 2))
    a = rng.normal(size=(n, K, 2, 2))
    cov = a @ a.transpose(0, 1, 3, 2) + 4.0 * np.eye(2)
    dump = {
        "filename": np.asarray(names),
        "kp_pred_full": y_hat,
        "heatmap_cov_full": cov,
        "vhead_cov_full": 2.0 * cov,
        "heatmap_empty": np.zeros((n, K), dtype=bool),
        "bbox_used": np.tile([120.0, 90.0, 610.0, 480.0], (n, 1)),
        "success": success,
        "q_pred": q_hat,
        "t_pred": t_hat,
        "confidence": rng.uniform(0.2, 0.9, size=(n, K)),
        "enc_feat": rng.normal(size=(n, 8)),
    }
    return dump, labels, names


@pytest.fixture(scope="module")
def frames() -> tuple[sh.ShiftFrames, sh.ShiftFrames, level_a.LevelAFit]:
    dump, labels, names = _dump_and_labels(160)
    tune, _ = sh.build_frames(dump, labels, names[:80], GEOM, ORDER)
    test, _ = sh.build_frames(dump, labels, names[80:], GEOM, ORDER)
    return tune, test, level_a.fit_level_a(sh.pose_frames(tune))


# --- arms ---------------------------------------------------------------------------------------


def test_split_frame_quantiles_broadcast() -> None:
    s = np.random.default_rng(0).exponential(size=99)
    q = sh.split_frame_quantiles(s, 0.1, 7)
    assert q.shape == (7,) and np.all(q == conformal_quantile(s, 0.1))


def test_mondrian_groups_and_failure_group(frames) -> None:
    tune, test, _ = frames
    bins = sh.fit_mondrian_bins(tune)
    assert bins.n_levels == (3, 3) and bins.failure_group == 9
    valid = sh.valid_mask("A1", test)
    groups = bins.groups(test, valid)
    assert np.all(groups[~valid] == 9) and np.all(groups[valid] < 9)
    rng = np.random.default_rng(1)
    s_cal = rng.exponential(size=len(test))
    q, table = sh.mondrian_frame_quantiles(s_cal, groups, groups, 0.2, bins.group_ids)
    for g in bins.group_ids:
        expected = conformal_quantile(s_cal[groups == g], 0.2) if (groups == g).any() else math.inf
        assert np.all(q[groups == g] == expected)
        assert table[str(g)]["n_cal"] == int((groups == g).sum())


def test_weighted_cp_with_true_logits_recovers_coverage_under_shift() -> None:
    """Covariate shift N(0,1) -> N(1,1) with |noise| growing in x: split CP under-covers, weighted
    CP with the true log density ratio (x - 1/2) is valid."""
    rng = np.random.default_rng(3)
    alpha, n, m, reps = 0.1, 600, 2000, 20
    cov_split, cov_weighted = [], []
    for _ in range(reps):
        x_cal, x_test = rng.normal(0, 1, n), rng.normal(1, 1, m)
        s_cal = np.abs(rng.normal(size=n)) * np.exp(x_cal)
        s_test = np.abs(rng.normal(size=m)) * np.exp(x_test)
        w_cal, w_test, _ = sh.likelihood_ratio_weights(x_cal - 0.5, x_test - 0.5, 1, 1)
        q_w = weighted_conformal_quantiles(s_cal, w_cal, w_test, alpha)
        cov_weighted.append(np.mean(s_test <= q_w))
        cov_split.append(np.mean(s_test <= conformal_quantile(s_cal, alpha)))
    assert np.mean(cov_split) < 0.85
    assert np.mean(cov_weighted) >= 1 - alpha - 0.01


def test_logit_shift_leaves_weighted_quantiles_unchanged() -> None:
    rng = np.random.default_rng(4)
    logit_cal, logit_test = rng.normal(size=300), rng.normal(0.5, 1, size=50)
    s = rng.exponential(size=300)
    w_cal, w_test, shift = sh.likelihood_ratio_weights(logit_cal, logit_test, 200, 100)
    ref_cal = weights_from_probabilities(expit(logit_cal), 200, 100)
    ref_test = weights_from_probabilities(expit(logit_test), 200, 100)
    np.testing.assert_allclose(w_cal * math.exp(shift), ref_cal, rtol=1e-10)
    for alpha in (0.05, 0.1, 0.3):
        np.testing.assert_array_equal(
            weighted_conformal_quantiles(s, w_cal, w_test, alpha),
            weighted_conformal_quantiles(s, ref_cal, ref_test, alpha),
        )
    # No overflow when the classifier separates the domains completely.
    w_cal, w_test, _ = sh.likelihood_ratio_weights(np.full(3, -800.0), np.full(2, 900.0), 1, 1)
    assert np.all(np.isfinite(w_cal)) and np.all(np.isfinite(w_test))


def test_domain_classifier_auc() -> None:
    rng = np.random.default_rng(5)
    same = sh.fit_domain_classifier(
        rng.normal(size=(300, 4)), rng.normal(size=(200, 4)),
        c=1.0, max_iter=500, standardise=True, cv_folds=5, cv_seed=0,
    )  # fmt: skip
    apart = sh.fit_domain_classifier(
        rng.normal(size=(300, 4)), rng.normal(4.0, 1.0, size=(200, 4)),
        c=1.0, max_iter=500, standardise=True, cv_folds=5, cv_seed=0,
    )  # fmt: skip
    assert len(same.cv_auc) == 5 and 0.35 < np.mean(same.cv_auc) < 0.65
    assert np.mean(apart.cv_auc) > 0.99
    assert apart.logits(np.full((1, 4), 4.0))[0] > 0 > apart.logits(np.zeros((1, 4)))[0]


def test_oracle_draws() -> None:
    a = sh.oracle_draws(100, 25, 5, 7, "lightbox")
    b = sh.oracle_draws(100, 25, 5, 7, "lightbox")
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert all(len(np.unique(x)) == 25 and x.min() >= 0 and x.max() < 100 for x in a)
    assert len({tuple(x) for x in a}) == 5
    assert not np.array_equal(a[0], sh.oracle_draws(100, 25, 1, 7, "sunlamp")[0])
    with pytest.raises(ValueError, match="cannot draw"):
        sh.oracle_draws(10, 25, 1, 7, "lightbox")
    assert sh.oracle_arm_tag(250) == "oracle_target_labels_n250"


# --- sets and sizes -----------------------------------------------------------------------------


@pytest.mark.parametrize("score_id", ["A1", "A2", "A3", "C2"])
def test_set_size_at_matches_level_a_at_scalar_q(frames, score_id: str) -> None:
    _, test, fit = frames
    q = 1.7
    if score_id == "C2":
        from poseconf.engine.level_c import c2_set

        pset = c2_set(test.kp, test.sigma, q)
    else:
        pset = level_a.pose_set(score_id, sh.pose_frames(test), fit, q)
    answered = sh.valid_mask(score_id, test)
    ref = level_a.set_size_report(pset, answered)
    got = sh.set_size_at(sh.unit_radii(score_id, test, fit), np.full(len(test), q), answered)
    for key in ref:
        for stat in ("median", "p90"):
            assert got[key][stat] == pytest.approx(ref[key][stat], rel=1e-12)


@pytest.mark.parametrize("score_id", ["B1", "B2", "C1"])
def test_set_size_at_matches_level_b_at_scalar_q(frames, score_id: str) -> None:
    _, test, fit = frames
    kp = test.c1 if score_id == "C1" else test.kp
    answered = kp.valid
    ref = level_b.set_size_report(level_b.keypoint_set(score_id, kp, 2.5), answered)
    got = sh.set_size_at(sh.unit_radii(score_id, test, fit), np.full(len(test), 2.5), answered)
    assert got["kp_px"]["median"] == pytest.approx(ref["kp_px"]["median"], rel=1e-12)


def test_set_size_keeps_infinity() -> None:
    unit = {"kp_px": np.array([1.0, np.inf, 2.0])}
    got = sh.set_size_at(unit, np.array([np.inf, 0.0, 1.0]), np.array([True, True, True]))
    assert got["kp_px"]["median"] == math.inf  # never NaN from 0 * inf
    assert sh.set_size_at(unit, np.zeros(3), np.zeros(3, bool)) is None


def test_test_labels_never_reach_sets_or_quantiles(frames) -> None:
    """Scramble every label column of the test frames: radii, groups and arm quantiles stay
    bit-identical (invariant 5)."""
    tune, test, fit = frames
    rng = np.random.default_rng(9)
    perm = rng.permutation(len(test))
    kp = test.kp
    scrambled_kp = level_b.KeypointFrames(
        **{
            **{f: getattr(kp, f) for f in kp.__dataclass_fields__},
            "q_gt": kp.q_gt[perm],
            "t_gt": kp.t_gt[perm] * 3.0,
            "y_gt": kp.y_gt[perm] + 50.0,
            "include": kp.include[perm],
        }
    )
    scrambled = sh.ShiftFrames(
        kp=scrambled_kp,
        cov_vhead=test.cov_vhead,
        conf=test.conf,
        bbox_used=test.bbox_used,
        bbox_gt=test.bbox_gt[perm] * 0.5,
        sigma=test.sigma,
    )
    bins = sh.fit_mondrian_bins(tune)
    for score_id in sh.SHIFT_SCORES:
        a, b = sh.unit_radii(score_id, test, fit), sh.unit_radii(score_id, scrambled, fit)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        valid = sh.valid_mask(score_id, test)
        np.testing.assert_array_equal(
            bins.groups(test, valid), bins.groups(scrambled, sh.valid_mask(score_id, scrambled))
        )


# --- rows -----------------------------------------------------------------------------------


@pytest.mark.parametrize("convention", ["abstain_allowed", "answer_required"])
def test_evaluate_row_fields_and_inf_roundtrip(frames, convention: str, tmp_path) -> None:
    tune, test, fit = frames
    for score_id in sh.SHIFT_SCORES:
        s_test = sh.scores(score_id, test, fit, convention)
        q = np.full(len(test), math.inf)
        q[::3] = 2.0
        row, covered, answered = sh.evaluate(
            score_id, test, s_test, q, sh.unit_radii(score_id, test, fit), convention
        )
        ref_cov, ref_ans = outcomes(s_test, sh.valid_mask(score_id, test), q, convention)
        assert np.array_equal(covered, ref_cov) and np.array_equal(answered, ref_ans)
        for field in (
            "n_total", "n_answered", "coverage", "coverage_ci95", "answer_rate",
            "silent_failure_rate", "set_size", "coverage_given_answered",
        ):  # fmt: skip
            assert field in row
        assert ("n_vacuous" in row) == (score_id in ("B1", "B2", "C1"))
        assert ("n_solved_sigma_undefined" in row) == (score_id == "C2")
        path = write_result_json(tmp_path / "r.json", {"schema": RESULT_SCHEMA, "rows": [row]})
        back = _script("make_tables")._decode(json.loads(path.read_text()))["rows"][0]
        assert back["set_size"] == row["set_size"]
    slices = sh.slice_labels(test, tune, with_iou=True)
    assert set(slices) == {"gt_range_tertile", "confidence_quintile", "gt_bbox_iou_bin"}
    assert "gt_bbox_iou_bin" not in sh.slice_labels(test, tune, with_iou=False)


# --- HIL label guards (CLAUDE.md invariant 4; DECISIONS L1, L2, W1, W2) --------------------------


def test_label_access_goes_through_the_guard(monkeypatch) -> None:
    calls = []

    def spy(root, run, domain, pool, *, tag):
        calls.append((domain, pool, tag))
        return {}, {}

    monkeypatch.setattr(sh, "load_dump_labels", spy)
    for domain in ("synthetic", "lightbox", "sunlamp"):
        sh.evaluation_labels(Path("."), "keypoint_a2", domain, tag="evaluation")
    assert calls == [
        ("synthetic", "all", "evaluation"),
        ("lightbox", "poolB", "evaluation"),
        ("sunlamp", "poolB", "evaluation"),
    ]
    calls.clear()
    sh.oracle_target_labels(Path("."), "keypoint_a2", "lightbox")
    assert calls == [("lightbox", "poolA", "oracle_target_labels")]
    with pytest.raises(ValueError, match="HIL poolA only"):
        sh.oracle_target_labels(Path("."), "keypoint_a2", "synthetic")


def test_real_guard_refuses_poola_without_oracle_tag(tmp_path) -> None:
    from poseconf.data.dump import load_dump_labels

    with pytest.raises(p1.HILLabelAccessError):
        load_dump_labels(tmp_path, "keypoint_a2", "lightbox", "poolA", tag="evaluation")
    with pytest.raises(p1.HILLabelAccessError):
        load_dump_labels(tmp_path, "keypoint_a2", "sunlamp", "poolA", tag=sh.WEIGHTED)


def test_weighted_arm_refuses_oracle_dumps() -> None:
    dump, _, names = _dump_and_labels(10)
    oracle_meta = {"oracle": True, "tag": "oracle_gt_box"}
    with pytest.raises(p1.HILLabelAccessError):
        sh.dump_features(dump, oracle_meta, names, arm=sh.WEIGHTED)
    with pytest.raises(p1.HILLabelAccessError):
        sh.require_label_free_dump(oracle_meta, sh.SPLIT)
    sh.require_label_free_dump(oracle_meta, sh.oracle_arm_tag(25))  # oracle arms may
    feats = sh.dump_features(dump, {"oracle": False}, names[::-1], arm=sh.WEIGHTED)
    np.testing.assert_array_equal(feats, dump["enc_feat"][::-1])


def test_script_reads_labels_only_through_the_shift_guards() -> None:
    source = (REPO_ROOT / "scripts" / "run_shift_matrix.py").read_text()
    assert "load_dump_labels" not in source
    assert "load_eval_labels" not in source and "load_sidecar" not in source
    assert source.count("oracle_target_labels(") == 1  # the oracle branch only
    engine = (REPO_ROOT / "src" / "poseconf" / "engine" / "shift.py").read_text()
    assert engine.count('"poolA"') == 1  # inside oracle_target_labels


# --- tables -------------------------------------------------------------------------------------


def _row(arm: str, domain: str, crop: str, oracle: bool, **extra) -> dict:
    return {
        "score": "A1", "arm": arm, "convention": "abstain_allowed", "alpha": 0.1,
        "domain": domain, "subset": "x", "crop_source": crop, "oracle": oracle,
        "n_total": 100, "n_answered": 50, "n_cal": 10, "coverage": 0.9,
        "coverage_ci95": [0.8, 0.95], "answer_rate": 0.5, "silent_failure_rate": 0.1,
        "coverage_given_answered": 0.8,
        "set_size": {"rot_deg": {"median": math.inf, "p90": math.inf},
                     "trans_frac": {"median": 0.01, "p90": 0.02}},
        **extra,
    }  # fmt: skip


def test_tables_keep_oracle_rows_in_their_own_section(tmp_path) -> None:
    files = {
        "a.json": [_row("split", "lightbox", "predicted_crop", False)],
        "b.json": [_row("split", "lightbox", "gt_crop", True, coverage=0.4242)],
        "c.json": [
            _row(
                "oracle_target_labels_n25",
                "lightbox",
                "predicted_crop",
                True,
                coverage=0.5151,
                coverage_min=0.4,
                coverage_max=0.6,
                n_draws=5,
            )
        ],  # fmt: skip
    }
    cells = []
    for name, rows in files.items():
        write_result_json(tmp_path / name, {"schema": RESULT_SCHEMA, "rows": rows})
        cells.append({"domain": "lightbox", "crop_source": rows[0]["crop_source"],
                      "arm": rows[0]["arm"], "file": name, "reason": None})  # fmt: skip
    cells.append({"domain": "synthetic", "crop_source": "predicted_crop", "arm": sh.WEIGHTED,
                  "file": None, "reason": "no target shift"})  # fmt: skip
    write_result_json(
        tmp_path / "index.json",
        {
            "schema": RESULT_SCHEMA, "run": "keypoint_a2", "cells": cells, "classifiers": {},
            "expected": {"alphas": [0.1], "conventions": ["abstain_allowed"]},
            "provenance": {"poseconf_git_sha": "0" * 40, "p1_commit": "1" * 40,
                           "split_manifest_sha256": "2" * 64},
        },
    )  # fmt: skip
    text = _script("make_tables").build(tmp_path)
    non_oracle = text.split("### Non-oracle rows")[1].split("### Oracle rows")[0]
    assert "0.9000" in non_oracle and "0.4242" not in non_oracle and "0.5151" not in non_oracle
    oracle = text.split("### Oracle rows")[1].split("## Few-label")[0]
    assert "0.4242" in oracle
    assert "0.5151 (0.4000–0.6000)" in text.split("## Few-label")[1]
    assert "∞" in non_oracle and "no target shift" in text
    for marker in {m.split("]")[0] for m in text.split("[^")[1:]}:
        assert f"[^{marker}]: " in text  # every dash has its footnote


# --- end to end on the real dumps ---------------------------------------------------------------


@pytest.mark.dataset
@pytest.mark.slow
def test_shift_matrix_end_to_end(local_paths, tmp_path) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "conformal.yaml").read_text())
    config["alpha_grid"] = [0.1]
    config["shift"].update(
        scores=["A1", "B2"], domains=["synthetic", "sunlamp"], crop_sources=["predicted_crop"]
    )
    config["arms"]["oracle_target_labels"]["n_values"] = [25]
    cfg_path = tmp_path / "conformal.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "shift"
    code = _script("run_shift_matrix").main(["--config", str(cfg_path), "--out-dir", str(out)])
    assert code == 0
    index = json.loads((out / "index.json").read_text())
    assert not index["unexplained"] and index["checks"]["predicted_crop_fit_vs_phase2"]["ok"]
    split = json.loads((out / "keypoint_a2_synthetic_predicted_crop_split.json").read_text())
    level_a_rows = json.loads(
        (REPO_ROOT / "results/level_a/keypoint_a2_predicted_crop_synthetic.json").read_text()
    )["rows"]
    for row in split["rows"]:
        if row["score"] == "A1":
            ref = next(
                r for r in level_a_rows
                if r["score"] == "A1" and r["convention"] == row["convention"] and r["alpha"] == 0.1
            )  # fmt: skip
            assert row["n_covered"] == ref["n_covered"]
    weighted = json.loads(
        (out / "keypoint_a2_sunlamp_predicted_crop_weighted_unlabeled_target.json").read_text()
    )
    assert weighted["classifier"]["target"] == "sunlamp_poolA"
    assert all(not r["oracle"] for r in weighted["rows"])
    oracle = json.loads(
        (out / "keypoint_a2_sunlamp_predicted_crop_oracle_target_labels_n25.json").read_text()
    )
    assert all(r["oracle"] and r["n_cal"] == 25 and len(r["draws"]) == 5 for r in oracle["rows"])
