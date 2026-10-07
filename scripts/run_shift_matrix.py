"""Phase 6: the coverage-under-shift matrix, every score x arm x convention x alpha x domain x crop.

Calibrates on synthetic `val_cal` (or, for `oracle_target_labels_n*`, on n labeled HIL poolA frames)
and evaluates on synthetic `val_test`, `lightbox_poolB` and `sunlamp_poolB`. Reads the Phase 3 dumps
and the Phase 5 variance sidecars (gitignored `dumps/`). CPU only, a few minutes.

Writes, under `results/shift/`:
* `<run>_<domain>_<crop>_<arm>.json` (kind `coverage`), one per domain x crop source x arm tag, with
  one row per score x convention x alpha;
* `index.json`: every expected cell, with its file or the reason it is empty.

    python scripts/run_shift_matrix.py --config configs/conformal.yaml

Exit code 1 if an expected cell is neither written nor explained, or a fit check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from poseconf.conformal.split import CONVENTIONS
from poseconf.conformal.weighted import weighted_conformal_quantiles
from poseconf.data.dump import (
    dump_file,
    labels_file,
    load_dump,
    variance_dump_file,
)
from poseconf.data.splits import load_split, manifest_sha256
from poseconf.engine import level_a
from poseconf.engine import shift as sh
from poseconf.engine.level_c import attach_variance
from poseconf.p1_adapter import (
    HIL_DOMAINS,
    load_p1_config,
    p1_commit,
    p1_paths,
    pose_quaternion_order,
    projection_geometry,
)
from poseconf.provenance import (
    POSECONF_ROOT,
    RESULT_SCHEMA,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    write_result_json,
)

#: The tag under which evaluation labels (synthetic `all`, HIL poolB) are requested.
EVAL_TAG = "evaluation"

DEFINITIONS = {
    "coverage": (
        "fraction of evaluated frames covered under the row's convention (docs/SCORES.md); "
        "Clopper-Pearson 95 % interval over n_total frames. Marginal and finite-sample only under "
        "exchangeability of calibration and evaluation frames; on HIL poolB this is a measured "
        "number, not a guarantee"
    ),
    "answered": "frame has a point estimate for the score and q > -inf (an empty set abstains)",
    "silent_failure_rate": "P(answered and not covered)",
    "coverage_given_answered": (
        "covered and answered / answered (null when nothing is answered): not a conformal "
        "quantity, reported beside the marginal coverage because abstentions count as covered "
        "under abstain_allowed; oracle rows: mean over draws with answers"
    ),
    "mondrian_conventions": (
        "invalid frames form their own Mondrian group, whose quantile is +inf (answer_required: "
        "whole-space set) or -inf (abstain_allowed: abstain); both count as covered and the other "
        "groups are identical, so the two conventions give the same Mondrian rows by construction"
    ),
    "set_size": (
        "median / p90 over answered frames of q_i x the frame's unit radius; +inf kept "
        "(order statistics). rot_deg: geodesic radius; *_frac: fraction of ||t_hat||; kp_px: "
        "largest keypoint radius in full-frame px (+inf with an unconstrained keypoint)"
    ),
    "quantile": (
        "split / oracle draw: the finite-sample quantile ceil((n+1)(1-alpha))-th order "
        "statistic, +inf if that exceeds n; mondrian: per-group quantiles; weighted: per-frame "
        "quantiles (Tibshirani et al. 2019), summarised"
    ),
    "oracle": (
        "true for every HIL gt_crop row (the crop box is the GT box) and every "
        "oracle_target_labels_n* arm (calibrated on labeled HIL poolA frames)"
    ),
    "oracle_target_labels": (
        "split CP on n labeled poolA frames, D draws; top-level coverage, answer_rate and "
        "silent_failure_rate are the mean over draws, coverage_min/max the range, coverage_ci95 the "
        "envelope of the per-draw Clopper-Pearson intervals, n_answered the rounded mean, set_size "
        "each statistic's median over draws; every draw is listed in `draws`"
    ),
    "slices": (
        "split arm, slices_alpha only: coverage by GT range tertile and by confidence quintile "
        "(edges from val_tune), and by GT-bbox IoU of the crop box (<0.5, 0.5-0.8, >0.8; "
        "predicted_crop only). GT quantities slice the evaluation; they never enter a score"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--out-dir", type=Path, default=None, help="default: shift.out_dir")
    return parser.parse_args(argv)


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(POSECONF_ROOT))
    except ValueError:
        return path.name


def _tagged(domain: str, names: list[str]) -> set[tuple[str, str]]:
    return {(domain, str(name)) for name in names}


def _checkpoint_sha(run: str) -> str:
    record = json.loads((POSECONF_ROOT / "results" / "p1_checkpoints.json").read_text("utf-8"))
    (entry,) = [c for c in record["checkpoints"] if c["run"] == run]
    if not entry["match"]:
        raise ValueError(f"{run} checkpoint does not match the P1 release manifest")
    return entry["sha256_local"]


class DumpCache:
    """Loads each domain x crop dump once, with its variance sidecar attached and checked."""

    def __init__(self, dumps_root: Path, run: str) -> None:
        self.root, self.run = dumps_root, run
        self._cache: dict[tuple[str, str], tuple[dict, dict, dict[str, str]]] = {}

    def get(self, domain: str, crop: str) -> tuple[dict, dict, dict[str, str]]:
        """`(arrays, meta, provenance)` of one dump."""
        key = (domain, crop)
        if key not in self._cache:
            path = dump_file(self.root, self.run, domain, crop, subset=False)
            side_path = variance_dump_file(self.root, self.run, domain, crop)
            dump, meta = load_dump(path)
            side, side_meta = load_dump(side_path)
            dump_sha = sha256_file(path)
            if side_meta["base_dump_sha256"] != dump_sha:
                raise ValueError(f"{side_path.name} was made from a different dump")
            if meta["domain"] != domain or meta["crop_source"] != crop:
                raise ValueError(f"{path.name} meta disagrees with ({domain}, {crop})")
            prov = {
                "dump": _rel(path),
                "dump_sha256": dump_sha,
                "variance_sidecar": _rel(side_path),
                "variance_sidecar_sha256": sha256_file(side_path),
                "variance_head_checkpoint_sha256": side_meta["variance_head"]["checkpoint_sha256"],
                "p1_checkpoint_sha256_dump": side_meta["provenance"]["p1_checkpoint_sha256"],
            }
            self._cache[key] = (attach_variance(dump, side), meta, prov)
        return self._cache[key]


def _label_prov(dumps_root: Path, run: str, domain: str, pool: str) -> dict[str, str]:
    path = labels_file(dumps_root, run, domain, pool)
    return {f"labels_{pool}": _rel(path), f"labels_{pool}_sha256": sha256_file(path)}


def _check_fit(fit: level_a.LevelAFit, conf: np.ndarray, path: Path, rtol: float) -> dict[str, Any]:
    """Refit (dump, float64) normalisers and g(conf) vs the committed Phase 2 fit (P1's sidecar,
    float32). g is compared as a function, on the `val_tune` confidences."""
    ref = json.loads(path.read_text("utf-8"))["fit"]
    pairs = {
        "c_R_rad": (fit.c_R, ref["c_R_rad"]),
        "c_t": (fit.c_t, ref["c_t"]),
        "c_z": (fit.c_z, ref["c_z"]),
        "c_xy": (fit.c_xy, ref["c_xy"]),
    }
    rel = {k: abs(a - b) / abs(b) for k, (a, b) in pairs.items()}
    g_ref = np.interp(conf, np.asarray(ref["g"]["x"]), np.asarray(ref["g"]["y"]))
    g_rel = float(np.max(np.abs(fit.difficulty(conf) - g_ref) / g_ref))
    worst = max(*rel.values(), g_rel)
    return {
        "reference": _rel(path),
        "why_not_bit_equal": "P1's sidecar stores q_pred, t_pred, confidence_mean as float32",
        "rel_diff": rel,
        "g_max_rel_diff_on_val_tune_conf": g_rel,
        "g_knots": [len(fit.g_x), len(ref["g"]["x"])],
        "rtol": rtol,
        "ok": worst <= rtol,
    }


def _base_row(
    score: str, arm: str, convention: str, alpha: float, domain: str, subset: str, crop: str,
    oracle: bool,
) -> dict[str, Any]:  # fmt: skip
    return {
        "score": score,
        "arm": arm,
        "convention": convention,
        "alpha": alpha,
        "domain": domain,
        "subset": subset,
        "crop_source": crop,
        "oracle": oracle,
        "valid_mask": "pnp_success_and_sigma_defined" if score == "C2" else "pnp_success",
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def _median_sizes(sizes: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    present = [s for s in sizes if s is not None]
    if not present:
        return None
    out: dict[str, Any] = {}
    for key in present[0]:
        out[key] = {
            stat: float(np.quantile([s[key][stat] for s in present], 0.5, method="inverted_cdf"))
            for stat in ("median", "p90")
        }
    out["n_draws_with_answers"] = len(present)
    return out


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    """Run the matrix; return the process exit code."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = config["shift"]
    out_dir = args.out_dir or (POSECONF_ROOT / cfg["out_dir"])
    alphas = [float(a) for a in config["alpha_grid"]]
    conventions = list(config["conventions"])
    score_ids = list(cfg["scores"])
    if not set(score_ids) <= set(sh.SHIFT_SCORES) or not set(conventions) <= set(CONVENTIONS):
        raise ValueError(f"unsupported scores {score_ids} or conventions {conventions}")
    run = cfg["run"]
    slices_alpha = float(cfg["slices_alpha"])
    oracle_cfg = config["arms"]["oracle_target_labels"]
    oracle_ns = [int(n) for n in oracle_cfg["n_values"]]
    n_draws = int(oracle_cfg["draws_per_n"])
    clf_cfg = cfg["classifier"]

    paths = p1_paths(args.paths)
    geometry = projection_geometry(run, paths=paths)
    order = pose_quaternion_order(run)
    cache = DumpCache(paths.dumps_root, run)
    names = {key: load_split(cfg[key]) for key in ("fit_split", "calibration_split")}
    tests = {domain: load_split(subset) for domain, subset in cfg["test_subsets"].items()}
    pools = {domain: load_split(pool) for domain, pool in cfg["target_pools"].items()}
    # Runtime disjointness, on (domain, filename): filenames repeat across domains. Fitting,
    # calibration and the classifier's target pool never contain an evaluation frame.
    evaluated = set().union(*(_tagged(d, test_names) for d, test_names in tests.items()))
    fitting = _tagged("synthetic", names["fit_split"]) | _tagged(
        "synthetic", names["calibration_split"]
    )
    target_pools = set().union(*(_tagged(d, pool) for d, pool in pools.items()))
    if (fitting | target_pools) & evaluated:
        raise RuntimeError("a fitting, calibration or poolA frame is also an evaluation frame")
    if set(names["fit_split"]) & set(names["calibration_split"]):
        raise RuntimeError("val_tune and val_cal overlap")

    common_prov = {
        "poseconf_git_sha": poseconf_git_sha(),
        "p1_commit": p1_commit(),
        "p1_checkpoint_sha256": _checkpoint_sha(run),
        "split_manifest_sha256": manifest_sha256(),
        "config_sha256": sha256_file(args.config),
        "pnp_config": load_p1_config(run)["pnp"],
    }
    syn_label_prov = _label_prov(paths.dumps_root, run, "synthetic", "all")
    syn_labels, _ = sh.evaluation_labels(paths.dumps_root, run, "synthetic", tag=EVAL_TAG)

    index: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    classifiers: dict[str, Any] = {}
    failed = False
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(domain: str, crop: str, arm: str, oracle: bool, body: dict[str, Any], prov: dict):
        stem = f"{run}_{domain}_{crop}_{arm}"
        result = {
            "schema": RESULT_SCHEMA,
            "kind": "coverage",
            "run": run,
            "domain": domain,
            "subset": cfg["test_subsets"][domain],
            "crop_source": crop,
            "arm": arm,
            "oracle": oracle,
            "definitions": DEFINITIONS,
            **body,
            "provenance": {**common_prov, **prov, "created_at": utc_now_iso()},
        }
        path = write_result_json(out_dir / f"{stem}.json", result)
        index.append(
            {"domain": domain, "crop_source": crop, "arm": arm, "oracle": oracle,
             "file": path.name, "n_rows": len(body["rows"]), "reason": None}
        )  # fmt: skip
        print(f"wrote {path.name} ({len(body['rows'])} rows)")

    def explain(domain: str, crop: str, arm: str, reason: str) -> None:
        index.append(
            {"domain": domain, "crop_source": crop, "arm": arm, "oracle": None, "file": None,
             "n_rows": 0, "reason": reason}
        )  # fmt: skip

    for crop in cfg["crop_sources"]:
        syn_dump, syn_meta, syn_prov = cache.get("synthetic", crop)
        tune, _ = sh.build_frames(syn_dump, syn_labels, names["fit_split"], geometry, order)
        cal, _ = sh.build_frames(syn_dump, syn_labels, names["calibration_split"], geometry, order)
        fit = level_a.fit_level_a(sh.pose_frames(tune))
        bins = sh.fit_mondrian_bins(tune)
        if crop == "predicted_crop":
            check = _check_fit(
                fit,
                tune.conf,
                POSECONF_ROOT / cfg["level_a_result"],
                float(cfg["normaliser_check_rtol"]),
            )
            checks["predicted_crop_fit_vs_phase2"] = check
            if not check["ok"]:
                print(f"FIT CHECK FAILED: {check}")
                failed = True
        fit_record = {"level_a": fit.as_dict(), "mondrian_bins": bins.as_dict()}
        cal_scores = {
            (s, c): sh.scores(s, cal, fit, c) for s in score_ids for c in conventions
        }  # fmt: skip
        cal_valid = {s: sh.valid_mask(s, cal) for s in score_ids}
        cal_groups = {s: bins.groups(cal, cal_valid[s]) for s in score_ids}

        for domain in cfg["domains"]:
            hil = domain in HIL_DOMAINS
            subset = cfg["test_subsets"][domain]
            if hil:
                dump, meta, dprov = cache.get(domain, crop)
                labels, _ = sh.evaluation_labels(paths.dumps_root, run, domain, tag=EVAL_TAG)
                label_prov = _label_prov(paths.dumps_root, run, domain, "poolB")
            else:
                dump, meta, dprov = syn_dump, syn_meta, syn_prov
                labels, label_prov = syn_labels, {}
            oracle_crop = bool(meta["oracle"])
            test, _ = sh.build_frames(dump, labels, tests[domain], geometry, order)
            m = len(test)
            prov = {
                "calibration_dump": syn_prov,
                "evaluation_dump": dprov,
                **syn_label_prov,
                **label_prov,
            }
            test_scores = {
                (s, c): sh.scores(s, test, fit, c) for s in score_ids for c in conventions
            }
            unit = {s: sh.unit_radii(s, test, fit) for s in score_ids}
            slices = sh.slice_labels(test, tune, with_iou=crop == "predicted_crop")
            calib = {
                "fit": cfg["fit_split"], "calibration": cfg["calibration_split"],
                "n_cal": len(cal), "evaluation": subset, "n_eval": m,
            }  # fmt: skip

            # ---- split
            rows = []
            for s in score_ids:
                for c in conventions:
                    for alpha in alphas:
                        q = sh.split_frame_quantiles(cal_scores[(s, c)], alpha, m)
                        row = _base_row(s, sh.SPLIT, c, alpha, domain, subset, crop, oracle_crop)
                        fields, covered, answered = sh.evaluate(
                            s, test, test_scores[(s, c)], q, unit[s], c
                        )
                        row.update(fields, quantile=float(q[0]), n_cal=len(cal))
                        if alpha == slices_alpha:
                            row["slices"] = sh.sliced_coverage(covered, answered, slices)
                        rows.append(row)
            write(domain, crop, sh.SPLIT, oracle_crop, {"calibration": calib, "fit": fit_record,
                                                         "rows": rows}, prov)  # fmt: skip

            # ---- mondrian
            rows = []
            for s in score_ids:
                g_test = bins.groups(test, sh.valid_mask(s, test))
                for c in conventions:
                    for alpha in alphas:
                        q, table = sh.mondrian_frame_quantiles(
                            cal_scores[(s, c)], cal_groups[s], g_test, alpha, bins.group_ids
                        )
                        row = _base_row(s, sh.MONDRIAN, c, alpha, domain, subset, crop, oracle_crop)
                        fields, _, _ = sh.evaluate(s, test, test_scores[(s, c)], q, unit[s], c)
                        row.update(fields, quantile=sh.q_summary(q), n_cal=len(cal))
                        row["groups"] = table
                        row["n_eval_per_group"] = {
                            str(g): int((g_test == g).sum()) for g in bins.group_ids
                        }
                        rows.append(row)
            write(domain, crop, sh.MONDRIAN, oracle_crop, {"calibration": calib, "fit": fit_record,
                                                            "rows": rows}, prov)  # fmt: skip

            if not hil:
                reason = "no target shift: synthetic val_test is the source distribution"
                explain(domain, crop, sh.WEIGHTED, reason)
                for n in oracle_ns:
                    explain(domain, crop, sh.oracle_arm_tag(n), reason)
                continue

            # ---- weighted_unlabeled_target (unlabeled poolA features only)
            if crop in cfg["weighted_crop_sources"]:
                source = sh.dump_features(syn_dump, syn_meta, names["fit_split"], arm=sh.WEIGHTED)
                target = sh.dump_features(dump, meta, pools[domain], arm=sh.WEIGHTED)
                trained = _tagged("synthetic", names["fit_split"]) | _tagged(domain, pools[domain])
                forbidden = _tagged("synthetic", names["calibration_split"]) | evaluated
                if trained & forbidden:
                    raise RuntimeError(
                        "domain-classifier training rows include an evaluation frame"
                    )
                clf = sh.fit_domain_classifier(
                    source, target, c=float(clf_cfg["C"]), max_iter=int(clf_cfg["max_iter"]),
                    standardise=bool(clf_cfg["standardise"]), cv_folds=int(clf_cfg["cv_folds"]),
                    cv_seed=int(clf_cfg["cv_seed"]),
                )  # fmt: skip
                logit_cal = clf.logits(
                    sh.dump_features(
                        syn_dump, syn_meta, names["calibration_split"], arm=sh.WEIGHTED
                    )
                )
                logit_test = clf.logits(
                    sh.dump_features(dump, meta, tests[domain], arm=sh.WEIGHTED)
                )
                w_cal, w_test, shift = sh.likelihood_ratio_weights(
                    logit_cal, logit_test, clf.n_source, clf.n_target
                )
                classifier = {
                    "features": config["arms"]["weighted_unlabeled_target"]["features"],
                    "source": cfg["fit_split"], "target": cfg["target_pools"][domain],
                    "n_source": clf.n_source, "n_target": clf.n_target,
                    "model": "StandardScaler + LogisticRegression (L2)" if clf_cfg["standardise"]
                    else "LogisticRegression (L2)",
                    "C": float(clf_cfg["C"]), "max_iter": int(clf_cfg["max_iter"]),
                    "cv_folds": int(clf_cfg["cv_folds"]), "cv_seed": int(clf_cfg["cv_seed"]),
                    "cv_auc": clf.cv_auc, "cv_auc_mean": float(np.mean(clf.cv_auc)),
                    "cv_auc_sd": float(np.std(clf.cv_auc, ddof=1)),
                    "logit_shift": shift,
                    "logit_cal_p50": float(np.median(logit_cal)),
                    "logit_eval_p50": float(np.median(logit_test)),
                    "weights": sh.weight_report(w_cal, w_test),
                }  # fmt: skip
                classifiers[domain] = classifier
                rows = []
                for s in score_ids:
                    for c in conventions:
                        for alpha in alphas:
                            q = weighted_conformal_quantiles(
                                cal_scores[(s, c)], w_cal, w_test, alpha
                            )
                            row = _base_row(s, sh.WEIGHTED, c, alpha, domain, subset, crop, False)
                            fields, _, _ = sh.evaluate(s, test, test_scores[(s, c)], q, unit[s], c)
                            row.update(fields, quantile=sh.q_summary(q), n_cal=len(cal))
                            row["ess_cal"] = classifier["weights"]["ess_cal"]
                            row["classifier_cv_auc_mean"] = classifier["cv_auc_mean"]
                            rows.append(row)
                write(domain, crop, sh.WEIGHTED, oracle_crop, {"calibration": calib,
                      "fit": fit_record, "classifier": classifier, "rows": rows}, prov)  # fmt: skip
            else:
                explain(
                    domain, crop, sh.WEIGHTED,
                    "gt_crop features are GT-box-conditioned on HIL (oracle_gt_box); the weighted "
                    "arm may consume unlabeled poolA only (DECISIONS, Phase 3 item L2)",
                )  # fmt: skip

            # ---- oracle_target_labels_n* (labeled poolA, oracle tag)
            pool_labels, _ = sh.oracle_target_labels(paths.dumps_root, run, domain)
            pool_frames, _ = sh.build_frames(dump, pool_labels, pools[domain], geometry, order)
            pool_scores = {
                (s, c): sh.scores(s, pool_frames, fit, c) for s in score_ids for c in conventions
            }  # fmt: skip
            oprov = {**prov, **_label_prov(paths.dumps_root, run, domain, "poolA")}
            for n in oracle_ns:
                arm = sh.oracle_arm_tag(n)
                draws = sh.oracle_draws(
                    len(pool_frames), n, n_draws, int(cfg["oracle_seed"]), domain
                )
                rows = []
                for s in score_ids:
                    for c in conventions:
                        for alpha in alphas:
                            per_draw = []
                            for d, idx in enumerate(draws):
                                q = sh.split_frame_quantiles(pool_scores[(s, c)][idx], alpha, m)
                                fields, _, _ = sh.evaluate(
                                    s, test, test_scores[(s, c)], q, unit[s], c
                                )
                                per_draw.append({"draw": d, **fields, "quantile": float(q[0])})
                            cov = np.array([p["coverage"] for p in per_draw])
                            row = _base_row(s, arm, c, alpha, domain, subset, crop, True)
                            row.update(
                                n_total=m,
                                n_answered=int(round(np.mean([p["n_answered"] for p in per_draw]))),
                                coverage=float(cov.mean()),
                                coverage_min=float(cov.min()),
                                coverage_max=float(cov.max()),
                                coverage_ci95=[
                                    min(p["coverage_ci95"][0] for p in per_draw),
                                    max(p["coverage_ci95"][1] for p in per_draw),
                                ],
                                answer_rate=float(np.mean([p["answer_rate"] for p in per_draw])),
                                coverage_given_answered=_mean_or_none(
                                    [p["coverage_given_answered"] for p in per_draw]
                                ),
                                silent_failure_rate=float(
                                    np.mean([p["silent_failure_rate"] for p in per_draw])
                                ),
                                set_size=_median_sizes([p["set_size"] for p in per_draw]),
                                quantile=[p["quantile"] for p in per_draw],
                                n_cal=n,
                                n_draws=len(draws),
                                draws=per_draw,
                            )
                            rows.append(row)
                body = {
                    "calibration": {
                        "fit": cfg["fit_split"], "calibration": cfg["target_pools"][domain],
                        "n_cal": n, "draws": n_draws, "oracle_seed": int(cfg["oracle_seed"]),
                        "draw_seed": "SeedSequence([oracle_seed, DOMAIN_KEYS[domain], n, draw])",
                        "evaluation": subset, "n_eval": m,
                    },
                    "fit": fit_record,
                    "rows": rows,
                }  # fmt: skip
                write(domain, crop, arm, True, body, oprov)

    index_record = {
        "schema": RESULT_SCHEMA,
        "kind": "shift_index",
        "run": run,
        "expected": {
            "domains": cfg["domains"],
            "crop_sources": cfg["crop_sources"],
            "arms": [
                sh.SPLIT,
                sh.MONDRIAN,
                sh.WEIGHTED,
                *(sh.oracle_arm_tag(n) for n in oracle_ns),
            ],
            "scores": score_ids,
            "conventions": conventions,
            "alphas": alphas,
        },  # fmt: skip
        "checks": checks,
        "classifiers": classifiers,
        "cells": index,
        "provenance": {**common_prov, "created_at": utc_now_iso()},
    }
    expected = {
        (d, c, a)
        for d in cfg["domains"]
        for c in cfg["crop_sources"]
        for a in index_record["expected"]["arms"]
    }
    seen = {(cell["domain"], cell["crop_source"], cell["arm"]) for cell in index}
    unexplained = sorted(expected - seen)
    index_record["unexplained"] = [list(cell) for cell in unexplained]
    write_result_json(out_dir / "index.json", index_record)
    if unexplained:
        print(f"UNEXPLAINED CELLS: {unexplained}")
        failed = True
    written = sum(cell["file"] is not None for cell in index)
    print(f"{written} files written, {len(index) - written} cells explained, -> {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
