"""Phase 4 Level B: B1/B2 joint keypoint sets, PURSE equivalence, propagation to pose.

Calibrates B1 (discs, radius q x predicted box diagonal) and B2 (heatmap-moment ellipses) on
`val_cal`, evaluates on `val_test` only, under both PnP-failure conventions and the full alpha grid,
then repeats over R random re-partitions of `val_cal` + `val_test`. Checks on every row that PURSE
membership of the label pose equals joint keypoint coverage. Measures the pose-space extent of the
sets with the linearised (every alpha) and sampled (headline alpha) estimators, timed, and compares
them with Level A at equal alpha. Synthetic only: no HIL file is opened. CPU only.

Writes, under `results/level_b/`:
* `<run>_<crop>_synthetic.json` (kind `level_b`) + `_resplits.npz` (per-draw arrays);
* `<run>_<crop>_synthetic_propagation.json` (kind `level_b_propagation`).

    python scripts/run_level_b.py --config configs/conformal.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from poseconf.conformal.metrics import coverage_summary, outcomes
from poseconf.conformal.split import CONVENTIONS, conformal_quantile
from poseconf.data.dump import dump_file, labels_file, load_dump, load_dump_labels
from poseconf.data.splits import load_split, manifest_sha256
from poseconf.engine.level_a import resplit_draws, summarise_draws
from poseconf.engine.level_b import (
    LEVEL_B_SCORES,
    check_cov_mapping,
    estimate_in_purse,
    join_dump,
    keypoint_set,
    linearised_frames,
    linearised_radii,
    propagation_report,
    purse_agreement,
    runtime_summary,
    sampled_frames,
    score_frames,
    set_size_report,
    unit_radius_px,
    worker_count,
)
from poseconf.p1_adapter import (
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

#: Level B is in-distribution only in Phase 4 (HIL evaluation waits for Phase 6).
DOMAIN = "synthetic"
ARM = "split"
_RAD_TO_DEG = 180.0 / math.pi

DEFINITIONS = {
    "purse": (
        "poses whose keypoints visible under the pose (P1's rule) and constrained lie in their "
        "sets; the label pose is a member iff the joint keypoint score <= q. The PURSE is "
        "unbounded: a pose projecting no constrained keypoint into the frame is a member at any "
        "q >= 0, so its global extent is pi / infinite. Every radius below describes the PURSE "
        "near the PnP estimate"
    ),
    "linearised_inner": (
        "first order about the PnP estimate over keypoints U visible under it, residual-aware: "
        "sqrt(||o||^2 + (q^2 - c) lambda), o = block of delta* = -H^-1 g, c = residual floor, "
        "lambda = lambda_max of the (H^-1) block; reached or exceeded by a point of the inner "
        "ellipsoid (sum_k Mahalanobis^2 <= q^2) inside the linearised PURSE, so a loose lower "
        "estimate of the linearised extent. NaN when c > q^2; those frames (n_nan, the "
        "high-residual ones) are excluded from the radius summaries, which therefore describe a "
        "selected subset. Approximation (linearisation, fixed U), not a bound"
    ),
    "linearised_outer": (
        "||o|| + sqrt((|U| q^2 - c) lambda): bounds the extent of the ellipsoid sum_k "
        "Mahalanobis^2 <= |U| q^2, which contains the linearised PURSE. NaN when c > |U| q^2 "
        "(then the linearised PURSE is empty; it can also be empty with a finite outer). "
        "Approximation, not a bound"
    ),
    "sampled": (
        "M keypoint configurations uniform in the sets, cv2.solvePnP ITERATIVE warm-started at the "
        "estimate, PURSE members kept; max deviation from the estimate. Inner approximation of "
        "the PURSE's extent near the estimate: a lower bound on how far the members local PnP "
        "reaches lie from it. NaN when no sample is accepted (n_nan)"
    ),
    "measured_ball_coverage": (
        "fraction of answered val_test frames with E_R <= rotation radius and ||t - t_hat|| <= "
        "translation radius, Clopper-Pearson 95 %; measured, no guarantee"
    ),
    "estimate_in_purse": (
        "fraction of answered frames whose PnP estimate lies in its own PURSE; where it does not, "
        "the PnP residual exceeds the set on some keypoint"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/level_b"))
    return parser.parse_args(argv)


def _synthetic_split(name: str) -> list[str]:
    if not name.startswith(f"{DOMAIN}_"):
        raise ValueError(f"Level B in Phase 4 reads synthetic splits only, got {name!r}")
    return load_split(name)


def _checkpoint_sha(run: str) -> str:
    record = json.loads((POSECONF_ROOT / "results" / "p1_checkpoints.json").read_text("utf-8"))
    (entry,) = [c for c in record["checkpoints"] if c["run"] == run]
    if not entry["match"]:
        raise ValueError(f"{run} checkpoint does not match the P1 release manifest")
    return entry["sha256_local"]


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(POSECONF_ROOT))
    except ValueError:
        return path.name


def _level_a_sizes(path: Path) -> dict[tuple[str, str, float], dict[str, Any]]:
    record = json.loads(path.read_text("utf-8"))
    return {
        (row["score"], row["convention"], float(row["alpha"])): row["set_size"]
        for row in record["rows"]
    }


def _median(report: dict[str, Any] | None, key: str) -> float | None:
    if not report or key not in report:
        return None
    return report[key]["median"]


def main(argv: list[str] | None = None) -> int:
    """Run Level B; return the process exit code (1 if any gate check fails)."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    level = config["level_b"]
    prop = config["propagation"]
    alphas = [float(a) for a in config["alpha_grid"]]
    conventions = list(config["conventions"])
    scores = list(level["scores"])
    run, crop = level["run"], level["crop_source"]
    if not set(scores) <= set(LEVEL_B_SCORES) or not set(conventions) <= set(CONVENTIONS):
        raise ValueError(f"unsupported scores {scores} or conventions {conventions}")
    if level["valid_mask"] != "pnp_success":
        raise ValueError(f"unsupported valid mask {level['valid_mask']!r}")

    paths = p1_paths(args.paths)
    geometry = projection_geometry(run, paths=paths)
    order = pose_quaternion_order(run)
    dump_path = dump_file(paths.dumps_root, run, DOMAIN, crop, subset=False)
    dump, dump_meta = load_dump(dump_path)
    if dump_meta["domain"] != DOMAIN or dump_meta["oracle"]:
        raise ValueError(f"expected a non-oracle synthetic dump, got {dump_meta['tag']!r}")
    labels, _ = load_dump_labels(paths.dumps_root, run, DOMAIN, "all", tag=ARM)

    cov_rel = check_cov_mapping(dump)
    if cov_rel > float(level["cov_mapping_rtol"]):
        raise ValueError(f"heatmap_cov_full not reproduced by crop_cov_to_full: {cov_rel}")
    cal, check_cal = join_dump(
        dump, labels, _synthetic_split(level["calibration_split"]), geometry, order
    )
    test, check_test = join_dump(
        dump, labels, _synthetic_split(level["test_split"]), geometry, order
    )
    worst_px = max(check_cal["max_abs_diff_px"], check_test["max_abs_diff_px"])
    if worst_px > float(level["label_projection_max_px"]):
        raise ValueError(f"projected label pose departs from P1's label product by {worst_px} px")

    pool = cal.concat(test)
    n_cal = len(cal)
    rs = config["repeated_splits"]
    rng = np.random.default_rng(int(rs["seed"]))
    perms = np.stack([rng.permutation(len(pool)) for _ in range(int(rs["n_resplits"]))])
    gate = level["gate"]

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"alphas": np.asarray(alphas)}
    quantiles: dict[tuple[str, str, float], float] = {}
    test_scores: dict[tuple[str, str], np.ndarray] = {}
    for score_id in scores:
        size_scale = unit_radius_px(score_id, pool)
        for convention in conventions:
            s_pool = score_frames(score_id, pool, convention)
            s_cal, s_test = s_pool[:n_cal], s_pool[n_cal:]
            test_scores[(score_id, convention)] = s_test
            draws = resplit_draws(
                s_pool, pool.valid, size_scale, perms, n_cal, alphas, convention, size_factor=1.0
            )
            n_fail = int((~pool.valid).sum())
            finite = s_pool[pool.valid]
            atoms = {
                "n_top_atom": n_fail if convention == "answer_required" else 0,
                "n_bottom_atom": n_fail if convention == "abstain_allowed" else 0,
            }
            prefix = f"{score_id}_{convention}"
            arrays[f"{prefix}_n_covered"] = draws.n_covered
            arrays[f"{prefix}_q"] = draws.q
            arrays[f"{prefix}_answer_rate"] = draws.answer_rate
            arrays[f"{prefix}_silent_failure_rate"] = draws.silent_failure_rate
            arrays[f"{prefix}_median_kp_px"] = draws.median_rot_deg
            for i, alpha in enumerate(alphas):
                q = conformal_quantile(s_cal, alpha)
                quantiles[(score_id, convention, alpha)] = q
                covered, answered = outcomes(s_test, test.valid, q, convention)
                summary = coverage_summary(covered, answered)
                kset = keypoint_set(score_id, test, q)
                bounded = test.include & ~test.unconstrained
                rows.append(
                    {
                        "score": score_id,
                        "arm": ARM,
                        "convention": convention,
                        "alpha": alpha,
                        "domain": DOMAIN,
                        "subset": level["test_split"],
                        "crop_source": crop,
                        "oracle": False,
                        "valid_mask": level["valid_mask"],
                        **summary,
                        "set_size": set_size_report(kset, answered),
                        "quantile": q,
                        "n_cal": n_cal,
                        "n_vacuous": int((test.valid & ~bounded.any(axis=1)).sum()),
                        "purse_agreement": purse_agreement(kset, test, s_test, order, geometry),
                        "resplits": summarise_draws(
                            draws,
                            i,
                            **atoms,
                            n_finite_ties=int(finite.size - np.unique(finite).size),
                            ks_min_p=float(gate["ks_min_p"]),
                            mean_band_se=float(gate["mean_band_se"]),
                            fixed_n_covered=summary["n_covered"],
                            size_name="kp_px",
                        ),
                    }
                )

    # ---------------------------------------------------------------- propagation (val_test)
    level_a_path = POSECONF_ROOT / level["level_a_result"]
    level_a = _level_a_sizes(level_a_path)
    linearised: dict[str, Any] = {}
    sampled: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    sp = prop["sampled"]
    sample_alpha = float(sp["alpha"])
    workers = worker_count(sp["workers"])
    by_score: dict[str, dict[str, Any]] = {}
    for si, score_id in enumerate(scores):
        extents, seconds = linearised_frames(score_id, test, geometry, order)
        lin_rows = []
        by_score[score_id] = {}
        for ci, convention in enumerate(conventions):
            for alpha in alphas:
                q = quantiles[(score_id, convention, alpha)]
                answered = np.flatnonzero(test.valid & (q > -math.inf))
                radii = linearised_radii(extents, answered, q)
                inside = estimate_in_purse(keypoint_set(score_id, test, q), test, geometry, order)
                resid = np.array([extents[int(r)].residual_max for r in answered])
                entry = {
                    "convention": convention,
                    "alpha": alpha,
                    "quantile": q,
                    "n_answered": int(answered.size),
                    "estimate_in_purse": float(inside[answered].mean()) if answered.size else None,
                    "residual_over_q": (
                        dict(
                            zip(
                                ("p50", "p90", "p99"),
                                (np.percentile(resid, [50, 90, 99]) / q).tolist(),
                                strict=True,
                            )
                        )
                        if answered.size and 0 < q < math.inf
                        else None
                    ),
                    "inner": propagation_report(
                        test, answered, radii["rot_inner"], radii["trans_inner"]
                    ),
                    "outer": propagation_report(
                        test, answered, radii["rot_outer"], radii["trans_outer"]
                    ),
                }
                lin_rows.append(entry)
                by_score[score_id][(convention, alpha)] = {"linearised": entry}
            if not (sp["enabled"] and sample_alpha in alphas):
                continue
            q = quantiles[(score_id, convention, sample_alpha)]
            answered = np.flatnonzero(test.valid & (q > -math.inf))
            if sp["max_frames"] is not None and answered.size > int(sp["max_frames"]):
                pick = np.random.default_rng(int(sp["seed"])).choice(
                    answered.size, size=int(sp["max_frames"]), replace=False
                )
                answered = np.sort(answered[pick])
            if not math.isfinite(q):
                sampled[f"{score_id}_{convention}"] = {
                    "alpha": sample_alpha,
                    "quantile": q,
                    "skipped": "q is infinite: the PURSE is the whole pose space",
                }
                continue
            results, wall = sampled_frames(
                keypoint_set(score_id, test, q),
                test,
                answered,
                geometry,
                order,
                samples=int(sp["samples"]),
                seed_key=[int(sp["seed"]), si, ci],
                workers=workers,
            )
            ext = [results[int(r)] for r in answered]
            n_solved = np.array([e.n_solved for e in ext])
            n_acc = np.array([e.n_accepted for e in ext])
            entry = {
                "convention": convention,
                "alpha": sample_alpha,
                "quantile": q,
                "samples": int(sp["samples"]),
                "frames": "all answered val_test frames"
                if sp["max_frames"] is None
                else f"seeded subset of {int(sp['max_frames'])}",
                "radius": propagation_report(
                    test,
                    answered,
                    np.array([e.rot for e in ext]),
                    np.array([e.trans for e in ext]),
                ),
                "acceptance": {
                    "median_accepted_fraction": float(np.median(n_acc / int(sp["samples"]))),
                    "min_accepted": int(n_acc.min()),
                    "n_frames_none_accepted": int((n_acc == 0).sum()),
                    "median_solved_fraction": float(np.median(n_solved / int(sp["samples"]))),
                },
                "runtime_per_frame": runtime_summary(np.array([e.seconds for e in ext])),
                "wall_s": wall,
                "workers": workers,
            }
            sampled[f"{score_id}_{convention}"] = entry
            by_score[score_id][(convention, sample_alpha)]["sampled"] = entry
        linearised[score_id] = {"runtime_per_frame": runtime_summary(seconds), "rows": lin_rows}

    for convention in conventions:
        for alpha in alphas:
            row: dict[str, Any] = {"convention": convention, "alpha": alpha, "level_a": {}}
            for a_id in ("A1", "A2", "A3"):
                size = level_a.get((a_id, convention, alpha))
                row["level_a"][a_id] = (
                    None
                    if size is None
                    else {
                        "rot_deg_median": _median(size, "rot_deg"),
                        "trans_frac_median": _median(size, "trans_frac"),
                        "boresight_frac_median": _median(size, "boresight_frac"),
                        "lateral_frac_median": _median(size, "lateral_frac"),
                    }
                )
            row["level_b"] = {}
            for score_id in scores:
                cell = by_score[score_id][(convention, alpha)]
                lin = cell["linearised"]
                out = {
                    f"linearised_{part}": {
                        "rot_deg_median": _median(lin[part], "rot_deg"),
                        "trans_frac_median": _median(lin[part], "trans_frac"),
                    }
                    for part in ("inner", "outer")
                }
                if "sampled" in cell:
                    out["sampled"] = {
                        "rot_deg_median": _median(cell["sampled"]["radius"], "rot_deg"),
                        "trans_frac_median": _median(cell["sampled"]["radius"], "trans_frac"),
                    }
                row["level_b"][score_id] = out
            comparison.append(row)

    # ---------------------------------------------------------------- write
    stem = f"{run}_{crop}_{DOMAIN}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"{stem}_resplits.npz"
    np.savez_compressed(npz_path, **arrays)
    labels_path = labels_file(paths.dumps_root, run, DOMAIN, "all")
    provenance = {
        "poseconf_git_sha": poseconf_git_sha(),
        "p1_commit": p1_commit(),
        "p1_checkpoint_sha256": _checkpoint_sha(run),
        "dump": _rel(dump_path),
        "dump_sha256": sha256_file(dump_path),
        "dump_poseconf_git_sha": dump_meta["provenance"]["poseconf_git_sha"],
        "labels": _rel(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "split_manifest_sha256": manifest_sha256(),
        "config_sha256": sha256_file(args.config),
        "created_at": utc_now_iso(),
        "pnp_config": load_p1_config(run)["pnp"],
    }
    n_agree = sum(r["purse_agreement"]["n_agree"] for r in rows)
    n_checked = sum(r["purse_agreement"]["n_checked"] for r in rows)
    checks = {
        "cov_mapping_max_rel_diff": cov_rel,
        "label_projection": {"val_cal": check_cal, "val_test": check_test},
        "purse_agreement": {
            "n_agree": n_agree,
            "n_checked": n_checked,
            "fraction": n_agree / n_checked,
            "scope": "val_test valid frames x score x convention x alpha",
        },
    }
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "level_b",
        "run": run,
        "domain": DOMAIN,
        "crop_source": crop,
        "arm": ARM,
        "oracle": False,
        "valid_mask": level["valid_mask"],
        "label_keypoints": "label pose projected with P1's project_points model; include = P1 visible",
        "splits": {
            "calibration": level["calibration_split"],
            "test": level["test_split"],
            "n": {"cal": len(cal), "test": len(test)},
            "n_solved": {"cal": int(cal.valid.sum()), "test": int(test.valid.sum())},
            "fit": "none: B1/B2 fit nothing, val_tune is not read",
        },
        "checks": checks,
        "resplit_protocol": {
            "pool": [level["calibration_split"], level["test_split"]],
            "R": int(rs["n_resplits"]),
            "seed": int(rs["seed"]),
            "common_random_numbers": "the same R permutations serve every score, convention, alpha",
            "reference_law": "conformal.metrics.resplit_coverage_law (exact given the pool)",
            "gate": gate,
            "arrays": npz_path.name,
        },
        "rows": rows,
        "provenance": {**provenance, "resplits_npz_sha256": sha256_file(npz_path)},
    }
    out = write_result_json(args.out_dir / f"{stem}.json", result)
    prop_result = {
        "schema": RESULT_SCHEMA,
        "kind": "level_b_propagation",
        "run": run,
        "domain": DOMAIN,
        "subset": level["test_split"],
        "crop_source": crop,
        "arm": ARM,
        "oracle": False,
        "definitions": DEFINITIONS,
        "units": {"rot_deg": "degrees", "trans_m": "metres", "trans_frac": "fraction of ||t_hat||"},
        "linearised": linearised,
        "sampled": sampled,
        "level_a_comparison": {
            "source": level["level_a_result"],
            "source_sha256": sha256_file(level_a_path),
            "note": (
                "medians over answered val_test frames at equal alpha and convention; linearised "
                "and sampled medians exclude their n_nan frames (see linearised / sampled)"
            ),
            "rows": comparison,
        },
        "provenance": {**provenance, "level_b_result": _rel(out)},
    }
    prop_out = write_result_json(args.out_dir / f"{stem}_propagation.json", prop_result)

    print(
        f"{'score':<5} {'convention':<16} {'alpha':>5} {'cov':>7} {'q':>8} {'mean':>8} "
        f"{'law':>8} {'ks_law':>7} {'purse':>11}  verdict"
    )
    for row in rows:
        r, q, pa = row["resplits"], row["quantile"], row["purse_agreement"]
        print(
            f"{row['score']:<5} {row['convention']:<16} {row['alpha']:>5.2f} "
            f"{row['coverage']:>7.4f} {q if math.isinf(q) else round(q, 4):>8} "
            f"{r['mean_coverage']:>8.5f} {r['law_mean']:>8.5f} {r['ks_law']['pvalue']:>7.3f} "
            f"{pa['n_agree']:>5}/{pa['n_checked']:<5}  {r['verdict']}"
        )
    for key, entry in sampled.items():
        if "radius" in entry:
            rad = entry["radius"]
            print(
                f"sampled {key}: rot median {rad['rot_deg']['median']:.3f} deg, "
                f"accepted {entry['acceptance']['median_accepted_fraction']:.3f}, "
                f"p50 {entry['runtime_per_frame']['p50_ms']:.1f} ms/frame"
            )
    print(f"wrote {out}, {prop_out} and {npz_path}")
    failed = any(row["resplits"]["verdict"] == "DEVIATES" for row in rows) or (
        n_agree / n_checked < float(gate["purse_agreement"])
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
