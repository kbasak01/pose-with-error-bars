"""Phase 5 Level C: C1 (learned keypoint ellipses) and C2 (pose balls scaled by linearised σ̂).

Reads the Phase 3 synthetic `predicted_crop` dump and its Phase 5 variance sidecar. Calibrates on
`val_cal`, evaluates on `val_test`, under both PnP-failure conventions and the full α grid, then
repeats over R random re-partitions of `val_cal` + `val_test` (Phase 2 protocol, same seed as
Level B, so C1 and B2 share their re-splits). Also reports the head's quality against the
heatmap-moment baseline (`val_tune` fits only the global NLL rescaling; `val_test` reported after),
and C1 vs B2 / C2 vs A1 set sizes at equal α. Synthetic only: no HIL file is opened. CPU only.

Writes, under `results/level_c/`:
* `<run>_<crop>_synthetic.json` (kind `level_c`) + `_resplits.npz` (per-draw arrays).

    python scripts/run_level_c.py --config configs/conformal.yaml
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
from scipy.stats import spearmanr

from poseconf.conformal.metrics import coverage_summary, outcomes
from poseconf.conformal.so3 import geodesic_distance
from poseconf.conformal.split import CONVENTIONS, conformal_quantile
from poseconf.data.dump import (
    dump_file,
    labels_file,
    load_dump,
    load_dump_labels,
    variance_dump_file,
)
from poseconf.data.splits import load_split, manifest_sha256
from poseconf.engine import level_a, level_b
from poseconf.engine.level_c import (
    LEVEL_C_SCORES,
    attach_variance,
    c2_scores,
    c2_set,
    check_variance_mapping,
    head_quality_report,
    pose_sigma,
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

#: Level C is in-distribution only in Phase 5 (HIL evaluation waits for Phase 6).
DOMAIN = "synthetic"
ARM = "split"

DEFINITIONS = {
    "C1": (
        "max_k sqrt(r_k^T S_k^-1 r_k) over in-frame constrained keypoints, S_k = the variance "
        "head's covariance mapped to the full frame; B2's code and empty-channel rule"
    ),
    "C2": (
        "max(E_R / sigma_R, ||t_hat - t|| / sigma_t); sigma = sqrt(lambda_max) of the rotation / "
        "translation block of H^-1, H = sum_k J_k^T S_k^-1 J_k at the PnP estimate over keypoints "
        "visible under it and constrained (C1's S_k). Solved frames with undefined sigma "
        "(|U| < 3 or singular H) are invalid: abstain or +inf by convention"
    ),
    "head_quality": (
        "2-D Gaussian NLL (nats per keypoint), Mahalanobis 1/2-sigma ellipse coverage (Gaussian "
        "reference 0.393 / 0.865; uncalibrated reliability, not a conformal guarantee) and "
        "Spearman(det(S)^(1/4), ||r||) on predicted-box crops; rescaled = one global factor "
        "per source fitted on val_tune, applied unchanged to val_test"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/level_c"))
    return parser.parse_args(argv)


def _synthetic_split(name: str) -> list[str]:
    if not name.startswith(f"{DOMAIN}_"):
        raise ValueError(f"Level C in Phase 5 reads synthetic splits only, got {name!r}")
    return load_split(name)


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(POSECONF_ROOT))
    except ValueError:
        return path.name


def _rows_by_key(path: Path) -> dict[tuple[str, str, float], dict[str, Any]]:
    record = json.loads(path.read_text("utf-8"))
    return {(r["score"], r["convention"], float(r["alpha"])): r for r in record["rows"]}


def _median(size: dict[str, Any] | None, key: str) -> float | None:
    return None if not size or key not in size else size[key]["median"]


def _ranking(frames: Any, sigma: Any) -> dict[str, Any]:
    """Spearman of C2's σ̂ against the realised pose error on solved frames with σ̂ (uses labels;
    evaluation only)."""
    ok = frames.valid & sigma.ok
    e_r = geodesic_distance(frames.q_hat[ok], frames.q_gt[ok])
    e_t = np.linalg.norm(frames.t_hat[ok] - frames.t_gt[ok], axis=1)
    rng = np.linalg.norm(frames.t_hat[ok], axis=1)
    return {
        "n": int(ok.sum()),
        "spearman_sigma_R_vs_E_R": float(spearmanr(sigma.sigma_R[ok], e_r).statistic),
        "spearman_sigma_t_vs_e_t": float(spearmanr(sigma.sigma_t[ok], e_t).statistic),
        "spearman_sigma_t_frac_vs_e_t_frac": float(
            spearmanr(sigma.sigma_t[ok] / rng, e_t / rng).statistic
        ),
        "sigma_R_deg_median": float(np.degrees(np.median(sigma.sigma_R[ok]))),
        "sigma_t_m_median": float(np.median(sigma.sigma_t[ok])),
    }


def main(argv: list[str] | None = None) -> int:
    """Run Level C; return the process exit code (1 if any gate check fails)."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    level = config["level_c"]
    alphas = [float(a) for a in config["alpha_grid"]]
    conventions = list(config["conventions"])
    scores = list(level["scores"])
    run, crop = level["run"], level["crop_source"]
    if not set(scores) <= set(LEVEL_C_SCORES) or not set(conventions) <= set(CONVENTIONS):
        raise ValueError(f"unsupported scores {scores} or conventions {conventions}")
    if level["valid_mask"] != "pnp_success":
        raise ValueError(f"unsupported valid mask {level['valid_mask']!r}")

    paths = p1_paths(args.paths)
    geometry = projection_geometry(run, paths=paths)
    order = pose_quaternion_order(run)
    dump_path = dump_file(paths.dumps_root, run, DOMAIN, crop, subset=False)
    side_path = variance_dump_file(paths.dumps_root, run, DOMAIN, crop)
    dump, dump_meta = load_dump(dump_path)
    sidecar, side_meta = load_dump(side_path)
    if dump_meta["domain"] != DOMAIN or dump_meta["oracle"]:
        raise ValueError(f"expected a non-oracle synthetic dump, got {dump_meta['tag']!r}")
    if side_meta["base_dump_sha256"] != sha256_file(dump_path):
        raise ValueError("variance sidecar was made from a different dump")
    dump = attach_variance(dump, sidecar)
    labels, _ = load_dump_labels(paths.dumps_root, run, DOMAIN, "all", tag=ARM)

    cov_rel = check_variance_mapping(dump)
    if cov_rel > float(level["cov_mapping_rtol"]):
        raise ValueError(f"vhead_cov_full not reproduced by crop_cov_to_full: {cov_rel}")
    cal, check_cal = level_b.join_dump(
        dump,
        labels,
        _synthetic_split(level["calibration_split"]),
        geometry,
        order,
        cov_key="vhead_cov_full",
    )
    test, check_test = level_b.join_dump(
        dump,
        labels,
        _synthetic_split(level["test_split"]),
        geometry,
        order,
        cov_key="vhead_cov_full",
    )
    tune, _ = level_b.join_dump(
        dump,
        labels,
        _synthetic_split(level["fit_split"]),
        geometry,
        order,
        cov_key="vhead_cov_full",
    )
    worst_px = max(check_cal["max_abs_diff_px"], check_test["max_abs_diff_px"])
    if worst_px > float(level["label_projection_max_px"]):
        raise ValueError(f"projected label pose departs from P1's label product by {worst_px} px")

    quality = head_quality_report(
        dump,
        labels,
        fit_names=_synthetic_split(level["fit_split"]),
        test_names=_synthetic_split(level["test_split"]),
    )

    pool = cal.concat(test)
    n_cal = len(cal)
    rs = config["repeated_splits"]
    rng = np.random.default_rng(int(rs["seed"]))
    perms = np.stack([rng.permutation(len(pool)) for _ in range(int(rs["n_resplits"]))])
    gate = level["gate"]

    # C2's per-frame σ̂ (label-free): pool for re-splits, val_tune for the ranking diagnostic.
    sigma_pool = pose_sigma(level_b.linearised_frames("C1", pool, geometry, order)[0])
    sigma_tune = pose_sigma(level_b.linearised_frames("C1", tune, geometry, order)[0])
    sigma_test = pose_sigma(level_b.linearised_frames("C1", test, geometry, order)[0])

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"alphas": np.asarray(alphas)}
    for score_id in scores:
        if score_id == "C1":
            size_scale = level_b.unit_radius_px("C1", pool)
            valid_pool = pool.valid
        else:
            size_scale = np.where(pool.valid & sigma_pool.ok, sigma_pool.sigma_R, np.nan)
            valid_pool = pool.valid & sigma_pool.ok
        for convention in conventions:
            if score_id == "C1":
                s_pool = level_b.score_frames("C1", pool, convention)
            else:
                s_pool = c2_scores(pool, sigma_pool, convention)
            s_cal, s_test = s_pool[:n_cal], s_pool[n_cal:]
            valid_test = valid_pool[n_cal:]
            draws = level_a.resplit_draws(
                s_pool,
                valid_pool,
                size_scale,
                perms,
                n_cal,
                alphas,
                convention,
                size_factor=1.0 if score_id == "C1" else 180.0 / math.pi,
            )
            n_fail = int((~valid_pool).sum())
            finite = s_pool[valid_pool]
            atoms = {
                "n_top_atom": n_fail if convention == "answer_required" else 0,
                "n_bottom_atom": n_fail if convention == "abstain_allowed" else 0,
            }
            size_name = "kp_px" if score_id == "C1" else "rot_deg"
            prefix = f"{score_id}_{convention}"
            arrays[f"{prefix}_n_covered"] = draws.n_covered
            arrays[f"{prefix}_q"] = draws.q
            arrays[f"{prefix}_answer_rate"] = draws.answer_rate
            arrays[f"{prefix}_silent_failure_rate"] = draws.silent_failure_rate
            arrays[f"{prefix}_median_{size_name}"] = draws.median_rot_deg
            for i, alpha in enumerate(alphas):
                q = conformal_quantile(s_cal, alpha)
                covered, answered = outcomes(s_test, valid_test, q, convention)
                summary = coverage_summary(covered, answered)
                row: dict[str, Any] = {
                    "score": score_id,
                    "arm": ARM,
                    "convention": convention,
                    "alpha": alpha,
                    "domain": DOMAIN,
                    "subset": level["test_split"],
                    "crop_source": crop,
                    "oracle": False,
                    "valid_mask": (
                        "pnp_success" if score_id == "C1" else "pnp_success_and_sigma_defined"
                    ),
                    **summary,
                    "quantile": q,
                    "n_cal": n_cal,
                }
                if score_id == "C1":
                    kset = level_b.keypoint_set("C1", test, q)
                    bounded = test.include & ~test.unconstrained
                    row["set_size"] = level_b.set_size_report(kset, answered)
                    row["n_vacuous"] = int((test.valid & ~bounded.any(axis=1)).sum())
                    row["purse_agreement"] = level_b.purse_agreement(
                        kset, test, s_test, order, geometry
                    )
                else:
                    row["set_size"] = level_a.set_size_report(c2_set(test, sigma_test, q), answered)
                    row["n_solved_sigma_undefined"] = int((test.valid & ~sigma_test.ok).sum())
                row["resplits"] = level_a.summarise_draws(
                    draws,
                    i,
                    **atoms,
                    n_finite_ties=int(finite.size - np.unique(finite).size),
                    ks_min_p=float(gate["ks_min_p"]),
                    mean_band_se=float(gate["mean_band_se"]),
                    fixed_n_covered=summary["n_covered"],
                    size_name=size_name,
                )
                rows.append(row)

    # Equal-α comparisons with the committed Level A / Level B results.
    level_a_path = POSECONF_ROOT / level["level_a_result"]
    level_b_path = POSECONF_ROOT / level["level_b_result"]
    a_rows, b_rows = _rows_by_key(level_a_path), _rows_by_key(level_b_path)
    comparison = []
    for row in rows:
        key = (row["convention"], row["alpha"])
        if row["score"] == "C1":
            ref = b_rows[("B2", *key)]
            comparison.append(
                {
                    "pair": "C1_vs_B2",
                    "convention": key[0],
                    "alpha": key[1],
                    "kp_px_median": [
                        _median(row["set_size"], "kp_px"),
                        _median(ref["set_size"], "kp_px"),
                    ],
                    "coverage": [row["coverage"], ref["coverage"]],
                }
            )
        else:
            ref = a_rows[("A1", *key)]
            comparison.append(
                {
                    "pair": "C2_vs_A1",
                    "convention": key[0],
                    "alpha": key[1],
                    "rot_deg_median": [
                        _median(row["set_size"], "rot_deg"),
                        _median(ref["set_size"], "rot_deg"),
                    ],
                    "trans_frac_median": [
                        _median(row["set_size"], "trans_frac"),
                        _median(ref["set_size"], "trans_frac"),
                    ],
                    "coverage": [row["coverage"], ref["coverage"]],
                    "answer_rate": [row["answer_rate"], ref["answer_rate"]],
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run}_{crop}_{DOMAIN}"
    npz_path = args.out_dir / f"{stem}_resplits.npz"
    np.savez_compressed(npz_path, **arrays)
    verdicts = [r["resplits"]["verdict"] for r in rows]
    purse = [r["purse_agreement"] for r in rows if "purse_agreement" in r]
    purse_ok = all(p["n_agree"] == p["n_checked"] for p in purse)
    head = side_meta["variance_head"]
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "level_c",
        "run": run,
        "domain": DOMAIN,
        "crop_source": crop,
        "arm": ARM,
        "oracle": False,
        "splits": {
            "fit": level["fit_split"],
            "calibration": level["calibration_split"],
            "test": level["test_split"],
        },
        "definitions": DEFINITIONS,
        "variance_head": head,
        "checks": {
            "vhead_cov_mapping_max_rel_diff": cov_rel,
            "label_projection_max_abs_diff_px": worst_px,
            "kp_pred_crop_bit_equal_to_dump": True,
            "n_unconstrained_keypoints_test": check_test["n_unconstrained_keypoints"],
            "purse_agreement_all_rows": purse_ok,
        },
        "head_quality": quality,
        "c2_sigma_ranking": {
            "val_tune": _ranking(tune, sigma_tune),
            "val_test": _ranking(test, sigma_test),
        },
        "resplit_protocol": {
            "pool": [level["calibration_split"], level["test_split"]],
            "R": int(rs["n_resplits"]),
            "seed": int(rs["seed"]),
            "common_random_numbers": (
                "the same R permutations serve every score, convention, alpha (and Level B)"
            ),
            "reference_law": "conformal.metrics.resplit_coverage_law (exact given the pool)",
            "gate": {k: float(v) for k, v in gate.items()},
            "arrays": npz_path.name,
        },
        "verdict_counts": {v: verdicts.count(v) for v in ("VALID", "DEGENERATE", "DEVIATES")},
        "comparison": comparison,
        "rows": rows,
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "p1_commit": p1_commit(),
            "p1_checkpoint_sha256": side_meta["provenance"]["p1_checkpoint_sha256"],
            "variance_head_checkpoint_sha256": head["checkpoint_sha256"],
            "dump": _rel(dump_path),
            "dump_sha256": sha256_file(dump_path),
            "variance_sidecar": _rel(side_path),
            "variance_sidecar_sha256": sha256_file(side_path),
            "labels": _rel(labels_file(paths.dumps_root, run, DOMAIN, "all")),
            "labels_sha256": sha256_file(labels_file(paths.dumps_root, run, DOMAIN, "all")),
            "level_a_result_sha256": sha256_file(level_a_path),
            "level_b_result_sha256": sha256_file(level_b_path),
            "split_manifest_sha256": manifest_sha256(),
            "config_sha256": sha256_file(args.config),
            "created_at": utc_now_iso(),
            "pnp_config": load_p1_config(run)["pnp"],
            "resplits_npz_sha256": sha256_file(npz_path),
        },
    }
    out = write_result_json(args.out_dir / f"{stem}.json", result)
    print(f"verdicts {result['verdict_counts']}; PURSE agreement all rows: {purse_ok} -> {out}")
    learned = quality["val_test"]["learned"]
    moment = quality["val_test"]["heatmap_moment"]
    print(
        f"val_test NLL learned {learned['nll_mean']:.4f} (rescaled {learned['nll_mean_rescaled']:.4f})"
        f" vs heatmap moment {moment['nll_mean']:.4f} (rescaled {moment['nll_mean_rescaled']:.4f})"
    )
    for row in rows:
        if row["alpha"] == float(config["headline_alpha"]):
            print(
                f"  {row['score']} {row['convention']}: coverage {row['coverage']:.4f} "
                f"{row['coverage_ci95']} answer {row['answer_rate']:.4f} q {row['quantile']:.4g} "
                f"[{row['resplits']['verdict']}]"
            )
    return 0 if ("DEVIATES" not in verdicts and purse_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
