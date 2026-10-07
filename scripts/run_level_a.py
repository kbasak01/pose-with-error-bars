"""Phase 2 Level A: A1-A3 split-conformal pose sets from P1's committed synthetic sidecar.

Fits normalisers and g(conf) on `val_tune`, calibrates on `val_cal`, evaluates on `val_test` only,
under both PnP-failure conventions and the full alpha grid, then repeats calibrate/evaluate over R
random re-partitions of `val_cal` + `val_test`. Synthetic only: no HIL file is opened.

Writes `results/level_a/<run>_<crop>_synthetic.json` (kind `level_a`) and the per-draw arrays as
`results/level_a/<run>_<crop>_synthetic_resplits.npz`. CPU only.

    python scripts/run_level_a.py --config configs/conformal.yaml --run keypoint_a2 \\
        --crop-source predicted_crop
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
from poseconf.data.splits import load_split, manifest_sha256
from poseconf.engine.level_a import (
    LEVEL_A_SCORES,
    Frames,
    check_sidecar_errors,
    fit_level_a,
    join_frames,
    pose_set,
    resplit_draws,
    score_frames,
    set_size_report,
    summarise_draws,
)
from poseconf.p1_adapter import (
    load_p1_config,
    load_sidecar,
    load_synthetic_labels,
    p1_commit,
    p1_paths,
)
from poseconf.provenance import (
    POSECONF_ROOT,
    RESULT_SCHEMA,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    write_result_json,
)

#: Level A is in-distribution only in Phase 2 (HIL evaluation waits for Phase 6).
DOMAIN = "synthetic"
ARM = "split"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--run", default="keypoint_a2")
    parser.add_argument("--crop-source", default="predicted_crop")
    parser.add_argument("--out-dir", type=Path, default=Path("results/level_a"))
    return parser.parse_args(argv)


def _synthetic_split(name: str) -> list[str]:
    if not name.startswith(f"{DOMAIN}_"):
        raise ValueError(f"Level A in Phase 2 reads synthetic splits only, got {name!r}")
    return load_split(name)


def _checkpoint_sha(run: str) -> str:
    record = json.loads((POSECONF_ROOT / "results" / "p1_checkpoints.json").read_text("utf-8"))
    (entry,) = [c for c in record["checkpoints"] if c["run"] == run]
    if not entry["match"]:
        raise ValueError(f"{run} checkpoint does not match the P1 release manifest")
    return entry["sha256_local"]


def _rot_scale(score_id: str, frames: Frames, fit: Any) -> np.ndarray:
    pset = pose_set(score_id, frames, fit, 1.0)
    return np.where(frames.valid, pset.rot_scale, np.nan)


def _concat(a: Frames, b: Frames) -> Frames:
    return Frames(
        **{
            name: np.concatenate([getattr(a, name), getattr(b, name)])
            for name in a.__dataclass_fields__
        }
    )


def main(argv: list[str] | None = None) -> int:
    """Run Level A; return the process exit code (1 if any cell DEVIATES)."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    level = config["level_a"]
    alphas = [float(a) for a in config["alpha_grid"]]
    conventions = list(config["conventions"])
    scores = list(level["scores"])
    if not set(scores) <= set(LEVEL_A_SCORES) or not set(conventions) <= set(CONVENTIONS):
        raise ValueError(f"unsupported scores {scores} or conventions {conventions}")

    sidecar = load_sidecar(args.run, DOMAIN, args.crop_source)
    labels = load_synthetic_labels("validation", paths=p1_paths(args.paths))
    frames = {
        role: join_frames(
            sidecar.fields, labels.filenames, labels.q, labels.t, _synthetic_split(level[key])
        )
        for role, key in (
            ("tune", "fit_split"),
            ("cal", "calibration_split"),
            ("test", "test_split"),
        )
    }
    tol = level["sidecar_check_tolerance"]
    join_check = {
        role: check_sidecar_errors(f, tol_rad=tol["e_r_rad"], tol_m=tol["e_t_m"])
        for role, f in frames.items()
    }
    fit = fit_level_a(frames["tune"])

    cal, test = frames["cal"], frames["test"]
    pool = _concat(cal, test)
    n_cal, n_pool = len(cal), len(pool)
    rs = config["repeated_splits"]
    rng = np.random.default_rng(int(rs["seed"]))
    perms = np.stack([rng.permutation(n_pool) for _ in range(int(rs["n_resplits"]))])
    gate = level["gate"]

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"alphas": np.asarray(alphas)}
    for score_id in scores:
        rot_scale = _rot_scale(score_id, pool, fit)
        for convention in conventions:
            s_pool = score_frames(score_id, pool, fit, convention)
            s_cal, s_test = s_pool[:n_cal], s_pool[n_cal:]
            draws = resplit_draws(s_pool, pool.valid, rot_scale, perms, n_cal, alphas, convention)
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
            arrays[f"{prefix}_median_rot_deg"] = draws.median_rot_deg
            for i, alpha in enumerate(alphas):
                q = conformal_quantile(s_cal, alpha)
                covered, answered = outcomes(s_test, test.valid, q, convention)
                summary = coverage_summary(covered, answered)
                rows.append(
                    {
                        "score": score_id,
                        "arm": ARM,
                        "convention": convention,
                        "alpha": alpha,
                        "domain": DOMAIN,
                        "subset": level["test_split"],
                        "crop_source": args.crop_source,
                        "oracle": False,
                        **summary,
                        "set_size": set_size_report(pose_set(score_id, test, fit, q), answered),
                        "quantile": q,
                        "n_cal": n_cal,
                        "resplits": summarise_draws(
                            draws,
                            i,
                            **atoms,
                            n_finite_ties=int(finite.size - np.unique(finite).size),
                            ks_min_p=float(gate["ks_min_p"]),
                            mean_band_se=float(gate["mean_band_se"]),
                        ),
                    }
                )

    stem = f"{args.run}_{args.crop_source}_{DOMAIN}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"{stem}_resplits.npz"
    np.savez_compressed(npz_path, **arrays)
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "level_a",
        "run": args.run,
        "domain": DOMAIN,
        "crop_source": args.crop_source,
        "arm": ARM,
        "oracle": False,
        "splits": {
            "fit": level["fit_split"],
            "calibration": level["calibration_split"],
            "test": level["test_split"],
            "n": {role: len(f) for role, f in frames.items()},
            "n_solved": {role: int(f.valid.sum()) for role, f in frames.items()},
        },
        "fit": fit.as_dict(),
        "join_check": join_check,
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
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "p1_commit": p1_commit(),
            "p1_checkpoint_sha256": _checkpoint_sha(args.run),
            "p1_sidecar": str(sidecar.path.relative_to(POSECONF_ROOT)),
            "p1_sidecar_sha256": sha256_file(sidecar.path),
            "p1_sidecar_meta": sidecar.meta,
            "split_manifest_sha256": manifest_sha256(),
            "config_sha256": sha256_file(args.config),
            "resplits_npz_sha256": sha256_file(npz_path),
            "created_at": utc_now_iso(),
            "pnp_config": load_p1_config(args.run)["pnp"],
        },
    }
    out = write_result_json(args.out_dir / f"{stem}.json", result)

    print(
        f"{'score':<5} {'convention':<16} {'alpha':>5} {'cov':>7} {'q':>8} {'mean':>8} "
        f"{'law':>8} {'ks_law':>7} {'ks_beta':>8}  verdict"
    )
    for row in rows:
        r = row["resplits"]
        q = row["quantile"]
        print(
            f"{row['score']:<5} {row['convention']:<16} {row['alpha']:>5.2f} "
            f"{row['coverage']:>7.4f} {q if math.isinf(q) else round(q, 3):>8} "
            f"{r['mean_coverage']:>8.5f} {r['law_mean']:>8.5f} {r['ks_law']['pvalue']:>7.3f} "
            f"{r['beta']['ks_pvalue']:>8.2g}  {r['verdict']}"
        )
    print(f"wrote {out} and {npz_path}")
    return 1 if any(row["resplits"]["verdict"] == "DEVIATES" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
