"""Level C re-split verdicts under independent permutation seeds (Phase 8 sensitivity check).

The committed Level C results use one set of R = 1,000 permutations (`repeated_splits.seed`),
shared by every score, convention, α and variance-head seed (common random numbers). A cell that
fails the per-cell KS gate under that one set may be a property of the permutation set rather
than of the cell. This script re-runs `scripts/run_level_c.py` unchanged except for
`repeated_splits.seed` (and, for one run, `n_resplits`), into a temporary directory, and records
the verdict counts and the named cell under each. It writes beside the committed results and
never replaces them. Gate and protocol are unchanged.

    python scripts/resplit_seed_sensitivity.py \
        --head vhead_a2_s2026=dumps/seed_2026 --head vhead_a2_s1337=dumps \
        --committed results/level_c/seed_2026/keypoint_a2_predicted_crop_synthetic.json \
                    results/level_c/keypoint_a2_predicted_crop_synthetic.json \
        --out results/level_c/seed_2026/resplit_seed_sensitivity.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from poseconf.provenance import (
    RESULT_SCHEMA,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    write_result_json,
)

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import run_level_c  # noqa: E402

RESULT_NAME = "keypoint_a2_predicted_crop_synthetic.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument(
        "--head",
        action="append",
        required=True,
        help="RUN_NAME=DUMPS_ROOT; the first one also gets the large-R run",
    )
    parser.add_argument(
        "--committed",
        type=Path,
        nargs="+",
        required=True,
        help="the committed Level C results (one per --head, same order)",
    )
    parser.add_argument("--perm-seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--large-seed", type=int, default=9001)
    parser.add_argument("--large-r", type=int, default=20000)
    parser.add_argument(
        "--cell",
        nargs=3,
        default=["C1", "answer_required", "0.5"],
        metavar=("SCORE", "CONVENTION", "ALPHA"),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)  # fmt: skip


def summarise(level_c: dict[str, Any], cell: tuple[str, str, float]) -> dict[str, Any]:
    """Verdict counts, the DEVIATES cells, and the named cell's re-split statistics."""
    score, convention, alpha = cell
    rows = [
        r
        for r in level_c["rows"]
        if r["score"] == score and r["convention"] == convention and abs(r["alpha"] - alpha) < 1e-12
    ]
    if len(rows) != 1:
        raise ValueError(f"cell {cell} not found exactly once")
    rs = rows[0]["resplits"]
    return {
        "variance_head": level_c["variance_head"]["run_name"],
        "R": rs["R"],
        "verdict_counts": level_c["verdict_counts"],
        "deviates_cells": [
            [r["score"], r["convention"], r["alpha"]]
            for r in level_c["rows"]
            if r["resplits"]["verdict"] == "DEVIATES"
        ],
        "cell": {
            "verdict": rs["verdict"],
            "ks_law_pvalue": rs["ks_law"]["pvalue"],
            "mean_coverage": rs["mean_coverage"],
            "law_mean": rs["law_mean"],
            "z_mc_se": (rs["mean_coverage"] - rs["law_mean"]) / rs["mc_se"],
            "in_plan_band": rs["in_plan_band"],
            "sd_ratio_observed_to_law": rs["sd_ratio_observed_to_law"],
        },
    }


def run_once(
    config: dict[str, Any], paths: dict[str, Any], dumps_root: str, seed: int, r: int, tmp: Path
) -> dict[str, Any]:
    """`run_level_c.main` with only the permutation seed (and R) changed; returns its result."""
    tag = f"{Path(dumps_root).name}_{seed}_{r}"
    cfg = json.loads(json.dumps(config))
    cfg["repeated_splits"]["seed"], cfg["repeated_splits"]["n_resplits"] = seed, r
    cfg_path, paths_path, out = tmp / f"{tag}.yaml", tmp / f"{tag}_paths.yaml", tmp / tag
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    paths_path.write_text(yaml.safe_dump({**paths, "dumps_root": dumps_root}), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        run_level_c.main(
            ["--config", str(cfg_path), "--paths", str(paths_path), "--out-dir", str(out)]
        )  # exit code 1 only means a gate cell failed; the result is what is recorded
    return json.loads((out / RESULT_NAME).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Write the sensitivity record; return the process exit code."""
    args = parse_args(argv)
    heads = [h.split("=", 1) for h in args.head]
    if len(heads) != len(args.committed):
        raise SystemExit("one --committed result per --head")
    cell = (args.cell[0], args.cell[1], float(args.cell[2]))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = yaml.safe_load(args.paths.read_text(encoding="utf-8"))
    committed_seed = int(config["repeated_splits"]["seed"])
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, ((run_name, dumps_root), committed) in enumerate(
            zip(heads, args.committed, strict=True)
        ):
            record = json.loads(committed.read_text(encoding="utf-8"))
            if record["variance_head"]["run_name"] != run_name:
                raise SystemExit(f"{committed} is not {run_name}")
            runs.append(
                {"permutation_seed": committed_seed, "committed": str(committed),
                 **summarise(record, cell)}
            )  # fmt: skip
            plan = [(s, int(config["repeated_splits"]["n_resplits"])) for s in args.perm_seeds]
            if i == 0:
                plan.append((args.large_seed, args.large_r))
            for seed, r in plan:
                result = run_once(config, paths, dumps_root, seed, r, Path(tmp))
                if result["variance_head"]["run_name"] != run_name:
                    raise SystemExit(f"{dumps_root} holds {result['variance_head']['run_name']}")
                runs.append({"permutation_seed": seed, **summarise(result, cell)})
                print(f"{run_name} seed {seed} R {r}: {runs[-1]['verdict_counts']}")
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "validity",
        "stage": "resplit_seed_sensitivity",
        "run": "keypoint_a2",
        "domain": "synthetic",
        "crop_source": "predicted_crop",
        "arm": "split",
        "oracle": False,
        "cell": {"score": cell[0], "convention": cell[1], "alpha": cell[2]},
        "definitions": {
            "what": (
                "Level C re-split validity re-run with independent permutation seeds; the "
                "committed rows (permutation seed 1337) are quoted, not replaced. Same scores, "
                "pool, gate (law KS p >= 0.01 and mean within 3 MC s.e. of the plan band) and law."
            ),
            "multiplicity": (
                "The gate is per cell with no multiplicity correction. With 26 non-degenerate "
                "cells per run at 0.01, about 0.26 false rejections are expected per run; cells "
                "share one permutation set, so their verdicts are correlated."
            ),
            "z_mc_se": "(mean re-split coverage - exact law mean) / Monte Carlo standard error",
        },
        "runs": runs,
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "created_at": utc_now_iso(),
            "config_sha256": sha256_file(args.config),
            "committed_results": {str(p): sha256_file(p) for p in args.committed},
        },
    }
    write_result_json(args.out, result)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
