"""Variance-head seed spread (Phase 8, VALIDATION_CHECKLIST D6) from committed Level C results.

Reads one Level C result and one training summary per seed (all CPU, no dataset) and writes
`results/level_c/variance_head_seed_spread.json`: the per-seed values of the head-quality and
C1/C2 coverage numbers the project quotes, and their range across seeds.

The selected head is fixed in advance (`vhead_a2_s1337`, Phase 5). The second seed is a
robustness check only: nothing is selected between seeds, and every other result keeps using the
first. Two seeds give a range, not a variance estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
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

#: Where the variance-head configs live (for records that predate the `seed` field).
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
#: The α the project quotes C1/C2 at (headline α of the calibration artifacts).
HEADLINE_ALPHA = 0.10
#: Default inputs, selected seed first.
DEFAULT_LEVEL_C = (
    Path("results/level_c/keypoint_a2_predicted_crop_synthetic.json"),
    Path("results/level_c/seed_2026/keypoint_a2_predicted_crop_synthetic.json"),
)
DEFAULT_TRAINING = (
    Path("results/level_c/variance_head_training.json"),
    Path("results/level_c/seed_2026/variance_head_training.json"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level-c", type=Path, nargs="+", default=list(DEFAULT_LEVEL_C))
    parser.add_argument("--training", type=Path, nargs="+", default=list(DEFAULT_TRAINING))
    parser.add_argument(
        "--out", type=Path, default=Path("results/level_c/variance_head_seed_spread.json")
    )
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(level_c: dict[str, Any], score: str, convention: str) -> dict[str, Any]:
    rows = [
        r
        for r in level_c["rows"]
        if r["score"] == score
        and r["convention"] == convention
        and abs(float(r["alpha"]) - HEADLINE_ALPHA) < 1e-12
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one {score}/{convention}/α={HEADLINE_ALPHA} row, got {len(rows)}"
        )
    return rows[0]


def _comparison(level_c: dict[str, Any], pair: str, convention: str) -> dict[str, Any]:
    rows = [
        c
        for c in level_c["comparison"]
        if c["pair"] == pair
        and c["convention"] == convention
        and abs(float(c["alpha"]) - HEADLINE_ALPHA) < 1e-12
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one {pair}/{convention} comparison, got {len(rows)}")
    return rows[0]


def _seed(training: dict[str, Any]) -> int:
    """The run's seed: recorded since Phase 8; for older records, read from the config file whose
    SHA-256 the record carries (refusing if no committed config matches)."""
    if "seed" in training:
        return int(training["seed"])
    want = training["provenance"]["config_sha256"]
    for path in sorted(CONFIG_DIR.glob("variance_head*.yaml")):
        if sha256_file(path) == want:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if config["run_name"] != training["run_name"]:
                raise ValueError(f"{path} is for {config['run_name']}, not {training['run_name']}")
            return int(config["seed"])
    raise ValueError(f"{training['run_name']}: no seed recorded and no config matches {want}")


def seed_values(level_c: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    """The quoted numbers for one seed, each copied from its result file."""
    head = level_c["variance_head"]
    if head["run_name"] != training["run_name"]:
        raise ValueError(f"Level C used {head['run_name']}, training is {training['run_name']}")
    if head["checkpoint_sha256"] != training["best_checkpoint_sha256"]:
        raise ValueError(f"{head['run_name']}: Level C checkpoint is not the training best.pt")
    learned = level_c["head_quality"]["val_test"]["learned"]
    values: dict[str, Any] = {
        "run_name": training["run_name"],
        "seed": _seed(training),
        "checkpoint_sha256": head["checkpoint_sha256"],
        "best_epoch": training["best_epoch"],
        "val_tune_nll_best": training["best_val_tune_nll"],
        "val_test_nll_mean": learned["nll_mean"],
        "val_test_nll_mean_rescaled": learned["nll_mean_rescaled"],
        "val_test_coverage_1sigma": learned["coverage_1sigma"],
        "val_test_coverage_2sigma": learned["coverage_2sigma"],
        "val_test_spearman_sigma_vs_error": learned["spearman_sigma_vs_error"],
        "verdict_counts": level_c["verdict_counts"],
    }
    for convention in ("abstain_allowed", "answer_required"):
        for score in ("C1", "C2"):
            r = _row(level_c, score, convention)
            values[f"{score}_{convention}"] = {
                "coverage": r["coverage"],
                "coverage_ci95": r["coverage_ci95"],
                "n_total": r["n_total"],
                "n_answered": r["n_answered"],
                "answer_rate": r["answer_rate"],
                "silent_failure_rate": r["silent_failure_rate"],
                "quantile": r["quantile"],
            }
        c1 = _comparison(level_c, "C1_vs_B2", convention)
        c2 = _comparison(level_c, "C2_vs_A1", convention)
        values[f"C1_{convention}"]["kp_px_median"] = c1["kp_px_median"][0]
        values[f"C2_{convention}"]["rot_deg_median"] = c2["rot_deg_median"][0]
        values[f"C2_{convention}"]["trans_frac_median"] = c2["trans_frac_median"][0]
    return values


def spread(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """`max - min` across seeds for every numeric leaf; identifiers and counts are left out."""
    skip = {"run_name", "seed", "checkpoint_sha256", "best_epoch", "verdict_counts", "n_total"}

    def walk(items: list[Any]) -> Any:
        first = items[0]
        if isinstance(first, dict):
            out = {k: walk([i[k] for i in items]) for k in first if k not in skip}
            return {k: v for k, v in out.items() if v is not None}
        if isinstance(first, bool) or not isinstance(first, int | float):
            return None
        return max(items) - min(items)

    return walk(per_seed)


def main(argv: list[str] | None = None) -> int:
    """Write the seed-spread record; return the process exit code."""
    args = parse_args(argv)
    if len(args.level_c) != len(args.training) or len(args.level_c) < 2:
        raise SystemExit("need one --level-c and one --training per seed, at least two seeds")
    per_seed = [
        seed_values(_load(lc), _load(tr))
        for lc, tr in zip(args.level_c, args.training, strict=True)
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "variance_head",
        "stage": "seed_spread",
        "run": "keypoint_a2",
        "domain": "synthetic",
        "subset": "synthetic_val_test",
        "crop_source": "predicted_crop",
        "arm": "split",
        "oracle": False,
        "alpha": HEADLINE_ALPHA,
        "selected_run": per_seed[0]["run_name"],
        "definitions": {
            "selection": (
                "The first seed is the head selected in Phase 5 and used by every other result. "
                "No selection happens between seeds; the others are a robustness check."
            ),
            "spread": (
                "max - min across seeds for each quoted number. With two seeds this is a range, "
                "not a variance estimate."
            ),
            "coverage": (
                "Split-conformal coverage on synthetic val_test with each seed's own val_cal "
                "quantile; marginal, finite-sample, under exchangeability of val_cal and val_test."
            ),
        },
        "seeds": per_seed,
        "spread": spread(per_seed),
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "created_at": utc_now_iso(),
            "level_c_results": {str(p): sha256_file(p) for p in args.level_c},
            "training_results": {str(p): sha256_file(p) for p in args.training},
        },
    }
    out = write_result_json(args.out, result)
    for s in per_seed:
        print(
            f"{s['run_name']}: val_test NLL {s['val_test_nll_mean']:.4f}, "
            f"C1 aa {s['C1_abstain_allowed']['coverage']:.4f}, "
            f"C2 aa {s['C2_abstain_allowed']['coverage']:.4f}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
