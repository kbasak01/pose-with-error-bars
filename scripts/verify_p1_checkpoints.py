"""Verify the four P1 checkpoints against the `phase-9-complete` release SHA256SUMS.txt.

Writes `results/p1_checkpoints.json` and exits non-zero if any hash differs. A missing manifest or
checkpoint is an error: nothing is written, because a verification that checked nothing is not a
result.

    python scripts/verify_p1_checkpoints.py --config configs/conformal.yaml \\
        --paths configs/paths.local.yaml --out results/p1_checkpoints.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from poseconf.p1_adapter import p1_commit, p1_paths
from poseconf.provenance import (
    RESULT_SCHEMA,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    verify_checkpoints,
    write_result_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument(
        "--sums", type=Path, default=None, help="override p1_release_sums from --paths"
    )
    parser.add_argument("--out", type=Path, default=Path("results/p1_checkpoints.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the verification; return the process exit code."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    runs = list(config["p1"]["checkpoints"])
    local = p1_paths(args.paths)
    sums_path = args.sums if args.sums is not None else local.release_sums

    records = verify_checkpoints(runs, local.p1_runs, sums_path)
    all_match = all(record["match"] for record in records)
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "checkpoints",
        "release_tag": config["p1"]["tag"],
        "n_checked": len(records),
        "all_match": all_match,
        "checkpoints": records,
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "p1_commit": p1_commit(),
            "p1_commit_expected": config["p1"]["commit"],
            "sums_sha256": sha256_file(sums_path),
            "config_sha256": sha256_file(args.config),
            "created_at": utc_now_iso(),
        },
    }
    write_result_json(args.out, result)
    for record in records:
        verdict = "match" if record["match"] else "MISMATCH"
        print(f"{record['run']:<22} {record['sha256_local'][:16]}…  {verdict}")
    print(f"wrote {args.out}: {sum(r['match'] for r in records)}/{len(records)} match")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
