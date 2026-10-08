"""Compare re-run result files with their committed versions, ignoring provenance.

Used at Phase 8 for committed files whose provenance is weaker than the rest: stamped with a
`-dirty` SHA, or with SHAs from different commits. The script re-produces the file elsewhere at a
clean tree, then compares it here. JSON is compared leaf by leaf, exactly, with the keys in
`IGNORED_KEYS` skipped. NPZ arrays are compared exactly. Writes a `rerun_check` result and exits
1 if any pair differs.

    python scripts/check_rerun.py \
        --pair results/level_a/x.json /tmp/rerun/x.json --pair ... \
        --out results/reproducibility/rerun_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from poseconf.provenance import (
    RESULT_SCHEMA,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    write_result_json,
)

#: Keys that record *when, where and from what commit* a file was made, not what it measured.
#: `figure` is an output path, which differs when the re-run writes its figure elsewhere.
IGNORED_KEYS = frozenset({"provenance", "created_at", "figure"})
#: How many differing leaves to list per pair before truncating.
MAX_LISTED = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        type=Path,
        required=True,
        metavar=("COMMITTED", "RERUN"),
    )
    parser.add_argument("--what", required=True, help="one line: what was re-run and how")
    parser.add_argument(
        "--rerun-root",
        type=Path,
        default=None,
        help="record re-run paths relative to this (temporary) directory, not absolute",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def json_differences(a: Any, b: Any, path: str = "") -> tuple[int, list[str]]:
    """Exact leaf-by-leaf comparison.

    Returns:
        `(leaves compared, paths that differ)`.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        n, diffs = 0, []
        for key in sorted(set(a) | set(b)):
            if key in IGNORED_KEYS:
                continue
            if key not in a or key not in b:
                diffs.append(f"{path}/{key} (missing on one side)")
                continue
            m, d = json_differences(a[key], b[key], f"{path}/{key}")
            n, diffs = n + m, diffs + d
        return n, diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 1, [f"{path} (length {len(a)} vs {len(b)})"]
        n, diffs = 0, []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            m, d = json_differences(x, y, f"{path}[{i}]")
            n, diffs = n + m, diffs + d
        return n, diffs
    return 1, ([] if a == b and type(a) is type(b) else [path or "/"])


def npz_differences(a: Path, b: Path) -> tuple[int, list[str]]:
    """Exact array comparison of two `.npz` files (NaN equal to NaN)."""
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        keys = sorted(set(x.files) | set(y.files))
        diffs = [k for k in keys if k not in x.files or k not in y.files]
        n = 0
        for k in keys:
            if k in diffs:
                continue
            n += int(x[k].size)
            same = x[k].shape == y[k].shape and x[k].dtype == y[k].dtype
            if same:
                same = bool(np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind in "fc"))
            if not same:
                diffs.append(k)
    return n, diffs


#: Provenance keys expected to differ between a committed file and its re-run.
EXPECTED_PROVENANCE_CHANGES = frozenset(
    {"poseconf_git_sha", "created_at", "source_poseconf_git_sha", "source_result"}
)


def compare(committed: Path, rerun: Path, rerun_root: Path | None = None) -> dict[str, Any]:
    """One pair's record: the numerical comparison, plus both sides' provenance.

    For JSON, `provenance_keys_differing` lists every provenance key (other than the commit and
    timestamp, which are expected to move) whose value differs: an input hash or config hash that
    changed is shown, not hidden by the comparison ignoring `provenance`. An `.npz` carries no git
    SHA of its own; pair it with the JSON its run wrote, whose SHA the dirty check covers.
    """
    old_prov: dict[str, Any] = {}
    new_prov: dict[str, Any] = {}
    if committed.suffix == ".npz":
        n, diffs = npz_differences(committed, rerun)
    else:
        old = json.loads(committed.read_text(encoding="utf-8"))
        new = json.loads(rerun.read_text(encoding="utf-8"))
        n, diffs = json_differences(old, new)
        old_prov, new_prov = old.get("provenance", {}), new.get("provenance", {})
    differing = sorted(
        k
        for k in set(old_prov) | set(new_prov)
        if k not in EXPECTED_PROVENANCE_CHANGES and old_prov.get(k) != new_prov.get(k)
    )
    return {
        "committed": str(committed),
        "committed_sha256": sha256_file(committed),
        "committed_poseconf_git_sha": old_prov.get("poseconf_git_sha"),
        "rerun": (
            f"<rerun-root>/{rerun.resolve().relative_to(rerun_root.resolve())}"
            if rerun_root is not None
            else str(rerun)
        ),
        "rerun_sha256": sha256_file(rerun),
        "rerun_poseconf_git_sha": new_prov.get("poseconf_git_sha"),
        "provenance_keys_differing": {
            k: {"committed": old_prov.get(k), "rerun": new_prov.get(k)} for k in differing
        },
        "leaves_compared": n,
        "identical": not diffs,
        "n_differences": len(diffs),
        "differences": diffs[:MAX_LISTED],
    }


def main(argv: list[str] | None = None) -> int:
    """Write the comparison record; return 1 if any pair differs."""
    args = parse_args(argv)
    pairs = [compare(committed, rerun, args.rerun_root) for committed, rerun in args.pair]
    dirty = [p["rerun"] for p in pairs if str(p["rerun_poseconf_git_sha"]).endswith("-dirty")]
    if dirty:
        raise SystemExit(f"re-runs must come from a clean tree; dirty: {dirty}")
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "rerun_check",
        "what": args.what,
        "ignored_keys": sorted(IGNORED_KEYS),
        "all_identical": all(p["identical"] for p in pairs),
        "pairs": pairs,
        "provenance": {"poseconf_git_sha": poseconf_git_sha(), "created_at": utc_now_iso()},
    }
    write_result_json(args.out, result)
    for p in pairs:
        state = "identical" if p["identical"] else f"{p['n_differences']} differences"
        print(f"{p['committed']}: {p['leaves_compared']} leaves, {state}")
    return 0 if result["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
