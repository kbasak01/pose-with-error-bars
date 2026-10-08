"""Stage the `phase-9-complete` release assets and write their SHA256SUMS manifest.

Assets: the 12 `ConformalPoseHead` calibration artifacts + their index, the selected variance-head
checkpoint, and its fp32 ONNX graph. Before anything is staged, every hash is cross-checked against
the committed record that already names it:
- the checkpoint against `payload.locks.variance_head_sha256` of every C1/C2 calibration artifact;
- the ONNX graph against `results/export/onnx_export.json`.
A mismatch refuses the build. The fp16 graph is not shipped: its variance outputs are not faithful
(`results/export/onnx_parity.json`).

Writes the staged files to `--stage` (gitignored), plus `results/release/SHA256SUMS.txt` (the same
manifest, committed) and `results/release/release.json`.

    python scripts/build_release.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from poseconf.provenance import (
    RESULT_SCHEMA,
    parse_sha256sums,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    write_result_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="phase-9-complete")
    parser.add_argument("--calibration-dir", type=Path, default=Path("results/calibration"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/vhead_a2_s1337/best.pt"))
    parser.add_argument("--checkpoint-name", default="vhead_a2_s1337_best.pt")
    parser.add_argument("--onnx", type=Path, default=Path("exports/vhead_a2_s1337.onnx"))
    parser.add_argument(
        "--export-record", type=Path, default=Path("results/export/onnx_export.json")
    )
    parser.add_argument("--stage", type=Path, default=Path("release"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/release"))
    return parser.parse_args(argv)


def calibration_files(calibration_dir: Path) -> list[Path]:
    """The calibration artifacts and their index, sorted by name.

    Raises:
        FileNotFoundError: If the directory has no artifact or no `index.json`.
    """
    files = sorted(calibration_dir.glob("*.json"))
    if not files or calibration_dir / "index.json" not in files:
        raise FileNotFoundError(f"{calibration_dir}: no calibration artifacts or no index.json")
    return files


def variance_head_locks(files: list[Path]) -> set[str]:
    """Every `payload.locks.variance_head_sha256` among the calibration artifacts (C1/C2)."""
    locks = set()
    for path in files:
        record = json.loads(path.read_text("utf-8"))
        lock = record.get("payload", {}).get("locks", {}).get("variance_head_sha256")
        if lock:
            locks.add(lock)
    return locks


def recorded_onnx_sha256(export_record: Path, graph: Path) -> str:
    """The SHA-256 that the export record gives for `graph` (matched by its `exports/` path).

    Raises:
        KeyError: If the record does not list the graph.
    """
    graphs = json.loads(export_record.read_text("utf-8"))["graphs"]
    for entry in graphs:
        if Path(entry["path"]).name == graph.name:
            return entry["sha256"]
    raise KeyError(f"{export_record} lists no graph named {graph.name}")


def manifest_text(hashes: dict[str, str]) -> str:
    """`sha256sum` format, sorted by name, so `sha256sum -c` verifies a download directly."""
    return "".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes))


def main(argv: list[str] | None = None) -> int:
    """Cross-check, stage and hash the release assets; return the process exit code."""
    args = parse_args(argv)
    cal = calibration_files(args.calibration_dir)
    for path in (args.checkpoint, args.onnx):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing: it is gitignored and lives on the workstation"
            )

    checkpoint_sha = sha256_file(args.checkpoint)
    locks = variance_head_locks(cal)
    if locks != {checkpoint_sha}:
        raise ValueError(f"checkpoint {checkpoint_sha} does not equal the artifact locks {locks}")
    onnx_sha = sha256_file(args.onnx)
    recorded = recorded_onnx_sha256(args.export_record, args.onnx)
    if onnx_sha != recorded:
        raise ValueError(f"{args.onnx}: {onnx_sha}, but {args.export_record} records {recorded}")

    sources = {path.name: path for path in cal}
    sources[args.checkpoint_name] = args.checkpoint
    sources[args.onnx.name] = args.onnx
    if len(sources) != len(cal) + 2:
        raise ValueError("asset names collide")
    args.stage.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, source in sources.items():
        shutil.copyfile(source, args.stage / name)
        hashes[name] = sha256_file(args.stage / name)
        if hashes[name] != sha256_file(source):
            raise ValueError(f"{name}: the staged copy differs from {source}")

    text = manifest_text(hashes)
    if parse_sha256sums(text) != hashes:
        raise ValueError("manifest does not round-trip")
    (args.stage / "SHA256SUMS.txt").write_text(text, "utf-8")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "SHA256SUMS.txt").write_text(text, "utf-8")
    record: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "kind": "release",
        "tag": args.tag,
        "n_assets": len(hashes),
        "assets": {
            name: {
                "sha256": hashes[name],
                "size_bytes": (args.stage / name).stat().st_size,
                "source": str(sources[name]),
            }
            for name in sorted(hashes)
        },
        "cross_checks": {
            "variance_head_sha256_equals_calibration_locks": True,
            "onnx_sha256_equals_export_record": True,
            "export_record": str(args.export_record),
        },
        "excluded": {
            "vhead_a2_s1337_fp16.onnx": "variance outputs not faithful in fp16 "
            "(results/export/onnx_parity.json)",
            "vhead_a2_s2026": "robustness seed; not selected, behind no calibration artifact",
            "P1 detector / keypoint_a2 graphs and checkpoints": "P1's own phase-9-complete release",
        },
        "licence_note": "Weights and graphs were trained on SPEED+ and carry its CC BY-NC-SA 4.0 "
        "NonCommercial and ShareAlike terms; the code is MIT.",
        "provenance": {"poseconf_git_sha": poseconf_git_sha(), "created_at": utc_now_iso()},
    }
    out = write_result_json(args.out_dir / "release.json", record)
    print(f"staged {len(hashes)} assets in {args.stage}/; wrote {out} and SHA256SUMS.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
