"""Export the variance graph (head on) and P1's keypoint + detector graphs (head off), fp32 and fp16.

All graphs go to the gitignored `exports/`; P1's are re-exported from the pinned checkpoints with
P1's own `spec_for_config`, so the head-off baseline is P1's artifact, not an approximation of it.
Writes `results/export/onnx_export.json`: per graph, the ONNX SHA-256, its source checkpoint
SHA-256(s), and P1's `graph_summary` (op histogram, forbidden ops, I/O signature, opset).
GPU needed for fp16 (P1 refuses a CPU half-precision export).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from poseconf.engine.train_variance import load_variance_net
from poseconf.export.to_onnx import VARIANCE_OUTPUTS, VarianceExportNet
from poseconf.p1_adapter import (
    load_p1_config,
    onnx_export,
    onnx_export_p1,
    onnx_forbidden_ops,
    onnx_graph_summary,
    p1_commit,
    p1_paths,
)
from poseconf.provenance import (
    RESULT_SCHEMA,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    write_result_json,
)

#: P1's stage-A graph, re-exported for the end-to-end frame budget.
DETECTOR_RUN = "detector"
OPSET = 17


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/variance_head.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("exports"))
    parser.add_argument("--fp16", action="store_true", help="also emit the fp16 graphs (CUDA)")
    parser.add_argument("--result", type=Path, default=Path("results/export/onnx_export.json"))
    return parser.parse_args(argv)


def _entry(path: Path, *, role: str, precision: str, sources: dict[str, str]) -> dict[str, Any]:
    summary = onnx_graph_summary(path)
    if summary["forbidden_ops_present"]:
        raise ValueError(f"{path} contains forbidden ops {summary['forbidden_ops_present']}")
    if summary["opset"] != OPSET:
        raise ValueError(f"{path} has opset {summary['opset']}, expected {OPSET}")
    return {
        "role": role,
        "precision": precision,
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "source_checkpoints_sha256": sources,
        "graph": summary,
    }


def main(argv: list[str] | None = None) -> int:
    """Export every graph and write the export record."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = p1_paths(args.paths)
    run, name = config["p1_run"], config["run_name"]
    vhead_ckpt = Path(config["out_root"]) / name / "best.pt"
    p1_ckpt = paths.p1_runs / run / "best.pt"
    crop = int(load_p1_config(run, paths=paths)["crop"]["size"])
    precisions = ["fp32", "fp16"] if args.fp16 else ["fp32"]
    if args.fp16 and not torch.cuda.is_available():
        raise RuntimeError("--fp16 needs CUDA (P1 refuses a CPU half-precision export)")

    graphs: list[dict[str, Any]] = []
    for precision in precisions:
        half = precision == "fp16"
        suffix = "_fp16" if half else ""
        # A fresh load per precision: P1's export halves the module in place.
        net, meta = load_variance_net(vhead_ckpt, paths=paths, device=torch.device("cpu"))
        out = onnx_export(
            VarianceExportNet(net),
            output_path=args.out_dir / f"{name}{suffix}.onnx",
            input_shape=(1, 1, crop, crop),
            output_names=VARIANCE_OUTPUTS,
            half_precision=half,
            opset=OPSET,
        )
        graphs.append(
            _entry(
                out,
                role="head_on",
                precision=precision,
                sources={
                    "p1_keypoint": sha256_file(p1_ckpt),
                    "variance_head": sha256_file(vhead_ckpt),
                },
            )
        )
        for p1_run, role in ((run, "head_off"), (DETECTOR_RUN, "detector")):
            out, ckpt = onnx_export_p1(
                p1_run,
                paths=paths,
                output_path=args.out_dir / f"{p1_run}{suffix}.onnx",
                half_precision=half,
            )
            graphs.append(
                _entry(out, role=role, precision=precision, sources={p1_run: sha256_file(ckpt)})
            )
        del net, meta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"exported {precision}")

    record = {
        "schema": RESULT_SCHEMA,
        "kind": "onnx_export",
        "run": run,
        "variance_head_run": name,
        "opset": OPSET,
        "forbidden_ops": sorted(onnx_forbidden_ops()),
        "variance_outputs": list(VARIANCE_OUTPUTS),
        "export_convention": (
            "P1's export_model: eval mode, opset 17, constant folding, dynamic batch only, "
            "onnx.checker + strict shape inference, forbidden ops refused, named I/O; fp16 = "
            "halved module (decode and variance readout keep explicit fp32 casts)"
        ),
        "graphs": graphs,
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "p1_commit": p1_commit(),
            "p1_checkpoint_sha256": sha256_file(p1_ckpt),
            "variance_head_checkpoint_sha256": sha256_file(vhead_ckpt),
            "torch": torch.__version__,
            "created_at": utc_now_iso(),
        },
    }
    write_result_json(args.result, record)
    print(f"wrote {args.result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
