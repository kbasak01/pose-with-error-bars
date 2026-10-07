"""Phase 7 latency: keypoint stage head on vs off, the conformal post-process, and the full frame.

Three result files under `results/latency/`, each with the A4000-is-not-a-Jetson caveat and the
provider each row actually ran on:

* `keypoint_stage.json`: P1's `benchmark_onnx` on the head-off graph (P1 `keypoint_a2`, re-exported
  by `scripts/export_onnx.py`) and the head-on variance graph, fp32/fp16, on every provider that
  constructs (CUDA, TensorRT), batch 1/4/8/16. TensorRT rows add the per-node provider split and a
  cold engine-build time.
* `postprocess.json`: CPU, one frame per call, 64 real `val_test` frames from the dumps cycled:
  `ConformalPoseHead.predict` per deployable score (C1/C2 include the crop -> full covariance
  mapping), linearised propagation alone, sampled propagation for M in {16, 64, 256}.
* `frame_budget.json`: P1's `end_to_end_frame_budget` (predicted crop, end to end) on ORT CUDA,
  head off (P1's pipeline) vs head on (variance graph + an `uncertainty` stage running C1 and C2).
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml

from poseconf.conformal.head import ConformalPoseHead
from poseconf.conformal.propagate import sampled_extent
from poseconf.conformal.scores import set_mahalanobis
from poseconf.conformal.so3 import quat_to_matrix
from poseconf.data.dump import dump_file, load_dump, variance_dump_file
from poseconf.data.splits import load_split, manifest_sha256
from poseconf.engine.level_c import attach_variance
from poseconf.export.bench import (
    HARDWARE_CAVEAT,
    UncertaintyFrame,
    time_calls,
    uncertainty_inputs,
)
from poseconf.p1_adapter import (
    eval_frames,
    onnx_benchmark,
    onnx_engine_build_seconds,
    onnx_frame_budget,
    onnx_methodology_minimums,
    onnx_provider_node_counts,
    onnx_resolve_providers,
    ort_pose_pipeline,
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

RUN = "keypoint_a2"
CROP = "predicted_crop"
DOMAIN = "synthetic"
SPLIT = "synthetic_val_test"
PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider")
BATCHES = (1, 4, 8, 16)
SAMPLED_M = (16, 64, 256)
SAMPLED_SEED = 1337
CONVENTION = "abstain_allowed"  # per-frame cost does not depend on the convention
BUDGET_SCORES = ("C1", "C2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/variance_head.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--exports", type=Path, default=Path("exports"))
    parser.add_argument("--calibration", type=Path, default=Path("results/calibration"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/latency"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--keypoint-stage", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--frame-budget", action="store_true")
    return parser.parse_args(argv)


def environment() -> dict[str, Any]:
    """Software and hardware the numbers were taken on."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "numpy": np.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _heads(directory: Path, paths: Any, vhead: Path, scores: Any) -> dict[str, ConformalPoseHead]:
    heads = {}
    for score_id in scores:
        artifact = directory / f"{RUN}_{CROP}_{score_id}_{CONVENTION}_a0.10.json"
        heads[score_id] = ConformalPoseHead.from_json(
            artifact,
            p1_checkpoint=paths.p1_runs / RUN / "best.pt",
            detector_checkpoint=paths.p1_runs / "detector" / "best.pt",
            variance_head_checkpoint=vhead if score_id in ("C1", "C2") else None,
        )
    return heads


def _record(kind_detail: str, body: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "kind": "latency",
        "what": kind_detail,
        "run": RUN,
        "hardware_caveat": HARDWARE_CAVEAT,
        "methodology": (
            "P1's: >= 50 warmup and >= 500 timed iterations, p50/p90/p99/max; network rows via "
            "speedpose.export.bench (provider verified by session construction, IOBinding, "
            "compute timed apart from H2D/D2H); CPU rows via time.perf_counter per call"
        ),
        **body,
        "environment": environment(),
        "provenance": provenance,
    }


# --------------------------------------------------------------------------------------------------
# keypoint stage
# --------------------------------------------------------------------------------------------------


def keypoint_stage(args: argparse.Namespace, name: str) -> dict[str, Any]:
    """Head off vs head on, per precision, provider and batch size."""
    graphs = {
        ("head_off", "fp32"): args.exports / f"{RUN}.onnx",
        ("head_off", "fp16"): args.exports / f"{RUN}_fp16.onnx",
        ("head_on", "fp32"): args.exports / f"{name}.onnx",
        ("head_on", "fp16"): args.exports / f"{name}_fp16.onnx",
    }
    warmup, iters = args.warmup, args.iters
    rows, trt, unavailable = [], [], {}
    for (role, precision), path in graphs.items():
        verified = onnx_resolve_providers(PROVIDERS, onnx_path=path)
        for provider in PROVIDERS:
            if provider not in verified:
                unavailable[f"{role}/{precision}/{provider}"] = "session construction failed"
                continue
            for row in onnx_benchmark(
                path,
                providers=(provider,),
                batch_sizes=BATCHES,
                warmup_iters=warmup,
                timed_iters=iters,
            ):
                rows.append({"role": role, "precision": precision, "graph": str(path), **row})
            print(f"{role} {precision} {provider}: done", flush=True)
            if provider == "TensorrtExecutionProvider":
                with tempfile.TemporaryDirectory() as workspace:
                    build = onnx_engine_build_seconds(path, batch=1, cache=Path(workspace))
                trt.append(
                    {
                        "role": role,
                        "precision": precision,
                        "node_provider_counts_batch1": onnx_provider_node_counts(
                            path, provider=provider, batch=1
                        ),
                        "engine_build_seconds_batch1": build,
                    }
                )
    ratios = []
    for provider in PROVIDERS:
        for precision in ("fp32", "fp16"):
            for batch in BATCHES:
                pick = {
                    r["role"]: r
                    for r in rows
                    if r["provider"] == provider
                    and r["precision"] == precision
                    and r["batch_size"] == batch
                }
                if {"head_off", "head_on"} <= set(pick):
                    off, on = pick["head_off"], pick["head_on"]
                    ratios.append(
                        {
                            "provider": provider,
                            "precision": precision,
                            "batch_size": batch,
                            "p50_head_on_minus_off_ms": on["p50_ms"] - off["p50_ms"],
                            "p99_head_on_minus_off_ms": on["p99_ms"] - off["p99_ms"],
                            "p50_ratio_on_over_off": on["p50_ms"] / off["p50_ms"],
                        }
                    )
    return {
        "graphs": {
            f"{r}/{p}": {"path": str(g), "sha256": sha256_file(g)} for (r, p), g in graphs.items()
        },
        "outputs_note": (
            "head_off is P1's keypoint graph (coords, confidence, heatmaps); head_on is the "
            "variance graph (coords, confidence, cov_chol, keypoint_empty). Compute is timed apart "
            "from D2H, so the different output sets affect d2h_ms, not p50/p99."
        ),
        "rows": rows,
        "head_on_vs_off": ratios,
        "tensorrt": trt,
        "unavailable": unavailable,
    }


# --------------------------------------------------------------------------------------------------
# post-process
# --------------------------------------------------------------------------------------------------


def _frames(paths: Any, n: int) -> dict[str, Any]:
    """The first n val_test frames of the dump + variance sidecar (predictions only)."""
    dump_path = dump_file(paths.dumps_root, RUN, DOMAIN, CROP, subset=False)
    dump, meta = load_dump(dump_path)
    side, side_meta = load_dump(variance_dump_file(paths.dumps_root, RUN, DOMAIN, CROP))
    if side_meta["base_dump_sha256"] != sha256_file(dump_path):
        raise ValueError("variance sidecar was made from a different dump")
    dump = attach_variance(dump, side)
    row_of = {name: i for i, name in enumerate(dump["filename"])}
    rows = np.array([row_of[name] for name in load_split(SPLIT)[:n]])
    return {"dump": dump, "rows": rows, "dump_sha256": sha256_file(dump_path)}


def postprocess(args: argparse.Namespace, paths: Any, vhead: Path) -> dict[str, Any]:
    """Single-frame CPU cost of the head and the propagation estimators."""
    minimums = onnx_methodology_minimums()
    data = _frames(paths, args.frames)
    dump, rows = data["dump"], data["rows"]
    heads = _heads(args.calibration, paths, vhead, ("A1", "A2", "A3", "B1", "C1", "C2"))

    def inputs(i: int) -> tuple[Any, Any]:
        r = rows[i]
        return uncertainty_inputs(
            valid=bool(dump["success"][r]),
            rotation_quat=dump["q_pred"][r],
            translation=dump["t_pred"][r],
            keypoints_full=dump["kp_pred_full"][r],
            confidence=dump["confidence"][r],
            bbox=dump["bbox_used"][r],
            cov_chol_crop=dump["vhead_chol_crop"][r],
            keypoint_empty=dump["heatmap_empty"][r],
            affine=dump["affine"][r],
        )

    solved = [i for i in range(len(rows)) if dump["success"][rows[i]]]
    timings: dict[str, Any] = {}
    for score_id, head in heads.items():

        def call(i: int, head: ConformalPoseHead = head) -> None:
            estimate, cov = inputs(i)
            head.predict(estimate, cov=cov)

        timings[f"predict_{score_id}"] = time_calls(
            call, n_inputs=len(rows), warmup=args.warmup, iters=args.iters, minimums=minimums
        )
        print(f"predict {score_id}: p50 {timings[f'predict_{score_id}']['p50_ms']:.4f} ms")

    c2 = heads["C2"]

    def linearised(k: int) -> None:
        estimate, cov = inputs(solved[k])
        c2.pose_sigma(estimate, cov)

    timings["linearised_propagation"] = time_calls(
        linearised, n_inputs=len(solved), warmup=args.warmup, iters=args.iters, minimums=minimums
    )

    c1 = heads["C1"]
    order = c2.payload["quaternion_order"]
    for m in SAMPLED_M:

        def sampled(k: int, m: int = m) -> None:
            estimate, cov = inputs(solved[k])
            kset = set_mahalanobis(
                estimate.keypoints[None],
                np.array([True]),
                c1.quantile,
                cov=cov[None],
                unconstrained=estimate.unconstrained[None],
            )
            rotation = quat_to_matrix(estimate.q_hat[None], order)[0]
            sampled_extent(
                kset,
                0,
                rotation,
                estimate.t_hat,
                c2.geometry,
                samples=m,
                rng=np.random.default_rng([SAMPLED_SEED, k]),
            )

        timings[f"sampled_propagation_M{m}"] = time_calls(
            sampled, n_inputs=len(solved), warmup=args.warmup, iters=args.iters, minimums=minimums
        )
        print(f"sampled M={m}: p50 {timings[f'sampled_propagation_M{m}']['p50_ms']:.3f} ms")

    return {
        "domain": DOMAIN,
        "subset": SPLIT,
        "crop_source": CROP,
        "frames": int(len(rows)),
        "frames_solved": len(solved),
        "convention": CONVENTION,
        "artifacts": {s: h.calibration_id for s, h in heads.items()},
        "definitions": {
            "predict_<score>": (
                "one frame: graph outputs -> FrameEstimate (+ cov_chol -> full-frame covariance "
                "via crop_cov_to_full) -> ConformalPoseHead.predict; C2 includes its linearisation"
            ),
            "linearised_propagation": "ConformalPoseHead.pose_sigma alone on solved frames",
            "sampled_propagation_M<m>": (
                "propagate.sampled_extent with m keypoint configurations on C1's sets at the C1 "
                "quantile, solved frames (offline estimator: an inner approximation)"
            ),
        },
        "timings": timings,
        "dump_sha256": data["dump_sha256"],
    }


# --------------------------------------------------------------------------------------------------
# full frame
# --------------------------------------------------------------------------------------------------


def frame_budget(args: argparse.Namespace, paths: Any, name: str, vhead: Path) -> dict[str, Any]:
    """P1's frame budget, head off vs head on (+ uncertainty stage), ORT CUDA, fp32 / fp16."""
    frames_index = eval_frames(DOMAIN, paths=paths)
    by_name = dict(zip(frames_index.filenames, frames_index.image_paths, strict=True))
    names = load_split(SPLIT)[: args.frames]
    images = [cv2.imread(by_name[n], cv2.IMREAD_GRAYSCALE) for n in names]
    if any(image is None for image in images):
        raise RuntimeError("failed to read a val_test frame for the frame budget")
    heads = list(_heads(args.calibration, paths, vhead, BUDGET_SCORES).values())
    provider = "CUDAExecutionProvider"
    budgets: dict[str, Any] = {}
    for precision in ("fp32", "fp16"):
        suffix = "" if precision == "fp32" else "_fp16"
        detector = args.exports / f"detector{suffix}.onnx"
        for role, graph in (
            ("head_off", f"{RUN}{suffix}.onnx"),
            ("head_on", f"{name}{suffix}.onnx"),
        ):
            pipeline = ort_pose_pipeline(
                RUN,
                paths=paths,
                keypoint_onnx=args.exports / graph,
                detector_onnx=detector,
                provider=provider,
                with_uncertainty=role == "head_on",
            )
            runner = UncertaintyFrame(pipeline, heads) if role == "head_on" else pipeline
            budgets[f"onnxruntime_cuda_{precision}_{role}"] = onnx_frame_budget(
                runner, images, warmup=args.warmup, iters=args.iters
            )
            print(f"frame budget {precision} {role}: done", flush=True)
            del runner, pipeline
            torch.cuda.empty_cache()
    deltas = {}
    for precision in ("fp32", "fp16"):
        off = budgets[f"onnxruntime_cuda_{precision}_head_off"]
        on = budgets[f"onnxruntime_cuda_{precision}_head_on"]
        deltas[precision] = {
            "total_p50_delta_ms": on["total_p50_ms"] - off["total_p50_ms"],
            "total_p99_delta_ms": on["total_p99_ms"] - off["total_p99_ms"],
            "uncertainty_p50_ms": on["uncertainty_p50_ms"],
            "uncertainty_p99_ms": on["uncertainty_p99_ms"],
            "keypoints_p50_delta_ms": on["keypoints_p50_ms"] - off["keypoints_p50_ms"],
        }
    return {
        "domain": DOMAIN,
        "subset": SPLIT,
        "frames": len(images),
        "arm": "predicted_crop (end to end, detector graph on ORT, no ground-truth box)",
        "provider": provider,
        "uncertainty_scores": list(BUDGET_SCORES),
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "stats_note": "P1's end_to_end_frame_budget reports p50 and p99 per stage only",
        "budgets": budgets,
        "head_on_minus_off": deltas,
        "note": (
            "Stage times from P1's PosePipeline.__call__ (inherited unchanged by the ORT "
            "pipeline); the head-on row adds an 'uncertainty' stage (covariance mapping + C1 and "
            "C2 predict). Totals are per-frame sums of stages, then percentiles."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the requested benchmarks."""
    args = parse_args(argv)
    if not (args.all or args.keypoint_stage or args.postprocess or args.frame_budget):
        raise SystemExit(
            "choose --all or at least one of --keypoint-stage/--postprocess/--frame-budget"
        )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = p1_paths(args.paths)
    name = config["run_name"]
    vhead = Path(config["out_root"]) / name / "best.pt"
    provenance = {
        "poseconf_git_sha": poseconf_git_sha(),
        "p1_commit": p1_commit(),
        "p1_checkpoint_sha256": sha256_file(paths.p1_runs / RUN / "best.pt"),
        "detector_checkpoint_sha256": sha256_file(paths.p1_runs / "detector" / "best.pt"),
        "variance_head_checkpoint_sha256": sha256_file(vhead),
        "split_manifest_sha256": manifest_sha256(),
        "export_record_sha256": sha256_file(Path("results/export/onnx_export.json")),
        "created_at": utc_now_iso(),
    }
    jobs = []
    if args.all or args.postprocess:
        jobs.append(("postprocess", lambda: postprocess(args, paths, vhead)))
    if args.all or args.keypoint_stage:
        jobs.append(("keypoint_stage", lambda: keypoint_stage(args, name)))
    if args.all or args.frame_budget:
        jobs.append(("frame_budget", lambda: frame_budget(args, paths, name, vhead)))
    for what, job in jobs:
        record = _record(what, job(), provenance)
        out = args.out_dir / f"{what}.json"
        write_result_json(out, record)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
