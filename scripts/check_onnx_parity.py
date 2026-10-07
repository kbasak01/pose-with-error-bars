"""PyTorch vs ONNX Runtime parity of the exported variance graph on real synthetic `val_test` crops.

Reference: the export wrapper in fp32 on CUDA with TF32 off (`torch.backends.*.allow_tf32 = False`).
Backends: the fp32 graph on ORT CUDA with `use_tf32 = 0`, and the fp16 graph on ORT CUDA (compared
against the fp32 reference, stated). Inputs: the first `--samples` names of the `val_test` manifest,
read as P1's crop-cache crops (images only; no label is read, nothing is fitted). Also P1's
four-cell TF32 ablation on the same crops, and P1's committed `keypoint_a2` parity record beside the
`coords` delta (an inherited miss). Writes `results/export/onnx_parity.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from poseconf.data.splits import load_split, manifest_sha256
from poseconf.engine.train_variance import load_variance_net
from poseconf.export.parity import FP32_VARIANCE_GATE, gate_status, parity_record
from poseconf.export.to_onnx import VARIANCE_OUTPUTS, VarianceExportNet
from poseconf.p1_adapter import (
    crop_dataset_filenames,
    keypoint_crop_dataset,
    load_p1_config,
    onnx_resolve_providers,
    onnx_tf32_ablation,
    p1_commit,
    p1_onnx_parity_record,
    p1_parity_tolerances,
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

PROVIDER = "CUDAExecutionProvider"
MIN_SAMPLES = 200  # plan / checklist minimum
TF32_ABLATION_SAMPLES = 192  # P1's
CPU_REFERENCE_SAMPLES = 256  # CPU control: same graph and crops, no CUDA kernels on either side


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/variance_head.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--exports", type=Path, default=Path("exports"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--split", default="synthetic_val_test")
    parser.add_argument(
        "--c1-artifact",
        type=Path,
        default=Path(
            "results/calibration/keypoint_a2_predicted_crop_C1_abstain_allowed_a0.10.json"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("results/export/onnx_parity.json"))
    return parser.parse_args(argv)


def _inputs(run: str, split: str, samples: int, paths: Any, seed: int) -> tuple[Any, list[str]]:
    """The first `samples` names of `split` (manifest order) as a Subset of P1's crop dataset."""
    if samples < MIN_SAMPLES:
        raise ValueError(f"parity needs at least {MIN_SAMPLES} inputs, got {samples}")
    dataset = keypoint_crop_dataset(run, "validation", paths=paths, augmentation="a0", seed=seed)
    index = {name: i for i, name in enumerate(crop_dataset_filenames(dataset))}
    names = [n for n in load_split(split) if n in index][:samples]
    if len(names) < samples:
        raise ValueError(f"only {len(names)} {split} frames are in the crop cache")
    return Subset(dataset, [index[n] for n in names]), names


def _session(path: Path, *, tf32: bool) -> Any:
    session = ort.InferenceSession(
        str(path), providers=[(PROVIDER, {"use_tf32": 1 if tf32 else 0})]
    )
    if session.get_providers()[0] != PROVIDER:
        raise RuntimeError(f"{path} did not get {PROVIDER}; refusing to label the row with it")
    return session


def _run(model: Any, session: Any, loader: Any, device: torch.device, dtype: Any) -> dict:
    """Stacked outputs of the torch wrapper and one ORT session over `loader`."""
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        b: {k: [] for k in VARIANCE_OUTPUTS} for b in ("torch", "ort")
    }
    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            for key, value in zip(VARIANCE_OUTPUTS, model(images.to(device)), strict=True):
                collected["torch"][key].append(value.cpu().numpy())
            feed = {"images": images.numpy().astype(dtype)}
            for key, value in zip(
                VARIANCE_OUTPUTS, session.run(list(VARIANCE_OUTPUTS), feed), strict=True
            ):
                collected["ort"][key].append(np.asarray(value))
    return {b: {k: np.concatenate(v) for k, v in o.items()} for b, o in collected.items()}


def main(argv: list[str] | None = None) -> int:
    """Measure parity; exit 1 if the fp32 variance gate is not met (reported either way)."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = p1_paths(args.paths)
    run, name = config["p1_run"], config["run_name"]
    vhead_ckpt = Path(config["out_root"]) / name / "best.pt"
    onnx_fp32 = args.exports / f"{name}.onnx"
    onnx_fp16 = args.exports / f"{name}_fp16.onnx"
    if PROVIDER not in onnx_resolve_providers((PROVIDER,), onnx_path=onnx_fp32):
        raise RuntimeError(f"{PROVIDER} cannot be constructed on this machine")

    p1_config = load_p1_config(run, paths=paths)
    seed = int(p1_config["runtime"]["seed"])
    subset, names = _inputs(run, args.split, args.samples, paths, seed)
    c1 = json.loads(args.c1_artifact.read_text(encoding="utf-8"))["payload"]

    restore = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        device = torch.device("cuda")
        net, _ = load_variance_net(vhead_ckpt, paths=paths, device=device)
        model = VarianceExportNet(net).eval()
        sessions = {
            "ort_fp32": _session(onnx_fp32, tf32=False),
            "ort_fp16": _session(onnx_fp16, tf32=False),
        }
        loader = DataLoader(subset, batch_size=48, shuffle=False, num_workers=4)
        runs = {
            "ort_fp32": _run(model, sessions["ort_fp32"], loader, device, np.float32),
            "ort_fp16": _run(model, sessions["ort_fp16"], loader, device, np.float16),
        }
        stacked = {"torch_fp32": runs["ort_fp32"]["torch"]}
        stacked.update({b: r["ort"] for b, r in runs.items()})
        ablation = onnx_tf32_ablation(
            model=model,
            dataset=subset,
            onnx_fp32=onnx_fp32,
            samples=TF32_ABLATION_SAMPLES,
            output_names=VARIANCE_OUTPUTS,
        )
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = restore

    # CPU control (P1's `cpu_reference`): if the miss survives with no CUDA kernel on either side,
    # it is not TF32 or cuDNN algorithm choice but the two frameworks' fp32 arithmetic.
    cpu_net, _ = load_variance_net(vhead_ckpt, paths=paths, device=torch.device("cpu"))
    cpu_session = ort.InferenceSession(str(onnx_fp32), providers=["CPUExecutionProvider"])
    cpu_subset = Subset(subset, range(CPU_REFERENCE_SAMPLES))
    cpu = _run(
        VarianceExportNet(cpu_net).eval(),
        cpu_session,
        DataLoader(cpu_subset, batch_size=32, shuffle=False),
        torch.device("cpu"),
        np.float32,
    )
    cpu_record = parity_record(
        {"torch_fp32": cpu["torch"], "cpu_reference": cpu["ort"]},
        c1_quantile=float(c1["quantile"]),
    )

    tolerances = p1_parity_tolerances()
    record = parity_record(stacked, c1_quantile=float(c1["quantile"]))
    p1_record, p1_path = p1_onnx_parity_record(run)
    out: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "kind": "parity",
        "run": run,
        "variance_head_run": name,
        "domain": "synthetic",
        "subset": args.split,
        "samples": len(names),
        "inputs": "P1 crop-cache crops (GT-box, margin_eval), first N names of the manifest",
        "provider": PROVIDER,
        "reference": "torch_fp32 on CUDA, TF32 off (cudnn and matmul)",
        "ort_options": {"use_tf32": 0},
        "onnx_fp32": {"path": str(onnx_fp32), "sha256": sha256_file(onnx_fp32)},
        "onnx_fp16": {"path": str(onnx_fp16), "sha256": sha256_file(onnx_fp16)},
        "tolerances": {
            "variance_fp32_tensor_max_abs": FP32_VARIANCE_GATE,
            "p1": tolerances,
            "fp16_note": (
                "fp16 is held to P1's stated fp16 relaxation (tensor max abs "
                f"{tolerances['fp16']['tensor_max_abs']}), by construction not by observation; "
                "the measured deltas are reported unrelaxed beside it"
            ),
        },
        **record,
        "cpu_reference": {
            "samples": CPU_REFERENCE_SAMPLES,
            "what": "torch fp32 on CPU vs the fp32 graph on ORT CPUExecutionProvider, same crops",
            **{key: block["cpu_reference"] for key, block in cpu_record.items()},
        },
        "tf32_ablation": ablation,
        "p1_inherited": {
            "record": str(p1_path.relative_to(POSECONF_ROOT)),
            "record_sha256": sha256_file(p1_path),
            "samples": p1_record["samples"],
            "split": p1_record["split"],
            "coords": {
                b: p1_record["tensor_parity"][b]["coords"] for b in ("ort_fp32", "ort_fp16")
            },
            "all_met": p1_record["gate_status"]["all_met"],
            "failed": p1_record["gate_status"]["failed"],
        },
        "gate_status": None,
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "p1_commit": p1_commit(),
            "p1_checkpoint_sha256": sha256_file(paths.p1_runs / run / "best.pt"),
            "variance_head_checkpoint_sha256": sha256_file(vhead_ckpt),
            "split_manifest_sha256": manifest_sha256(),
            "c1_artifact": str(args.c1_artifact),
            "c1_calibration_id": json.loads(args.c1_artifact.read_text("utf-8"))["calibration_id"],
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "created_at": utc_now_iso(),
        },
    }
    out["gate_status"] = gate_status(out, float(tolerances["fp16"]["tensor_max_abs"]))
    write_result_json(args.out, out)
    print(json.dumps(out["gate_status"], indent=1))
    return 0 if out["gate_status"]["variance_outputs_fp32_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
