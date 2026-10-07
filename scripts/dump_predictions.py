"""Phase 3: dump P1's per-frame predictions, heatmap moments and features through P1's own stages.

Writes, under `dumps_root/<run>/` (gitignored, never committed):
`<domain>_<arm>.npz` (predictions), `<domain>_<arm>_subset.npz` (heatmaps + decoder features of a
fixed 256-frame subset) and the label files (`synthetic_labels.npz`,
`<hil>_labels_pool{A,B}.npz`). See `poseconf.data.dump` for the layout. GPU; run as a background job:

    python scripts/dump_predictions.py --run keypoint_a2 --all-domains --all-arms --device cuda --tf32 off

`--limit N` dumps the first N frames for debugging and is refused anywhere under `dumps_root`, so a
truncated dump can never be mistaken for the real one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from poseconf.data.dump import (
    CROP_ARMS,
    arm_tag,
    dump_file,
    labels_file,
    p1_reference_arm,
    run_dump,
    subset_rows,
    write_dump,
    write_labels,
)
from poseconf.data.splits import load_split, manifest_sha256
from poseconf.p1_adapter import (
    DOMAINS,
    HIL_DOMAINS,
    EvalFrames,
    eval_frames,
    eval_split,
    keypoint_labels_file,
    load_eval_labels,
    load_p1_config,
    load_pipeline,
    p1_commit,
    p1_paths,
)
from poseconf.provenance import poseconf_git_sha, sha256_file, utc_now_iso

#: Schema tag of the dump files' meta record.
DUMP_SCHEMA = "poseconf.dump.v1"

#: Tag under which poolA labels are written: only oracle_* arms may ever read them back.
POOL_A_LABEL_TAG = "oracle_target_labels"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--run", default=None, help="P1 keypoint run (default: dump.run)")
    domains = parser.add_mutually_exclusive_group(required=True)
    domains.add_argument("--domain", nargs="+", choices=DOMAINS)
    domains.add_argument("--all-domains", action="store_true")
    arms = parser.add_mutually_exclusive_group(required=True)
    arms.add_argument("--crop-source", nargs="+", choices=CROP_ARMS)
    arms.add_argument("--all-arms", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tf32", choices=("off", "on"), default=None, help="default: dump.tf32")
    parser.add_argument("--limit", type=int, default=None, help="debug: first N frames only")
    parser.add_argument(
        "--out", type=Path, default=None, help="output root (required with --limit)"
    )
    return parser.parse_args(argv)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _set_backends(tf32: bool) -> dict[str, Any]:
    """Fix the numerics-relevant backend flags and return them for the meta record."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    # P1's evaluation never turns benchmark on; autotuning could pick a different algorithm.
    torch.backends.cudnn.benchmark = False
    return {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def _runtime(device: torch.device, flags: dict[str, Any], dump_cfg: dict[str, Any]) -> dict:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        **flags,
        "batch_size": int(dump_cfg["batch_size"]),
        "solve_workers": int(dump_cfg["solve_workers"]),
        "solve_seed": int(dump_cfg["solve_seed"]),
        "cpu_count": os.cpu_count(),
        "decode_threads": int(dump_cfg["decode_threads"]),
    }


def _write_label_files(
    domain: str, frames: EvalFrames, *, run: str, out_root: Path, paths: Any, base_meta: dict
) -> list[Path]:
    """Write the domain's label files; HIL poolA and poolB go to separate files."""
    names = [str(name) for name in frames.filenames]
    if domain not in HIL_DOMAINS:
        groups = {"all": (names, None)}
    else:
        groups = {}
        for pool, tag in (("poolA", POOL_A_LABEL_TAG), ("poolB", None)):
            members = set(load_split(f"{domain}_{pool}"))
            groups[pool] = ([name for name in names if name in members], tag)
    written = []
    for pool, (members, tag) in groups.items():
        labels = load_eval_labels(domain, paths=paths, frames=members, tag=tag)
        meta = {
            **base_meta,
            "kind": "labels",
            "pool": pool,
            "read_tag": tag,
            "n_frames": len(members),
        }
        written.append(write_labels(labels_file(out_root, run, domain, pool), labels, meta))
    return written


def main(argv: list[str] | None = None) -> int:
    """Dump every requested domain x arm."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dump_cfg = config["dump"]
    run = args.run or dump_cfg["run"]
    domains = list(DOMAINS) if args.all_domains else args.domain
    arms = list(CROP_ARMS) if args.all_arms else args.crop_source
    paths = p1_paths(args.paths)

    if args.limit is not None:
        if args.out is None or _is_within(args.out, paths.dumps_root):
            raise SystemExit(
                "--limit needs --out outside dumps_root: a truncated dump is not a dump"
            )
        out_root = args.out
    else:
        out_root = args.out or paths.dumps_root

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    tf32 = (args.tf32 == "on") if args.tf32 is not None else bool(dump_cfg["tf32"])
    runtime = _runtime(device, _set_backends(tf32), dump_cfg)

    checkpoints = {
        name: sha256_file(paths.p1_runs / name / "best.pt") for name in ("detector", run)
    }
    common = {
        "schema": DUMP_SCHEMA,
        "run": run,
        "limit": args.limit,
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "p1_commit": p1_commit(),
            "p1_checkpoint_sha256": checkpoints,
            "split_manifest_sha256": manifest_sha256(),
            "config_sha256": sha256_file(args.config),
            "created_at": utc_now_iso(),
            "pnp_config": load_p1_config(run)["pnp"],
        },
        "runtime": runtime,
    }

    for domain in domains:
        frames = eval_frames(domain, paths=paths)
        if args.limit is not None:
            frames = EvalFrames(
                domain, frames.filenames[: args.limit], frames.image_paths[: args.limit]
            )
        domain_meta = {
            **common,
            "domain": domain,
            "split": eval_split(domain),
            "keypoint_labels_sha256": sha256_file(keypoint_labels_file(domain, paths=paths)),
        }
        for path in _write_label_files(
            domain, frames, run=run, out_root=out_root, paths=paths, base_meta=domain_meta
        ):
            print(f"[{domain}] labels -> {path}", flush=True)
        subset = subset_rows(
            len(frames), int(dump_cfg["subset_size"]), int(dump_cfg["subset_seed"])
        )

        for arm in arms:
            tag = arm_tag(domain, arm)
            boxes_gt = None
            if arm == "gt_crop":
                names = [str(name) for name in frames.filenames]
                boxes_gt = load_eval_labels(domain, paths=paths, frames=names, tag=tag).bbox_tight
            started = time.perf_counter()
            print(f"[{domain} {arm}] {len(frames)} frames, tag={tag}", flush=True)
            pipeline = load_pipeline(run, crop_source=arm, paths=paths, device=device)
            output = run_dump(
                pipeline,
                frames,
                crop_source=arm,
                boxes_gt=boxes_gt,
                batch_size=int(dump_cfg["batch_size"]),
                subset=subset,
                solve_workers=int(dump_cfg["solve_workers"]),
                solve_seed=int(dump_cfg["solve_seed"]),
                decode_threads=int(dump_cfg["decode_threads"]),
                moment_tol_crop_px=float(dump_cfg["moment_mean_tol_crop_px"]),
                log=lambda line, d=domain, a=arm: print(f"[{d} {a}]{line}", flush=True),
            )
            meta = {
                **domain_meta,
                "kind": "predictions",
                "crop_source": arm,
                "p1_reference_arm": p1_reference_arm(arm),
                "tag": tag,
                "oracle": tag.startswith("oracle_"),
                "info": output.info,
                "units": {
                    "kp_pred_crop": "crop px",
                    "kp_pred_full": "full-frame px",
                    "heatmap_cov_crop": "crop px^2",
                    "heatmap_cov_full": "full-frame px^2",
                    "heatmap_entropy": "nats",
                    "t_pred": "m",
                    "q_pred": "P1 label order (scalar_first), canonical sign",
                },
            }
            main_path = write_dump(
                dump_file(out_root, run, domain, arm, subset=False),
                output.fields,
                meta,
                compress=True,
            )
            subset_path = write_dump(
                dump_file(out_root, run, domain, arm, subset=True),
                output.subset,
                {**meta, "kind": "subset"},
                compress=False,
            )
            info = output.info
            print(
                f"[{domain} {arm}] solved {info['n_solved']}/{info['n_frames']} "
                f"({info['n_solved'] / max(info['n_frames'], 1):.4f}); "
                f"moment dev {info['max_moment_mean_dev_crop_px']:.2e} px; "
                f"{time.perf_counter() - started:.0f} s -> {main_path}, {subset_path}",
                flush=True,
            )
            del pipeline
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
