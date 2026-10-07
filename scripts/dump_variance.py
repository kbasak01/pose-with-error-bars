"""Phase 5: variance-head sidecars for the Phase 3 prediction dumps (GPU).

For each domain x arm, re-runs P1's keypoint stage on the dump's own boxes (same order, same batch
size, TF32 off) with the trained head reading P1's decoder features, and writes
`dumps_root/<run>/<domain>_<arm>_vhead.npz` (`filename, kp_pred_crop, vhead_chol_crop,
vhead_cov_crop, vhead_cov_full`). The parity-gated Phase 3 dumps are not rewritten; P1's
coordinates must equal them bit for bit or the script stops. Predictions only: no label is read
(the HIL `gt_crop` arm's input box is the label box already stored in that oracle dump).

    python scripts/dump_variance.py --variance-head runs/vhead_a2_s1337/best.pt \\
        --all-domains --all-arms --device cuda --tf32 off
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

from poseconf.data.dump import (
    CROP_ARMS,
    arm_tag,
    dump_file,
    load_dump,
    run_variance_dump,
    variance_dump_file,
    write_dump,
)
from poseconf.engine.train_variance import load_variance_net
from poseconf.p1_adapter import DOMAINS, eval_frames, load_pipeline, p1_commit, p1_paths
from poseconf.provenance import poseconf_git_sha, sha256_file, utc_now_iso

#: Schema tag of the sidecar's meta record.
VHEAD_SCHEMA = "poseconf.dump_vhead.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--variance-head", type=Path, required=True, help="selected best.pt")
    domains = parser.add_mutually_exclusive_group(required=True)
    domains.add_argument("--domain", nargs="+", choices=DOMAINS)
    domains.add_argument("--all-domains", action="store_true")
    arms = parser.add_mutually_exclusive_group(required=True)
    arms.add_argument("--crop-source", nargs="+", choices=CROP_ARMS)
    arms.add_argument("--all-arms", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tf32", choices=("off", "on"), default="off")
    parser.add_argument("--dumps-root", type=Path, default=None, help="default: paths dumps_root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write one sidecar per requested domain x arm."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dump_cfg = config["dump"]
    run = dump_cfg["run"]
    paths = p1_paths(args.paths)
    root = args.dumps_root or paths.dumps_root
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    tf32 = args.tf32 == "on"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False

    net, head_meta = load_variance_net(args.variance_head, paths=paths, device=device)
    if head_meta["p1_run"] != run:
        raise SystemExit(f"head was trained on {head_meta['p1_run']}, dump is {run}")
    head = net.head.eval()
    domains = list(DOMAINS) if args.all_domains else args.domain
    arms = list(CROP_ARMS) if args.all_arms else args.crop_source
    for domain in domains:
        frames = eval_frames(domain, paths=paths)
        for arm in arms:
            started = time.perf_counter()
            base_path = dump_file(root, run, domain, arm, subset=False)
            dump, base_meta = load_dump(base_path)
            if base_meta["limit"] is not None:
                raise SystemExit(f"{base_path} is a truncated debug dump")
            batch_size = int(base_meta["runtime"]["batch_size"])
            pipeline = load_pipeline(run, crop_source=arm, paths=paths, device=device)
            arrays = run_variance_dump(
                pipeline,
                frames,
                head,
                dump,
                batch_size=batch_size,
                decode_threads=int(dump_cfg["decode_threads"]),
                log=lambda line, d=domain, a=arm: print(f"[{d} {a}]{line}", flush=True),
            )
            meta = {
                "schema": VHEAD_SCHEMA,
                "kind": "variance_sidecar",
                "run": run,
                "domain": domain,
                "crop_source": arm,
                "tag": arm_tag(domain, arm),
                "oracle": base_meta["oracle"],
                "base_dump": base_path.name,
                "base_dump_sha256": sha256_file(base_path),
                "kp_pred_crop_bit_equal_to_base": True,
                "variance_head": {
                    "checkpoint": str(args.variance_head),
                    "checkpoint_sha256": sha256_file(args.variance_head),
                    "run_name": head_meta["config"]["run_name"],
                    "epoch": head_meta["epoch"],
                    "val_tune_nll": head_meta["val_tune_nll"],
                    "trained_at_git_sha": head_meta["git_sha"],
                },
                "provenance": {
                    "poseconf_git_sha": poseconf_git_sha(),
                    "p1_commit": p1_commit(),
                    "p1_checkpoint_sha256": head_meta["p1_checkpoint_sha256"],
                    "created_at": utc_now_iso(),
                },
                "runtime": {
                    "torch": torch.__version__,
                    "device": str(device),
                    "tf32": tf32,
                    "cudnn_benchmark": False,
                    "batch_size": batch_size,
                },
                "units": {
                    "vhead_chol_crop": "crop px (lower Cholesky l0, l1, l2)",
                    "vhead_cov_crop": "crop px^2",
                    "vhead_cov_full": "full-frame px^2",
                },
            }
            path = write_dump(
                variance_dump_file(root, run, domain, arm), arrays, meta, compress=True
            )
            print(
                f"[{domain} {arm}] {len(frames)} frames, coords bit-equal to the dump; "
                f"{time.perf_counter() - started:.0f} s -> {path}",
                flush=True,
            )
            del pipeline
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
