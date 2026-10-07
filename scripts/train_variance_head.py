"""Phase 5: train the variance head on the frozen P1 `keypoint_a2` trunk (β-NLL, synthetic train).

Full run (GPU, ~20 epochs):

    python scripts/train_variance_head.py --config configs/variance_head.yaml

Overfit gate (`make train-var-smoke`): one fixed A2 training batch, `--max-steps` β-NLL steps; the
head's plain NLL on that batch must fall below the best homoscedastic Gaussian's (exit 1 otherwise).

Writes `runs/<run_name>/` (gitignored): `config.yaml`, `git_sha.txt`, `metrics.jsonl`, `best.pt`,
`last.pt`, `diagnostics_val_tune.json`, `smoke.json`; and the committed summaries
`results/level_c/variance_head_smoke.json` / `variance_head_training.json` (kind `variance_head`).
Selection sees `val_tune` only; `val_cal`, `val_test` and HIL frames are never loaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from poseconf import p1_adapter as p1
from poseconf.data.splits import load_split, manifest_sha256
from poseconf.engine.head_metrics import head_quality
from poseconf.engine.train_variance import (
    AMP_DTYPES,
    chol_stats,
    collect,
    load_variance_net,
    overfit_one_batch,
    train_head,
)
from poseconf.models.variance_head import (
    HeadConfig,
    VarianceHead,
    VarianceKeypointNet,
    chol_to_cov,
    p1_state_hash,
)
from poseconf.provenance import (
    RESULT_SCHEMA,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    write_result_json,
)

#: Committed summaries.
RESULTS_DIR = Path("results") / "level_c"

#: Synthetic-validation subsets selection must never see (CLAUDE.md invariant 3).
_FORBIDDEN_SPLITS = ("synthetic_val_cal", "synthetic_val_test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/variance_head.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument(
        "--out", type=Path, default=None, help="run dir (default out_root/run_name)"
    )
    parser.add_argument("--overfit-batch", action="store_true", help="the overfit gate")
    parser.add_argument("--max-steps", type=int, default=400, help="overfit steps")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--device", default=None, help="default: cuda if available")
    parser.add_argument("--limit-train", type=int, default=None, help="debug: seeded subset")
    parser.add_argument("--limit-val", type=int, default=None, help="debug: seeded subset")
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, help="committed summaries go here"
    )
    return parser.parse_args(argv)


def _subset(dataset: Any, limit: int | None, seed: int) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    rows = np.sort(np.random.default_rng(seed).permutation(len(dataset))[:limit])
    return torch.utils.data.Subset(dataset, rows.tolist())


def val_tune_dataset(run: str, *, paths: p1.P1Paths, seed: int) -> tuple[Any, dict[str, int]]:
    """P1's a0 validation crops restricted to `val_tune`; refuses if any other frame slips in.

    Returns:
        `(Subset, counts)`; counts has the manifest size and the served size (P1 serves only
        rows its crop cache could box, so the two may differ).
    """
    dataset = p1.keypoint_crop_dataset(run, "validation", paths=paths, augmentation="a0", seed=seed)
    names = p1.crop_dataset_filenames(dataset)
    tune = set(load_split("synthetic_val_tune"))
    keep = np.flatnonzero(np.isin(names, list(tune)))
    served = set(names[keep].tolist())
    for other in _FORBIDDEN_SPLITS:
        if served & set(load_split(other)):
            raise RuntimeError(f"val_tune selection would read {other} frames")
    return torch.utils.data.Subset(dataset, keep.tolist()), {
        "manifest": len(tune),
        "served": len(keep),
    }


def _set_backends(tf32: bool) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    return {"tf32": tf32, "cudnn_benchmark": False}


def _provenance(config_path: Path, run: str, paths: p1.P1Paths) -> dict[str, Any]:
    return {
        "poseconf_git_sha": poseconf_git_sha(),
        "p1_commit": p1.p1_commit(),
        "p1_checkpoint_sha256": sha256_file(paths.p1_runs / run / "best.pt"),
        "split_manifest_sha256": manifest_sha256(),
        "config_sha256": sha256_file(config_path),
        "created_at": utc_now_iso(),
    }


def final_diagnostics(
    checkpoint: Path,
    loader: Any,
    *,
    paths: p1.P1Paths,
    device: torch.device,
    amp: Any,
    head_config: HeadConfig,
    rho_saturation: float,
) -> dict[str, Any]:
    """The selected head vs heatmap moments vs confidence on `val_tune` (GT-box eval crops)."""
    net, meta = load_variance_net(checkpoint, paths=paths, device=device)
    arrays = collect(net, loader, device, amp)
    mask = arrays.mask & ~arrays.moment_empty
    learned_cov = chol_to_cov(torch.from_numpy(arrays.chol)).numpy()
    eig = np.linalg.eigvalsh(learned_cov[arrays.mask])
    return {
        "split": "val_tune",
        "crops": "P1 a0 eval crops (GT box from the crop cache, margin_eval)",
        "selected_epoch": meta["epoch"],
        "n_frames": len(arrays.filenames),
        "n_empty_channels_masked_out": int((arrays.mask & arrays.moment_empty).sum()),
        "learned": head_quality(learned_cov, arrays.residual, mask, confidence=arrays.confidence),
        "heatmap_moment": head_quality(arrays.moment_cov, arrays.residual, mask),
        "chol": chol_stats(arrays.chol, mask, head_config, rho_saturation=rho_saturation),
        "min_eigenvalue_px2_all_labelled": float(eig.min()),
        "any_nonfinite": bool(not np.isfinite(arrays.chol).all()),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the overfit gate or the full training."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = p1.p1_paths(args.paths)
    run, seed = config["p1_run"], int(config["seed"])
    train_cfg = config["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    out_dir = args.out or Path(config["out_root"]) / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backends = _set_backends(bool(train_cfg["tf32"]))
    p1.seed_everything(seed)

    p1_amp = p1.load_p1_config(run)["runtime"]["amp_dtype"]
    if p1_amp != train_cfg["amp_dtype"]:
        raise SystemExit(f"train.amp_dtype {train_cfg['amp_dtype']} != P1's {p1_amp}")
    amp = AMP_DTYPES[train_cfg["amp_dtype"]]
    head_config = HeadConfig.from_dict(config["head"])
    model, _ = p1.load_keypoint_model(run, paths=paths)
    state_hash = p1_state_hash(model)
    net = VarianceKeypointNet(model, VarianceHead(head_config)).to(device)
    provenance = _provenance(args.config, run, paths)
    debug = args.limit_train is not None or args.limit_val is not None

    train_set = _subset(
        p1.keypoint_crop_dataset(
            run, train_cfg["split"], paths=paths, augmentation=train_cfg["augmentation"], seed=seed
        ),
        args.limit_train,
        seed,
    )

    if args.overfit_batch:
        batch_size = int(config["overfit"]["batch_size"])
        loader = p1.keypoint_loader(
            train_set, batch_size=batch_size, shuffle=True, workers=0, seed=seed, pin_memory=False
        )
        batch = next(iter(loader))
        smoke = overfit_one_batch(
            net,
            batch,
            device=device,
            amp=amp,
            beta=float(config["loss"]["beta"]),
            lr=float(config["overfit"]["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
            grad_clip=float(train_cfg["grad_clip"]),
            max_steps=args.max_steps,
        )
        smoke["p1_state_hash_unchanged"] = p1_state_hash(net.p1) == state_hash
        smoke["passed"] = bool(smoke["passed"] and smoke["p1_state_hash_unchanged"])
        (out_dir / "smoke.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
        if not debug:
            write_result_json(
                args.results_dir / "variance_head_smoke.json",
                {
                    "schema": RESULT_SCHEMA,
                    "kind": "variance_head",
                    "stage": "overfit_smoke",
                    "run_name": config["run_name"],
                    "p1_run": run,
                    "device": str(device),
                    "batch": {"split": "train", "augmentation": train_cfg["augmentation"]},
                    "batch_size": batch_size,
                    **smoke,
                    "provenance": provenance,
                },
            )
        verdict = "PASS" if smoke["passed"] else "FAIL"
        print(
            f"overfit gate {verdict}: final NLL {smoke['nll_final']:.4f} vs homoscedastic floor "
            f"{smoke['floor_homoscedastic_nll']:.4f} after {args.max_steps} steps"
        )
        return 0 if smoke["passed"] else 1

    if (out_dir / "best.pt").exists():
        raise SystemExit(f"{out_dir}/best.pt exists; choose another --out rather than overwrite")
    shutil.copyfile(args.config, out_dir / "config.yaml")
    (out_dir / "git_sha.txt").write_text(provenance["poseconf_git_sha"] + "\n", encoding="utf-8")
    (out_dir / "metrics.jsonl").unlink(missing_ok=True)

    val_set, val_counts = val_tune_dataset(run, paths=paths, seed=seed)
    val_set = _subset(val_set, args.limit_val, seed)
    sel = config["selection"]
    train_loader = p1.keypoint_loader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        workers=int(train_cfg["workers"]),
        seed=seed,
        pin_memory=bool(train_cfg["pin_memory"]),
    )
    val_loader = p1.keypoint_loader(
        val_set,
        batch_size=int(sel["batch_size"]),
        shuffle=False,
        workers=int(sel["workers"]),
        seed=seed,
        pin_memory=bool(train_cfg["pin_memory"]),
    )
    meta = {
        "head_config": config["head"],
        "config": config,
        "p1_run": run,
        "p1_checkpoint_sha256": provenance["p1_checkpoint_sha256"],
        "p1_state_hash": state_hash,
        "git_sha": provenance["poseconf_git_sha"],
        "seed": seed,
        "selection": "min val_tune plain Gaussian NLL",
    }
    print(
        f"training {config['run_name']} on {device}: {len(train_set)} train, "
        f"{len(val_set)} val_tune ({val_counts}); out {out_dir}",
        flush=True,
    )
    summary = train_head(
        net,
        train_loader,
        val_loader,
        config=config,
        head_config=head_config,
        device=device,
        out_dir=out_dir,
        checkpoint_meta=meta,
        log=lambda line: print(line, flush=True),
    )
    diagnostics = final_diagnostics(
        out_dir / "best.pt",
        val_loader,
        paths=paths,
        device=device,
        amp=amp,
        head_config=head_config,
        rho_saturation=float(config["diagnostics"]["rho_saturation"]),
    )
    (out_dir / "diagnostics_val_tune.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    if not debug:
        write_result_json(
            args.results_dir / "variance_head_training.json",
            {
                "schema": RESULT_SCHEMA,
                "kind": "variance_head",
                "stage": "training",
                "run_name": config["run_name"],
                "p1_run": run,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
                "backends": backends,
                "data": {
                    "train": {"split": "synthetic train", "n": len(train_set)},
                    "selection": {"split": "val_tune", **val_counts},
                },
                "units": {"sigma": "crop px", "nll": "nats per keypoint (2-D)"},
                **{key: value for key, value in summary.items() if key != "history"},
                "history": summary["history"],
                "diagnostics_val_tune": diagnostics,
                "best_checkpoint_sha256": sha256_file(out_dir / "best.pt"),
                "provenance": provenance,
            },
        )
    print(
        f"best epoch {summary['best_epoch']}: val_tune NLL {summary['best_val_tune_nll']:.4f}; "
        f"P1 hash unchanged: {summary['p1_state_hash_before'] == summary['p1_state_hash_after']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
