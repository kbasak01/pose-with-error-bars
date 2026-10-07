"""β-NLL training of the variance head on a frozen P1 trunk (IMPLEMENTATION_PLAN §1.6, Phase 5).

* Data: P1's `SpeedPlusCrop` on synthetic `train` with P1's A2 augmentation (through the adapter);
  selection on synthetic `val_tune` only (P1's eval crops, a0). `val_cal`, `val_test` and every HIL
  frame are never loaded here.
* Residuals: P1's own coordinates (stop-gradient by construction: the trunk runs under `no_grad`)
  minus the label, in crop px, masked exactly as P1's coordinate loss is (`batch["mask"]`,
  normalised by P1's `_masked_mean`). The trunk runs under P1's autocast dtype, as
  `PosePipeline.keypoints` runs it, so training residuals come from the inference numerics.
* Loss: 2-D β-NLL (Seitzer et al. 2022), `stopgrad(det Σ^{β/2}) · NLL`, the 2-D analogue of
  `stopgrad(σ^{2β})`; β = 0.5.
* Selection: the epoch with the lowest plain Gaussian NLL on `val_tune`.

The P1 state hash is taken before the first step and checked after the last; a change raises.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from poseconf import p1_adapter as p1
from poseconf.data.dump import heatmap_moments
from poseconf.engine.head_metrics import head_quality
from poseconf.models.variance_head import (
    HeadConfig,
    VarianceHead,
    VarianceKeypointNet,
    chol_to_cov,
    p1_state_hash,
)

__all__ = [
    "EvalArrays",
    "beta_nll",
    "chol_stats",
    "collect",
    "gaussian_nll",
    "homoscedastic_floor",
    "load_variance_net",
    "overfit_one_batch",
    "train_head",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)

#: Autocast dtypes by P1's `runtime.amp_dtype` name.
AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def gaussian_nll(chol: Tensor, residual: Tensor) -> Tensor:
    """Per-keypoint 2-D Gaussian NLL from the Cholesky factor.

    Args:
        chol: (..., 3) `(l0, l1, l2)`, `Σ = L Lᵀ`.
        residual: (..., 2) residuals, same units.

    Returns:
        (...) NLL in nats, `½(rᵀΣ⁻¹r + log det Σ) + log 2π`.
    """
    l0, l1, l2 = chol.unbind(-1)
    z0 = residual[..., 0] / l0
    z1 = (residual[..., 1] - l1 * z0) / l2
    log_det = 2.0 * (torch.log(l0) + torch.log(l2))
    return 0.5 * (z0 * z0 + z1 * z1 + log_det) + _LOG_TWO_PI


def beta_nll(chol: Tensor, residual: Tensor, beta: float) -> Tensor:
    """Per-keypoint 2-D β-NLL: `stopgrad(det Σ^{β/2}) · NLL`; β = 0 is the plain NLL.

    `det Σ^{1/2} = l0 · l2`, so the weight is `(l0 l2)^β`, the 2-D analogue of `σ^{2β}`.
    """
    nll = gaussian_nll(chol, residual)
    if beta == 0.0:
        return nll
    weight = (chol[..., 0] * chol[..., 2]).detach().pow(beta)
    return weight * nll


def homoscedastic_floor(residual: NDArray[np.float64]) -> float:
    """Mean NLL of the best single zero-mean Gaussian for these (n, 2) residuals (its MLE).

    The overfit gate's "analytic floor": with `S = rᵀr / n`, the mean NLL is
    `½(2 + log det S) + log 2π`. A per-keypoint head that has learned anything about the batch
    must fall below it.

    Raises:
        ValueError: With fewer than two residuals or a singular second moment.
    """
    r = np.asarray(residual, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 2 or len(r) < 2:
        raise ValueError("need (n >= 2, 2) residuals")
    second = r.T @ r / len(r)
    det = float(np.linalg.det(second))
    if det <= 0.0:
        raise ValueError("residual second moment is singular")
    return 0.5 * (2.0 + math.log(det)) + _LOG_TWO_PI


def chol_stats(
    chol: NDArray[np.float64], mask: NDArray[np.bool_], config: HeadConfig, *, rho_saturation: float
) -> dict[str, Any]:
    """σ distribution, clamp fractions and ρ saturation over the masked keypoints.

    Args:
        chol: (n, K, 3) Cholesky factors.
        mask: (n, K) keypoints that count.
        config: Head clamps.
        rho_saturation: |ρ| at or above which ρ counts as saturated.

    Returns:
        A JSON-ready mapping.
    """
    c = np.asarray(chol, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    sx = c[:, 0]
    sy = np.hypot(c[:, 1], c[:, 2])
    rho = c[:, 1] / sy
    both = np.concatenate([sx, sy])
    tol = 1.0e-4
    return {
        "n": int(len(c)),
        "sigma_px_quantiles": {
            q: float(np.quantile(both, float(q))) for q in ("0.01", "0.1", "0.5", "0.9", "0.99")
        },
        "frac_sigma_at_min": float(np.mean(both <= config.sigma_min * (1.0 + tol))),
        "frac_sigma_at_max": float(np.mean(both >= config.sigma_max * (1.0 - tol))),
        "frac_rho_saturated": float(np.mean(np.abs(rho) >= rho_saturation)),
        "rho_abs_median": float(np.median(np.abs(rho))),
        "sigma_px_std_across_keypoints": float(np.std(np.sqrt(sx * sy))),
    }


@dataclass(frozen=True)
class EvalArrays:
    """One pass over an evaluation loader (all numpy, float64 where numeric).

    Attributes:
        filenames: (n,) image filenames.
        chol: (n, K, 3) head output, crop px.
        residual: (n, K, 2) P1 coordinate − label, crop px.
        mask: (n, K) P1's loss mask (visible ∧ in crop).
        confidence: (n, K) P1 confidence.
        moment_cov: (n, K, 2, 2) heatmap-moment covariance (B2's Σ̂), crop px².
        moment_empty: (n, K) empty heatmap channel.
    """

    filenames: NDArray[np.str_]
    chol: NDArray[np.float64]
    residual: NDArray[np.float64]
    mask: NDArray[np.bool_]
    confidence: NDArray[np.float64]
    moment_cov: NDArray[np.float64]
    moment_empty: NDArray[np.bool_]


def _autocast(device: torch.device, dtype: torch.dtype) -> Any:
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda")


@torch.no_grad()
def collect(
    net: VarianceKeypointNet, loader: Iterable[dict[str, Any]], device: torch.device, amp: Any
) -> EvalArrays:
    """Run the net over a loader and keep everything the diagnostics need.

    Args:
        net: The wrapped network.
        loader: Batches from `SpeedPlusCrop`.
        device: Where to run.
        amp: Autocast dtype (P1's).

    Returns:
        The collected arrays.
    """
    net.eval()
    soft_argmax, stride = net.p1.head, float(net.p1.stride)
    parts: dict[str, list[Any]] = {key: [] for key in EvalArrays.__dataclass_fields__}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with _autocast(device, amp):
            coords, confidence, heatmaps, chol = net(images)
        moments = heatmap_moments(heatmaps, soft_argmax, stride)
        target = batch["keypoints"].to(device).float()
        parts["filenames"].extend(str(name) for name in batch["filename"])
        parts["chol"].append(chol.double().cpu().numpy())
        parts["residual"].append((coords.float() - target).double().cpu().numpy())
        parts["mask"].append(batch["mask"].numpy())
        parts["confidence"].append(confidence.float().double().cpu().numpy())
        parts["moment_cov"].append(moments.cov_crop.double().cpu().numpy())
        parts["moment_empty"].append(moments.empty.cpu().numpy())
    return EvalArrays(
        filenames=np.asarray(parts.pop("filenames")).astype(np.str_),
        **{key: np.concatenate(value) for key, value in parts.items()},
    )


def evaluate(
    arrays: EvalArrays, config: HeadConfig, *, beta: float, rho_saturation: float
) -> dict[str, Any]:
    """Selection metrics on one collected split (plain and β NLL, reliability, σ stats)."""
    mask = arrays.mask & ~arrays.moment_empty
    chol = torch.from_numpy(arrays.chol)
    residual = torch.from_numpy(arrays.residual)
    weights = torch.from_numpy(mask)
    plain = p1.masked_mean(gaussian_nll(chol, residual), weights)
    weighted = p1.masked_mean(beta_nll(chol, residual, beta), weights)
    cov = chol_to_cov(chol).numpy()
    quality = head_quality(cov, arrays.residual, mask, confidence=arrays.confidence)
    return {
        "nll": float(plain),
        "beta_nll": float(weighted),
        "coverage_1sigma": quality["coverage_1sigma"],
        "coverage_2sigma": quality["coverage_2sigma"],
        "spearman_sigma_vs_error": quality["spearman_sigma_vs_error"],
        "chol": chol_stats(arrays.chol, mask, config, rho_saturation=rho_saturation),
    }


def _gpu_utilisation() -> float | None:
    """Instantaneous GPU utilisation (%) from `nvidia-smi`, or None when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return float(out.stdout.split()[0])


def _step(
    net: VarianceKeypointNet,
    batch: dict[str, Any],
    optimiser: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp: Any,
    beta: float,
    grad_clip: float,
) -> tuple[float, float]:
    """One optimiser step on one batch; returns `(β-NLL, plain NLL)` (masked means)."""
    images = batch["image"].to(device, non_blocking=True)
    target = batch["keypoints"].to(device, non_blocking=True).float()
    mask = batch["mask"].to(device, non_blocking=True)
    with _autocast(device, amp):
        coords, _, _, chol = net(images)
    residual = coords.float() - target
    loss = p1.masked_mean(beta_nll(chol, residual, beta), mask)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.head.parameters(), grad_clip)
    optimiser.step()
    with torch.no_grad():
        plain = p1.masked_mean(gaussian_nll(chol.detach(), residual), mask)
    return float(loss.detach()), float(plain)


def _optimiser(head: VarianceHead, train_cfg: dict[str, Any], lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        head.parameters(), lr=lr, weight_decay=float(train_cfg["weight_decay"])
    )


def overfit_one_batch(
    net: VarianceKeypointNet,
    batch: dict[str, Any],
    *,
    device: torch.device,
    amp: Any,
    beta: float,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    max_steps: int,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The overfit gate: β-NLL steps on one fixed batch; plain NLL must fall below the floor.

    Args:
        net: The wrapped network (head freshly initialised).
        batch: One fixed training batch.
        device: Where to run.
        amp: Autocast dtype.
        beta: β of the loss.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Gradient-norm clip.
        max_steps: Steps to run.
        log: Progress sink.

    Returns:
        `{floor, nll_start, nll_final, nll_min, steps, passed, curve}`.
    """
    net.train()
    with torch.no_grad(), _autocast(device, amp):
        coords, _, _, _ = net(batch["image"].to(device))
    residual = (coords.float().cpu() - batch["keypoints"].float()).double().numpy()
    mask = batch["mask"].numpy()
    floor = homoscedastic_floor(residual[mask])
    optimiser = torch.optim.AdamW(net.head.parameters(), lr=lr, weight_decay=weight_decay)
    curve = []
    for step in range(1, max_steps + 1):
        _, plain = _step(
            net, batch, optimiser, device=device, amp=amp, beta=beta, grad_clip=grad_clip
        )
        curve.append(plain)
        if step % 50 == 0 or step == 1:
            log(f"  overfit step {step}: nll {plain:.4f} (floor {floor:.4f})")
    net.eval()
    with torch.no_grad(), _autocast(device, amp):
        coords, _, _, chol = net(batch["image"].to(device))
    final = float(
        p1.masked_mean(
            gaussian_nll(chol.float().cpu(), coords.float().cpu() - batch["keypoints"].float()),
            batch["mask"],
        )
    )
    return {
        "floor_homoscedastic_nll": floor,
        "n_keypoints": int(mask.sum()),
        "nll_start": curve[0],
        "nll_final": final,
        "nll_min_during_training": float(min(curve)),
        "steps": max_steps,
        "passed": bool(final < floor),
        "curve_every_10": curve[::10],
    }


def train_head(
    net: VarianceKeypointNet,
    train_loader: Any,
    val_loader: Any,
    *,
    config: dict[str, Any],
    head_config: HeadConfig,
    device: torch.device,
    out_dir: Path,
    checkpoint_meta: dict[str, Any],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Full training with per-epoch `val_tune` selection.

    Writes `metrics.jsonl` (one record per epoch), `last.pt` every epoch and `best.pt` when the
    `val_tune` plain NLL improves.

    Args:
        net: The wrapped network.
        train_loader: Shuffled synthetic `train` loader (A2).
        val_loader: `val_tune` loader (a0).
        config: The variance-head config.
        head_config: Head clamps.
        device: Where to run.
        out_dir: Run directory.
        checkpoint_meta: Provenance stored in every checkpoint.
        log: Progress sink.

    Returns:
        Summary: best epoch and NLL, per-epoch history, hash check, timings.

    Raises:
        RuntimeError: If the P1 state hash changed during training.
    """
    train_cfg, loss_cfg, diag_cfg = config["train"], config["loss"], config["diagnostics"]
    amp = AMP_DTYPES[train_cfg["amp_dtype"]]
    beta = float(loss_cfg["beta"])
    epochs = int(train_cfg["epochs"])
    hash_before = p1_state_hash(net.p1)
    optimiser = _optimiser(net.head, train_cfg, float(train_cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimiser, gamma=float(train_cfg["lr_gamma"])
    )
    util_every = int(diag_cfg["gpu_util_every_steps"])
    best_nll, best_epoch, history = math.inf, -1, []
    started_all = time.perf_counter()

    for epoch in range(1, epochs + 1):
        net.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        wait = 0.0
        sums = np.zeros(2)
        steps = 0
        utils: list[float] = []
        tick = time.perf_counter()
        for batch in train_loader:
            wait += time.perf_counter() - tick
            weighted, plain = _step(
                net,
                batch,
                optimiser,
                device=device,
                amp=amp,
                beta=beta,
                grad_clip=float(train_cfg["grad_clip"]),
            )
            if not (math.isfinite(weighted) and math.isfinite(plain)):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {steps}")
            sums += (weighted, plain)
            steps += 1
            if steps % util_every == 0:
                value = _gpu_utilisation()
                if value is not None:
                    utils.append(value)
            tick = time.perf_counter()
        train_s = time.perf_counter() - started
        lr = optimiser.param_groups[0]["lr"]
        scheduler.step()

        val_started = time.perf_counter()
        arrays = collect(net, val_loader, device, amp)
        val = evaluate(
            arrays, head_config, beta=beta, rho_saturation=float(diag_cfg["rho_saturation"])
        )
        record = {
            "epoch": epoch,
            "lr": lr,
            "train_beta_nll": sums[0] / steps,
            "train_nll": sums[1] / steps,
            "train_steps": steps,
            "val_tune": val,
            "epoch_train_s": round(train_s, 2),
            "epoch_val_s": round(time.perf_counter() - val_started, 2),
            "data_wait_s": round(wait, 2),
            "train_images_per_s": round(steps * int(train_cfg["batch_size"]) / train_s, 1),
            "gpu_util_mean_pct": float(np.mean(utils)) if utils else None,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        }
        improved = val["nll"] < best_nll
        if improved:
            best_nll, best_epoch = val["nll"], epoch
        record["best_epoch_so_far"] = best_epoch
        history.append(record)
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        state = {
            **checkpoint_meta,
            "head": net.head.state_dict(),
            "epoch": epoch,
            "val_tune_nll": val["nll"],
        }
        torch.save(state, out_dir / "last.pt")
        if improved:
            torch.save(state, out_dir / "best.pt")
        log(
            f"epoch {epoch}/{epochs}: train β-NLL {record['train_beta_nll']:.4f} "
            f"NLL {record['train_nll']:.4f} | val_tune NLL {val['nll']:.4f} "
            f"1σ {val['coverage_1sigma']:.3f} 2σ {val['coverage_2sigma']:.3f} | "
            f"{train_s:.0f}s {'*' if improved else ''}"
        )

    hash_after = p1_state_hash(net.p1)
    if hash_after != hash_before:
        raise RuntimeError("P1 state hash changed during training: the trunk is not frozen")
    return {
        "best_epoch": best_epoch,
        "best_val_tune_nll": best_nll,
        "epochs": epochs,
        "p1_state_hash_before": hash_before,
        "p1_state_hash_after": hash_after,
        "wall_clock_s": round(time.perf_counter() - started_all, 1),
        "history": history,
    }


def load_variance_net(
    checkpoint: Path, *, paths: p1.P1Paths, device: torch.device
) -> tuple[VarianceKeypointNet, dict[str, Any]]:
    """Rebuild the selected network, refusing a checkpoint trained on a different P1 trunk.

    Args:
        checkpoint: A `best.pt` written by `train_head`.
        paths: Local paths (the P1 checkpoint is read from `p1_runs`).
        device: Where to put it.

    Returns:
        `(net in eval mode, checkpoint dict without the weights)`.

    Raises:
        ValueError: If the P1 checkpoint's SHA-256 or the loaded trunk's state hash differs from
            the ones recorded at training time.
    """
    from poseconf.provenance import sha256_file

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run = state["p1_run"]
    actual_sha = sha256_file(paths.p1_runs / run / "best.pt")
    if actual_sha != state["p1_checkpoint_sha256"]:
        raise ValueError(
            f"{checkpoint} was trained on P1 {run} sha256 {state['p1_checkpoint_sha256']}, "
            f"but {paths.p1_runs / run / 'best.pt'} has {actual_sha}"
        )
    model, _ = p1.load_keypoint_model(run, paths=paths)
    if p1_state_hash(model) != state["p1_state_hash"]:
        raise ValueError("loaded P1 trunk's state hash differs from the one trained on")
    head = VarianceHead(HeadConfig.from_dict(state["head_config"]))
    head.load_state_dict(state["head"])
    net = VarianceKeypointNet(model, head).to(device).eval()
    meta = {key: value for key, value in state.items() if key != "head"}
    return net, meta
