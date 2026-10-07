"""Per-frame prediction dump (`dumps/<run>/<domain>_<arm>.npz`), Phase 3.

The dump runs P1's own pipeline stages in the order P1's evaluation runs them
(`speedpose.engine.evaluate._evaluate_through_pipeline`), so a frame cannot be solved differently
from P1:

1. `prepare_frame` -> `detect_boxes`, batches of `train.batch_size` (predicted_crop only);
2. `crop_one` -> `normalise_crop` -> `pipeline.keypoints()` in the same batches, observed through
   `p1_adapter.keypoint_hooks` (heatmaps, pooled encoder feature, decoder features);
3. one PnP pass over all frames with `p1_adapter.solve_frames` (P1's `solve_single`, P1's chunk
   seeding).

Nothing here re-implements a crop, a decode or a solve. The only computation added is
`heatmap_moments`, the spread of the distribution P1's soft-argmax refines over; its mean is
checked against P1's coordinate on every frame, so the moment is centred on P1's point.

Files (see docs/DECISIONS.md, 2026-10-07 Phase 3 entries):

* `<domain>_<arm>.npz` — predictions, label-free (except the `gt_crop` arm's input box, which is
  the label box: that arm is an oracle on HIL and is tagged `oracle_gt_box`).
* `<domain>_<arm>_subset.npz` — raw heatmaps and decoder features for a fixed 256-frame subset.
* `synthetic_labels.npz`, `<hil>_labels_poolA.npz`, `<hil>_labels_poolB.npz` — ground truth,
  physically split so that poolA labels are opened only by `oracle_*` callers.

PnP failures are never dropped: every frame has a row, failures carry NaN poses and a reason.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from poseconf import p1_adapter as p1

__all__ = [
    "CROP_ARMS",
    "DEGENERATE_BOX_REASON",
    "DumpOutput",
    "HeatmapMoments",
    "arm_tag",
    "crop_cov_to_full",
    "dump_file",
    "heatmap_moments",
    "labels_file",
    "load_dump",
    "load_dump_labels",
    "p1_reference_arm",
    "run_dump",
    "run_variance_dump",
    "variance_dump_file",
    "subset_rows",
    "write_dump",
    "write_labels",
]

#: Crop arms the dump runs. `gt_crop` is P1's `pipeline_gt_crop` code path (GT box through the
#: pipeline), not P1's crop-cache `gt_crop` arm.
CROP_ARMS = ("predicted_crop", "gt_crop")

#: P1 sidecar arm each dump arm reproduces.
_P1_REFERENCE_ARM = {"predicted_crop": "predicted_crop", "gt_crop": "pipeline_gt_crop"}

#: P1's failure reason for a box `crop_one` refuses (`_evaluate_through_pipeline`).
DEGENERATE_BOX_REASON = "detector box was degenerate"

#: Label pools per domain: synthetic has one file, HIL domains one per committed pool.
_LABEL_POOLS = {
    "synthetic": ("all",),
    "lightbox": ("poolA", "poolB"),
    "sunlamp": ("poolA", "poolB"),
}


def p1_reference_arm(crop_source: str) -> str:
    """The P1 sidecar arm a dump arm must reproduce.

    Raises:
        ValueError: On an unknown arm.
    """
    if crop_source not in _P1_REFERENCE_ARM:
        raise ValueError(f"unknown dump arm {crop_source!r}; expected one of {CROP_ARMS}")
    return _P1_REFERENCE_ARM[crop_source]


def arm_tag(domain: str, crop_source: str) -> str:
    """The arm tag a dump carries: GT-box crops on a HIL domain are an oracle.

    Args:
        domain: One of `p1_adapter.DOMAINS`.
        crop_source: One of `CROP_ARMS`.

    Returns:
        `oracle_gt_box` for HIL `gt_crop`, else the crop source itself.
    """
    p1_reference_arm(crop_source)
    if crop_source == "gt_crop" and domain in p1.HIL_DOMAINS:
        return "oracle_gt_box"
    return crop_source


def dump_file(dumps_root: Path, run: str, domain: str, crop_source: str, *, subset: bool) -> Path:
    """Path of a prediction (or subset) dump."""
    suffix = "_subset" if subset else ""
    return Path(dumps_root) / run / f"{domain}_{crop_source}{suffix}.npz"


def variance_dump_file(dumps_root: Path, run: str, domain: str, crop_source: str) -> Path:
    """Path of the Phase 5 variance-head sidecar of a prediction dump."""
    return Path(dumps_root) / run / f"{domain}_{crop_source}_vhead.npz"


def labels_file(dumps_root: Path, run: str, domain: str, pool: str) -> Path:
    """Path of a label file; `pool` is `all` for synthetic, `poolA`/`poolB` for HIL.

    Raises:
        ValueError: On a pool the domain does not have.
    """
    if pool not in _LABEL_POOLS.get(domain, ()):
        raise ValueError(f"{domain} has label pools {_LABEL_POOLS.get(domain)}, not {pool!r}")
    stem = f"{domain}_labels" if pool == "all" else f"{domain}_labels_{pool}"
    return Path(dumps_root) / run / f"{stem}.npz"


def subset_rows(n: int, size: int, seed: int) -> NDArray[np.int64]:
    """The fixed heatmap/decoder-feature subset: `size` rows drawn without replacement, sorted.

    Depends only on `(n, size, seed)`, so both arms of a domain store the same frames.
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(size, n), replace=False)).astype(np.int64)


@dataclass(frozen=True)
class HeatmapMoments:
    """Moments of the soft-argmax refine distribution, per keypoint.

    Attributes:
        mean_crop: (B, K, 2) mean in crop px; equals P1's coordinate by construction.
        cov_crop: (B, K, 2, 2) covariance in crop px².
        entropy: (B, K) Shannon entropy (nats) of the normalised rectified heatmap.
        peak: (B, K) raw heatmap maximum (P1's confidence).
        empty: (B, K) the refine weights are all zero (P1 decodes such a channel to the origin;
            its covariance is then 0).
    """

    mean_crop: torch.Tensor
    cov_crop: torch.Tensor
    entropy: torch.Tensor
    peak: torch.Tensor
    empty: torch.Tensor


def heatmap_moments(heatmaps: torch.Tensor, head: Any, stride: float) -> HeatmapMoments:
    """Mean and covariance of the distribution P1's two-pass soft-argmax refines over.

    The locate pass and the window are built exactly as `SoftArgmax2d.forward` builds them, and
    both centroids go through the head's own `_centroid`, so `mean_crop` is P1's coordinate. The
    covariance is the second central moment of the same normalised weights,
    `relu(h)^refine * window / sum`. All in fp32, as P1's head computes.

    Args:
        heatmaps: (B, K, H, W) raw heatmaps (the network's third output).
        head: The network's `SoftArgmax2d` (`locate_exponent`, `refine_exponent`,
            `window_sigma`, `_centroid`).
        stride: Crop px per heatmap px (`KeypointNet.stride`).

    Returns:
        The moments, on the heatmaps' device.
    """
    maps = heatmaps.float()
    height, width = maps.shape[-2:]
    xs = torch.arange(width, dtype=maps.dtype, device=maps.device)
    ys = torch.arange(height, dtype=maps.dtype, device=maps.device)
    rectified = torch.relu(maps)

    locate = head._centroid(rectified.pow(head.locate_exponent), xs, ys)
    two_sigma_squared = 2.0 * head.window_sigma * head.window_sigma
    offset_x = xs.view(1, 1, 1, width) - locate[..., 0].unsqueeze(-1).unsqueeze(-1)
    offset_y = ys.view(1, 1, height, 1) - locate[..., 1].unsqueeze(-1).unsqueeze(-1)
    window = torch.exp(-(offset_x * offset_x + offset_y * offset_y) / two_sigma_squared)
    weights = rectified.pow(head.refine_exponent) * window
    mean = head._centroid(weights, xs, ys)

    tiny = torch.finfo(weights.dtype).tiny
    mass = weights.sum(dim=(-2, -1), keepdim=True)
    probability = weights / mass.clamp_min(tiny)
    dx = xs.view(1, 1, 1, width) - mean[..., 0].unsqueeze(-1).unsqueeze(-1)
    dy = ys.view(1, 1, height, 1) - mean[..., 1].unsqueeze(-1).unsqueeze(-1)
    cxx = (probability * dx * dx).sum(dim=(-2, -1))
    cyy = (probability * dy * dy).sum(dim=(-2, -1))
    cxy = (probability * dx * dy).sum(dim=(-2, -1))
    cov = torch.stack((torch.stack((cxx, cxy), -1), torch.stack((cxy, cyy), -1)), -2)

    rectified_mass = rectified.sum(dim=(-2, -1), keepdim=True).clamp_min(tiny)
    spread = rectified / rectified_mass
    entropy = -torch.special.xlogy(spread, spread).sum(dim=(-2, -1))

    return HeatmapMoments(
        mean_crop=(mean + 0.5) * stride - 0.5,
        cov_crop=cov * (stride * stride),
        entropy=entropy,
        peak=maps.amax(dim=(-2, -1)),
        empty=mass.squeeze(-1).squeeze(-1) == 0,
    )


def crop_cov_to_full(
    cov_crop: NDArray[np.float64], affines: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map (n, K, 2, 2) crop-px covariances to full-frame px²: `L Σ Lᵀ`.

    `L` is the linear part of P1's inverse affine (crop -> full frame), the same inversion that
    maps the keypoints, so ellipses and points live in one frame.

    Args:
        cov_crop: (n, K, 2, 2) covariances in crop px².
        affines: (n, 2, 3) full-frame-to-crop affines.

    Returns:
        (n, K, 2, 2) covariances in full-frame px².
    """
    linear = np.stack([p1.invert_affine(affine)[:, :2] for affine in affines])
    return np.einsum("nij,nkjl,nml->nkim", linear, cov_crop, linear)


@dataclass(frozen=True)
class DumpOutput:
    """What one dump pass produced.

    Attributes:
        fields: Per-frame arrays for `<domain>_<arm>.npz`.
        subset: Arrays for `<domain>_<arm>_subset.npz`.
        info: Run facts for the meta record (chunks, timings, moment check).
    """

    fields: dict[str, NDArray[Any]]
    subset: dict[str, NDArray[Any]]
    info: dict[str, Any]


def _read_grayscale(path: str) -> NDArray[np.uint8]:
    """Decode one frame as P1's evaluation does.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read source image: {path}")
    return image


def _image_batches(
    paths: Sequence[str], batch_size: int, pool: ThreadPoolExecutor
) -> Iterator[tuple[int, list[NDArray[np.uint8]]]]:
    """Yield `(start, images)` in order, decoding the next batch while this one is consumed."""
    pending: tuple[int, list[Future]] | None = None
    for start in range(0, len(paths), batch_size):
        futures = [pool.submit(_read_grayscale, path) for path in paths[start : start + batch_size]]
        if pending is not None:
            yield pending[0], [future.result() for future in pending[1]]
        pending = (start, futures)
    if pending is not None:
        yield pending[0], [future.result() for future in pending[1]]


def run_dump(
    pipeline: Any,
    frames: p1.EvalFrames,
    *,
    crop_source: str,
    boxes_gt: NDArray[np.float64] | None,
    batch_size: int,
    subset: NDArray[np.int64],
    solve_workers: int,
    solve_seed: int,
    decode_threads: int,
    moment_tol_crop_px: float,
    log: Callable[[str], None] = print,
    log_every: int = 50,
) -> DumpOutput:
    """Run P1's stages over one domain and collect everything Level B/C needs.

    Args:
        pipeline: P1 `PosePipeline` (with a detector for predicted_crop, without for gt_crop).
        frames: The domain's evaluation frames, in P1's order.
        crop_source: One of `CROP_ARMS`.
        boxes_gt: (n, 4) tight GT boxes for `gt_crop`; must be None for `predicted_crop`.
        batch_size: P1's `train.batch_size` (both network stages batch by it in P1's evaluation).
        subset: Sorted rows whose heatmaps and decoder features are kept.
        solve_workers: P1's `runtime.num_workers` (sets P1's PnP chunking).
        solve_seed: P1's `runtime.seed`.
        decode_threads: Threads decoding JPEGs ahead of the network.
        moment_tol_crop_px: Largest allowed |moment mean - P1 coordinate| on non-empty channels.
        log: Progress sink.
        log_every: Batches between progress lines.

    Returns:
        The dump arrays and run facts.

    Raises:
        ValueError: On an arm/box mismatch.
        RuntimeError: If a moment mean departs from P1's coordinate by more than the tolerance.
    """
    n = len(frames)
    if crop_source == "predicted_crop":
        if boxes_gt is not None:
            raise ValueError("predicted_crop takes its boxes from the detector; pass boxes_gt=None")
    elif crop_source == "gt_crop":
        if boxes_gt is None or len(boxes_gt) != n:
            raise ValueError("gt_crop needs one GT box per frame")
    else:
        raise ValueError(f"unknown crop_source {crop_source!r}; expected one of {CROP_ARMS}")

    model = pipeline.keypoint_model
    num_keypoints = int(model.num_keypoints)
    crop_size = int(pipeline.crop_size)
    in_subset = np.zeros(n, dtype=bool)
    in_subset[subset] = True
    subset_position = {int(row): index for index, row in enumerate(subset)}
    timings: dict[str, float] = {}
    pipeline.seed_rng()

    with ThreadPoolExecutor(max_workers=decode_threads) as pool:
        started = time.perf_counter()
        if crop_source == "predicted_crop":
            boxes = np.empty((n, 4), dtype=np.float64)
            for batch_index, (start, images) in enumerate(
                _image_batches(frames.image_paths, batch_size, pool)
            ):
                prepared = torch.stack([pipeline.prepare_frame(image) for image in images])
                boxes[start : start + len(images)] = pipeline.detect_boxes(prepared)
                if batch_index % log_every == 0:
                    log(f"  detect {start + len(images)}/{n}")
        else:
            boxes = np.asarray(boxes_gt, dtype=np.float64).copy()
        timings["detect_s"] = time.perf_counter() - started

        started = time.perf_counter()
        coords = np.empty((n, num_keypoints, 2))
        confidence = np.empty((n, num_keypoints))
        affines = np.empty((n, 2, 3))
        croppable = np.ones(n, dtype=bool)
        cov_crop = np.empty((n, num_keypoints, 2, 2))
        entropy = np.empty((n, num_keypoints), dtype=np.float32)
        peak = np.empty((n, num_keypoints), dtype=np.float32)
        empty = np.empty((n, num_keypoints), dtype=bool)
        enc_feat: NDArray[np.float16] | None = None
        sub_heatmaps: NDArray[np.float16] | None = None
        sub_decoder: NDArray[np.float16] | None = None
        worst_moment_dev = 0.0
        placeholder = np.zeros((crop_size, crop_size), dtype=np.uint8)

        with p1.keypoint_hooks(pipeline) as capture:
            for batch_index, (start, images) in enumerate(
                _image_batches(frames.image_paths, batch_size, pool)
            ):
                stop = start + len(images)
                tensors = []
                for offset, image in enumerate(images):
                    try:
                        crop, affine = pipeline.crop_one(image, boxes[start + offset])
                    except ValueError:
                        # P1 carries an uncroppable box through as a blank crop with an identity
                        # affine, solves it, and then charges it as a failure. Same here.
                        crop, affine = placeholder, np.eye(2, 3, dtype=np.float64)
                        croppable[start + offset] = False
                    affines[start + offset] = affine
                    tensors.append(pipeline.normalise_crop(crop))
                rows = np.arange(start, stop)
                capture.keep_decoder = bool(in_subset[rows].any())
                batch_coords, batch_conf = pipeline.keypoints(torch.stack(tensors))
                coords[start:stop], confidence[start:stop] = batch_coords, batch_conf

                moments = heatmap_moments(capture.heatmaps, model.head, float(model.stride))
                mean = moments.mean_crop.cpu().double().numpy()
                hit = ~moments.empty.cpu().numpy()
                if hit.any():
                    deviation = float(np.abs(mean - batch_coords)[hit].max())
                    worst_moment_dev = max(worst_moment_dev, deviation)
                    if deviation > moment_tol_crop_px:
                        raise RuntimeError(
                            f"heatmap moment mean departs from P1's coordinate by {deviation:.3g} "
                            f"crop px (> {moment_tol_crop_px}) in frames {start}..{stop - 1}"
                        )
                cov_crop[start:stop] = moments.cov_crop.cpu().double().numpy()
                entropy[start:stop] = moments.entropy.cpu().numpy()
                peak[start:stop] = moments.peak.cpu().numpy()
                empty[start:stop] = moments.empty.cpu().numpy()

                pooled = capture.encoder_pooled.cpu().numpy()
                if enc_feat is None:
                    enc_feat = np.empty((n, pooled.shape[1]), dtype=np.float16)
                enc_feat[start:stop] = pooled.astype(np.float16)

                keep = rows[in_subset[rows]]
                if keep.size:
                    local = torch.as_tensor(keep - start, device=capture.heatmaps.device)
                    maps = capture.heatmaps[local].float().cpu().numpy().astype(np.float16)
                    decoder = capture.decoder[local].float().cpu().numpy().astype(np.float16)
                    if sub_heatmaps is None:
                        sub_heatmaps = np.empty((len(subset), *maps.shape[1:]), dtype=np.float16)
                        sub_decoder = np.empty((len(subset), *decoder.shape[1:]), dtype=np.float16)
                    for local_index, row in enumerate(keep):
                        sub_heatmaps[subset_position[int(row)]] = maps[local_index]
                        sub_decoder[subset_position[int(row)]] = decoder[local_index]
                if batch_index % log_every == 0:
                    log(f"  keypoints {stop}/{n}")
        timings["keypoints_s"] = time.perf_counter() - started

    started = time.perf_counter()
    solved = p1.solve_frames(
        pipeline, coords, confidence, affines, workers=solve_workers, seed=solve_seed
    )
    timings["solve_s"] = time.perf_counter() - started

    success = solved.success.copy()
    q_pred, t_pred = solved.q.copy(), solved.t.copy()
    n_inliers = solved.n_inliers.copy()
    rmse = solved.reprojection_rmse.copy()
    reasons = solved.failure_reason.astype(object)
    bad = ~croppable
    success[bad], q_pred[bad], t_pred[bad] = False, np.nan, np.nan
    n_inliers[bad], rmse[bad], reasons[bad] = 0, np.nan, DEGENERATE_BOX_REASON

    kp_full = np.stack([p1.crop_to_full(coords[i], affines[i]) for i in range(n)])
    cov_full = crop_cov_to_full(cov_crop, affines)

    fields = {
        "filename": np.asarray(frames.filenames).astype(np.str_),
        "bbox_used": boxes,
        "affine": affines,
        "croppable": croppable,
        "kp_pred_crop": coords,
        "kp_pred_full": kp_full,
        "confidence": confidence,
        "heatmap_cov_crop": cov_crop,
        "heatmap_cov_full": cov_full,
        "heatmap_peak": peak,
        "heatmap_entropy": entropy,
        "heatmap_empty": empty,
        "enc_feat": enc_feat,
        "success": success,
        "failure_reason": reasons.astype(np.str_),
        "q_pred": q_pred,
        "t_pred": t_pred,
        "n_inliers": n_inliers,
        "reprojection_rmse": rmse,
        "subset_mask": in_subset,
    }
    subset_fields = {
        "filename": fields["filename"][subset],
        "rows": subset,
        "heatmaps": sub_heatmaps,
        "decoder_feat": sub_decoder,
    }
    info = {
        "n_frames": n,
        "n_solved": int(success.sum()),
        "n_uncroppable": int(bad.sum()),
        "max_moment_mean_dev_crop_px": worst_moment_dev,
        "n_empty_heatmap_channels": int(empty.sum()),
        "solve_chunks": [list(chunk) for chunk in solved.chunks],
        "timings_s": {key: round(value, 2) for key, value in timings.items()},
    }
    return DumpOutput(fields=fields, subset=subset_fields, info=info)


def run_variance_dump(
    pipeline: Any,
    frames: p1.EvalFrames,
    head: torch.nn.Module,
    dump: dict[str, NDArray[Any]],
    *,
    batch_size: int,
    decode_threads: int,
    log: Callable[[str], None] = print,
    log_every: int = 50,
) -> dict[str, NDArray[Any]]:
    """Phase 5 sidecar: the variance head's covariance for every frame of an existing dump.

    Re-runs only the keypoint stage, on the dump's own boxes, in the dump's batch order and size,
    with the head reading P1's decoder features and heatmaps through `keypoint_hooks` under the
    pipeline's autocast. P1's path is untouched; the affine and coordinates must equal the
    Phase 3 dump bit for bit, otherwise this raises (the covariance would describe another point).

    Args:
        pipeline: P1 `PosePipeline` for the dump's arm.
        frames: The domain's evaluation frames, in the dump's order.
        head: A trained `VarianceHead` (on the pipeline's device, eval mode).
        dump: The Phase 3 prediction-dump arrays for this domain x arm.
        batch_size: The dump's batch size (P1's `train.batch_size`).
        decode_threads: Threads decoding JPEGs ahead of the network.
        log: Progress sink.
        log_every: Batches between progress lines.

    Returns:
        `filename, kp_pred_crop, vhead_chol_crop (n, K, 3), vhead_cov_crop, vhead_cov_full`.

    Raises:
        RuntimeError: If an affine or coordinate differs from the dump.
    """
    from poseconf.models.variance_head import chol_to_cov

    n = len(frames)
    if list(np.asarray(dump["filename"]).astype(str)) != [str(f) for f in frames.filenames]:
        raise RuntimeError("frames are not in the dump's order")
    model = pipeline.keypoint_model
    crop_size = int(pipeline.crop_size)
    boxes = np.asarray(dump["bbox_used"], dtype=np.float64)
    croppable = np.asarray(dump["croppable"], dtype=bool)
    placeholder = np.zeros((crop_size, crop_size), dtype=np.uint8)
    coords = np.empty((n, int(model.num_keypoints), 2))
    chol = np.empty((n, int(model.num_keypoints), 3))
    device = pipeline.device
    with (
        ThreadPoolExecutor(max_workers=decode_threads) as pool,
        p1.keypoint_hooks(pipeline, keep_decoder=True) as capture,
    ):
        for batch_index, (start, images) in enumerate(
            _image_batches(frames.image_paths, batch_size, pool)
        ):
            stop = start + len(images)
            tensors = []
            for offset, image in enumerate(images):
                row = start + offset
                if croppable[row]:
                    crop, affine = pipeline.crop_one(image, boxes[row])
                else:
                    crop, affine = placeholder, np.eye(2, 3, dtype=np.float64)
                if not np.array_equal(affine, dump["affine"][row]):
                    raise RuntimeError(f"frame {row}: affine differs from the dump")
                tensors.append(pipeline.normalise_crop(crop))
            batch_coords, _ = pipeline.keypoints(torch.stack(tensors))
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=pipeline.amp_dtype,
                    enabled=device.type == "cuda",
                ),
            ):
                batch_chol = head(capture.decoder, capture.heatmaps, model.head)
            coords[start:stop] = batch_coords
            chol[start:stop] = batch_chol.double().cpu().numpy()
            if batch_index % log_every == 0:
                log(f"  vhead {stop}/{n}")
    expected = np.asarray(dump["kp_pred_crop"], dtype=np.float64)
    if not np.array_equal(coords, expected):
        bad = int(np.flatnonzero((coords != expected).any(axis=(1, 2)))[0])
        raise RuntimeError(f"frame {bad}: P1 coordinates differ from the Phase 3 dump")
    cov_crop = chol_to_cov(torch.from_numpy(chol)).numpy()
    return {
        "filename": np.asarray(frames.filenames).astype(np.str_),
        "kp_pred_crop": coords,
        "vhead_chol_crop": chol,
        "vhead_cov_crop": cov_crop,
        "vhead_cov_full": crop_cov_to_full(cov_crop, np.asarray(dump["affine"], dtype=np.float64)),
    }


def write_dump(
    path: Path, arrays: dict[str, NDArray[Any]], meta: dict[str, Any], *, compress: bool
) -> Path:
    """Write a dump npz with a JSON `meta` record.

    Raises:
        ValueError: On a missing array or ragged leading dimensions.
    """
    missing = [key for key, value in arrays.items() if value is None]
    if missing:
        raise ValueError(f"dump arrays {missing} were never filled")
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"ragged dump arrays: {lengths}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    save(path, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    return path


def load_dump(path: Path) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Read a prediction or subset dump.

    Returns:
        `(arrays, meta)`.

    Raises:
        FileNotFoundError: If the dump is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no dump {path}; run scripts/dump_predictions.py first")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "meta"}
        meta = json.loads(str(archive["meta"]))
    return arrays, meta


def write_labels(path: Path, labels: p1.EvalLabels, meta: dict[str, Any]) -> Path:
    """Write one label file (`filename, q_gt, t_gt, kp_gt_full, in_frame, bbox_gt`)."""
    return write_dump(
        path,
        {
            "filename": labels.filenames,
            "q_gt": labels.q,
            "t_gt": labels.t,
            "kp_gt_full": labels.keypoints_2d,
            "in_frame": labels.in_frame,
            "bbox_gt": labels.bbox_tight,
        },
        meta,
        compress=True,
    )


def load_dump_labels(
    dumps_root: Path, run: str, domain: str, pool: str, *, tag: str | None
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Read a label file, refusing HIL poolA without an `oracle_*` tag (CLAUDE.md invariant 4).

    Args:
        dumps_root: The dumps root.
        run: P1 run.
        domain: One of `p1_adapter.DOMAINS`.
        pool: `all` (synthetic), `poolA` or `poolB` (HIL).
        tag: The calling arm's tag.

    Returns:
        `(arrays, meta)`.

    Raises:
        p1_adapter.HILLabelAccessError: On HIL poolA without an `oracle_*` tag.
    """
    path = labels_file(dumps_root, run, domain, pool)
    if domain in p1.HIL_DOMAINS and pool == "poolA" and not (tag or "").startswith("oracle_"):
        raise p1.HILLabelAccessError(
            f"{path.name}: HIL poolA labels are readable only by an oracle_* arm (tag={tag!r})"
        )
    return load_dump(path)
