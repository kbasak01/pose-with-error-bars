"""Figures regenerated from committed results (+ the gitignored dumps and the read-only dataset).

    python scripts/make_figures.py --figure all
    python scripts/make_figures.py --figure level_b_overlay

`all` skips a figure whose inputs are absent (with a message); a figure asked for by name raises.

Figures:
* `level_b_overlay` -> `assets/level_b_overlay_<score>.png`: a gallery of synthetic `val_test`
  frames with the keypoint sets at the headline alpha (q read from
  `results/level_b/<run>_<crop>_synthetic.json`), the true keypoints and the PnP estimate's
  projection. Frames are drawn at random (seeded) from answered frames, some whose estimate lies
  inside its own PURSE and some outside it. SPEED+ imagery: CC BY-NC-SA 4.0.
* Phase 6, from `results/shift/` (`make shift-matrix`):
  * `shift_coverage_vs_nominal` -> `assets/shift_coverage_vs_nominal_<convention>.png`: measured vs
    nominal coverage per domain, `split` arm, `predicted_crop`, one line per score.
  * `shift_silent_failure` -> `assets/shift_silent_failure.png`: silent-failure rate per arm x
    domain (A1, alpha = 0.10, `abstain_allowed`).
  * `shift_few_label_recovery` -> `assets/shift_few_label_recovery.png`: coverage and set size vs
    n labeled poolA frames (oracle), mean and min-max over draws.
  * `shift_size_vs_coverage` -> `assets/shift_size_vs_coverage.png`: median rotation radius vs
    coverage per arm x domain; infinite sets on a marked rail, never dropped.
  * `shift_gallery` -> `assets/shift_gallery.png`: 8 synthetic `val_test` frames with the A1 pose
    set (wireframes at sampled boundary poses), the C1 keypoint ellipses and the true pose, on
    P1's wireframe. Synthetic SPEED+ imagery only: CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from poseconf.conformal.propagate import project  # noqa: E402
from poseconf.conformal.so3 import quat_to_matrix  # noqa: E402
from poseconf.data.dump import dump_file, load_dump, load_dump_labels  # noqa: E402
from poseconf.data.splits import load_split  # noqa: E402
from poseconf.engine import shift as sh  # noqa: E402
from poseconf.engine.level_b import (  # noqa: E402
    estimate_in_purse,
    join_dump,
    keypoint_set,
    score_frames,
    unit_shapes,
)
from poseconf.engine.level_c import attach_variance  # noqa: E402
from poseconf.p1_adapter import (  # noqa: E402
    p1_paths,
    pose_quaternion_order,
    projection_geometry,
    wireframe_edges,
)

#: Overlay colours: reference-palette categorical slots 1-3, dark-surface steps (imagery is dark).
#: Each role also has its own marker shape, so colour never carries identity alone.
SET_COLOUR = "#3987e5"
TRUTH_COLOUR = "#d95926"
ESTIMATE_COLOUR = "#199e70"
SURFACE = "#1a1a19"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
LICENCE = "SPEED+ imagery, CC BY-NC-SA 4.0 (Park et al. 2022)"

#: Data charts (light surface): reference-palette categorical slots in fixed order, light steps.
#: The slot order is the CVD-safety mechanism; every series also has its own marker shape.
LIGHT_SURFACE = "#fcfcfb"
LIGHT_TEXT = "#0b0b0b"
LIGHT_TEXT_2 = "#52514e"
LIGHT_GRID = "#e4e3df"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
DOMAINS = ("synthetic", "lightbox", "sunlamp")
DOMAIN_LABEL = {
    "synthetic": "synthetic val_test",
    "lightbox": "lightbox poolB",
    "sunlamp": "sunlamp poolB",
}


class MissingInputs(FileNotFoundError):
    """A figure's inputs are not on this machine."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure", default="all", help="figure name or 'all'")
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("assets"))
    parser.add_argument("--score", default="B2", choices=["B1", "B2"])
    parser.add_argument("--convention", default="abstain_allowed")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n-inside", type=int, default=5, help="frames, estimate inside PURSE")
    parser.add_argument("--n-outside", type=int, default=3, help="frames, estimate outside PURSE")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--pad", type=float, default=0.15, help="crop margin, fraction of box")
    parser.add_argument("--shift-dir", type=Path, default=Path("results/shift"))
    return parser.parse_args(argv)


def _quantile(result: Path, score: str, convention: str, alpha: float) -> float:
    record = json.loads(result.read_text("utf-8"))
    rows = [
        r
        for r in record["rows"]
        if r["score"] == score and r["convention"] == convention and r["alpha"] == alpha
    ]
    if len(rows) != 1:
        raise ValueError(f"{result} has {len(rows)} rows for {score} {convention} {alpha}")
    q = rows[0]["quantile"]
    return math.inf if q == "inf" else float(q)


def _ellipse(ax: plt.Axes, centre: np.ndarray, shape: np.ndarray, q: float, width: float) -> None:
    eig, vec = np.linalg.eigh(shape)
    angle = math.degrees(math.atan2(vec[1, 1], vec[0, 1]))
    ax.add_patch(
        Ellipse(
            centre,
            2 * q * math.sqrt(eig[1]),
            2 * q * math.sqrt(eig[0]),
            angle=angle,
            fill=False,
            edgecolor=SET_COLOUR,
            linewidth=width,
        )
    )


def level_b_overlay(args: argparse.Namespace) -> Path:
    """Gallery of keypoint sets, true keypoints and the estimate's projection."""
    import yaml

    config = yaml.safe_load(args.config.read_text("utf-8"))
    level = config["level_b"]
    run, crop = level["run"], level["crop_source"]
    result = Path("results/level_b") / f"{run}_{crop}_synthetic.json"
    if not result.is_file():
        raise MissingInputs(f"{result} missing; run `make level-b` first")
    paths = p1_paths(args.paths)
    dump_path = dump_file(paths.dumps_root, run, "synthetic", crop, subset=False)
    image_dir = paths.speedplus_root / "synthetic" / "images"
    if not dump_path.is_file() or not image_dir.is_dir():
        raise MissingInputs(f"needs {dump_path} and {image_dir}")

    q = _quantile(result, args.score, args.convention, args.alpha)
    if not math.isfinite(q):
        raise ValueError(f"q is {q} at alpha {args.alpha}: nothing to draw")
    geometry = projection_geometry(run, paths=paths)
    order = pose_quaternion_order(run)
    dump, _ = load_dump(dump_path)
    labels, _ = load_dump_labels(paths.dumps_root, run, "synthetic", "all", tag="split")
    test, _ = join_dump(dump, labels, load_split(level["test_split"]), geometry, order)
    kset = keypoint_set(args.score, test, q)
    scores = score_frames(args.score, test, args.convention)
    inside = estimate_in_purse(kset, test, geometry, order)

    rng = np.random.default_rng(args.seed)
    pick_in = rng.choice(np.flatnonzero(inside), size=args.n_inside, replace=False)
    pick_out = rng.choice(np.flatnonzero(test.valid & ~inside), size=args.n_outside, replace=False)
    rows = [*np.sort(pick_in), *np.sort(pick_out)]

    shapes = unit_shapes(kset)
    # Everything per frame in frame (sorted-row) order, as `keypoint_errors` returns it.
    in_order = np.sort(np.asarray(rows))
    est_points, est_visible = project(
        quat_to_matrix(test.q_hat[in_order], order), test.t_hat[in_order], geometry
    )
    est_err = kset.keypoint_errors(est_points, np.isin(np.arange(len(test)), in_order))
    position = {int(row): i for i, row in enumerate(in_order)}

    cols = 4
    nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.0 * cols, 4.7 * nrows + 1.1))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        ax.set_axis_off()
    for panel, row in enumerate(rows):
        ax = axes.ravel()[panel]
        image = cv2.imread(str(image_dir / str(test.filenames[row])), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise MissingInputs(f"cannot read {test.filenames[row]}")
        x0, y0, x1, y1 = dump["bbox_used"][list(dump["filename"]).index(test.filenames[row])]
        side = max(x1 - x0, y1 - y0) * (1 + 2 * args.pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_facecolor(SURFACE)  # crops that run past the image edge show the surface
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_xlim(cx - side / 2, cx + side / 2)
        ax.set_ylim(cy + side / 2, cy - side / 2)
        i = position[int(row)]
        errors, ev = est_err[i], est_visible[i]
        for k in np.flatnonzero(kset.constrained[row]):
            violated = bool(ev[k] and errors[k] > q)
            _ellipse(ax, test.y_hat[row, k], shapes[row, k], q, 2.6 if violated else 1.2)
        free = kset.unconstrained[row]  # whole-image sets: marked, never silently dropped
        if free.any():
            ax.scatter(*test.y_hat[row, free].T, marker="+", s=60, c=TEXT_SECONDARY, linewidths=1.4)
        vis = test.include[row]
        ax.scatter(*test.y_gt[row, vis].T, marker="x", s=34, c=TRUTH_COLOUR, linewidths=1.6)
        ax.scatter(
            *est_points[i, ev].T,
            marker="s",
            s=30,
            facecolors="none",
            edgecolors=ESTIMATE_COLOUR,
            linewidths=1.4,
        )
        covered = scores[row] <= q
        ax.set_title(
            f"{test.filenames[row]}\n{args.score} score {scores[row]:.3g} "
            f"({'covered' if covered else 'NOT covered'}) · estimate "
            f"{'in' if inside[row] else 'OUTSIDE'} PURSE",
            color=TEXT_PRIMARY if inside[row] else TEXT_SECONDARY,
            fontsize=8.5,
        )
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(SET_COLOUR if inside[row] else TEXT_SECONDARY)
            spine.set_linestyle("-" if inside[row] else "--")

    handles = [
        Line2D([], [], color=SET_COLOUR, linewidth=1.2, label=f"{args.score} set at q = {q:.3g}"),
        Line2D([], [], color=SET_COLOUR, linewidth=2.6, label="set the estimate's keypoint leaves"),
        Line2D([], [], color=TRUTH_COLOUR, marker="x", linestyle="", label="true keypoint"),
        Line2D(
            [],
            [],
            color=ESTIMATE_COLOUR,
            marker="s",
            markerfacecolor="none",
            linestyle="",
            label="PnP estimate, projected",
        ),
    ]
    if kset.unconstrained[rows].any():
        handles.append(
            Line2D(
                [],
                [],
                color=TEXT_SECONDARY,
                marker="+",
                linestyle="",
                label="unconstrained keypoint (empty heatmap: set = whole image)",
            )
        )
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0),
    )
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    fig.suptitle(
        f"{args.score} keypoint sets, alpha = {args.alpha:g}, {args.convention}, arm split, "
        f"{level['test_split']} ({run}, {crop}). First {args.n_inside}: estimate inside "
        f"its PURSE; last {args.n_outside} (dashed frame): outside.\nPURSE = poses whose in-frame "
        f"keypoints all fall in their sets. Seeded selection, not a coverage sample. {LICENCE}",
        color=TEXT_PRIMARY,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.06, wspace=0.04, hspace=0.22)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"level_b_overlay_{args.score}.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------------------
# Phase 6: coverage under shift
# --------------------------------------------------------------------------------------------------


def _decode(value: object) -> object:
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _shift_rows(args: argparse.Namespace) -> list[dict]:
    """Every row of every shift-matrix file the index lists."""
    index = args.shift_dir / "index.json"
    if not index.is_file():
        raise MissingInputs(f"{index} missing; run `make shift-matrix` first")
    cells = json.loads(index.read_text("utf-8"))["cells"]
    rows = []
    for cell in cells:
        if cell["file"]:
            rows += _decode(json.loads((args.shift_dir / cell["file"]).read_text("utf-8")))["rows"]
    return rows


def _pick(rows: list[dict], **want: object) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in want.items())]


def _one(rows: list[dict], **want: object) -> dict | None:
    hits = _pick(rows, **want)
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} rows match {want}")
    return hits[0] if hits else None


def _light_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(LIGHT_SURFACE)
    ax.grid(True, color=LIGHT_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIGHT_TEXT_2)
    ax.tick_params(colors=LIGHT_TEXT_2, labelsize=8.5)
    ax.xaxis.label.set_color(LIGHT_TEXT)
    ax.yaxis.label.set_color(LIGHT_TEXT)
    ax.title.set_color(LIGHT_TEXT)


def _light_figure(
    nrows: int, ncols: int, size: tuple[float, float]
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    fig.patch.set_facecolor(LIGHT_SURFACE)
    for ax in axes.ravel():
        _light_axes(ax)
    return fig, axes


def _save(fig: plt.Figure, args: argparse.Namespace, name: str) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / name
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor(), metadata={"Software": None})
    plt.close(fig)
    return out


def _infinite(row: dict) -> bool:
    size = row.get("set_size") or {}
    return any(isinstance(v, dict) and math.isinf(v.get("median", 0.0)) for v in size.values())


def shift_coverage_vs_nominal(args: argparse.Namespace) -> Path:
    """Measured vs nominal coverage, split arm, predicted_crop: one panel per domain."""
    rows = _pick(_shift_rows(args), arm="split", crop_source="predicted_crop")
    scores = sorted({r["score"] for r in rows})
    written = []
    for convention in ("abstain_allowed", "answer_required"):
        fig, axes = _light_figure(1, 3, (14.5, 5.0))
        for ax, domain in zip(axes[0], DOMAINS, strict=True):
            ax.plot([0, 1], [0, 1], color=LIGHT_TEXT_2, linestyle="--", linewidth=1.0)
            for i, score in enumerate(scores):
                sel = sorted(
                    _pick(rows, score=score, domain=domain, convention=convention),
                    key=lambda r: r["alpha"],
                )
                x = np.array([1 - r["alpha"] for r in sel])
                y = np.array([r["coverage"] for r in sel])
                lo = y - np.array([r["coverage_ci95"][0] for r in sel])
                hi = np.array([r["coverage_ci95"][1] for r in sel]) - y
                ax.errorbar(x, y, yerr=[lo, hi], color=SLOTS[i], linewidth=1.5, elinewidth=1.0,
                            capsize=0, zorder=3)  # fmt: skip
                inf = np.array([_infinite(r) for r in sel])
                ax.scatter(x[~inf], y[~inf], marker=MARKERS[i], s=42, color=SLOTS[i],
                           edgecolors=LIGHT_SURFACE, linewidths=1.2, zorder=4, label=score)  # fmt: skip
                ax.scatter(x[inf], y[inf], marker=MARKERS[i], s=42, facecolors=LIGHT_SURFACE,
                           edgecolors=SLOTS[i], linewidths=1.4, zorder=4)  # fmt: skip
            n = sel[0]["n_total"] if sel else 0
            ax.set_title(f"{DOMAIN_LABEL[domain]} (n = {n})", fontsize=10.5)
            ax.set_xlim(0.45, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel("nominal coverage 1 − α")
        axes[0, 0].set_ylabel("measured coverage (95 % Clopper–Pearson)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor=LIGHT_SURFACE,
                              markeredgecolor=LIGHT_TEXT_2, label="hollow: set = ∞ (no information)"))  # fmt: skip
        handles.append(
            Line2D([], [], color=LIGHT_TEXT_2, linestyle="--", label="measured = nominal")
        )
        legend = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                            fontsize=9, bbox_to_anchor=(0.5, 0.0))  # fmt: skip
        for text in legend.get_texts():
            text.set_color(LIGHT_TEXT)
        fig.suptitle(
            f"Coverage under shift: calibrated on synthetic val_cal (arm split, predicted_crop, "
            f"{convention}). Guaranteed only on synthetic; HIL is measured.",
            color=LIGHT_TEXT, fontsize=11.5,
        )  # fmt: skip
        fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.2, wspace=0.12)
        written.append(_save(fig, args, f"shift_coverage_vs_nominal_{convention}.png"))
    return written[0].parent / "shift_coverage_vs_nominal_*.png"


SILENT_ARMS = (
    ("split", "split"),
    ("mondrian", "mondrian"),
    ("weighted_unlabeled_target", "weighted (unlabeled poolA)"),
    ("oracle_target_labels_n1000", "oracle: 1,000 labeled poolA"),
)


def shift_silent_failure(args: argparse.Namespace) -> Path:
    """Silent-failure rate (answered and wrong) per arm x domain, A1, alpha 0.10, abstain."""
    rows = _pick(_shift_rows(args), score="A1", convention="abstain_allowed", alpha=0.1,
                 crop_source="predicted_crop")  # fmt: skip
    fig, axes = _light_figure(1, 1, (10.5, 5.2))
    ax = axes[0, 0]
    width = 0.2
    for j, (arm, label) in enumerate(SILENT_ARMS):
        for d, domain in enumerate(DOMAINS):
            x = d + (j - 1.5) * (width + 0.02)
            row = _one(rows, arm=arm, domain=domain)
            if row is None:
                ax.text(x, 0.004, "n/a", ha="center", va="bottom", fontsize=8, color=LIGHT_TEXT_2,
                        rotation=90)  # fmt: skip
                continue
            value = row["silent_failure_rate"]
            ax.bar(x, value, width=width, color=SLOTS[j], edgecolor=LIGHT_SURFACE, linewidth=2,
                   hatch="//" if row["oracle"] else None,
                   label=label if d == 1 else None)  # fmt: skip
            ax.text(x, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=8,
                    color=LIGHT_TEXT)  # fmt: skip
            if _infinite(row):
                ax.text(x, value + 0.03, "set ∞", ha="center", fontsize=7.5, color=LIGHT_TEXT_2)
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels([DOMAIN_LABEL[d] for d in DOMAINS], color=LIGHT_TEXT)
    ax.set_ylabel("silent-failure rate  P(answered ∧ truth ∉ set)")
    ax.axhline(0.1, color=LIGHT_TEXT_2, linestyle="--", linewidth=1.0)
    ax.text(-0.45, 0.103, "α = 0.10", fontsize=8, color=LIGHT_TEXT_2, ha="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(LIGHT_TEXT)
    ax.set_title(
        "A1 pose sets, α = 0.10, abstain_allowed, predicted_crop: confident wrong answers per arm "
        "(hatched = oracle, uses target labels)",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.1)
    return _save(fig, args, "shift_silent_failure.png")


def shift_few_label_recovery(args: argparse.Namespace) -> Path:
    """Coverage and median rotation radius vs n labeled poolA frames (oracle), A1 and C2."""
    rows = _pick(_shift_rows(args), convention="abstain_allowed", alpha=0.1,
                 crop_source="predicted_crop")  # fmt: skip
    scores = ("A1", "C2")
    fig, axes = _light_figure(2, 2, (12.0, 8.0))
    for c, domain in enumerate(DOMAINS[1:]):
        top, bottom = axes[0, c], axes[1, c]
        for i, score in enumerate(scores):
            oracle = sorted(
                (r for r in _pick(rows, score=score, domain=domain)
                 if r["arm"].startswith("oracle_target_labels")),
                key=lambda r: r["n_cal"],
            )  # fmt: skip
            if not oracle:
                raise MissingInputs(f"no oracle rows for {score} {domain}")
            n = np.array([r["n_cal"] for r in oracle])
            mean = np.array([r["coverage"] for r in oracle])
            top.fill_between(n, [r["coverage_min"] for r in oracle],
                             [r["coverage_max"] for r in oracle], color=SLOTS[i], alpha=0.18,
                             linewidth=0)  # fmt: skip
            top.plot(n, mean, color=SLOTS[i], linewidth=2, marker=MARKERS[i], markersize=7,
                     markeredgecolor=LIGHT_SURFACE, label=f"{score}: oracle, mean (band: min–max of 5 draws)")  # fmt: skip
            base = _one(rows, score=score, domain=domain, arm="split")
            top.axhline(base["coverage"], color=SLOTS[i], linestyle=":", linewidth=1.6,
                        label=f"{score}: split, synthetic calibration ({base['coverage']:.3f})")  # fmt: skip
            radius = [r["set_size"]["rot_deg"]["median"] if r["set_size"] else np.nan
                      for r in oracle]  # fmt: skip
            bottom.plot(n, radius, color=SLOTS[i], linewidth=2, marker=MARKERS[i], markersize=7,
                        markeredgecolor=LIGHT_SURFACE, label=f"{score}: oracle (median over draws)")  # fmt: skip
            bottom.axhline(base["set_size"]["rot_deg"]["median"], color=SLOTS[i], linestyle=":",
                           linewidth=1.6, label=f"{score}: split")  # fmt: skip
        top.axhline(0.9, color=LIGHT_TEXT_2, linestyle="--", linewidth=1.0)
        top.text(26, 0.903, "nominal 0.90", fontsize=8, color=LIGHT_TEXT_2, ha="left", va="bottom")
        n_eval = oracle[0]["n_total"]
        top.set_title(f"{DOMAIN_LABEL[domain]} (n = {n_eval} evaluated)", fontsize=10.5)
        for ax in (top, bottom):
            ax.set_xscale("log")
            ax.set_xticks([25, 50, 100, 250, 500, 1000])
            ax.set_xticklabels(["25", "50", "100", "250", "500", "1000"])
        top.set_ylim(0.6, 1.0)
        bottom.set_xlabel("n labeled poolA frames used for calibration")
    axes[0, 0].set_ylabel("measured coverage on poolB")
    axes[1, 0].set_ylabel("median rotation radius (°), answered frames")
    for ax, loc in zip(axes.ravel(), ("lower right", "lower right", "upper right", "upper right"),
                       strict=True):  # fmt: skip
        legend = ax.legend(frameon=False, fontsize=7.8, loc=loc)
        for text in legend.get_texts():
            text.set_color(LIGHT_TEXT)
    fig.suptitle(
        "How many labeled real frames buy coverage back? oracle_target_labels_n*, α = 0.10, "
        "abstain_allowed, predicted_crop (oracle: uses HIL labels)",
        color=LIGHT_TEXT, fontsize=11.5,
    )  # fmt: skip
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.07, hspace=0.25, wspace=0.14)
    return _save(fig, args, "shift_few_label_recovery.png")


def shift_size_vs_coverage(args: argparse.Namespace) -> Path:
    """Median rotation radius vs coverage per arm x domain; infinite sets on a marked rail."""
    rows = _pick(_shift_rows(args), score="A1", convention="abstain_allowed", alpha=0.1,
                 crop_source="predicted_crop")  # fmt: skip
    fig, axes = _light_figure(1, 1, (10.0, 6.0))
    ax = axes[0, 0]
    points = []
    for j, (arm, _label) in enumerate(SILENT_ARMS):
        for d, domain in enumerate(DOMAINS):
            row = _one(rows, arm=arm, domain=domain)
            if row is not None and row["set_size"] is not None:
                points.append((j, d, row))
    finite = [r["set_size"]["rot_deg"]["median"] for _, _, r in points
              if math.isfinite(r["set_size"]["rot_deg"]["median"])]  # fmt: skip
    rail = max(finite) * 2.5
    for j, d, row in points:
        radius = row["set_size"]["rot_deg"]["median"]
        # A split set's radius q c_R is fixed at calibration, the same on every domain: offset the
        # domains slightly (log x) so no point hides another.
        x = (rail if math.isinf(radius) else radius) * 1.04 ** (d - 1)
        lo, hi = row["coverage_ci95"]
        ax.errorbar(x, row["coverage"], yerr=[[row["coverage"] - lo], [hi - row["coverage"]]],
                    color=SLOTS[d], linewidth=1.2, capsize=0, zorder=3)  # fmt: skip
        ax.scatter(x, row["coverage"], marker=MARKERS[j], s=70, color=SLOTS[d],
                   edgecolors=LIGHT_SURFACE, linewidths=1.4, zorder=4,
                   hatch="///" if row["oracle"] else None)  # fmt: skip
    ax.axvline(rail, color=LIGHT_TEXT_2, linewidth=1.0)
    ax.text(rail * 1.04, 0.05, "∞ (whole space)", rotation=90, fontsize=8.5, color=LIGHT_TEXT_2,
            va="bottom")  # fmt: skip
    ax.axhline(0.9, color=LIGHT_TEXT_2, linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(min(finite) / 1.5, rail * 1.4)
    ticks = [t for t in (1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30) if min(finite) / 1.5 < t < rail * 0.9]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("median rotation radius of answered sets (°, log)")
    ax.set_ylabel("measured coverage (95 % Clopper–Pearson)")
    handles = [Line2D([], [], marker="o", linestyle="", color=SLOTS[d], markersize=8,
                      label=DOMAIN_LABEL[domain]) for d, domain in enumerate(DOMAINS)]  # fmt: skip
    handles += [Line2D([], [], marker=MARKERS[j], linestyle="", color=LIGHT_TEXT_2, markersize=8,
                       markerfacecolor="none", label=label)
                for j, (_, label) in enumerate(SILENT_ARMS)]  # fmt: skip
    legend = ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    for text in legend.get_texts():
        text.set_color(LIGHT_TEXT)
    ax.set_title(
        "Coverage is only informative beside set size: A1, α = 0.10, abstain_allowed, "
        "predicted_crop\n(colour = domain, marker = arm; domains offset slightly in x: a split "
        "set has the same radius everywhere)",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.1)
    return _save(fig, args, "shift_size_vs_coverage.png")


def _boundary_poses(rotation: np.ndarray, t: np.ndarray, r_rot: float, r_t: float, count: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """`count` poses on the A1 set boundary: rotation angle r_rot about random axes (geodesic
    distance r_rot) and translation offset r_t in random directions."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    dirs = rng.normal(size=(count, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rots = Rotation.from_rotvec(axes * r_rot).as_matrix() @ rotation
    return rots, t + dirs * r_t


def _draw_wireframe(ax: plt.Axes, points: np.ndarray, visible: np.ndarray, edges, **style) -> None:
    for a, b in edges:
        if visible[a] and visible[b]:
            ax.plot(*np.stack([points[a], points[b]]).T, **style)


def shift_gallery(args: argparse.Namespace) -> Path:
    """Predicted A1 pose sets and C1 keypoint sets on P1's wireframe, 8 synthetic val_test frames."""
    import yaml

    rows_path = args.shift_dir / "keypoint_a2_synthetic_predicted_crop_split.json"
    if not rows_path.is_file():
        raise MissingInputs(f"{rows_path} missing; run `make shift-matrix` first")
    record = _decode(json.loads(rows_path.read_text("utf-8")))
    config = yaml.safe_load(args.config.read_text("utf-8"))
    cfg = config["shift"]
    run = cfg["run"]
    paths = p1_paths(args.paths)
    dump_path = dump_file(paths.dumps_root, run, "synthetic", "predicted_crop", subset=False)
    image_dir = paths.speedplus_root / "synthetic" / "images"
    if not dump_path.is_file() or not image_dir.is_dir():
        raise MissingInputs(f"needs {dump_path} and {image_dir}")
    alpha, convention = args.alpha, args.convention
    q_a1 = _one(record["rows"], score="A1", convention=convention, alpha=alpha)["quantile"]
    q_c1 = _one(record["rows"], score="C1", convention=convention, alpha=alpha)["quantile"]
    if not (math.isfinite(q_a1) and math.isfinite(q_c1)):
        raise ValueError(f"q is infinite at alpha {alpha}: nothing to draw")
    fit = record["fit"]["level_a"]
    c_r, c_t = float(fit["c_R_rad"]), float(fit["c_t"])

    geometry = projection_geometry(run, paths=paths)
    order = pose_quaternion_order(run)
    edges = wireframe_edges()
    dump, _ = load_dump(dump_path)
    side, _ = load_dump(dump_path.with_name(dump_path.stem + "_vhead.npz"))
    dump = attach_variance(dump, side)
    labels, _ = sh.evaluation_labels(paths.dumps_root, run, "synthetic", tag="figure")
    test, _ = sh.build_frames(dump, labels, load_split(cfg["test_subsets"]["synthetic"]),
                              geometry, order)  # fmt: skip
    kp = test.c1
    kset = keypoint_set("C1", kp, q_c1)
    shapes = unit_shapes(kset)
    a1 = sh.scores("A1", test, _fit_from_record(fit), convention)

    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(np.flatnonzero(kp.valid), size=8, replace=False))
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 9.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, row in zip(axes.ravel(), rows, strict=True):
        image = cv2.imread(str(image_dir / str(kp.filenames[row])), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise MissingInputs(f"cannot read {kp.filenames[row]}")
        x0, y0, x1, y1 = test.bbox_used[row]
        span = max(x1 - x0, y1 - y0) * (1 + 2 * args.pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_facecolor(SURFACE)
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy + span / 2, cy - span / 2)
        rotation = quat_to_matrix(kp.q_hat[row][None], order)[0]
        r_t = q_a1 * c_t * float(np.linalg.norm(kp.t_hat[row]))
        rots, trans = _boundary_poses(rotation, kp.t_hat[row], q_a1 * c_r, r_t, 24, args.seed + row)
        pts, vis = project(rots, trans, geometry)
        for p_i, v_i in zip(pts, vis, strict=True):
            _draw_wireframe(ax, p_i, v_i, edges, color=SET_COLOUR, linewidth=0.7, alpha=0.35)
        est, est_vis = project(rotation[None], kp.t_hat[row][None], geometry)
        _draw_wireframe(ax, est[0], est_vis[0], edges, color=SET_COLOUR, linewidth=1.8)
        truth, truth_vis = project(quat_to_matrix(kp.q_gt[row][None], order), kp.t_gt[row][None],
                                   geometry)  # fmt: skip
        _draw_wireframe(ax, truth[0], truth_vis[0], edges, color=TRUTH_COLOUR, linewidth=1.3,
                        linestyle="--")  # fmt: skip
        for k in np.flatnonzero(kset.constrained[row]):
            _ellipse_colour(ax, kp.y_hat[row, k], shapes[row, k], q_c1, ESTIMATE_COLOUR)
        covered = a1[row] <= q_a1
        ax.set_title(f"{kp.filenames[row]} · A1 score {a1[row]:.2f} "
                     f"({'covered' if covered else 'NOT covered'})",
                     color=TEXT_PRIMARY if covered else TEXT_SECONDARY, fontsize=8.5)  # fmt: skip
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_SECONDARY)
    handles = [
        Line2D([], [], color=SET_COLOUR, linewidth=1.8, label="PnP estimate (P1 wireframe)"),
        Line2D(
            [],
            [],
            color=SET_COLOUR,
            linewidth=0.7,
            alpha=0.6,
            label=f"A1 set boundary: {math.degrees(q_a1 * c_r):.2f}° × {100 * q_a1 * c_t:.2f} % of ‖t̂‖ (24 samples)",
        ),  # fmt: skip
        Line2D([], [], color=TRUTH_COLOUR, linewidth=1.3, linestyle="--", label="true pose"),
        Line2D(
            [], [], color=ESTIMATE_COLOUR, linewidth=1.2, label=f"C1 keypoint sets, q = {q_c1:.3g}"
        ),
    ]
    legend = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
                        bbox_to_anchor=(0.5, 0))  # fmt: skip
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    fig.suptitle(
        f"Predicted sets on 8 seeded answered synthetic val_test frames (arm split, α = {alpha:g}, "
        f"{convention}, {run}, predicted_crop). Seeded selection, not a coverage sample.\n{LICENCE}",
        color=TEXT_PRIMARY, fontsize=10.5,
    )  # fmt: skip
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.06, wspace=0.04, hspace=0.14)
    return _save(fig, args, "shift_gallery.png")


def _ellipse_colour(
    ax: plt.Axes, centre: np.ndarray, shape: np.ndarray, q: float, colour: str
) -> None:
    eig, vec = np.linalg.eigh(shape)
    angle = math.degrees(math.atan2(vec[1, 1], vec[0, 1]))
    ax.add_patch(Ellipse(centre, 2 * q * math.sqrt(eig[1]), 2 * q * math.sqrt(eig[0]), angle=angle,
                         fill=False, edgecolor=colour, linewidth=1.1))  # fmt: skip


def _fit_from_record(fit: dict):
    from poseconf.engine.level_a import LevelAFit

    return LevelAFit(
        c_R=float(fit["c_R_rad"]), c_t=float(fit["c_t"]), c_z=float(fit["c_z"]),
        c_xy=float(fit["c_xy"]), g_x=np.asarray(fit["g"]["x"]), g_y=np.asarray(fit["g"]["y"]),
        n_fit=int(fit["n_fit_solved"]),
    )  # fmt: skip


FIGURES: dict[str, Callable[[argparse.Namespace], Path]] = {
    "level_b_overlay": level_b_overlay,
    "shift_coverage_vs_nominal": shift_coverage_vs_nominal,
    "shift_silent_failure": shift_silent_failure,
    "shift_few_label_recovery": shift_few_label_recovery,
    "shift_size_vs_coverage": shift_size_vs_coverage,
    "shift_gallery": shift_gallery,
}


def main(argv: list[str] | None = None) -> int:
    """Build the requested figure(s)."""
    args = parse_args(argv)
    names = list(FIGURES) if args.figure == "all" else [args.figure]
    for name in names:
        if name not in FIGURES:
            raise ValueError(f"unknown figure {name!r}; known: {sorted(FIGURES)}")
        try:
            print(f"wrote {FIGURES[name](args)}")
        except MissingInputs as missing:
            if args.figure != "all":
                raise
            print(f"skipped {name}: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
