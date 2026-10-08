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
from matplotlib.patches import Ellipse, Patch  # noqa: E402

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
            ref = _one(rows, score="A1", domain=domain, convention=convention, alpha=0.1)
            answer = ref["answer_rate"]
            if convention == "abstain_allowed":
                bound, text = 1 - answer, f"floor 1 − answer rate = {1 - answer:.3f}"
            else:
                bound, text = answer, f"ceiling = answer rate {answer:.3f} (finite sets)"
            ax.axhline(bound, color=LIGHT_TEXT_2, linestyle=":", linewidth=1.2)
            if convention == "abstain_allowed":
                ax.text(0.995, bound - 0.045, text, fontsize=8, color=LIGHT_TEXT_2, ha="right")
            else:
                ax.text(0.46, bound + 0.015, text, fontsize=8, color=LIGHT_TEXT_2, ha="left")
            ax.set_title(f"{DOMAIN_LABEL[domain]} (n = {n}; A1 answer rate {answer:.3f})",
                         fontsize=10.5)  # fmt: skip
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
        bound_label = (
            "abstentions count as covered: coverage ≥ 1 − answer rate"
            if convention == "abstain_allowed"
            else "failures uncovered unless q = ∞: coverage ≤ answer rate"
        )
        handles.append(Line2D([], [], color=LIGHT_TEXT_2, linestyle=":", label=bound_label))
        legend = fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                            fontsize=9, bbox_to_anchor=(0.5, 0.0))  # fmt: skip
        for text in legend.get_texts():
            text.set_color(LIGHT_TEXT)
        fig.suptitle(
            f"Coverage under shift: calibrated on synthetic val_cal (arm split, predicted_crop, "
            f"{convention}).\nMarginal, finite-sample coverage ≥ 1 − α holds on synthetic val_test "
            f"(exchangeable with val_cal); on HIL poolB it is measured, not guaranteed.",
            color=LIGHT_TEXT, fontsize=11.5,
        )  # fmt: skip
        fig.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.24, wspace=0.12)
        written.append(_save(fig, args, f"shift_coverage_vs_nominal_{convention}.png"))
    return written[0].parent / "shift_coverage_vs_nominal_*.png"


WEIGHTED_ARM = "weighted_unlabeled_target"
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
    answer = {d: _one(rows, arm="split", domain=d)["answer_rate"] for d in DOMAINS}
    ax.set_xticklabels([f"{DOMAIN_LABEL[d]}\nanswer rate {answer[d]:.3f}" for d in DOMAINS],
                       color=LIGHT_TEXT)  # fmt: skip
    ax.set_ylabel("silent-failure rate  P(answered ∧ truth ∉ set)")
    ax.axhline(0.1, color=LIGHT_TEXT_2, linestyle="--", linewidth=1.0)
    ax.text(2.55, 0.113, "α = 0.10", fontsize=8, color=LIGHT_TEXT_2, ha="right")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(LIGHT_TEXT)
    ax.set_title(
        "A1 pose sets, α = 0.10, abstain_allowed, predicted_crop: confident wrong answers per arm "
        "(hatched = oracle, uses target labels)",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.13)
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
            if score == "A1":
                for r in oracle:
                    top.text(r["n_cal"], 0.03, f"ans\n{r['answer_rate']:.3f}", ha="center",
                             fontsize=7.5, color=LIGHT_TEXT_2,
                             transform=top.get_xaxis_transform())  # fmt: skip
            radius = [r["set_size"]["rot_deg"]["median"] if r["set_size"] else np.nan
                      for r in oracle]  # fmt: skip
            for r, value in zip(oracle, radius, strict=True):
                answered = r["set_size"]["n_draws_with_answers"] if r["set_size"] else 0
                if score == "A1" and answered < r["n_draws"] and math.isfinite(value):
                    bottom.annotate(f"{answered}/{r['n_draws']} draws answer", (r["n_cal"], value),
                                    textcoords="offset points", xytext=(0, 8), ha="center",
                                    fontsize=7.5, color=LIGHT_TEXT_2)  # fmt: skip
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
    axes[1, 0].set_ylabel("median rotation radius (°)\n(answered frames; draws with answers)")
    for ax, loc in zip(axes.ravel(), ("lower right", "lower right", "upper right", "upper right"),
                       strict=True):  # fmt: skip
        anchor = (1.0, 0.13) if loc == "lower right" else (1.0, 1.0)
        legend = ax.legend(frameon=False, fontsize=7.8, loc=loc, bbox_to_anchor=anchor)
        for text in legend.get_texts():
            text.set_color(LIGHT_TEXT)
    fig.suptitle(
        "How many labeled real frames buy coverage back? oracle_target_labels_n*, α = 0.10, "
        "abstain_allowed, predicted_crop (oracle: uses HIL labels)\nAn abstention counts as "
        "covered: read coverage with the mean A1 answer rate (ans, bottom of each panel).\n"
        "A draw with q = −∞ abstains on every frame.",
        color=LIGHT_TEXT, fontsize=11.5,
    )  # fmt: skip
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.07, hspace=0.25, wspace=0.14)
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
        if SILENT_ARMS[j][0] == WEIGHTED_ARM:
            # The other arms share their domain's answer rate (in the legend); the weighted arm's
            # differs, so it is labelled at the point: lightbox left of the rail, sunlamp right.
            ax.annotate(f"ans {row['answer_rate']:.3f}", (x, row["coverage"]),
                        textcoords="offset points", xytext=(-52, -14) if d == 1 else (8, -14),
                        fontsize=7.5, color=LIGHT_TEXT_2)  # fmt: skip
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
    shared = {}
    for d, domain in enumerate(DOMAINS):
        rates = {round(r["answer_rate"], 3) for _, dd, r in points
                 if dd == d and r["arm"] != WEIGHTED_ARM}  # fmt: skip
        if len(rates) != 1:
            raise ValueError(f"{domain}: non-weighted arms differ in answer rate {sorted(rates)}")
        shared[domain] = rates.pop()
    handles = [Line2D([], [], marker="o", linestyle="", color=SLOTS[d], markersize=8,
                      label=f"{DOMAIN_LABEL[domain]} (answer rate {shared[domain]:.3f})")
               for d, domain in enumerate(DOMAINS)]  # fmt: skip
    handles += [Line2D([], [], marker=MARKERS[j], linestyle="", color=LIGHT_TEXT_2, markersize=8,
                       markerfacecolor="none", label=label)
                for j, (_, label) in enumerate(SILENT_ARMS)]  # fmt: skip
    legend = ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower left", ncol=2,
                       title="answer rate in the legend: split, mondrian and oracle share it; "
                             "weighted labelled at its points", title_fontsize=8)  # fmt: skip
    legend.get_title().set_color(LIGHT_TEXT_2)
    for text in legend.get_texts():
        text.set_color(LIGHT_TEXT)
    ax.set_title(
        "Coverage is only informative beside set size and answer rate: A1, α = 0.10, "
        "abstain_allowed, "
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


# --------------------------------------------------------------------------------------------------
# Phase 9 showcase: the headline outcome chart and two GIFs
# --------------------------------------------------------------------------------------------------

#: "No pose" is the absence of an answer, not a series: neutral grey on both surfaces.
NO_POSE_LIGHT = "#b9b8b1"
NO_POSE_DARK = "#6f6e69"
#: Answered-and-missed (silent failure): categorical slot 4 (yellow), always hatched as well.
MISSED_LIGHT = "#eda100"
MISSED_DARK = "#c98500"
#: GIFs: render size and palette budget, frame holds (ms), poses drawn on each set boundary.
GIF_DPI = 90
GIF_COLOURS = 96
TOUR_PER_DOMAIN = 6
TOUR_FRAME_MS = 2200
SWEEP_FRAME_MS = 1500
SWEEP_LAST_MS = 3500
BOUNDARY_POSES = 24
#: The alpha sweep runs from loose to tight over the committed alpha grid.
SWEEP_ALPHAS = (0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01)
SWEEP_DOMAINS = ("synthetic", "lightbox")
#: Sweep chart series: dark-surface slots 3 and 7 (blue and orange already mean set and truth).
SWEEP_COLOURS = ("#199e70", "#9085e9")
OUTCOMES = ("covered", "no_pose", "missed")


def _outcomes(row: dict) -> dict[str, int]:
    """Split a coverage row into frame counts: answered and covered, no pose, answered and missed.

    Under `abstain_allowed` a frame with no pose is covered (an abstention); under
    `answer_required` it is uncovered (finite q). The parts must reproduce the row's own coverage
    and silent-failure rate, or this raises.
    """
    n, n_answered, n_covered = row["n_total"], row["n_answered"], row["n_covered"]
    no_pose = n - n_answered
    if row["convention"] == "abstain_allowed":
        covered = n_covered - no_pose
    elif math.isfinite(row["quantile"]):
        covered = n_covered
    else:
        raise ValueError("answer_required with q = inf: every frame is covered, no outcome split")
    missed = n_answered - covered
    if min(covered, missed, no_pose) < 0:
        raise ValueError(f"negative outcome count in {row['domain']} {row['convention']}")
    if not (math.isclose(missed / n, row["silent_failure_rate"], abs_tol=1e-12)
            and math.isclose(n_covered / n, row["coverage"], abs_tol=1e-12)):  # fmt: skip
        raise ValueError(f"outcome split does not reproduce {row['domain']} {row['convention']}")
    return {"covered": covered, "no_pose": no_pose, "missed": missed}


def headline_outcomes(args: argparse.Namespace) -> Path:
    """Every evaluated frame, by outcome, per domain: A1, split, alpha 0.10, both conventions."""
    rows = _pick(_shift_rows(args), score="A1", arm="split", alpha=0.1,
                 crop_source="predicted_crop")  # fmt: skip
    conventions = ("abstain_allowed", "answer_required")
    fig, axes = _light_figure(2, 1, (11.0, 7.0))
    colours = {"covered": SLOTS[0], "no_pose": NO_POSE_LIGHT, "missed": MISSED_LIGHT}
    ink = {"covered": "#ffffff", "no_pose": LIGHT_TEXT, "missed": LIGHT_TEXT}
    hatch = {"covered": None, "no_pose": None, "missed": "///"}
    rule = {
        "abstain_allowed": "abstain_allowed: no pose = abstention, counted as covered",
        "answer_required": "answer_required: no pose = a miss (the set must contain the truth)",
    }
    for ax, convention in zip(axes[:, 0], conventions, strict=True):
        for d, domain in enumerate(DOMAINS):
            row = _one(rows, domain=domain, convention=convention)
            if row is None:
                raise MissingInputs(f"no A1 split row for {domain} {convention}")
            parts = _outcomes(row)
            y, left = len(DOMAINS) - 1 - d, 0.0
            for name in OUTCOMES:
                width = parts[name] / row["n_total"]
                ax.barh(y, width, left=left, height=0.62, color=colours[name],
                        edgecolor=LIGHT_SURFACE, linewidth=2, hatch=hatch[name])  # fmt: skip
                if width >= 0.05:
                    ax.text(left + width / 2, y, f"{width:.3f}", ha="center", va="center",
                            fontsize=8.5, color=ink[name], zorder=6,
                            bbox={"facecolor": colours[name], "edgecolor": "none", "pad": 1.5})  # fmt: skip
                left += width
            lo, hi = row["coverage_ci95"]
            ax.errorbar(row["coverage"], y + 0.42, xerr=[[row["coverage"] - lo], [hi - row["coverage"]]],
                        fmt="v", color=LIGHT_TEXT, markersize=7, elinewidth=1.4, capsize=3,
                        zorder=5)  # fmt: skip
            ax.text(1.015, y, f"coverage {row['coverage']:.3f}\n[{lo:.3f}, {hi:.3f}]",
                    va="center", fontsize=8.5, color=LIGHT_TEXT,
                    transform=ax.get_yaxis_transform())  # fmt: skip
        ax.axvline(0.9, color=LIGHT_TEXT, linestyle="--", linewidth=1.1, zorder=4)
        ax.set_yticks(range(len(DOMAINS)))
        labels = []
        for domain in reversed(DOMAINS):
            row = _one(rows, domain=domain, convention=convention)
            labels.append(f"{DOMAIN_LABEL[domain]}\nn = {row['n_total']:,} · answer rate "
                          f"{row['answer_rate']:.3f}")  # fmt: skip
        ax.set_yticklabels(labels, color=LIGHT_TEXT, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(DOMAINS) - 0.2)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(rule[convention], fontsize=10, loc="left", color=LIGHT_TEXT)
    axes[-1, 0].set_xlabel("fraction of evaluated frames")
    handles = [
        Patch(facecolor=colours["covered"], label="answered, truth ∈ set"),
        Patch(facecolor=colours["no_pose"], label="no pose (PnP failed)"),
        Patch(facecolor=colours["missed"], hatch="///", edgecolor=LIGHT_SURFACE,
                                 label="answered, truth ∉ set (silent failure)"),
        Line2D([], [], marker="v", color=LIGHT_TEXT, linestyle="", markersize=7,
               label="measured coverage, 95 % Clopper–Pearson"),
        Line2D([], [], color=LIGHT_TEXT, linestyle="--", label="nominal 1 − α = 0.90"),
    ]  # fmt: skip
    legend = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
                        bbox_to_anchor=(0.5, 0.0))  # fmt: skip
    for text in legend.get_texts():
        text.set_color(LIGHT_TEXT)
    fig.suptitle(
        "One synthetic calibration, three domains: A1 pose sets, arm split, α = 0.10, "
        "predicted_crop, q from synthetic val_cal.\nOn synthetic val_test the marginal coverage "
        "guarantee applies (exchangeable with val_cal);\non HIL poolB coverage is measured, not "
        "guaranteed.",
        color=LIGHT_TEXT, fontsize=11,
    )  # fmt: skip
    fig.subplots_adjust(left=0.2, right=0.87, top=0.84, bottom=0.17, hspace=0.42)
    return _save(fig, args, "headline_outcomes.png")


class _Domain:
    """One domain's predicted_crop frames with the shift study's own A1 fit, scores and q."""

    def __init__(self, args: argparse.Namespace, domain: str) -> None:
        import yaml

        cfg = yaml.safe_load(args.config.read_text("utf-8"))["shift"]
        self.run, self.domain = cfg["run"], domain
        path = args.shift_dir / f"{self.run}_{domain}_predicted_crop_split.json"
        if not path.is_file():
            raise MissingInputs(f"{path} missing; run `make shift-matrix` first")
        self.record = _decode(json.loads(path.read_text("utf-8")))
        paths = p1_paths(args.paths)
        dump_path = dump_file(paths.dumps_root, self.run, domain, "predicted_crop", subset=False)
        self.image_dir = paths.speedplus_root / domain / "images"
        side_path = dump_path.with_name(dump_path.stem + "_vhead.npz")
        if not (dump_path.is_file() and side_path.is_file() and self.image_dir.is_dir()):
            raise MissingInputs(f"needs {dump_path}, {side_path} and {self.image_dir}")
        self.geometry = projection_geometry(self.run, paths=paths)
        self.order = pose_quaternion_order(self.run)
        dump, _ = load_dump(dump_path)
        dump = attach_variance(dump, load_dump(side_path)[0])
        # HIL: poolB labels only (evaluation), through the guarded loader (invariant 4a).
        labels, _ = sh.evaluation_labels(paths.dumps_root, self.run, domain, tag="figure")
        names = load_split(cfg["test_subsets"][domain])
        self.frames, _ = sh.build_frames(dump, labels, names, self.geometry, self.order)
        fit = self.record["fit"]["level_a"]
        self.c_r, self.c_t = float(fit["c_R_rad"]), float(fit["c_t"])
        self.scores = sh.scores("A1", self.frames, _fit_from_record(fit), "abstain_allowed")

    def row(self, alpha: float) -> dict:
        """The committed A1 abstain_allowed row at `alpha`, checked against these frames."""
        row = _one(self.record["rows"], score="A1", convention="abstain_allowed", alpha=alpha)
        if row is None:
            raise MissingInputs(f"no A1 abstain_allowed row at alpha {alpha} for {self.domain}")
        covered = int(np.sum(self.scores <= row["quantile"]))
        if covered != row["n_covered"] or len(self.scores) != row["n_total"]:
            raise ValueError(f"{self.domain} alpha {alpha}: frames give {covered} covered, the "
                             f"committed row {row['n_covered']}")  # fmt: skip
        return row

    def outcome(self, i: int, q: float) -> str:
        """`covered`, `missed` or `no_pose` for frame `i` at quantile `q`."""
        if not self.frames.kp.valid[i]:
            return "no_pose"
        return "covered" if self.scores[i] <= q else "missed"

    def draw(self, ax: plt.Axes, i: int, q: float, *, pad: float, seed: int) -> None:
        """Image crop with the A1 set at `q` (boundary poses), the estimate and the true pose."""
        kp, edges = self.frames.kp, wireframe_edges()
        image = cv2.imread(str(self.image_dir / str(kp.filenames[i])), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise MissingInputs(f"cannot read {kp.filenames[i]}")
        boxes = [self.frames.bbox_gt[i]]
        if np.all(np.isfinite(self.frames.bbox_used[i])):
            boxes.append(self.frames.bbox_used[i])
        boxes = np.stack(boxes)
        x0, y0 = boxes[:, :2].min(axis=0)
        x1, y1 = boxes[:, 2:].max(axis=0)
        span = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_facecolor(SURFACE)
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy + span / 2, cy - span / 2)
        if kp.valid[i]:
            rotation = quat_to_matrix(kp.q_hat[i][None], self.order)[0]
            r_t = q * self.c_t * float(np.linalg.norm(kp.t_hat[i]))
            rots, trans = _boundary_poses(rotation, kp.t_hat[i], q * self.c_r, r_t, BOUNDARY_POSES,
                                          seed)  # fmt: skip
            pts, vis = project(rots, trans, self.geometry)
            for p_i, v_i in zip(pts, vis, strict=True):
                _draw_wireframe(ax, p_i, v_i, edges, color=SET_COLOUR, linewidth=0.7, alpha=0.4)
            est, est_vis = project(rotation[None], kp.t_hat[i][None], self.geometry)
            _draw_wireframe(ax, est[0], est_vis[0], edges, color=SET_COLOUR, linewidth=1.8)
        truth, truth_vis = project(quat_to_matrix(kp.q_gt[i][None], self.order), kp.t_gt[i][None],
                                   self.geometry)  # fmt: skip
        _draw_wireframe(ax, truth[0], truth_vis[0], edges, color=TRUTH_COLOUR, linewidth=1.4,
                        linestyle="--")  # fmt: skip
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_SECONDARY)


STAMP = {
    "covered": ("✓ COVERED", "answered; the true pose is inside the set"),
    "missed": ("✗ MISSED: silent failure", "answered; the true pose is outside the set"),
    "no_pose": ("○ NO POSE: abstained", "PnP failed: no estimate, no set"),
}


def _dark_figure(size: tuple[float, float]) -> plt.Figure:
    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    return fig


def _render(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _save_gif(frames: list[np.ndarray], durations: list[int], args: argparse.Namespace,
              name: str) -> Path:  # fmt: skip
    from PIL import Image

    images = [Image.fromarray(f).quantize(colors=GIF_COLOURS, dither=Image.Dither.NONE)
              for f in frames]  # fmt: skip
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / name
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True)  # fmt: skip
    return out


def _outcome_bar(ax: plt.Axes, parts: dict[str, int], n: int, highlight: str) -> None:
    """One stacked bar of every evaluated frame by outcome (dark surface), `highlight` opaque."""
    colours = {"covered": SET_COLOUR, "no_pose": NO_POSE_DARK, "missed": MISSED_DARK}
    left = 0.0
    for name in OUTCOMES:
        width = parts[name] / n
        ax.barh(0, width, left=left, height=0.8, color=colours[name], edgecolor=SURFACE,
                linewidth=2, hatch="///" if name == "missed" else None,
                alpha=1.0 if name == highlight else 0.35)  # fmt: skip
        if width >= 0.08:
            ax.text(left + width / 2, 0, f"{width:.2f}", ha="center", va="center", fontsize=8,
                    color=TEXT_PRIMARY)  # fmt: skip
        left += width
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axvline(0.9, color=TEXT_PRIMARY, linestyle="--", linewidth=1.0)
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Frames per outcome in proportion to `counts` (largest remainder), at least one each."""
    present = [k for k in OUTCOMES if counts[k] > 0]
    alloc = {k: 1 if k in present else 0 for k in OUTCOMES}
    n = sum(counts.values())
    spare = total - len(present)
    exact = {k: spare * counts[k] / n for k in present}
    for k in present:
        alloc[k] += int(exact[k])
    order = sorted(present, key=lambda k: exact[k] - int(exact[k]), reverse=True)
    for k in order[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def shift_domain_tour(args: argparse.Namespace) -> Path:
    """GIF: the synthetic-calibrated A1 set (alpha 0.10) on seeded frames of all three domains."""
    alpha = 0.1
    frames, durations = [], []
    domains = [_Domain(args, domain) for domain in DOMAINS]
    total = TOUR_PER_DOMAIN * len(domains)
    for d, dom in enumerate(domains):
        row = dom.row(alpha)
        q = row["quantile"]
        parts = _outcomes(row)
        outcome = np.array([dom.outcome(i, q) for i in range(len(dom.frames))])
        alloc = _allocate(parts, TOUR_PER_DOMAIN)
        rng = np.random.default_rng(args.seed + d)
        picks = [(name, int(i)) for name in OUTCOMES
                 for i in np.sort(rng.choice(np.flatnonzero(outcome == name), size=alloc[name],
                                             replace=False))]  # fmt: skip
        status = ("the marginal ≥ 0.90 guarantee applies (exchangeable with val_cal)"
                  if dom.domain == "synthetic" else "HIL: measured, not guaranteed")  # fmt: skip
        for name, i in picks:
            fig = _dark_figure((11.0, 6.0))
            ax = fig.add_axes((0.01, 0.1, 0.5, 0.82))
            dom.draw(ax, i, q, pad=args.pad, seed=args.seed + i)
            title, detail = STAMP[name]
            score = dom.scores[i]
            fig.text(0.54, 0.86, DOMAIN_LABEL[dom.domain], fontsize=17, color=TEXT_PRIMARY,
                     weight="bold")  # fmt: skip
            fig.text(0.54, 0.79, str(dom.frames.kp.filenames[i]), fontsize=10,
                     color=TEXT_SECONDARY)  # fmt: skip
            fig.text(0.54, 0.69, title, fontsize=16, weight="bold",
                     color=TEXT_PRIMARY if name != "missed" else MISSED_DARK)  # fmt: skip
            fig.text(0.54, 0.64, detail, fontsize=10, color=TEXT_SECONDARY)
            if name != "no_pose":
                fig.text(0.54, 0.59, f"A1 score {score:.2f}  vs  q = {q:.2f}", fontsize=10,
                         color=TEXT_SECONDARY)  # fmt: skip
            fig.text(0.54, 0.51, "Over all frames of this domain (A1, split, α = 0.10, "
                     "abstain_allowed):", fontsize=9.5, color=TEXT_PRIMARY)  # fmt: skip
            bar = fig.add_axes((0.54, 0.43, 0.42, 0.06))
            _outcome_bar(bar, parts, row["n_total"], name)
            lo, hi = row["coverage_ci95"]
            fig.text(0.54, 0.38, f"coverage {row['coverage']:.3f} [{lo:.3f}, {hi:.3f}] · "
                     f"n = {row['n_total']:,} · answer rate {row['answer_rate']:.3f}",
                     fontsize=9.5, color=TEXT_PRIMARY)  # fmt: skip
            fig.text(0.54, 0.34, f"silent-failure rate {row['silent_failure_rate']:.3f}\n{status}",
                     fontsize=9, color=TEXT_SECONDARY, va="top")  # fmt: skip
            fig.text(0.54, 0.25,
                     "blue = estimate + set boundary (one synthetic q, every domain)\n"
                     "orange dashed = true pose   ·   bar: covered | no pose | missed (hatched)\n"
                     "dashed line = nominal 0.90",
                     fontsize=8.5, color=TEXT_SECONDARY, va="top")  # fmt: skip
            fig.text(0.54, 0.1,
                     f"Seeded frames: {TOUR_PER_DOMAIN} per domain, split across outcomes in "
                     "proportion to the\nmeasured rates (at least one each). Not a coverage "
                     f"sample.   {len(frames) + 1}/{total}",
                     fontsize=8, color=TEXT_SECONDARY)  # fmt: skip
            fig.text(0.01, 0.03, f"{dom.run}, predicted_crop · {LICENCE}", fontsize=8,
                     color=TEXT_SECONDARY)  # fmt: skip
            frames.append(_render_gif(fig))
            durations.append(TOUR_FRAME_MS)
    return _save_gif(frames, durations, args, "shift_domain_tour.gif")


def _render_gif(fig: plt.Figure) -> np.ndarray:
    fig.set_dpi(GIF_DPI)
    return _render(fig)


def alpha_sweep(args: argparse.Namespace) -> Path:
    """GIF: alpha from 0.50 to 0.01; the A1 set grows on two frames while measured coverage
    tracks nominal on synthetic and falls short on lightbox (split, abstain_allowed)."""
    domains = [_Domain(args, domain) for domain in SWEEP_DOMAINS]
    rows = {dom.domain: [dom.row(a) for a in SWEEP_ALPHAS] for dom in domains}
    q = np.array([r["quantile"] for r in rows[SWEEP_DOMAINS[0]]])
    for dom in domains:
        if not np.allclose([r["quantile"] for r in rows[dom.domain]], q, rtol=0, atol=0):
            raise ValueError("split q differs between domains: not one synthetic calibration")
    # One answered frame per domain whose score lies inside the sweep's q range, so its stamp
    # changes during the sweep (seeded; an illustration, not a sample).
    picks = []
    for d, dom in enumerate(domains):
        s = dom.scores
        band = dom.frames.kp.valid & (s > q[0]) & (s <= q[-1])
        if not band.any():
            raise ValueError(f"no {dom.domain} frame inside the q range")
        picks.append(int(np.random.default_rng(args.seed + d).choice(np.flatnonzero(band))))
    frames, durations = [], []
    for a, alpha in enumerate(SWEEP_ALPHAS):
        fig = _dark_figure((13.0, 5.6))
        for d, (dom, i) in enumerate(zip(domains, picks, strict=True)):
            ax = fig.add_axes((0.005 + d * 0.255, 0.17, 0.245, 0.64))
            dom.draw(ax, i, q[a], pad=args.pad, seed=args.seed + i)
            name = dom.outcome(i, q[a])
            ax.set_title(f"{DOMAIN_LABEL[dom.domain]}\n{STAMP[name][0]}", fontsize=10.5,
                         color=TEXT_PRIMARY if name != "missed" else MISSED_DARK)  # fmt: skip
        chart = fig.add_axes((0.585, 0.2, 0.4, 0.61))
        chart.set_facecolor(SURFACE)
        chart.plot([0.45, 1], [0.45, 1], color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
        nominal = np.array([1 - x for x in SWEEP_ALPHAS])
        for d, dom in enumerate(domains):
            colour = SWEEP_COLOURS[d]
            cov = np.array([r["coverage"] for r in rows[dom.domain]])
            lo = np.array([r["coverage_ci95"][0] for r in rows[dom.domain]])
            hi = np.array([r["coverage_ci95"][1] for r in rows[dom.domain]])
            chart.fill_between(nominal, lo, hi, color=colour, alpha=0.25, linewidth=0)
            chart.plot(nominal, cov, color=colour, linewidth=1.6, marker=MARKERS[d], markersize=5,
                       label=f"{DOMAIN_LABEL[dom.domain]} (n = {rows[dom.domain][0]['n_total']:,}, "
                             f"answer rate {rows[dom.domain][0]['answer_rate']:.3f})")  # fmt: skip
            chart.plot(nominal[a], cov[a], marker=MARKERS[d], markersize=12, color=colour,
                       markeredgecolor=TEXT_PRIMARY, markeredgewidth=1.5, zorder=5)  # fmt: skip
            chart.annotate(f"{cov[a]:.3f}", (nominal[a], cov[a]), textcoords="offset points",
                           xytext=(10 if nominal[a] < 0.95 else -44, -16 if d else 8), fontsize=9.5, color=TEXT_PRIMARY)  # fmt: skip
        chart.axvline(nominal[a], color=TEXT_SECONDARY, linewidth=0.8)
        chart.set_xlim(0.45, 1.0)
        chart.set_ylim(0.3, 1.02)
        chart.grid(True, color="#3a3a38", linewidth=0.8)
        for side in ("top", "right"):
            chart.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            chart.spines[side].set_color(TEXT_SECONDARY)
        chart.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
        chart.set_xlabel("nominal coverage 1 − α", color=TEXT_PRIMARY)
        chart.set_ylabel("measured coverage (95 % CI band)", color=TEXT_PRIMARY)
        legend = chart.legend(frameon=False, fontsize=8, loc="upper left")
        for text in legend.get_texts():
            text.set_color(TEXT_PRIMARY)
        radius = math.degrees(q[a] * domains[0].c_r)
        fig.text(0.01, 0.93,
                 f"α = {alpha:g}   q = {q[a]:.2f}   set = {radius:.2f}° rotation × "
                 f"{100 * q[a] * domains[0].c_t:.2f} % of ‖t̂‖ translation (A1, split, "
                 "abstain_allowed, predicted_crop)",
                 fontsize=12, color=TEXT_PRIMARY, weight="bold")  # fmt: skip
        fig.text(0.01, 0.88,
                 "The same synthetic-calibrated set on both frames. Blue: estimate + sampled set "
                 "boundary; orange dashed: true pose.",
                 fontsize=8.5, color=TEXT_SECONDARY)  # fmt: skip
        fig.text(0.01, 0.05,
                 "Frames are seeded picks whose score lies in the sweep's q range: an "
                 "illustration, not a sample.",
                 fontsize=8, color=TEXT_SECONDARY)  # fmt: skip
        fig.text(0.01, 0.015,
                 "Synthetic val_test: the marginal ≥ 1 − α guarantee applies (exchangeable with "
                 "val_cal). lightbox poolB: measured, not guaranteed; abstentions count as "
                 f"covered.   {LICENCE}",
                 fontsize=8, color=TEXT_SECONDARY)  # fmt: skip
        frames.append(_render_gif(fig))
        durations.append(SWEEP_LAST_MS if a == len(SWEEP_ALPHAS) - 1 else SWEEP_FRAME_MS)
    return _save_gif(frames, durations, args, "alpha_sweep.gif")


FIGURES: dict[str, Callable[[argparse.Namespace], Path]] = {
    "level_b_overlay": level_b_overlay,
    "shift_coverage_vs_nominal": shift_coverage_vs_nominal,
    "shift_silent_failure": shift_silent_failure,
    "shift_few_label_recovery": shift_few_label_recovery,
    "shift_size_vs_coverage": shift_size_vs_coverage,
    "shift_gallery": shift_gallery,
    "headline_outcomes": headline_outcomes,
    "shift_domain_tour": shift_domain_tour,
    "alpha_sweep": alpha_sweep,
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
