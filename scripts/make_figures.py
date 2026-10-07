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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from poseconf.conformal.propagate import project  # noqa: E402
from poseconf.conformal.so3 import quat_to_matrix  # noqa: E402
from poseconf.data.dump import dump_file, load_dump, load_dump_labels  # noqa: E402
from poseconf.data.splits import load_split  # noqa: E402
from poseconf.engine.level_b import (  # noqa: E402
    estimate_in_purse,
    join_dump,
    keypoint_set,
    score_frames,
    unit_shapes,
)
from poseconf.p1_adapter import (  # noqa: E402
    p1_paths,
    pose_quaternion_order,
    projection_geometry,
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
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0),
    )
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    fig.suptitle(
        f"{args.score} keypoint sets, alpha = {args.alpha:g}, {args.convention}, arm split, "
        f"{level['test_split']} ({run}, {crop}). First {args.n_inside}: estimate inside "
        f"its PURSE; last {args.n_outside} (dashed frame): outside.\n{LICENCE}",
        color=TEXT_PRIMARY,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.06, wspace=0.04, hspace=0.22)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"level_b_overlay_{args.score}.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return out


FIGURES: dict[str, Callable[[argparse.Namespace], Path]] = {"level_b_overlay": level_b_overlay}


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
