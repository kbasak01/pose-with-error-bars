"""`/coverage-check`: repeated-split validity of one Level A score under one convention.

Reads the committed Level A result and its per-draw arrays (no recomputation), compares each alpha's
R re-split coverages with the exact atom-aware re-split law and with Beta(n+1-l, l), and writes
`results/validity/<score>_<convention>.json` (kind `validity`) plus
`assets/validity_<score>_<convention>.png`. Exits 1 if any alpha DEVIATES.

    python scripts/coverage_check.py --score A1 --convention abstain_allowed
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from poseconf.conformal.metrics import resplit_coverage_law  # noqa: E402
from poseconf.provenance import (  # noqa: E402
    RESULT_SCHEMA,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    write_result_json,
)

#: Reference palette slots 1-3 (dataviz skill; validated all-pairs) and text inks.
_BARS, _LAW, _BETA = "#2a78d6", "#eb6834", "#1baf7a"
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"
#: Panel x-range: law mean +/- this many law standard deviations.
_X_SPAN_SD = 5.0
_HIST_BINS = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score", required=True, choices=["A1", "A2", "A3"])
    parser.add_argument(
        "--convention", default="abstain_allowed", choices=["abstain_allowed", "answer_required"]
    )
    parser.add_argument(
        "--level-a",
        type=Path,
        default=Path("results/level_a/keypoint_a2_predicted_crop_synthetic.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/validity"))
    parser.add_argument("--fig-dir", type=Path, default=Path("assets"))
    return parser.parse_args(argv)


def _decode(value: Any) -> Any:
    return (
        {"inf": math.inf, "-inf": -math.inf}.get(value, value) if isinstance(value, str) else value
    )


def _panel(ax: Any, row: dict[str, Any], counts: np.ndarray) -> None:
    r = row["resplits"]
    n, m, alpha = r["n_cal"], r["n_test"], row["alpha"]
    ax.set_title(f"α = {alpha:g}  ·  {r['verdict']}", fontsize=9, color=_INK, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=7)
    if r["verdict"] == "DEGENERATE":
        q_note = "+∞" if r["frac_q_pos_inf"] == 1.0 else "−∞"
        ax.text(
            0.5,
            0.5,
            f"q = {q_note} in all {r['R']} draws\ncoverage ≡ 1, set = whole space"
            if q_note == "+∞"
            else f"q = −∞ in all {r['R']} draws\nalways abstain",
            ha="center",
            va="center",
            fontsize=8,
            color=_MUTED,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["left", "bottom"]].set_visible(False)
        return
    atoms = r["pool_failure_atom"]
    law = resplit_coverage_law(
        n, m, alpha, n_top_atom=atoms["top_+inf"], n_bottom_atom=atoms["bottom_-inf"]
    )
    lo = r["law_mean"] - _X_SPAN_SD * r["law_sd"]
    hi = r["law_mean"] + _X_SPAN_SD * r["law_sd"]
    cov = counts / m
    ax.hist(
        cov,
        bins=_HIST_BINS,
        range=(lo, hi),
        density=True,
        color=_BARS,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.6,
        label=f"R = {r['R']} re-splits",
    )
    grid = np.arange(m + 1) / m
    keep = (grid >= lo) & (grid <= hi)
    ax.plot(grid[keep], law[keep] * m, color=_LAW, linewidth=2, label="exact re-split law")
    x = np.linspace(lo, hi, 400)
    ax.plot(
        x,
        stats.beta.pdf(x, r["beta"]["a"], r["beta"]["b"]),
        color=_BETA,
        linewidth=2,
        linestyle="--",
        label="Beta(n+1−l, l)",
    )
    ax.axvline(1 - alpha, color=_MUTED, linewidth=1, linestyle=":")
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.text(
        0.02,
        0.97,
        f"KS p (law) {r['ks_law']['pvalue']:.3f}\nKS p (Beta) {r['beta']['ks_pvalue']:.1e}",
        ha="left",
        va="top",
        fontsize=7,
        color=_MUTED,
        transform=ax.transAxes,
    )


def figure(rows: list[dict[str, Any]], counts: np.ndarray, title: str, path: Path) -> None:
    """Small multiples: one panel per alpha (histogram vs both reference laws)."""
    cols = 4
    nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(12, 2.6 * nrows), squeeze=False)
    for ax, row, c in zip(axes.flat, rows, counts, strict=False):
        _panel(ax, row, c)
    for ax in list(axes.flat)[len(rows) :]:
        ax.set_visible(False)
    handles, labels = next(
        (ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0]),
        ([], []),
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(title, fontsize=10, color=_INK, x=0.01, ha="left")
    fig.supxlabel(
        "coverage on the test half (dotted line: 1 − α)", fontsize=8, color=_MUTED, y=0.075
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Write the validity JSON and figure; return 1 if any alpha deviates."""
    args = parse_args(argv)
    level_a = json.loads(args.level_a.read_text(encoding="utf-8"))
    npz_path = args.level_a.parent / level_a["resplit_protocol"]["arrays"]
    if sha256_file(npz_path) != level_a["provenance"]["resplits_npz_sha256"]:
        raise ValueError(f"{npz_path} does not match the hash recorded in {args.level_a}")
    rows = [
        row
        for row in level_a["rows"]
        if row["score"] == args.score and row["convention"] == args.convention
    ]
    with np.load(npz_path) as arrays:
        counts = arrays[f"{args.score}_{args.convention}_n_covered"]
        alphas = arrays["alphas"].tolist()
    if [row["alpha"] for row in rows] != alphas:
        raise ValueError("alpha order in the JSON rows and the npz arrays disagree")

    cases = []
    for row in rows:
        r = row["resplits"]
        cases.append(
            {
                "alpha": row["alpha"],
                "verdict": r["verdict"],
                "mean_coverage": r["mean_coverage"],
                "mc_se": r["mc_se"],
                "law_mean": r["law_mean"],
                "in_plan_band": r["in_plan_band"],
                "plan_band": r["plan_band"],
                "ks_law_pvalue": r["ks_law"]["pvalue"],
                "ks_beta_pvalue": r["beta"]["ks_pvalue"],
                "sd_ratio_observed_to_law": (
                    r["sd_coverage"] / r["law_sd"] if r["law_sd"] > 0 else None
                ),
                "frac_q_pos_inf": r["frac_q_pos_inf"],
                "frac_q_neg_inf": r["frac_q_neg_inf"],
                "fixed_split": {
                    "coverage": row["coverage"],
                    "coverage_ci95": row["coverage_ci95"],
                    "n_total": row["n_total"],
                    "answer_rate": row["answer_rate"],
                    "silent_failure_rate": row["silent_failure_rate"],
                    "quantile": _decode(row["quantile"]),
                    "set_size": row["set_size"],
                },
            }
        )
    fig_path = args.fig_dir / f"validity_{args.score}_{args.convention}.png"
    title = (
        f"{args.score}, {args.convention}, arm split — coverage over {rows[0]['resplits']['R']} "
        f"re-splits of val_cal ∪ val_test (synthetic, keypoint_a2 predicted_crop)"
    )
    figure(rows, counts, title, fig_path)
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "validity",
        "score": args.score,
        "arm": "split",
        "convention": args.convention,
        "domain": "synthetic",
        "subset": "synthetic_val_cal+synthetic_val_test re-splits",
        "crop_source": level_a["crop_source"],
        "oracle": False,
        "n_cal": rows[0]["resplits"]["n_cal"],
        "n_test": rows[0]["resplits"]["n_test"],
        "reference_law": level_a["resplit_protocol"]["reference_law"],
        "cases": cases,
        "n_valid": sum(c["verdict"] == "VALID" for c in cases),
        "n_degenerate": sum(c["verdict"] == "DEGENERATE" for c in cases),
        "n_deviates": sum(c["verdict"] == "DEVIATES" for c in cases),
        "figure": str(fig_path),
        "provenance": {
            **level_a["provenance"],
            "source_result": str(args.level_a),
            "source_result_sha256": sha256_file(args.level_a),
            "poseconf_git_sha": poseconf_git_sha(),
            "created_at": utc_now_iso(),
        },
    }
    out = write_result_json(args.out_dir / f"{args.score}_{args.convention}.json", result)
    for c in cases:
        print(
            f"{args.score} {args.convention:<16} α={c['alpha']:<5g} {c['verdict']:<10} "
            f"mean {c['mean_coverage']:.5f} ± {c['mc_se']:.5f}  law {c['law_mean']:.5f}  "
            f"KS law {c['ks_law_pvalue']:.3f}  KS Beta {c['ks_beta_pvalue']:.1e}"
        )
    print(f"wrote {out} and {fig_path}")
    return 1 if result["n_deviates"] else 0


if __name__ == "__main__":
    sys.exit(main())
