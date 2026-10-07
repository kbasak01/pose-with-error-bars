"""`results/TABLES.md` from the committed shift-matrix JSON (`results/shift/`). Never typed numbers.

    python scripts/make_tables.py [--shift-dir results/shift] [--out results/TABLES.md]

Layout (shift-study-runner contract): one section per convention; the headline alpha first, then the
alpha grid; non-oracle rows first and oracle rows (HIL `gt_crop`, `oracle_target_labels_n*`) in their
own headed sub-tables; a few-label recovery table; weighted-CP rows with classifier AUC and ESS.
Every `—` carries a footnote, and infinity is printed as ∞.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_ORDER = ("synthetic", "lightbox", "sunlamp")
CROP_ORDER = ("predicted_crop", "gt_crop")
NON_ORACLE_ARMS = ("split", "mondrian", "weighted_unlabeled_target")
DASH = "—"
FOOTNOTES = {
    "no_answer": "no frame answered (q = −∞ or no point estimate), so there is no set to measure",
    "not_score": "not defined for this score (pose radii for A/C2, keypoint radii for B/C1)",
    "not_weighted": "classifier AUC / ESS apply to the weighted arm only",
    "no_iou": "gt_crop: the crop box is the GT box, so IoU is 1 by construction",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shift-dir", type=Path, default=ROOT / "results" / "shift")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "TABLES.md")
    return parser.parse_args(argv)


def _decode(value: Any) -> Any:
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def load(shift_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """`(index, files)`; files are the decoded result objects the index lists."""
    index = _decode(json.loads((shift_dir / "index.json").read_text("utf-8")))
    files = [
        _decode(json.loads((shift_dir / cell["file"]).read_text("utf-8")))
        for cell in index["cells"]
        if cell["file"]
    ]
    return index, files


def _num(x: float | None, digits: int) -> str:
    if x is None:
        return DASH
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    return f"{x:.{digits}f}"


class Notes:
    """Collects the footnotes a table uses, so each `—` points at its reason."""

    def __init__(self) -> None:
        self.used: list[str] = []

    def dash(self, key: str) -> str:
        """A dash with the footnote marker for `key`."""
        if key not in self.used:
            self.used.append(key)
        return f"{DASH}[^{key}]"

    def render(self) -> list[str]:
        """Footnote definitions."""
        return [f"[^{key}]: {FOOTNOTES[key]}" for key in self.used]


def _size(row: dict[str, Any], key: str, stat: str, notes: Notes, scale: float = 1.0) -> str:
    size = row.get("set_size")
    if size is None:
        return notes.dash("no_answer")
    if key not in size:
        return notes.dash("not_score")
    value = size[key][stat]
    return _num(value * scale if not math.isinf(value) else value, 2)


def _trans(row: dict[str, Any], stat: str, notes: Notes) -> str:
    size = row.get("set_size")
    if size is None:
        return notes.dash("no_answer")
    if "trans_frac" in size:
        return _size(row, "trans_frac", stat, notes, 100.0)
    if "boresight_frac" in size:
        z = _size(row, "boresight_frac", stat, notes, 100.0)
        xy = _size(row, "lateral_frac", stat, notes, 100.0)
        return f"z {z} / xy {xy}"
    return notes.dash("not_score")


def _cov(row: dict[str, Any]) -> str:
    lo, hi = row["coverage_ci95"]
    return f"{_num(row['coverage'], 4)} [{_num(lo, 4)}, {_num(hi, 4)}]"


def _cga(row: dict[str, Any], notes: Notes) -> str:
    value = row.get("coverage_given_answered")
    return notes.dash("no_answer") if value is None else _num(value, 3)


def _vacuous(row: dict[str, Any], notes: Notes) -> str:
    return str(row["n_vacuous"]) if "n_vacuous" in row else notes.dash("not_score")


def _arm_rank(arm: str) -> tuple[int, int]:
    if arm in NON_ORACLE_ARMS:
        return NON_ORACLE_ARMS.index(arm), 0
    return len(NON_ORACLE_ARMS), int(arm.rsplit("_n", 1)[-1]) if "_n" in arm else 0


def _key(row: dict[str, Any]) -> tuple:
    return (
        row["score"],
        DOMAIN_ORDER.index(row["domain"]),
        CROP_ORDER.index(row["crop_source"]),
        _arm_rank(row["arm"]),
    )


def _infinite_set(row: dict[str, Any]) -> bool:
    """Whether the median answered set is unbounded (q = +inf, or an unconstrained keypoint)."""
    size = row.get("set_size") or {}
    return any(isinstance(v, dict) and math.isinf(v.get("median", 0.0)) for v in size.values())


HEADER = (
    "| score | domain | crop | arm | n answered / total | coverage [95 % CI] | coverage given "
    "answered | answer rate | silent-failure rate | rot radius median / p90 (°) | trans radius "
    "median / p90 (% of ‖t̂‖) | kp radius median (px) | n_vacuous |"
)
SEP = "|" + "---|" * 13


def _row_line(row: dict[str, Any], notes: Notes) -> str:
    cells = [
        row["score"],
        row["domain"],
        row["crop_source"],
        f"`{row['arm']}`",
        f"{row['n_answered']} / {row['n_total']}",
        _cov(row),
        _cga(row, notes),
        _num(row["answer_rate"], 3),
        _num(row["silent_failure_rate"], 3),
        f"{_size(row, 'rot_deg', 'median', notes)} / {_size(row, 'rot_deg', 'p90', notes)}",
        f"{_trans(row, 'median', notes)} / {_trans(row, 'p90', notes)}",
        _size(row, "kp_px", "median", notes),
        _vacuous(row, notes),
    ]
    return "| " + " | ".join(cells) + " |"


def _is_oracle(row: dict[str, Any]) -> bool:
    return bool(row["oracle"])


def _headline(rows: list[dict[str, Any]], convention: str, alpha: float, notes: Notes) -> list[str]:
    sel = [r for r in rows if r["convention"] == convention and r["alpha"] == alpha]
    non = sorted([r for r in sel if not _is_oracle(r)], key=_key)
    gt_hil = sorted(
        [r for r in sel if _is_oracle(r) and not r["arm"].startswith("oracle_target_labels")],
        key=_key,
    )
    out = [
        f"### Non-oracle rows — α = {alpha}, `{convention}`",
        "",
        f"Arms `split`, `mondrian`, `weighted_unlabeled_target`; calibration on synthetic "
        f"`val_cal` (n_cal = {non[0]['n_cal'] if non else DASH}); n = evaluated frames per row.",
        "",
        HEADER,
        SEP,
        *(_row_line(r, notes) for r in non),
        "",
        f"### Oracle rows — HIL `gt_crop` (crop box = GT box), α = {alpha}, `{convention}`",
        "",
        "Tagged `oracle_gt_box` in the dumps: the crop uses a test label, so these rows are not "
        "deployable numbers. Arms `split` and `mondrian`, calibrated on synthetic `gt_crop` "
        "`val_cal`.",
        "",
        HEADER,
        SEP,
        *(_row_line(r, notes) for r in gt_hil),
        "",
    ]
    return out


def _grid(
    rows: list[dict[str, Any]], convention: str, alphas: list[float], oracle: bool
) -> list[str]:
    sel = [
        r
        for r in rows
        if r["convention"] == convention
        and _is_oracle(r) == oracle
        and not r["arm"].startswith("oracle_target_labels")
    ]
    cells: dict[tuple, dict[float, dict[str, Any]]] = defaultdict(dict)
    for r in sel:
        cells[_key(r)][r["alpha"]] = r
    header = "| score | domain | crop | arm | " + " | ".join(f"α = {a}" for a in alphas) + " |"
    lines = [
        "Cell: coverage [95 % CI] · answer rate; **set ∞** marks a row whose median answered set "
        "is unbounded (the whole space covers everything: read it as no information, not as "
        "coverage).",
        "",
        header,
        "|" + "---|" * (4 + len(alphas)),
    ]
    for key in sorted(cells):
        first = next(iter(cells[key].values()))
        parts = []
        for a in alphas:
            r = cells[key][a]
            flag = " · **set ∞**" if _infinite_set(r) else ""
            parts.append(f"{_cov(r)} · ans {_num(r['answer_rate'], 3)}{flag}")
        lines.append(
            f"| {first['score']} | {first['domain']} | {first['crop_source']} | `{first['arm']}` | "
            + " | ".join(parts)
            + " |"
        )
    return lines


def _recovery(
    rows: list[dict[str, Any]], split_rows: list[dict[str, Any]], alpha: float, notes: Notes
) -> list[str]:
    sel = [r for r in rows if r["arm"].startswith("oracle_target_labels") and r["alpha"] == alpha]
    out = []
    for convention in ("abstain_allowed", "answer_required"):
        for domain in DOMAIN_ORDER[1:]:
            for crop in CROP_ORDER:
                block = [
                    r
                    for r in sel
                    if r["convention"] == convention
                    and r["domain"] == domain
                    and r["crop_source"] == crop
                ]
                if not block:
                    continue
                n_eval = block[0]["n_total"]
                out += [
                    f"#### {domain} poolB, `{crop}`, `{convention}`, α = {alpha} "
                    f"(n = {n_eval} evaluated; {block[0]['n_draws']} draws per n)",
                    "",
                    "| score | calibration | coverage mean (min–max) | envelope of per-draw 95 % "
                    "CIs | answer rate (mean) | silent-failure rate (mean) | rot radius median (°) "
                    "| kp radius median (px) |",
                    "|---|---|---|---|---|---|---|---|",
                ]
                for score in sorted({r["score"] for r in block}):
                    base = [
                        r
                        for r in split_rows
                        if r["score"] == score
                        and r["domain"] == domain
                        and r["crop_source"] == crop
                        and r["convention"] == convention
                        and r["alpha"] == alpha
                    ]
                    for b in base:
                        lo, hi = b["coverage_ci95"]
                        out.append(
                            f"| {score} | `split` (synthetic val_cal, n = {b['n_cal']}) | "
                            f"{_num(b['coverage'], 4)} | [{_num(lo, 4)}, {_num(hi, 4)}] | "
                            f"{_num(b['answer_rate'], 3)} | {_num(b['silent_failure_rate'], 3)} | "
                            f"{_size(b, 'rot_deg', 'median', notes)} | "
                            f"{_size(b, 'kp_px', 'median', notes)} |"
                        )
                    for r in sorted(
                        (r for r in block if r["score"] == score), key=lambda r: r["n_cal"]
                    ):
                        lo, hi = r["coverage_ci95"]
                        out.append(
                            f"| {score} | `{r['arm']}` | {_num(r['coverage'], 4)} "
                            f"({_num(r['coverage_min'], 4)}–{_num(r['coverage_max'], 4)}) | "
                            f"[{_num(lo, 4)}, {_num(hi, 4)}] | {_num(r['answer_rate'], 3)} | "
                            f"{_num(r['silent_failure_rate'], 3)} | "
                            f"{_size(r, 'rot_deg', 'median', notes)} | "
                            f"{_size(r, 'kp_px', 'median', notes)} |"
                        )
                out.append("")
    return out


def _classifiers(index: dict[str, Any]) -> list[str]:
    out = [
        "| domain | source → target | n_source / n_target | 5-fold CV AUC (mean ± sd) | ESS of "
        "val_cal weights (of n) | max normalised val_cal weight |",
        "|---|---|---|---|---|---|",
    ]
    for domain, c in index["classifiers"].items():
        w = c["weights"]
        out.append(
            f"| {domain} | {c['source']} → {c['target']} | {c['n_source']} / {c['n_target']} | "
            f"{_num(c['cv_auc_mean'], 4)} ± {_num(c['cv_auc_sd'], 4)} | "
            f"{_num(w['ess_cal'], 2)} (of {w['n_cal']}) | {_num(w['max_normalised_weight_cal'], 4)} |"
        )
    return out


def _slices(rows: list[dict[str, Any]], alpha: float, notes: Notes) -> list[str]:
    sel = sorted([r for r in rows if "slices" in r and r["alpha"] == alpha], key=_key)
    out = []
    for slicing in ("gt_range_tertile", "confidence_quintile", "gt_bbox_iou_bin"):
        bins = sorted({b for r in sel for b in r["slices"].get(slicing, {})}, key=int)
        out += [
            f"#### `{slicing}` — `split`, α = {alpha} (cell: coverage [95 % CI], n)",
            "",
            "| score | domain | crop | convention | " + " | ".join(f"bin {b}" for b in bins) + " |",
            "|" + "---|" * (4 + len(bins)),
        ]
        for r in sel:
            cells = []
            for b in bins:
                s = r["slices"].get(slicing, {}).get(b)
                if slicing not in r["slices"]:
                    cells.append(notes.dash("no_iou"))
                elif s is None:
                    cells.append(f"{DASH} (n = 0)")
                else:
                    lo, hi = s["coverage_ci95"]
                    cells.append(
                        f"{_num(s['coverage'], 3)} [{_num(lo, 3)}, {_num(hi, 3)}], {s['n_total']}"
                    )
            out.append(
                f"| {r['score']} | {r['domain']} | {r['crop_source']} | {r['convention']} | "
                + " | ".join(cells)
                + " |"
            )
        out.append("")
    return out


def build(shift_dir: Path) -> str:
    """The full TABLES.md text."""
    index, files = load(shift_dir)
    rows = [row for f in files for row in f["rows"]]
    alphas = [float(a) for a in index["expected"]["alphas"]]
    headline = 0.1 if 0.1 in alphas else alphas[0]
    notes = Notes()
    prov = index["provenance"]
    out = [
        "# Coverage under shift — tables",
        "",
        f"Generated by `scripts/make_tables.py` from `results/shift/*.json` (index "
        f"`results/shift/index.json`); do not edit by hand. Run `{index['run']}`, poseconf "
        f"`{prov['poseconf_git_sha'][:7]}`, P1 `{prov['p1_commit'][:7]}`, split manifest "
        f"`{prov['split_manifest_sha256'][:12]}`.",
        "",
        "**What these numbers are.** Coverage is the measured fraction of evaluated frames whose "
        "set contains the truth (abstentions count as covered under `abstain_allowed`), with a "
        "Clopper–Pearson 95 % interval. The split-conformal guarantee is marginal, finite-sample "
        "and holds only under exchangeability of calibration and evaluation frames: on synthetic "
        "`val_test`. On `lightbox_poolB` / `sunlamp_poolB` it is a measured number, not a "
        "guarantee. An ∞ set covers everything, so read coverage with the set size and answer "
        "rate beside it. `coverage given answered` is a slice reported beside the marginal "
        "coverage, not a conformal quantity. Sunlamp poolB has few answered frames; read its CIs.",
        "",
        "## Cells",
        "",
        "| domain | crop | arm | file or reason |",
        "|---|---|---|---|",
    ]
    for cell in index["cells"]:
        what = f"`{cell['file']}`" if cell["file"] else f"{DASH} {cell['reason']}"
        out.append(f"| {cell['domain']} | {cell['crop_source']} | `{cell['arm']}` | {what} |")
    out += ["", "## Domain classifier (`weighted_unlabeled_target`)", "", *_classifiers(index), ""]
    out += [
        "Features: P1's pooled 512-d encoder output from the `predicted_crop` dumps. Weighted rows "
        "use the val_cal weights above; per-row ESS and AUC repeat these values.",
        "",
    ]
    for convention in index["expected"]["conventions"]:
        out += [f"## `{convention}`", "", f"### Headline α = {headline}", ""]
        out += _headline(rows, convention, headline, notes)
    out += [
        f"## Few-label recovery (`oracle_target_labels_n*`, oracle) — α = {headline}",
        "",
        "Split CP calibrated on n labeled poolA frames (5 seeded draws per n, failures kept), "
        "evaluated on poolB. Normalisers and σ̂ are the synthetic `val_tune` fits. The `split` row "
        "is the synthetic-calibrated baseline on the same poolB frames.",
        "",
    ]
    split_rows = [r for r in rows if r["arm"] == "split"]
    out += _recovery(rows, split_rows, headline, notes)
    out += [f"## Conditional coverage slices (`split` arm, α = {headline})", ""]
    out += _slices(rows, headline, notes)
    for convention in index["expected"]["conventions"]:
        out += [f"## α grid — `{convention}`", "", "### Non-oracle rows", ""]
        out += _grid(rows, convention, alphas, oracle=False)
        out += ["", "### Oracle rows — HIL `gt_crop`", ""]
        out += _grid(rows, convention, alphas, oracle=True)
        out += [""]
    out += ["## Notes", "", *notes.render(), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Write TABLES.md."""
    args = parse_args(argv)
    if not (args.shift_dir / "index.json").is_file():
        raise FileNotFoundError(f"no {args.shift_dir / 'index.json'}; run make shift-matrix first")
    text = build(args.shift_dir)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
