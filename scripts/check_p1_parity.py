"""Phase 3 gate: does the prediction dump reproduce P1's committed `keypoint_a2` results?

For every domain x arm in `dumps_root/<run>/`, compares the dump with the P1 sidecar arm it must
reproduce (`predicted_crop` -> `predicted_crop`, `gt_crop` -> `pipeline_gt_crop`):
success-flag agreement, label-free pose deltas, solve counts vs P1's results JSON, and |Δe_r|,
|Δe_t| where labels may be read (synthetic; HIL poolB only). On HIL poolA the gate uses the
label-free bound angle(q̂_dump, q̂_P1) ≥ |Δe_r|. Writes `results/dump_summary.json`
(kind `dump_summary`) even when the gate misses, and exits non-zero on a miss.

    python scripts/check_p1_parity.py --run keypoint_a2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from poseconf.data.dump import CROP_ARMS, dump_file, load_dump, load_dump_labels, p1_reference_arm
from poseconf.data.splits import manifest_sha256
from poseconf.engine.parity import ErrorDeltas, parity_cell
from poseconf.p1_adapter import (
    DOMAINS,
    HIL_DOMAINS,
    SIDECAR_LABEL_FREE_COLUMNS,
    load_p1_config,
    load_sidecar,
    p1_commit,
    p1_paths,
    p1_result_counts,
    pose_errors,
)
from poseconf.provenance import (
    RESULT_SCHEMA,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    write_result_json,
)

#: Dump columns the comparison reads (all label-free).
_DUMP_COLUMNS = ("filename", "success", "q_pred", "t_pred", "n_inliers", "failure_reason")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--run", default=None, help="P1 keypoint run (default: dump.run)")
    parser.add_argument("--dumps-root", type=Path, default=None, help="default: dumps_root")
    parser.add_argument("--domain", nargs="+", choices=DOMAINS, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/dump_summary.json"))
    return parser.parse_args(argv)


def _error_deltas(
    domain: str,
    arm: str,
    dump: dict[str, Any],
    *,
    run: str,
    dumps_root: Path,
) -> ErrorDeltas:
    """e_r/e_t of dump and P1 on the frames whose labels may be read: synthetic all, HIL poolB."""
    pool = "all" if domain not in HIL_DOMAINS else "poolB"
    labels, _ = load_dump_labels(dumps_root, run, domain, pool, tag=None)
    names = [str(name) for name in labels["filename"]]
    position = {str(name): row for row, name in enumerate(dump["filename"])}
    present = [name for name in names if name in position]
    label_rows = np.array([row for row, name in enumerate(names) if name in position], dtype=int)
    rows = np.array([position[name] for name in present], dtype=int)
    e_r_dump, e_t_dump = pose_errors(
        labels["q_gt"][label_rows],
        labels["t_gt"][label_rows],
        dump["q_pred"][rows],
        dump["t_pred"][rows],
    )
    sidecar = load_sidecar(
        run,
        domain,
        p1_reference_arm(arm),
        columns=["e_r", "e_t"],
        frames=present,
        tag=None,
    )
    order = {str(name): row for row, name in enumerate(sidecar.fields["filename"])}
    p1_rows = np.array([order[name] for name in present], dtype=int)
    return ErrorDeltas(
        scope=pool,
        filenames=np.array(present),
        e_r_dump=e_r_dump,
        e_t_dump=e_t_dump,
        e_r_p1=sidecar.fields["e_r"][p1_rows].astype(np.float64),
        e_t_p1=sidecar.fields["e_t"][p1_rows].astype(np.float64),
    )


def main(argv: list[str] | None = None) -> int:
    """Compare every dump arm present and write the summary."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dump_cfg = config["dump"]
    run = args.run or dump_cfg["run"]
    paths = p1_paths(args.paths)
    dumps_root = args.dumps_root or paths.dumps_root
    domains = args.domain or list(dump_cfg["domains"])

    cells = []
    dump_provenance: dict[str, Any] = {}
    missing = []
    for domain in domains:
        for arm in CROP_ARMS:
            path = dump_file(dumps_root, run, domain, arm, subset=False)
            if not path.is_file():
                missing.append(str(path))
                continue
            arrays, meta = load_dump(path)
            dump = {key: arrays[key] for key in _DUMP_COLUMNS}
            reference = p1_reference_arm(arm)
            p1 = load_sidecar(run, domain, reference, columns=list(SIDECAR_LABEL_FREE_COLUMNS))
            full = meta.get("limit") is None
            counts = p1_result_counts(run, domain, reference)
            if full and counts["count"] != len(dump["filename"]):
                raise SystemExit(
                    f"{path}: {len(dump['filename'])} frames but P1's JSON has {counts['count']}"
                )
            if full and not np.array_equal(
                np.asarray(dump["filename"]).astype(str), p1.fields["filename"].astype(str)
            ):
                raise SystemExit(f"{path}: frame order differs from P1's sidecar")
            cell = parity_cell(
                dump,
                p1.fields,
                errors=_error_deltas(domain, arm, arrays, run=run, dumps_root=dumps_root),
                p1_solved_count=int(counts["solved_count"]) if full else None,
                gate=dump_cfg["parity_gate"],
                max_listed=int(dump_cfg["max_disagreements_listed"]),
            )
            cells.append(
                {
                    "domain": domain,
                    "crop_source": arm,
                    "p1_reference_arm": reference,
                    "tag": meta["tag"],
                    "oracle": bool(meta["oracle"]),
                    "subset": f"{domain}_{meta['split']}",
                    "truncated_to": meta.get("limit"),
                    **cell,
                }
            )
            dump_provenance[f"{domain}/{arm}"] = {
                "dump_sha256": sha256_file(path),
                "dump_git_sha": meta["provenance"]["poseconf_git_sha"],
                "dump_created_at": meta["provenance"]["created_at"],
                "p1_checkpoint_sha256": meta["provenance"]["p1_checkpoint_sha256"],
                "keypoint_labels_sha256": meta["keypoint_labels_sha256"],
                "runtime": meta["runtime"],
                "info": meta["info"],
            }
            verdict = "MET" if cell["gate"]["met"] else "MISSED"
            print(
                f"{domain:<9} {arm:<14} n={cell['n_total']:>6} "
                f"agree={cell['success_flag_agreement']:.6f} "
                f"solved {cell['solved']['dump']}/{cell['solved']['p1_json']} "
                f"|de_r| med={cell['errors']['abs_de_r_rad']['median']} "
                f"bound med={cell['label_free_bound']['rotation_delta_rad']['median']}  {verdict}",
                flush=True,
            )
    if missing:
        print("missing dumps (run scripts/dump_predictions.py):\n  " + "\n  ".join(missing))
    if not cells:
        return 2

    checkpoints = {json.dumps(item["p1_checkpoint_sha256"]) for item in dump_provenance.values()}
    if len(checkpoints) != 1:
        raise SystemExit(f"dumps were made with different checkpoints: {sorted(checkpoints)}")
    all_met = all(cell["gate"]["met"] for cell in cells) and not missing
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "dump_summary",
        "run": run,
        "gate": {
            **dump_cfg["parity_gate"],
            "met": bool(all_met),
            "missing_dumps": missing,
            "hil_policy": (
                "HIL labels read for poolB only (|de_r|, |de_t|); poolA is gated by the "
                "label-free bound angle(q_dump, q_p1) >= |de_r| (CLAUDE.md invariant 4)"
            ),
        },
        "cells": cells,
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "p1_commit": p1_commit(),
            "p1_checkpoint_sha256": json.loads(checkpoints.pop()),
            "split_manifest_sha256": manifest_sha256(),
            "config_sha256": sha256_file(args.config),
            "created_at": utc_now_iso(),
            "pnp_config": load_p1_config(run)["pnp"],
            "dumps": dump_provenance,
        },
    }
    write_result_json(args.out, result)
    print(f"gate {'MET' if all_met else 'MISSED'} -> {args.out}")
    return 0 if all_met else 1


if __name__ == "__main__":
    sys.exit(main())
