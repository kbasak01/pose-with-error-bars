"""Build the deployable `ConformalPoseHead` calibration artifacts from committed result rows.

One artifact per (score, convention) at the headline alpha, under `results/calibration/`. Nothing is
recomputed: each quantile, n_cal and normaliser is copied from the committed Level A / B / C result
the coverage numbers were reported from (CLAUDE.md invariant 8), and the artifact is locked to the
P1 (and, for C1/C2, variance-head) checkpoint SHA-256 that result records.

`--verify` (needs the gitignored dumps) replays `val_test` frame by frame through
`ConformalPoseHead.predict`, scoring each `Answer`/`Abstain` against the labels, and requires the
replayed n_covered and n_answered to equal the committed row exactly. It writes
`results/calibration/index.json`. A3 and the other pose scores replay from P1's sidecar (what Level A
scored); B1/C1/C2 from the dump, with C1's covariance mapped crop -> full frame by the deployment
path (`propagate.crop_cov_to_full`), not read back from the dump's full-frame column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from poseconf.conformal.head import (
    Abstain,
    ConformalPoseHead,
    FrameEstimate,
    build_artifact,
)
from poseconf.conformal.propagate import crop_cov_to_full
from poseconf.data.dump import dump_file, load_dump, load_dump_labels, variance_dump_file
from poseconf.data.splits import load_split
from poseconf.engine import level_a, level_b
from poseconf.engine.level_c import attach_variance
from poseconf.p1_adapter import (
    load_sidecar,
    load_synthetic_labels,
    p1_commit,
    p1_paths,
    pose_quaternion_order,
    projection_geometry,
)
from poseconf.provenance import (
    POSECONF_ROOT,
    RESULT_SCHEMA,
    poseconf_git_sha,
    sha256_file,
    utc_now_iso,
    write_result_json,
)

DOMAIN = "synthetic"
_POSE_FROM_SIDECAR = ("A1", "A2", "A3")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--verify", action="store_true", help="replay val_test (needs dumps)")
    return parser.parse_args(argv)


def _rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(POSECONF_ROOT))


def _row(record: dict[str, Any], score_id: str, convention: str, alpha: float) -> dict[str, Any]:
    rows = [
        r
        for r in record["rows"]
        if r["score"] == score_id and r["convention"] == convention and r["alpha"] == alpha
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {score_id}/{convention}/{alpha}, got {len(rows)}")
    return rows[0]


def _normalisers(score_id: str, record: dict[str, Any]) -> dict[str, Any]:
    if score_id not in _POSE_FROM_SIDECAR:
        return {}
    fit = record["fit"]
    if score_id == "A1":
        return {"c_R": fit["c_R_rad"], "c_t": fit["c_t"]}
    if score_id == "A2":
        return {"c_R": fit["c_R_rad"], "c_z": fit["c_z"], "c_xy": fit["c_xy"]}
    return {"c_R": fit["c_R_rad"], "c_t": fit["c_t"], "g_x": fit["g"]["x"], "g_y": fit["g"]["y"]}


def _variance_sha(record: dict[str, Any]) -> str:
    sha = record["provenance"]["variance_head_checkpoint_sha256"]
    if sha != record["variance_head"]["checkpoint_sha256"]:
        raise ValueError("level_c result disagrees with itself on the variance-head checkpoint")
    return sha


def build(config: dict[str, Any], paths: Any) -> list[tuple[Path, dict[str, Any], dict]]:
    """Write every artifact; return `(path, artifact, source row)` per artifact."""
    block = config["calibration"]
    alpha = float(block["alpha"])
    run, crop = block["run"], block["crop_source"]
    order = pose_quaternion_order(run)
    geometry = projection_geometry(run, paths=paths)
    out_dir = Path(block["out_dir"])
    git_sha = poseconf_git_sha()
    written = []
    p1_shas = set()
    for score_id in block["scores"]:
        source = Path(block["sources"][score_id])
        record = json.loads(source.read_text(encoding="utf-8"))
        if (record["run"], record["crop_source"], record["arm"], record["domain"]) != (
            run,
            crop,
            block["arm"],
            DOMAIN,
        ):
            raise ValueError(f"{source} is not the {run}/{crop}/{block['arm']}/{DOMAIN} result")
        if record["splits"]["calibration"] != block["calibration_split"]:
            raise ValueError(f"{source} was calibrated on {record['splits']['calibration']}")
        p1_shas.add(record["provenance"]["p1_checkpoint_sha256"])
        for convention in config["conventions"]:
            row = _row(record, score_id, convention, alpha)
            artifact = build_artifact(
                score_id=score_id,
                alpha=alpha,
                convention=convention,
                quantile=float(row["quantile"]),
                n_cal=int(row["n_cal"]),
                normalisers=_normalisers(score_id, record),
                quaternion_order=order,
                p1_checkpoint_sha256=record["provenance"]["p1_checkpoint_sha256"],
                variance_head_sha256=_variance_sha(record) if score_id in ("C1", "C2") else None,
                geometry=geometry if score_id == "C2" else None,
                context={
                    "run": run,
                    "crop_source": crop,
                    "arm": block["arm"],
                    "calibration_subset": block["calibration_split"],
                    "domain": DOMAIN,
                },
                provenance={
                    "source_result": _rel(source),
                    "source_result_sha256": sha256_file(source),
                    "split_manifest_sha256": record["provenance"]["split_manifest_sha256"],
                    "p1_commit": record["provenance"]["p1_commit"],
                    "poseconf_git_sha": git_sha,
                    "created_at": utc_now_iso(),
                },
            )
            path = out_dir / f"{run}_{crop}_{score_id}_{convention}_a{alpha:.2f}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", "utf-8")
            written.append((path, artifact, row))
            print(f"wrote {path}  q = {artifact['payload']['quantile']}")
    if len(p1_shas) != 1:
        raise ValueError(f"source results disagree on the P1 checkpoint: {sorted(p1_shas)}")
    return written


# --------------------------------------------------------------------------------------------------
# --verify: replay val_test through the head
# --------------------------------------------------------------------------------------------------


def _replay(head: ConformalPoseHead, estimates, covs, truth) -> tuple[int, int, dict[str, int]]:
    """(n_covered, n_answered, abstain reasons) for frame-by-frame head outputs."""
    n_cov = n_ans = 0
    reasons: dict[str, int] = {}
    for i, estimate in enumerate(estimates):
        out = head.predict(estimate, cov=None if covs is None else covs[i])
        if isinstance(out, Abstain):
            reasons[out.reason] = reasons.get(out.reason, 0) + 1
            n_cov += int(out.covered_if_abstained)
            continue
        n_ans += 1
        n_cov += int(truth(out.set, i))
    return n_cov, n_ans, reasons


def verify(config: dict[str, Any], paths: Any, written: list) -> dict[str, Any]:
    """Replay every artifact on `val_test`; the result record for `index.json`."""
    block = config["calibration"]
    run, crop = block["run"], block["crop_source"]
    names = load_split(block["test_split"])
    order = pose_quaternion_order(run)
    geometry = projection_geometry(run, paths=paths)
    k = geometry.wireframe.shape[0]

    # Pose scores: P1's sidecar, as Level A scored it.
    sidecar = load_sidecar(run, DOMAIN, crop)
    labels = load_synthetic_labels("validation", paths=paths)
    pose = level_a.join_frames(sidecar.fields, labels.filenames, labels.q, labels.t, names)
    side_index = {n: i for i, n in enumerate(sidecar.fields["filename"])}
    side_box = np.asarray(sidecar.fields["bbox_used"], dtype=np.float64)
    pose_estimates = [
        FrameEstimate(
            valid=bool(pose.valid[i]),
            q_hat=pose.q_hat[i],
            t_hat=pose.t_hat[i],
            keypoints=np.zeros((k, 2)),  # unused by A1-A3
            confidence_mean=float(pose.conf[i]),
            bbox=side_box[side_index[name]],
        )
        for i, name in enumerate(pose.filenames)
    ]

    # Keypoint scores (and C2): the dump plus the variance sidecar.
    dump_path = dump_file(paths.dumps_root, run, DOMAIN, crop, subset=False)
    dump, meta = load_dump(dump_path)
    side, side_meta = load_dump(variance_dump_file(paths.dumps_root, run, DOMAIN, crop))
    if side_meta["base_dump_sha256"] != sha256_file(dump_path):
        raise ValueError("variance sidecar was made from a different dump")
    dump = attach_variance(dump, side)
    dlabels, _ = load_dump_labels(paths.dumps_root, run, DOMAIN, "all", tag=block["arm"])
    kp, _ = level_b.join_dump(dump, dlabels, names, geometry, order, cov_key="vhead_cov_full")
    row_of = {n: i for i, n in enumerate(dump["filename"])}
    rows = np.array([row_of[n] for n in kp.filenames])
    cov_full = crop_cov_to_full(dump["vhead_cov_crop"][rows], dump["affine"][rows])
    conf_mean = np.asarray(dump["confidence"], np.float32)[rows].mean(axis=1, dtype=np.float32)
    kp_estimates = [
        FrameEstimate(
            valid=bool(kp.valid[i]),
            q_hat=kp.q_hat[i],
            t_hat=kp.t_hat[i],
            keypoints=kp.y_hat[i],
            confidence_mean=float(conf_mean[i]),
            bbox=np.asarray(dump["bbox_used"], dtype=np.float64)[rows[i]],
            unconstrained=kp.unconstrained[i],
        )
        for i in range(len(kp))
    ]

    def pose_truth(s, i):
        return s.contains(pose.q_gt[i][None], pose.t_gt[i][None])[0]

    def kp_truth(s, i):
        return s.contains(kp.y_gt[i][None], kp.include[i][None])[0]

    def c2_truth(s, i):
        return s.contains(kp.q_gt[i][None], kp.t_gt[i][None])[0]

    # The checkpoints actually on disk are hashed, as a deployment would: the sidecar's head must be
    # the file in `runs/`, and the head refuses any artifact locked to something else.
    vhead_file = Path(side_meta["variance_head"]["checkpoint"])
    vhead_sha = sha256_file(vhead_file)
    if vhead_sha != side_meta["variance_head"]["checkpoint_sha256"]:
        raise ValueError(f"{vhead_file} is not the checkpoint the variance sidecar was made with")
    p1_sha = sha256_file(paths.p1_runs / run / "best.pt")

    checks = []
    for path, artifact, row in written:
        sid = artifact["payload"]["score_id"]
        head = ConformalPoseHead.from_dict(
            artifact, p1_checkpoint_sha256=p1_sha, variance_head_sha256=vhead_sha
        )
        if sid in _POSE_FROM_SIDECAR:
            got = _replay(head, pose_estimates, None, pose_truth)
        elif sid == "C2":
            got = _replay(head, kp_estimates, cov_full, c2_truth)
        else:
            got = _replay(head, kp_estimates, cov_full if sid == "C1" else None, kp_truth)
        n_cov, n_ans, reasons = got
        match = n_cov == row["n_covered"] and n_ans == row["n_answered"]
        checks.append(
            {
                "artifact": _rel(path),
                "calibration_id": artifact["calibration_id"],
                "score": sid,
                "convention": artifact["payload"]["convention"],
                "alpha": artifact["payload"]["alpha"],
                "quantile": artifact["payload"]["quantile"],
                "n_total": len(names),
                "replay_n_covered": n_cov,
                "replay_n_answered": n_ans,
                "row_n_covered": row["n_covered"],
                "row_n_answered": row["n_answered"],
                "abstain_reasons": reasons,
                "match": match,
            }
        )
        print(
            f"{path.name}: covered {n_cov}/{row['n_covered']} answered {n_ans}/{row['n_answered']}"
        )

    return {
        "schema": RESULT_SCHEMA,
        "kind": "calibration",
        "run": run,
        "domain": DOMAIN,
        "crop_source": crop,
        "arm": block["arm"],
        "oracle": False,
        "subset": block["test_split"],
        "definition": (
            "Each artifact replayed frame by frame through ConformalPoseHead.predict on val_test; "
            "Answer scored by set.contains against the label, Abstain by covered_if_abstained "
            "(metrics.outcomes semantics). match = replay equals the committed row exactly."
        ),
        "all_match": all(c["match"] for c in checks),
        "checks": checks,
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "p1_commit": p1_commit(),
            "p1_checkpoint_sha256": p1_sha,
            "variance_head_checkpoint": str(vhead_file),
            "variance_head_checkpoint_sha256": vhead_sha,
            "dump": str(dump_path),
            "dump_sha256": sha256_file(dump_path),
            "dump_poseconf_git_sha": meta.get("poseconf_git_sha"),
            "split_manifest_sha256": written[0][1]["provenance"]["split_manifest_sha256"],
            "created_at": utc_now_iso(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Build (and optionally verify) the artifacts; exit 1 if a replay does not match."""
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = p1_paths(args.paths)
    written = build(config, paths)
    if not args.verify:
        return 0
    record = verify(config, paths, written)
    out = Path(config["calibration"]["out_dir"]) / "index.json"
    write_result_json(out, record)
    print(f"wrote {out}  all_match = {record['all_match']}")
    return 0 if record["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
