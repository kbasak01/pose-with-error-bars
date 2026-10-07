"""A 6-frame fake SPEED+ tree plus random-init P1 checkpoints, for dataset-free dump tests.

Nothing here is SPEED+ data: the intrinsics are round numbers, the poses are drawn from a seeded
generator, and the images are black frames with P1's wireframe keypoints drawn as dots. Labels are
produced with P1's own projection (through the adapter), in the format of P1's Phase 2d label
products, so the dump reads the fixture exactly as it reads the real tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from poseconf import p1_adapter

#: Fixture image size, the SPEED+ frame size P1's configs require.
WIDTH, HEIGHT = 1920, 1200

#: Round-number pinhole intrinsics (not SPEED+'s).
FOCAL_PX = 3000.0

#: Dot radius used to draw keypoints into the fixture images.
DOT_RADIUS_PX = 6


@dataclass(frozen=True)
class TinySpeedPlus:
    """Handles to a built fixture tree.

    Attributes:
        root: Fixture root.
        paths_yaml: A `paths.local.yaml` pointing at the fixture.
        paths: The validated `P1Paths`.
        filenames: Frame filenames in split order.
    """

    root: Path
    paths_yaml: Path
    paths: p1_adapter.P1Paths
    filenames: list[str]


def _random_quaternion(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def build_tiny_speedplus(root: Path, *, n_frames: int = 6, seed: int = 0) -> TinySpeedPlus:
    """Build the fixture tree under `root` and return handles to it.

    Args:
        root: An empty directory.
        n_frames: Number of synthetic validation frames.
        seed: Seed for poses and checkpoint initialisation.

    Returns:
        The fixture handles.
    """
    rng = np.random.default_rng(seed)
    speedplus = root / "speedplus"
    images = speedplus / "synthetic" / "images"
    images.mkdir(parents=True)
    camera = {
        "Nu": WIDTH,
        "Nv": HEIGHT,
        "cameraMatrix": [[FOCAL_PX, 0.0, WIDTH / 2], [0.0, FOCAL_PX, HEIGHT / 2], [0.0, 0.0, 1.0]],
        "distCoeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    (speedplus / "camera.json").write_text(json.dumps(camera), encoding="utf-8")

    filenames = [f"img{index:06d}.jpg" for index in range(1, n_frames + 1)]
    records, keypoints, visible, boxes, ranges = [], [], [], [], []
    for name in filenames:
        q = _random_quaternion(rng)
        t = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2), rng.uniform(6.0, 10.0)])
        points, valid = p1_adapter.project_keypoints(q, t, camera_json=speedplus / "camera.json")
        inside = (
            valid
            & (points[:, 0] >= 0)
            & (points[:, 0] < WIDTH)
            & (points[:, 1] >= 0)
            & (points[:, 1] < HEIGHT)
        )
        frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for x, y in points[inside]:
            cv2.circle(frame, (round(x), round(y)), DOT_RADIUS_PX, 255, -1)
        cv2.imwrite(str(images / name), frame)
        seen = points[inside]
        boxes.append([*seen.min(axis=0), *seen.max(axis=0)])
        keypoints.append(np.where(valid[:, None], points, np.nan))
        visible.append(inside)
        ranges.append(float(np.linalg.norm(t)))
        records.append(
            {"filename": name, "q_vbs2tango_true": q.tolist(), "r_Vo2To_vbs_true": t.tolist()}
        )
    (speedplus / "synthetic" / "validation.json").write_text(json.dumps(records), encoding="utf-8")

    labels = root / "labels"
    labels.mkdir()
    np.savez_compressed(
        labels / "synthetic_validation.npz",
        filenames=np.array(filenames),
        keypoints_2d=np.asarray(keypoints, dtype=np.float32),
        visible=np.asarray(visible, dtype=bool),
        bbox_tight=np.asarray(boxes, dtype=np.float32),
        range_m=np.asarray(ranges, dtype=np.float32),
        meta=np.array(json.dumps({"fixture": True})),
    )

    runs = root / "runs"
    runs.mkdir()
    paths_yaml = root / "paths.local.yaml"
    paths_yaml.write_text(
        yaml.safe_dump(
            {
                "speedplus_root": str(speedplus),
                "p1_runs": str(runs),
                "dumps_root": str(root / "dumps"),
                "p1_release_sums": str(root / "SHA256SUMS.txt"),
                "p1_keypoint_labels": str(labels),
            }
        ),
        encoding="utf-8",
    )
    paths = p1_adapter.p1_paths(paths_yaml)
    p1_adapter.write_random_init_checkpoints(runs, paths=paths, seed=seed)
    return TinySpeedPlus(root=root, paths_yaml=paths_yaml, paths=paths, filenames=filenames)
