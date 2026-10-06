"""Checkpoint verification against the P1 release manifest.

Unit tests run on fake files (CI-safe). The real verification is `@pytest.mark.dataset` and skips
until the release `SHA256SUMS.txt` is placed at `p1_release_sums`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from poseconf.provenance import (
    RESULT_SCHEMA,
    match_release_entry,
    parse_sha256sums,
    sha256_file,
    verify_checkpoints,
    write_result_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = ["detector", "keypoint_a2"]


def _fake_runs(root: Path) -> dict[str, str]:
    digests = {}
    for run in RUNS:
        (root / run).mkdir(parents=True)
        payload = f"weights of {run}".encode()
        (root / run / "best.pt").write_bytes(payload)
        digests[run] = hashlib.sha256(payload).hexdigest()
    return digests


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    path.write_bytes(b"x" * 3_000_000)  # spans several read chunks
    assert sha256_file(path) == hashlib.sha256(b"x" * 3_000_000).hexdigest()


def test_parse_sha256sums_formats() -> None:
    a, b = "a" * 64, "B" * 64
    parsed = parse_sha256sums(f"# comment\n{a}  keypoint_a2_best.pt\n\n{b} *detector.pt\n")
    assert parsed == {"keypoint_a2_best.pt": a, "detector.pt": b.lower()}


@pytest.mark.parametrize("line", ["deadbeef  file.pt", "a" * 64, f"{'g' * 64}  file.pt"])
def test_parse_sha256sums_rejects_malformed(line: str) -> None:
    with pytest.raises(ValueError):
        parse_sha256sums(line)


def test_parse_sha256sums_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="twice"):
        parse_sha256sums(f"{'a' * 64}  x.pt\n{'b' * 64}  x.pt\n")


@pytest.mark.parametrize("name", ["keypoint_a2-best.pt", "./keypoint_a2-best.pt"])
def test_match_release_entry_accepts_release_spelling(name: str) -> None:
    names = [name, "keypoint_a2.onnx", "keypoint_a2_fp16.onnx", "keypoint_a1-best.pt"]
    assert match_release_entry("keypoint_a2", names) == name


@pytest.mark.parametrize("name", ["keypoint_a2/best.pt", "keypoint_a2_best.pt", "keypoint_a2.pt"])
def test_match_release_entry_refuses_other_spellings(name: str) -> None:
    with pytest.raises(KeyError, match="no entry"):
        match_release_entry("keypoint_a2", [name])


def test_match_release_entry_does_not_confuse_prefixes() -> None:
    with pytest.raises(KeyError, match="no entry"):
        match_release_entry("keypoint_a2", ["keypoint_a2_smoke-best.pt", "keypoint_a20-best.pt"])


def test_match_release_entry_refuses_ambiguity() -> None:
    with pytest.raises(KeyError, match="2 entries"):
        match_release_entry("keypoint_a2", ["keypoint_a2-best.pt", "./keypoint_a2-best.pt"])


def test_verify_checkpoints_all_match(tmp_path: Path) -> None:
    digests = _fake_runs(tmp_path / "runs")
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text("".join(f"{d}  {r}-best.pt\n" for r, d in digests.items()), encoding="utf-8")
    records = verify_checkpoints(RUNS, tmp_path / "runs", sums)
    assert [r["run"] for r in records] == RUNS
    assert all(r["match"] for r in records)
    assert all(r["path"] == f"{r['run']}/best.pt" for r in records)  # no machine paths


def test_verify_checkpoints_reports_mismatch(tmp_path: Path) -> None:
    digests = _fake_runs(tmp_path / "runs")
    digests["detector"] = "0" * 64
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text("".join(f"{d}  {r}-best.pt\n" for r, d in digests.items()), encoding="utf-8")
    records = {r["run"]: r["match"] for r in verify_checkpoints(RUNS, tmp_path / "runs", sums)}
    assert records == {"detector": False, "keypoint_a2": True}


def test_verify_checkpoints_missing_manifest(tmp_path: Path) -> None:
    _fake_runs(tmp_path / "runs")
    with pytest.raises(FileNotFoundError, match="release manifest"):
        verify_checkpoints(RUNS, tmp_path / "runs", tmp_path / "absent.txt")


def test_verify_checkpoints_missing_checkpoint(tmp_path: Path) -> None:
    digests = _fake_runs(tmp_path / "runs")
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text("".join(f"{d}  {r}-best.pt\n" for r, d in digests.items()), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="keypoint_a1"):
        verify_checkpoints([*RUNS, "keypoint_a1"], tmp_path / "runs", sums)


def test_write_result_json_keeps_infinity(tmp_path: Path) -> None:
    path = write_result_json(
        tmp_path / "r.json", {"schema": RESULT_SCHEMA, "q": math.inf, "s": [-math.inf, 1.0]}
    )
    assert json.loads(path.read_text()) == {"schema": RESULT_SCHEMA, "q": "inf", "s": ["-inf", 1.0]}


def test_write_result_json_refuses_nan_and_bad_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NaN"):
        write_result_json(tmp_path / "r.json", {"schema": RESULT_SCHEMA, "q": math.nan})
    with pytest.raises(ValueError, match="schema"):
        write_result_json(tmp_path / "r.json", {"q": 1.0})


@pytest.mark.dataset
def test_real_checkpoints_match_release(local_paths) -> None:
    import yaml

    if not local_paths.release_sums.is_file():
        pytest.skip(f"release SHA256SUMS.txt not yet placed at {local_paths.release_sums}")
    config = yaml.safe_load((REPO_ROOT / "configs" / "conformal.yaml").read_text())
    runs = config["p1"]["checkpoints"]
    records = verify_checkpoints(runs, local_paths.p1_runs, local_paths.release_sums)
    assert len(records) == 4
    mismatched = [r["run"] for r in records if not r["match"]]
    assert not mismatched, f"checkpoint hash mismatch vs release: {mismatched}"
