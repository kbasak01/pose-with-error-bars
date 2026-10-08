"""Phase 9 release: the committed manifest matches the committed records (no binary needed)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from poseconf.provenance import parse_sha256sums, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE = REPO_ROOT / "results" / "release"
CALIBRATION = REPO_ROOT / "results" / "calibration"


def _script(name: str) -> ModuleType:
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def manifest() -> dict[str, str]:
    path = RELEASE / "SHA256SUMS.txt"
    if not path.is_file():
        pytest.skip("no release manifest committed yet")
    return parse_sha256sums(path.read_text("utf-8"))


def test_manifest_lists_every_calibration_file_with_its_committed_hash(manifest):
    files = sorted(CALIBRATION.glob("*.json"))
    assert len(files) == 13  # 12 artifacts + index.json
    for path in files:
        assert manifest[path.name] == sha256_file(path), path.name


def test_binary_hashes_equal_the_records_that_already_name_them(manifest):
    build = _script("build_release")
    locks = build.variance_head_locks(sorted(CALIBRATION.glob("*.json")))
    assert locks == {manifest["vhead_a2_s1337_best.pt"]}
    onnx = build.recorded_onnx_sha256(
        REPO_ROOT / "results" / "export" / "onnx_export.json", Path("vhead_a2_s1337.onnx")
    )
    assert manifest["vhead_a2_s1337.onnx"] == onnx


def test_release_record_matches_the_manifest(manifest):
    record = json.loads((RELEASE / "release.json").read_text("utf-8"))
    assert record["n_assets"] == len(manifest) == 15
    assert {k: v["sha256"] for k, v in record["assets"].items()} == manifest
    assert not record["provenance"]["poseconf_git_sha"].endswith("-dirty")


def test_manifest_text_is_sorted_sha256sum_format():
    build = _script("build_release")
    text = build.manifest_text({"b.json": "1" * 64, "a.pt": "2" * 64})
    assert text == f"{'2' * 64}  a.pt\n{'1' * 64}  b.json\n"
    assert parse_sha256sums(text) == {"a.pt": "2" * 64, "b.json": "1" * 64}
