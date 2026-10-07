"""Split manifests: the rule, the committed files, and byte-identical regeneration."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

from poseconf.data.splits import (
    DOMAIN_KEYS,
    HIL_DOMAINS,
    MANIFEST_NAMES,
    SUMS_FILENAME,
    assign_splits,
    domain_fractions,
    load_split,
    manifest_sha256,
    manifest_text,
    split_sizes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "splits"
CONFIG = yaml.safe_load((REPO_ROOT / "configs" / "conformal.yaml").read_text(encoding="utf-8"))

#: IMPLEMENTATION_PLAN.md section 1.2 sizes.
EXPECTED_SIZES = {
    "synthetic_val_tune": 2399,
    "synthetic_val_cal": 4798,
    "synthetic_val_test": 4797,
    "lightbox_poolA": 3370,
    "lightbox_poolB": 3370,
    "sunlamp_poolA": 1395,
    "sunlamp_poolB": 1396,
}

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import make_splits  # noqa: E402

# --- the rule ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "fractions", "expected"),
    [
        (11994, [0.2, 0.4, 0.4], [2399, 4798, 4797]),
        (6740, [0.5, 0.5], [3370, 3370]),
        (2791, [0.5, 0.5], [1395, 1396]),  # 1395.5 rounds half-down
        (10, [0.25, 0.75], [2, 8]),  # 2.5 rounds half-down
        (0, [0.5, 0.5], [0, 0]),
    ],
)
def test_split_sizes(n: int, fractions: list[float], expected: list[int]) -> None:
    assert split_sizes(n, fractions) == expected


@pytest.mark.parametrize("fractions", [[0.2, 0.4], [0.5, 0.6], [], [0.0, 1.0], [1.2, -0.2]])
def test_split_sizes_rejects_bad_fractions(fractions: list[float]) -> None:
    with pytest.raises(ValueError):
        split_sizes(100, fractions)


def test_assign_is_deterministic_disjoint_and_complete() -> None:
    names = [f"img{i:06d}.jpg" for i in range(1, 1001)]
    fractions = {"a": 0.2, "b": 0.4, "c": 0.4}
    first = assign_splits(names, fractions, seed=20261006, key=0)
    again = assign_splits(list(reversed(names)), fractions, seed=20261006, key=0)
    assert first == again  # input order does not matter
    assert [len(v) for v in first.values()] == [200, 400, 400]
    union = [n for v in first.values() for n in v]
    assert sorted(union) == names and len(set(union)) == len(names)
    assert all(v == sorted(v) for v in first.values())
    # Domain keys are separate streams; so are seeds.
    assert assign_splits(names, fractions, seed=20261006, key=1) != first
    assert assign_splits(names, fractions, seed=1, key=0) != first


def test_assign_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        assign_splits(["a", "a", "b"], {"x": 0.5, "y": 0.5}, seed=0, key=0)


def test_manifest_text_is_canonical() -> None:
    assert manifest_text(["b", "a"]) == "a\nb\n"
    with pytest.raises(ValueError):
        manifest_text(["a ", "b"])


def test_domain_keys_are_pinned() -> None:
    # Renumbering a key would re-split a domain.
    assert DOMAIN_KEYS == {"synthetic": 0, "lightbox": 1, "sunlamp": 2}


# --- the committed manifests --------------------------------------------------------------------


def test_committed_manifests_sizes_and_roles() -> None:
    loaded = {name: load_split(name) for name in MANIFEST_NAMES}
    assert {k: len(v) for k, v in loaded.items()} == EXPECTED_SIZES
    synthetic = [loaded[f"synthetic_{s}"] for s in ("val_tune", "val_cal", "val_test")]
    union = set().union(*map(set, synthetic))
    assert len(union) == sum(map(len, synthetic)) == 11994  # pairwise disjoint
    for domain in HIL_DOMAINS:
        a, b = set(loaded[f"{domain}_poolA"]), set(loaded[f"{domain}_poolB"])
        assert not a & b
    assert len(manifest_sha256()) == 64


def test_load_split_rejects_tampering(tmp_path: Path) -> None:
    for item in SPLITS_DIR.iterdir():
        shutil.copy(item, tmp_path / item.name)
    path = tmp_path / "synthetic_val_cal.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_split("synthetic_val_cal", tmp_path)
    with pytest.raises(KeyError):
        load_split("val_cal", tmp_path)


def test_fractions_from_config() -> None:
    fractions = domain_fractions(CONFIG)
    assert set(fractions) == {"synthetic", *HIL_DOMAINS}
    assert [n for f in fractions.values() for n in f] == list(MANIFEST_NAMES)


def test_regeneration_is_byte_identical(tmp_path: Path) -> None:
    """Synthetic from P1's committed sidecar; HIL from the committed union (no dataset)."""
    args = make_splits.parse_args(["--out-dir", str(SPLITS_DIR), "--hil-names-from-manifest"])
    make_splits.build(args, tmp_path)
    for filename in [f"{n}.txt" for n in MANIFEST_NAMES] + [SUMS_FILENAME]:
        assert (tmp_path / filename).read_bytes() == (SPLITS_DIR / filename).read_bytes(), filename


# --- against the dataset (local only) ------------------------------------------------------------


@pytest.mark.dataset
def test_hil_unions_equal_image_listing(local_paths) -> None:
    from poseconf.p1_adapter import list_image_filenames

    for domain in HIL_DOMAINS:
        union = sorted(load_split(f"{domain}_poolA") + load_split(f"{domain}_poolB"))
        assert union == list_image_filenames(domain, paths=local_paths)


@pytest.mark.dataset
def test_synthetic_union_equals_validation_labels(local_paths) -> None:
    from poseconf.p1_adapter import load_synthetic_labels

    labels = load_synthetic_labels("validation", paths=local_paths)
    union = sorted(
        load_split("synthetic_val_tune")
        + load_split("synthetic_val_cal")
        + load_split("synthetic_val_test")
    )
    assert union == sorted(labels.filenames.tolist())


@pytest.mark.dataset
def test_make_splits_check_mode_against_dataset() -> None:
    assert make_splits.main(["--check"]) == 0
