"""Shared test fixtures.

* `repo_root` -- this repository.
* `local_paths` -- the validated `configs/paths.local.yaml`. Tests using it must also carry
  `@pytest.mark.dataset`, which CI filters out; the fixture additionally skips with the reason when
  the file is absent or invalid, so a clean clone sees a skip rather than an error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATHS_FILE = REPO_ROOT / "configs" / "paths.local.yaml"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def local_paths():
    """`p1_paths()` on the real local file, or a skip naming why it is unusable."""
    from poseconf.p1_adapter import p1_paths

    try:
        return p1_paths(LOCAL_PATHS_FILE)
    except (FileNotFoundError, KeyError, ValueError) as error:
        pytest.skip(f"configs/paths.local.yaml unusable on this machine: {error}")
