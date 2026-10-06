"""Provenance helpers shared by every script that writes to `results/`.

Hashing, git SHAs, timestamps, the SHA256SUMS manifest format, and a JSON writer that keeps `inf`
as a value. See the `results-provenance` skill for the schema these feed. No torch, no speedpose.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "POSECONF_ROOT",
    "RESULT_SCHEMA",
    "match_release_entry",
    "parse_sha256sums",
    "poseconf_git_sha",
    "sha256_file",
    "utc_now_iso",
    "verify_checkpoints",
    "write_result_json",
]

#: The poseconf repository root. `src/poseconf/provenance.py` -> two up.
POSECONF_ROOT = Path(__file__).resolve().parents[2]

#: Schema tag carried by every result JSON.
RESULT_SCHEMA = "poseconf.result.v1"

#: Read size for streaming hashes; checkpoints are ~250 MB.
_HASH_CHUNK_BYTES = 1 << 20

#: Length of a hex SHA-256 digest.
_SHA256_HEX_LEN = 64


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, streamed.

    Args:
        path: File to hash.

    Returns:
        The lowercase hex digest.

    Raises:
        FileNotFoundError: If `path` is not a file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cannot hash {path}: not a file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str], cwd: Path) -> str:
    """Run a read-only git command and return its stripped stdout.

    Raises:
        RuntimeError: If git fails.
    """
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def poseconf_git_sha(root: str | Path = POSECONF_ROOT) -> str:
    """The poseconf commit, suffixed `-dirty` when tracked files have uncommitted changes.

    Untracked files and the submodule's own state do not count as dirty here; the submodule pin is
    recorded separately as `p1_commit`.

    Args:
        root: Repository root.

    Returns:
        A full SHA, possibly with `-dirty`.
    """
    root = Path(root)
    sha = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain", "--untracked-files=no", "--ignore-submodules"], root)
    return f"{sha}-dirty" if status else sha


def utc_now_iso() -> str:
    """Current UTC time, ISO-8601 with seconds precision and a `Z` suffix."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _encode_nonfinite(value: Any) -> Any:
    """Recursively replace float infinities with the strings `"inf"` / `"-inf"`.

    Raises:
        ValueError: On NaN — a NaN in a result is a bug upstream, not a value.
    """
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN in a result object; refusing to serialise it")
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {key: _encode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_encode_nonfinite(item) for item in value]
    return value


def write_result_json(path: str | Path, obj: dict[str, Any]) -> Path:
    """Write a result object as JSON, keeping infinities as `"inf"` strings.

    Args:
        path: Destination; parent directories are created.
        obj: The result mapping. Must carry `schema` == `RESULT_SCHEMA`.

    Returns:
        The written path.

    Raises:
        ValueError: If the schema tag is missing or wrong, or the object contains NaN.
    """
    if obj.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"result object must carry schema={RESULT_SCHEMA!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_encode_nonfinite(obj), indent=2, sort_keys=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse `sha256sum` output: one `<hex>  <name>` (or `<hex> *<name>`) per line.

    Args:
        text: Manifest contents. Blank lines and `#` comments are ignored.

    Returns:
        Mapping of file name to lowercase hex digest.

    Raises:
        ValueError: On a malformed line or a duplicated name.
    """
    entries: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"SHA256SUMS line {lineno} is malformed: {raw!r}")
        digest, name = parts[0].lower(), parts[1].lstrip("*").strip()
        if len(digest) != _SHA256_HEX_LEN or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"SHA256SUMS line {lineno} has no valid SHA-256: {raw!r}")
        if name in entries:
            raise ValueError(f"SHA256SUMS lists {name!r} twice")
        entries[name] = digest
    return entries


def match_release_entry(run: str, names: list[str] | set[str] | dict[str, str]) -> str:
    """Find the manifest entry for a run's `best.pt`.

    Release assets cannot contain `/`, and the exact naming of P1's release assets is not recorded
    in P1's repository, so three spellings are accepted: `<run>/best.pt`, `<run>_best.pt` and
    `<run>.pt` (a leading `./` is ignored). Anything else is refused rather than guessed.

    Args:
        run: Run name, e.g. `"keypoint_a2"`.
        names: Manifest file names.

    Returns:
        The single matching manifest name.

    Raises:
        KeyError: If no entry, or more than one entry, matches.
    """
    accepted = {f"{run}/best.pt", f"{run}_best.pt", f"{run}.pt"}
    hits = sorted(name for name in names if name.removeprefix("./") in accepted)
    if len(hits) != 1:
        problem = "no entry" if not hits else f"{len(hits)} entries ({hits})"
        raise KeyError(
            f"SHA256SUMS has {problem} for run {run!r}; accepted names are {sorted(accepted)}. "
            f"Manifest lists: {sorted(names)}"
        )
    return hits[0]


def verify_checkpoints(
    runs: list[str], runs_root: str | Path, sums_path: str | Path
) -> list[dict[str, Any]]:
    """Hash `<runs_root>/<run>/best.pt` for each run and compare with a SHA256SUMS manifest.

    Args:
        runs: Run names to check.
        runs_root: Directory holding `<run>/best.pt`.
        sums_path: The release manifest.

    Returns:
        One record per run: `run`, `path` (relative to `runs_root`, so no machine path is
        committed), `size_bytes`, `sha256_local`, `release_name`, `sha256_release`, `match`.

    Raises:
        FileNotFoundError: If the manifest or a checkpoint is missing.
        KeyError: If the manifest has no unambiguous entry for a run.
        ValueError: If `runs` is empty or the manifest is malformed.
    """
    if not runs:
        raise ValueError("no runs to verify")
    sums_path, runs_root = Path(sums_path), Path(runs_root)
    if not sums_path.is_file():
        raise FileNotFoundError(
            f"release manifest {sums_path} not found. Place the phase-9-complete SHA256SUMS.txt "
            "there, or set p1_release_sums in configs/paths.local.yaml."
        )
    manifest = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    missing = [run for run in runs if not (runs_root / run / "best.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint(s) missing under {runs_root}: {missing}")

    records = []
    for run in runs:
        checkpoint = runs_root / run / "best.pt"
        name = match_release_entry(run, manifest)
        local = sha256_file(checkpoint)
        records.append(
            {
                "run": run,
                "path": f"{run}/best.pt",
                "size_bytes": checkpoint.stat().st_size,
                "sha256_local": local,
                "release_name": name,
                "sha256_release": manifest[name],
                "match": local == manifest[name],
            }
        )
    return records
