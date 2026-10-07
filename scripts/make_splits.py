"""Write the seeded split manifests to `splits/` (or check that regeneration is byte-identical).

Filename sources, none of which reads a label:
* synthetic validation: the `filename` column of P1's committed synthetic sidecar (no dataset);
* lightbox / sunlamp: a directory listing of `<speedplus_root>/<domain>/images/` (names only), or,
  with `--hil-names-from-manifest`, the committed poolA + poolB union (for CI, which has no dataset).
Counts are checked against P1's committed `results/dataset_audit.json`.

    python scripts/make_splits.py --config configs/conformal.yaml            # write splits/
    python scripts/make_splits.py --config configs/conformal.yaml --check    # diff only
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

from poseconf.data.splits import (
    DEFAULT_MANIFEST_DIR,
    DOMAIN_KEYS,
    HIL_DOMAINS,
    MANIFEST_NAMES,
    SUMS_FILENAME,
    assign_splits,
    domain_fractions,
    load_split,
    write_manifests,
)
from poseconf.p1_adapter import (
    dataset_audit_counts,
    list_image_filenames,
    load_sidecar,
    p1_paths,
)

#: Crop arm whose synthetic sidecar supplies the validation filenames (every arm has the same set).
_NAME_SOURCE_CROP = "predicted_crop"

#: P1 audit key per domain.
_AUDIT_KEYS = {
    "synthetic": "synthetic/validation",
    "lightbox": "lightbox/test",
    "sunlamp": "sunlamp/test",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/conformal.yaml"))
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a temp dir and compare byte-for-byte with --out-dir; write nothing",
    )
    parser.add_argument(
        "--hil-names-from-manifest",
        action="store_true",
        help="take HIL filenames from the committed poolA+poolB in --out-dir (no dataset needed)",
    )
    return parser.parse_args(argv)


def source_names(args: argparse.Namespace, run: str) -> dict[str, list[str]]:
    """Filenames per domain, checked against P1's audit counts."""
    names = {"synthetic": load_sidecar(run, "synthetic", _NAME_SOURCE_CROP).fields["filename"]}
    names["synthetic"] = [str(name) for name in names["synthetic"]]
    if args.hil_names_from_manifest:
        for domain in HIL_DOMAINS:
            names[domain] = sorted(
                load_split(f"{domain}_poolA", args.out_dir)
                + load_split(f"{domain}_poolB", args.out_dir)
            )
    else:
        local = p1_paths(args.paths)
        for domain in HIL_DOMAINS:
            names[domain] = list_image_filenames(domain, paths=local)
    counts = dataset_audit_counts()
    for domain, domain_names in names.items():
        expected = counts[_AUDIT_KEYS[domain]]
        if len(domain_names) != expected or len(set(domain_names)) != expected:
            raise ValueError(
                f"{domain}: {len(domain_names)} names ({len(set(domain_names))} unique), "
                f"P1 audit says {expected}"
            )
    return names


def build(args: argparse.Namespace, out_dir: Path) -> Path:
    """Generate every manifest into `out_dir`; return the SHA256SUMS path."""
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["splits"]["seed"])
    names = source_names(args, config["p1"]["primary_run"])
    splits: dict[str, list[str]] = {}
    for domain, fractions in domain_fractions(config).items():
        splits.update(assign_splits(names[domain], fractions, seed=seed, key=DOMAIN_KEYS[domain]))
    return write_manifests(splits, out_dir)


def main(argv: list[str] | None = None) -> int:
    """Write or check the manifests; return the process exit code."""
    args = parse_args(argv)
    if not args.check:
        sums = build(args, args.out_dir)
        for name in MANIFEST_NAMES:
            n = len(load_split(name, args.out_dir))
            print(f"{name:<22} {n:>6}")
        print(f"wrote {sums}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        build(args, Path(tmp))
        differ = [
            filename
            for filename in [f"{name}.txt" for name in MANIFEST_NAMES] + [SUMS_FILENAME]
            if not (args.out_dir / filename).is_file()
            or (Path(tmp) / filename).read_bytes() != (args.out_dir / filename).read_bytes()
        ]
    if differ:
        print(f"NOT byte-identical: {differ}")
        return 1
    print(f"byte-identical: {len(MANIFEST_NAMES)} manifests + {SUMS_FILENAME} in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
