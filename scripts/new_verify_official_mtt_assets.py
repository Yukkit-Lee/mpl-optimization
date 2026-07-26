#!/usr/bin/env python3
"""Verify raw dataset archives and official non-ZCA MTT tensors by SHA256."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "migration/new_official_data_manifest.json"
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("NEW_DATA_ROOT", root.parent / "new_data"))
    )
    parser.add_argument(
        "--official-root", type=Path,
        default=Path(os.environ.get(
            "NEW_OFFICIAL_MTT_ROOT",
            Path(os.environ.get("NEW_DATA_ROOT", root.parent / "new_data"))
            / "mtt_data/georgecazenavette",
        ))
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for record in manifest["official_mtt_files"]:
        path = args.official_root / record["relative_path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            failures.append(str(path))
    for record in manifest["raw_archives"]:
        path = args.data_root / record["autodl_relative_path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            failures.append(str(path))
    if failures:
        print("Verification failed for:")
        for path in failures:
            print("  {}".format(path))
        return 1
    print("Verified {} official MTT files and {} raw archives".format(
        len(manifest["official_mtt_files"]), len(manifest["raw_archives"])
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

