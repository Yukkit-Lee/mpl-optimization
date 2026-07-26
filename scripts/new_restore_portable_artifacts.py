#!/usr/bin/env python3
"""Restore Git-tracked Stage-B selected initializers to runtime paths."""

import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = ROOT / "portable_artifacts/stage_b_selected"
DESTINATION = ROOT / "new_init_model/new_public_mtt_mainline/stage_b"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["records"]:
        source = SOURCE / record["relative_path"]
        if sha256_file(source) != record["sha256"]:
            raise RuntimeError("Portable initializer SHA mismatch: {}".format(source))
        destination = DESTINATION / record["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(destination))
    print("Restored {} selected initializers to {}".format(
        len(manifest["records"]), DESTINATION
    ))


if __name__ == "__main__":
    main()

