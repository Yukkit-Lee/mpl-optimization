#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用途：根据迁移包中的 SHA256SUMS 校验恢复后的不可替代资产。"""

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="校验迁移资产 SHA256")
    parser.add_argument("sum_file", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    failed = 0
    for line in args.sum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, rel = line.split(None, 1)
        path = args.root / rel.strip()
        if not path.exists():
            print(f"MISSING  {path}")
            failed += 1
            continue
        actual = sha256(path)
        ok = actual == expected
        print(f"{'OK' if ok else 'FAILED'}  {path}")
        failed += int(not ok)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

