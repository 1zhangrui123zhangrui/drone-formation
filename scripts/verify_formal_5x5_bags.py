#!/usr/bin/env python3
"""Verify the formal 5-scene x 5-seed bag set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCENES = [
    "scene01_hover",
    "scene02_circle",
    "scene03_reconfig",
    "scene04_wind",
    "scene05_longtime",
]
SEEDS = [f"seed{i:02d}" for i in range(1, 6)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag-dir",
        default="data/raw_bags/v2/formal_5x5",
        help="Directory containing formal scene_seed bags.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run scripts/audit_bag_quality.py for each present bag.",
    )
    args = parser.parse_args()

    root = Path(args.bag_dir)
    expected = [root / f"{scene}_{seed}.bag" for scene in SCENES for seed in SEEDS]
    missing = [p for p in expected if not p.exists()]
    extra = sorted(p for p in root.glob("*.bag") if p not in expected)

    print(f"[verify] formal dir: {root}")
    print(f"[verify] expected: {len(expected)} bags")
    print(f"[verify] present:  {sum(p.exists() for p in expected)} bags")

    if missing:
        print("[verify] missing:")
        for p in missing:
            print(f"  - {p.name}")

    if extra:
        print("[verify] extra non-formal bags:")
        for p in extra:
            print(f"  - {p.name}")

    failures = []
    if args.audit:
        audit_script = Path("scripts/audit_bag_quality.py")
        for p in expected:
            if not p.exists():
                continue
            print(f"[verify] audit {p.name}")
            result = subprocess.run([sys.executable, str(audit_script), str(p)], check=False)
            if result.returncode != 0:
                failures.append(p.name)

    if missing or extra or failures:
        if failures:
            print("[verify] audit failures:")
            for name in failures:
                print(f"  - {name}")
        print("[verify] FAIL")
        return 1

    print("[verify] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
