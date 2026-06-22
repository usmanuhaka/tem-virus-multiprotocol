#!/usr/bin/env python3
"""Validate the four shipped split manifests.

Checks, per protocol: train/val/test sizes, 14-class coverage in every split,
label_id consistency with the alphabetical class order, source-aware
group-spanning leakage (B-G14 / C-G09), and exact (class, filename) overlaps
across splits. Exits non-zero if any manifest fails.

Example
-------
    python scripts/verify_splits.py --splits-dir data/splits
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temvirus.config import PROTOCOL_MANIFEST  # noqa: E402
from temvirus.splits import validate_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    args = ap.parse_args()

    all_ok = True
    for protocol, fname in PROTOCOL_MANIFEST.items():
        path = args.splits_dir / fname
        if not path.exists():
            print(f"[MISSING] {protocol}: {path}")
            all_ok = False
            continue
        rep = validate_manifest(path, protocol)
        flag = "OK " if rep["ok"] else "FAIL"
        extra = ""
        if "groups_spanning_splits" in rep:
            extra = f" | groups={rep['groups']} spanning={rep['groups_spanning_splits']}"
        print(f"[{flag}] {protocol:15s} sizes(train/val/test)={rep['sizes']} "
              f"classes={rep['n_classes']}{extra}")
        for prob in rep["problems"]:
            print(f"         - {prob}")
        all_ok = all_ok and rep["ok"]

    print("\nAll manifests valid." if all_ok else "\nSome manifests FAILED validation.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
