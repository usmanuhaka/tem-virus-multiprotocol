"""Protocol definitions and manifest validation.

The four protocols are produced in notebook 01 and *shipped as the manifests*
under ``data/splits/``, those CSVs are the canonical, version-controlled split
definitions. This module documents each protocol and validates a manifest for
the leakage properties it should satisfy (it does not re-derive splits from the
raw dataset).

Protocols
---------
- ``A``              : the dataset's original crop-level train/val/test split.
- ``B_G14``          : source-aware split grouping all crops from the same RAW
                       source image (``G14_RAWSource``) into one split.
- ``A_clean_strict`` : Protocol A with the 8 visually validated exact-duplicate
                       (Hamming-0) crop pairs removed from training (near-dups kept).
- ``C_G09``          : source-aware split grouped by acquisition date +
                       magnification (``G09_Date_Magnification``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import CLASS_NAMES
from .data import load_manifest

PROTOCOL_GROUP_KEY = {
    "A": None,
    "B_G14": "G14_RAWSource",
    "A_clean_strict": None,
    "C_G09": "G09_Date_Magnification",
}

EXPECTED_SIZES = {  # (train, validation, test) from final_split_size_summary
    "A": (5740, 2249, 1900),
    "B_G14": (5735, 2251, 1903),
    "A_clean_strict": (5732, 2249, 1900),
    "C_G09": (5640, 2312, 1937),
}


def validate_manifest(path: str | Path, protocol: str) -> Dict[str, object]:
    """Validate one manifest. Returns a report dict; ``ok`` is the overall flag."""
    df = load_manifest(path)
    report: Dict[str, object] = {"protocol": protocol, "path": str(path), "problems": []}
    problems: List[str] = report["problems"]  # type: ignore[assignment]

    # split sizes
    counts = df["split"].value_counts().to_dict()
    sizes = (counts.get("train", 0), counts.get("validation", 0), counts.get("test", 0))
    report["sizes"] = sizes
    if protocol in EXPECTED_SIZES and sizes != EXPECTED_SIZES[protocol]:
        problems.append(f"split sizes {sizes} != expected {EXPECTED_SIZES[protocol]}")

    # class coverage (14 classes present in every split)
    classes = sorted(df["class_name"].unique().tolist())
    report["n_classes"] = len(classes)
    if classes != sorted(CLASS_NAMES):
        problems.append(f"class set mismatch: {classes}")
    for sp in ("train", "validation", "test"):
        present = df[df["split"] == sp]["class_name"].nunique()
        if present != len(CLASS_NAMES):
            problems.append(f"split '{sp}' covers {present}/{len(CLASS_NAMES)} classes")

    # label_id consistency with alphabetical class order
    idx = {c: i for i, c in enumerate(sorted(CLASS_NAMES))}
    bad = df[df.apply(lambda r: idx.get(r["class_name"]) != int(r["label_id"]), axis=1)]
    if len(bad):
        problems.append(f"{len(bad)} rows have label_id inconsistent with alphabetical order")

    # source-aware leakage: a group must not span splits
    key = PROTOCOL_GROUP_KEY.get(protocol)
    if key and key in df.columns:
        spanning = (df.groupby(key)["split"].nunique() > 1).sum()
        report["groups"] = int(df[key].nunique())
        report["groups_spanning_splits"] = int(spanning)
        if spanning > 0:
            problems.append(f"{spanning} '{key}' groups span more than one split (leakage)")

    # crop-level leakage: the same physical crop (unique full filepath) must not
    # appear in more than one split. (Note: bare filenames repeat across source
    # images, so they are NOT a valid uniqueness key, filepath is.)
    dup = int((df.groupby("filepath")["split"].nunique() > 1).sum())
    report["filepath_split_overlaps"] = dup
    if dup > 0:
        problems.append(f"{dup} filepaths appear in more than one split (crop-level leakage)")

    report["ok"] = len(problems) == 0
    return report
