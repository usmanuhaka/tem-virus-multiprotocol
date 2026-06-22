#!/usr/bin/env python3
"""Combine DenseNet201 + EfficientNetV2-S softmaxes into the selected ensemble.

Takes the val/test softmax ``.npy`` files written by ``evaluate.py`` for both
models, derives ground-truth labels from the protocol manifest (the eval loader
is unshuffled, so manifest row order matches the softmax row order), selects the
best validation strategy, applies it to the test split, prints metrics, and
saves the ensemble softmaxes.

Example
-------
    python scripts/ensemble.py --protocol A --seed 42 \
        --dense-val  outputs/densenet201_A_seed42_validation_softmax.npy \
        --dense-test outputs/densenet201_A_seed42_test_softmax.npy \
        --effnet-val  outputs/effnetv2s_A_seed42_validation_softmax.npy \
        --effnet-test outputs/effnetv2s_A_seed42_test_softmax.npy \
        --out outputs/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temvirus.config import PROTOCOL_MANIFEST  # noqa: E402
from temvirus.data import load_manifest  # noqa: E402
from temvirus.ensemble import evaluate_softmax, select_and_apply  # noqa: E402


def labels_for_split(splits_dir: Path, protocol: str, split: str) -> np.ndarray:
    df = load_manifest(splits_dir / PROTOCOL_MANIFEST[protocol])
    sub = df[df["split"] == split]
    return sub["label_id"].to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="A", choices=list(PROTOCOL_MANIFEST))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dense-val", type=Path, required=True)
    ap.add_argument("--dense-test", type=Path, required=True)
    ap.add_argument("--effnet-val", type=Path, required=True)
    ap.add_argument("--effnet-test", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    dense_val, dense_test = np.load(args.dense_val), np.load(args.dense_test)
    eff_val, eff_test = np.load(args.effnet_val), np.load(args.effnet_test)
    y_val = labels_for_split(args.splits_dir, args.protocol, "validation")
    y_test = labels_for_split(args.splits_dir, args.protocol, "test")

    for name, arr, y in [("dense_val", dense_val, y_val), ("eff_val", eff_val, y_val),
                         ("dense_test", dense_test, y_test), ("eff_test", eff_test, y_test)]:
        if arr.shape[0] != len(y):
            raise ValueError(f"{name} rows ({arr.shape[0]}) != labels ({len(y)}) for protocol {args.protocol}")

    best, ens_val, ens_test, val_results = select_and_apply(eff_val, dense_val, eff_test, dense_test, y_val)
    test_metrics = evaluate_softmax(ens_test, y_test, best)

    print(f"Protocol {args.protocol} seed {args.seed}")
    print(f"  best validation strategy : {best}")
    print(f"  ensemble TEST accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"  ensemble TEST macro-F1   : {test_metrics['macro_f1']:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"ensemble_{args.protocol}_seed{args.seed}"
    np.save(args.out / f"{stem}_val_softmax.npy", ens_val)
    np.save(args.out / f"{stem}_test_softmax.npy", ens_test)
    with open(args.out / f"{stem}_summary.json", "w") as f:
        json.dump({"protocol": args.protocol, "seed": args.seed, "best_strategy": best,
                   "test_metrics": test_metrics, "validation_strategies": val_results}, f, indent=2)
    print(f"  saved ensemble softmaxes + summary to {args.out}")


if __name__ == "__main__":
    main()
