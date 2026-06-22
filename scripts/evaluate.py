#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a split with 4-view TTA.

Example
-------
    python scripts/evaluate.py --checkpoint outputs/densenet201_A_seed42_best.pt \
        --model densenet201 --protocol A --split test \
        --data-root /path/to/tem_dataset --out outputs/

Prints accuracy / macro-F1 and writes the per-sample softmax (.npy) and a
predictions CSV (used downstream by the ensemble).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temvirus.config import CLASS_NAMES, PROTOCOL_MANIFEST, TrainConfig  # noqa: E402
from temvirus.data import build_loaders, load_manifest  # noqa: E402
from temvirus.engine import evaluate_with_tta, get_device  # noqa: E402
from temvirus.models import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model", default="densenet201", choices=["densenet201", "effnetv2s"])
    ap.add_argument("--protocol", default="A", choices=list(PROTOCOL_MANIFEST))
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    import torch
    cfg = TrainConfig(model=args.model, protocol=args.protocol, seed=args.seed)
    class_names = sorted(CLASS_NAMES)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    manifest = load_manifest(args.splits_dir / PROTOCOL_MANIFEST[cfg.protocol])
    train_loader, val_loader, test_loader, _ = build_loaders(manifest, class_to_idx, cfg, args.data_root)
    loader = {"validation": val_loader, "test": test_loader}[args.split]

    device = get_device()
    model = build_model(cfg.model, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)

    import torch.nn as nn
    res = evaluate_with_tta(model, loader, nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing), device)
    print(f"{args.model} {args.protocol} {args.split}: acc={res['accuracy']:.4f} "
          f"macro_f1={res['macro_f1']:.4f} weighted_f1={res['weighted_f1']:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.model}_{cfg.protocol}_seed{cfg.seed}_{args.split}"
    np.save(args.out / f"{stem}_softmax.npy", res["probs"])
    pd.DataFrame({
        "true_label": [class_names[i] for i in res["y_true"]],
        "pred_label": [class_names[i] for i in res["y_pred"]],
        "confidence": res["probs"].max(axis=1),
    }).to_csv(args.out / f"{stem}_predictions.csv", index=False)
    print(f"Saved softmax + predictions to {args.out}")


if __name__ == "__main__":
    main()
