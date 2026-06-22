#!/usr/bin/env python3
"""Train one model for one protocol and seed (faithful CP3.6 pipeline).

Example
-------
    python scripts/train.py --model densenet201 --protocol A --seed 42 \
        --data-root /path/to/tem_dataset --out outputs/

Requires the protocol manifest under --splits-dir (default: data/splits) and the
image dataset under --data-root. Writes ``<model>_<protocol>_seed<seed>_best.pt``
(with ``model_state_dict`` + metadata) and a training-history JSON to --out.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temvirus.config import CLASS_NAMES, PROTOCOL_MANIFEST, TrainConfig  # noqa: E402
from temvirus.data import build_loaders, load_manifest  # noqa: E402
from temvirus.engine import fit, get_device  # noqa: E402
from temvirus.models import build_model  # noqa: E402
from temvirus.seed import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="densenet201", choices=["densenet201", "effnetv2s"])
    ap.add_argument("--protocol", default="A", choices=list(PROTOCOL_MANIFEST))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", type=Path, required=True, help="root of the TEM image dataset")
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = TrainConfig(model=args.model, protocol=args.protocol, seed=args.seed)
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    set_seed(cfg.seed)
    class_names = sorted(CLASS_NAMES)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    manifest_path = args.splits_dir / PROTOCOL_MANIFEST[cfg.protocol]
    manifest = load_manifest(manifest_path)
    train_loader, val_loader, _test_loader, sizes = build_loaders(
        manifest, class_to_idx, cfg, args.data_root)
    print(f"Protocol {cfg.protocol} | train/val/test = {sizes} | model {cfg.model} | seed {cfg.seed}")

    device = get_device()
    model = build_model(cfg.model, pretrained=cfg.pretrained)
    history, best_epoch, best_f1, ckpt = fit(
        model, train_loader, val_loader, cfg, device, args.out, class_names)
    print(f"\nDone. Best val macro-F1 = {best_f1:.4f} at epoch {best_epoch}. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
