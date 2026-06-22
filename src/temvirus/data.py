"""Data: dataset, transforms, dataloaders, and manifest handling.

The dataset and transforms are ported from the CP3.6 block: each crop is read as
grayscale and replicated to 3 channels, resized to 224, ImageNet-normalized;
training adds horizontal/vertical flips and light ColorJitter. torch/torchvision
are imported lazily so the rest of the package (and the test suite) can import
without them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PIL import Image

from .config import IMAGENET_MEAN, IMAGENET_STD


# Manifests label the held-out split "validation"; the notebook code used "val".
def normalize_split(value: str) -> str:
    v = str(value).strip().lower()
    return {"val": "validation", "valid": "validation"}.get(v, v)


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"filepath", "filename", "class_name", "label_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["split"] = df["split"].map(normalize_split)
    return df


def resolve_image_path(row, data_root: Optional[Path]) -> str:
    """Map a manifest row to a readable image path.

    The manifest ``filepath`` is the original (Colab) path. When ``data_root`` is
    given we look for ``data_root/<class_name>/<filename>`` then
    ``data_root/<filename>``; otherwise we fall back to the manifest path.
    """
    if data_root is not None:
        by_class = data_root / str(row["class_name"]) / str(row["filename"])
        if by_class.exists():
            return str(by_class)
        flat = data_root / str(row["filename"])
        if flat.exists():
            return str(flat)
    return str(row["filepath"])


def build_transforms(img_size: int):
    """Return (train_transform, eval_transform). Mirrors CP3.6 exactly."""
    import torchvision.transforms as T

    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def make_dataset(df: pd.DataFrame, class_to_idx, transform, data_root: Optional[Path]):
    from torch.utils.data import Dataset

    root = Path(data_root) if data_root is not None else None

    class TEMVirusDataset(Dataset):
        def __init__(self, frame):
            self.df = frame.reset_index(drop=True).copy()

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = resolve_image_path(row, root)
            label = class_to_idx[row["class_name"]]
            image = Image.open(path).convert("L")
            image = Image.merge("RGB", (image, image, image))
            if transform is not None:
                image = transform(image)
            return image, label, path

    return TEMVirusDataset(df)


def build_loaders(manifest: pd.DataFrame, class_to_idx, cfg, data_root: Optional[Path]):
    """Build train/val/test dataloaders for one protocol manifest."""
    import torch
    from torch.utils.data import DataLoader

    train_tf, eval_tf = build_transforms(cfg.img_size)
    train_df = manifest[manifest["split"] == "train"].copy()
    val_df = manifest[manifest["split"] == "validation"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()

    train_ds = make_dataset(train_df, class_to_idx, train_tf, data_root)
    val_ds = make_dataset(val_df, class_to_idx, eval_tf, data_root)
    test_ds = make_dataset(test_df, class_to_idx, eval_tf, data_root)

    gen = torch.Generator()
    gen.manual_seed(cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, (len(train_df), len(val_df), len(test_df))
