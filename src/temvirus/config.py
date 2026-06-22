"""Canonical configuration.

All defaults are the values used by the enhanced ("CP3.6") pipeline in the
notebooks, the configuration that produced the released DenseNet201 weights and
the headline results. Do not change these if the intent is to reproduce the
paper; expose them via the YAML configs / CLI flags instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

# Class order = label_id order from the split manifests (alphabetical).
CLASS_NAMES: List[str] = [
    "Adenovirus", "Astrovirus", "CCHF", "Cowpox", "Ebola", "Influenza",
    "Lassa", "Marburg", "Nipah virus", "Norovirus", "Orf", "Papilloma",
    "Rift Valley", "Rotavirus",
]
NUM_CLASSES: int = len(CLASS_NAMES)          # 14

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# timm model names actually used.
MODELS: Dict[str, str] = {
    "densenet201": "densenet201",
    "effnetv2s": "tf_efficientnetv2_s.in21k_ft_in1k",
}

# Seeds per protocol (from the seed-level result tables).
PROTOCOL_SEEDS: Dict[str, List[int]] = {
    "A": [42, 123, 456, 789, 1024],
    "B_G14": [42, 123, 456, 789, 1024],
    "A_clean_strict": [42, 123, 456],
    "C_G09": [42, 123, 456],
}

# Protocol -> shipped manifest filename (under data/splits/).
PROTOCOL_MANIFEST: Dict[str, str] = {
    "A": "protocol_A_official.csv",
    "B_G14": "protocol_B_G14.csv",
    "A_clean_strict": "protocol_A_clean_strict.csv",
    "C_G09": "protocol_C_G09.csv",
}


@dataclass
class TrainConfig:
    """Hyperparameters for one training run (one model, one protocol, one seed)."""
    model: str = "densenet201"           # key into MODELS
    protocol: str = "A"                  # key into PROTOCOL_SEEDS / PROTOCOL_MANIFEST
    seed: int = 42

    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 35

    base_lr: float = 1e-4
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    t_0: int = 10
    t_mult: int = 2

    label_smoothing: float = 0.1
    grad_clip_max_norm: float = 1.0

    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5

    early_stop_patience: int = 10
    pretrained: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def timm_name(model_key: str) -> str:
    if model_key not in MODELS:
        raise KeyError(f"Unknown model key '{model_key}'. Options: {list(MODELS)}")
    return MODELS[model_key]
