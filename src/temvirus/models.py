"""Model builder — timm, ImageNet-pretrained, 14-class head."""
from __future__ import annotations

from .config import NUM_CLASSES, timm_name


def build_model(model_key: str = "densenet201", pretrained: bool = True, num_classes: int = NUM_CLASSES):
    """Create the timm model exactly as in the notebooks (``make_model``)."""
    import timm
    return timm.create_model(timm_name(model_key), pretrained=pretrained, num_classes=num_classes)


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
