"""4-view test-time augmentation — ported verbatim from CP3.6.

Views: identity, horizontal flip, vertical flip, both; softmax of each is
averaged. This is the exact inference setting used for the paper and the demo.
"""
from __future__ import annotations


def tta_predict(model, images):
    import torch
    augmentations = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),   # horizontal
        lambda x: torch.flip(x, dims=[2]),   # vertical
        lambda x: torch.flip(x, dims=[2, 3]),  # both
    ]
    probs_list = []
    for aug in augmentations:
        logits = model(aug(images))
        probs_list.append(torch.softmax(logits.float(), dim=1))
    return torch.mean(torch.stack(probs_list, dim=0), dim=0)
