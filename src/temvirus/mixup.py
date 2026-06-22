"""Mixup — ported verbatim from CP3.6."""
from __future__ import annotations

import numpy as np


def mixup_data(x, y, alpha: float = 0.2):
    import torch
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_loss(criterion, logits, y_a, y_b, lam: float):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
