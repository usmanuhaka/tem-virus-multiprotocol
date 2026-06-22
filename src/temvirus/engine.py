"""Training engine, scheduler, train/eval loops, the ``fit`` driver, and
checkpointing. Ported from the CP3.6 block; torch imported lazily.

``fit`` runs the warmup→cosine-restart schedule, mixup-augmented AMP training,
TTA validation each epoch, early stopping on validation macro-F1, and writes
``best.pt`` in the same format the demo expects (a dict with ``model_state_dict``
plus metadata).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import compute_metrics
from .mixup import mixup_data, mixup_loss
from .tta import tta_predict


def get_device(prefer: str | None = None):
    import torch
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WarmupCosineRestartScheduler:
    """Linear warmup then CosineAnnealingWarmRestarts, verbatim from CP3.6."""

    def __init__(self, optimizer, warmup_epochs=5, warmup_start_lr=1e-6,
                 base_lr=1e-4, min_lr=1e-6, t_0=10, t_mult=2):
        import torch.optim as optim
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.t_0 = t_0
        self.t_mult = t_mult
        self.last_epoch = -1
        self.restart_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult, eta_min=min_lr)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            progress = epoch / max(1, self.warmup_epochs - 1)
            lr = self.warmup_start_lr + progress * (self.base_lr - self.warmup_start_lr)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        else:
            self.restart_scheduler.step(epoch - self.warmup_epochs)
            lr = self.optimizer.param_groups[0]["lr"]
        self.last_epoch = epoch
        return lr

    def state_dict(self):
        return {
            "warmup_epochs": self.warmup_epochs, "warmup_start_lr": self.warmup_start_lr,
            "base_lr": self.base_lr, "min_lr": self.min_lr, "t_0": self.t_0,
            "t_mult": self.t_mult, "last_epoch": self.last_epoch,
            "restart_scheduler": self.restart_scheduler.state_dict(),
        }

    def load_state_dict(self, sd):
        self.warmup_epochs = sd["warmup_epochs"]; self.warmup_start_lr = sd["warmup_start_lr"]
        self.base_lr = sd["base_lr"]; self.min_lr = sd["min_lr"]; self.t_0 = sd["t_0"]
        self.t_mult = sd["t_mult"]; self.last_epoch = sd["last_epoch"]
        self.restart_scheduler.load_state_dict(sd["restart_scheduler"])


def _make_scaler(device):
    import torch
    try:
        return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch_num):
    import torch
    import torch.nn as nn
    model.train()
    running_loss = 0.0
    all_targets, all_preds = [], []
    last_grad_norm = 0.0

    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        apply_mixup = np.random.random() < cfg.mixup_prob
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            if apply_mixup:
                mixed, y_a, y_b, lam = mixup_data(images, labels, alpha=cfg.mixup_alpha)
                logits = model(mixed)
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
        if torch.isnan(loss):
            raise ValueError(f"NaN loss at epoch {epoch_num}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        preds = torch.argmax(logits.detach(), dim=1)
        running_loss += loss.item() * images.size(0)
        all_targets.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        last_grad_norm = float(grad_norm)

    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = float(running_loss / len(loader.dataset))
    metrics["grad_norm_last"] = last_grad_norm
    return metrics


def evaluate_with_tta(model, loader, criterion, device):
    """TTA evaluation. Returns metrics dict augmented with arrays (probs/y_true/y_pred)."""
    import torch
    model.eval()
    running_loss = 0.0
    all_targets, all_preds, all_probs, all_paths = [], [], [], []
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                probs = tta_predict(model, images)
                loss = criterion(model(images), labels)
            preds = torch.argmax(probs, dim=1)
            running_loss += loss.item() * images.size(0)
            all_targets.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.append(probs.detach().cpu())
            all_paths.extend(list(paths))
    probs_np = torch.cat(all_probs, dim=0).numpy()
    out = compute_metrics(all_targets, all_preds)
    out["loss"] = float(running_loss / len(loader.dataset))
    out["y_true"] = np.asarray(all_targets)
    out["y_pred"] = np.asarray(all_preds)
    out["probs"] = probs_np
    out["paths"] = all_paths
    return out


def save_checkpoint(path, model, cfg, val_metrics, epoch, class_names):
    import torch
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_metrics": {k: v for k, v in val_metrics.items()
                        if k not in ("y_true", "y_pred", "probs", "paths")},
        "class_names": class_names,
        "config": cfg.to_dict(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def fit(model, train_loader, val_loader, cfg, device, out_dir, class_names):
    """Full training driver. Returns (history, best_epoch, best_macro_f1, best_ckpt_path)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = device or get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    scheduler = WarmupCosineRestartScheduler(
        optimizer, warmup_epochs=cfg.warmup_epochs, warmup_start_lr=cfg.warmup_start_lr,
        base_lr=cfg.base_lr, min_lr=cfg.min_lr, t_0=cfg.t_0, t_mult=cfg.t_mult)
    scaler = _make_scaler(device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / f"{cfg.model}_{cfg.protocol}_seed{cfg.seed}_best.pt"

    history = []
    best_macro_f1 = -1.0
    best_epoch = None
    patience = 0

    for epoch in range(cfg.max_epochs):
        lr = scheduler.step(epoch)
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg, epoch + 1)
        va = evaluate_with_tta(model, val_loader, criterion, device)
        row = {"epoch": epoch + 1, "lr": lr,
               "train_loss": tr["loss"], "train_macro_f1": tr["macro_f1"],
               "val_loss": va["loss"], "val_acc": va["accuracy"], "val_macro_f1": va["macro_f1"]}
        history.append(row)
        print(f"[{cfg.model} {cfg.protocol} s{cfg.seed}] epoch {epoch+1:02d}/{cfg.max_epochs} "
              f"lr={lr:.2e} train_f1={tr['macro_f1']:.4f} val_f1={va['macro_f1']:.4f} "
              f"val_acc={va['accuracy']:.4f}")

        if va["macro_f1"] > best_macro_f1:
            best_macro_f1 = va["macro_f1"]
            best_epoch = epoch + 1
            patience = 0
            save_checkpoint(best_ckpt, model, cfg, va, epoch + 1, class_names)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (no val macro-F1 gain for "
                      f"{cfg.early_stop_patience} epochs).")
                break

    with open(out_dir / f"{cfg.model}_{cfg.protocol}_seed{cfg.seed}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return history, best_epoch, best_macro_f1, str(best_ckpt)
