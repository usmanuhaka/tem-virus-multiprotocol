"""Softmax ensemble, combination strategies and validation-based selection.

Ported verbatim from the CP3.6 ensemble cells. Six strategies are evaluated on
the validation split; the one with the highest validation macro-F1 is applied to
the test split (this is how each protocol/seed's "Ensemble" result was chosen).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .metrics import compute_metrics


def evaluate_softmax(softmax: np.ndarray, y_true: np.ndarray, strategy_name: str) -> dict:
    pred = softmax.argmax(axis=1)
    out = {"strategy": strategy_name}
    out.update(compute_metrics(y_true, pred))
    return out


def simple_average(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * a + 0.5 * b


def weighted_by_val_f1(a: np.ndarray, b: np.ndarray, w_a: float, w_b: float) -> np.ndarray:
    return w_a * a + w_b * b


def geometric_mean(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    out = np.exp((np.log(a + eps) + np.log(b + eps)) / 2.0)
    return out / out.sum(axis=1, keepdims=True)


def max_confidence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    use_a = a.max(axis=1) > b.max(axis=1)
    return np.where(use_a[:, None], a, b)


def val_f1_weights(effnet_val: np.ndarray, dense_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
    fe = evaluate_softmax(effnet_val, y_val, "EffNetV2S_only")["macro_f1"]
    fd = evaluate_softmax(dense_val, y_val, "DenseNet201_only")["macro_f1"]
    total = fe + fd
    return fe / total, fd / total


def all_strategies(effnet: np.ndarray, dense: np.ndarray, w_eff: float, w_dense: float) -> Dict[str, np.ndarray]:
    return {
        "EffNetV2S_only": effnet,
        "DenseNet201_only": dense,
        "simple_average": simple_average(effnet, dense),
        "weighted_by_valF1": weighted_by_val_f1(effnet, dense, w_eff, w_dense),
        "geometric_mean": geometric_mean(effnet, dense),
        "max_confidence": max_confidence(effnet, dense),
    }


def select_and_apply(effnet_val, dense_val, effnet_test, dense_test, y_val):
    """Pick the best validation strategy and apply it to the test softmaxes.

    Returns (best_strategy_name, ensemble_val_softmax, ensemble_test_softmax,
    validation_results_table).
    """
    w_eff, w_dense = val_f1_weights(effnet_val, dense_val, y_val)
    val_strats = all_strategies(effnet_val, dense_val, w_eff, w_dense)
    test_strats = all_strategies(effnet_test, dense_test, w_eff, w_dense)

    val_results = [evaluate_softmax(sm, y_val, name) for name, sm in val_strats.items()]
    best = max(val_results, key=lambda r: r["macro_f1"])["strategy"]
    return best, val_strats[best], test_strats[best], val_results
