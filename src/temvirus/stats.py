"""Paired statistical comparisons — ported verbatim from the stats cells.

For each protocol and each comparison (TTA+mixup-DN vs baseline, ensemble vs
baseline, ensemble vs TTA+mixup-DN), the per-seed accuracy differences feed a
mean, std, Wilcoxon signed-rank p, paired Cohen's d, and a 10k-sample bootstrap
95% CI. Significance uses a Bonferroni threshold of 0.05/12 (12 comparisons).

Note (carried from the paper): the signed-rank test is underpowered at these
seed counts (its floor p-value cannot cross the Bonferroni threshold), so the
bootstrap CI and effect size are the informative quantities.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

BONFERRONI_THRESHOLD = 0.05 / 12


def paired_cohens_d(diff) -> float:
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.inf if np.mean(diff) != 0 else 0.0
    return float(np.mean(diff) / sd)


def bootstrap_ci_mean(diff, n_iter: int = 10000, seed: int = 42):
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = [np.mean(rng.choice(diff, size=n, replace=True)) for _ in range(n_iter)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def safe_wilcoxon(diff) -> float:
    from scipy.stats import wilcoxon
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return np.nan
    if np.allclose(diff, 0):
        return 1.0
    try:
        _stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
        return float(p)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Wilcoxon failed for diff={diff}: {exc}")
        return np.nan


def build_comparison_row(protocol: str, comparison: str,
                         left_values: Dict[int, float], right_values: Dict[int, float]) -> dict:
    matched = sorted(set(left_values) & set(right_values))
    diffs = np.asarray([float(left_values[s]) - float(right_values[s]) for s in matched], dtype=float)
    if len(diffs) == 0:
        return {"protocol": protocol, "comparison": comparison, "n_seeds": 0,
                "matched_seeds": "", "mean_diff": np.nan, "std_diff": np.nan,
                "wilcoxon_p": np.nan, "cohens_d": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "bonferroni_threshold": BONFERRONI_THRESHOLD,
                "significant_after_bonferroni": False}
    ci_low, ci_high = bootstrap_ci_mean(diffs, n_iter=10000, seed=42)
    return {
        "protocol": protocol, "comparison": comparison, "n_seeds": len(diffs),
        "matched_seeds": ",".join(map(str, matched)),
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        "wilcoxon_p": safe_wilcoxon(diffs),
        "cohens_d": paired_cohens_d(diffs),
        "ci_low": ci_low, "ci_high": ci_high,
        "bonferroni_threshold": BONFERRONI_THRESHOLD,
        "significant_after_bonferroni": False,
    }


def build_comparison_table(baseline: Dict[str, Dict[int, float]],
                           dn_ttamixup: Dict[str, Dict[int, float]],
                           ensemble: Dict[str, Dict[int, float]]) -> pd.DataFrame:
    """Build the 12-row comparison table (4 protocols x 3 comparisons)."""
    rows = []
    for protocol in baseline:
        rows.append(build_comparison_row(protocol, "TTA+Mixup_DN_vs_Baseline",
                                         dn_ttamixup.get(protocol, {}), baseline[protocol]))
        rows.append(build_comparison_row(protocol, "Ensemble_vs_Baseline",
                                         ensemble.get(protocol, {}), baseline[protocol]))
        rows.append(build_comparison_row(protocol, "Ensemble_vs_TTAMixup_DN",
                                         ensemble.get(protocol, {}), dn_ttamixup.get(protocol, {})))
    df = pd.DataFrame(rows)
    df["significant_after_bonferroni"] = (
        pd.to_numeric(df["wilcoxon_p"], errors="coerce") < BONFERRONI_THRESHOLD)
    return df
