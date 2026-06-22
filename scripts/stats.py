#!/usr/bin/env python3
"""Build the paired statistical comparison table (12 rows) from per-seed results.

Reads the seed-level ensemble table (which carries per-seed DenseNet201 and
ensemble test accuracies) and, optionally, a baseline per-seed accuracy table.
Produces ``statistical_tests_results.csv`` with mean diff, std, Wilcoxon p,
Cohen's d, bootstrap 95% CI, and the Bonferroni (0.05/12) significance flag.

Example
-------
    python scripts/stats.py \
        --seed-level results/summaries/CP4_ensemble_seed_level.csv \
        --baseline-csv results/summaries/baseline_seed_level.csv \
        --out results/summaries/statistical_tests_results.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temvirus.stats import BONFERRONI_THRESHOLD, build_comparison_row  # noqa: E402

COMPARISONS = ["TTA+Mixup_DN_vs_Baseline", "Ensemble_vs_Baseline", "Ensemble_vs_TTAMixup_DN"]


def nested(df: pd.DataFrame, value_col: str) -> dict:
    """{protocol: {seed: value}} from a long table with protocol/seed columns."""
    out: dict = {}
    for _, r in df.iterrows():
        out.setdefault(str(r["protocol"]), {})[int(r["seed"])] = float(r[value_col])
    return out


def autodetect_accuracy_col(df: pd.DataFrame) -> str:
    for c in ["accuracy", "baseline_accuracy", "baseline_test_accuracy", "test_accuracy"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find an accuracy column in baseline CSV (cols: {list(df.columns)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-level", type=Path, required=True,
                    help="CP4_ensemble_seed_level.csv (densenet_test_accuracy + ensemble_test_accuracy)")
    ap.add_argument("--baseline-csv", type=Path, default=None,
                    help="optional long table: protocol, seed, <accuracy>")
    ap.add_argument("--out", type=Path, default=Path("statistical_tests_results.csv"))
    args = ap.parse_args()

    sl = pd.read_csv(args.seed_level)
    dn = nested(sl, "densenet_test_accuracy")
    ens = nested(sl, "ensemble_test_accuracy")

    baseline: dict = {}
    if args.baseline_csv and args.baseline_csv.exists():
        bdf = pd.read_csv(args.baseline_csv)
        baseline = nested(bdf, autodetect_accuracy_col(bdf))
    else:
        print("[note] no baseline CSV provided, baseline comparisons will be empty.")

    protocols = [p for p in ["A", "B_G14", "A_clean_strict", "C_G09"] if p in ens] or sorted(ens)
    rows = []
    for p in protocols:
        rows.append(build_comparison_row(p, "TTA+Mixup_DN_vs_Baseline", dn.get(p, {}), baseline.get(p, {})))
        rows.append(build_comparison_row(p, "Ensemble_vs_Baseline", ens.get(p, {}), baseline.get(p, {})))
        rows.append(build_comparison_row(p, "Ensemble_vs_TTAMixup_DN", ens.get(p, {}), dn.get(p, {})))

    out_df = pd.DataFrame(rows)
    out_df["significant_after_bonferroni"] = (
        pd.to_numeric(out_df["wilcoxon_p"], errors="coerce") < BONFERRONI_THRESHOLD)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out} (Bonferroni threshold {BONFERRONI_THRESHOLD:.6f})")


if __name__ == "__main__":
    main()
