# Reproducing the results with the clean pipeline (`src/` + `scripts/`)

This document is the **reproduction entry point**. The two notebooks under
`notebooks/` are the *experimental record* (research log): they contain
exploration, configuration trials, recovery/resume steps, and stale outputs, and
they are tied to Google Colab + Google Drive paths, so they are **not** intended
to be run end-to-end with *Run All*. The code under `src/temvirus/` and
`scripts/` re-expresses the **same** method, same model, augmentation,
schedule, mixup, TTA, ensemble, and statistics, in a structured, deterministic,
Colab-free form. No hyperparameter or result is changed.

## What runs where

| Stage | Notebook (record) | Clean code (reproduction) |
|---|---|---|
| Dataset checks, near-duplicate detection, source-aware splits | `01_…baseline.ipynb` | **shipped manifests** in `data/splits/` + `scripts/verify_splits.py` |
| DenseNet201 baseline | `01_…baseline.ipynb` | (baseline is the torchvision reference; the enhanced DN201 below supersedes it for the headline numbers) |
| Enhanced pipeline (TTA + mixup), architecture sweep, ensemble, stats | `02_…ensemble.ipynb` | `scripts/train.py`, `scripts/evaluate.py`, `scripts/ensemble.py`, `scripts/stats.py` |

The four protocol splits are **provided** as manifests (one row per crop) so that
anyone reproduces the *identical* partitions without re-running the
dataset-dependent split construction.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-repro.txt
# GPU strongly recommended for training (the notebooks used an NVIDIA L4).
```

## 0. Validate the splits (no dataset or GPU needed)

```bash
python scripts/verify_splits.py --splits-dir data/splits
```

Confirms split sizes, 14-class coverage, `label_id` order, and the absence of
source-group leakage (B-G14 / C-G09) and exact (class, filename) overlaps.

## 1. Get the images

Download the TEM virus images from Mendeley Data
(`doi:10.17632/x4dwwfwtw3.3`) and note the local root. The manifests carry
`class_name` + `filename`; `--data-root` is resolved as
`<data-root>/<class_name>/<filename>` (with a flat fallback).

## 2. Train (per model × protocol × seed)

```bash
python scripts/train.py --model densenet201 --protocol A --seed 42 \
    --data-root /path/to/tem_dataset --out outputs/
python scripts/train.py --model effnetv2s  --protocol A --seed 42 \
    --data-root /path/to/tem_dataset --out outputs/
```

Seeds per protocol: A and B-G14 use `42 123 456 789 1024`; A-clean-strict and
C-G09 use `42 123 456` (see `configs/protocols.yaml`). Each run writes
`<model>_<protocol>_seed<seed>_best.pt` (selected on validation macro-F1, same
checkpoint format the demo loads) plus a history JSON.

## 3. Evaluate with TTA → softmaxes

```bash
for split in validation test; do
  python scripts/evaluate.py --checkpoint outputs/densenet201_A_seed42_best.pt \
      --model densenet201 --protocol A --split $split \
      --data-root /path/to/tem_dataset --out outputs/
  python scripts/evaluate.py --checkpoint outputs/effnetv2s_A_seed42_best.pt \
      --model effnetv2s --protocol A --split $split \
      --data-root /path/to/tem_dataset --out outputs/
done
```

## 4. Ensemble (select best validation strategy → apply to test)

```bash
python scripts/ensemble.py --protocol A --seed 42 \
    --dense-val  outputs/densenet201_A_seed42_validation_softmax.npy \
    --dense-test outputs/densenet201_A_seed42_test_softmax.npy \
    --effnet-val  outputs/effnetv2s_A_seed42_validation_softmax.npy \
    --effnet-test outputs/effnetv2s_A_seed42_test_softmax.npy \
    --out outputs/
```

Six strategies (simple average, val-F1-weighted, geometric mean, max-confidence,
and each single model) are scored on validation; the best by macro-F1 is applied
to the test split, exactly the selection rule used in the paper.

## 5. Statistics

Collect per-seed accuracies and build the 12-row comparison table (mean diff,
std, Wilcoxon p, Cohen's d, bootstrap 95% CI, Bonferroni 0.05/12):

```bash
python scripts/stats.py \
    --seed-level results/summaries/CP4_ensemble_seed_level.csv \
    --baseline-csv results/summaries/baseline_seed_level.csv \
    --out results/summaries/statistical_tests_results.csv
```

As noted in the paper, the signed-rank test is underpowered at these seed
counts; the bootstrap CI and effect size are the informative quantities.

## Faithfulness

Every constant and routine here is ported from the notebooks (the source of
truth): IMG_SIZE 224, batch 32, AdamW (lr 1e-4, weight decay 1e-4), label
smoothing 0.1, mixup α=0.2 with p=0.5, gradient clip 1.0, 5-epoch warmup then
cosine annealing with warm restarts (T₀=10, T_mult=2, η_min=1e-6), early stop
patience 10, 4-view TTA (identity + H/V/both flips, softmax-averaged), ImageNet
normalization. If any single run in the notebooks used a slightly different value
(e.g. EfficientNetV2-S label smoothing), set it via the YAML config / CLI.
