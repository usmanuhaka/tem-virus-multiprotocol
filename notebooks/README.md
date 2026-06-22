# Notebooks: experimental record (research log)

> **These notebooks are the experimental record, not the reproduction entry
> point.** They document the full research process (exploration, configuration
> trials, checkpoint/resume and recovery steps, and stale intermediate outputs),
> and they are coupled to Google Colab + Google Drive paths. They are **not**
> intended to run end-to-end with *Run All* from a clean environment.
>
> To reproduce the results, use the clean, deterministic pipeline under
> [`../src/temvirus/`](../src/temvirus) + [`../scripts/`](../scripts). See
> [`../REPRODUCE.md`](../REPRODUCE.md). That code re-expresses the **same** method
> (model, augmentation, schedule, mixup, TTA, ensemble, statistics) and the same
> hyperparameters, and produces the same numbers.

Cell outputs have been stripped for a lightweight repo; executed copies with full
outputs are available from the author on request.

## `01_phase1_dataset_verification_densenet201_baseline.ipynb`

- Dataset integrity and inventory checks.
- Crop → RAW source mapping (parsing the dataset's MATLAB metadata).
- Near-duplicate detection (perceptual/content hashing) and visual validation.
- Construction of the source-aware split manifests (Protocols A, B-G14, C-G09)
  and the leakage analysis.
- Multi-seed DenseNet201 **baseline** training/evaluation.

## `02_phase2_4_equivariant_sweep_tta_mixup_ensemble.ipynb`

- Equivariant (group-CNN) baseline experiment.
- Architecture sweep (EfficientNetV2-S, ConvNeXt-T, …).
- The enhanced pipeline: **TTA + mixup + warm-restart scheduling**.
- Multi-seed runs across all four protocols for DenseNet201 and EfficientNetV2-S.
- Two-model **softmax ensemble** and per-protocol fusion-rule selection.
- Generation of the paper's tables and figures.

## Relationship to the clean pipeline

| In these notebooks | Clean, runnable equivalent |
|---|---|
| Dataset checks, near-duplicate detection, source-aware split construction | the resulting **manifests** in [`../data/splits/`](../data/splits), validated by [`../scripts/verify_splits.py`](../scripts/verify_splits.py) |
| Enhanced training (TTA + mixup + warm-restart schedule), multi-seed runs | [`../scripts/train.py`](../scripts/train.py), [`../scripts/evaluate.py`](../scripts/evaluate.py) |
| Two-model softmax ensemble + fusion-rule selection | [`../scripts/ensemble.py`](../scripts/ensemble.py) |
| Paired statistics (bootstrap CI, Wilcoxon, Cohen's d, Bonferroni) | [`../scripts/stats.py`](../scripts/stats.py) |

> Reproducibility note: the baseline (Phase 1) and the enhanced pipeline use
> different training configurations, so the "TTA + mixup vs baseline" comparison
> reflects a combined pipeline-level effect rather than a single-factor ablation
> (see the paper's Limitations).
