# Notebooks

The reproducible pipeline, in two phases. **Cell outputs have been stripped**
(for a lightweight repo); executed copies with full outputs are linked in the
paper's Code Availability statement (Google Colab).

## `01_phase1_dataset_verification_densenet201_baseline.ipynb`

- Dataset integrity and inventory checks.
- Crop → RAW source mapping (parsing the dataset's MATLAB metadata).
- Near-duplicate detection (perceptual hashing) and visual validation.
- Construction of the source-aware split manifests (Protocols A, B-G14, C-G09)
  and the leakage analysis.
- Multi-seed DenseNet201 **baseline** training/evaluation.

## `02_phase2_4_equivariant_sweep_tta_mixup_ensemble.ipynb`

- Equivariant (group-CNN) baseline experiment.
- Architecture sweep (EfficientNetV2-S, ConvNeXt-T, …).
- The enhanced pipeline: **TTA + mixup + warm-restart scheduling**.
- Multi-seed runs across all four protocols for DenseNet201 and
  EfficientNetV2-S.
- Two-model **softmax ensemble** and per-protocol fusion-rule selection.
- Generation of the paper's tables and figures.

## Running

Install `../requirements.txt`, set the dataset root (and mount Drive if on
Colab) in the first cells, then run top to bottom. GPU strongly recommended
(developed on an NVIDIA L4).

> Reproducibility note: the baseline (Phase 1) and the enhanced pipeline use
> different training configurations, so the "TTA + mixup vs baseline"
> comparison reflects a combined pipeline-level effect rather than a
> single-factor ablation (see the paper's Limitations).
