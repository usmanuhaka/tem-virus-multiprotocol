# Multi-Protocol Re-Evaluation of TEM Virus Image Classification

Code, source-aware split manifests, and results for the paper:

> **Multi-Protocol Re-Evaluation of TEM Virus Image Classification with Source-Aware Splits and Ensemble Learning**
> Muhammad Haq Usmanuhaka. *Scientific Journal of Informatics (SJI)*, Universitas Negeri Semarang, 2026 (under review).

This project re-evaluates transmission electron microscopy (TEM) virus image
classification on the public Matuszewski & Sintorn corpus. The standard random
crop-level split leaks visually near-identical crops from the same source image
across train/validation/test. We build **source-aware splits**, quantify
near-duplicate leakage, and report performance under four evaluation protocols
using multi-seed DenseNet201 and EfficientNetV2-S models with test-time
augmentation (TTA), mixup, and a two-model softmax ensemble.

---

## Key results (test set)

Test **accuracy**, mean ± std across seeds (5 seeds for A and B-G14; 3 seeds for
A-clean-strict and C-G09):

| Protocol | Baseline DN201 | DN201 + TTA + mixup | EffNetV2-S + TTA + mixup | Ensemble | Ensemble (best seed) |
|---|---|---|---|---|---|
| A (official) | 0.9585 ± 0.0091 | 0.9697 ± 0.0053 | 0.9651 ± 0.0087 | **0.9709 ± 0.0052** | 0.9768 |
| B-G14 (source-aware) | 0.9589 ± 0.0039 | 0.9636 ± 0.0055 | 0.9605 ± 0.0056 | **0.9663 ± 0.0042** | 0.9721 |
| A-clean-strict | 0.9612 ± 0.0077 | 0.9728 ± 0.0034 | 0.9647 ± 0.0071 | **0.9716 ± 0.0005** | 0.9721 |
| C-G09 (date+magnification) | 0.9463 ± 0.0108 | 0.9596 ± 0.0016 | 0.9601 ± 0.0021 | **0.9633 ± 0.0015** | 0.9649 |

Macro **F1** (mean ± std): Ensemble reaches 0.9640 (A), 0.9678 (B-G14),
0.9654 (A-clean-strict), 0.9398 (C-G09). Full numbers are in
[`results/tables/`](results/tables) and per-seed values in
[`results/summaries/`](results/summaries).

**Takeaways.** TTA + mixup and the ensemble improve consistently over the
baseline across all four protocols (positive mean differences, large effect
sizes, bootstrap CIs excluding zero). Source-aware splitting (B-G14) does not
collapse accuracy, indicating the task remains learnable without crop-level
leakage. See the paper for the full statistical discussion (and the note that
the signed-rank test is underpowered at these seed counts).

---

## Evaluation protocols

| Protocol | Definition |
|---|---|
| **A (official)** | The dataset's original crop-level train/val/test split. |
| **B-G14** | Source-aware split that keeps all crops from the same RAW source image in a single split (grouping `G14_RAWSource`). |
| **A-clean-strict** | Protocol A with the 8 visually validated exact-duplicate (Hamming-0) crop pairs removed from training. Isolates exact-duplicate removal; near-duplicates are retained. |
| **C-G09** | Source-aware split grouped by acquisition date + magnification (`G09_Date_Magnification`). |

---

## Repository structure

```
tem-virus-multiprotocol/
├── notebooks/                         # Reproducible pipeline (outputs stripped)
│   ├── 01_phase1_dataset_verification_densenet201_baseline.ipynb
│   └── 02_phase2_4_equivariant_sweep_tta_mixup_ensemble.ipynb
├── data/
│   └── splits/                        # Source-aware split manifests (one row per crop)
│       ├── protocol_A_official.csv
│       ├── protocol_B_G14.csv
│       ├── protocol_A_clean_strict.csv
│       └── protocol_C_G09.csv
├── results/
│   ├── tables/                        # Paper tables 1-6 (CSV)
│   ├── summaries/                     # Per-seed metrics, paired deltas, stats, ensemble
│   └── figures/                       # Headline figures (PNG)
├── app/                               # Streamlit inference demo
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── README.md
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Dataset

The images are from the public TEM virus dataset of **Matuszewski & Sintorn
(2021)**, *"TEM virus images: Benchmark dataset and deep learning
classification,"* Computer Methods and Programs in Biomedicine 209, 106318,
[doi:10.1016/j.cmpb.2021.106318](https://doi.org/10.1016/j.cmpb.2021.106318).
It contains 14 virus classes; this project uses the
`context_virus_1nm_256x256` crops.

This repository ships only the **split manifests**, not the images. Obtain the
dataset from the source above (the dataset is governed by its own license),
then point the manifests at your local copy.

> **Add the exact dataset download URL here** once confirmed (e.g., the Zenodo
> record or supplementary link from the dataset paper).

### Split manifest columns

Each `data/splits/*.csv` has one row per crop:

| Column | Meaning |
|---|---|
| `filepath` | Original (Colab) path to the `.tif` crop — remap to your local dataset root. |
| `filename` | Crop file name. |
| `class_name` | Virus class (one of 14). |
| `label_id` | Integer class id, 0–13, alphabetical (0 = Adenovirus … 13 = Rotavirus). |
| `split` | `train` / `validation` / `test`. |
| `raw_source_id` | Identifier of the originating RAW source image. |
| `G14_RAWSource` / `G09_Date_Magnification` | Grouping keys used to build source-aware splits. |

---

## Reproducing the experiments

1. **Environment**

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

   The notebooks were developed on Google Colab (NVIDIA L4 GPU). Live,
   executed copies with outputs are linked in the paper's Code Availability
   statement.

2. **Phase 1** — `notebooks/01_...baseline.ipynb`: dataset verification,
   crop→RAW mapping, near-duplicate detection, source-aware split construction,
   and the multi-seed DenseNet201 baseline.

3. **Phases 2–4** — `notebooks/02_...ensemble.ipynb`: equivariant baseline,
   architecture sweep, the TTA + mixup pipeline, multi-seed runs across all
   protocols, and the two-model softmax ensemble. Tables and figures are
   regenerated here.

Set the dataset root and (in Colab) mount Drive at the top of each notebook.

---

## Pretrained weights

The DenseNet201 + TTA + mixup weights (`best.pt`, ~74 MB per seed) are **not**
committed to git. Distribute them via a GitHub Release or a Zenodo archive and
download into `weights/`:

```bash
mkdir -p weights
# Example (replace with your Release/Zenodo URL):
# curl -L -o weights/best.pt "<RELEASE_OR_ZENODO_URL>/best.pt"
```

> **Add the weights download URL here** after creating a Release / Zenodo
> record.

---

## Streamlit demo

A single-model classifier (DenseNet201 + TTA + mixup, Protocol A) that takes a
TEM crop and returns the predicted virus class with calibrated 4-view TTA.

```bash
pip install -r app/requirements.txt
# point the app at your weights (sidebar) or:
export WEIGHTS_PATH=weights/best.pt
streamlit run app/streamlit_app.py
```

See [`app/README.md`](app/README.md) for deployment (Streamlit Community Cloud)
and hosting the weights. The demo reproduces the paper's inference setting:
224×224 input, ImageNet normalization, and 4-view TTA (original + horizontal +
vertical + both flips), softmax-averaged.

---

## Citation

If you use this code or the split manifests, please cite the paper (see
[`CITATION.cff`](CITATION.cff)) and the dataset paper (Matuszewski & Sintorn,
2021).

## License

Code and derived artifacts: MIT (see [`LICENSE`](LICENSE)). The underlying TEM
virus image dataset remains under its original authors' terms.
