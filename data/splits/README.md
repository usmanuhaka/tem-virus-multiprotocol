# Source-aware split manifests

One row per crop. These define the four evaluation protocols in the paper.
Images are **not** included — obtain the dataset (Matuszewski & Sintorn, 2021)
and remap the `filepath` column to your local root.

| File | Protocol | train / val / test | Total |
|---|---|---|---|
| `protocol_A_official.csv` | A (official crop-level split) | 5740 / 2249 / 1900 | 9889 |
| `protocol_B_G14.csv` | B-G14 (group = RAW source image) | 5735 / 2251 / 1903 | 9889 |
| `protocol_C_G09.csv` | C-G09 (group = date + magnification) | 5640 / 2312 / 1937 | 9889 |
| `protocol_A_clean_strict.csv` | A minus 8 exact-duplicate crops (train) | 5732 / 2249 / 1900 | 9881 |

## Columns

- `filepath` — original Colab path to the `.tif` crop (remap to your root).
- `filename` — crop file name.
- `class_name` — virus class (14 total).
- `label_id` — integer id 0–13, alphabetical: 0 Adenovirus, 1 Astrovirus,
  2 CCHF, 3 Cowpox, 4 Ebola, 5 Influenza, 6 Lassa, 7 Marburg, 8 Nipah virus,
  9 Norovirus, 10 Orf, 11 Papilloma, 12 Rift Valley, 13 Rotavirus.
- `split` — `train` / `validation` / `test`.
- `raw_source_id` — originating RAW source image id.
- `G14_RAWSource`, `G09_Date_Magnification`, `group_id` — grouping keys for the
  source-aware splits (column names vary by manifest).

## Why source-aware splits

In Protocol A, multiple crops taken from the same RAW micrograph can land in
different splits, so a test crop may be near-identical to a training crop. The
B-G14 and C-G09 manifests prevent this by keeping all crops from one source
(or one acquisition session) within a single split. A-clean-strict isolates the
narrower effect of removing 8 exact-duplicate (Hamming-0) pairs.
