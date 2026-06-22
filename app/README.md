# Streamlit demo: TEM Virus Classifier

Single-model inference for the **DenseNet201 + TTA + mixup** classifier
(Protocol A). Upload a TEM crop; the app returns the predicted virus class and
top-K probabilities using 4-view test-time augmentation.

## Run locally

```bash
pip install -r requirements.txt

# Option A: local checkpoint (the app's default WEIGHTS_PATH is weights/best.pt)
export WEIGHTS_PATH=weights/best.pt

# Option B: auto-download from a URL on first run
export WEIGHTS_URL="https://github.com/usmanuhaka/tem-virus-multiprotocol/releases/download/v1.0.0/best.pt"

streamlit run streamlit_app.py
```

## Inference details (must match training)

- Model: `timm.create_model("densenet201", num_classes=14)`
- Input: 224×224, ImageNet mean/std normalization
- TTA: four views (original, horizontal flip, vertical flip, both), softmax-averaged
- Checkpoint: a dict containing `model_state_dict` (the loader also accepts a
  raw `state_dict` or a `state_dict` key, and strips a `module.` prefix)

Class order (label id 0→13): Adenovirus, Astrovirus, CCHF, Cowpox, Ebola,
Influenza, Lassa, Marburg, Nipah virus, Norovirus, Orf, Papilloma, Rift Valley,
Rotavirus.

## Hosting the weights

`best.pt` is ~74 MB, too large for a normal git commit. Recommended options:

- **GitHub Release asset:** attach `best.pt` to a tagged release; use its URL
  in `WEIGHTS_URL`.
- **Hugging Face Hub:** upload to a model repo and download with
  `huggingface_hub.hf_hub_download`.
- **Zenodo:** a citable DOI for the weights.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. On share.streamlit.io, create an app pointing to `app/streamlit_app.py`.
3. In the app's **Secrets/Settings**, set `WEIGHTS_URL` to your weights URL so
   the model downloads on first launch (the repo itself stays lightweight).

## Notes

This is a single-model demo. The paper's best configuration is a DenseNet201 +
EfficientNetV2-S softmax ensemble; an ensemble demo additionally requires the
EfficientNetV2-S weights and an averaging step over both models' softmax
outputs.
