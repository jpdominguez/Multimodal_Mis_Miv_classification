# Trained models

Weights of the two models compared in the study, one per cross-validation split (5 folds × 3 repetitions, grouped
by patient). All models use a DenseNet121 image branch pretrained on ImageNet and the same training recipe
(`pipeline/05_train.py`).

| Folder | Model | Input |
|---|---|---|
| `image_only/` | Image-only model (experiment `image`) | Dermoscopic image, 224×224 |
| `image_metadata/` | Image + metadata, concatenation fusion (experiment `concat_meta`) | Dermoscopic image + age, sex, anatomical site and lesion diameter |

**Files:**
- `repR_foldF.pth`: PyTorch `state_dict` of the model trained on the training part of repetition `R` (0-2), fold `F` (0-4).
- `repR_foldF_metadata_encoder.json` (`image_metadata/` only): metadata encoding fitted on that training fold. It holds the normalization statistics (training-fold means, standard deviations and modes) and the order of the 13 input features.

The output is a 2-class softmax; class 1 is invasive melanoma (Miv).

## Loading

```python
import json, sys
import torch
sys.path.insert(0, "pipeline")
from common import MetaEncoder, build_torch_components

T = build_torch_components()

# Image only
model = T["MultimodalNet"]("image", 0)
model.load_state_dict(torch.load("models/image_only/rep0_fold0.pth", map_location="cpu"))

# Image + metadata
cfg = json.load(open("models/image_metadata/rep0_fold0_metadata_encoder.json"))
enc = MetaEncoder(cfg["meta_vars"], cfg["age_enc"], cfg["diam_enc"], cfg["missing"])
enc.stats, enc.names = cfg["stats"], cfg["names"]
model = T["MultimodalNet"]("concat", enc.dim)
model.load_state_dict(torch.load("models/image_metadata/rep0_fold0.pth", map_location="cpu"))
model.eval()
# meta = enc.transform(df)   # df with columns age, sex, site, diam_long, diam_short, source
```

Images are resized to 224×224 and normalized with ImageNet statistics (`CachedImageDataset` in `pipeline/common.py`).
