# Cross-region U-Net checkpoints (best pair)

These are the two strongest **cross-region** U-Nets from `src.train_unet_xregion`
(GPU v3 run): train on one continent, validate on the other, early-stop on **Val IoU**.

## What `*_meta.json` is for

It is a small **sidecar** next to the `.pt` weights so inference (and humans) know:

- `best_model_path`: which file to `torch.load` (same folder as this JSON).
- `in_channels`, `base_channels`, `patch_size`, `aef_downsample`: rebuild the same `UNetSmall` and read rasters at the same resolution.
- `best_threshold`: probability cutoff chosen on the **opposite continent’s** validation patches (Union-IoU–friendly operating point).
- `train_tiles` / `val_tiles`, `label_strategy`, `loss_weights`: reproducibility / documentation.

## Ensemble inference

From the repo root (with data downloaded and deps installed):

```bash
python -m scripts.predict_xregion_ensemble \
  --data-dir ./data/makeathon-challenge \
  --meta-a ./models/xregion_best/unet_xregion_americas_to_asia_v3_meta.json \
  --meta-b ./models/xregion_best/unet_xregion_asia_to_americas_v3_meta.json \
  --ensemble-mode mean \
  --aef-downsample 1 --patch-size 128 --stride 64 \
  --submission-out ./artifacts/predictions/submission.geojson
```

Weights are tracked with **Git LFS** (`*.pt` in this directory).
