# OsapiensTUMhackathon

## Quickstart (Colab or local)

Install dependencies:

```bash
python -m pip install -U pip
python -m pip install -r ONI-makeathon-challenge-2026-main/requirements.txt
```

Download the dataset from the public S3 bucket:

```bash
python -m src.download_data --local-dir ./data
```

This downloads `makeathon-challenge/` into `./data/`:

```text
data/makeathon-challenge/
```

## Basic Training (Baseline)

Train a simple baseline model on AlphaEarth embeddings with consensus labels:

```bash
python -m src.train_baseline --data-dir ./data/makeathon-challenge
```

The trained model is saved to `./artifacts/baseline_aef_logreg.joblib` by default.

## Baseline2 (AEF Temporal Diffs)

Train an AEF temporal-diff baseline (AEF, AEF-2020, AEF-year-1):

```bash
python -m src.train_baseline2 --data-dir ./data/makeathon-challenge
```

Region holdout validation (train on Thailand, validate on Colombia or vice versa):

```bash
python -m src.train_baseline2 \
	--data-dir ./data/makeathon-challenge \
	--val-region colombia
```

## Baseline3 (Region Ensemble)

Train two region-holdout models plus a full-data model, and save an ensemble bundle:

```bash
python -m src.train_baseline3 \
	--data-dir ./data/makeathon-challenge \
	--max-samples 200000 \
	--per-tile-samples 50000
```

## Patch Baseline (XGBoost)

Train a patch-based XGBoost model on AlphaEarth embeddings:

```bash
python -m src.train_patch_xgboost --data-dir ./data/makeathon-challenge
```

## Patch U-Net (AEF)

Train a small U-Net on AlphaEarth embeddings with random patch sampling:

```bash
python -m src.train_unet --data-dir ./data/makeathon-challenge
```

Augmentations (optional):

```bash
python -m src.train_unet \
	--data-dir ./data/makeathon-challenge \
	--aug-flip-rotate-prob 0.5 \
	--aug-noise-std 0.02 \
	--aug-dropout-prob 0.3 \
	--aug-dropout-frac 0.1 \
	--aug-scale-min 0.9 \
	--aug-scale-max 1.1
```

## XGBoost + Temporal Features

Train an XGBoost pixel model with AEF + NDVI/S1 temporal features:

```bash
python -m src.train_xgb_temporal --data-dir ./data/makeathon-challenge
```

Augmentations (feature-space, optional):

```bash
python -m src.train_xgb_temporal \
	--data-dir ./data/makeathon-challenge \
	--aug-noise-std 0.01 \
	--aug-dropout-prob 0.3 \
	--aug-dropout-frac 0.1
```

## Patch XGBoost Augmentations

```bash
python -m src.train_patch_xgboost \
	--data-dir ./data/makeathon-challenge \
	--patch-size 32 \
	--stride 32 \
	--aug-flip-rotate-prob 0.5 \
	--aug-noise-std 0.02 \
	--aug-dropout-prob 0.3 \
	--aug-dropout-frac 0.1 \
	--aug-scale-min 0.9 \
	--aug-scale-max 1.1
```

## Pixel XGBoost Augmentations

```bash
python -m src.train_baseline \
	--data-dir ./data/makeathon-challenge \
	--aug-noise-std 0.01 \
	--aug-dropout-prob 0.3 \
	--aug-dropout-frac 0.1

## Polygon Prediction (GLAD/RADD Comparison)

Predict deforestation within a polygon and compare with GLAD/RADD consensus labels:

```bash
python scripts/predict_polygon.py \
	--data-dir /content/drive/MyDrive/makeathon-challenge \
	--model-path /content/drive/MyDrive/artifacts/baseline_aef_logreg.joblib \
	--model-type aef_xgb \
	--year 2020 \
	--polygon-geojson /content/drive/MyDrive/my_polygon.geojson \
	--out-dir /content/drive/MyDrive/outputs
```

You can also pass a WKT polygon via `--polygon-wkt` and optionally provide `--tile-id`.

## Sanity Check (Train Tiles)

Run a quick sanity check that writes overlays, polygons, and a JSON report:

```bash
python scripts/sanity_check.py \
	--data-dir /content/drive/MyDrive/makeathon-challenge \
	--model-type baseline2 \
	--model-path /content/drive/MyDrive/artifacts/baseline2_aef_xgb.joblib \
	--tile-ids 18NWG_6_6,48QWD_2_2 \
	--year 2020
```
```

## Cross-Region U-Net Ensemble (best generalization to Africa test set)

Trains two U-Nets where one continent is train and the other is OOD validation,
then ensembles their predictions. Forces region-invariant features → the model
that wins early stopping on Asia is exactly the one that should also work on
Africa.

### v3 — GPU (ROCm) settings

On the AMD MI300X box install the ROCm wheel of torch first:

```bash
.venv/bin/pip install --upgrade --index-url https://download.pytorch.org/whl/rocm6.4 torch
export HSA_OVERRIDE_GFX_VERSION=9.4.2   # MI300X (gfx942) needs this
```

Train both directions in parallel on the GPU (full-res AEF, 128² patches,
preload, mixed precision, cosine LR):

```bash
HSA_OVERRIDE_GFX_VERSION=9.4.2 python -m src.train_unet_xregion \
  --data-dir ./data/makeathon-challenge \
  --train-region americas --val-region asia \
  --epochs 40 --samples-per-epoch 1600 --val-samples 600 \
  --batch-size 32 --aef-downsample 1 --patch-size 128 \
  --base-channels 48 --amp --preload --cosine-lr \
  --early-stop-patience 8 --positive-oversample-prob 0.6 \
  --lr 3e-4 --seed 7 --no-progress-bar \
  --tag americas_to_asia_v3 --artifact-dir ./artifacts/xregion_v3
```

Repeat with `--train-region asia --val-region americas`, `--seed 11`, and
`--tag asia_to_americas_v3` for the second model. Add seed/`--base-channels`
variants (e.g. `--seed 42`, `--base-channels 64`) for ensemble diversity.

### Legacy CPU settings (no GPU)

```bash
OMP_NUM_THREADS=8 nohup python -m src.train_unet_xregion \
  --data-dir ./data/makeathon-challenge \
  --train-region americas --val-region asia \
  --epochs 18 --samples-per-epoch 800 --val-samples 240 \
  --aef-downsample 2 --num-threads 8 --no-progress-bar \
  --tag americas_to_asia --artifact-dir ./artifacts/xregion \
  > logs/train_am_to_as.log 2>&1 &
```

Each model writes `artifacts/xregion/unet_xregion_<tag>_{best,last}.pt`,
`_history.{csv,png}`, and a `_meta.json` with the per-model best threshold
chosen from a sweep on the OOD validation set.

Notes:
- Loss is `0.4*BCE + 0.3*Dice + 0.3*Tversky(α=0.7,β=0.3)` — the Tversky term
  penalises false positives more than false negatives, which is what the
  Union-IoU leaderboard metric rewards.
- Label strategy `xregion` does majority voting over whichever of GLAD-S2,
  RADD and GLAD-L are available for the tile, so Asia tiles (no GLAD-S2) are
  also usable.

Inference + ensemble + submission:

```bash
python -m scripts.predict_xregion_ensemble \
  --data-dir ./data/makeathon-challenge \
  --meta-a ./artifacts/xregion/unet_xregion_americas_to_asia_meta.json \
  --meta-b ./artifacts/xregion/unet_xregion_asia_to_americas_meta.json \
  --ensemble-mode mean \
  --aef-downsample 2 --patch-size 64 --stride 32 \
  --submission-out ./artifacts/xregion/predictions/submission.geojson
```

Use `--ensemble-mode intersection` for the most precision-friendly variant
(usually best for Union IoU when both models are well-calibrated).

**Best cross-region pair (tracked in-repo with Git LFS):** see
`models/xregion_best/README.md`. After `git lfs pull`, run inference with the
same `meta.json` paths and **full-res** settings matching training
(`--aef-downsample 1 --patch-size 128`):

```bash
python -m scripts.predict_xregion_ensemble \
  --data-dir ./data/makeathon-challenge \
  --meta-a ./models/xregion_best/unet_xregion_americas_to_asia_v3_meta.json \
  --meta-b ./models/xregion_best/unet_xregion_asia_to_americas_v3_meta.json \
  --ensemble-mode mean \
  --aef-downsample 1 --patch-size 128 --stride 64 \
  --submission-out ./artifacts/predictions/submission.geojson
```

## Submission format & scoring

**Submission (one file)**

- Upload a single **`.geojson`** file.
- Top-level type must be a GeoJSON **`FeatureCollection`**.
- Each feature geometry must be a **`Polygon`** or **`MultiPolygon`**.
- **Everything inside a polygon counts as predicted deforestation** (the interior is the positive class).
- **`properties.time_step`** (optional): if you predict *when* deforestation occurred, use **`YYMM`** (e.g. **`2204`** = April 2022). If you do not predict time, set it to **`null`** or **omit** the property.

**Scoring (leaderboard)**

| | |
| --- | --- |
| **Primary** | **Union IoU** |
| **Also reported** | Polygon recall, polygon-level FPR, year accuracy |

For converting a binary prediction raster to submission GeoJSON, see `ONI-makeathon-challenge-2026-main/submission_utils.py`.

## Notes

- The notebook walkthrough lives in [ONI-makeathon-challenge-2026-main/challenge.ipynb](ONI-makeathon-challenge-2026-main/challenge.ipynb).
- You can also use the Makefile in [ONI-makeathon-challenge-2026-main/Makefile](ONI-makeathon-challenge-2026-main/Makefile) for local setup.
