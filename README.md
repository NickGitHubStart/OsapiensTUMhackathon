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

```

## Baseline4 (Patch XGBoost, Multimodal Cache)

Build the baseline4 cache (train + test):

```bash
python -m src.build_cache_baseline4 \
	--data-dir ./data/makeathon-challenge \
	--cache-dir ./data/makeathon-challenge-cache/baseline4 \
	--split both
```

Train patch-based baseline4 models:

```bash
python -m src.train_baseline4 \
	--cache-dir ./data/makeathon-challenge-cache/baseline4 \
	--patch-size 16 \
	--stride-train 16 \
	--stride-infer 8 \
	--include-gradient
```

Run inference from cache and export GeoJSON:

```bash
python scripts/predict_baseline4.py \
	--cache-dir ./data/makeathon-challenge-cache/baseline4 \
	--bundle-path ./artifacts/baseline4_patch_xgb_ensemble.joblib \
	--out-dir ./submission
```

## Submission Merge (Final Step)

Merge per-tile GeoJSONs into a single submission file:

```bash
python scripts/merge_geojson_tiles.py \
	--in-dir ./submission/baseline3 \
	--pattern "pred_*.geojson" \
	--out-file ./submission/submission.geojson
```

If your per-tile files are named like `{tile_id}.geojson`, use `--pattern "*.geojson"`.

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

## Notes

- The notebook walkthrough lives in [ONI-makeathon-challenge-2026-main/challenge.ipynb](ONI-makeathon-challenge-2026-main/challenge.ipynb).
- You can also use the Makefile in [ONI-makeathon-challenge-2026-main/Makefile](ONI-makeathon-challenge-2026-main/Makefile) for local setup.
