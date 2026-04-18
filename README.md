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
```

## Notes

- The notebook walkthrough lives in [ONI-makeathon-challenge-2026-main/challenge.ipynb](ONI-makeathon-challenge-2026-main/challenge.ipynb).
- You can also use the Makefile in [ONI-makeathon-challenge-2026-main/Makefile](ONI-makeathon-challenge-2026-main/Makefile) for local setup.
