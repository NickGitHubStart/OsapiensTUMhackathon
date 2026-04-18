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

## XGBoost + Temporal Features

Train an XGBoost pixel model with AEF + NDVI/S1 temporal features:

```bash
python -m src.train_xgb_temporal --data-dir ./data/makeathon-challenge
```

## Notes

- The notebook walkthrough lives in [ONI-makeathon-challenge-2026-main/challenge.ipynb](ONI-makeathon-challenge-2026-main/challenge.ipynb).
- You can also use the Makefile in [ONI-makeathon-challenge-2026-main/Makefile](ONI-makeathon-challenge-2026-main/Makefile) for local setup.
