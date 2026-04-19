# Deforestation Challenge - Current State

## Leaderboard status
- 6 submissions used of 10
- Best score: Sub 4 at 44.98% IoU (recall 70.5%, FPR 44.6%)
- Target: 50-55% IoU
- Top leaderboard: 54% IoU

## Goal for next submission
Retrain Baseline 3 XGBoost ensemble with label confidence weights.
Expected: +3-5 IoU, landing ~48-50%.

## Architecture
- 3-model XGBoost ensemble (Thailand-holdout, Colombia-holdout, all-data)
- Per-pixel features: 192-dim (AEF + AEF-2020 diff + AEF-year-1 diff)
- Consensus labels from RADD + GLAD-L + GLAD-S2
- Inference: average the 3 models, threshold, UTM postprocessing (close+open+area filter)

## Key files
- scripts/predict_baseline3_all.py - inference + submission pipeline (WORKING)
- src/train_baseline3.py - training script (MODIFY THIS FOR SUB 5)
- src/data_utils.py - postprocess_prediction function
- Bundle format: dict with "models" (co_holdout, th_holdout, all_data), "feature_spec", etc.

## Label encoding
- RADD: val = confidence_digit * 10000 + days_since_2014-12-31. Confidence 2=low, 3=high.
- GLAD-L: 0=none, 2=probable, 3=confirmed
- GLAD-S2: 0=none, 1=single-obs, 2=low, 3=medium, 4=high
- GLAD-S2 missing for tiles 47/48 (Thailand/SE Asia) — must handle

## Training data
- Thailand tiles (MGRS 47, 48): 48 tiles
- Colombia tiles (MGRS 18, 19): 48 tiles
- Each tile: AEF embeddings (64-dim × n_years), labels per pixel

## Environment
- Colab. Data at /content/drive/MyDrive/makeathon-challenge/
- Python 3.12, XGBoost, rasterio, geopandas
- numpy<2 (ABI issue)

## What NOT to change
- Feature extraction (stay at 192-dim AEF-only)
- Model architecture (XGBClassifier with default-ish params)
- Inference script (already working, don't break it)

Hackathon deadline mode. I'm switching from Copilot mid-task. Repo context:

STATE:
- Baseline 3 (XGBoost ensemble on AEF features, 3 models: TH-holdout, CO-holdout, all-data)
- Current best: 44.98% Union IoU on leaderboard (submission 4)
- Goal: reach 50%+ in 2-3 submissions
- Submissions left: 6 of 10

KEY FILES:
- src/train_baseline3.py — training script
- scripts/predict_baseline3_all.py — inference, produces GeoJSON submission
- src/data_utils.py — shared utilities, has postprocess_prediction()
- Cache location: /content/drive/MyDrive/cache/ (or similar)
- Bundle: /content/drive/MyDrive/artifacts/baseline3_aef_xgb_ensemble.joblib

WHAT WORKED (don't undo):
- UTM-space postprocessing (reproject 4326 → UTM → close 5x5 → open 3x3 → area filter → polygonize → back to 4326)
- min-area-ha 1.0
- ensemble-mode geomean
- threshold 0.50

NEXT TASK: Add label confidence weighting to training.
- Use RADD/GLAD-L/GLAD-S2 confidence values (not just binary) to compute per-pixel confidence
- Pass as sample_weight to XGBoost.fit()
- Retrain bundle as baseline3_weighted.joblib
- DO NOT change features, architecture, or the 3-model ensemble structure

Ask me questions before writing code. Read the existing scripts first to match their style.


