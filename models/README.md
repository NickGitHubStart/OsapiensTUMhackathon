# models/

Only one family of models is kept here: the three XGBoost classifiers behind Submissions 1–3. All other model families we tried (cross-region U-Nets, distilled students, the wider "ultimate" student) **did not improve on the leaderboard** and have been removed from the repo. See the "Learnings" section in the top-level `README.md` for what we tried and why it did not work.

## `baseline_xgb/` — current LB best at 44.98 %

| File | Used for | Notes |
|---|---|---|
| `baseline_aef_logreg.{joblib,json}` | **Baseline 1** — AEF only, single year (64 features per pixel) | Validation F1 ≈ 0.79 (same-region). Sanity-check that the foundation embedding alone is a strong signal. |
| `baseline2_aef_logreg.{joblib,json}` | **Baseline 2** — AEF + temporal differences (192 features per pixel) | Validation F1 ≈ 0.87 (same-region). Confirms that temporal change relative to the 2020 forest baseline is the real deforestation signal. |
| `baseline3_aef_logreg.{joblib,json}` | **Baseline 3** — regional ensemble (Thailand-holdout, Colombia-holdout, all-data) | Cross-region F1 ≈ 0.65 (Thailand) / 0.59 (Colombia). The model behind our **44.98 %** leaderboard score (Sub 3). |

The file names contain `_logreg` for legacy reasons — these are XGBoost classifiers. The `.json` companion files describe the training configuration of each model.

## How to load

```python
import joblib
clf = joblib.load("models/baseline_xgb/baseline3_aef_logreg.joblib")
proba = clf.predict_proba(features)[:, 1]   # P(deforestation) per pixel
```

For Submission 3 we combined the three Baseline 3 models with the **geometric mean** at inference (penalises model disagreement more than arithmetic mean). See `src/train_baseline3.py` and `scripts/predict_polygon.py` for the exact pipeline.
