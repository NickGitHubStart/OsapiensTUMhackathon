# submissions/

The single GeoJSON file we have submitted that produced our current best leaderboard result.

| File | Sub # | Approach | Union IoU | Recall | FPR | Year |
|---|---|---|---|---|---|---|
| `sub3_baseline3_44.98pct.geojson` | 3 | Baseline 3 XGB ensemble (geometric mean) · threshold 0.50 · UTM closing 5×5 + opening 3×3 · min area 1.0 ha · single `time_step=null` | **44.98 %** | 70.52 % | 44.61 % | 0 % |
| `sub6_baseline3_peryear.geojson`  | 6 (candidate) | **Same model and same post-processing as Sub 3**, but predictions are made per year (2021–2025), polygons get a `time_step` of `YY06` for the year of maximum probability inside their footprint. Built with `python -m scripts.predict_polygon`. | _(pending)_ | _(pending)_ | _(pending)_ | _expected > 0_ |

Earlier baseline submissions (Sub 1 = 32.79 %, Sub 2 = 38.09 %) used the same model with progressively tighter post-processing — they are documented in the top-level `README.md` but not stored as separate files because the model behind them is identical to Sub 3.

Submissions 4 and 5 (the U-Net ensemble experiments, scoring 36.99 % and 35.20 %) did not improve on Sub 3 and have been removed. The lessons from those attempts are summarised under "Learnings" in the top-level `README.md`.

Why Sub 6 should beat Sub 3: the leaderboard "Year" component scored 0 on Sub 3 because every polygon was emitted with `time_step=null`. Sub 6 keeps the exact same Union IoU recipe (same model bundle, same threshold, same UTM morphology, same area filter) and only adds the temporal label, so the Union IoU is expected to stay essentially the same as 44.98 % while the Year metric goes from 0 to a non-zero value.

## Submission format (Osapiens spec)

Every file here is validated against:

- `.geojson` extension.
- Top-level `FeatureCollection`.
- All features are `Polygon` or `MultiPolygon`.
- `properties.time_step` is either a valid `YYMM` integer (e.g. `2204` for April 2022) or `null` (allowed by spec).
