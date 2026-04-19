# Osapiens Terra — Deforestation Detection (TUM Makeathon 2026)

End-to-end pipeline for the Osapiens Terra deforestation challenge: predict deforestation polygons across 2021–2025 in unseen tiles after training only on Asia (Thailand) and Americas (Colombia). The hidden test set includes Africa.

> **Best leaderboard score: `44.98 %` Union IoU** — Submission 3, regional XGBoost ensemble + UTM post-processing.

---

## Repository layout

```
.
├── README.md
├── pyproject.toml             # python 3.10, deps via uv
├── data/                      # downloaded challenge dataset (S3, gitignored)
├── metadata/                  # train_tiles.geojson, test_tiles.geojson (tile AOIs)
├── models/
│   ├── README.md
│   └── baseline_xgb/          # the 3 XGBoost models behind Subs 1–3
│       ├── baseline_aef_logreg.{joblib,json}    # Baseline 1
│       ├── baseline2_aef_logreg.{joblib,json}   # Baseline 2
│       └── baseline3_aef_logreg.{joblib,json}   # Baseline 3 (current best)
├── src/
│   ├── data_utils.py          # raster I/O, label rasterisation
│   ├── download_data.py       # pull S3 dataset
│   ├── enhanced_labels.py     # weak-label loaders (RADD, GLAD-S2, GLAD-L, xregion)
│   ├── train_baseline.py      # Baseline 1
│   ├── train_baseline2.py     # Baseline 2
│   └── train_baseline3.py     # Baseline 3 (regional ensemble — current best)
├── scripts/
│   └── predict_polygon.py     # turn raster predictions into a GeoJSON submission
└── submissions/
    ├── README.md
    ├── sub3_baseline3_44.98pct.geojson         # 44.98 % leaderboard file
    ├── sub6_baseline3_peryear.geojson          # 40.71 % per-year polygons
    ├── sub7_sub3polys_with_timestep.geojson    # Sub 3 polys + time_step (candidate)
    └── sub8_baseline3plus_singlemask.geojson   # Baseline 3+ + time_step (candidate)
```

---

## Data distribution

The S3 dataset (`data/makeathon-challenge/`, ~43 GB) is structured as:

```
data/makeathon-challenge/
├── aef-embeddings/                 # AlphaEarth foundation model output
│   ├── train/  (16 tiles × 6 years)   # 2020–2025, 64 channels per tile-year
│   └── test/   (5 tiles × 6 years)
├── sentinel-1/                     # SAR backscatter (RTC) — multiple orbits per month
│   ├── train/  (~2575 scenes total, ~116 per tile)
│   └── test/
├── sentinel-2/                     # L2A monthly composites
│   ├── train/  (~71 monthly composites per tile)
│   └── test/
└── labels/                         # weak labels (only train tiles)
    ├── radd/        # 16 / 16 train tiles
    ├── gladl/       # 16 / 16 train tiles
    └── glads2/      #  8 / 16 train tiles
```

**Tiles**

| Split | # tiles | Tile naming | Region(s) |
|---|---|---|---|
| Train | 16 | e.g. `18NWG_6_6` (Colombia), `47QQV_2_4` (Thailand) | Colombia + Thailand |
| Test | 5 | `18NVJ_1_6`, `18NYH_2_1`, `33NTE_5_1`, `47QMA_6_2`, `48PWA_0_6` | Colombia, **Cameroon (Africa, OOD)**, Thailand |

The hidden Africa tile (`33NTE_5_1`) is the dominant generalisation challenge — it does not appear in any training data.

**Labels**

A pixel is labelled positive (`1` = deforestation) if **2 of 3** weak sources (RADD, GLAD-L, GLAD-S2) agree post-2020. Where GLAD-S2 is missing (8 of 16 tiles), the `xregion` strategy uses a 2-of-2 rule on RADD + GLAD-L instead. The metric (`Union IoU`) is computed against a hidden ground truth that we never see.

---

## Approach (from the slides)

### Baseline 1 — AEF only, single year

- **Input**: AlphaEarth Foundations (AEF) embeddings — 64-dimensional features per pixel from a single target year.
- **Model**: Pixel-wise XGBoost classifier.
- **Validation F1 (Class 1)**: ~0.79 (same-region held-out pixels).
- **Notes**: AEF is a pretrained global foundation model that encodes multimodal satellite information into compact 64-dim vectors per pixel. Simplest possible starting point.

### Baseline 2 — AEF + temporal differences

- **Input**: 192-dimensional feature vector per pixel:
  - `AEF[Y]` — current state (64 dims)
  - `AEF[Y] − AEF[2020]` — cumulative change from baseline (64 dims)
  - `AEF[Y] − AEF[Y-1]` — year-over-year change (64 dims)
- **Model**: Pixel-wise XGBoost classifier.
- **Validation F1 (Class 1)**: ~0.87 (same-region held-out pixels).
- **Notes**: The jump from 0.79 → 0.87 confirms that temporal change (especially relative to the 2020 forest baseline) is the key deforestation signal, not the static land-cover state.

### Baseline 3 — Regional ensemble

- **Input**: Same 192-dim temporal feature vector as Baseline 2.
- **Model**: Ensemble of 3 XGBoost models:
  1. Thailand-holdout model (trained on Colombia only)
  2. Colombia-holdout model (trained on Thailand only)
  3. All-data model (trained on both regions)

  At inference the per-pixel probabilities from all three are combined.
- **Cross-region F1 (Class 1)**:
  - Thailand holdout: ~0.65
  - Colombia holdout: ~0.59
- **Notes**: Cross-region validation is harsher than same-region (Baselines 1/2) but realistic for Africa. Regional ensembling reduces overfitting to any single biome.

---

## Leaderboard submissions

All three confirmed leaderboard results below use the **Baseline 3 regional XGB ensemble**. Only the threshold and post-processing change.

### Submission 1 — sanity check (32.79 %)

- Threshold 0.45 · arithmetic mean ensemble · min area 0.5 ha · no spatial post-processing.
- Union IoU **32.79 %** · Recall 79.06 % · FPR 64.09 %.
- High recall, very high FPR — the model finds most deforestation but also flags a lot of noise.

### Submission 2 — threshold tuning (38.09 %, +5.3)

- Same model. Threshold raised 0.45 → **0.55**. Everything else identical.
- Union IoU **38.09 %** · Recall 76.11 % · FPR 56.74 %.
- A simple threshold increase removes many low-confidence false positives with only a small drop in recall. Quick and effective gain.

### Submission 3 — UTM post-processing (44.98 %, +6.9) ⭐ CURRENT BEST

- Threshold 0.50 · **geometric mean** of the 3 model probabilities (penalises model disagreement) · reproject to local UTM CRS at 10 m to match Sentinel-2 · **morphological closing 5×5** + **opening 3×3** · min polygon area **1.0 ha** computed correctly in UTM as `abs(transform.a × transform.e) / 10000`.
- Union IoU **44.98 %** · Recall 70.52 % · FPR 44.61 %. Year **0 %** (single submission with `time_step=null`).
- Proper spatial post-processing in the correct coordinate system — the largest single jump and the current top score.

### Submission 6 — Sub 3 recipe + per-year polygons (40.71 %, −4.3)

- **Same Baseline 3 ensemble, same geometric mean, same threshold 0.50, same UTM closing 5×5 + opening 3×3, same 1.0 ha min area.** The change was that inference was done **per year** (732 polygons, one per year of max prob per pixel) instead of a single fused mask.
- File: `submissions/sub6_baseline3_peryear.geojson`.
- Result: Union IoU **40.71 %** · Recall 72.78 % · FPR 51.98 % · Year **7.31 %**.
- **Lesson**: per-year polygonisation injects extra false positives (FPR up 7 pts vs Sub 3) — splitting the mask 5 ways is not free. The Year metric finally went non-zero, but the Union IoU loss outweighed the gain. Conclusion: keep Sub 3's single fused mask, just add `time_step` on top — that's Sub 7.

### Submission 7 — Sub 3 polygons, relabelled (candidate)

- **Byte-identical 711 polygons of Sub 3** (so the proven 44.98 % spatial mask is preserved exactly), with `properties.time_step = YY06` set per polygon by computing the **mean Baseline 3 probability inside each polygon for each year 2021–2025 and picking the argmax year**.
- File: `submissions/sub7_sub3polys_with_timestep.geojson` (711 polygons, all `Polygon`).
- Year split: 2106=77, 2206=81, 2306=208, 2406=118, 2506=227.
- **Expectation**: Union IoU ≈ **44.98 %** (geometry is identical to Sub 3), Recall ≈ 70.5 %, FPR ≈ 44.6 %, Year **>0** (vs Sub 3's 0). **Why**: the spatial mask is unchanged, so the Union IoU / Recall / FPR are mathematically the same as Sub 3; the only added information is the temporal label, which is computed from the same model probabilities Sub 3 was generated from, and uses the most defensible heuristic ("the year the model is most confident inside this polygon") rather than a noisy "first crossing" signal.

### Submission 8 — Baseline 3+, single fused mask + per-polygon `time_step` (candidate)

- New, beefier model: `baseline3plus_aef_xgb.joblib` — **800 trees, max_depth 8, learning rate 0.04, 1.5 M training samples** (vs 400 trees, depth 6, lr 0.05, 200 K samples for the original Baseline 3). Cross-region F1: Thailand 0.612 (P 0.874 · R 0.471), Colombia 0.593 (P 0.909 · R 0.440) — **higher precision, lower recall** than Baseline 3.
- Built with the **same Sub 3 post-processing recipe**: per-year geom-mean → max over years → threshold 0.50 → UTM closing 5×5 + opening 3×3 → 1.0 ha min area. Then per-polygon `time_step` from year-of-max mean prob inside polygon (same logic as Sub 7).
- File: `submissions/sub8_baseline3plus_singlemask.geojson` (674 polygons — 5 % fewer than Sub 3, the model is more conservative).
- Year split: 2106=42, 2206=82, 2306=80, 2406=252, 2506=218.
- **Expectation**: a small Union IoU gain over Sub 3 — likely **45–47 %**. **Why**: Sub 3's main weakness was FPR 44.61 %; the new model's higher precision (0.87–0.91 vs 0.86–0.93) and noticeably lower recall mean it should produce fewer false-positive polygons (already visible: 5 % fewer polygons than Sub 3). Plus a non-zero Year score from the same `time_step` strategy as Sub 7. **Risk**: if recall drops too far on the Africa OOD tile (33NTE), Union IoU could *also* drop — that's the only scenario where Sub 8 underperforms Sub 3. We rate Sub 7 as the safer bet (high confidence, ≈ 44.98 % + Year), Sub 8 as the higher-upside bet (45–47 % + Year).

---

## Quick start

```bash
# 1. install deps
uv sync                              # or: pip install -e .

# 2. download the challenge dataset (~43 GB)
python -m src.download_data --out data/makeathon-challenge

# 3. train all three baselines
python -m src.train_baseline                  # Baseline 1
python -m src.train_baseline2                 # Baseline 2
python -m src.train_baseline3                 # Baseline 3 (current best)

# 4. produce a submission with the proven post-processing
#    (per-year predictions + per-polygon time_step in YY06 format)
python -m scripts.predict_polygon \
       --data-dir ./data/makeathon-challenge \
       --bundle models/baseline_xgb/baseline3_aef_logreg.joblib \
       --out submissions/my_submission.geojson \
       --threshold 0.50 --min-area-ha 1.0 \
       --closing 5 --opening 3 --month 6 \
       --years 2021,2022,2023,2024,2025
```

Hardware: AMD MI300X (192 GB HBM) was used for training, but XGBoost runs fine on CPU; the entire training pipeline takes ~30 minutes on a single machine.

---

## Learnings from things that did **not** improve the score

These are kept here so anyone picking up the project knows where the dead ends are.

1. **An 11-model U-Net ensemble does not automatically beat a well-tuned XGBoost baseline on this task.** We trained 8 cross-region U-Nets on the 192-channel temporal AEF stack, then 2 distilled students from their soft labels, then a wider "ultimate" student. Local cross-region IoU reached **0.622**, but the actual leaderboard Union IoU was **36.99 %** (Sub 4) and **35.20 %** (Sub 5), both well below the 44.98 % XGB baseline. The U-Nets overfit cross-region validation in a way that does not transfer to Africa.

2. **`time_step` matters.** Submissions without per-polygon `time_step` get **Year = 0 %**. Even our 35.20 % submission (which had per-year `time_step` in YYMM format) only got Year = 18 %, because we used the "first year of detection" as the timestamp and that is wrong for events occurring later in the time series. A correct strategy is to assign each polygon to the year with the **maximum** prediction probability, not the first crossing.

3. **`cap_frac` per-tile (cap on positive fraction) hurts on the Africa OOD tile.** This was added to suppress over-prediction on tiles where the model was uncalibrated, but on Africa where the model is **under-calibrated** it suppressed real positives too. The proven Sub 3 recipe uses no such cap.

4. **Cross-region holdout is a noisy proxy for the Africa test.** Local IoU 0.622 (cross-region) → 0.35 leaderboard. Use it for **direction**, not for **magnitude**. The XGB cross-region F1 ≈ 0.6 actually predicted Africa performance much better than U-Net cross-region IoU 0.62.

5. **AEF data quality is uneven across tiles.** A few test tiles (notably `47QMA_6_2`) had a small fraction of NaN pixels in the AEF embeddings. Our pipeline correctly masks these to zero, but it limits how confidently any model can predict on those tiles regardless of architecture.

6. **The biggest single leaderboard jump was post-processing, not modelling.** Sub 2 → Sub 3 = +6.9 IoU just from switching to geometric mean + UTM-space morphology + 1.0 ha area filter. Spending compute on better post-processing pays off more than spending it on bigger models, on this dataset.

---

## Submission format (Osapiens spec)

Every file we submit satisfies:

- `.geojson` extension.
- Top-level `FeatureCollection`.
- All features are `Polygon` or `MultiPolygon`.
- `properties.time_step` is either a valid `YYMM` integer (e.g. `2204` for April 2022) or `null` (allowed by spec).

`scripts/predict_polygon.py` enforces all four rules.
