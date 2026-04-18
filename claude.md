The real problem (what the jury will actually score):
The hard constraint is that test data is in Africa but training has none. So generalization dominates everything. A model that scores 0.85 on Thailand but 0.4 on unseen African biomes will lose to a model that scores 0.7 on both. This should shape every decision — architecture, augmentation, feature choice, validation strategy.
The second hard constraint is noisy, conflicting labels. This means you need to not blindly trust any single label source, and your loss function / training regime needs to account for it.
My strategic take on how to attack it:
Lean heavily on the foundation model embeddings. They're the single biggest generalization lever you have — models like Prithvi, Clay, Presto, SatMAE have seen global data and encode biome-agnostic features. Raw S1/S2 pixels tend to overfit to regional spectral signatures. Embeddings abstract that away.
Build a strong dumb baseline first. Gradient boosting (LightGBM/XGBoost) on per-pixel foundation embeddings + a handful of hand-engineered temporal features (NDVI delta, max NDVI drop, SAR backscatter change, temporal variance) is often within 5% of a U-Net and takes 2 hours instead of 2 days. Don't skip this — it anchors everything.
For the segmentation model, a U-Net with multimodal input channels and temporal modeling (either 3D conv, ConvLSTM, or temporal self-attention per pixel before a 2D decoder) is the standard play. If you want to be fancy, fine-tune Prithvi or Clay with a segmentation head — they already handle multi-temporal inputs.
For noisy labels, pick one or combine:

Treat each label source as a separate output head (multi-task), the model learns shared features that agree
Consensus labels (only train where ≥2 sources agree; treat disagreement pixels as "uncertain" with lower weight)
Co-teaching / loss truncation (discard the top X% highest-loss samples per batch)
Label smoothing + symmetric cross-entropy

For generalization, heavy augmentation is essential: spectral jitter, random biome-style color transforms, cutmix between regions, temporal shuffling. Validate by holding out entire regions, not random pixels — if your Thailand-trained model drops 15 points on Colombia-held-out, that's your real Africa score estimate.
Bonus "when did it happen": per-pixel change-point detection on the time series. BFAST is the classical approach; a transformer over the time series with temporal attention weights often works better and gives you free interpretability.


Submission budget (rough plan):

Sub 1 — Dumb baseline sanity check. LightGBM on foundation embeddings, minimal features. Confirms your pipeline works end-to-end (data loading → prediction → submission format) and calibrates what "Africa score" looks like vs. your local validation. The gap between this number and your local CV tells you how much your validation is lying to you.
Sub 2 — Strong single model. Your best U-Net / fine-tuned foundation model after local iteration. Confirms the architecture jump is real.
Subs 3–7 — Targeted experiments. Each one should test one hypothesis where local validation is ambiguous. Examples: "does adding S1 actually help on unseen regions?", "does consensus labeling beat multi-head?", "does TTA help?". Don't submit if you can answer it locally.
Subs 8–9 — Ensembles / combinations. Only after you know which individual components work.
Sub 10 — Final submission. Your best ensemble, held in reserve.

Local validation is now everything. Build a held-out validation split that mimics the test distribution as closely as possible. Since test is Africa and you have Thailand + Colombia + one other region, do leave-one-region-out cross-validation. Train on two regions, validate on the third, rotate. Your generalization score is the worst of the three folds, not the average — because Africa might be more different than any training region is from the others.
Don't chase the leaderboard. If a submission scores worse than expected, resist the urge to immediately submit a "fix". Diagnose locally. Each submission should teach you something you couldn't learn otherwise.
Keep a submission log. For each sub, write down: what changed, local CV score, leaderboard score, gap. After 3-4 submissions you'll have a calibration curve between your CV and the real score — this is gold.
Ensemble cheaply. Average predictions from your LOROCV folds (you already trained 3 models, one per held-out region) as your final submission. Free ensemble, usually +1-3% over any single model, and it's naturally regularized toward generalization.
One tactical note: if submissions are limited but you can submit a "dummy" to check format, do that first with all zeros or a trivial prediction so you don't waste Sub 1 on a format bug. Check the rules — some competitions let you submit a baseline without it counting.


Let me see the middle section I missed — AlphaEarth embeddings and S1 details matter a lot here.Okay, reading the notebook shifts my advice meaningfully. There are specific things here that are easy to miss and will silently destroy your score if you don't handle them. Let me flag the critical stuff first, then answer the repo/Colab question concretely.

## Things the notebook reveals that change the plan

**1. AlphaEarth (AEF) is your single biggest asset — and it's easy to misuse.**
AEF is Google's 2025 foundation model (the paper is linked). It's 64-dim embeddings trained globally on S1+S2+more, *annual* cadence, *EPSG:4326* CRS. Two implications:
- It's already seen Africa-like biomes in pretraining. This is your generalization engine. A model that leans heavily on AEF will transfer to the Africa test set far better than one trained on raw S1/S2 pixels.
- It's in a different CRS than S1/S2 (which are UTM). If you don't reproject carefully, you'll train on misaligned data and not notice. Pick one CRS (I'd pick the tile's UTM) and resample *everything* to it on load.

**2. Deforestation is only counted after 2020 AND where the pixel was forest in 2020.** You need a **forest-in-2020 mask**. Don't try to learn this from weak labels — use an external source:
- **ESA WorldCover 2020** (10m, free, global, `forest = class 10`) is the cleanest option. Download once, crop per tile.
- Or Hansen Global Forest Change `treecover2000` + `lossyear < 2020` to derive "still forest in 2020".

Without this mask, you'll get hammered by false positives in areas that were already cleared before 2020 but where GLAD/RADD flag re-clearing or regrowth cycles.

**3. The RADD encoding is a trap.** A value like `31847` is **not** a class — it's `confidence_digit * 10000 + days_since_2014_12_31`. You need to decode:
```python
confidence = radd_val // 10000       # 2 = low, 3 = high
days       = radd_val %  10000       # since 2014-12-31
alert_date = pd.Timestamp("2014-12-31") + pd.Timedelta(days=days)
post_2020  = alert_date >= pd.Timestamp("2020-01-01")
```
Lots of teams will treat the raw integer as a class and get garbage.

**4. Consensus labels across the 3 sources is the cleanest supervision.** The three sensors fail in different ways (RADD = radar geometry noise, GLAD-S2/L = clouds). A pixel where ≥2 sources agree on deforestation, *and* was forest in 2020, is a high-quality positive. A pixel where all 3 say "no alert" for years 2021–2024 is a high-quality negative. Everything else is uncertain — either discard or weight down. This alone can be worth several points.

**5. Submission is polygons, not a raster.** So your post-processing matters:
- Morphological opening (remove single-pixel noise)
- Filter polygons below some area threshold (example shows 0.5 ha min)
- The `raster_to_geojson` utility handles the format, but garbage in = garbage polygons out

**6. S2 months are cloudy/missing.** "Single best cloud-free scene per month, not mosaicked" means some months are great, some are junk. Your time-series model must handle variable validity per month (mask tokens, use attention, or precompute annual composites).

## Concrete repo structure for Colab portability

Given what I now see (data on S3 with a `make download_data_from_s3` target), here's what I'd build:

```
repo/
├── configs/
│   ├── paths.local.yaml       # /data/makeathon-challenge
│   ├── paths.colab.yaml       # /content/drive/MyDrive/... OR /content/data
│   └── paths.kaggle.yaml      # /kaggle/input/osapiens-data
├── src/
│   ├── data/
│   │   ├── io.py              # rasterio readers, always reprojects to tile UTM
│   │   ├── labels.py          # decode RADD/GLAD, build consensus masks
│   │   ├── forest_mask.py     # ESA WorldCover 2020 → per-tile forest mask
│   │   └── dataset.py         # PyTorch Dataset, returns (modalities, labels, mask)
│   ├── models/
│   │   ├── aef_head.py        # lightweight model on AEF embeddings (generalizes best)
│   │   └── unet_temporal.py   # S1+S2 temporal U-Net (fits more, overfits region more)
│   ├── train.py
│   ├── predict.py
│   └── postprocess.py         # raster → filtered polygons → GeoJSON
├── notebooks/
│   ├── 00_eda.ipynb
│   └── colab_runner.ipynb     # 20-line wrapper: pip install, set config, call train.py
├── submission_utils.py        # provided, don't modify
├── Makefile
└── requirements.txt
```

The key discipline: **zero ML logic in notebooks.** The notebook does `!git clone`, `!pip install`, mounts Drive, points to a config, and calls `python -m src.train --config configs/paths.colab.yaml`. Everything else is in `.py` files you can version, test, and run anywhere.

## One thing I'd strongly reconsider about Colab

60GB + monthly time series + 3 label sources = **you will be I/O-bound on Colab**, especially with Drive-mounted data. Before committing:

**Precompute a per-tile cached tensor.** For each training tile, precompute once:
- AEF stack: `(n_years, 64, H, W)` — small, load once per tile
- S2 annual composites: median of cloud-free months per year → `(n_years, 12, H, W)` instead of `(n_years, 12_months, 12_bands, H, W)`. Cuts data by 12x.
- S1 annual mean + std of VV backscatter → `(n_years, 2, H, W)`
- Consensus label mask + forest-2020 mask: `(n_years, H, W)` + `(H, W)`

Save as `.npz` or `.zarr`. Your dataset probably shrinks from 60GB → 5–10GB, loads instantly on Colab, and you can iterate 20x faster. For the "when did it happen" bonus, keep the monthly S1/S2 for tiles you think have events and only load those.

This is probably the single highest-ROI engineering decision on the whole project.

---

Want me to actually write:
- (a) the `labels.py` with RADD/GLAD decoding and consensus-mask construction,
- (b) the precompute-cache script that turns 60GB → ~8GB, or
- (c) the `colab_runner.ipynb` + config pattern?

Pick whichever is blocking you most and I'll draft it.


xgboost results:
INFO:__main__:Validation report:
              precision    recall  f1-score   support

           0       0.86      0.93      0.89     28137
           1       0.79      0.63      0.70     11863

    accuracy                           0.84     40000
   macro avg       0.82      0.78      0.80     40000
weighted avg       0.84      0.84      0.84     40000

INFO:__main__:Saved model to artifacts/baseline_aef_logreg.joblib


downloaded files using  directory structure
```

== aef-embeddings/test ==
files: 30
 - 33NTE_5_1_2020.tiff
 - 18NYH_2_1_2021.tiff
 - 18NVJ_1_6_2024.tiff
 - 18NYH_2_1_2025.tiff
 - 33NTE_5_1_2021.tiff
 - 33NTE_5_1_2023.tiff
 - 18NYH_2_1_2024.tiff
 - 18NVJ_1_6_2022.tiff
 - 33NTE_5_1_2024.tiff
 - 18NVJ_1_6_2025.tiff

== aef-embeddings/train ==
files: 96
 - 48PXC_7_7_2020.tiff
 - 48QWD_2_2_2022.tiff
 - 48PXC_7_7_2021.tiff
 - 18NWG_6_6_2024.tiff
 - 18NWJ_8_9_2025.tiff
 - 18NXJ_7_6_2025.tiff
 - 18NWH_1_4_2021.tiff
 - 18NXH_6_8_2025.tiff
 - 48PXC_7_7_2025.tiff
 - 47QQV_2_4_2021.tiff

== labels/train/gladl ==
files: 160
 - gladl_18NXH_6_8_alert25.tif
 - gladl_18NYH_9_9_alertDate21.tif
 - gladl_18NYH_9_9_alert22.tif
 - gladl_18NXH_6_8_alert21.tif
 - gladl_18NWH_1_4_alertDate23.tif
 - gladl_18NWM_9_4_alert25.tif
 - gladl_48PUT_0_8_alert25.tif
 - gladl_48PWV_7_8_alert25.tif
 - gladl_18NWG_6_6_alertDate25.tif
 - gladl_18NXJ_7_6_alert24.tif

== labels/train/glads2 ==
files: 16
 - glads2_18NWJ_8_9_alertDate.tif
 - glads2_18NXH_6_8_alert.tif
 - glads2_19NBD_4_4_alert.tif
 - glads2_18NWH_1_4_alert.tif
 - glads2_18NWG_6_6_alert.tif
 - glads2_18NWH_1_4_alertDate.tif
 - glads2_18NXJ_7_6_alert.tif
 - glads2_18NWG_6_6_alertDate.tif
 - glads2_18NWJ_8_9_alert.tif
 - glads2_18NWM_9_4_alert.tif

== labels/train/radd ==
files: 16
 - radd_18NXJ_7_6_labels.tif
 - radd_48PYB_3_6_labels.tif
 - radd_18NWJ_8_9_labels.tif
 - radd_48PXC_7_7_labels.tif
 - radd_48QVE_3_0_labels.tif
 - radd_18NYH_9_9_labels.tif
 - radd_19NBD_4_4_labels.tif
 - radd_18NWM_9_4_labels.tif
 - radd_18NXH_6_8_labels.tif
 - radd_48PWV_7_8_labels.tif

== metadata ==
files: 2
 - test_tiles.geojson
 - train_tiles.geojson

== sentinel-1/test/18NVJ_1_6__s1_rtc ==
files: 101
 - 18NVJ_1_6__s1_rtc_2023_5_descending.tif
 - 18NVJ_1_6__s1_rtc_2025_6_descending.tif
 - 18NVJ_1_6__s1_rtc_2025_2_descending.tif
 - 18NVJ_1_6__s1_rtc_2023_1_descending.tif
 - 18NVJ_1_6__s1_rtc_2020_4_descending.tif
 - 18NVJ_1_6__s1_rtc_2025_7_descending.tif
 - 18NVJ_1_6__s1_rtc_2020_7_ascending.tif
 - 18NVJ_1_6__s1_rtc_2020_5_descending.tif
 - 18NVJ_1_6__s1_rtc_2024_9_descending.tif
 - 18NVJ_1_6__s1_rtc_2020_12_descending.tif

== sentinel-1/test/18NYH_2_1__s1_rtc ==
files: 102
 - 18NYH_2_1__s1_rtc_2022_8_descending.tif
 - 18NYH_2_1__s1_rtc_2020_6_ascending.tif
 - 18NYH_2_1__s1_rtc_2022_12_descending.tif
 - 18NYH_2_1__s1_rtc_2021_3_descending.tif
 - 18NYH_2_1__s1_rtc_2020_1_ascending.tif
 - 18NYH_2_1__s1_rtc_2021_7_ascending.tif
 - 18NYH_2_1__s1_rtc_2024_9_descending.tif
 - 18NYH_2_1__s1_rtc_2021_9_ascending.tif
 - 18NYH_2_1__s1_rtc_2024_7_descending.tif
 - 18NYH_2_1__s1_rtc_2025_5_descending.tif

== sentinel-1/test/33NTE_5_1__s1_rtc ==
files: 73
 - 33NTE_5_1__s1_rtc_2021_6_ascending.tif
 - 33NTE_5_1__s1_rtc_2022_10_ascending.tif
 - 33NTE_5_1__s1_rtc_2020_4_ascending.tif
 - 33NTE_5_1__s1_rtc_2025_11_ascending.tif
 - 33NTE_5_1__s1_rtc_2023_9_ascending.tif
 - 33NTE_5_1__s1_rtc_2024_3_ascending.tif
 - 33NTE_5_1__s1_rtc_2020_8_ascending.tif
 - 33NTE_5_1__s1_rtc_2020_1_ascending.tif
 - 33NTE_5_1__s1_rtc_2020_12_ascending.tif
 - 33NTE_5_1__s1_rtc_2021_5_ascending.tif

== sentinel-1/test/47QMA_6_2__s1_rtc ==
files: 143
 - 47QMA_6_2__s1_rtc_2025_10_ascending.tif
 - 47QMA_6_2__s1_rtc_2021_2_descending.tif
 - 47QMA_6_2__s1_rtc_2021_5_descending.tif
 - 47QMA_6_2__s1_rtc_2024_8_ascending.tif
 - 47QMA_6_2__s1_rtc_2020_9_descending.tif
 - 47QMA_6_2__s1_rtc_2023_8_ascending.tif
 - 47QMA_6_2__s1_rtc_2020_6_ascending.tif
 - 47QMA_6_2__s1_rtc_2023_11_descending.tif
 - 47QMA_6_2__s1_rtc_2020_2_ascending.tif
 - 47QMA_6_2__s1_rtc_2021_8_descending.tif

== sentinel-1/test/48PWA_0_6__s1_rtc ==
files: 144
 - 48PWA_0_6__s1_rtc_2021_3_descending.tif
 - 48PWA_0_6__s1_rtc_2022_12_descending.tif
 - 48PWA_0_6__s1_rtc_2022_11_ascending.tif
 - 48PWA_0_6__s1_rtc_2022_4_descending.tif
 - 48PWA_0_6__s1_rtc_2021_7_descending.tif
 - 48PWA_0_6__s1_rtc_2023_2_ascending.tif
 - 48PWA_0_6__s1_rtc_2020_12_ascending.tif
 - 48PWA_0_6__s1_rtc_2023_3_ascending.tif
 - 48PWA_0_6__s1_rtc_2020_1_descending.tif
 - 48PWA_0_6__s1_rtc_2021_1_descending.tif

== sentinel-1/train/18NWG_6_6__s1_rtc ==
files: 102
 - 18NWG_6_6__s1_rtc_2021_2_ascending.tif
 - 18NWG_6_6__s1_rtc_2023_6_descending.tif
 - 18NWG_6_6__s1_rtc_2021_10_descending.tif
 - 18NWG_6_6__s1_rtc_2025_11_ascending.tif
 - 18NWG_6_6__s1_rtc_2021_4_ascending.tif
 - 18NWG_6_6__s1_rtc_2023_5_descending.tif
 - 18NWG_6_6__s1_rtc_2025_6_descending.tif
 - 18NWG_6_6__s1_rtc_2020_5_descending.tif
 - 18NWG_6_6__s1_rtc_2021_12_descending.tif
 - 18NWG_6_6__s1_rtc_2020_4_descending.tif

== sentinel-1/train/18NWH_1_4__s1_rtc ==
files: 102
 - 18NWH_1_4__s1_rtc_2021_5_ascending.tif
 - 18NWH_1_4__s1_rtc_2021_10_ascending.tif
 - 18NWH_1_4__s1_rtc_2025_11_ascending.tif
 - 18NWH_1_4__s1_rtc_2020_9_ascending.tif
 - 18NWH_1_4__s1_rtc_2025_8_descending.tif
 - 18NWH_1_4__s1_rtc_2023_11_descending.tif
 - 18NWH_1_4__s1_rtc_2021_6_ascending.tif
 - 18NWH_1_4__s1_rtc_2020_2_ascending.tif
 - 18NWH_1_4__s1_rtc_2025_3_descending.tif
 - 18NWH_1_4__s1_rtc_2022_8_descending.tif

== sentinel-1/train/18NWJ_8_9__s1_rtc ==
files: 102
 - 18NWJ_8_9__s1_rtc_2020_3_descending.tif
 - 18NWJ_8_9__s1_rtc_2023_3_descending.tif
 - 18NWJ_8_9__s1_rtc_2022_11_descending.tif
 - 18NWJ_8_9__s1_rtc_2021_10_descending.tif
 - 18NWJ_8_9__s1_rtc_2020_10_ascending.tif
 - 18NWJ_8_9__s1_rtc_2020_8_ascending.tif
 - 18NWJ_8_9__s1_rtc_2024_8_descending.tif
 - 18NWJ_8_9__s1_rtc_2023_2_descending.tif
 - 18NWJ_8_9__s1_rtc_2024_2_descending.tif
 - 18NWJ_8_9__s1_rtc_2024_12_descending.tif

== sentinel-1/train/18NWM_9_4__s1_rtc ==
files: 70
 - 18NWM_9_4__s1_rtc_2024_9_descending.tif
 - 18NWM_9_4__s1_rtc_2021_2_descending.tif
 - 18NWM_9_4__s1_rtc_2020_10_descending.tif
 - 18NWM_9_4__s1_rtc_2023_9_descending.tif
 - 18NWM_9_4__s1_rtc_2020_9_descending.tif
 - 18NWM_9_4__s1_rtc_2022_9_descending.tif
 - 18NWM_9_4__s1_rtc_2022_3_descending.tif
 - 18NWM_9_4__s1_rtc_2023_8_descending.tif
 - 18NWM_9_4__s1_rtc_2024_4_descending.tif
 - 18NWM_9_4__s1_rtc_2025_12_descending.tif

== sentinel-1/train/18NXH_6_8__s1_rtc ==
files: 102
 - 18NXH_6_8__s1_rtc_2020_1_descending.tif
 - 18NXH_6_8__s1_rtc_2022_4_descending.tif
 - 18NXH_6_8__s1_rtc_2025_9_descending.tif
 - 18NXH_6_8__s1_rtc_2025_7_descending.tif
 - 18NXH_6_8__s1_rtc_2020_7_ascending.tif
 - 18NXH_6_8__s1_rtc_2021_10_descending.tif
 - 18NXH_6_8__s1_rtc_2022_7_descending.tif
 - 18NXH_6_8__s1_rtc_2021_6_ascending.tif
 - 18NXH_6_8__s1_rtc_2022_5_descending.tif
 - 18NXH_6_8__s1_rtc_2025_12_descending.tif

== sentinel-1/train/18NXJ_7_6__s1_rtc ==
files: 102
 - 18NXJ_7_6__s1_rtc_2020_10_ascending.tif
 - 18NXJ_7_6__s1_rtc_2023_4_descending.tif
 - 18NXJ_7_6__s1_rtc_2021_12_descending.tif
 - 18NXJ_7_6__s1_rtc_2022_5_descending.tif
 - 18NXJ_7_6__s1_rtc_2022_9_descending.tif
 - 18NXJ_7_6__s1_rtc_2021_10_descending.tif
 - 18NXJ_7_6__s1_rtc_2021_1_ascending.tif
 - 18NXJ_7_6__s1_rtc_2024_11_descending.tif
 - 18NXJ_7_6__s1_rtc_2021_4_ascending.tif
 - 18NXJ_7_6__s1_rtc_2025_10_descending.tif

== sentinel-1/train/18NYH_9_9__s1_rtc ==
files: 102
 - 18NYH_9_9__s1_rtc_2021_2_ascending.tif
 - 18NYH_9_9__s1_rtc_2020_12_ascending.tif
 - 18NYH_9_9__s1_rtc_2025_2_descending.tif
 - 18NYH_9_9__s1_rtc_2025_12_descending.tif
 - 18NYH_9_9__s1_rtc_2022_5_descending.tif
 - 18NYH_9_9__s1_rtc_2025_1_descending.tif
 - 18NYH_9_9__s1_rtc_2022_4_descending.tif
 - 18NYH_9_9__s1_rtc_2022_10_descending.tif
 - 18NYH_9_9__s1_rtc_2020_3_descending.tif
 - 18NYH_9_9__s1_rtc_2020_4_ascending.tif

== sentinel-1/train/19NBD_4_4__s1_rtc ==
files: 59
 - 19NBD_4_4__s1_rtc_2025_10_descending.tif
 - 19NBD_4_4__s1_rtc_2025_6_ascending.tif
 - 19NBD_4_4__s1_rtc_2025_11_ascending.tif
 - 19NBD_4_4__s1_rtc_2025_1_descending.tif
 - 19NBD_4_4__s1_rtc_2020_4_ascending.tif
 - 19NBD_4_4__s1_rtc_2025_9_descending.tif
 - 19NBD_4_4__s1_rtc_2021_3_descending.tif
 - 19NBD_4_4__s1_rtc_2020_11_ascending.tif
 - 19NBD_4_4__s1_rtc_2021_4_descending.tif
 - 19NBD_4_4__s1_rtc_2021_12_ascending.tif

== sentinel-1/train/47QMB_0_8__s1_rtc ==
files: 143
 - 47QMB_0_8__s1_rtc_2020_6_descending.tif
 - 47QMB_0_8__s1_rtc_2023_11_ascending.tif
 - 47QMB_0_8__s1_rtc_2022_4_descending.tif
 - 47QMB_0_8__s1_rtc_2020_4_ascending.tif
 - 47QMB_0_8__s1_rtc_2024_12_descending.tif
 - 47QMB_0_8__s1_rtc_2021_3_descending.tif
 - 47QMB_0_8__s1_rtc_2022_12_descending.tif
 - 47QMB_0_8__s1_rtc_2024_11_ascending.tif
 - 47QMB_0_8__s1_rtc_2024_8_descending.tif
 - 47QMB_0_8__s1_rtc_2025_2_ascending.tif

== sentinel-1/train/47QQV_2_4__s1_rtc ==
files: 144
 - 47QQV_2_4__s1_rtc_2024_7_descending.tif
 - 47QQV_2_4__s1_rtc_2022_11_descending.tif
 - 47QQV_2_4__s1_rtc_2024_7_ascending.tif
 - 47QQV_2_4__s1_rtc_2024_12_ascending.tif
 - 47QQV_2_4__s1_rtc_2025_4_descending.tif
 - 47QQV_2_4__s1_rtc_2023_11_ascending.tif
 - 47QQV_2_4__s1_rtc_2024_8_ascending.tif
 - 47QQV_2_4__s1_rtc_2021_6_ascending.tif
 - 47QQV_2_4__s1_rtc_2025_4_ascending.tif
 - 47QQV_2_4__s1_rtc_2024_11_descending.tif

== sentinel-1/train/48PUT_0_8__s1_rtc ==
files: 142
 - 48PUT_0_8__s1_rtc_2023_3_descending.tif
 - 48PUT_0_8__s1_rtc_2020_8_ascending.tif
 - 48PUT_0_8__s1_rtc_2024_1_ascending.tif
 - 48PUT_0_8__s1_rtc_2025_8_ascending.tif
 - 48PUT_0_8__s1_rtc_2021_6_ascending.tif
 - 48PUT_0_8__s1_rtc_2024_11_descending.tif
 - 48PUT_0_8__s1_rtc_2022_12_ascending.tif
 - 48PUT_0_8__s1_rtc_2021_12_ascending.tif
 - 48PUT_0_8__s1_rtc_2021_8_ascending.tif
 - 48PUT_0_8__s1_rtc_2020_9_ascending.tif

== sentinel-1/train/48PWV_7_8__s1_rtc ==
files: 143
 - 48PWV_7_8__s1_rtc_2025_2_descending.tif
 - 48PWV_7_8__s1_rtc_2020_9_descending.tif
 - 48PWV_7_8__s1_rtc_2025_5_ascending.tif
 - 48PWV_7_8__s1_rtc_2025_7_descending.tif
 - 48PWV_7_8__s1_rtc_2023_5_descending.tif
 - 48PWV_7_8__s1_rtc_2024_3_ascending.tif
 - 48PWV_7_8__s1_rtc_2025_7_ascending.tif
 - 48PWV_7_8__s1_rtc_2020_12_descending.tif
 - 48PWV_7_8__s1_rtc_2024_12_ascending.tif
 - 48PWV_7_8__s1_rtc_2025_10_ascending.tif

== sentinel-1/train/48PXC_7_7__s1_rtc ==
files: 144
 - 48PXC_7_7__s1_rtc_2025_4_descending.tif
 - 48PXC_7_7__s1_rtc_2020_6_descending.tif
 - 48PXC_7_7__s1_rtc_2021_1_ascending.tif
 - 48PXC_7_7__s1_rtc_2020_3_descending.tif
 - 48PXC_7_7__s1_rtc_2021_6_ascending.tif
 - 48PXC_7_7__s1_rtc_2024_8_descending.tif
 - 48PXC_7_7__s1_rtc_2021_10_descending.tif
 - 48PXC_7_7__s1_rtc_2023_6_ascending.tif
 - 48PXC_7_7__s1_rtc_2023_10_descending.tif
 - 48PXC_7_7__s1_rtc_2025_6_descending.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_10_ascending ==
files: 1
 - 96c6ee62-fc9e-48fe-b135-f70ffa804058.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_11_ascending ==
files: 1
 - 5f077939-d1e5-4dff-931e-ab4d52593860.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_12_ascending ==
files: 1
 - 756f871e-bc75-491f-bb20-5429e93ef0ad.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_1_ascending ==
files: 1
 - 7f1aeb64-6cef-43e4-b1f9-e5fd7b69b4a8.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_2_ascending ==
files: 1
 - a492c55f-dd32-490f-924b-c23383178f5f.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_3_ascending ==
files: 1
 - 6e893e46-e14b-47ed-9090-5a303b58f484.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_4_ascending ==
files: 1
 - 59e30ad6-8a5c-405c-9727-60bfc0ee0d4e.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_5_ascending ==
files: 1
 - 28dca82a-9720-433a-a627-fc810a58e51d.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_6_ascending ==
files: 1
 - ea407978-9584-4c7f-9509-1ca96d24def4.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_7_ascending ==
files: 1
 - 567c855a-3848-42c0-8cf3-4c225399ba83.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_8_ascending ==
files: 1
 - fa1e2502-8e2f-4155-8528-4a4e390ef57d.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_9_ascending ==
files: 1
 - 6ff013ef-a114-4e5e-b40a-9b8809a965f8.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_10_ascending ==
files: 1
 - a4d00d59-7833-45ac-a3bc-6e749a43ae00.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_11_ascending ==
files: 1
 - 26315890-75a8-4600-ae6b-9479c172870a.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_12_ascending ==
files: 1
 - fb4233a0-3b69-4056-8a25-1fb1a725f45c.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_1_ascending ==
files: 1
 - 1b110c63-9946-45d3-b980-a72b61484b86.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_2_ascending ==
files: 1
 - ff8a6f3a-19e5-428b-aa40-c7bead608ad0.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_3_ascending ==
files: 1
 - 7e5f892c-1020-4f7e-a97e-9e8fcba8e8cd.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_4_ascending ==
files: 1
 - f9f15b88-e1c2-4dd3-b656-56b75481c430.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_5_ascending ==
files: 1
 - 9e1549e6-a98b-47b4-964d-43348c500858.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_6_ascending ==
files: 1
 - 03ad30c8-1cc2-4ff3-b974-ff6b3ff58f76.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_7_ascending ==
files: 1
 - f1a3cd13-3bc9-499e-935d-944c3ee58154.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_8_ascending ==
files: 1
 - b8e2bc9d-fcc9-4106-9446-5a1232d2ee90.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2021_9_ascending ==
files: 1
 - 83e43265-9ae7-4825-a8a5-b9ca954e7175.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_10_ascending ==
files: 1
 - bf344639-e729-461f-900c-6a90e5d6f4cf.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_11_ascending ==
files: 1
 - 6cbad00f-ce89-4b1f-b886-d817f8ac63a8.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_12_ascending ==
files: 1
 - dca7c018-b5b1-402e-bdeb-109453e6d0af.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_1_ascending ==
files: 1
 - aa47b56d-3a65-4673-a81b-85ba839137c7.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_2_ascending ==
files: 1
 - 8e98a612-a6f5-43b8-a15e-2cbb6b5dccd0.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_3_ascending ==
files: 1
 - a7f67b94-09c6-4e99-a960-8763cea51215.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_4_ascending ==
files: 1
 - ed18bbc8-1a13-4142-8e17-1d74b08aff53.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_5_ascending ==
files: 1
 - beb1629d-1218-441f-9737-5890de20347f.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_6_ascending ==
files: 1
 - a64761b2-5553-4c5c-87ff-6ced69a358ca.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_7_ascending ==
files: 1
 - 1e3de866-16ee-4cc6-942c-537e3d9bfb4b.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_8_ascending ==
files: 1
 - 7a094084-5696-4aaa-b2d2-546ccf875077.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2022_9_ascending ==
files: 1
 - 1822394e-3117-42d6-9cf1-f04d02733b55.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_10_ascending ==
files: 1
 - 8dd2f31f-438b-4851-b506-f996088320bb.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_11_ascending ==
files: 1
 - aa72da1f-e666-4dfe-8d8e-769483bc0ca4.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_12_ascending ==
files: 1
 - fcd1d016-93d2-45b9-adcd-69ab6c66a545.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_1_ascending ==
files: 1
 - 03078010-49fd-42e3-b46c-abdbed1c4f9f.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_2_ascending ==
files: 1
 - 2cf16d92-8ac3-4f31-8e0f-12d7aefda95e.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_3_ascending ==
files: 1
 - 6cf7ae9a-6f16-4996-8704-ebf03e82949c.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_4_ascending ==
files: 1
 - 9a0f4273-57e8-4759-bef4-caf17b27cf1e.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_5_ascending ==
files: 1
 - 04808153-c168-40c3-a690-2a4a72f0e0e2.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_6_ascending ==
files: 1
 - a9bb01c3-4e76-4ec3-9028-f693ff566650.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_7_ascending ==
files: 1
 - d892adec-54d6-4082-ba04-ee67e9fe00e1.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_8_ascending ==
files: 1
 - 51e8eec2-a965-4a6b-94c9-d7371fcec190.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2023_9_ascending ==
files: 1
 - 87bb0674-a6ed-4fdb-aaa5-d216c43aef58.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_10_ascending ==
files: 1
 - ec90df1a-4975-44b5-a79a-d8f2234f823d.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_11_ascending ==
files: 1
 - ba22acff-03e8-41e6-b092-b63c9e67eec7.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_12_ascending ==
files: 1
 - 0862c06c-3f14-4168-897b-d79422adae59.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_1_ascending ==
files: 1
 - e659d016-762c-4288-bd09-cd9ac3394d03.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_2_ascending ==
files: 1
 - 123750dd-09f0-4cab-85e4-5bf0caa873cd.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_3_ascending ==
files: 1
 - d19812ec-51ec-41dd-b211-ed5862c1bd79.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_4_ascending ==
files: 1
 - 5987d195-dc1e-4f1c-a025-731e7aca8dde.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_5_ascending ==
files: 1
 - 92a0a9c4-5cda-4c09-8f2b-27b339ff9c13.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_6_ascending ==
files: 1
 - 1a74776d-c6a9-45e9-8e9f-22042bf9220a.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_7_ascending ==
files: 1
 - cb7d82ca-3b64-4dfc-af75-8b3bee2d98b1.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_8_ascending ==
files: 1
 - ec8ad9bd-7b79-4301-b9ee-01b36e9241da.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2024_9_ascending ==
files: 1
 - cc9a254e-19a9-4fcf-94d7-a6848fb4f545.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_10_ascending ==
files: 1
 - 1b3cae7c-c9b4-447e-99c1-3ff446f547ce.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_11_ascending ==
files: 1
 - 8ae1207e-e6ce-421d-b3a2-7b1b7742c3ab.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_12_ascending ==
files: 1
 - 5df2bed1-cedf-4f6e-9a01-39de51c6cce4.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_1_ascending ==
files: 1
 - b12db5cd-34ea-4a81-a58a-53fcbb0c3ceb.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_2_ascending ==
files: 1
 - fbda1f2a-51f1-42e2-b6cd-ec04e4c5da0f.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_3_ascending ==
files: 1
 - bcfde5e5-3a4f-44de-b1cb-07b1b66f0c79.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_4_ascending ==
files: 1
 - fa08855d-467e-4624-9504-7d5ee7f7a46a.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_5_ascending ==
files: 1
 - 80867c00-ae93-436d-9b29-ba8d34d5a55e.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_7_ascending ==
files: 1
 - f09d6ff3-edd8-4e11-844f-d8e51fe8d4c9.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_8_ascending ==
files: 1
 - da5e1c79-ebba-45ae-b7b1-ed7f9b45b087.tif

== sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2025_9_ascending ==
files: 1
 - 8d8764a5-7fc5-4dce-81f1-6bd41c9e61c8.tif

== sentinel-1/train/48PYB_3_6__s1_rtc ==
files: 145
 - 48PYB_3_6__s1_rtc_2021_12_descending.tif
 - 48PYB_3_6__s1_rtc_2025_11_descending.tif
 - 48PYB_3_6__s1_rtc_2024_4_descending.tif
 - 48PYB_3_6__s1_rtc_2021_3_descending.tif
 - 48PYB_3_6__s1_rtc_2021_11_descending.tif
 - 48PYB_3_6__s1_rtc_2025_10_ascending.tif
 - 48PYB_3_6__s1_rtc_2020_8_descending.tif
 - 48PYB_3_6__s1_rtc_2021_1_ascending.tif
 - 48PYB_3_6__s1_rtc_2025_4_ascending.tif
 - 48PYB_3_6__s1_rtc_2024_10_descending.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_10_ascending ==
files: 1
 - 4853b6f6-0c05-4724-9eed-bb74fe475d7c.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_11_ascending ==
files: 1
 - dffc0383-b193-4add-805e-3b2b498c7df2.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_12_ascending ==
files: 1
 - c1e4da93-6fad-4389-a1bb-6e26065506f6.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_1_ascending ==
files: 1
 - e5d107bd-3b2a-4143-83de-336790b2510d.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_2_ascending ==
files: 1
 - d4a399be-a18e-4d31-9bf9-16f6e0ccbd78.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_3_ascending ==
files: 1
 - 85f66932-1e05-4111-b52c-58a39e87dee9.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_4_ascending ==
files: 1
 - c99a7d76-7947-4533-95d8-7343f72cdeb3.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_5_ascending ==
files: 1
 - cc8d5597-da4c-47ec-a081-1fbebafa0ff2.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_6_ascending ==
files: 1
 - 3baacf53-d1ca-48d2-9cfe-c28516574862.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_7_ascending ==
files: 1
 - 8822d425-0b13-49b9-a0ff-222e079b6d8d.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_8_ascending ==
files: 1
 - 6d462098-501c-423d-872f-81f13d9094a5.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2020_9_ascending ==
files: 1
 - ed1fbf8c-9395-49ba-b25a-ff16aef0abcb.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_10_ascending ==
files: 1
 - 1fe6116b-4675-4b84-bb90-642149290bdb.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_11_ascending ==
files: 1
 - 087cb842-faa8-4ed8-bf86-69e7459f4d80.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_12_ascending ==
files: 1
 - f2f12551-cf12-4cd1-b588-e821099b4641.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_1_ascending ==
files: 1
 - fa07fb59-daa9-41e0-9137-3375fb76fa26.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_2_ascending ==
files: 1
 - d09da991-48a7-4e64-b5f1-37f10c4e00f1.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_3_ascending ==
files: 1
 - f73a48c3-b36e-4cae-9912-b34876fd7932.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_4_ascending ==
files: 1
 - e650390f-be46-4ec7-824c-62acc6107ef4.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_5_ascending ==
files: 1
 - e2cf55f6-46f9-44f5-b6f7-268b61f39641.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_6_ascending ==
files: 1
 - 13be2077-ff7c-403e-8f4c-20e2ca612f94.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_7_ascending ==
files: 1
 - dd9aa8a8-43b1-4b38-8c9a-c6bab8d76ebf.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_8_ascending ==
files: 1
 - f2e0155b-34a7-4b7c-a5ab-8dadf5829552.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2021_9_ascending ==
files: 1
 - a35d53a5-1aec-4a50-b477-ffa1db06ecee.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_10_ascending ==
files: 1
 - a32ec72d-1322-4e1a-a747-5d19c0516980.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_11_ascending ==
files: 1
 - 192bb9a2-2609-4ecf-88e3-0fe0154c1d7f.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_12_ascending ==
files: 1
 - 5866841a-8a14-4917-b16b-1f5dab7a35c5.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_1_ascending ==
files: 1
 - ac7d753a-8979-4e91-85de-2c24869c7e84.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_2_ascending ==
files: 1
 - 8deb229c-21b6-4732-9787-b13e54ea7faa.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_3_ascending ==
files: 1
 - b7031a85-49fa-4792-b991-faa716c7b96f.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_4_ascending ==
files: 1
 - 675747b6-da9e-4f7e-afc9-598d227b4e9b.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_5_ascending ==
files: 1
 - 021aacbc-4005-445a-b5e4-81ae959da423.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_6_ascending ==
files: 1
 - 5758f431-4763-4285-bb86-61950fab5c8b.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_7_ascending ==
files: 1
 - d583f268-1ba0-4f75-ba8c-26f9a8e0efca.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_8_ascending ==
files: 1
 - 979bc471-5e85-450d-9791-69b9d3375211.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2022_9_ascending ==
files: 1
 - 76036c5a-44db-4756-9f98-d6df481979d8.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_10_ascending ==
files: 1
 - 959f7ff3-95bd-4abb-929d-1308a2cfec66.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_11_ascending ==
files: 1
 - 4f9d9229-e041-4eb1-bd92-b48507a27461.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_12_ascending ==
files: 1
 - 34863e10-bfbe-41d8-aabe-24c4193b77a2.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_1_ascending ==
files: 1
 - 21b54f4b-77bc-489f-9d0f-e9b955f31ecf.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_2_ascending ==
files: 1
 - 86bd1556-ea62-4a09-b8e5-c5e986eb7eff.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_3_ascending ==
files: 1
 - fb071c37-0aa3-41d0-bf91-6f5c4e5795d0.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_4_ascending ==
files: 1
 - 06f1a03a-5d2c-4964-8b5f-e5106605061a.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_5_ascending ==
files: 1
 - a43bc0b3-5887-4305-bcee-81e9ee6945a1.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_6_ascending ==
files: 1
 - d283fc45-7d6a-480d-8cd8-459a4108a3d6.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_7_ascending ==
files: 1
 - 4df58860-eabd-4ee5-ac92-6fa753572867.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_8_ascending ==
files: 1
 - 45f3328f-a4e0-4581-aa0a-45a336dfeeed.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2023_9_ascending ==
files: 1
 - 0db5a8cd-b5b1-425c-bd8e-e074deabbc4f.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_10_ascending ==
files: 1
 - 588d5505-3249-47ef-8d61-d18c3f7813e3.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_11_ascending ==
files: 1
 - d1c90309-006d-4b3d-932a-4445e371b004.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_1_ascending ==
files: 1
 - 72c5a38f-f5d0-4b1b-af9c-3f5a651c7711.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_2_ascending ==
files: 1
 - ac83a6fa-6dbf-4141-8dc8-0851ca8d5803.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_3_ascending ==
files: 1
 - bb964b4c-a4a4-4ed5-93c4-9caad06f43f5.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_4_ascending ==
files: 1
 - 91c3c7c2-9c93-45c3-9553-d66a587bde5c.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_5_ascending ==
files: 1
 - 41c3b957-78e5-40e0-8f8e-05f46ad7a3a3.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_6_ascending ==
files: 1
 - 531a8ee0-1718-446b-9464-347dfefd7939.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_7_ascending ==
files: 1
 - 6a7156ec-de4d-4d5f-8879-bd65393150c8.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_8_ascending ==
files: 1
 - 74dc601a-f9b4-4d0f-a391-b1adc2fa3e1c.tif

== sentinel-1/train/48PYB_3_6__s1_rtc/48PYB_3_6__s1_rtc_2024_9_ascending ==
files: 1
 - 49a69e13-2e1a-4f0c-b683-5f420a82947a.tif

== sentinel-1/train/48QVE_3_0__s1_rtc ==
files: 141
 - 48QVE_3_0__s1_rtc_2021_12_ascending.tif
 - 48QVE_3_0__s1_rtc_2021_5_descending.tif
 - 48QVE_3_0__s1_rtc_2022_6_descending.tif
 - 48QVE_3_0__s1_rtc_2020_10_ascending.tif
 - 48QVE_3_0__s1_rtc_2020_2_ascending.tif
 - 48QVE_3_0__s1_rtc_2024_6_ascending.tif
 - 48QVE_3_0__s1_rtc_2024_8_descending.tif
 - 48QVE_3_0__s1_rtc_2020_3_ascending.tif
 - 48QVE_3_0__s1_rtc_2020_12_ascending.tif
 - 48QVE_3_0__s1_rtc_2023_7_descending.tif

== sentinel-1/train/48QWD_2_2__s1_rtc ==
files: 141
 - 48QWD_2_2__s1_rtc_2025_3_ascending.tif
 - 48QWD_2_2__s1_rtc_2025_4_descending.tif
 - 48QWD_2_2__s1_rtc_2024_3_descending.tif
 - 48QWD_2_2__s1_rtc_2022_6_ascending.tif
 - 48QWD_2_2__s1_rtc_2023_11_ascending.tif
 - 48QWD_2_2__s1_rtc_2022_8_ascending.tif
 - 48QWD_2_2__s1_rtc_2020_4_ascending.tif
 - 48QWD_2_2__s1_rtc_2024_6_descending.tif
 - 48QWD_2_2__s1_rtc_2022_7_ascending.tif
 - 48QWD_2_2__s1_rtc_2023_10_descending.tif

== sentinel-2/test/18NVJ_1_6__s2_l2a ==
files: 72
 - 18NVJ_1_6__s2_l2a_2024_11.tif
 - 18NVJ_1_6__s2_l2a_2022_9.tif
 - 18NVJ_1_6__s2_l2a_2025_12.tif
 - 18NVJ_1_6__s2_l2a_2022_11.tif
 - 18NVJ_1_6__s2_l2a_2023_6.tif
 - 18NVJ_1_6__s2_l2a_2023_9.tif
 - 18NVJ_1_6__s2_l2a_2022_10.tif
 - 18NVJ_1_6__s2_l2a_2020_1.tif
 - 18NVJ_1_6__s2_l2a_2025_7.tif
 - 18NVJ_1_6__s2_l2a_2025_9.tif

== sentinel-2/test/18NYH_2_1__s2_l2a ==
files: 71
 - 18NYH_2_1__s2_l2a_2023_8.tif
 - 18NYH_2_1__s2_l2a_2020_7.tif
 - 18NYH_2_1__s2_l2a_2022_3.tif
 - 18NYH_2_1__s2_l2a_2024_1.tif
 - 18NYH_2_1__s2_l2a_2025_11.tif
 - 18NYH_2_1__s2_l2a_2025_5.tif
 - 18NYH_2_1__s2_l2a_2020_4.tif
 - 18NYH_2_1__s2_l2a_2021_4.tif
 - 18NYH_2_1__s2_l2a_2020_11.tif
 - .DS_Store

== sentinel-2/test/33NTE_5_1__s2_l2a ==
files: 73
 - 33NTE_5_1__s2_l2a_2021_10.tif
 - 33NTE_5_1__s2_l2a_2025_3.tif
 - 33NTE_5_1__s2_l2a_2024_6.tif
 - 33NTE_5_1__s2_l2a_2022_12.tif
 - 33NTE_5_1__s2_l2a_2021_8.tif
 - 33NTE_5_1__s2_l2a_2025_6.tif
 - 33NTE_5_1__s2_l2a_2020_10.tif
 - 33NTE_5_1__s2_l2a_2022_11.tif
 - 33NTE_5_1__s2_l2a_2020_6.tif
 - 33NTE_5_1__s2_l2a_2021_3.tif

== sentinel-2/test/47QMA_6_2__s2_l2a ==
files: 61
 - 47QMA_6_2__s2_l2a_2023_11.tif
 - 47QMA_6_2__s2_l2a_2025_5.tif
 - 47QMA_6_2__s2_l2a_2023_6.tif
 - 47QMA_6_2__s2_l2a_2023_3.tif
 - 47QMA_6_2__s2_l2a_2023_5.tif
 - 47QMA_6_2__s2_l2a_2024_5.tif
 - .DS_Store
 - 47QMA_6_2__s2_l2a_2025_11.tif
 - 47QMA_6_2__s2_l2a_2024_8.tif
 - 47QMA_6_2__s2_l2a_2023_9.tif

== sentinel-2/test/48PWA_0_6__s2_l2a ==
files: 71
 - 48PWA_0_6__s2_l2a_2020_8.tif
 - 48PWA_0_6__s2_l2a_2022_9.tif
 - 48PWA_0_6__s2_l2a_2021_3.tif
 - 48PWA_0_6__s2_l2a_2023_3.tif
 - 48PWA_0_6__s2_l2a_2025_1.tif
 - 48PWA_0_6__s2_l2a_2025_7.tif
 - 48PWA_0_6__s2_l2a_2024_9.tif
 - 48PWA_0_6__s2_l2a_2022_1.tif
 - 48PWA_0_6__s2_l2a_2022_6.tif
 - 48PWA_0_6__s2_l2a_2020_4.tif

== sentinel-2/train/18NWG_6_6__s2_l2a ==
files: 73
 - 18NWG_6_6__s2_l2a_2021_3.tif
 - 18NWG_6_6__s2_l2a_2025_1.tif
 - 18NWG_6_6__s2_l2a_2025_8.tif
 - 18NWG_6_6__s2_l2a_2024_7.tif
 - 18NWG_6_6__s2_l2a_2022_8.tif
 - 18NWG_6_6__s2_l2a_2020_2.tif
 - 18NWG_6_6__s2_l2a_2021_7.tif
 - 18NWG_6_6__s2_l2a_2021_11.tif
 - 18NWG_6_6__s2_l2a_2024_3.tif
 - .DS_Store

== sentinel-2/train/18NWH_1_4__s2_l2a ==
files: 73
 - 18NWH_1_4__s2_l2a_2020_4.tif
 - 18NWH_1_4__s2_l2a_2025_12.tif
 - 18NWH_1_4__s2_l2a_2022_1.tif
 - 18NWH_1_4__s2_l2a_2023_9.tif
 - 18NWH_1_4__s2_l2a_2025_7.tif
 - 18NWH_1_4__s2_l2a_2023_3.tif
 - 18NWH_1_4__s2_l2a_2024_6.tif
 - 18NWH_1_4__s2_l2a_2025_2.tif
 - 18NWH_1_4__s2_l2a_2022_12.tif
 - 18NWH_1_4__s2_l2a_2024_10.tif

== sentinel-2/train/18NWJ_8_9__s2_l2a ==
files: 72
 - 18NWJ_8_9__s2_l2a_2022_5.tif
 - 18NWJ_8_9__s2_l2a_2024_9.tif
 - 18NWJ_8_9__s2_l2a_2020_3.tif
 - 18NWJ_8_9__s2_l2a_2024_10.tif
 - 18NWJ_8_9__s2_l2a_2021_10.tif
 - 18NWJ_8_9__s2_l2a_2020_2.tif
 - 18NWJ_8_9__s2_l2a_2023_10.tif
 - 18NWJ_8_9__s2_l2a_2020_6.tif
 - 18NWJ_8_9__s2_l2a_2025_8.tif
 - 18NWJ_8_9__s2_l2a_2020_9.tif

== sentinel-2/train/18NWM_9_4__s2_l2a ==
files: 73
 - 18NWM_9_4__s2_l2a_2021_10.tif
 - 18NWM_9_4__s2_l2a_2021_4.tif
 - 18NWM_9_4__s2_l2a_2024_2.tif
 - 18NWM_9_4__s2_l2a_2025_9.tif
 - 18NWM_9_4__s2_l2a_2025_8.tif
 - .DS_Store
 - 18NWM_9_4__s2_l2a_2021_2.tif
 - 18NWM_9_4__s2_l2a_2024_11.tif
 - 18NWM_9_4__s2_l2a_2023_8.tif
 - 18NWM_9_4__s2_l2a_2020_4.tif

== sentinel-2/train/18NXH_6_8__s2_l2a ==
files: 72
 - 18NXH_6_8__s2_l2a_2022_1.tif
 - 18NXH_6_8__s2_l2a_2021_6.tif
 - 18NXH_6_8__s2_l2a_2025_9.tif
 - 18NXH_6_8__s2_l2a_2021_4.tif
 - 18NXH_6_8__s2_l2a_2024_11.tif
 - 18NXH_6_8__s2_l2a_2020_5.tif
 - 18NXH_6_8__s2_l2a_2025_4.tif
 - .DS_Store
 - 18NXH_6_8__s2_l2a_2020_1.tif
 - 18NXH_6_8__s2_l2a_2025_5.tif

== sentinel-2/train/18NXJ_7_6__s2_l2a ==
files: 73
 - 18NXJ_7_6__s2_l2a_2022_3.tif
 - 18NXJ_7_6__s2_l2a_2023_5.tif
 - 18NXJ_7_6__s2_l2a_2020_1.tif
 - 18NXJ_7_6__s2_l2a_2023_3.tif
 - 18NXJ_7_6__s2_l2a_2021_9.tif
 - 18NXJ_7_6__s2_l2a_2025_9.tif
 - 18NXJ_7_6__s2_l2a_2025_5.tif
 - 18NXJ_7_6__s2_l2a_2025_2.tif
 - 18NXJ_7_6__s2_l2a_2021_12.tif
 - 18NXJ_7_6__s2_l2a_2025_8.tif

== sentinel-2/train/18NYH_9_9__s2_l2a ==
files: 73
 - 18NYH_9_9__s2_l2a_2024_9.tif
 - 18NYH_9_9__s2_l2a_2023_1.tif
 - 18NYH_9_9__s2_l2a_2020_10.tif
 - 18NYH_9_9__s2_l2a_2025_10.tif
 - 18NYH_9_9__s2_l2a_2023_12.tif
 - 18NYH_9_9__s2_l2a_2020_3.tif
 - 18NYH_9_9__s2_l2a_2020_12.tif
 - 18NYH_9_9__s2_l2a_2021_10.tif
 - 18NYH_9_9__s2_l2a_2022_11.tif
 - 18NYH_9_9__s2_l2a_2020_6.tif

== sentinel-2/train/19NBD_4_4__s2_l2a ==
files: 73
 - 19NBD_4_4__s2_l2a_2023_10.tif
 - 19NBD_4_4__s2_l2a_2025_1.tif
 - 19NBD_4_4__s2_l2a_2024_2.tif
 - 19NBD_4_4__s2_l2a_2025_10.tif
 - 19NBD_4_4__s2_l2a_2024_8.tif
 - 19NBD_4_4__s2_l2a_2021_6.tif
 - 19NBD_4_4__s2_l2a_2023_5.tif
 - 19NBD_4_4__s2_l2a_2025_11.tif
 - 19NBD_4_4__s2_l2a_2025_2.tif
 - .DS_Store

== sentinel-2/train/47QMB_0_8__s2_l2a ==
files: 73
 - 47QMB_0_8__s2_l2a_2022_2.tif
 - 47QMB_0_8__s2_l2a_2020_3.tif
 - 47QMB_0_8__s2_l2a_2021_6.tif
 - 47QMB_0_8__s2_l2a_2025_2.tif
 - 47QMB_0_8__s2_l2a_2023_3.tif
 - 47QMB_0_8__s2_l2a_2022_7.tif
 - 47QMB_0_8__s2_l2a_2022_5.tif
 - 47QMB_0_8__s2_l2a_2020_10.tif
 - 47QMB_0_8__s2_l2a_2025_3.tif
 - 47QMB_0_8__s2_l2a_2023_12.tif

== sentinel-2/train/47QQV_2_4__s2_l2a ==
files: 73
 - 47QQV_2_4__s2_l2a_2020_5.tif
 - 47QQV_2_4__s2_l2a_2022_7.tif
 - 47QQV_2_4__s2_l2a_2025_4.tif
 - 47QQV_2_4__s2_l2a_2025_3.tif
 - 47QQV_2_4__s2_l2a_2022_1.tif
 - 47QQV_2_4__s2_l2a_2021_11.tif
 - 47QQV_2_4__s2_l2a_2020_8.tif
 - 47QQV_2_4__s2_l2a_2024_2.tif
 - 47QQV_2_4__s2_l2a_2024_1.tif
 - 47QQV_2_4__s2_l2a_2022_4.tif

== sentinel-2/train/48PUT_0_8__s2_l2a ==
files: 73
 - 48PUT_0_8__s2_l2a_2022_9.tif
 - 48PUT_0_8__s2_l2a_2020_11.tif
 - 48PUT_0_8__s2_l2a_2020_4.tif
 - 48PUT_0_8__s2_l2a_2020_7.tif
 - 48PUT_0_8__s2_l2a_2024_8.tif
 - .DS_Store
 - 48PUT_0_8__s2_l2a_2023_11.tif
 - 48PUT_0_8__s2_l2a_2024_10.tif
 - 48PUT_0_8__s2_l2a_2025_1.tif
 - 48PUT_0_8__s2_l2a_2021_10.tif

== sentinel-2/train/48PWV_7_8__s2_l2a ==
files: 73
 - 48PWV_7_8__s2_l2a_2023_12.tif
 - 48PWV_7_8__s2_l2a_2020_3.tif
 - 48PWV_7_8__s2_l2a_2021_9.tif
 - 48PWV_7_8__s2_l2a_2023_6.tif
 - 48PWV_7_8__s2_l2a_2021_2.tif
 - 48PWV_7_8__s2_l2a_2024_4.tif
 - 48PWV_7_8__s2_l2a_2022_2.tif
 - 48PWV_7_8__s2_l2a_2020_2.tif
 - 48PWV_7_8__s2_l2a_2023_5.tif
 - 48PWV_7_8__s2_l2a_2025_10.tif

== sentinel-2/train/48PXC_7_7__s2_l2a ==
files: 73
 - 48PXC_7_7__s2_l2a_2021_1.tif
 - 48PXC_7_7__s2_l2a_2024_2.tif
 - 48PXC_7_7__s2_l2a_2024_11.tif
 - 48PXC_7_7__s2_l2a_2022_7.tif
 - 48PXC_7_7__s2_l2a_2024_1.tif
 - 48PXC_7_7__s2_l2a_2023_5.tif
 - 48PXC_7_7__s2_l2a_2023_8.tif
 - 48PXC_7_7__s2_l2a_2020_12.tif
 - 48PXC_7_7__s2_l2a_2022_8.tif
 - 48PXC_7_7__s2_l2a_2024_8.tif

== sentinel-2/train/48PYB_3_6__s2_l2a ==
files: 73
 - 48PYB_3_6__s2_l2a_2021_6.tif
 - 48PYB_3_6__s2_l2a_2023_9.tif
 - 48PYB_3_6__s2_l2a_2021_11.tif
 - 48PYB_3_6__s2_l2a_2024_12.tif
 - 48PYB_3_6__s2_l2a_2020_11.tif
 - 48PYB_3_6__s2_l2a_2024_10.tif
 - 48PYB_3_6__s2_l2a_2025_8.tif
 - 48PYB_3_6__s2_l2a_2022_5.tif
 - 48PYB_3_6__s2_l2a_2023_6.tif
 - .DS_Store

== sentinel-2/train/48QVE_3_0__s2_l2a ==
files: 73
 - 48QVE_3_0__s2_l2a_2025_9.tif
 - 48QVE_3_0__s2_l2a_2022_5.tif
 - 48QVE_3_0__s2_l2a_2021_4.tif
 - 48QVE_3_0__s2_l2a_2024_1.tif
 - 48QVE_3_0__s2_l2a_2022_2.tif
 - 48QVE_3_0__s2_l2a_2020_5.tif
 - 48QVE_3_0__s2_l2a_2021_10.tif
 - 48QVE_3_0__s2_l2a_2020_10.tif
 - 48QVE_3_0__s2_l2a_2025_4.tif
 - .DS_Store

== sentinel-2/train/48QWD_2_2__s2_l2a ==
files: 73
 - 48QWD_2_2__s2_l2a_2021_11.tif
 - 48QWD_2_2__s2_l2a_2023_7.tif
 - 48QWD_2_2__s2_l2a_2020_4.tif
 - 48QWD_2_2__s2_l2a_2023_10.tif
 - 48QWD_2_2__s2_l2a_2024_10.tif
 - 48QWD_2_2__s2_l2a_2023_8.tif
 - 48QWD_2_2__s2_l2a_2025_1.tif
 - 48QWD_2_2__s2_l2a_2023_12.tif
 - 48QWD_2_2__s2_l2a_2023_1.tif
 - 48QWD_2_2__s2_l2a_2024_5.tif
```