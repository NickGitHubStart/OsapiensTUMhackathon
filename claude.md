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


Per-modality model recommendations
AEF (64-dim embeddings, annual, EPSG:4326)
Best bet: lightweight temporal classifier on per-pixel embedding sequences.
Concretely, for each pixel you have a sequence of (n_years, 64) embeddings. Deforestation shows up as an abrupt change in the embedding trajectory. Models that work well:

Difference features + LightGBM — compute AEF[year] - AEF[2020] (64 dims) and AEF[year] - AEF[year-1] (64 dims) per pixel per year, concat with raw embedding, feed to LightGBM. Stupidly effective baseline. Trains in minutes. No overfitting risk.
Small 1D CNN or MLP over years — input (n_years, 64), output per-year deforestation probability. Residual connection on the year-to-year diff. ~100k params, trains fast.
Tiny transformer (4 layers, 128 dim) over the year sequence — only if you have time. Marginal gains.

Avoid: big U-Nets on AEF. The spatial resolution is lower (embeddings are typically coarser than 10m), and spatial context in AEF is already baked in by the foundation model. You gain little from 2D convs over them.
Why this is your main model: AEF is global-pretrained, so a simple classifier on top is closer to linear-probing a foundation model — which is exactly the setup that generalizes best to new regions.
Sentinel-1 (VV backscatter, monthly, 2 orbits)
Best bet: temporal U-Net on annual statistics.
Don't feed 60 monthly tiles to a model. Compute per-year features per pixel:

Mean VV in dB
Std VV in dB (volatility — deforestation spikes this)
Min, max, 10th/90th percentiles
Mean difference vs. 2020 baseline
Separate features for ascending vs descending orbits (they see different geometries)

Result: ~12 features × n_years per pixel. Feed to:

2D U-Net (ResNet-18 or EfficientNet-B0 encoder, small) treating the features as channels. ~5M params. Trains fast, uses spatial context (which matters for radar speckle).
Or concat these as extra channels to your S2/AEF model rather than a separate model.

Why radar annual stats, not raw monthly: monthly VV has heavy speckle noise, variable orbit coverage, and seasonal moisture effects. Annual aggregates kill most of the noise while preserving the clearing signal. Clearing causes a sustained drop in mean VV + temporary spike in std — that's what you want the model to see.
Sentinel-2 (12 bands, monthly)
Best bet: temporal U-Net on cloud-free annual composites + spectral indices.
S2 has the richest signal but cloud gaps are brutal in tropics. Do this:

Mask clouds per month using SCL if provided, or band thresholds (B09 water vapor > threshold, or (B2+B3+B4)/3 > 0.3 rough proxy).
Compute annual median composite per band → (n_years, 12, H, W).
Derive indices per composite:

NDVI = (B8 - B4) / (B8 + B4) — the deforestation signal
NBR = (B8 - B12) / (B8 + B12) — burn-sensitive, catches slash-and-burn
NDMI = (B8 - B11) / (B8 + B11) — moisture, drops when canopy removed
Tasseled cap brightness/greenness/wetness if you want to go fancy


Temporal features: ΔNDVI(year vs 2020), ΔNBR, max NDVI, min NDVI.

Feed to:

U-Net with ResNet-18/34 encoder on the per-year feature stack. Standard, well-understood, not too overfitty.
Temporal U-Net (3D conv on the year dimension, or ConvLSTM bottleneck) if you want to model the trajectory explicitly. Higher ceiling, higher overfitting risk, more engineering.

Be careful: S2 is where your Thailand → Africa gap will bite hardest. Augment aggressively: spectral jitter (random gain per band), random channel dropout, cross-region CutMix.
The winning strategy (IMO)
Don't train three separate models. Train one multimodal model, but tiered by trust:
Main model: per-pixel head on concat(AEF_features, S1_annual_stats, S2_annual_indices)
            → simple MLP or LightGBM per pixel, then spatial smoothing
            OR
            → U-Net with all features as channels

AEF should dominate the feature importance.
S1 should add ~3-5 points via cloud-immunity in missing S2 months.
S2 should add ~2-3 points via spectral precision, but weighted lower.
Three reasons:

AEF alone is probably 80% of your score on Africa. Everything else is refinement.
Late fusion beats early fusion for generalization. Each modality's annual features are already summarized — you're not forcing the model to re-learn noise filtering per region.
One model, three feature banks lets you ablate: turn off S2 features at inference if your local CV on held-out Colombia says S2 hurts generalization. You can't do that with three separately-trained models.

If I had to pick one architecture to submit first
LightGBM on per-pixel feature bank (AEF + S1 annual + S2 annual indices + forest-2020 mask), followed by morphological smoothing → polygons.
This will:

Train in under an hour on Colab
Not overfit regionally (tree models on summarized features generalize shockingly well)
Give you a baseline to beat with a U-Net
Tell you which features actually matter via feature importance

Then Sub 2 is a U-Net with the same features as channels, trained with leave-one-region-out. If the U-Net doesn't beat LightGBM by ≥3 points, keep LightGBM — it will generalize better to Africa.