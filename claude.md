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