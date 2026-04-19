Here's a cleaned-up, slide-ready version. I'll fix a few inaccuracies, clarify what each baseline actually did, and structure it as you'd want for a presentation.

---

# Slide 1: Baseline Models Overview

## Baseline 1 — AEF-only, single year

**Input:** AlphaEarth Foundations (AEF) embeddings — 64-dimensional foundation-model features per pixel, from a single target year.

**Model:** Pixel-wise XGBoost classifier.

**Validation F1 (Class 1):** ~0.79 (same-region held-out pixels)

**Notes / speaker:** AEF is a pretrained global foundation model that encodes multimodal satellite information (optical + radar) into 64 dims per pixel per year. Using just one year's embedding as input is the simplest possible starting point.

---

## Baseline 2 — AEF + temporal differences

**Input:** 192-dim feature vector per pixel:
- AEF[target year] — current state (64 dims)
- AEF[target year] − AEF[2020] — cumulative change from baseline (64 dims)
- AEF[target year] − AEF[target year − 1] — year-over-year change (64 dims)

**Model:** Pixel-wise XGBoost classifier.

**Validation F1 (Class 1):** ~0.87 (same-region held-out pixels)

**Notes / speaker:** The jump from 0.79 → 0.87 F1 confirms that *change* between years is the real deforestation signal, not the raw land-cover state. The anchor to 2020 is chosen because the challenge defines deforestation as "forest in 2020, non-forest later."

---

## Baseline 3 — Regional ensemble

**Input:** Same 192-dim feature vector as Baseline 2.

**Model:** Ensemble of 3 XGBoost models bundled together:
1. **Thailand-holdout model** — trained on Colombia only, validated on Thailand
2. **Colombia-holdout model** — trained on Thailand only, validated on Colombia
3. **All-data model** — trained on both regions combined

At inference: probabilities from all three models are averaged.

**Cross-region F1 (Class 1):**
- Thailand holdout: ~0.65
- Colombia holdout: ~0.59

**Notes / speaker:** Unlike Baseline 1/2 which were evaluated on same-region held-out pixels (optimistic), Baseline 3 was evaluated across regions (trained on one, tested on the other) — a realistic estimate for the Africa test set. The lower F1 is not a worse model; it is an honest measurement under harder generalization conditions. Ensembling across regionally-diverse models reduces overfitting to any single region.

---


All three submissions below use the **Baseline 3 ensemble** (the 3-model bundle described above). Labels during training were built via **majority vote across three weak-label sources**: RADD, GLAD-L, and GLAD-S2. A pixel is labeled positive if at least 2 of the 3 sources flag deforestation post-2020.

---

## Submission 1 — Baseline inference (sanity check)

**Settings:**
- Model: Baseline 3 ensemble
- Probability threshold: 0.45
- Minimum polygon area: 0.5 ha
- No spatial postprocessing

**Results:**
- Union IoU: **32.79%**
- Poly Recall: 79.06%
- Poly FPR: 64.09%

**Takeaway:** High recall, very high false-positive rate. The model finds most deforestation but also flags a lot of noise.

---

## Submission 2 — Threshold tuning

**Change:** Probability threshold raised from 0.45 to 0.55. Everything else identical to Submission 1.

**Results:**
- Union IoU: **38.09%** (+5.3)
- Poly Recall: 76.11%
- Poly FPR: 56.74% (−7.4)

**Takeaway:** Raising the threshold removes low-confidence predictions (disproportionately FPs) while barely touching recall. A simple no-cost change worth over 5 IoU points.

---

## Submission 3 — UTM-space postprocessing

**Changes:**
- Probability threshold: 0.50 (slightly relaxed)
- Ensemble combination: **geometric mean** of three model probabilities (stricter than arithmetic mean — penalizes disagreement between models)
- **UTM reprojection for postprocessing:** the binary prediction raster is reprojected from EPSG:4326 to the tile's local UTM CRS (10m resolution, matching Sentinel-2) so that spatial operations use meaningful distance and area units
- **Morphological closing (5×5)** followed by **opening (3×3)** to fill gaps and remove salt-and-pepper noise, applied in UTM space
- **Area filter** at 1.0 ha minimum, computed correctly as `abs(transform.a × transform.e) / 10000` hectares per pixel in UTM
- Polygons reprojected back to EPSG:4326 for the GeoJSON submission

**Results:**
- Union IoU: **44.98%** (+6.9)
- Poly Recall: 70.52%
- Poly FPR: 44.61% (−12.1)

**Takeaway:** Proper spatial postprocessing — in the correct coordinate system, with morphology and area filtering — gave the largest single improvement of any change so far. This submission is the current best.

---

# What's Next

## Submission 4 (planned) — Label confidence weighting

**Motivation:** The three weak-label sources (RADD, GLAD-L, GLAD-S2) each provide *graded confidence values*, not just binary flags. Our current training uses only a binary consensus (majority vote) and discards this information.

**Planned change:**
- For each pixel, compute a continuous consensus confidence by weighting each source's native confidence score (e.g., GLAD-L: probable=0.5, confirmed=1.0; RADD: low-conf=0.5, high-conf=1.0; GLAD-S2: 5-level scale)
- Pass these as `sample_weight` to XGBoost training so the model focuses on pixels where the label is trustworthy and under-weights the ambiguous middle ground
- Retrain the Baseline 3 ensemble with weighted training, save as a new bundle
- Inference pipeline unchanged

**Expected effect:** Cleaner per-pixel probabilities from training on higher-quality labels. Primarily an FPR reduction (fewer "confidently wrong" predictions driven by noisy labels). Expected IoU gain: +2-4 points.

---

# Corrections and clarifications I made to your notes

A few things in your original notes that I fixed or clarified in the above:

1. **"first and second: RADD, GLAD-L, GLAD majority vote"** — this applies to all three submissions, not just 1 and 2. The labels used for *training* are the same across submissions; what changed was the *inference/postprocessing* pipeline.

2. **"EPSC 43%"** → EPSG:4326 (the geographic lat/lon coordinate system). This was just a typo worth fixing for slides.

3. **"ensembling of baseline what models"** — clarified that all three submissions use the Baseline 3 ensemble, which itself is 3 XGBoost models (TH-holdout, CO-holdout, all-data) averaged at inference. Submissions 1 and 2 use arithmetic mean averaging; submission 3 switches to geometric mean.

4. **F1 score context for Baseline 3** — called out explicitly that the F1 numbers are not comparable to Baseline 1/2 because they were evaluated on a harder split (cross-region vs same-region). This is important because otherwise the audience may think Baseline 3 is a worse model.

5. **"normalfeature embeddings subtracted from 2020 the last year"** — rephrased as clearer temporal difference descriptions in the Baseline 2 slide.

Let me know if you want the slides tightened further (e.g., shorter bullets for slide display, or longer prose for speaker notes), or if you want me to restructure into a specific slide template format.
