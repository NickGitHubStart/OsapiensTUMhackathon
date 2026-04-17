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

