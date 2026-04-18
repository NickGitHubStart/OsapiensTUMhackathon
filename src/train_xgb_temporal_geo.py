"""Temporal XGBoost (AEF + NDVI/S1) with geographic validation and enhanced labels.

Use ``src.train_xgb_temporal`` for the original random pixel split. Use this
script on the cluster for leave-one-region-out validation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

from src.data_utils import iter_aef_files, label_tile_ids
from src.enhanced_labels import load_tile_labels_enhanced
from src.geo_validation import resolve_train_val_tiles
from src.train_xgb_temporal import (
    _load_ndvi_stack,
    _load_s1_stack,
    _sample_pixels,
    _temporal_features,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XGBoost on AEF + temporal features with geographic val split."
    )
    parser.add_argument("--data-dir", default="./data/makeathon-challenge")
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--max-samples-per-tile", type=int, default=50_000)
    parser.add_argument("--neg-pos-ratio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--model-out",
        default="./artifacts/baseline_aef_xgb_temporal_geo.joblib",
    )
    parser.add_argument("--geojson-path", type=Path, default=None)
    parser.add_argument("--val-region", type=str, default="americas")
    parser.add_argument("--val-tiles", type=str, default="")
    parser.add_argument("--use-random-val-split", action="store_true")
    parser.add_argument(
        "--label-strategy",
        choices=("multi_source", "glads2_radd"),
        default="multi_source",
    )
    parser.add_argument("--forest-mask-dir", type=Path, default=None)
    parser.add_argument(
        "--metrics-plot",
        type=Path,
        default=None,
        help="If set, save PNG with train/val logloss curves and metric bars (needs matplotlib)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    aef_dir = data_dir / "aef-embeddings" / "train"
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    geojson = args.geojson_path or (data_dir / "metadata" / "train_tiles.geojson")
    val_tiles_arg: set[str] | None = None
    if args.val_tiles.strip():
        val_tiles_arg = {t.strip() for t in args.val_tiles.split(",") if t.strip()}

    label_tiles = label_tile_ids(data_dir / "labels" / "train")
    logger.info("Label tiles available: %d", len(label_tiles))

    if args.use_random_val_split:
        train_tile_ids = set(label_tiles)
        val_tile_ids: set[str] = set()
        split_desc = "legacy_random_pixels"
    else:
        train_tile_ids, val_tile_ids, split_desc = resolve_train_val_tiles(
            label_tiles,
            val_region=args.val_region,
            val_tiles=val_tiles_arg,
            geojson_path=geojson,
            use_random_split=False,
            random_split_fraction=0.2,
            seed=args.seed,
        )
    logger.info(
        "Split (%s): train_tiles=%d val_tiles=%d",
        split_desc,
        len(train_tile_ids),
        len(val_tile_ids),
    )
    if val_tile_ids:
        logger.info("Val tiles: %s", sorted(val_tile_ids))

    xs_train: list[np.ndarray] = []
    ys_train: list[np.ndarray] = []
    xs_val: list[np.ndarray] = []
    ys_val: list[np.ndarray] = []
    xs_pool: list[np.ndarray] = []
    ys_pool: list[np.ndarray] = []

    for aef_path in iter_aef_files(aef_dir):
        tile_id, year_str = aef_path.stem.rsplit("_", 1)
        year = int(year_str)
        if year < 2020 or tile_id not in label_tiles:
            continue

        in_val = tile_id in val_tile_ids
        in_train = tile_id in train_tile_ids
        if not args.use_random_val_split and not in_val and not in_train:
            continue

        with rasterio.open(aef_path) as src:
            aef = src.read().astype(np.float32)
            ref_profile = src.profile

        labels = load_tile_labels_enhanced(
            data_dir,
            tile_id,
            ref_profile,
            year=year,
            forest_mask_dir=args.forest_mask_dir,
            label_strategy=args.label_strategy,
        )
        if labels is None:
            continue

        try:
            ndvi_stack = _load_ndvi_stack(tile_id, data_dir, ref_profile)
            s1_stack = _load_s1_stack(tile_id, data_dir, ref_profile)
            ndvi_delta, ndvi_max_drop, s1_change, ndvi_var = _temporal_features(
                ndvi_stack, s1_stack
            )
        except RuntimeError as exc:
            logger.warning("Skipping %s (%s)", tile_id, exc)
            continue

        channels, height, width = aef.shape
        flat = aef.reshape(channels, height * width).transpose(1, 0)
        temporal = np.stack([ndvi_delta, ndvi_max_drop, s1_change, ndvi_var], axis=0)
        temporal = temporal.reshape(4, height * width).transpose(1, 0)
        features = np.concatenate([flat, temporal], axis=1)

        pos_idx = labels.positive.reshape(-1)
        neg_idx = labels.negative.reshape(-1)
        pos_features = features[pos_idx]
        neg_features = features[neg_idx]

        if pos_features.size == 0 or neg_features.size == 0:
            continue

        rng = np.random.default_rng(args.seed)
        n_pos = pos_features.shape[0]
        n_neg = min(neg_features.shape[0], n_pos * args.neg_pos_ratio)
        neg_sel = rng.choice(neg_features.shape[0], size=n_neg, replace=False)
        x = np.concatenate([pos_features, neg_features[neg_sel]], axis=0)
        y = np.concatenate(
            [np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)], axis=0
        )
        x, y = _sample_pixels(x, y, args.max_samples_per_tile, args.seed + hash(tile_id) % 10000)

        if args.use_random_val_split:
            xs_pool.append(x)
            ys_pool.append(y)
        else:
            if in_train:
                xs_train.append(x)
                ys_train.append(y)
            if in_val:
                xs_val.append(x)
                ys_val.append(y)

        logger.info("Loaded %s (%s): %d pos, %d neg", tile_id, year, n_pos, n_neg)

    if args.use_random_val_split:
        x_all = np.concatenate(xs_pool, axis=0)
        y_all = np.concatenate(ys_pool, axis=0)
        x_all, y_all = _sample_pixels(x_all, y_all, args.max_samples, args.seed)
        x_train, x_val, y_train, y_val = train_test_split(
            x_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
        )
    else:
        if not xs_train or not xs_val:
            raise RuntimeError("Need both train and val tiles; adjust val-region or val-tiles.")
        x_train = np.concatenate(xs_train, axis=0)
        y_train = np.concatenate(ys_train, axis=0)
        x_val = np.concatenate(xs_val, axis=0)
        y_val = np.concatenate(ys_val, axis=0)
        x_train, y_train = _sample_pixels(x_train, y_train, args.max_samples, args.seed)
        x_val, y_val = _sample_pixels(
            x_val, y_val, max(10_000, args.max_samples // 5), args.seed + 1
        )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=args.seed,
    )
    # Log train vs validation logloss each boosting round (for optional plots).
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        verbose=False,
    )
    y_pred = model.predict(x_val)
    y_proba = model.predict_proba(x_val)[:, 1]
    logger.info("Validation report:\n%s", classification_report(y_val, y_pred))

    try:
        auc = roc_auc_score(y_val, y_proba)
        logger.info("Validation ROC-AUC (proba vs label): %.4f", auc)
    except ValueError as exc:
        logger.warning("Could not compute ROC-AUC: %s", exc)

    if args.metrics_plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.metrics_plot.parent.mkdir(parents=True, exist_ok=True)
        evals = model.evals_result()
        # eval_set order: (train), (val) -> validation_0, validation_1
        k0, k1 = "validation_0", "validation_1"
        if k0 not in evals or k1 not in evals:
            keys = list(evals.keys())
            logger.warning("Unexpected evals_result keys %s; skipping plot", keys)
        else:
            train_loss = evals[k0]["logloss"]
            val_loss = evals[k1]["logloss"]
            rounds = np.arange(1, len(val_loss) + 1)

            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average=None, labels=[0, 1], zero_division=0
            )

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(rounds, train_loss, label="train set (monitoring)")
            axes[0].plot(rounds, val_loss, label="validation / held-out tiles")
            axes[0].set_xlabel("Boosting round")
            axes[0].set_ylabel("Log loss")
            axes[0].set_title("XGBoost learning curves")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            xpos = np.arange(2)
            w = 0.25
            axes[1].bar(xpos - w, [prec[0], prec[1]], width=w, label="precision")
            axes[1].bar(xpos, [rec[0], rec[1]], width=w, label="recall")
            axes[1].bar(xpos + w, [f1[0], f1[1]], width=w, label="F1")
            axes[1].set_xticks(xpos)
            axes[1].set_xticklabels(["class 0 (no deforest.)", "class 1 (deforest.)"])
            axes[1].set_ylabel("Score")
            axes[1].set_ylim(0, 1.05)
            axes[1].set_title("Validation per-class metrics")
            axes[1].legend()
            axes[1].grid(True, axis="y", alpha=0.3)

            fig.tight_layout()
            fig.savefig(args.metrics_plot, dpi=150)
            plt.close(fig)
            logger.info("Saved metrics plot to %s", args.metrics_plot)

    joblib.dump(model, model_out)
    logger.info("Saved model to %s", model_out)


if __name__ == "__main__":
    main()
