"""Train a patch-based XGBoost baseline on AlphaEarth embeddings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

from src.data_utils import (
    build_label_mask,
    iter_aef_files,
    label_tile_ids,
    load_tile_labels,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _iter_patch_coords(height: int, width: int, patch_size: int, stride: int) -> Iterable[tuple[int, int]]:
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            yield y, x


def _patch_features(patch: np.ndarray) -> np.ndarray:
    mean = patch.mean(axis=(1, 2))
    std = patch.std(axis=(1, 2))
    return np.concatenate([mean, std], axis=0)


def _sample_rows(features: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if features.shape[0] <= max_samples:
        return features, labels

    rng = np.random.default_rng(seed)
    idx = rng.choice(features.shape[0], size=max_samples, replace=False)
    return features[idx], labels[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a patch-based XGBoost baseline on AlphaEarth embeddings."
    )
    parser.add_argument(
        "--data-dir",
        default="./data/makeathon-challenge",
        help="Path to the downloaded dataset root",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="Square patch size in pixels",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Stride between patches",
    )
    parser.add_argument(
        "--pos-frac",
        type=float,
        default=0.05,
        help="Minimum positive fraction to label a patch positive",
    )
    parser.add_argument(
        "--neg-frac",
        type=float,
        default=0.95,
        help="Minimum negative fraction to label a patch negative",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="Maximum total patch samples to train on",
    )
    parser.add_argument(
        "--neg-pos-ratio",
        type=int,
        default=3,
        help="Number of negatives per positive",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed",
    )
    parser.add_argument(
        "--model-out",
        default="./artifacts/baseline_aef_patch_xgb.joblib",
        help="Where to save the trained model",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    aef_dir = data_dir / "aef-embeddings" / "train"
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    pos_feats: list[np.ndarray] = []
    neg_feats: list[np.ndarray] = []

    label_tiles = label_tile_ids(data_dir / "labels" / "train")
    logger.info("Label tiles available: %d", len(label_tiles))

    for aef_path in iter_aef_files(aef_dir):
        tile_id, year_str = aef_path.stem.rsplit("_", 1)
        year = int(year_str)
        if year < 2020:
            continue

        if tile_id not in label_tiles:
            logger.info("Skipping %s (labels not available)", tile_id)
            continue

        with rasterio.open(aef_path) as src:
            aef = src.read().astype(np.float32)
            ref_profile = src.profile

        labels = load_tile_labels(data_dir, tile_id, ref_profile)
        if labels is None:
            logger.warning("Missing labels for %s, skipping", tile_id)
            continue

        label_mask = build_label_mask(labels)
        channels, height, width = aef.shape

        for y, x in _iter_patch_coords(height, width, args.patch_size, args.stride):
            label_patch = label_mask[y : y + args.patch_size, x : x + args.patch_size]
            valid = label_patch >= 0
            valid_count = int(valid.sum())
            if valid_count == 0:
                continue

            pos_frac = float((label_patch == 1).sum()) / valid_count
            neg_frac = float((label_patch == 0).sum()) / valid_count

            if pos_frac >= args.pos_frac:
                feat = _patch_features(aef[:, y : y + args.patch_size, x : x + args.patch_size])
                pos_feats.append(feat)
            elif neg_frac >= args.neg_frac:
                feat = _patch_features(aef[:, y : y + args.patch_size, x : x + args.patch_size])
                neg_feats.append(feat)

        logger.info(
            "Scanned %s (%s): pos=%d, neg=%d",
            tile_id,
            year,
            len(pos_feats),
            len(neg_feats),
        )

    if not pos_feats or not neg_feats:
        raise RuntimeError("No training patches were collected.")

    rng = np.random.default_rng(args.seed)
    pos_arr = np.stack(pos_feats, axis=0)
    neg_arr = np.stack(neg_feats, axis=0)

    max_neg = min(neg_arr.shape[0], pos_arr.shape[0] * args.neg_pos_ratio)
    neg_idx = rng.choice(neg_arr.shape[0], size=max_neg, replace=False)
    neg_arr = neg_arr[neg_idx]

    x_all = np.concatenate([pos_arr, neg_arr], axis=0)
    y_all = np.concatenate(
        [np.ones(pos_arr.shape[0], dtype=np.uint8), np.zeros(neg_arr.shape[0], dtype=np.uint8)],
        axis=0,
    )

    x_all, y_all = _sample_rows(x_all, y_all, args.max_samples, args.seed)

    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
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
    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)
    logger.info("Validation report:\n%s", classification_report(y_val, y_pred))

    joblib.dump(model, model_out)
    logger.info("Saved model to %s", model_out)


if __name__ == "__main__":
    main()
