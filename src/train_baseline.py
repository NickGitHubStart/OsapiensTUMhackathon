"""Train a basic baseline on AlphaEarth embeddings."""

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
    apply_feature_channel_dropout,
    apply_feature_noise,
    iter_aef_files,
    label_tile_ids,
    load_tile_labels,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _sample_pixels(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    if n <= max_samples:
        return features, labels

    idx = rng.choice(n, size=max_samples, replace=False)
    return features[idx], labels[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a basic baseline on AlphaEarth embeddings."
    )
    parser.add_argument(
        "--data-dir",
        default="./data/makeathon-challenge",
        help="Path to the downloaded dataset root",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="Maximum total samples to train on",
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
        "--aug-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std for feature augmentation",
    )
    parser.add_argument(
        "--aug-dropout-prob",
        type=float,
        default=0.0,
        help="Probability of feature channel dropout",
    )
    parser.add_argument(
        "--aug-dropout-frac",
        type=float,
        default=0.1,
        help="Fraction of channels to drop when applying dropout",
    )
    parser.add_argument(
        "--model-out",
        default="./artifacts/baseline_aef_logreg.joblib",
        help="Where to save the trained model",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    aef_dir = data_dir / "aef-embeddings" / "train"
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)

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

        # Flatten features to (N, C)
        channels, height, width = aef.shape
        flat = aef.reshape(channels, height * width).transpose(1, 0)

        pos_idx = labels.positive.reshape(-1)
        neg_idx = labels.negative.reshape(-1)

        pos_features = flat[pos_idx]
        neg_features = flat[neg_idx]

        if pos_features.size == 0 or neg_features.size == 0:
            logger.info(
                "Skipping %s (pos=%d, neg=%d)",
                tile_id,
                pos_features.shape[0],
                neg_features.shape[0],
            )
            continue

        n_pos = pos_features.shape[0]
        n_neg = min(neg_features.shape[0], n_pos * args.neg_pos_ratio)
        neg_sel = rng.choice(neg_features.shape[0], size=n_neg, replace=False)

        x = np.concatenate([pos_features, neg_features[neg_sel]], axis=0)
        x = apply_feature_noise(x, rng, args.aug_noise_std)
        x = apply_feature_channel_dropout(
            x, rng, args.aug_dropout_prob, args.aug_dropout_frac
        )
        y = np.concatenate(
            [np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)], axis=0
        )

        xs.append(x)
        ys.append(y)

        logger.info(
            "Loaded %s (%s): %d pos, %d neg",
            tile_id,
            year,
            n_pos,
            n_neg,
        )

    if not xs:
        raise RuntimeError("No training samples were collected.")

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    x_all, y_all = _sample_pixels(x_all, y_all, args.max_samples, args.seed)

    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )

    model = XGBClassifier(
        n_estimators=300,
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
