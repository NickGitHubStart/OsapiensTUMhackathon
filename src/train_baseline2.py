"""Train a baseline with AEF temporal diff features."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

from src.data_utils import iter_aef_files, label_tile_ids, load_tile_labels

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _group_aef_by_tile(aef_dir: Path) -> dict[str, dict[int, Path]]:
    tiles: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in iter_aef_files(aef_dir):
        tile_id, year_str = path.stem.rsplit("_", 1)
        try:
            year = int(year_str)
        except ValueError:
            continue
        tiles[tile_id][year] = path
    return tiles


def _sample_rows(
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
        description="Train baseline2 with AEF temporal diff features."
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
        "--model-out",
        default="./artifacts/baseline2_aef_xgb.joblib",
        help="Where to save the trained model",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    aef_dir = data_dir / "aef-embeddings" / "train"
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    label_tiles = label_tile_ids(data_dir / "labels" / "train")
    logger.info("Label tiles available: %d", len(label_tiles))

    tiles = _group_aef_by_tile(aef_dir)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)

    for tile_id, years in tiles.items():
        if tile_id not in label_tiles:
            logger.info("Skipping %s (labels not available)", tile_id)
            continue

        if 2020 not in years:
            logger.info("Skipping %s (missing 2020 AEF)", tile_id)
            continue

        with rasterio.open(years[2020]) as src:
            aef_2020 = src.read().astype(np.float32)
            ref_profile = src.profile

        labels = load_tile_labels(data_dir, tile_id, ref_profile)
        if labels is None:
            logger.warning("Missing labels for %s, skipping", tile_id)
            continue

        for year, path in sorted(years.items()):
            if year < 2020:
                continue

            with rasterio.open(path) as src:
                aef = src.read().astype(np.float32)

            if year == 2020:
                diff_prev = np.zeros_like(aef)
            else:
                prev_path = years.get(year - 1)
                if prev_path is None:
                    diff_prev = np.zeros_like(aef)
                else:
                    with rasterio.open(prev_path) as src:
                        aef_prev = src.read().astype(np.float32)
                    diff_prev = aef - aef_prev

            diff_2020 = aef - aef_2020

            channels, height, width = aef.shape
            base = aef.reshape(channels, height * width).transpose(1, 0)
            d2020 = diff_2020.reshape(channels, height * width).transpose(1, 0)
            dprev = diff_prev.reshape(channels, height * width).transpose(1, 0)

            features = np.concatenate([base, d2020, dprev], axis=1)

            pos_idx = labels.positive.reshape(-1)
            neg_idx = labels.negative.reshape(-1)

            pos_features = features[pos_idx]
            neg_features = features[neg_idx]

            if pos_features.size == 0 or neg_features.size == 0:
                logger.info(
                    "Skipping %s (%s) (pos=%d, neg=%d)",
                    tile_id,
                    year,
                    pos_features.shape[0],
                    neg_features.shape[0],
                )
                continue

            n_pos = pos_features.shape[0]
            n_neg = min(neg_features.shape[0], n_pos * args.neg_pos_ratio)
            neg_sel = rng.choice(neg_features.shape[0], size=n_neg, replace=False)

            x = np.concatenate([pos_features, neg_features[neg_sel]], axis=0)
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
