"""Train a basic baseline on AlphaEarth embeddings."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class TileLabels:
    positive: np.ndarray
    negative: np.ndarray


def _reproject_to_match(src_path: Path, ref_profile: dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.uint16)
        reproject(
            source=src.read(1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
        )
        return dst


def _load_tile_labels(data_dir: Path, tile_id: str, ref_profile: dict) -> TileLabels:
    labels_dir = data_dir / "labels" / "train"

    glads2_alert = labels_dir / "glads2" / f"glads2_{tile_id}_alert.tif"
    glads2_date = labels_dir / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    radd_labels = labels_dir / "radd" / f"radd_{tile_id}_labels.tif"

    glads2_alert_r = _reproject_to_match(glads2_alert, ref_profile)
    glads2_date_r = _reproject_to_match(glads2_date, ref_profile)
    radd_r = _reproject_to_match(radd_labels, ref_profile)

    # GLAD-S2: day offset since 2019-01-01
    glads2_date = np.datetime64("2019-01-01") + glads2_date_r.astype("timedelta64[D]")
    glads2_pos = (glads2_alert_r >= 2) & (glads2_date >= np.datetime64("2020-01-01"))

    # RADD: confidence digit * 10000 + days since 2014-12-31
    radd_conf = radd_r // 10000
    radd_days = radd_r % 10000
    radd_date = np.datetime64("2014-12-31") + radd_days.astype("timedelta64[D]")
    radd_pos = (radd_conf >= 2) & (radd_date >= np.datetime64("2020-01-01"))

    positive = glads2_pos & radd_pos
    negative = (glads2_alert_r == 0) & (radd_r == 0)

    return TileLabels(positive=positive, negative=negative)


def _iter_aef_files(aef_dir: Path) -> Iterable[Path]:
    yield from sorted(aef_dir.glob("*.tiff"))


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

    for aef_path in _iter_aef_files(aef_dir):
        tile_id, year_str = aef_path.stem.rsplit("_", 1)
        year = int(year_str)
        if year < 2021:
            continue

        with rasterio.open(aef_path) as src:
            aef = src.read().astype(np.float32)
            ref_profile = src.profile

        labels = _load_tile_labels(data_dir, tile_id, ref_profile)

        # Flatten features to (N, C)
        channels, height, width = aef.shape
        flat = aef.reshape(channels, height * width).transpose(1, 0)

        pos_idx = labels.positive.reshape(-1)
        neg_idx = labels.negative.reshape(-1)

        pos_features = flat[pos_idx]
        neg_features = flat[neg_idx]

        if pos_features.size == 0 or neg_features.size == 0:
            logger.info("Skipping %s (no positives or negatives)", tile_id)
            continue

        rng = np.random.default_rng(args.seed)
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
