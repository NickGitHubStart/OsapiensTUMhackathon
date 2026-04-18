"""Train a pixel-wise XGBoost model with temporal hand-engineered features."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

S2_PATTERN = re.compile(r"_s2_l2a_(\d{4})_(\d{1,2})\.tif$")
S1_PATTERN = re.compile(r"_s1_rtc_(\d{4})_(\d{1,2})_.*\.tif$")


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


def _reproject_array(src_array: np.ndarray, src_transform, src_crs, ref_profile: dict) -> np.ndarray:
    dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=Resampling.nearest,
    )
    return dst


def _load_tile_labels(data_dir: Path, tile_id: str, ref_profile: dict) -> TileLabels | None:
    labels_dir = data_dir / "labels" / "train"

    glads2_alert = labels_dir / "glads2" / f"glads2_{tile_id}_alert.tif"
    glads2_date = labels_dir / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    radd_labels = labels_dir / "radd" / f"radd_{tile_id}_labels.tif"

    if not (glads2_alert.exists() and glads2_date.exists() and radd_labels.exists()):
        logger.warning("Missing labels for %s, skipping", tile_id)
        return None

    glads2_alert_r = _reproject_to_match(glads2_alert, ref_profile)
    glads2_date_r = _reproject_to_match(glads2_date, ref_profile)
    radd_r = _reproject_to_match(radd_labels, ref_profile)

    glads2_date = np.datetime64("2019-01-01") + glads2_date_r.astype("timedelta64[D]")
    glads2_pos = (glads2_alert_r >= 2) & (glads2_date >= np.datetime64("2020-01-01"))

    radd_conf = radd_r // 10000
    radd_days = radd_r % 10000
    radd_date = np.datetime64("2014-12-31") + radd_days.astype("timedelta64[D]")
    radd_pos = (radd_conf >= 2) & (radd_date >= np.datetime64("2020-01-01"))

    positive = glads2_pos & radd_pos
    negative = (glads2_alert_r == 0) & (radd_r == 0)

    return TileLabels(positive=positive, negative=negative)


def _iter_aef_files(aef_dir: Path) -> Iterable[Path]:
    yield from sorted(aef_dir.glob("*.tiff"))


def _label_tile_ids(labels_dir: Path) -> set[str]:
    glads2_alert = {
        p.name.replace("glads2_", "").replace("_alert.tif", "")
        for p in (labels_dir / "glads2").glob("glads2_*_alert.tif")
    }
    glads2_date = {
        p.name.replace("glads2_", "").replace("_alertDate.tif", "")
        for p in (labels_dir / "glads2").glob("glads2_*_alertDate.tif")
    }
    radd = {
        p.name.replace("radd_", "").replace("_labels.tif", "")
        for p in (labels_dir / "radd").glob("radd_*_labels.tif")
    }
    return glads2_alert & glads2_date & radd


def _iter_s2_files(tile_id: str, s2_dir: Path) -> list[Path]:
    return sorted((s2_dir / f"{tile_id}__s2_l2a").glob("*.tif"))


def _iter_s1_files(tile_id: str, s1_dir: Path) -> list[Path]:
    return sorted((s1_dir / f"{tile_id}__s1_rtc").glob(f"{tile_id}__s1_rtc_*.tif"))


def _parse_s2_date(path: Path) -> tuple[int, int] | None:
    match = S2_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_s1_date(path: Path) -> tuple[int, int] | None:
    match = S1_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    denom = nir + red
    return np.where(denom != 0, (nir - red) / denom, 0.0).astype(np.float32)


def _load_ndvi_stack(tile_id: str, data_dir: Path, ref_profile: dict) -> list[np.ndarray]:
    s2_dir = data_dir / "sentinel-2" / "train"
    ndvi_items: list[tuple[tuple[int, int], np.ndarray]] = []

    for path in _iter_s2_files(tile_id, s2_dir):
        date = _parse_s2_date(path)
        if date is None:
            continue

        with rasterio.open(path) as src:
            red = src.read(4).astype(np.float32)
            nir = src.read(8).astype(np.float32)
            ndvi = _compute_ndvi(nir, red)
            ndvi = _reproject_array(ndvi, src.transform, src.crs, ref_profile)

        ndvi_items.append((date, ndvi))

    ndvi_items.sort(key=lambda x: x[0])
    return [item[1] for item in ndvi_items]


def _load_s1_stack(tile_id: str, data_dir: Path, ref_profile: dict) -> list[np.ndarray]:
    s1_dir = data_dir / "sentinel-1" / "train"
    s1_items: list[tuple[tuple[int, int], np.ndarray]] = []

    for path in _iter_s1_files(tile_id, s1_dir):
        date = _parse_s1_date(path)
        if date is None:
            continue

        with rasterio.open(path) as src:
            backscatter = src.read(1).astype(np.float32)
            db = np.where(backscatter > 0, 10 * np.log10(backscatter), np.nan)
            db = np.nan_to_num(db, nan=0.0).astype(np.float32)
            db = _reproject_array(db, src.transform, src.crs, ref_profile)

        s1_items.append((date, db))

    s1_items.sort(key=lambda x: x[0])
    return [item[1] for item in s1_items]


def _temporal_features(ndvi_stack: list[np.ndarray], s1_stack: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    if not ndvi_stack:
        raise RuntimeError("No Sentinel-2 NDVI stack available.")

    ndvi_arr = np.stack(ndvi_stack, axis=0)
    ndvi_delta = ndvi_arr[-1] - ndvi_arr[0]
    ndvi_var = np.var(ndvi_arr, axis=0)

    if ndvi_arr.shape[0] > 1:
        drops = ndvi_arr[:-1] - ndvi_arr[1:]
        ndvi_max_drop = np.max(drops, axis=0)
    else:
        ndvi_max_drop = np.zeros_like(ndvi_arr[0])

    if not s1_stack:
        s1_change = np.zeros_like(ndvi_arr[0])
    else:
        s1_arr = np.stack(s1_stack, axis=0)
        s1_change = np.max(s1_arr, axis=0) - np.min(s1_arr, axis=0)

    return ndvi_delta, ndvi_max_drop, s1_change, ndvi_var


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
        description="Train an XGBoost pixel model with temporal features (NDVI/S1)."
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
        default="./artifacts/baseline_aef_xgb_temporal.joblib",
        help="Where to save the trained model",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    aef_dir = data_dir / "aef-embeddings" / "train"
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    label_tiles = _label_tile_ids(data_dir / "labels" / "train")
    logger.info("Label tiles available: %d", len(label_tiles))

    for aef_path in _iter_aef_files(aef_dir):
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

        labels = _load_tile_labels(data_dir, tile_id, ref_profile)
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
            logger.info(
                "Skipping %s (pos=%d, neg=%d)",
                tile_id,
                pos_features.shape[0],
                neg_features.shape[0],
            )
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
