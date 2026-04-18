"""Shared data helpers for training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


@dataclass
class TileLabels:
    positive: np.ndarray
    negative: np.ndarray


def reproject_to_match(src_path: Path, ref_profile: dict) -> np.ndarray:
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


def reproject_array(src_array: np.ndarray, src_transform, src_crs, ref_profile: dict) -> np.ndarray:
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


def load_tile_labels(data_dir: Path, tile_id: str, ref_profile: dict) -> TileLabels | None:
    labels_dir = data_dir / "labels" / "train"

    glads2_alert = labels_dir / "glads2" / f"glads2_{tile_id}_alert.tif"
    glads2_date = labels_dir / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    radd_labels = labels_dir / "radd" / f"radd_{tile_id}_labels.tif"

    if not (glads2_alert.exists() and glads2_date.exists() and radd_labels.exists()):
        return None

    glads2_alert_r = reproject_to_match(glads2_alert, ref_profile)
    glads2_date_r = reproject_to_match(glads2_date, ref_profile)
    radd_r = reproject_to_match(radd_labels, ref_profile)

    glads2_date = np.datetime64("2019-01-01") + glads2_date_r.astype("timedelta64[D]")
    glads2_pos = (glads2_alert_r >= 2) & (glads2_date >= np.datetime64("2020-01-01"))

    radd_conf = radd_r // 10000
    radd_days = radd_r % 10000
    radd_date = np.datetime64("2014-12-31") + radd_days.astype("timedelta64[D]")
    radd_pos = (radd_conf >= 2) & (radd_date >= np.datetime64("2020-01-01"))

    positive = glads2_pos & radd_pos
    negative = (glads2_alert_r == 0) & (radd_r == 0)

    return TileLabels(positive=positive, negative=negative)


def iter_aef_files(aef_dir: Path) -> Iterable[Path]:
    yield from sorted(aef_dir.glob("*.tiff"))


def label_tile_ids(labels_dir: Path) -> set[str]:
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


def build_label_mask(labels: TileLabels) -> np.ndarray:
    mask = np.full(labels.positive.shape, -1, dtype=np.int8)
    mask[labels.negative] = 0
    mask[labels.positive] = 1
    return mask


def apply_feature_noise(features: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return features
    return features + rng.normal(0.0, sigma, size=features.shape).astype(features.dtype)


def apply_feature_channel_dropout(
    features: np.ndarray, rng: np.random.Generator, drop_prob: float, drop_frac: float
) -> np.ndarray:
    if drop_prob <= 0 or drop_frac <= 0:
        return features

    if rng.random() >= drop_prob:
        return features

    n_features = features.shape[1]
    n_drop = max(1, int(n_features * drop_frac))
    drop_idx = rng.choice(n_features, size=n_drop, replace=False)
    features = features.copy()
    features[:, drop_idx] = 0.0
    return features


def _resize_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    _, in_h, in_w = arr.shape
    if in_h == out_h and in_w == out_w:
        return arr

    y_idx = (np.linspace(0, in_h - 1, out_h)).round().astype(int)
    x_idx = (np.linspace(0, in_w - 1, out_w)).round().astype(int)
    return arr[:, y_idx][:, :, x_idx]


def apply_spatial_aug(
    patch: np.ndarray,
    label: np.ndarray | None,
    rng: np.random.Generator,
    flip_rotate_prob: float,
    scale_min: float,
    scale_max: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if flip_rotate_prob > 0 and rng.random() < flip_rotate_prob:
        k = int(rng.integers(0, 4))
        patch = np.rot90(patch, k=k, axes=(1, 2))
        if label is not None:
            label = np.rot90(label, k=k, axes=(0, 1))

        if rng.random() < 0.5:
            patch = patch[:, :, ::-1]
            if label is not None:
                label = label[:, ::-1]
        if rng.random() < 0.5:
            patch = patch[:, ::-1, :]
            if label is not None:
                label = label[::-1, :]

    if scale_max > 1.0 or scale_min < 1.0:
        scale = float(rng.uniform(scale_min, scale_max))
        _, h, w = patch.shape
        crop_h = max(1, min(h, int(round(h / scale))))
        crop_w = max(1, min(w, int(round(w / scale))))
        y0 = int(rng.integers(0, h - crop_h + 1))
        x0 = int(rng.integers(0, w - crop_w + 1))
        patch = patch[:, y0 : y0 + crop_h, x0 : x0 + crop_w]
        patch = _resize_nearest(patch, h, w)
        if label is not None:
            label = label[y0 : y0 + crop_h, x0 : x0 + crop_w]
            label = _resize_nearest(label[None, ...], h, w)[0]

    return patch, label


def apply_patch_noise(patch: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return patch
    return patch + rng.normal(0.0, sigma, size=patch.shape).astype(patch.dtype)


def apply_patch_channel_dropout(
    patch: np.ndarray, rng: np.random.Generator, drop_prob: float, drop_frac: float
) -> np.ndarray:
    if drop_prob <= 0 or drop_frac <= 0:
        return patch
    if rng.random() >= drop_prob:
        return patch

    n_channels = patch.shape[0]
    n_drop = max(1, int(n_channels * drop_frac))
    drop_idx = rng.choice(n_channels, size=n_drop, replace=False)
    patch = patch.copy()
    patch[drop_idx, :, :] = 0.0
    return patch
