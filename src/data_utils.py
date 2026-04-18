"""Shared data helpers for training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_opening, label as cc_label


@dataclass
class TileLabels:
    positive: np.ndarray
    negative: np.ndarray


def s2_cloud_mask(s2_stack: np.ndarray) -> np.ndarray:
    blue = s2_stack[1].astype(np.float32)
    cirrus = s2_stack[9].astype(np.float32)
    is_cloud = (blue > 2500) | (cirrus > 200)
    no_data = (s2_stack[1] == 0) & (s2_stack[2] == 0) & (s2_stack[3] == 0)
    return is_cloud | no_data


def postprocess_prediction(
    pred_binary: np.ndarray,
    transform,
    min_area_ha: float = 0.5,
) -> np.ndarray:
    opened = binary_opening(pred_binary.astype(bool), structure=np.ones((3, 3), dtype=bool))
    labels, _ = cc_label(opened)
    pixel_area_ha = abs(transform.a * transform.e) / 10_000
    min_pixels = int(np.ceil(min_area_ha / pixel_area_ha))

    component_sizes = np.bincount(labels.ravel())
    keep_labels = np.where(component_sizes >= min_pixels)[0]
    keep_labels = keep_labels[keep_labels != 0]

    keep_mask = np.isin(labels, keep_labels)
    return keep_mask.astype(np.uint8)


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

    pos_sources: list[np.ndarray] = []
    neg_sources: list[np.ndarray] = []

    if glads2_alert.exists() and glads2_date.exists():
        glads2_alert_r = reproject_to_match(glads2_alert, ref_profile)
        glads2_date_r = reproject_to_match(glads2_date, ref_profile)
        glads2_date = np.datetime64("2019-01-01") + glads2_date_r.astype("timedelta64[D]")
        glads2_pos = (glads2_alert_r >= 2) & (
            glads2_date >= np.datetime64("2020-01-01")
        )
        glads2_neg = glads2_alert_r == 0
        pos_sources.append(glads2_pos)
        neg_sources.append(glads2_neg)

    if radd_labels.exists():
        radd_r = reproject_to_match(radd_labels, ref_profile)
        radd_conf = radd_r // 10000
        radd_days = radd_r % 10000
        radd_date = np.datetime64("2014-12-31") + radd_days.astype("timedelta64[D]")
        radd_pos = (radd_conf >= 2) & (radd_date >= np.datetime64("2020-01-01"))
        radd_neg = radd_r == 0
        pos_sources.append(radd_pos)
        neg_sources.append(radd_neg)

    gladl_alert_paths = list((labels_dir / "gladl").glob(f"gladl_{tile_id}_alert*.tif"))
    for alert_path in sorted(gladl_alert_paths):
        year_str = alert_path.stem.split("_alert")[-1]
        date_path = alert_path.with_name(f"gladl_{tile_id}_alertDate{year_str}.tif")
        if not date_path.exists():
            continue

        try:
            year = int(f"20{year_str}")
        except ValueError:
            continue

        alert_r = reproject_to_match(alert_path, ref_profile)
        date_r = reproject_to_match(date_path, ref_profile)
        date = np.datetime64(f"{year}-01-01") + date_r.astype("timedelta64[D]")
        year_pos = (alert_r >= 2) & (date >= np.datetime64("2020-01-01"))

        if "gladl_pos" not in locals():
            gladl_pos = year_pos
            gladl_neg = alert_r == 0
        else:
            gladl_pos = gladl_pos | year_pos
            gladl_neg = gladl_neg & (alert_r == 0)

    if "gladl_pos" in locals():
        pos_sources.append(gladl_pos)
        neg_sources.append(gladl_neg)

    if not pos_sources:
        return None

    pos_stack = np.stack(pos_sources, axis=0)
    neg_stack = np.stack(neg_sources, axis=0)
    n_sources = pos_stack.shape[0]
    threshold = (n_sources + 1) // 2

    pos_votes = pos_stack.sum(axis=0)
    neg_votes = neg_stack.sum(axis=0)

    positive = (pos_votes >= threshold) & (neg_votes < threshold)
    negative = (neg_votes >= threshold) & (pos_votes < threshold)

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
    glads2_tiles = glads2_alert & glads2_date

    radd_tiles = {
        p.name.replace("radd_", "").replace("_labels.tif", "")
        for p in (labels_dir / "radd").glob("radd_*_labels.tif")
    }

    gladl_alert = (labels_dir / "gladl").glob("gladl_*_alert*.tif")
    gladl_date = (labels_dir / "gladl").glob("gladl_*_alertDate*.tif")

    def _parse_gladl(path: Path) -> tuple[str, str] | None:
        name = path.name
        if "_alertDate" in name:
            tile = name.replace("gladl_", "").split("_alertDate")[0]
            year = name.split("_alertDate")[-1].replace(".tif", "")
        elif "_alert" in name:
            tile = name.replace("gladl_", "").split("_alert")[0]
            year = name.split("_alert")[-1].replace(".tif", "")
        else:
            return None
        return tile, year

    gladl_alert_map: dict[str, set[str]] = {}
    for path in gladl_alert:
        parsed = _parse_gladl(path)
        if parsed is None:
            continue
        tile, year = parsed
        gladl_alert_map.setdefault(tile, set()).add(year)

    gladl_date_map: dict[str, set[str]] = {}
    for path in gladl_date:
        parsed = _parse_gladl(path)
        if parsed is None:
            continue
        tile, year = parsed
        gladl_date_map.setdefault(tile, set()).add(year)

    gladl_tiles: set[str] = set()
    for tile, years in gladl_alert_map.items():
        if tile in gladl_date_map and years & gladl_date_map[tile]:
            gladl_tiles.add(tile)

    return glads2_tiles | radd_tiles | gladl_tiles


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
