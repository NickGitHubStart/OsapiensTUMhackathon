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
