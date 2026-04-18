"""Run baseline4 patch-based inference from cached features."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
import sys

import joblib
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import binary_opening, label as cc_label

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

try:
    sys.path.insert(0, str(repo_root / "ONI-makeathon-challenge-2026-main"))
    from submission_utils import raster_to_geojson
except Exception as exc:  # pragma: no cover
    raise RuntimeError("submission_utils.py not found") from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _iter_patch_coords(height: int, width: int, patch_size: int, stride: int):
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            yield y, x


def _patch_vector(patch: np.ndarray, include_gradient: bool) -> np.ndarray:
    mean = np.nanmean(patch, axis=(1, 2))
    std = np.nanstd(patch, axis=(1, 2))
    vmin = np.nanmin(patch, axis=(1, 2))
    vmax = np.nanmax(patch, axis=(1, 2))
    p10 = np.nanpercentile(patch, 10, axis=(1, 2))
    p90 = np.nanpercentile(patch, 90, axis=(1, 2))

    center_idx = patch.shape[1] // 2
    center = patch[:, center_idx, center_idx]

    parts = [mean, std, vmin, vmax, p10, p90, center]

    if include_gradient:
        patch_filled = np.nan_to_num(patch, nan=0.0)
        dy, dx = np.gradient(patch_filled, axis=(1, 2))
        grad_mag = np.sqrt(dx ** 2 + dy ** 2)
        grad_mean = np.mean(grad_mag, axis=(1, 2))
        parts.append(grad_mean)

    return np.concatenate(parts, axis=0)


def _spatial_features(
    ndvi_drop: np.ndarray,
    vv_drop: np.ndarray,
    forest_mask: np.ndarray,
    ndvi_thresh: float,
    vv_thresh: float,
) -> np.ndarray:
    ndvi_mask = ndvi_drop > ndvi_thresh
    vv_mask = vv_drop > vv_thresh

    ndvi_frac = np.nanmean(ndvi_mask)
    vv_frac = np.nanmean(vv_mask)
    forest_frac = np.mean(forest_mask)

    if np.isnan(ndvi_frac):
        ndvi_frac = 0.0
    if np.isnan(vv_frac):
        vv_frac = 0.0

    return np.array([ndvi_frac, vv_frac, forest_frac], dtype=np.float32)


def _predict_tile_proba(
    features: np.ndarray,
    ndvi_drop: np.ndarray,
    vv_drop: np.ndarray,
    forest_mask: np.ndarray,
    model,
    patch_size: int,
    stride: int,
    ndvi_thresh: float,
    vv_thresh: float,
    include_gradient: bool,
) -> np.ndarray:
    height, width = forest_mask.shape
    out = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for y0, x0 in _iter_patch_coords(height, width, patch_size, stride):
        patch_feat = features[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
        patch_ndvi_drop = ndvi_drop[y0 : y0 + patch_size, x0 : x0 + patch_size]
        patch_vv_drop = vv_drop[y0 : y0 + patch_size, x0 : x0 + patch_size]
        patch_forest = forest_mask[y0 : y0 + patch_size, x0 : x0 + patch_size]

        vec = _patch_vector(patch_feat, include_gradient)
        spatial = _spatial_features(patch_ndvi_drop, patch_vv_drop, patch_forest, ndvi_thresh, vv_thresh)
        vec = np.concatenate([vec, spatial], axis=0)[None, :]

        pred = float(model.predict(vec)[0])
        out[y0 : y0 + patch_size, x0 : x0 + patch_size] += pred
        counts[y0 : y0 + patch_size, x0 : x0 + patch_size] += 1.0

    counts[counts == 0] = 1.0
    return out / counts


def _postprocess_mask(
    pred: np.ndarray,
    transform: Affine,
    min_area_ha: float,
    apply_opening: bool,
) -> np.ndarray:
    if apply_opening:
        pred = binary_opening(pred.astype(bool), structure=np.ones((3, 3), dtype=bool))
    else:
        pred = pred.astype(bool)

    labels, _ = cc_label(pred)
    pixel_area_ha = abs(transform.a * transform.e) / 10_000
    min_pixels = int(np.ceil(min_area_ha / pixel_area_ha))

    component_sizes = np.bincount(labels.ravel())
    keep_labels = np.where(component_sizes >= min_pixels)[0]
    keep_labels = keep_labels[keep_labels != 0]

    keep_mask = np.isin(labels, keep_labels)
    return keep_mask.astype(np.uint8)


def _write_geojson(mask: np.ndarray, transform: Affine, crs: str, out_path: Path) -> None:
    meta = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
    }

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(mask.astype(np.uint8), 1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raster_to_geojson(tmp_path, output_path=str(out_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline4 patch-based inference.")
    parser.add_argument("--cache-dir", default="./data/makeathon-challenge-cache/baseline4")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--tile-ids", default="")
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--ensemble", choices=["all_data", "average_all", "holdout_thailand", "holdout_colombia"], default="all_data")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--ndvi-drop-threshold", type=float, default=None)
    parser.add_argument("--vv-drop-threshold", type=float, default=None)
    parser.add_argument("--include-gradient", action="store_true")
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    parser.add_argument("--apply-opening", action="store_true")
    parser.add_argument("--out-dir", default="./submission")

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir) / args.split
    out_dir = Path(args.out_dir)

    bundle = joblib.load(args.bundle_path)
    models = bundle.get("models", {})
    thresholds = bundle.get("thresholds", {})

    if args.tile_ids:
        tile_ids = [t.strip() for t in args.tile_ids.split(",") if t.strip()]
        tiles = [cache_dir / f"{t}.npz" for t in tile_ids]
    else:
        tiles = sorted(cache_dir.glob("*.npz"))

    if not tiles:
        raise RuntimeError("No tiles found to predict.")

    if args.ensemble == "average_all":
        model_keys = list(models.keys())
    else:
        model_keys = [args.ensemble]

    for key in model_keys:
        if key not in models:
            raise ValueError(f"Model key not found in bundle: {key}")

    if args.threshold >= 0:
        threshold = args.threshold
    else:
        if args.ensemble == "average_all":
            vals = [thresholds.get(k, 0.5) for k in model_keys]
            threshold = float(np.mean(vals)) if vals else 0.5
        else:
            threshold = float(thresholds.get(model_keys[0], 0.5))

    patch_size = args.patch_size or int(bundle.get("patch_size", 16))
    stride = args.stride or int(bundle.get("stride_infer", 8))
    ndvi_thresh = args.ndvi_drop_threshold
    if ndvi_thresh is None:
        ndvi_thresh = float(bundle.get("ndvi_drop_threshold", 0.2))
    vv_thresh = args.vv_drop_threshold
    if vv_thresh is None:
        vv_thresh = float(bundle.get("vv_drop_threshold", 1.5))
    include_gradient = args.include_gradient or bool(bundle.get("include_gradient", False))

    for path in tiles:
        tile_id = path.stem
        data = np.load(path)
        features = data["features"].astype(np.float32)
        ndvi_drop = data["ndvi_drop"].astype(np.float32)
        vv_drop = data["vv_drop"].astype(np.float32)
        forest_mask = data["forest_mask"].astype(np.uint8)
        transform = Affine.from_gdal(*data["transform"].tolist())
        crs = str(data["crs"]) if data["crs"].size else ""

        probas = []
        for key in model_keys:
            model = models[key]
            proba = _predict_tile_proba(
                features,
                ndvi_drop,
                vv_drop,
                forest_mask,
                model,
                patch_size,
                stride,
                ndvi_thresh,
                vv_thresh,
                include_gradient,
            )
            probas.append(proba)

        proba = np.mean(np.stack(probas, axis=0), axis=0)
        pred = (proba > threshold).astype(np.uint8)
        pred[forest_mask == 0] = 0

        pred = _postprocess_mask(pred, transform, args.min_area_ha, args.apply_opening)
        out_path = out_dir / f"{tile_id}.geojson"
        _write_geojson(pred, transform, crs, out_path)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
