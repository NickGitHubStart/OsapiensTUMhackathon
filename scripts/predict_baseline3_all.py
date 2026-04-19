"""Generate GeoJSON predictions for baseline3 across all tiles."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
import sys
import re

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import binary_closing, binary_opening, label as cc_label
from shapely.geometry import shape

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

try:
    sys.path.insert(0, str(repo_root / "ONI-makeathon-challenge-2026-main"))
    from submission_utils import raster_to_geojson
except Exception as exc:  # pragma: no cover
    raise RuntimeError("submission_utils.py not found") from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_TILE_META_CACHE: dict[str, tuple[gpd.GeoDataFrame, str]] = {}


def _resolve_model_keys(models: dict, ensemble: str) -> list[str]:
    if not models:
        raise ValueError("No models found in bundle.")

    if ensemble == "average_all":
        return list(models.keys())

    if "all_data" in models:
        return ["all_data"]
    if "all" in models:
        return ["all"]
    return [next(iter(models.keys()))]


def _collect_tiles(data_dir: Path, split: str) -> list[str]:
    if split == "test":
        meta_path = data_dir / "metadata" / "test_tiles.geojson"
    else:
        meta_path = data_dir / "metadata" / "train_tiles.geojson"

    if meta_path.exists():
        gdf = gpd.read_file(meta_path)
        for col in ("name", "tile_id", "id"):
            if col in gdf.columns:
                return [str(name) for name in gdf[col].tolist()]
        raise RuntimeError(f"No tile id column found in {meta_path}")

    aef_dir = data_dir / "aef-embeddings" / split
    tile_ids = set()
    for path in aef_dir.glob("*.tiff"):
        tile_id = path.stem.rsplit("_", 1)[0]
        tile_ids.add(tile_id)
    return sorted(tile_ids)


def _available_years(aef_dir: Path, tile_id: str) -> list[int]:
    years: list[int] = []
    for path in aef_dir.glob(f"{tile_id}_*.tiff"):
        try:
            year = int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        years.append(year)
    return sorted(set(years))


def _predict_baseline2(aef: np.ndarray, aef_2020: np.ndarray, aef_prev: np.ndarray | None, model) -> np.ndarray:
    diff_2020 = aef - aef_2020
    if aef_prev is None:
        diff_prev = np.zeros_like(aef)
    else:
        diff_prev = aef - aef_prev

    channels, height, width = aef.shape
    base = aef.reshape(channels, height * width).transpose(1, 0)
    d2020 = diff_2020.reshape(channels, height * width).transpose(1, 0)
    dprev = diff_prev.reshape(channels, height * width).transpose(1, 0)
    features = np.concatenate([base, d2020, dprev], axis=1)
    proba = model.predict_proba(features)[:, 1]
    return proba.reshape(height, width)


def _combine_probs(probs: list[np.ndarray], mode: str) -> np.ndarray:
    stack = np.stack(probs, axis=0) 
    if mode == "mean":
        return np.mean(stack, axis=0)
    if mode == "min":
        return np.min(stack, axis=0)
    if mode == "geomean":
        eps = 1e-6
        return np.exp(np.mean(np.log(np.clip(stack, eps, 1.0)), axis=0))
    raise ValueError(f"Unknown ensemble mode: {mode}")


def _write_geojson(mask: np.ndarray, ref_profile: dict, out_path: Path, min_area_ha: float) -> dict:
    meta = ref_profile.copy()
    meta.update(dtype="uint8", nodata=0, count=1)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(mask.astype(np.uint8), 1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return raster_to_geojson(tmp_path, output_path=str(out_path), min_area_ha=min_area_ha)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _get_tile_meta(data_dir: Path, split: str) -> tuple[gpd.GeoDataFrame, str] | None:
    cache_key = f"{data_dir.resolve()}::{split}"
    if cache_key in _TILE_META_CACHE:
        return _TILE_META_CACHE[cache_key]

    meta_path = data_dir / "metadata" / ("test_tiles.geojson" if split == "test" else "train_tiles.geojson")
    if not meta_path.exists():
        return None

    gdf = gpd.read_file(meta_path)
    id_col = None
    for col in ("name", "tile_id", "id"):
        if col in gdf.columns:
            id_col = col
            break
    if id_col is None:
        return None

    _TILE_META_CACHE[cache_key] = (gdf, id_col)
    return gdf, id_col


def _parse_origin_epsg(origin: str) -> int | None:
    match = re.search(r"SRID=(\d+)", origin)
    if not match:
        return None
    return int(match.group(1))


def _get_tile_utm_profile(data_dir: Path, split: str, tile_id: str, src_profile: dict) -> dict | None:
    s2_dir = data_dir / "sentinel-2" / split / f"{tile_id}__s2_l2a"
    if s2_dir.exists():
        s2_files = sorted(s2_dir.glob("*.tif"))
        if s2_files:
            with rasterio.open(s2_files[0]) as src:
                return {
                    "crs": src.crs,
                    "transform": src.transform,
                    "height": src.height,
                    "width": src.width,
                }

    meta = _get_tile_meta(data_dir, split)
    if meta is None:
        return None
    gdf, id_col = meta
    rows = gdf[gdf[id_col] == tile_id]
    if rows.empty:
        return None

    origin = rows.iloc[0].get("origin")
    if not isinstance(origin, str):
        return None
    epsg = _parse_origin_epsg(origin)
    if epsg is None:
        return None

    dst_crs = CRS.from_epsg(epsg)
    bounds = array_bounds(src_profile["height"], src_profile["width"], src_profile["transform"])
    dst_transform, width, height = calculate_default_transform(
        src_profile["crs"], dst_crs, src_profile["width"], src_profile["height"], *bounds, resolution=10
    )
    return {
        "crs": dst_crs,
        "transform": dst_transform,
        "height": height,
        "width": width,
    }


def _postprocess_and_polygonize(
    pred: np.ndarray,
    src_profile: dict,
    utm_profile: dict,
    out_path: Path,
    min_area_ha: float,
    close_kernel: int,
    open_kernel: int,
) -> dict:
    dst = np.zeros((utm_profile["height"], utm_profile["width"]), dtype=np.uint8)
    reproject(
        source=pred.astype(np.uint8),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=utm_profile["transform"],
        dst_crs=utm_profile["crs"],
        resampling=Resampling.nearest,
    )

    raw_pos = int(pred.sum())
    closed = binary_closing(dst.astype(bool), structure=np.ones((close_kernel, close_kernel), dtype=bool))
    opened = binary_opening(closed, structure=np.ones((open_kernel, open_kernel), dtype=bool))
    morph_pos = int(opened.sum())

    labels, _ = cc_label(opened)
    pixel_area_ha = abs(utm_profile["transform"].a * utm_profile["transform"].e) / 10_000
    min_pixels = int(np.ceil(min_area_ha / pixel_area_ha))

    component_sizes = np.bincount(labels.ravel())
    keep_labels = np.where(component_sizes >= min_pixels)[0]
    keep_labels = keep_labels[keep_labels != 0]
    keep_mask = np.isin(labels, keep_labels)
    filtered_pos = int(keep_mask.sum())
    predicted_area_ha = filtered_pos * pixel_area_ha

    logger.info(
        "%s: raw=%d morph=%d filtered=%d area_ha=%.2f",
        out_path.stem,
        raw_pos,
        morph_pos,
        filtered_pos,
        predicted_area_ha,
    )

    if filtered_pos == 0:
        raise ValueError("All polygons removed by postprocessing.")

    polygons = [
        shape(geom)
        for geom, value in shapes(keep_mask.astype(np.uint8), mask=keep_mask, transform=utm_profile["transform"])
        if value == 1
    ]

    if not polygons:
        raise ValueError("No polygons generated after postprocessing.")

    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=utm_profile["crs"])
    gdf = gdf.to_crs("EPSG:4326")
    gdf["time_step"] = None
    geojson = json.loads(gdf.to_json())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(geojson))
    return geojson


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline3 GeoJSON predictions for all tiles.")
    parser.add_argument("--data-dir", default="./data/makeathon-challenge")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--year", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ha", type=float, default=2.0)
    parser.add_argument("--apply-postprocess", action="store_true")
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--ensemble", choices=["all_data", "average_all"], default="all_data")
    parser.add_argument(
        "--ensemble-mode",
        choices=["mean", "min", "geomean"],
        default="mean",
    )
    parser.add_argument("--tile-ids", default="", help="Comma-separated tile ids to run.")
    parser.add_argument("--out-dir", default="./submission/baseline3")
    parser.add_argument("--merge-out", default="")
    parser.add_argument("--debug-stats", action="store_true", help="Log per-tile proba stats.")
    parser.add_argument(
        "--allow-empty-merge",
        action="store_true",
        help="Write merged GeoJSON even if it has zero features.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.bundle_path)
    models = bundle.get("models", {})

    model_keys = _resolve_model_keys(models, args.ensemble)

    for key in model_keys:
        if key not in models:
            raise ValueError(f"Model key not found in bundle: {key}")

    aef_dir = data_dir / "aef-embeddings" / args.split
    tiles = _collect_tiles(data_dir, args.split)
    if args.tile_ids:
        tiles = [t.strip() for t in args.tile_ids.split(",") if t.strip()]
        if not tiles:
            raise ValueError("No tile ids provided in --tile-ids.")

    all_features: list[dict] = []
    total_features = 0
    tiles_with_features = 0
    tiles_written = 0
    tiles_skipped = 0

    for tile_id in tiles:
        years = _available_years(aef_dir, tile_id)
        if 2020 not in years:
            logger.warning("Skipping %s (missing 2020 AEF)", tile_id)
            continue

        if args.year:
            target_years = [args.year]
        else:
            target_years = [year for year in years if year > 2020]
            if not target_years:
                target_years = [max(years)]
                logger.warning("%s: no post-2020 AEF years, using %s", tile_id, target_years[0])

        aef_2020_path = aef_dir / f"{tile_id}_2020.tiff"
        if not aef_2020_path.exists():
            logger.warning("Skipping %s (missing 2020 AEF)", tile_id)
            continue

        with rasterio.open(aef_2020_path) as src:
            aef_2020 = src.read().astype(np.float32)

        year_probs: list[np.ndarray] = []
        ref_profile = None

        for target_year in target_years:
            aef_path = aef_dir / f"{tile_id}_{target_year}.tiff"
            if not aef_path.exists():
                logger.warning("Skipping %s (%s missing)", tile_id, aef_path.name)
                continue

            with rasterio.open(aef_path) as src:
                aef = src.read().astype(np.float32)
                if ref_profile is None:
                    ref_profile = src.profile

            prev_path = aef_dir / f"{tile_id}_{target_year - 1}.tiff"
            aef_prev = None
            if target_year > 2020 and prev_path.exists():
                with rasterio.open(prev_path) as src:
                    aef_prev = src.read().astype(np.float32)

            probs: list[np.ndarray] = []
            for key in model_keys:
                probs.append(_predict_baseline2(aef, aef_2020, aef_prev, models[key]))
            year_probs.append(_combine_probs(probs, args.ensemble_mode))

        if not year_probs or ref_profile is None:
            logger.warning("Skipping %s (no usable AEF years)", tile_id)
            continue

        if args.year:
            proba = year_probs[0]
        else:
            proba = np.max(np.stack(year_probs, axis=0), axis=0)

        pred = (proba > args.threshold).astype(np.uint8)

        if args.debug_stats:
            proba_min = float(np.nanmin(proba))
            proba_max = float(np.nanmax(proba))
            proba_mean = float(np.nanmean(proba))
            frac_on = float(np.mean(proba > args.threshold))
            logger.info(
                "%s: proba min=%.4f max=%.4f mean=%.4f above_thr=%.5f",
                tile_id,
                proba_min,
                proba_max,
                proba_mean,
                frac_on,
            )

        if args.apply_postprocess:
            utm_profile = _get_tile_utm_profile(data_dir, args.split, tile_id, ref_profile)
            if utm_profile is None:
                logger.warning("%s: missing UTM profile, falling back to raster_to_geojson", tile_id)
                try:
                    geojson = _write_geojson(pred, ref_profile, out_dir / f"{tile_id}.geojson", min_area_ha=args.min_area_ha)
                except ValueError as exc:
                    logger.warning("%s: %s", tile_id, exc)
                    tiles_skipped += 1
                    continue
            else:
                try:
                    geojson = _postprocess_and_polygonize(
                        pred,
                        ref_profile,
                        utm_profile,
                        out_dir / f"{tile_id}.geojson",
                        min_area_ha=args.min_area_ha,
                        close_kernel=args.close_kernel,
                        open_kernel=args.open_kernel,
                    )
                except ValueError as exc:
                    logger.warning("%s: %s", tile_id, exc)
                    tiles_skipped += 1
                    continue
        else:
            try:
                geojson = _write_geojson(pred, ref_profile, out_dir / f"{tile_id}.geojson", min_area_ha=args.min_area_ha)
            except ValueError as exc:
                logger.warning("%s: %s", tile_id, exc)
                tiles_skipped += 1
                continue

        features = geojson.get("features", [])
        n_features = len(features)
        total_features += n_features
        if n_features > 0:
            tiles_with_features += 1
        tiles_written += 1

        if args.merge_out:
            all_features.extend(features)

        logger.info("Wrote %s", out_dir / f"{tile_id}.geojson")

    if args.merge_out:
        if not all_features and not args.allow_empty_merge:
            raise RuntimeError(
                "Merged GeoJSON has zero features. "
                "Lower --threshold or --min-area-ha, or inspect per-tile outputs."
            )
        merged = {"type": "FeatureCollection", "features": all_features}
        merged_path = Path(args.merge_out)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text(json.dumps(merged))
        logger.info("Wrote merged GeoJSON to %s", merged_path)

    logger.info(
        "Tiles: %d written, %d skipped; features: %d total across %d tiles",
        tiles_written,
        tiles_skipped,
        total_features,
        tiles_with_features,
    )


if __name__ == "__main__":
    main()
