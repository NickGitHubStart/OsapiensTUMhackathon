"""Generate GeoJSON predictions for baseline3 across all tiles."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
import sys

import geopandas as gpd
import joblib
import numpy as np
import rasterio

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data_utils import postprocess_prediction

try:
    sys.path.insert(0, str(repo_root / "ONI-makeathon-challenge-2026-main"))
    from submission_utils import raster_to_geojson
except Exception as exc:  # pragma: no cover
    raise RuntimeError("submission_utils.py not found") from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _collect_tiles(data_dir: Path, split: str) -> list[str]:
    if split == "test":
        meta_path = data_dir / "metadata" / "test_tiles.geojson"
    else:
        meta_path = data_dir / "metadata" / "train_tiles.geojson"

    if meta_path.exists():
        gdf = gpd.read_file(meta_path)
        return [str(name) for name in gdf["name"].tolist()]

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline3 GeoJSON predictions for all tiles.")
    parser.add_argument("--data-dir", default="./data/makeathon-challenge")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--year", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    parser.add_argument("--apply-postprocess", action="store_true")
    parser.add_argument("--ensemble", choices=["all_data", "average_all"], default="all_data")
    parser.add_argument("--out-dir", default="./submission/baseline3")
    parser.add_argument("--merge-out", default="")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.bundle_path)
    models = bundle.get("models", {})

    if args.ensemble == "average_all":
        model_keys = list(models.keys())
    else:
        model_keys = ["all_data"]

    for key in model_keys:
        if key not in models:
            raise ValueError(f"Model key not found in bundle: {key}")

    aef_dir = data_dir / "aef-embeddings" / args.split
    tiles = _collect_tiles(data_dir, args.split)

    all_features: list[dict] = []

    for tile_id in tiles:
        years = _available_years(aef_dir, tile_id)
        if 2020 not in years:
            logger.warning("Skipping %s (missing 2020 AEF)", tile_id)
            continue

        target_year = args.year if args.year else (max(years) if years else 0)
        if target_year == 0:
            logger.warning("Skipping %s (no AEF years)", tile_id)
            continue
        if target_year not in years:
            logger.warning("Skipping %s (missing %s AEF)", tile_id, target_year)
            continue

        aef_path = aef_dir / f"{tile_id}_{target_year}.tiff"
        aef_2020_path = aef_dir / f"{tile_id}_2020.tiff"
        prev_path = aef_dir / f"{tile_id}_{target_year - 1}.tiff"

        if not aef_path.exists() or not aef_2020_path.exists():
            logger.warning("Skipping %s (missing AEF files)", tile_id)
            continue

        with rasterio.open(aef_path) as src:
            aef = src.read().astype(np.float32)
            ref_profile = src.profile

        with rasterio.open(aef_2020_path) as src:
            aef_2020 = src.read().astype(np.float32)

        aef_prev = None
        if target_year > 2020 and prev_path.exists():
            with rasterio.open(prev_path) as src:
                aef_prev = src.read().astype(np.float32)

        probs: list[np.ndarray] = []
        for key in model_keys:
            probs.append(_predict_baseline2(aef, aef_2020, aef_prev, models[key]))
        proba = np.mean(np.stack(probs, axis=0), axis=0)

        pred = (proba > args.threshold).astype(np.uint8)
        if args.apply_postprocess:
            pred = postprocess_prediction(pred, ref_profile["transform"], min_area_ha=args.min_area_ha)
            geojson = _write_geojson(pred, ref_profile, out_dir / f"{tile_id}.geojson", min_area_ha=0.0)
        else:
            geojson = _write_geojson(pred, ref_profile, out_dir / f"{tile_id}.geojson", min_area_ha=args.min_area_ha)

        if args.merge_out:
            for feature in geojson.get("features", []):
                all_features.append(feature)

        logger.info("Wrote %s", out_dir / f"{tile_id}.geojson")

    if args.merge_out:
        merged = {"type": "FeatureCollection", "features": all_features}
        merged_path = Path(args.merge_out)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text(json.dumps(merged))
        logger.info("Wrote merged GeoJSON to %s", merged_path)


if __name__ == "__main__":
    main()
