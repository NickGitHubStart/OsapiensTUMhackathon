"""Build a Sub-3-style submission from any baseline3 bundle, with per-polygon
year-of-max time_step.

Pipeline (per tile):
  1. Predict per-year geometric-mean probability for years 2021-2025 with the
     given baseline3 ensemble bundle.
  2. Take per-pixel MAX over years -> single probability map (same shape as
     the proven 44.98 % Sub 3 spatial mask).
  3. Threshold at --threshold (default 0.50), reproject to local UTM @ 10 m,
     closing kxk + opening jxj, polygonise, drop polygons < --min-area-ha.
  4. For each surviving polygon, look up which year had the highest MEAN
     probability inside it and write properties.time_step = YY*100 + month.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import binary_closing, binary_opening
from shapely.geometry import shape

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _features(aef_y, aef_2020, aef_prev):
    diff_2020 = aef_y - aef_2020
    diff_prev = aef_y - aef_prev if aef_prev is not None else np.zeros_like(aef_y)
    c, h, w = aef_y.shape
    feat = np.concatenate(
        [aef_y.reshape(c, h * w).T, diff_2020.reshape(c, h * w).T, diff_prev.reshape(c, h * w).T],
        axis=1,
    )
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def _predict_year(aef_y, aef_2020, aef_prev, bundle):
    feats = _features(aef_y, aef_2020, aef_prev)
    h, w = aef_y.shape[1], aef_y.shape[2]
    log_sum = np.zeros(feats.shape[0], dtype=np.float64)
    for _name, mdl in bundle["models"].items():
        p = mdl.predict_proba(feats)[:, 1]
        log_sum += np.log(np.clip(p, 1e-6, 1.0 - 1e-6))
    return np.exp(log_sum / len(bundle["models"])).astype(np.float32).reshape(h, w)


def _utm_for_tile(profile):
    src_crs = profile["crs"]
    if "utm" in str(src_crs).lower():
        return src_crs
    h, w = profile["height"], profile["width"]
    minx, miny, maxx, maxy = rasterio.transform.array_bounds(h, w, profile["transform"])
    pt = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([(minx + maxx) / 2], [(miny + maxy) / 2]),
        crs=src_crs,
    )
    return pt.estimate_utm_crs()


def _reproject(arr, src_profile, utm_crs, resampling=Resampling.bilinear, dtype=np.float32):
    h, w = src_profile["height"], src_profile["width"]
    minx, miny, maxx, maxy = rasterio.transform.array_bounds(h, w, src_profile["transform"])
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_profile["crs"], utm_crs, w, h, minx, miny, maxx, maxy, resolution=10.0
    )
    dst = np.zeros((dst_h, dst_w), dtype=dtype)
    reproject(
        source=arr.astype(dtype),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_transform,
        dst_crs=utm_crs,
        resampling=resampling,
    )
    return dst, dst_transform, (dst_h, dst_w)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("./data/makeathon-challenge"))
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min-area-ha", type=float, default=1.0)
    ap.add_argument("--closing", type=int, default=5)
    ap.add_argument("--opening", type=int, default=3)
    ap.add_argument("--month", type=int, default=6)
    ap.add_argument("--years", default="2021,2022,2023,2024,2025")
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(",")]
    bundle = joblib.load(args.bundle)
    logger.info("Loaded bundle from %s with %d models", args.bundle, len(bundle["models"]))

    aef_test = args.data_dir / "aef-embeddings" / "test"
    test_tiles = sorted(set(p.stem.rsplit("_", 1)[0] for p in aef_test.glob("*.tiff")))

    all_geoms_wgs = []
    all_ts: list[int] = []

    for tile_id in test_tiles:
        logger.info("==== %s ====", tile_id)
        a2020 = aef_test / f"{tile_id}_2020.tiff"
        if not a2020.is_file():
            logger.warning("missing 2020 AEF; skip")
            continue
        with rasterio.open(a2020) as src:
            aef_2020 = src.read().astype(np.float32)
            ref_profile = src.profile.copy()
        utm_crs = _utm_for_tile(ref_profile)

        per_year_utm = []
        utm_transform, utm_shape = None, None
        for y in years:
            yp = aef_test / f"{tile_id}_{y}.tiff"
            if not yp.is_file():
                logger.warning("  %d missing; zeros", y)
                if utm_shape is not None:
                    per_year_utm.append(np.zeros(utm_shape, dtype=np.float32))
                continue
            with rasterio.open(yp) as src:
                aef_y = src.read().astype(np.float32)
            pp = aef_test / f"{tile_id}_{y - 1}.tiff"
            aef_prev = None
            if pp.is_file():
                with rasterio.open(pp) as src:
                    aef_prev = src.read().astype(np.float32)
            prob_native = _predict_year(aef_y, aef_2020, aef_prev, bundle)
            prob_utm, utm_transform, utm_shape = _reproject(prob_native, ref_profile, utm_crs)
            per_year_utm.append(prob_utm)
            logger.info("  %d: prob_mean=%.4f", y, prob_utm.mean())

        prob_stack = np.stack(per_year_utm, axis=0)
        prob_max = prob_stack.max(axis=0)

        binary = (prob_max > args.threshold).astype(np.uint8)
        if not binary.any():
            continue
        if args.closing > 0:
            binary = binary_closing(binary, structure=np.ones((args.closing, args.closing))).astype(np.uint8)
        if args.opening > 0:
            binary = binary_opening(binary, structure=np.ones((args.opening, args.opening))).astype(np.uint8)
        if not binary.any():
            continue

        polys_utm = []
        for geom, val in shapes(binary, mask=binary.astype(bool), transform=utm_transform):
            if val == 1:
                polys_utm.append(shape(geom))
        gdf_utm = gpd.GeoDataFrame(geometry=polys_utm, crs=utm_crs)
        gdf_utm = gdf_utm[gdf_utm.area / 10_000.0 >= args.min_area_ha].reset_index(drop=True)
        if gdf_utm.empty:
            continue

        for geom_utm in gdf_utm.geometry:
            mask = rasterize(
                [(geom_utm, 1)],
                out_shape=utm_shape,
                transform=utm_transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            if not mask.any():
                ts = years[-1] % 100 * 100 + args.month
            else:
                means = prob_stack[:, mask].mean(axis=1)
                ts = years[int(means.argmax())] % 100 * 100 + args.month
            all_ts.append(int(ts))

        gdf_wgs = gdf_utm.to_crs("EPSG:4326")
        all_geoms_wgs.extend(list(gdf_wgs.geometry))
        logger.info("  polys=%d", len(gdf_utm))

    if not all_geoms_wgs:
        raise RuntimeError("no polygons produced")

    features = [
        {
            "type": "Feature",
            "properties": {"time_step": ts},
            "geometry": json.loads(gpd.GeoSeries([g], crs="EPSG:4326").to_json())["features"][0]["geometry"],
        }
        for g, ts in zip(all_geoms_wgs, all_ts)
    ]
    container = {
        "type": "FeatureCollection",
        "name": "submission",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(container, fh)
    dist: dict[int, int] = {}
    for ts in all_ts:
        dist[ts] = dist.get(ts, 0) + 1
    logger.info("Wrote %s (%d polygons)", args.out, len(all_ts))
    logger.info("time_step distribution: %s", dist)


if __name__ == "__main__":
    main()
