"""Attach per-polygon time_step to an existing submission GeoJSON.

Loads polygons from a base GeoJSON (e.g. our 44.98 % Sub 3) and, for every
polygon, computes the per-year mean Baseline-3 probability inside it. The
year with the highest mean probability is written as ``properties.time_step``
in YYMM format (default month 06).

This lets us keep Sub 3's proven spatial mask (= Union IoU 44.98 %) while
adding a non-zero Year score to the leaderboard total.
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
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject, calculate_default_transform

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _features(aef_y: np.ndarray, aef_2020: np.ndarray, aef_prev: np.ndarray | None) -> np.ndarray:
    diff_2020 = aef_y - aef_2020
    diff_prev = aef_y - aef_prev if aef_prev is not None else np.zeros_like(aef_y)
    c, h, w = aef_y.shape
    feat = np.concatenate(
        [
            aef_y.reshape(c, h * w).T,
            diff_2020.reshape(c, h * w).T,
            diff_prev.reshape(c, h * w).T,
        ],
        axis=1,
    )
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def _predict_year_geo_mean(aef_y, aef_2020, aef_prev, bundle) -> np.ndarray:
    feats = _features(aef_y, aef_2020, aef_prev)
    h, w = aef_y.shape[1], aef_y.shape[2]
    log_sum = np.zeros(feats.shape[0], dtype=np.float64)
    for _name, mdl in bundle["models"].items():
        p = mdl.predict_proba(feats)[:, 1]
        log_sum += np.log(np.clip(p, 1e-6, 1.0 - 1e-6))
    return np.exp(log_sum / len(bundle["models"])).astype(np.float32).reshape(h, w)


def _utm_for_tile(profile: dict) -> "rasterio.crs.CRS":
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


def _reproject_probs_to_utm(prob_native, profile, utm_crs, resolution=10.0):
    h, w = profile["height"], profile["width"]
    minx, miny, maxx, maxy = rasterio.transform.array_bounds(h, w, profile["transform"])
    dst_transform, dst_w, dst_h = calculate_default_transform(
        profile["crs"], utm_crs, w, h, minx, miny, maxx, maxy, resolution=resolution
    )
    dst = np.zeros((dst_h, dst_w), dtype=np.float32)
    reproject(
        source=prob_native.astype(np.float32),
        destination=dst,
        src_transform=profile["transform"],
        src_crs=profile["crs"],
        dst_transform=dst_transform,
        dst_crs=utm_crs,
        resampling=Resampling.bilinear,
    )
    return dst, dst_transform, (dst_h, dst_w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-geojson", type=Path, required=True,
                    help="Submission to relabel (e.g. sub3_baseline3_44.98pct.geojson)")
    ap.add_argument("--data-dir", type=Path, default=Path("./data/makeathon-challenge"))
    ap.add_argument("--bundle", type=Path, default=Path("./models/baseline_xgb/baseline3_aef_logreg.joblib"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--years", default="2021,2022,2023,2024,2025")
    ap.add_argument("--month", type=int, default=6, help="Month component of YYMM time_step")
    ap.add_argument("--metadata-dir", type=Path, default=Path("./metadata"))
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(",")]
    bundle = joblib.load(args.bundle)
    logger.info("Loaded bundle with %d models", len(bundle["models"]))

    base = gpd.read_file(args.base_geojson)
    if base.crs is None:
        base = base.set_crs("EPSG:4326")
    else:
        base = base.to_crs("EPSG:4326")
    logger.info("Loaded %d base polygons from %s", len(base), args.base_geojson)

    test_tiles_path = args.metadata_dir / "test_tiles.geojson"
    test_tiles = gpd.read_file(test_tiles_path).to_crs("EPSG:4326")
    logger.info("Test tiles in metadata: %s", list(test_tiles["name"]))

    aef_test = args.data_dir / "aef-embeddings" / "test"

    final_ts = np.full(len(base), -1, dtype=np.int32)

    for _, tile_row in test_tiles.iterrows():
        tile_id = tile_row["name"]
        tile_geom = tile_row.geometry
        idxs = base.index[base.intersects(tile_geom)].tolist()
        if not idxs:
            logger.info("[%s] no base polygons intersect this tile", tile_id)
            continue
        logger.info("[%s] %d polygons", tile_id, len(idxs))

        aef_2020_path = aef_test / f"{tile_id}_2020.tiff"
        if not aef_2020_path.is_file():
            logger.warning("[%s] missing 2020 AEF, defaulting time_step to first year", tile_id)
            for i in idxs:
                final_ts[i] = years[0] % 100 * 100 + args.month
            continue
        with rasterio.open(aef_2020_path) as src:
            aef_2020 = src.read().astype(np.float32)
            ref_profile = src.profile.copy()

        utm_crs = _utm_for_tile(ref_profile)

        per_year_utm = []
        utm_transform = None
        utm_shape = None
        for y in years:
            ypath = aef_test / f"{tile_id}_{y}.tiff"
            if not ypath.is_file():
                logger.warning("[%s] missing AEF for %d", tile_id, y)
                if utm_shape is not None:
                    per_year_utm.append(np.zeros(utm_shape, dtype=np.float32))
                continue
            with rasterio.open(ypath) as src:
                aef_y = src.read().astype(np.float32)
            prev_path = aef_test / f"{tile_id}_{y - 1}.tiff"
            aef_prev = None
            if prev_path.is_file():
                with rasterio.open(prev_path) as src:
                    aef_prev = src.read().astype(np.float32)
            prob_native = _predict_year_geo_mean(aef_y, aef_2020, aef_prev, bundle)
            prob_utm, utm_transform, utm_shape = _reproject_probs_to_utm(
                prob_native, ref_profile, utm_crs
            )
            per_year_utm.append(prob_utm)
            logger.info("  %d: prob_mean=%.4f", y, prob_utm.mean())

        prob_stack_utm = np.stack(per_year_utm, axis=0)

        sub_polys = base.loc[idxs].to_crs(utm_crs)
        for local_i, (gi, geom) in enumerate(zip(idxs, sub_polys.geometry), start=1):
            mask = rasterize(
                [(geom, 1)],
                out_shape=utm_shape,
                transform=utm_transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            if not mask.any():
                final_ts[gi] = years[-1] % 100 * 100 + args.month
                continue
            means = prob_stack_utm[:, mask].mean(axis=1)
            best = int(means.argmax())
            year = years[best]
            final_ts[gi] = year % 100 * 100 + args.month

    if (final_ts < 0).any():
        n_unknown = int((final_ts < 0).sum())
        logger.warning(
            "%d polygons did not match any test tile; assigning fallback %d", n_unknown, years[-1]
        )
        final_ts[final_ts < 0] = years[-1] % 100 * 100 + args.month

    out_features = []
    base_records = json.load(open(args.base_geojson))["features"]
    for i, f in enumerate(base_records):
        out_features.append(
            {
                "type": "Feature",
                "properties": {"time_step": int(final_ts[i])},
                "geometry": f["geometry"],
            }
        )
    container = {
        "type": "FeatureCollection",
        "name": "submission",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": out_features,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(container, fh)
    dist: dict[int, int] = {}
    for ts in final_ts.tolist():
        dist[ts] = dist.get(ts, 0) + 1
    logger.info("Wrote %s", args.out)
    logger.info("time_step distribution: %s", dist)


if __name__ == "__main__":
    main()
