"""Build a leaderboard-ready GeoJSON submission from the Baseline 3 XGB ensemble.

This is the production version of the post-processing pipeline behind our
44.98 % submission, with one critical addition: per-polygon ``time_step`` is
filled in by year-of-max-probability so we get credit on the leaderboard's
Year metric instead of scoring 0 there.

Pipeline per tile:
  1. Load AEF embeddings for the baseline year 2020 and target years 2021–2025.
  2. For every target year Y, build the 192-channel feature vector
     ``[AEF[Y], AEF[Y]-AEF[2020], AEF[Y]-AEF[Y-1]]`` and predict the
     deforestation probability with each of the 3 ensemble models.
     The three probabilities are combined with the **geometric mean** to
     match Submission 3 exactly.
  3. Per pixel, pick the year with the maximum probability across 2021–2025
     and threshold it at ``--threshold`` (default 0.50).
  4. Reproject the binary mask to the tile's local UTM CRS at 10 m/pixel
     (matches Sentinel-2) and apply morphological **closing 5×5** then
     **opening 3×3** in UTM space.
  5. Polygonise the post-processed mask, drop polygons smaller than
     ``--min-area-ha`` (default 1.0 ha), and reproject back to EPSG:4326.
  6. For every polygon, look up the dominant year-of-max-probability within
     its footprint and write ``time_step = YY * 100 + month`` (default month
     is June so we use ``2106`` for 2021, ``2206`` for 2022, etc.).
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
from rasterio.features import shapes
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import binary_closing, binary_opening
from shapely.geometry import shape

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Inference

def _load_aef(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def _load_aef_profile(path: Path) -> dict:
    with rasterio.open(path) as src:
        return src.profile.copy()


def _features(aef_y: np.ndarray, aef_2020: np.ndarray, aef_prev: np.ndarray | None) -> np.ndarray:
    """Build (N, 192) feature matrix matching baseline2/3 training."""
    diff_2020 = aef_y - aef_2020
    if aef_prev is None:
        diff_prev = np.zeros_like(aef_y)
    else:
        diff_prev = aef_y - aef_prev
    c, h, w = aef_y.shape
    flat_y = aef_y.reshape(c, h * w).T
    flat_d2020 = diff_2020.reshape(c, h * w).T
    flat_dprev = diff_prev.reshape(c, h * w).T
    features = np.concatenate([flat_y, flat_d2020, flat_dprev], axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _predict_year_geo_mean(
    aef_y: np.ndarray,
    aef_2020: np.ndarray,
    aef_prev: np.ndarray | None,
    bundle: dict,
) -> np.ndarray:
    """Return geometric-mean probability map (H, W) for one year."""
    feats = _features(aef_y, aef_2020, aef_prev)
    h, w = aef_y.shape[1], aef_y.shape[2]
    log_sum = np.zeros(feats.shape[0], dtype=np.float64)
    n = 0
    for _name, mdl in bundle["models"].items():
        proba = mdl.predict_proba(feats)[:, 1]
        proba = np.clip(proba, 1e-6, 1.0 - 1e-6)
        log_sum += np.log(proba)
        n += 1
    geo_mean = np.exp(log_sum / n).astype(np.float32)
    return geo_mean.reshape(h, w)


# ---------------------------------------------------------------------------
# Post-processing

def _morph_in_utm(
    mask: np.ndarray,
    src_profile: dict,
    closing_size: int,
    opening_size: int,
) -> tuple[np.ndarray, dict]:
    """Reproject mask to local UTM at 10 m, run closing+opening, return both."""
    src_crs = src_profile["crs"]
    src_transform = src_profile["transform"]
    h, w = src_profile["height"], src_profile["width"]

    minx, miny, maxx, maxy = rasterio.transform.array_bounds(h, w, src_transform)
    if "+proj=utm" in str(src_crs).lower() or "utm" in str(src_crs).lower():
        utm_crs = src_crs
    else:
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([(minx + maxx) / 2], [(miny + maxy) / 2]),
            crs=src_crs,
        )
        utm_crs = gdf.estimate_utm_crs()

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, utm_crs, w, h, minx, miny, maxx, maxy, resolution=10.0
    )

    mask_utm = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=mask.astype(np.uint8),
        destination=mask_utm,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=utm_crs,
        resampling=Resampling.nearest,
    )

    if closing_size > 0:
        mask_utm = binary_closing(mask_utm, structure=np.ones((closing_size, closing_size))).astype(np.uint8)
    if opening_size > 0:
        mask_utm = binary_opening(mask_utm, structure=np.ones((opening_size, opening_size))).astype(np.uint8)

    utm_profile = {
        "crs": utm_crs,
        "transform": dst_transform,
        "height": dst_h,
        "width": dst_w,
    }
    return mask_utm, utm_profile


def _polygonise(mask_utm: np.ndarray, utm_profile: dict, min_area_ha: float) -> gpd.GeoDataFrame:
    geoms = []
    for geom, value in shapes(
        mask_utm, mask=mask_utm.astype(bool), transform=utm_profile["transform"]
    ):
        if value == 1:
            geoms.append(shape(geom))
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=utm_profile["crs"])
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=utm_profile["crs"])
    gdf = gdf[gdf.area / 10_000.0 >= min_area_ha].reset_index(drop=True)
    return gdf


def _argmax_year_at_polygons(
    polys_utm: gpd.GeoDataFrame,
    argmax_year_native: np.ndarray,
    src_profile: dict,
    utm_profile: dict,
    years: list[int],
) -> list[int]:
    """For every polygon return the dominant year (mode of pixels inside)."""
    if polys_utm.empty:
        return []

    argmax_utm = np.zeros((utm_profile["height"], utm_profile["width"]), dtype=np.int16)
    reproject(
        source=argmax_year_native.astype(np.int16),
        destination=argmax_utm,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=utm_profile["transform"],
        dst_crs=utm_profile["crs"],
        resampling=Resampling.nearest,
    )

    from rasterio.features import rasterize

    out: list[int] = []
    for i, geom in enumerate(polys_utm.geometry, start=1):
        m = rasterize(
            [(geom, 1)],
            out_shape=(utm_profile["height"], utm_profile["width"]),
            transform=utm_profile["transform"],
            fill=0,
            dtype="uint8",
        ).astype(bool)
        if not m.any():
            out.append(years[0])
            continue
        idxs, counts = np.unique(argmax_utm[m], return_counts=True)
        out.append(int(years[int(idxs[counts.argmax()])]))
    return out


# ---------------------------------------------------------------------------
# Main

def build_submission(
    data_dir: Path,
    bundle_path: Path,
    out_path: Path,
    threshold: float,
    min_area_ha: float,
    closing_size: int,
    opening_size: int,
    month: int,
    years: list[int],
) -> dict:
    bundle = joblib.load(bundle_path)
    n_models = len(bundle.get("models", {}))
    if n_models == 0:
        raise RuntimeError("baseline3 bundle has no models")
    logger.info("Loaded baseline3 bundle with %d models", n_models)

    aef_test_dir = data_dir / "aef-embeddings" / "test"
    if not aef_test_dir.is_dir():
        raise FileNotFoundError(f"missing {aef_test_dir}")

    test_tiles = sorted(set(p.stem.rsplit("_", 1)[0] for p in aef_test_dir.glob("*.tiff")))
    logger.info("Test tiles: %s", test_tiles)

    all_geoms: list = []
    all_ts: list[int] = []
    summary: dict = {"per_tile": {}, "config": {
        "threshold": threshold, "min_area_ha": min_area_ha,
        "closing": closing_size, "opening": opening_size,
        "month": month, "years": years, "ensemble": "geometric_mean",
    }}

    for tile_id in test_tiles:
        logger.info("==== %s ====", tile_id)
        aef_2020_path = aef_test_dir / f"{tile_id}_2020.tiff"
        if not aef_2020_path.is_file():
            logger.warning("  no 2020 AEF; skipping")
            continue
        aef_2020 = _load_aef(aef_2020_path)
        ref_profile = _load_aef_profile(aef_2020_path)

        per_year_probs: list[np.ndarray] = []
        for y in years:
            aef_y_path = aef_test_dir / f"{tile_id}_{y}.tiff"
            if not aef_y_path.is_file():
                logger.warning("  missing AEF for %s/%d; using zeros", tile_id, y)
                per_year_probs.append(np.zeros(aef_2020.shape[1:], dtype=np.float32))
                continue
            aef_y = _load_aef(aef_y_path)
            aef_prev_path = aef_test_dir / f"{tile_id}_{y - 1}.tiff"
            aef_prev = _load_aef(aef_prev_path) if aef_prev_path.is_file() else None
            proba = _predict_year_geo_mean(aef_y, aef_2020, aef_prev, bundle)
            per_year_probs.append(proba)
            logger.info("  %d: prob_mean=%.4f prob>thr=%.4f", y, proba.mean(), (proba > threshold).mean())

        prob_stack = np.stack(per_year_probs, axis=0)
        argmax_year = prob_stack.argmax(axis=0).astype(np.int16)
        max_prob = prob_stack.max(axis=0)

        binary_native = (max_prob > threshold).astype(np.uint8)
        if not binary_native.any():
            summary["per_tile"][tile_id] = {"polys": 0, "year_dist": {}}
            continue

        mask_utm, utm_profile = _morph_in_utm(
            binary_native, ref_profile, closing_size, opening_size
        )
        if not mask_utm.any():
            summary["per_tile"][tile_id] = {"polys": 0, "year_dist": {}}
            continue

        polys_utm = _polygonise(mask_utm, utm_profile, min_area_ha)
        poly_years = _argmax_year_at_polygons(polys_utm, argmax_year, ref_profile, utm_profile, years)
        polys_wgs = polys_utm.to_crs("EPSG:4326")

        year_counts: dict[int, int] = {}
        for geom, year in zip(polys_wgs.geometry, poly_years):
            yy = year % 100
            ts = yy * 100 + month
            all_geoms.append(geom)
            all_ts.append(ts)
            year_counts[year] = year_counts.get(year, 0) + 1
        logger.info("  polys=%d  by_year=%s", len(polys_wgs), year_counts)
        summary["per_tile"][tile_id] = {"polys": len(polys_wgs), "year_dist": year_counts}

    if not all_geoms:
        raise RuntimeError("no polygons produced")
    final = gpd.GeoDataFrame({"time_step": all_ts}, geometry=all_geoms, crs="EPSG:4326")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_file(out_path, driver="GeoJSON")
    summary["n_polys_total"] = len(final)
    summary["out_path"] = str(out_path)
    return summary


def _validate_geojson(path: Path) -> None:
    gj = json.load(open(path))
    assert gj["type"] == "FeatureCollection", "not a FeatureCollection"
    feats = gj["features"]
    assert all(f["geometry"]["type"] in ("Polygon", "MultiPolygon") for f in feats), \
        "non-Polygon geometry present"
    bad_ts = []
    for f in feats:
        v = (f.get("properties") or {}).get("time_step")
        if v is None:
            continue
        s = str(v)
        if not (len(s) == 4 and s.isdigit() and 1 <= int(s[2:]) <= 12):
            bad_ts.append(v)
    assert not bad_ts, f"invalid time_step values: {bad_ts[:5]}"
    logger.info("VALIDATED: %d features, all Polygon/MultiPolygon, time_step format OK", len(feats))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("./data/makeathon-challenge"))
    ap.add_argument("--bundle", type=Path, default=Path("./models/baseline_xgb/baseline3_aef_logreg.joblib"))
    ap.add_argument("--out", type=Path, default=Path("./submissions/sub_baseline3_per_year.geojson"))
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min-area-ha", type=float, default=1.0)
    ap.add_argument("--closing", type=int, default=5, help="UTM-space closing kernel size (0 disables)")
    ap.add_argument("--opening", type=int, default=3, help="UTM-space opening kernel size (0 disables)")
    ap.add_argument("--month", type=int, default=6, help="Month component for YYMM time_step (1..12)")
    ap.add_argument("--years", default="2021,2022,2023,2024,2025")
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(",")]
    summary = build_submission(
        args.data_dir, args.bundle, args.out,
        args.threshold, args.min_area_ha,
        args.closing, args.opening,
        args.month, years,
    )
    _validate_geojson(args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_tile"}, indent=2))
    print(json.dumps(summary["per_tile"], indent=2))


if __name__ == "__main__":
    main()
