"""Build baseline4 feature cache (multimodal per-pixel features)."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from src.data_utils import reproject_to_match, s2_cloud_mask

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

S2_PATTERN = re.compile(r"_s2_l2a_(\d{4})_(\d{1,2})\.tif$")
S1_PATTERN = re.compile(r"_s1_rtc_(\d{4})_(\d{1,2})_(ascending|descending)\.tif$")


def _iter_aef_files(aef_dir: Path) -> Iterable[Path]:
    yield from sorted(aef_dir.glob("*.tiff"))


def _group_aef_by_tile(aef_dir: Path) -> dict[str, dict[int, Path]]:
    tiles: dict[str, dict[int, Path]] = {}
    for path in _iter_aef_files(aef_dir):
        tile_id, year_str = path.stem.rsplit("_", 1)
        try:
            year = int(year_str)
        except ValueError:
            continue
        tiles.setdefault(tile_id, {})[year] = path
    return tiles


def _collect_feature_years(data_dir: Path, min_year: int) -> list[int]:
    years: set[int] = set()
    for split in ["train", "test"]:
        aef_dir = data_dir / "aef-embeddings" / split
        if not aef_dir.exists():
            continue
        for path in _iter_aef_files(aef_dir):
            try:
                year = int(path.stem.rsplit("_", 1)[1])
            except ValueError:
                continue
            if year >= min_year:
                years.add(year)
    return sorted(years)


def _parse_s2_date(path: Path) -> tuple[int, int] | None:
    match = S2_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_s1_date(path: Path) -> tuple[int, int, str] | None:
    match = S1_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), match.group(3)


def _reproject_array(
    src_array: np.ndarray,
    src_transform,
    src_crs,
    ref_profile: dict,
    resampling: Resampling,
) -> np.ndarray:
    if src_array.ndim == 2:
        dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        reproject(
            source=src_array,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=resampling,
        )
        return dst

    if src_array.ndim == 3:
        bands = src_array.shape[0]
        dst = np.zeros((bands, ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        for band_idx in range(bands):
            reproject(
                source=src_array[band_idx],
                destination=dst[band_idx],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                resampling=resampling,
            )
        return dst

    raise ValueError(f"Unsupported array dimensions for reprojection: {src_array.ndim}")


def _compute_index(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denom = numerator + denominator
    return np.where(denom != 0, (numerator - denominator) / denom, np.nan).astype(np.float32)


def _safe_nan_stat(fn, arr: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice")
        warnings.filterwarnings("ignore", message="All-NaN axis encountered")
        return fn(arr, axis=axis).astype(np.float32)


def _compute_s2_year_stats(s2_files: list[Path], ref_profile: dict) -> dict[str, np.ndarray] | None:
    if not s2_files:
        return None

    ndvi_list: list[np.ndarray] = []
    nbr_list: list[np.ndarray] = []
    ndmi_list: list[np.ndarray] = []

    for path in s2_files:
        with rasterio.open(path) as src:
            s2_stack = src.read().astype(np.float32)
            cloud_mask = s2_cloud_mask(s2_stack)

            red = s2_stack[3]
            nir = s2_stack[7]
            swir1 = s2_stack[10]
            swir2 = s2_stack[11]

            ndvi = _compute_index(nir, red)
            nbr = _compute_index(nir, swir2)
            ndmi = _compute_index(nir, swir1)

            ndvi[cloud_mask] = np.nan
            nbr[cloud_mask] = np.nan
            ndmi[cloud_mask] = np.nan

            if src.crs != ref_profile["crs"] or src.transform != ref_profile["transform"]:
                ndvi = _reproject_array(ndvi, src.transform, src.crs, ref_profile, Resampling.nearest)
                nbr = _reproject_array(nbr, src.transform, src.crs, ref_profile, Resampling.nearest)
                ndmi = _reproject_array(ndmi, src.transform, src.crs, ref_profile, Resampling.nearest)

        ndvi_list.append(ndvi)
        nbr_list.append(nbr)
        ndmi_list.append(ndmi)

    ndvi_stack = np.stack(ndvi_list, axis=0)
    nbr_stack = np.stack(nbr_list, axis=0)
    ndmi_stack = np.stack(ndmi_list, axis=0)

    return {
        "ndvi_median": _safe_nan_stat(np.nanmedian, ndvi_stack, axis=0),
        "ndvi_min": _safe_nan_stat(np.nanmin, ndvi_stack, axis=0),
        "ndvi_std": _safe_nan_stat(np.nanstd, ndvi_stack, axis=0),
        "nbr_median": _safe_nan_stat(np.nanmedian, nbr_stack, axis=0),
        "nbr_min": _safe_nan_stat(np.nanmin, nbr_stack, axis=0),
        "nbr_std": _safe_nan_stat(np.nanstd, nbr_stack, axis=0),
        "ndmi_median": _safe_nan_stat(np.nanmedian, ndmi_stack, axis=0),
        "ndmi_min": _safe_nan_stat(np.nanmin, ndmi_stack, axis=0),
        "ndmi_std": _safe_nan_stat(np.nanstd, ndmi_stack, axis=0),
    }


def _compute_s1_year_stats(s1_files: list[Path], ref_profile: dict) -> dict[str, dict[str, np.ndarray]]:
    per_orbit: dict[str, list[np.ndarray]] = {"ascending": [], "descending": []}

    for path in s1_files:
        parsed = _parse_s1_date(path)
        if parsed is None:
            continue
        _, _, orbit = parsed

        with rasterio.open(path) as src:
            backscatter = src.read(1).astype(np.float32)
            db = np.where(backscatter > 0, 10 * np.log10(backscatter), np.nan)
            if src.crs != ref_profile["crs"] or src.transform != ref_profile["transform"]:
                db = _reproject_array(db, src.transform, src.crs, ref_profile, Resampling.nearest)
        per_orbit[orbit].append(db)

    stats: dict[str, dict[str, np.ndarray]] = {}
    for orbit, stacks in per_orbit.items():
        if not stacks:
            continue
        arr = np.stack(stacks, axis=0)
        stats[orbit] = {
            "mean": _safe_nan_stat(np.nanmean, arr, axis=0),
            "std": _safe_nan_stat(np.nanstd, arr, axis=0),
            "p10": _safe_nan_stat(lambda a, axis: np.nanpercentile(a, 10, axis=axis), arr, axis=0),
        }
    return stats


def _load_consensus_labels(
    data_dir: Path,
    tile_id: str,
    ref_profile: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
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
        glads2_pos = (glads2_alert_r >= 2) & (glads2_date >= np.datetime64("2020-01-01"))
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
    gladl_pos: np.ndarray | None = None
    gladl_neg: np.ndarray | None = None

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
        year_neg = alert_r == 0

        if gladl_pos is None:
            gladl_pos = year_pos
            gladl_neg = year_neg
        else:
            gladl_pos = gladl_pos | year_pos
            gladl_neg = gladl_neg & year_neg

    if gladl_pos is not None and gladl_neg is not None:
        pos_sources.append(gladl_pos)
        neg_sources.append(gladl_neg)

    if not pos_sources:
        return None

    pos_stack = np.stack(pos_sources, axis=0)
    neg_stack = np.stack(neg_sources, axis=0)

    pos_votes = pos_stack.sum(axis=0)
    neg_votes = neg_stack.sum(axis=0)
    n_available = pos_votes + neg_votes
    threshold = (n_available // 2) + 1

    positive = (pos_votes >= threshold) & (neg_votes < threshold)
    negative = (neg_votes >= threshold) & (pos_votes < threshold)

    labels = np.full(pos_votes.shape, -1, dtype=np.int8)
    labels[positive] = 1
    labels[negative] = 0

    weights = np.zeros(pos_votes.shape, dtype=np.float32)
    agree = np.maximum(pos_votes, neg_votes)
    valid = n_available > 0
    weights[valid] = agree[valid] / n_available[valid]
    weights[labels == -1] = 0.0

    return labels, weights


def _build_feature_names(feature_years: list[int]) -> list[str]:
    names: list[str] = []

    for band in range(64):
        names.append(f"aef_2020_b{band:02d}")

    for year in feature_years:
        if year == 2020:
            continue
        for band in range(64):
            names.append(f"aef_diff_2020_{year}_b{band:02d}")
        for band in range(64):
            names.append(f"aef_diff_prev_{year}_b{band:02d}")

    for orbit in ["ascending", "descending"]:
        names.append(f"s1_{orbit}_mean_2020")
        names.append(f"s1_{orbit}_std_2020")
        names.append(f"s1_{orbit}_p10_2020")

    for year in feature_years:
        if year == 2020:
            continue
        for orbit in ["ascending", "descending"]:
            names.append(f"s1_{orbit}_mean_diff_2020_{year}")
            names.append(f"s1_{orbit}_std_diff_2020_{year}")
            names.append(f"s1_{orbit}_p10_diff_2020_{year}")

    s2_indices = ["ndvi", "nbr", "ndmi"]
    s2_stats = ["median", "min", "std"]

    for idx in s2_indices:
        for stat in s2_stats:
            names.append(f"s2_{idx}_{stat}_2020")

    for year in feature_years:
        if year == 2020:
            continue
        for idx in s2_indices:
            for stat in s2_stats:
                names.append(f"s2_{idx}_{stat}_diff_2020_{year}")

    return names


def _ensure_feature_spec(cache_dir: Path, feature_years: list[int]) -> dict:
    spec_path = cache_dir / "feature_spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        if spec.get("feature_years") != feature_years:
            raise ValueError("Feature years mismatch with existing cache spec.")
        return spec

    feature_names = _build_feature_names(feature_years)
    spec = {
        "feature_years": feature_years,
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "aef_bands": 64,
        "s1_orbits": ["ascending", "descending"],
        "s1_stats": ["mean", "std", "p10"],
        "s2_indices": ["ndvi", "nbr", "ndmi"],
        "s2_stats": ["median", "min", "std"],
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2))
    return spec


def _build_tile_cache(
    data_dir: Path,
    split: str,
    tile_id: str,
    years: list[int],
    aef_paths: dict[int, Path],
    out_dir: Path,
    feature_dim: int,
    force: bool,
) -> None:
    out_path = out_dir / f"{tile_id}.npz"
    if out_path.exists() and not force:
        logger.info("Cache exists for %s (%s), skipping", tile_id, split)
        return

    s2_root = data_dir / "sentinel-2" / split
    s1_root = data_dir / "sentinel-1" / split

    s2_files = sorted((s2_root / f"{tile_id}__s2_l2a").glob("*.tif"))
    s2_2020 = [p for p in s2_files if _parse_s2_date(p) and _parse_s2_date(p)[0] == 2020]
    if not s2_2020:
        logger.warning("Skipping %s (%s): missing 2020 Sentinel-2", tile_id, split)
        return

    with rasterio.open(s2_2020[0]) as src:
        ref_profile = src.profile

    if 2020 not in aef_paths:
        logger.warning("Skipping %s (%s): missing 2020 AEF", tile_id, split)
        return

    with rasterio.open(aef_paths[2020]) as src:
        aef_2020 = src.read().astype(np.float32)
        aef_2020 = _reproject_array(
            aef_2020,
            src.transform,
            src.crs,
            ref_profile,
            Resampling.bilinear,
        )

    features: list[np.ndarray] = [band for band in aef_2020]

    aef_prev: np.ndarray | None = None
    prev_year: int | None = None

    for year in years:
        if year == 2020:
            aef_prev = aef_2020
            prev_year = 2020
            continue

        path = aef_paths.get(year)
        if path is None:
            nan = np.full_like(aef_2020, np.nan, dtype=np.float32)
            features.extend([band for band in nan])
            features.extend([band for band in nan])
            continue

        with rasterio.open(path) as src:
            aef_year = src.read().astype(np.float32)
            aef_year = _reproject_array(
                aef_year,
                src.transform,
                src.crs,
                ref_profile,
                Resampling.bilinear,
            )

        diff_2020 = aef_year - aef_2020
        if prev_year == year - 1 and aef_prev is not None:
            diff_prev = aef_year - aef_prev
        else:
            diff_prev = np.full_like(aef_2020, np.nan, dtype=np.float32)

        features.extend([band for band in diff_2020])
        features.extend([band for band in diff_prev])
        aef_prev = aef_year
        prev_year = year

    s1_files = sorted((s1_root / f"{tile_id}__s1_rtc").glob(f"{tile_id}__s1_rtc_*.tif"))
    s1_by_year: dict[int, list[Path]] = {}
    for path in s1_files:
        parsed = _parse_s1_date(path)
        if parsed is None:
            continue
        year, _, _ = parsed
        s1_by_year.setdefault(year, []).append(path)

    s1_stats_by_year: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for year in years:
        if year not in s1_by_year:
            continue
        s1_stats_by_year[year] = _compute_s1_year_stats(s1_by_year[year], ref_profile)

    if 2020 not in s1_stats_by_year or not all(
        orbit in s1_stats_by_year[2020] for orbit in ["ascending", "descending"]
    ):
        logger.warning("Skipping %s (%s): missing 2020 Sentinel-1", tile_id, split)
        return

    s1_2020 = s1_stats_by_year[2020]
    for orbit in ["ascending", "descending"]:
        features.append(s1_2020[orbit]["mean"])
        features.append(s1_2020[orbit]["std"])
        features.append(s1_2020[orbit]["p10"])

    for year in years:
        if year == 2020:
            continue
        stats = s1_stats_by_year.get(year)
        for orbit in ["ascending", "descending"]:
            if stats is None or orbit not in stats:
                nan = np.full_like(aef_2020[0], np.nan, dtype=np.float32)
                features.extend([nan, nan, nan])
            else:
                features.append(stats[orbit]["mean"] - s1_2020[orbit]["mean"])
                features.append(stats[orbit]["std"] - s1_2020[orbit]["std"])
                features.append(stats[orbit]["p10"] - s1_2020[orbit]["p10"])

    s2_by_year: dict[int, list[Path]] = {}
    for path in s2_files:
        parsed = _parse_s2_date(path)
        if parsed is None:
            continue
        year, _ = parsed
        s2_by_year.setdefault(year, []).append(path)

    s2_stats_by_year: dict[int, dict[str, np.ndarray]] = {}
    for year in years:
        stats = _compute_s2_year_stats(s2_by_year.get(year, []), ref_profile)
        if stats is not None:
            s2_stats_by_year[year] = stats

    if 2020 not in s2_stats_by_year:
        logger.warning("Skipping %s (%s): missing 2020 Sentinel-2", tile_id, split)
        return

    s2_2020 = s2_stats_by_year[2020]
    for key in [
        "ndvi_median",
        "ndvi_min",
        "ndvi_std",
        "nbr_median",
        "nbr_min",
        "nbr_std",
        "ndmi_median",
        "ndmi_min",
        "ndmi_std",
    ]:
        features.append(s2_2020[key])

    for year in years:
        if year == 2020:
            continue
        stats = s2_stats_by_year.get(year)
        for key in [
            "ndvi_median",
            "ndvi_min",
            "ndvi_std",
            "nbr_median",
            "nbr_min",
            "nbr_std",
            "ndmi_median",
            "ndmi_min",
            "ndmi_std",
        ]:
            if stats is None:
                nan = np.full_like(aef_2020[0], np.nan, dtype=np.float32)
                features.append(nan)
            else:
                features.append(stats[key] - s2_2020[key])

    feature_stack = np.stack(features, axis=0).astype(np.float32)
    if feature_stack.shape[0] != feature_dim:
        raise RuntimeError(
            f"Feature dim mismatch for {tile_id}: {feature_stack.shape[0]} != {feature_dim}"
        )

    ndvi_future = [s2_stats_by_year[y]["ndvi_median"] for y in years if y != 2020 and y in s2_stats_by_year]
    if ndvi_future:
        ndvi_min_future = _safe_nan_stat(np.nanmin, np.stack(ndvi_future, axis=0), axis=0)
        ndvi_drop = s2_2020["ndvi_median"] - ndvi_min_future
    else:
        ndvi_drop = np.full_like(aef_2020[0], np.nan, dtype=np.float32)

    vv_2020 = 0.5 * (s1_2020["ascending"]["mean"] + s1_2020["descending"]["mean"])
    vv_future = []
    for year in years:
        if year == 2020:
            continue
        stats = s1_stats_by_year.get(year)
        if stats is None or "ascending" not in stats or "descending" not in stats:
            continue
        vv_future.append(0.5 * (stats["ascending"]["mean"] + stats["descending"]["mean"]))

    if vv_future:
        vv_min_future = _safe_nan_stat(np.nanmin, np.stack(vv_future, axis=0), axis=0)
        vv_drop = vv_2020 - vv_min_future
    else:
        vv_drop = np.full_like(aef_2020[0], np.nan, dtype=np.float32)

    forest_mask = np.ones_like(aef_2020[0], dtype=np.uint8)

    if split == "train":
        labels = _load_consensus_labels(data_dir, tile_id, ref_profile)
        if labels is None:
            logger.warning("Skipping %s (%s): missing labels", tile_id, split)
            return
        label_mask, weight = labels
    else:
        label_mask = np.full_like(aef_2020[0], -1, dtype=np.int8)
        weight = np.zeros_like(aef_2020[0], dtype=np.float32)

    transform_gdal = np.array(ref_profile["transform"].to_gdal(), dtype=np.float64)
    crs_wkt = str(ref_profile["crs"]) if ref_profile["crs"] is not None else ""

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=feature_stack,
        label=label_mask.astype(np.int8),
        weight=weight.astype(np.float32),
        ndvi_drop=ndvi_drop.astype(np.float32),
        vv_drop=vv_drop.astype(np.float32),
        forest_mask=forest_mask,
        transform=transform_gdal,
        crs=crs_wkt,
    )
    logger.info("Wrote cache for %s (%s)", tile_id, split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline4 feature cache.")
    parser.add_argument("--data-dir", default="./data/makeathon-challenge")
    parser.add_argument("--cache-dir", default="./data/makeathon-challenge-cache/baseline4")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--feature-years", default="")
    parser.add_argument("--min-year", type=int, default=2020)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (use 1 to disable).",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    if args.feature_years:
        feature_years = [int(y.strip()) for y in args.feature_years.split(",") if y.strip()]
        feature_years = sorted({y for y in feature_years if y >= args.min_year})
    else:
        feature_years = _collect_feature_years(data_dir, args.min_year)

    if not feature_years or 2020 not in feature_years:
        raise RuntimeError("Feature years must include 2020.")

    spec = _ensure_feature_spec(cache_dir, feature_years)

    splits = [args.split] if args.split != "both" else ["train", "test"]
    for split in splits:
        aef_dir = data_dir / "aef-embeddings" / split
        if not aef_dir.exists():
            logger.warning("Missing AEF dir for split=%s", split)
            continue

        tiles = _group_aef_by_tile(aef_dir)
        tile_items = sorted(tiles.items())
        if args.max_tiles:
            tile_items = tile_items[: args.max_tiles]
        out_dir = cache_dir / split
        total_tiles = len(tile_items)

        logger.info("Building cache for split=%s (%d tiles)", split, total_tiles)

        if args.num_workers <= 1:
            for idx, (tile_id, year_map) in enumerate(tile_items, start=1):
                if total_tiles > 0:
                    pct = (idx / total_tiles) * 100
                    logger.info("Processing %s (%d/%d, %.1f%%)", tile_id, idx, total_tiles, pct)

                _build_tile_cache(
                    data_dir=data_dir,
                    split=split,
                    tile_id=tile_id,
                    years=feature_years,
                    aef_paths=year_map,
                    out_dir=out_dir,
                    feature_dim=spec["feature_dim"],
                    force=args.force,
                )
        else:
            logger.info("Using %d workers for split=%s", args.num_workers, split)
            futures = {}
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                for tile_id, year_map in tile_items:
                    future = executor.submit(
                        _build_tile_cache,
                        data_dir,
                        split,
                        tile_id,
                        feature_years,
                        year_map,
                        out_dir,
                        spec["feature_dim"],
                        args.force,
                    )
                    futures[future] = tile_id

                for idx, future in enumerate(as_completed(futures), start=1):
                    tile_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.warning("Tile %s failed: %s", tile_id, exc)

                    if total_tiles > 0:
                        pct = (idx / total_tiles) * 100
                        logger.info("Completed %s (%d/%d, %.1f%%)", tile_id, idx, total_tiles, pct)


if __name__ == "__main__":
    main()
