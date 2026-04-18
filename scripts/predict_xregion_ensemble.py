"""Cross-region ensemble inference for the deforestation challenge.

Loads the two cross-region U-Nets produced by ``src.train_unet_xregion`` and
predicts on the test tiles, aggregating predictions across years and
ensembling the two models' probabilities. Outputs:

- per-tile binary GeoTIFF predictions (in each tile's native CRS)
- a single GeoJSON ``FeatureCollection`` (EPSG:4326) suitable for submission

Ensembling modes:

- ``mean`` (default): average per-pixel sigmoid probabilities.
- ``intersection``: AND of binary predictions at each model's threshold (very
  conservative; tends to maximise precision and Union IoU when both models
  agree).
- ``union``: OR of binary predictions (boost recall).

Test-time augmentation (TTA) averages predictions over 4 rotations.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from shapely.geometry import shape

from src.train_unet import UNetSmall

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


YEAR_RE = re.compile(r"_(\d{4})$")


# ---------------------------------------------------------------------------
# Inference helpers


def _resolve_weight_path(meta_path: Path, stored: str) -> Path:
    """Resolve ``best_model_path`` from JSON: same-dir as meta, then CWD-relative."""
    raw = Path(stored)
    if raw.is_absolute():
        return raw
    colocated = meta_path.parent / raw.name
    if colocated.is_file():
        return colocated
    nested = (meta_path.parent / raw).resolve()
    if nested.is_file():
        return nested
    cwd_p = (Path.cwd() / raw).resolve()
    if cwd_p.is_file():
        return cwd_p
    raise FileNotFoundError(
        f"Weight file not found for meta {meta_path}: {stored!r} "
        f"(tried {colocated}, {nested}, {cwd_p})"
    )


def _load_model(meta_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    meta = json.loads(meta_path.read_text())
    in_channels = int(meta.get("in_channels", 64))
    base_channels = int(meta.get("base_channels", 32))
    model = UNetSmall(in_channels=in_channels, base_channels=base_channels).to(device)
    weight_path = _resolve_weight_path(meta_path, meta["best_model_path"])
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, meta


def _iter_patches(height: int, width: int, patch_size: int, stride: int):
    ys = list(range(0, max(1, height - patch_size + 1), stride))
    xs = list(range(0, max(1, width - patch_size + 1), stride))
    if ys[-1] != height - patch_size:
        ys.append(max(0, height - patch_size))
    if xs[-1] != width - patch_size:
        xs.append(max(0, width - patch_size))
    for y in ys:
        for x in xs:
            yield y, x


def _predict_unet_tile(
    model: torch.nn.Module,
    aef: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    device: torch.device,
    tta: bool,
) -> np.ndarray:
    aef = np.nan_to_num(aef, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    channels, height, width = aef.shape
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if pad_h or pad_w:
        aef = np.pad(aef, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        height += pad_h
        width += pad_w

    accum = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    rotations = [0, 1, 2, 3] if tta else [0]
    with torch.no_grad():
        for k in rotations:
            rotated = np.rot90(aef, k=k, axes=(1, 2))
            rotated = np.ascontiguousarray(rotated)
            rot_accum = np.zeros((height, width), dtype=np.float32)
            rot_counts = np.zeros((height, width), dtype=np.float32)
            for y, x in _iter_patches(height, width, patch_size, stride):
                patch = rotated[:, y : y + patch_size, x : x + patch_size]
                patch_t = torch.from_numpy(patch[None, ...]).to(device)
                logits = model(patch_t)
                prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
                rot_accum[y : y + patch_size, x : x + patch_size] += prob
                rot_counts[y : y + patch_size, x : x + patch_size] += 1.0
            rot_counts[rot_counts == 0] = 1.0
            avg = rot_accum / rot_counts
            # Rotate back to canonical orientation
            avg_back = np.rot90(avg, k=-k, axes=(0, 1))
            accum += avg_back
            counts += 1.0
    counts[counts == 0] = 1.0
    out = accum / counts
    if pad_h or pad_w:
        out = out[: height - pad_h, : width - pad_w]
    return out


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble inference for cross-region U-Nets.")
    parser.add_argument("--data-dir", default="./data/makeathon-challenge")
    parser.add_argument("--meta-a", required=True, help="meta.json from first model")
    parser.add_argument("--meta-b", required=True, help="meta.json from second model")
    parser.add_argument("--out-dir", default="./artifacts/xregion/predictions")
    parser.add_argument(
        "--ensemble-mode",
        choices=("mean", "intersection", "union"),
        default="mean",
        help="How to combine the two models' predictions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Final binarisation threshold for ``mean`` mode. If unset, uses the "
        "average of the two per-model best thresholds from meta.json.",
    )
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--aef-downsample", type=int, default=2)
    parser.add_argument(
        "--years",
        default="2021,2022,2023,2024,2025",
        help="Comma-separated years to aggregate predictions over (max-pool).",
    )
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    parser.add_argument(
        "--submission-out",
        default="./artifacts/xregion/predictions/submission.geojson",
    )
    parser.add_argument(
        "--include-tiles",
        default=None,
        help="Optional comma-separated list of tile_ids to limit inference to.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_out = Path(args.submission_out)
    submission_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, meta_a = _load_model(Path(args.meta_a), device)
    model_b, meta_b = _load_model(Path(args.meta_b), device)
    thr_a = float(meta_a.get("best_threshold", 0.5))
    thr_b = float(meta_b.get("best_threshold", 0.5))
    final_threshold = (
        args.threshold if args.threshold is not None else float(np.mean([thr_a, thr_b]))
    )
    logger.info(
        "Loaded models: A=%s (thr=%.2f, val_iou=%.3f), B=%s (thr=%.2f, val_iou=%.3f)",
        meta_a.get("tag"),
        thr_a,
        meta_a.get("best_val_iou", 0.0),
        meta_b.get("tag"),
        thr_b,
        meta_b.get("best_val_iou", 0.0),
    )
    logger.info("Ensemble=%s, final_threshold=%.3f", args.ensemble_mode, final_threshold)

    test_dir = Path(args.data_dir) / "aef-embeddings" / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"test AEF dir not found: {test_dir}")

    years = [int(y) for y in args.years.split(",") if y.strip()]
    include = None
    if args.include_tiles:
        include = {t.strip() for t in args.include_tiles.split(",") if t.strip()}

    tile_files: dict[str, list[Path]] = {}
    for p in sorted(test_dir.glob("*.tiff")):
        tile_id, year_str = p.stem.rsplit("_", 1)
        year = int(year_str)
        if year not in years:
            continue
        if include is not None and tile_id not in include:
            continue
        tile_files.setdefault(tile_id, []).append(p)

    if not tile_files:
        raise RuntimeError("No test AEF tiles matched the year filter.")

    all_polygons: list = []
    all_crs: list = []

    for tile_id, paths in tile_files.items():
        logger.info("Tile %s: %d years", tile_id, len(paths))
        max_prob: np.ndarray | None = None
        ref_transform = None
        ref_crs = None
        ref_height = None
        ref_width = None

        for path in sorted(paths):
            with rasterio.open(path) as src:
                if args.aef_downsample > 1:
                    out_h = max(1, src.height // args.aef_downsample)
                    out_w = max(1, src.width // args.aef_downsample)
                    aef = src.read(
                        out_shape=(src.count, out_h, out_w),
                        resampling=rasterio.enums.Resampling.nearest,
                    ).astype(np.float32)
                    sx = src.width / out_w
                    sy = src.height / out_h
                    transform = src.transform * src.transform.scale(sx, sy)
                    height = out_h
                    width = out_w
                else:
                    aef = src.read().astype(np.float32)
                    transform = src.transform
                    height = src.height
                    width = src.width
                crs = src.crs

            prob_a = _predict_unet_tile(
                model_a, aef,
                patch_size=args.patch_size, stride=args.stride,
                device=device, tta=not args.no_tta,
            )
            prob_b = _predict_unet_tile(
                model_b, aef,
                patch_size=args.patch_size, stride=args.stride,
                device=device, tta=not args.no_tta,
            )

            if args.ensemble_mode == "mean":
                ensemble = 0.5 * (prob_a + prob_b)
                year_pred = (ensemble >= final_threshold).astype(np.uint8)
            elif args.ensemble_mode == "intersection":
                ensemble = np.minimum(prob_a, prob_b)
                pa = (prob_a >= thr_a).astype(np.uint8)
                pb = (prob_b >= thr_b).astype(np.uint8)
                year_pred = (pa & pb).astype(np.uint8)
            else:  # union
                ensemble = np.maximum(prob_a, prob_b)
                pa = (prob_a >= thr_a).astype(np.uint8)
                pb = (prob_b >= thr_b).astype(np.uint8)
                year_pred = (pa | pb).astype(np.uint8)

            if max_prob is None:
                max_prob = ensemble.astype(np.float32)
                ref_transform = transform
                ref_crs = crs
                ref_height = height
                ref_width = width
                acc_pred = year_pred.astype(np.uint8)
            else:
                if ensemble.shape != max_prob.shape:
                    logger.warning(
                        "Shape mismatch on %s year=%s; skipping (got %s vs %s)",
                        tile_id,
                        path.stem,
                        ensemble.shape,
                        max_prob.shape,
                    )
                    continue
                np.maximum(max_prob, ensemble.astype(np.float32), out=max_prob)
                acc_pred = np.maximum(acc_pred, year_pred)

        if max_prob is None:
            continue

        # final binarisation: max-pool across years, then threshold at final_threshold
        if args.ensemble_mode == "mean":
            tile_binary = (max_prob >= final_threshold).astype(np.uint8)
        else:
            tile_binary = acc_pred

        # Save tile prediction GeoTIFF for debugging / reproducibility.
        tile_out = out_dir / f"{tile_id}_pred.tif"
        profile = {
            "driver": "GTiff",
            "height": ref_height,
            "width": ref_width,
            "count": 1,
            "dtype": "uint8",
            "crs": ref_crs,
            "transform": ref_transform,
            "compress": "lzw",
            "nodata": 0,
        }
        with rasterio.open(tile_out, "w", **profile) as dst:
            dst.write(tile_binary, 1)
        logger.info(
            "Tile %s: %d positive px (%.3f%%), saved %s",
            tile_id,
            int(tile_binary.sum()),
            100.0 * float(tile_binary.sum()) / (ref_height * ref_width),
            tile_out,
        )

        # Vectorise foreground -> WGS84 polygons.
        if tile_binary.sum() == 0:
            continue
        polys = [
            shape(geom)
            for geom, value in shapes(tile_binary, mask=tile_binary, transform=ref_transform)
            if value == 1
        ]
        if not polys:
            continue
        # Reproject polygon CRS
        try:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame(geometry=polys, crs=ref_crs)
            gdf = gdf.to_crs("EPSG:4326")
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            keep = gdf_utm.area / 10_000 >= args.min_area_ha
            gdf = gdf[keep].reset_index(drop=True)
            for geom in gdf.geometry:
                all_polygons.append(geom)
        except Exception as exc:
            logger.warning("Polygonization failed for %s: %s", tile_id, exc)

    if not all_polygons:
        raise RuntimeError("No polygons produced; lower threshold or check models.")

    import geopandas as gpd
    final_gdf = gpd.GeoDataFrame(geometry=all_polygons, crs="EPSG:4326")
    final_gdf["time_step"] = None
    final_gdf.to_file(submission_out, driver="GeoJSON")
    logger.info(
        "Wrote %d polygons across %d tiles to %s",
        len(final_gdf),
        len(tile_files),
        submission_out,
    )


if __name__ == "__main__":
    main()
