"""Colab-friendly inference script for multiple models.

Supports:
- aef_xgb: pixel-wise XGBoost on AEF embeddings
- temporal_xgb: pixel-wise XGBoost on AEF + NDVI/S1 temporal features
- patch_xgb: patch-wise XGBoost on AEF embeddings (upsampled to full mask)
- unet: patch-wise U-Net on AEF embeddings (upsampled to full mask)
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
import matplotlib.pyplot as plt
import geopandas as gpd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

S2_PATTERN = re.compile(r"_s2_l2a_(\d{4})_(\d{1,2})\.tif$")
S1_PATTERN = re.compile(r"_s1_rtc_(\d{4})_(\d{1,2})_.*\.tif$")


def _reproject_array(src_array: np.ndarray, src_transform, src_crs, ref_profile: dict) -> np.ndarray:
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


def _resolve_tile_dir(root: Path, tile_id: str, suffix: str) -> Path | None:
    for split in ["train", "test"]:
        candidate = root / split / f"{tile_id}__{suffix}"
        if candidate.exists():
            return candidate
    return None


def _iter_s2_files(tile_id: str, root: Path) -> list[Path]:
    tile_dir = _resolve_tile_dir(root, tile_id, "s2_l2a")
    if tile_dir is None:
        return []
    return sorted(tile_dir.glob("*.tif"))


def _iter_s1_files(tile_id: str, root: Path) -> list[Path]:
    tile_dir = _resolve_tile_dir(root, tile_id, "s1_rtc")
    if tile_dir is None:
        return []
    return sorted(tile_dir.glob(f"{tile_id}__s1_rtc_*.tif"))


def _parse_s2_date(path: Path) -> tuple[int, int] | None:
    match = S2_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_s1_date(path: Path) -> tuple[int, int] | None:
    match = S1_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    denom = nir + red
    return np.where(denom != 0, (nir - red) / denom, 0.0).astype(np.float32)


def _load_ndvi_stack(tile_id: str, data_dir: Path, ref_profile: dict) -> list[np.ndarray]:
    s2_root = data_dir / "sentinel-2"
    ndvi_items: list[tuple[tuple[int, int], np.ndarray]] = []

    for path in _iter_s2_files(tile_id, s2_root):
        date = _parse_s2_date(path)
        if date is None:
            continue

        with rasterio.open(path) as src:
            red = src.read(4).astype(np.float32)
            nir = src.read(8).astype(np.float32)
            ndvi = _compute_ndvi(nir, red)
            ndvi = _reproject_array(ndvi, src.transform, src.crs, ref_profile)

        ndvi_items.append((date, ndvi))

    ndvi_items.sort(key=lambda x: x[0])
    return [item[1] for item in ndvi_items]


def _load_s1_stack(tile_id: str, data_dir: Path, ref_profile: dict) -> list[np.ndarray]:
    s1_root = data_dir / "sentinel-1"
    s1_items: list[tuple[tuple[int, int], np.ndarray]] = []

    for path in _iter_s1_files(tile_id, s1_root):
        date = _parse_s1_date(path)
        if date is None:
            continue

        with rasterio.open(path) as src:
            backscatter = src.read(1).astype(np.float32)
            db = np.where(backscatter > 0, 10 * np.log10(backscatter), np.nan)
            db = np.nan_to_num(db, nan=0.0).astype(np.float32)
            db = _reproject_array(db, src.transform, src.crs, ref_profile)

        s1_items.append((date, db))

    s1_items.sort(key=lambda x: x[0])
    return [item[1] for item in s1_items]


def _temporal_features(ndvi_stack: list[np.ndarray], s1_stack: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    if not ndvi_stack:
        raise RuntimeError("No Sentinel-2 NDVI stack available.")

    ndvi_arr = np.stack(ndvi_stack, axis=0)
    ndvi_delta = ndvi_arr[-1] - ndvi_arr[0]
    ndvi_var = np.var(ndvi_arr, axis=0)

    if ndvi_arr.shape[0] > 1:
        drops = ndvi_arr[:-1] - ndvi_arr[1:]
        ndvi_max_drop = np.max(drops, axis=0)
    else:
        ndvi_max_drop = np.zeros_like(ndvi_arr[0])

    if not s1_stack:
        s1_change = np.zeros_like(ndvi_arr[0])
    else:
        s1_arr = np.stack(s1_stack, axis=0)
        s1_change = np.max(s1_arr, axis=0) - np.min(s1_arr, axis=0)

    return ndvi_delta, ndvi_max_drop, s1_change, ndvi_var


def _iter_patch_coords(height: int, width: int, patch_size: int, stride: int) -> Iterable[tuple[int, int]]:
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            yield y, x


def _patch_features(patch: np.ndarray) -> np.ndarray:
    mean = patch.mean(axis=(1, 2))
    std = patch.std(axis=(1, 2))
    return np.concatenate([mean, std], axis=0)


def _make_rgb(aef: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(42)
    chosen = sorted(rng.choice(aef.shape[0], size=3, replace=False).tolist())
    bands = aef[chosen].astype(np.float32)

    def _norm(band: np.ndarray) -> np.ndarray:
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            return np.zeros_like(band)
        lo, hi = np.percentile(valid, [2, 98])
        return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

    rgb = np.stack([_norm(bands[i]) for i in range(3)], axis=-1)
    return rgb


def _find_tile_polygon(data_dir: Path, tile_id: str):
    for name in ["train_tiles.geojson", "test_tiles.geojson"]:
        path = data_dir / "metadata" / name
        if not path.exists():
            continue
        gdf = gpd.read_file(path)
        match = gdf[gdf["name"] == tile_id]
        if not match.empty:
            return match.iloc[0].geometry
    raise FileNotFoundError(f"Tile {tile_id} not found in metadata.")


def _predict_pixel_xgb(aef: np.ndarray, model) -> np.ndarray:
    channels, height, width = aef.shape
    flat = aef.reshape(channels, height * width).transpose(1, 0)
    proba = model.predict_proba(flat)[:, 1]
    return proba.reshape(height, width)


def _predict_temporal_xgb(aef: np.ndarray, tile_id: str, data_dir: Path, ref_profile: dict, model) -> np.ndarray:
    ndvi_stack = _load_ndvi_stack(tile_id, data_dir, ref_profile)
    s1_stack = _load_s1_stack(tile_id, data_dir, ref_profile)
    ndvi_delta, ndvi_max_drop, s1_change, ndvi_var = _temporal_features(ndvi_stack, s1_stack)

    channels, height, width = aef.shape
    flat = aef.reshape(channels, height * width).transpose(1, 0)

    temporal = np.stack([ndvi_delta, ndvi_max_drop, s1_change, ndvi_var], axis=0)
    temporal = temporal.reshape(4, height * width).transpose(1, 0)

    features = np.concatenate([flat, temporal], axis=1)
    proba = model.predict_proba(features)[:, 1]
    return proba.reshape(height, width)


def _predict_patch_xgb(aef: np.ndarray, model, patch_size: int, stride: int) -> np.ndarray:
    channels, height, width = aef.shape
    out = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for y, x in _iter_patch_coords(height, width, patch_size, stride):
        patch = aef[:, y : y + patch_size, x : x + patch_size]
        feat = _patch_features(patch)[None, :]
        proba = float(model.predict_proba(feat)[0, 1])
        out[y : y + patch_size, x : x + patch_size] += proba
        counts[y : y + patch_size, x : x + patch_size] += 1.0

    counts[counts == 0] = 1.0
    return out / counts


def _predict_unet(aef: np.ndarray, model_path: Path, patch_size: int, stride: int) -> np.ndarray:
    import torch
    from src.train_unet import UNetSmall  # local import

    channels, height, width = aef.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetSmall(in_channels=channels).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for y, x in _iter_patch_coords(height, width, patch_size, stride):
            patch = aef[:, y : y + patch_size, x : x + patch_size]
            patch = torch.from_numpy(patch[None, ...]).to(device)
            logits = model(patch)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
            out[y : y + patch_size, x : x + patch_size] += prob
            counts[y : y + patch_size, x : x + patch_size] += 1.0

    counts[counts == 0] = 1.0
    return out / counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab inference for multiple models.")
    parser.add_argument("--data-dir", default="/content/data/makeathon-challenge")
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument(
        "--model-type",
        choices=["aef_xgb", "temporal_xgb", "patch_xgb", "unet"],
        required=True,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", default="./outputs")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aef_path = data_dir / "aef-embeddings" / "train" / f"{args.tile_id}_{args.year}.tiff"
    if not aef_path.exists():
        aef_path = data_dir / "aef-embeddings" / "test" / f"{args.tile_id}_{args.year}.tiff"

    if not aef_path.exists():
        raise FileNotFoundError(f"AEF file not found for {args.tile_id} {args.year}")

    with rasterio.open(aef_path) as src:
        aef = src.read().astype(np.float32)
        ref_profile = src.profile
        transform = src.transform

    tile_geom = _find_tile_polygon(data_dir, args.tile_id)
    height, width = aef.shape[1], aef.shape[2]
    polygon_mask = rasterize(
        [(tile_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    if args.model_type in {"aef_xgb", "temporal_xgb", "patch_xgb"}:
        model = joblib.load(args.model_path)
        if args.model_type == "aef_xgb":
            proba = _predict_pixel_xgb(aef, model)
        elif args.model_type == "temporal_xgb":
            proba = _predict_temporal_xgb(aef, args.tile_id, data_dir, ref_profile, model)
        else:
            proba = _predict_patch_xgb(aef, model, args.patch_size, args.stride)
    else:
        proba = _predict_unet(aef, Path(args.model_path), args.patch_size, args.stride)

    pred_mask = (proba > args.threshold).astype(np.uint8)

    rgb = _make_rgb(aef)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("AEF RGB")
    axes[0].axis("off")

    axes[1].imshow(polygon_mask, cmap="gray")
    axes[1].set_title("Tile Polygon Mask")
    axes[1].axis("off")

    axes[2].imshow(rgb)
    axes[2].imshow(pred_mask, cmap="Reds", alpha=0.45)
    axes[2].set_title("Prediction Mask Overlay")
    axes[2].axis("off")

    fig.tight_layout()
    out_path = out_dir / f"{args.tile_id}_{args.model_type}_overlay.png"
    fig.savefig(out_path, dpi=200)
    logger.info("Saved overlay to %s", out_path)


if __name__ == "__main__":
    main()
