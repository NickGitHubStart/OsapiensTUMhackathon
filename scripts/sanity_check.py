"""Run sanity checks on training tiles and save metrics + overlays + polygons."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import joblib
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data_utils import (
    build_label_mask,
    iter_aef_files,
    label_tile_ids,
    load_tile_labels,
    postprocess_prediction,
    reproject_array,
    s2_cloud_mask,
)

try:
    sys.path.insert(0, str(repo_root / "ONI-makeathon-challenge-2026-main"))
    from submission_utils import raster_to_geojson
except Exception as exc:  # pragma: no cover
    raise RuntimeError("submission_utils.py not found") from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = np.logical_and(y_true == 1, y_pred == 1).sum()
    union = np.logical_or(y_true == 1, y_pred == 1).sum()
    if union == 0:
        return float("nan")
    return float(intersection / union)


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

    return np.stack([_norm(bands[i]) for i in range(3)], axis=-1)


def _iter_s2_files(tile_id: str, s2_dir: Path) -> list[Path]:
    return sorted((s2_dir / f"{tile_id}__s2_l2a").glob("*.tif"))


def _iter_s1_files(tile_id: str, s1_dir: Path) -> list[Path]:
    return sorted((s1_dir / f"{tile_id}__s1_rtc").glob(f"{tile_id}__s1_rtc_*.tif"))


def _compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    denom = nir + red
    return np.where(denom != 0, (nir - red) / denom, 0.0).astype(np.float32)


def _load_ndvi_stack(tile_id: str, data_dir: Path, ref_profile: dict) -> list[np.ndarray]:
    s2_dir = data_dir / "sentinel-2" / "train"
    ndvi_items: list[tuple[tuple[int, int], np.ndarray]] = []

    for path in _iter_s2_files(tile_id, s2_dir):
        name = path.name
        parts = name.split("_")
        if len(parts) < 2:
            continue
        try:
            year = int(parts[-2])
            month = int(parts[-1].replace(".tif", ""))
        except ValueError:
            continue

        with rasterio.open(path) as src:
            s2_stack = src.read().astype(np.float32)
            cloud_mask = s2_cloud_mask(s2_stack)
            red = s2_stack[3]
            nir = s2_stack[7]
            ndvi = _compute_ndvi(nir, red)
            ndvi[cloud_mask] = np.nan
            ndvi = reproject_array(ndvi, src.transform, src.crs, ref_profile)
        ndvi_items.append(((year, month), ndvi))

    ndvi_items.sort(key=lambda x: x[0])
    return [item[1] for item in ndvi_items]


def _load_s1_stack(tile_id: str, data_dir: Path, ref_profile: dict) -> list[np.ndarray]:
    s1_dir = data_dir / "sentinel-1" / "train"
    s1_items: list[tuple[tuple[int, int], np.ndarray]] = []

    for path in _iter_s1_files(tile_id, s1_dir):
        name = path.name
        parts = name.split("_")
        if len(parts) < 3:
            continue
        try:
            year = int(parts[-3])
            month = int(parts[-2])
        except ValueError:
            continue

        with rasterio.open(path) as src:
            backscatter = src.read(1).astype(np.float32)
            db = np.where(backscatter > 0, 10 * np.log10(backscatter), np.nan)
            db = np.nan_to_num(db, nan=0.0).astype(np.float32)
            db = reproject_array(db, src.transform, src.crs, ref_profile)
        s1_items.append(((year, month), db))

    s1_items.sort(key=lambda x: x[0])
    return [item[1] for item in s1_items]


def _temporal_features(ndvi_stack: list[np.ndarray], s1_stack: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    if not ndvi_stack:
        raise RuntimeError("No Sentinel-2 NDVI stack available.")

    ndvi_arr = np.stack(ndvi_stack, axis=0)
    ndvi_delta = ndvi_arr[-1] - ndvi_arr[0]
    ndvi_var = np.nanvar(ndvi_arr, axis=0)

    if ndvi_arr.shape[0] > 1:
        drops = ndvi_arr[:-1] - ndvi_arr[1:]
        ndvi_max_drop = np.nanmax(drops, axis=0)
        all_nan = np.all(np.isnan(drops), axis=0)
        ndvi_max_drop[all_nan] = np.nan
    else:
        ndvi_max_drop = np.full_like(ndvi_arr[0], np.nan)

    if not s1_stack:
        s1_change = np.zeros_like(ndvi_arr[0])
    else:
        s1_arr = np.stack(s1_stack, axis=0)
        s1_change = np.nanmax(s1_arr, axis=0) - np.nanmin(s1_arr, axis=0)

    return ndvi_delta, ndvi_max_drop, s1_change, ndvi_var


def _predict_pixel_xgb(aef: np.ndarray, model) -> np.ndarray:
    channels, height, width = aef.shape
    flat = aef.reshape(channels, height * width).transpose(1, 0)
    proba = model.predict_proba(flat)[:, 1]
    return proba.reshape(height, width)


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

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = aef[:, y : y + patch_size, x : x + patch_size]
            mean = patch.mean(axis=(1, 2))
            std = patch.std(axis=(1, 2))
            feat = np.concatenate([mean, std], axis=0)[None, :]
            proba = float(model.predict_proba(feat)[0, 1])
            out[y : y + patch_size, x : x + patch_size] += proba
            counts[y : y + patch_size, x : x + patch_size] += 1.0

    counts[counts == 0] = 1.0
    return out / counts


def _predict_unet(aef: np.ndarray, model_path: Path, patch_size: int, stride: int) -> np.ndarray:
    import torch
    from src.train_unet import UNetSmall

    channels, height, width = aef.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetSmall(in_channels=channels).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = aef[:, y : y + patch_size, x : x + patch_size]
                patch = torch.from_numpy(patch[None, ...]).to(device)
                logits = model(patch)
                prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
                out[y : y + patch_size, x : x + patch_size] += prob
                counts[y : y + patch_size, x : x + patch_size] += 1.0

    counts[counts == 0] = 1.0
    return out / counts


def _write_polygon_geojson(mask: np.ndarray, ref_profile: dict, out_path: Path) -> None:
    meta = ref_profile.copy()
    meta.update(dtype="uint8", nodata=0, count=1)

    processed = postprocess_prediction(mask, ref_profile["transform"])

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(processed.astype(np.uint8), 1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raster_to_geojson(tmp_path, output_path=str(out_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check model predictions on train tiles.")
    parser.add_argument("--data-dir", default="./data/makeathon-challenge")
    parser.add_argument(
        "--model-type",
        choices=["baseline1", "baseline2", "baseline3", "temporal_xgb", "patch_xgb", "unet"],
        required=True,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tile-ids", default="")
    parser.add_argument("--num-tiles", type=int, default=2)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--out-dir", default="")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    model_name = model_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("./artifacts/sanity") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tile_ids:
        tile_ids = [t.strip() for t in args.tile_ids.split(",") if t.strip()]
    else:
        label_tiles = sorted(label_tile_ids(data_dir / "labels" / "train"))
        tile_ids = label_tiles[: args.num_tiles]

    if not tile_ids:
        raise RuntimeError("No tiles selected for sanity check.")

    model = None
    ensemble = None
    if args.model_type in {"baseline1", "baseline2", "temporal_xgb", "patch_xgb"}:
        model = joblib.load(model_path)
    elif args.model_type == "baseline3":
        ensemble = joblib.load(model_path)
    else:
        model = None

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    tile_reports: dict[str, dict] = {}

    for tile_id in tile_ids:
        aef_path = data_dir / "aef-embeddings" / "train" / f"{tile_id}_{args.year}.tiff"
        if not aef_path.exists():
            logger.warning("Missing AEF for %s %s", tile_id, args.year)
            continue

        with rasterio.open(aef_path) as src:
            aef = src.read().astype(np.float32)
            ref_profile = src.profile

        labels = load_tile_labels(data_dir, tile_id, ref_profile)
        if labels is None:
            logger.warning("Missing labels for %s", tile_id)
            continue

        label_mask = build_label_mask(labels)

        if args.model_type == "baseline1":
            proba = _predict_pixel_xgb(aef, model)
        elif args.model_type == "baseline2":
            aef_2020_path = data_dir / "aef-embeddings" / "train" / f"{tile_id}_2020.tiff"
            if not aef_2020_path.exists():
                logger.warning("Missing 2020 AEF for %s", tile_id)
                continue
            with rasterio.open(aef_2020_path) as src:
                aef_2020 = src.read().astype(np.float32)

            prev_path = data_dir / "aef-embeddings" / "train" / f"{tile_id}_{args.year - 1}.tiff"
            aef_prev = None
            if args.year > 2020 and prev_path.exists():
                with rasterio.open(prev_path) as src:
                    aef_prev = src.read().astype(np.float32)

            proba = _predict_baseline2(aef, aef_2020, aef_prev, model)
        elif args.model_type == "baseline3":
            aef_2020_path = data_dir / "aef-embeddings" / "train" / f"{tile_id}_2020.tiff"
            if not aef_2020_path.exists():
                logger.warning("Missing 2020 AEF for %s", tile_id)
                continue
            with rasterio.open(aef_2020_path) as src:
                aef_2020 = src.read().astype(np.float32)

            prev_path = data_dir / "aef-embeddings" / "train" / f"{tile_id}_{args.year - 1}.tiff"
            aef_prev = None
            if args.year > 2020 and prev_path.exists():
                with rasterio.open(prev_path) as src:
                    aef_prev = src.read().astype(np.float32)

            models = ensemble.get("models", {})
            probs: list[np.ndarray] = []
            for _, mdl in models.items():
                probs.append(_predict_baseline2(aef, aef_2020, aef_prev, mdl))
            if not probs:
                raise RuntimeError("No models found in baseline3 ensemble.")
            proba = np.mean(np.stack(probs, axis=0), axis=0)
        elif args.model_type == "temporal_xgb":
            proba = _predict_temporal_xgb(aef, tile_id, data_dir, ref_profile, model)
        elif args.model_type == "patch_xgb":
            proba = _predict_patch_xgb(aef, model, args.patch_size, args.stride)
        else:
            proba = _predict_unet(aef, model_path, args.patch_size, args.stride)

        pred = (proba > args.threshold).astype(np.uint8)

        valid = label_mask >= 0
        y_true = label_mask[valid].astype(np.uint8)
        y_pred = pred[valid].astype(np.uint8)

        if y_true.size == 0:
            logger.warning("No valid labels for %s", tile_id)
            continue

        all_true.append(y_true)
        all_pred.append(y_pred)

        tile_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        tile_report["iou"] = _compute_iou(y_true, y_pred)
        tile_reports[tile_id] = tile_report

        rgb = _make_rgb(aef)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(rgb)
        axes[0].set_title(f"AEF RGB | {tile_id}")
        axes[0].axis("off")

        axes[1].imshow(rgb)
        axes[1].imshow(label_mask, cmap="gray", vmin=-1, vmax=1, alpha=0.55)
        axes[1].set_title("GLAD/RADD Consensus")
        axes[1].axis("off")

        axes[2].imshow(rgb)
        axes[2].imshow(pred, cmap="Reds", alpha=0.45)
        axes[2].set_title("Prediction Overlay")
        axes[2].axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / f"{tile_id}_overlay.png", dpi=200)
        plt.close(fig)

        _write_polygon_geojson(pred, ref_profile, out_dir / f"{tile_id}_pred.geojson")

    if not all_true:
        raise RuntimeError("No labeled pixels found for sanity check.")

    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)
    overall = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
    overall["iou"] = _compute_iou(y_true_all, y_pred_all)

    payload = {
        "script": "sanity_check.py",
        "model_type": args.model_type,
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "args": vars(args),
        "tiles": tile_ids,
        "overall": overall,
        "per_tile": tile_reports,
    }
    (out_dir / "sanity_report.json").write_text(json.dumps(payload, indent=2))
    logger.info("Saved sanity report to %s", out_dir / "sanity_report.json")


if __name__ == "__main__":
    main()
