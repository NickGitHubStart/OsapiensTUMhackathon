"""Train baseline4 patch-based XGBoost models from cached features."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from rasterio.transform import Affine
from scipy.ndimage import binary_opening, label as cc_label
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _infer_region(tile_id: str) -> str:
    try:
        zone = int(tile_id[:2])
    except ValueError:
        return "other"
    if zone in {47, 48}:
        return "thailand"
    if zone in {18, 19}:
        return "colombia"
    return "other"


def _iter_patch_coords(height: int, width: int, patch_size: int, stride: int) -> Iterable[tuple[int, int]]:
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


def _extract_patch_samples(
    features: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    ndvi_drop: np.ndarray,
    vv_drop: np.ndarray,
    forest_mask: np.ndarray,
    patch_size: int,
    stride: int,
    ndvi_thresh: float,
    vv_thresh: float,
    include_gradient: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = label.shape

    xs: list[np.ndarray] = []
    ys: list[float] = []
    ws: list[float] = []

    for y0, x0 in _iter_patch_coords(height, width, patch_size, stride):
        patch_label = label[y0 : y0 + patch_size, x0 : x0 + patch_size]
        patch_weight = weight[y0 : y0 + patch_size, x0 : x0 + patch_size]

        denom = patch_weight.sum()
        if denom <= 0:
            continue

        pos_mask = patch_label == 1
        y_val = float((patch_weight * pos_mask).sum() / denom)
        w_val = float(patch_weight.mean())

        patch_feat = features[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
        patch_ndvi_drop = ndvi_drop[y0 : y0 + patch_size, x0 : x0 + patch_size]
        patch_vv_drop = vv_drop[y0 : y0 + patch_size, x0 : x0 + patch_size]
        patch_forest = forest_mask[y0 : y0 + patch_size, x0 : x0 + patch_size]

        vec = _patch_vector(patch_feat, include_gradient)
        spatial = _spatial_features(patch_ndvi_drop, patch_vv_drop, patch_forest, ndvi_thresh, vv_thresh)
        vec = np.concatenate([vec, spatial], axis=0)

        xs.append(vec.astype(np.float32))
        ys.append(y_val)
        ws.append(w_val)

    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.stack(xs, axis=0), np.array(ys, dtype=np.float32), np.array(ws, dtype=np.float32)


def _sample_rows(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x, y, w
    idx = rng.choice(x.shape[0], size=max_samples, replace=False)
    return x[idx], y[idx], w[idx]


def _make_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:logistic",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def _predict_tile_proba(
    features: np.ndarray,
    ndvi_drop: np.ndarray,
    vv_drop: np.ndarray,
    forest_mask: np.ndarray,
    model: XGBRegressor,
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


def _metrics_from_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _evaluate_model(
    tiles: list[Path],
    model: XGBRegressor,
    patch_size: int,
    stride: int,
    ndvi_thresh: float,
    vv_thresh: float,
    include_gradient: bool,
    thresholds: list[float],
    min_area_ha: float,
    apply_opening: bool,
) -> tuple[float, dict]:
    preds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for path in tiles:
        tile_id = path.stem
        data = np.load(path)
        features = data["features"].astype(np.float32)
        label = data["label"].astype(np.int8)
        ndvi_drop = data["ndvi_drop"].astype(np.float32)
        vv_drop = data["vv_drop"].astype(np.float32)
        forest_mask = data["forest_mask"].astype(np.uint8)
        transform = Affine.from_gdal(*data["transform"].tolist())

        if np.all(label < 0):
            logger.info("Skipping %s (no labels)", tile_id)
            continue

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
        preds.append((proba, label, transform))

    if not preds:
        return 0.5, {"error": "no labeled tiles"}

    best_thr = 0.5
    best_iou = -1.0
    best_metrics: dict | None = None

    for thr in thresholds:
        tp = fp = fn = 0
        for proba, label, transform in preds:
            binary = (proba > thr).astype(np.uint8)
            binary = _postprocess_mask(binary, transform, min_area_ha, apply_opening)

            valid = label >= 0
            y_true = label[valid]
            y_pred = binary[valid]

            tp += int(((y_true == 1) & (y_pred == 1)).sum())
            fp += int(((y_true == 0) & (y_pred == 1)).sum())
            fn += int(((y_true == 1) & (y_pred == 0)).sum())

        metrics = _metrics_from_counts(tp, fp, fn)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_thr = thr
            best_metrics = metrics

    return best_thr, best_metrics or {}


def _load_tiles(cache_dir: Path) -> dict[str, list[Path]]:
    tiles_by_region: dict[str, list[Path]] = {"thailand": [], "colombia": [], "other": []}
    for path in sorted((cache_dir / "train").glob("*.npz")):
        region = _infer_region(path.stem)
        tiles_by_region.setdefault(region, []).append(path)
    return tiles_by_region


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline4 patch-based XGBoost models.")
    parser.add_argument("--cache-dir", default="./data/makeathon-challenge-cache/baseline4")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--stride-train", type=int, default=16)
    parser.add_argument("--stride-infer", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--per-tile-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ndvi-drop-threshold", type=float, default=0.2)
    parser.add_argument("--vv-drop-threshold", type=float, default=1.5)
    parser.add_argument("--include-gradient", action="store_true")
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    parser.add_argument("--apply-opening", action="store_true")
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--model-out", default="./artifacts/baseline4_patch_xgb_ensemble.joblib")

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    spec_path = cache_dir / "feature_spec.json"
    feature_spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}

    if args.thresholds:
        thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    else:
        thresholds = [round(x, 2) for x in np.linspace(0.1, 0.9, 17)]

    tiles_by_region = _load_tiles(cache_dir)
    rng = np.random.default_rng(args.seed)

    xs_by_region: dict[str, list[np.ndarray]] = {"thailand": [], "colombia": [], "other": []}
    ys_by_region: dict[str, list[np.ndarray]] = {"thailand": [], "colombia": [], "other": []}
    ws_by_region: dict[str, list[np.ndarray]] = {"thailand": [], "colombia": [], "other": []}

    for region, tiles in tiles_by_region.items():
        for path in tiles:
            data = np.load(path)
            features = data["features"].astype(np.float32)
            label = data["label"].astype(np.int8)
            weight = data["weight"].astype(np.float32)
            ndvi_drop = data["ndvi_drop"].astype(np.float32)
            vv_drop = data["vv_drop"].astype(np.float32)
            forest_mask = data["forest_mask"].astype(np.uint8)

            if np.all(label < 0):
                continue

            x_tile, y_tile, w_tile = _extract_patch_samples(
                features,
                label,
                weight,
                ndvi_drop,
                vv_drop,
                forest_mask,
                args.patch_size,
                args.stride_train,
                args.ndvi_drop_threshold,
                args.vv_drop_threshold,
                args.include_gradient,
            )

            if x_tile.size == 0:
                continue

            if args.per_tile_samples > 0 and x_tile.shape[0] > args.per_tile_samples:
                idx = rng.choice(x_tile.shape[0], size=args.per_tile_samples, replace=False)
                x_tile = x_tile[idx]
                y_tile = y_tile[idx]
                w_tile = w_tile[idx]

            xs_by_region[region].append(x_tile)
            ys_by_region[region].append(y_tile)
            ws_by_region[region].append(w_tile)

    if not any(xs_by_region[region] for region in xs_by_region):
        raise RuntimeError("No training patches found in cache.")

    def _concat(region: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not xs_by_region[region]:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return (
            np.concatenate(xs_by_region[region], axis=0),
            np.concatenate(ys_by_region[region], axis=0),
            np.concatenate(ws_by_region[region], axis=0),
        )

    reports: dict[str, dict] = {}
    models: dict[str, XGBRegressor] = {}
    thresholds_out: dict[str, float] = {}

    for holdout in ["thailand", "colombia"]:
        if not tiles_by_region[holdout]:
            continue

        train_regions = [r for r in ["thailand", "colombia"] if r != holdout and tiles_by_region[r]]
        if not train_regions:
            continue

        x_train = np.concatenate([_concat(r)[0] for r in train_regions], axis=0)
        y_train = np.concatenate([_concat(r)[1] for r in train_regions], axis=0)
        w_train = np.concatenate([_concat(r)[2] for r in train_regions], axis=0)

        x_train, y_train, w_train = _sample_rows(x_train, y_train, w_train, args.max_samples, rng)

        model = _make_model(args.seed)
        model.fit(x_train, y_train, sample_weight=w_train)

        best_thr, metrics = _evaluate_model(
            tiles_by_region[holdout],
            model,
            args.patch_size,
            args.stride_infer,
            args.ndvi_drop_threshold,
            args.vv_drop_threshold,
            args.include_gradient,
            thresholds,
            args.min_area_ha,
            args.apply_opening,
        )

        models[f"holdout_{holdout}"] = model
        thresholds_out[f"holdout_{holdout}"] = best_thr
        reports[f"holdout_{holdout}"] = metrics
        logger.info("Holdout %s: IoU=%.4f (thr=%.2f)", holdout, metrics.get("iou", 0.0), best_thr)

    regions = [r for r in ["thailand", "colombia"] if tiles_by_region[r]]
    if regions:
        per_region = args.max_samples // len(regions) if args.max_samples > 0 else 0

        xs_all: list[np.ndarray] = []
        ys_all: list[np.ndarray] = []
        ws_all: list[np.ndarray] = []

        for region in regions:
            x_reg, y_reg, w_reg = _concat(region)
            if x_reg.size == 0:
                continue
            if per_region > 0:
                x_reg, y_reg, w_reg = _sample_rows(x_reg, y_reg, w_reg, per_region, rng)
            xs_all.append(x_reg)
            ys_all.append(y_reg)
            ws_all.append(w_reg)

        if xs_all:
            x_all = np.concatenate(xs_all, axis=0)
            y_all = np.concatenate(ys_all, axis=0)
            w_all = np.concatenate(ws_all, axis=0)

            model_all = _make_model(args.seed)
            model_all.fit(x_all, y_all, sample_weight=w_all)

            best_thr, metrics = _evaluate_model(
                sum([tiles_by_region[r] for r in regions], []),
                model_all,
                args.patch_size,
                args.stride_infer,
                args.ndvi_drop_threshold,
                args.vv_drop_threshold,
                args.include_gradient,
                thresholds,
                args.min_area_ha,
                args.apply_opening,
            )

            models["all_data"] = model_all
            thresholds_out["all_data"] = best_thr
            reports["all_data"] = metrics
            logger.info("All-data: IoU=%.4f (thr=%.2f)", metrics.get("iou", 0.0), best_thr)

    payload = {
        "script": "train_baseline4.py",
        "cache_dir": str(cache_dir),
        "feature_spec": feature_spec,
        "patch_size": args.patch_size,
        "stride_train": args.stride_train,
        "stride_infer": args.stride_infer,
        "ndvi_drop_threshold": args.ndvi_drop_threshold,
        "vv_drop_threshold": args.vv_drop_threshold,
        "include_gradient": args.include_gradient,
        "min_area_ha": args.min_area_ha,
        "apply_opening": args.apply_opening,
        "thresholds": thresholds,
        "reports": reports,
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "thresholds": thresholds_out,
            "report": payload,
            "patch_size": args.patch_size,
            "stride_infer": args.stride_infer,
            "ndvi_drop_threshold": args.ndvi_drop_threshold,
            "vv_drop_threshold": args.vv_drop_threshold,
            "include_gradient": args.include_gradient,
            "min_area_ha": args.min_area_ha,
            "apply_opening": args.apply_opening,
        },
        model_out,
    )

    report_path = model_out.with_suffix(".json")
    report_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved bundle to %s", model_out)


if __name__ == "__main__":
    main()
