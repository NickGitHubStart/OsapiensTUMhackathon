"""Train baseline3 with region holdout models and an ensemble bundle."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import rasterio
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.data_utils import iter_aef_files, label_tile_ids, load_tile_label_confidence

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _group_aef_by_tile(aef_dir: Path) -> dict[str, dict[int, Path]]:
    tiles: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in iter_aef_files(aef_dir):
        tile_id, year_str = path.stem.rsplit("_", 1)
        try:
            year = int(year_str)
        except ValueError:
            continue
        tiles[tile_id][year] = path
    return tiles


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


def _sample_rows(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    if n <= max_samples:
        return features, labels, weights

    idx = rng.choice(n, size=max_samples, replace=False)
    return features[idx], labels[idx], weights[idx]


def _sample_features(features: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    n = features.shape[0]
    if n <= max_samples:
        return features
    idx = rng.choice(n, size=max_samples, replace=False)
    return features[idx]


def _sample_features_with_weights(
    features: np.ndarray,
    weights: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = features.shape[0]
    if n <= max_samples:
        return features, weights
    idx = rng.choice(n, size=max_samples, replace=False)
    return features[idx], weights[idx]


def _make_model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def _compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def _train_and_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[XGBClassifier, dict]:
    model = _make_model(seed)
    model.fit(x_train, y_train, sample_weight=w_train)
    y_pred = model.predict(x_val)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    iou = _compute_iou(y_val, y_pred)
    report["iou"] = iou
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline3 with region holdout and full-data ensemble."
    )
    parser.add_argument(
        "--data-dir",
        default="./data/makeathon-challenge",
        help="Path to the downloaded dataset root",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="Maximum total samples to train on",
    )
    parser.add_argument(
        "--per-tile-samples",
        type=int,
        default=50000,
        help="Maximum samples to keep per tile before concatenation",
    )
    parser.add_argument(
        "--neg-pos-ratio",
        type=int,
        default=3,
        help="Number of negatives per positive",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed",
    )
    parser.add_argument(
        "--model-out",
        default="./artifacts/baseline3_weighted.joblib",
        help="Where to save the ensemble bundle",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    aef_dir = data_dir / "aef-embeddings" / "train"
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    label_tiles = label_tile_ids(data_dir / "labels" / "train")
    logger.info("Label tiles available: %d", len(label_tiles))

    tiles = _group_aef_by_tile(aef_dir)
    xs_by_region: dict[str, list[np.ndarray]] = defaultdict(list)
    ys_by_region: dict[str, list[np.ndarray]] = defaultdict(list)
    ws_by_region: dict[str, list[np.ndarray]] = defaultdict(list)
    rng = np.random.default_rng(args.seed)
    loaded_tiles = 0
    total_samples = 0

    for tile_id, years in tiles.items():
        if tile_id not in label_tiles:
            logger.info("Skipping %s (labels not available)", tile_id)
            continue

        if 2020 not in years:
            logger.info("Skipping %s (missing 2020 AEF)", tile_id)
            continue

        with rasterio.open(years[2020]) as src:
            aef_2020 = src.read().astype(np.float32)
            ref_profile = src.profile

        label_payload = load_tile_label_confidence(data_dir, tile_id, ref_profile)
        if label_payload is None:
            logger.warning("Missing labels for %s, skipping", tile_id)
            continue
        label_mask, weight_mask = label_payload

        for year, path in sorted(years.items()):
            if year < 2020:
                continue

            with rasterio.open(path) as src:
                aef = src.read().astype(np.float32)

            if year == 2020:
                diff_prev = np.zeros_like(aef)
            else:
                prev_path = years.get(year - 1)
                if prev_path is None:
                    diff_prev = np.zeros_like(aef)
                else:
                    with rasterio.open(prev_path) as src:
                        aef_prev = src.read().astype(np.float32)
                    diff_prev = aef - aef_prev

            diff_2020 = aef - aef_2020

            channels, height, width = aef.shape
            base = aef.reshape(channels, height * width).transpose(1, 0)
            d2020 = diff_2020.reshape(channels, height * width).transpose(1, 0)
            dprev = diff_prev.reshape(channels, height * width).transpose(1, 0)

            features = np.concatenate([base, d2020, dprev], axis=1)

            flat_labels = label_mask.reshape(-1)
            flat_weights = weight_mask.reshape(-1)
            pos_idx = flat_labels == 1
            neg_idx = flat_labels == 0

            pos_features = features[pos_idx]
            neg_features = features[neg_idx]
            pos_weights = flat_weights[pos_idx]
            neg_weights = flat_weights[neg_idx]

            if pos_features.size == 0 or neg_features.size == 0:
                logger.info(
                    "Skipping %s (%s) (pos=%d, neg=%d)",
                    tile_id,
                    year,
                    pos_features.shape[0],
                    neg_features.shape[0],
                )
                continue

            n_pos = pos_features.shape[0]
            n_neg = min(neg_features.shape[0], n_pos * args.neg_pos_ratio)
            neg_sel = rng.choice(neg_features.shape[0], size=n_neg, replace=False)
            neg_features = neg_features[neg_sel]
            neg_weights = neg_weights[neg_sel]

            pos_features, pos_weights = _sample_features_with_weights(
                pos_features, pos_weights, args.per_tile_samples, rng
            )
            neg_features, neg_weights = _sample_features_with_weights(
                neg_features, neg_weights, args.per_tile_samples, rng
            )

            x = np.concatenate([pos_features, neg_features], axis=0)
            y = np.concatenate(
                [
                    np.ones(pos_features.shape[0], dtype=np.uint8),
                    np.zeros(neg_features.shape[0], dtype=np.uint8),
                ],
                axis=0,
            )
            w = np.concatenate([pos_weights, neg_weights], axis=0)

            region = _infer_region(tile_id)
            xs_by_region[region].append(x)
            ys_by_region[region].append(y)
            ws_by_region[region].append(w)

            loaded_tiles += 1
            total_samples += x.shape[0]

            logger.info(
                "Loaded %s (%s): %d pos, %d neg",
                tile_id,
                year,
                n_pos,
                n_neg,
            )

    if not xs_by_region:
        raise RuntimeError("No training samples were collected.")

    reports: dict[str, dict] = {}
    models: dict[str, XGBClassifier] = {}

    for holdout in ["thailand", "colombia"]:
        if holdout not in xs_by_region:
            continue

        train_regions = [r for r in xs_by_region.keys() if r != holdout]
        x_train = np.concatenate(
            [np.concatenate(xs_by_region[r], axis=0) for r in train_regions], axis=0
        )
        y_train = np.concatenate(
            [np.concatenate(ys_by_region[r], axis=0) for r in train_regions], axis=0
        )
        w_train = np.concatenate(
            [np.concatenate(ws_by_region[r], axis=0) for r in train_regions], axis=0
        )
        x_val = np.concatenate(xs_by_region[holdout], axis=0)
        y_val = np.concatenate(ys_by_region[holdout], axis=0)
        w_val = np.concatenate(ws_by_region[holdout], axis=0)

        x_train, y_train, w_train = _sample_rows(
            x_train, y_train, w_train, args.max_samples, args.seed
        )
        x_val, y_val, w_val = _sample_rows(
            x_val, y_val, w_val, args.max_samples, args.seed + 1
        )

        model, report = _train_and_eval(x_train, y_train, w_train, x_val, y_val, args.seed)
        models[f"holdout_{holdout}"] = model
        reports[f"holdout_{holdout}"] = report
        logger.info(
            "Holdout %s: IoU=%.4f  precision=%.4f  recall=%.4f  f1=%.4f",
            holdout,
            report.get("iou", 0.0),
            report.get("1", {}).get("precision", 0.0),
            report.get("1", {}).get("recall", 0.0),
            report.get("1", {}).get("f1-score", 0.0),
        )

    all_regions = list(xs_by_region.keys())
    x_all = np.concatenate([np.concatenate(xs_by_region[r], axis=0) for r in all_regions], axis=0)
    y_all = np.concatenate([np.concatenate(ys_by_region[r], axis=0) for r in all_regions], axis=0)
    w_all = np.concatenate([np.concatenate(ws_by_region[r], axis=0) for r in all_regions], axis=0)
    x_all, y_all, w_all = _sample_rows(x_all, y_all, w_all, args.max_samples, args.seed)

    model_all = _make_model(args.seed)
    model_all.fit(x_all, y_all, sample_weight=w_all)
    models["all"] = model_all

    bundle = {
        "models": models,
        "model_names": list(models.keys()),
        "feature_spec": "aef + (aef-2020) + (aef-year-1)",
        "region_rule": "47/48=thailand, 18/19=colombia",
    }
    joblib.dump(bundle, model_out)
    logger.info("Saved ensemble bundle to %s", model_out)

    report_path = model_out.with_suffix(".json")
    payload = {
        "script": "train_baseline3.py",
        "data_dir": str(data_dir),
        "model_out": str(model_out),
        "args": vars(args),
        "loaded_tiles": loaded_tiles,
        "total_samples": int(total_samples),
        "reports": reports,
    }
    report_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved report to %s", report_path)


if __name__ == "__main__":
    main()
