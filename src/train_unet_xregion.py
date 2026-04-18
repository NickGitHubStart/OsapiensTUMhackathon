"""Cross-region U-Net training for the deforestation challenge.

Trains on one region (e.g. Americas) and validates on the other (e.g. Asia)
so the validation signal forces the model to learn region-invariant features.
This is intended to be run twice (Americas->Asia and Asia->Americas) so the
two checkpoints can be ensembled at inference time for a leaderboard score
that generalizes to the held-out Africa test continent.

Highlights vs. ``train_unet_geo_v2``:
- Strict ``--train-region``/``--val-region`` (americas|asia) selection.
- ``xregion`` label strategy that tolerates Asia tiles missing GLAD-S2.
- BCE + Dice + Tversky loss (alpha=0.7, beta=0.3 by default to favour IoU).
- Best checkpoint selected by validation **IoU** (the leaderboard metric).
- Per-tile validation tile IDs logged for transparency.
- Threshold sweep on the OOD validation set; best threshold saved next to the
  weights so inference uses model-specific operating points.
- Defaults tuned for CPU training with 8 threads (set OMP_NUM_THREADS=8).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import deque
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency

    def tqdm(it, **_kwargs):  # type: ignore[misc]
        return it

from src.data_utils import (
    apply_patch_channel_dropout,
    apply_patch_noise,
    apply_spatial_aug,
    build_label_mask,
    iter_aef_files,
)
from src.enhanced_labels import label_tile_ids_xregion, load_tile_labels_enhanced
from src.geo_validation import infer_region_from_tile_id, load_tile_regions_from_geojson
from src.train_unet import UNetSmall, _masked_bce_logits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


REGION_CHOICES = ("americas", "asia")


# ---------------------------------------------------------------------------
# Dataset


class _TileCache:
    def __init__(self, max_items: int = 16) -> None:
        self.max_items = max_items
        self._data: dict[tuple[Path, str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._order: deque[tuple[Path, str, str]] = deque()

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value) -> None:
        if key in self._data:
            return
        if len(self._order) >= self.max_items:
            old_key = self._order.popleft()
            self._data.pop(old_key, None)
        self._data[key] = value
        self._order.append(key)


def _read_tile_cached(
    aef_path: Path,
    data_dir: Path,
    label_strategy: str,
    forest_mask_dir: Path | None,
    aef_downsample: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one (AEF tensor, label mask) pair from disk."""
    tile_id, year_str = aef_path.stem.rsplit("_", 1)
    year = int(year_str)
    aef_downsample = max(1, int(aef_downsample))
    with rasterio.open(aef_path) as src:
        if aef_downsample > 1:
            out_h = max(1, src.height // aef_downsample)
            out_w = max(1, src.width // aef_downsample)
            aef = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.nearest,
            ).astype(np.float32)
            sx = src.width / out_w
            sy = src.height / out_h
            transform = src.transform * src.transform.scale(sx, sy)
            ref_profile = src.profile.copy()
            ref_profile.update(height=out_h, width=out_w, transform=transform)
        else:
            aef = src.read().astype(np.float32)
            ref_profile = src.profile

    labels = load_tile_labels_enhanced(
        data_dir,
        tile_id,
        ref_profile,
        year=year,
        forest_mask_dir=forest_mask_dir,
        label_strategy=label_strategy,
    )
    if labels is None:
        return None
    label_mask = build_label_mask(labels)
    if not np.isfinite(aef).all():
        aef = np.nan_to_num(aef, nan=0.0, posinf=0.0, neginf=0.0)
    return aef, label_mask


class AEFXRegionDataset(Dataset):
    def __init__(
        self,
        aef_paths: list[Path],
        data_dir: Path,
        patch_size: int,
        samples_per_epoch: int,
        seed: int,
        min_labeled_frac: float,
        min_pos_frac: float,
        positive_oversample_prob: float,
        max_tries: int,
        label_strategy: str,
        forest_mask_dir: Path | None,
        aug_flip_rotate_prob: float,
        aug_noise_std: float,
        aug_dropout_prob: float,
        aug_dropout_frac: float,
        aug_scale_min: float,
        aug_scale_max: float,
        aef_downsample: int,
        preloaded: list[tuple[np.ndarray, np.ndarray]] | None = None,
        positive_indices: list[np.ndarray] | None = None,
    ) -> None:
        self.aef_paths = aef_paths
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.min_labeled_frac = min_labeled_frac
        self.min_pos_frac = min_pos_frac
        self.positive_oversample_prob = positive_oversample_prob
        self.max_tries = max_tries
        self.label_strategy = label_strategy
        self.forest_mask_dir = forest_mask_dir
        self.aug_flip_rotate_prob = aug_flip_rotate_prob
        self.aug_noise_std = aug_noise_std
        self.aug_dropout_prob = aug_dropout_prob
        self.aug_dropout_frac = aug_dropout_frac
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aef_downsample = max(1, int(aef_downsample))
        # If preloaded is provided we skip on-disk caching entirely.
        self._preloaded = preloaded
        self._positive_indices = positive_indices
        self._cache = _TileCache(max_items=24) if preloaded is None else None

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_tile(self, aef_path: Path):
        cache_key = (aef_path, self.label_strategy, str(self.forest_mask_dir or ""))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        loaded = _read_tile_cached(
            aef_path,
            self.data_dir,
            self.label_strategy,
            self.forest_mask_dir,
            self.aef_downsample,
        )
        if loaded is not None:
            self._cache.set(cache_key, loaded)
        return loaded

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        want_positive = rng.random() < self.positive_oversample_prob

        for attempt in range(self.max_tries):
            tile_idx = int(rng.integers(0, len(self.aef_paths)))
            if self._preloaded is not None:
                aef, label_mask = self._preloaded[tile_idx]
                pos_idx_array = (
                    self._positive_indices[tile_idx] if self._positive_indices else None
                )
            else:
                aef_path = self.aef_paths[tile_idx]
                loaded = self._load_tile(aef_path)
                if loaded is None:
                    continue
                aef, label_mask = loaded
                pos_idx_array = None
            _, height, width = aef.shape
            if height < self.patch_size or width < self.patch_size:
                continue

            if want_positive:
                if pos_idx_array is None:
                    pos_idx_array = np.argwhere(label_mask == 1)
                if pos_idx_array.size > 0:
                    py, px = pos_idx_array[int(rng.integers(0, len(pos_idx_array)))]
                    y0 = int(np.clip(py - self.patch_size // 2, 0, height - self.patch_size))
                    x0 = int(np.clip(px - self.patch_size // 2, 0, width - self.patch_size))
                else:
                    y0 = int(rng.integers(0, height - self.patch_size + 1))
                    x0 = int(rng.integers(0, width - self.patch_size + 1))
            else:
                y0 = int(rng.integers(0, height - self.patch_size + 1))
                x0 = int(rng.integers(0, width - self.patch_size + 1))

            label_patch = label_mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
            features = aef[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]

            features, label_patch = apply_spatial_aug(
                features,
                label_patch,
                rng,
                self.aug_flip_rotate_prob,
                self.aug_scale_min,
                self.aug_scale_max,
            )
            features = np.ascontiguousarray(features)
            label_patch = np.ascontiguousarray(label_patch)

            valid = label_patch >= 0
            if valid.mean() < self.min_labeled_frac:
                continue

            if want_positive:
                pos_frac = float((label_patch == 1).sum()) / float(label_patch.size)
                if pos_frac < self.min_pos_frac and attempt < self.max_tries - 1:
                    continue

            features = apply_patch_noise(features, rng, self.aug_noise_std)
            features = apply_patch_channel_dropout(
                features, rng, self.aug_dropout_prob, self.aug_dropout_frac
            )

            labels = label_patch.astype(np.float32)
            labels[labels < 0] = 0.0
            mask = valid.astype(np.float32)
            if not np.isfinite(features).all():
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return (
                torch.from_numpy(features),
                torch.from_numpy(labels[None, ...]),
                torch.from_numpy(mask[None, ...]),
            )

        raise RuntimeError(
            "Failed to sample a valid patch; lower --min-labeled-frac or check labels."
        )


# ---------------------------------------------------------------------------
# Losses & metrics


def _masked_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probs = torch.sigmoid(logits) * mask
    targets = targets * mask
    intersection = (probs * targets).sum()
    denom = probs.sum() + targets.sum()
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice


def _masked_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Tversky loss; alpha penalises FP, beta penalises FN.

    For Union IoU on the leaderboard, slightly higher alpha (e.g. 0.7) reduces
    over-prediction since over-predicted polygons immediately hurt Union IoU.
    """
    probs = torch.sigmoid(logits) * mask
    targets = targets * mask
    tp = (probs * targets).sum()
    fp = (probs * (1.0 - targets)).sum()
    fn = ((1.0 - probs) * targets).sum()
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky


def _confusion_components(
    probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    valid = mask > 0.5
    pred_pos = (probs >= threshold) & valid
    label_pos = (labels > 0.5) & valid
    tp = float((pred_pos & label_pos).sum())
    fp = float((pred_pos & ~label_pos).sum())
    fn = float((~pred_pos & label_pos).sum())
    return tp, fp, fn


def _metrics_from_components(tp: float, fp: float, fn: float) -> tuple[float, float, float, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return precision, recall, f1, iou


# ---------------------------------------------------------------------------
# Region selection


def _build_region_map(label_tiles: set[str], geojson_path: Path) -> dict[str, str]:
    region_map: dict[str, str] = {}
    if geojson_path.is_file():
        region_map.update(load_tile_regions_from_geojson(geojson_path))
    for tid in label_tiles:
        if tid not in region_map:
            region_map[tid] = infer_region_from_tile_id(tid)
    return region_map


def _filter_tiles(label_tiles: set[str], region_map: dict[str, str], region: str) -> set[str]:
    return {t for t in label_tiles if region_map.get(t) == region}


# ---------------------------------------------------------------------------
# History helpers


def _save_history_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_iou",
        "best_threshold",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_history_plot(rows: list[dict[str, float]], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    val_iou = [float(r["val_iou"]) for r in rows]
    val_f1 = [float(r["val_f1"]) for r in rows]
    val_prec = [float(r["val_precision"]) for r in rows]
    val_rec = [float(r["val_recall"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train loss")
    axes[0].plot(epochs, val_loss, label="val loss (OOD)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} — losses")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_iou, label="val IoU @ best thr")
    axes[1].plot(epochs, val_f1, label="val F1 @ best thr")
    axes[1].plot(epochs, val_prec, label="val precision", alpha=0.6)
    axes[1].plot(epochs, val_rec, label="val recall", alpha=0.6)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title(f"{title} — OOD validation metrics")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-region U-Net (train one continent, validate on the other)."
    )
    parser.add_argument("--data-dir", default="./data/makeathon-challenge")
    parser.add_argument(
        "--train-region",
        choices=REGION_CHOICES,
        required=True,
        help="Region used for training tiles.",
    )
    parser.add_argument(
        "--val-region",
        choices=REGION_CHOICES,
        required=True,
        help="Region used for OOD validation tiles.",
    )
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=1200)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--min-labeled-frac", type=float, default=0.05)
    parser.add_argument("--min-pos-frac", type=float, default=0.001)
    parser.add_argument("--positive-oversample-prob", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bce-weight", type=float, default=0.4)
    parser.add_argument("--dice-weight", type=float, default=0.3)
    parser.add_argument("--tversky-weight", type=float, default=0.3)
    parser.add_argument("--tversky-alpha", type=float, default=0.7)
    parser.add_argument("--tversky-beta", type=float, default=0.3)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument(
        "--label-strategy",
        choices=("multi_source", "glads2_radd", "xregion"),
        default="xregion",
    )
    parser.add_argument("--forest-mask-dir", type=Path, default=None)
    parser.add_argument("--aug-flip-rotate-prob", type=float, default=0.5)
    parser.add_argument("--aug-noise-std", type=float, default=0.02)
    parser.add_argument("--aug-dropout-prob", type=float, default=0.3)
    parser.add_argument("--aug-dropout-frac", type=float, default=0.1)
    parser.add_argument("--aug-scale-min", type=float, default=0.9)
    parser.add_argument("--aug-scale-max", type=float, default=1.1)
    parser.add_argument(
        "--aef-downsample",
        type=int,
        default=2,
        help="Read AEF at lower resolution (1=full, 2/4 for faster CPU training).",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="If >0, set torch.set_num_threads to this. Useful when running two "
        "trainings in parallel on the same machine.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. With --preload, 0 is fastest (data lives in main proc).",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload all train+val tiles into RAM upfront (recommended on a fat box).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed-precision autocast on CUDA/ROCm.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="U-Net base channel width (32=small, 48=medium, 64=large).",
    )
    parser.add_argument(
        "--cosine-lr",
        action="store_true",
        help="Cosine LR schedule with warmup over total epochs.",
    )
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument(
        "--tag",
        default=None,
        help="Override artifact filename tag (defaults to <train>_to_<val>).",
    )
    parser.add_argument("--artifact-dir", default="./artifacts")
    parser.add_argument("--geojson-path", type=Path, default=None)

    args = parser.parse_args()
    if args.train_region == args.val_region:
        raise ValueError("--train-region and --val-region must differ for cross-region training.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    logger.info("torch num_threads = %d", torch.get_num_threads())

    data_dir = Path(args.data_dir)
    aef_dir = data_dir / "aef-embeddings" / "train"
    geojson = args.geojson_path or (data_dir / "metadata" / "train_tiles.geojson")
    label_tiles = label_tile_ids_xregion(data_dir / "labels" / "train")

    region_map = _build_region_map(label_tiles, geojson)
    train_tiles = _filter_tiles(label_tiles, region_map, args.train_region)
    val_tiles = _filter_tiles(label_tiles, region_map, args.val_region)
    if not train_tiles or not val_tiles:
        raise RuntimeError(
            f"Empty tile set: train={len(train_tiles)}, val={len(val_tiles)}. "
            "Check region inference."
        )
    logger.info(
        "Cross-region split: train_region=%s (%d tiles) -> val_region=%s (%d tiles)",
        args.train_region,
        len(train_tiles),
        args.val_region,
        len(val_tiles),
    )
    logger.info("Train tiles: %s", sorted(train_tiles))
    logger.info("Val tiles:   %s", sorted(val_tiles))

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for aef_path in iter_aef_files(aef_dir):
        tile_id, year_str = aef_path.stem.rsplit("_", 1)
        year = int(year_str)
        if year < 2020:
            continue
        if tile_id in train_tiles:
            train_paths.append(aef_path)
        elif tile_id in val_tiles:
            val_paths.append(aef_path)
    if not train_paths or not val_paths:
        raise RuntimeError("Need non-empty train/val AEF paths after filtering.")

    tag = args.tag or f"{args.train_region}_to_{args.val_region}"
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_model_out = artifact_dir / f"unet_xregion_{tag}_best.pt"
    last_model_out = artifact_dir / f"unet_xregion_{tag}_last.pt"
    history_csv = artifact_dir / f"unet_xregion_{tag}_history.csv"
    history_plot = artifact_dir / f"unet_xregion_{tag}_history.png"
    meta_json = artifact_dir / f"unet_xregion_{tag}_meta.json"

    train_preloaded = None
    train_pos_idx = None
    val_preloaded = None
    val_pos_idx = None
    if args.preload:
        logger.info("Preloading %d train + %d val tiles into RAM...", len(train_paths), len(val_paths))
        train_preloaded = []
        train_pos_idx = []
        for p in train_paths:
            loaded = _read_tile_cached(
                p, data_dir, args.label_strategy, args.forest_mask_dir, args.aef_downsample,
            )
            if loaded is None:
                continue
            aef, lm = loaded
            train_preloaded.append((aef, lm))
            train_pos_idx.append(np.argwhere(lm == 1))
        val_preloaded = []
        val_pos_idx = []
        for p in val_paths:
            loaded = _read_tile_cached(
                p, data_dir, args.label_strategy, args.forest_mask_dir, args.aef_downsample,
            )
            if loaded is None:
                continue
            aef, lm = loaded
            val_preloaded.append((aef, lm))
            val_pos_idx.append(np.argwhere(lm == 1))
        # Filter aef_paths to those that successfully loaded so indexing matches.
        train_paths = train_paths[: len(train_preloaded)]
        val_paths = val_paths[: len(val_preloaded)]
        n_train_pos = sum(int(arr.size > 0) for arr in train_pos_idx)
        n_val_pos = sum(int(arr.size > 0) for arr in val_pos_idx)
        logger.info(
            "Preloaded train tiles=%d (with positives=%d), val tiles=%d (with positives=%d)",
            len(train_preloaded),
            n_train_pos,
            len(val_preloaded),
            n_val_pos,
        )

    train_ds = AEFXRegionDataset(
        aef_paths=train_paths,
        data_dir=data_dir,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        min_labeled_frac=args.min_labeled_frac,
        min_pos_frac=args.min_pos_frac,
        positive_oversample_prob=args.positive_oversample_prob,
        max_tries=80,
        label_strategy=args.label_strategy,
        forest_mask_dir=args.forest_mask_dir,
        aug_flip_rotate_prob=args.aug_flip_rotate_prob,
        aug_noise_std=args.aug_noise_std,
        aug_dropout_prob=args.aug_dropout_prob,
        aug_dropout_frac=args.aug_dropout_frac,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
        aef_downsample=args.aef_downsample,
        preloaded=train_preloaded,
        positive_indices=train_pos_idx,
    )
    val_ds = AEFXRegionDataset(
        aef_paths=val_paths,
        data_dir=data_dir,
        patch_size=args.patch_size,
        samples_per_epoch=args.val_samples,
        seed=args.seed + 999,
        min_labeled_frac=args.min_labeled_frac,
        min_pos_frac=0.0,
        positive_oversample_prob=0.0,
        max_tries=80,
        label_strategy=args.label_strategy,
        forest_mask_dir=args.forest_mask_dir,
        aug_flip_rotate_prob=0.0,
        aug_noise_std=0.0,
        aug_dropout_prob=0.0,
        aug_dropout_frac=0.0,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
        aef_downsample=args.aef_downsample,
        preloaded=val_preloaded,
        positive_indices=val_pos_idx,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    sample_aef, _, _ = train_ds[0]
    in_channels = sample_aef.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall(in_channels=in_channels, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )
    use_amp = bool(args.amp and torch.cuda.is_available())
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(
        "device=%s amp=%s base_channels=%d in_channels=%d cosine_lr=%s",
        device,
        use_amp,
        args.base_channels,
        in_channels,
        bool(args.cosine_lr),
    )

    history: list[dict[str, float]] = []
    best_iou = -1.0
    best_epoch = 0
    best_threshold = 0.5
    patience_ctr = 0

    threshold_grid = np.linspace(0.2, 0.8, 13)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0
        bad_train_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"[{tag}] train epoch {epoch}/{args.epochs}",
            leave=False,
            disable=args.no_progress_bar,
        )
        for features, labels, mask in pbar:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
            mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

            if mask.sum() < 1.0:
                bad_train_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(features)
                bce = _masked_bce_logits(logits, labels, mask) if args.bce_weight > 0 else 0.0
                dice = _masked_dice_loss(logits, labels, mask) if args.dice_weight > 0 else 0.0
                tversky = (
                    _masked_tversky_loss(
                        logits, labels, mask, args.tversky_alpha, args.tversky_beta
                    )
                    if args.tversky_weight > 0
                    else 0.0
                )
                loss = (
                    args.bce_weight * bce
                    + args.dice_weight * dice
                    + args.tversky_weight * tversky
                )
            if not torch.isfinite(loss):
                bad_train_batches += 1
                continue
            if use_amp:
                scaler.scale(loss).backward()
                if args.max_grad_norm and args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.max_grad_norm and args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
            train_loss_sum += float(loss.detach().cpu())
            n_train_batches += 1
            try:
                pbar.set_postfix(loss=float(loss.detach().cpu()))
            except Exception:
                pass

        if scheduler is not None:
            scheduler.step()
        train_loss = train_loss_sum / max(1, n_train_batches)
        if bad_train_batches:
            logger.warning("[%s] skipped %d bad train batches", tag, bad_train_batches)

        # ---- Validation: collect probs + labels for threshold sweep
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        # Accumulate per-threshold confusion components.
        comp_tp = np.zeros_like(threshold_grid, dtype=np.float64)
        comp_fp = np.zeros_like(threshold_grid, dtype=np.float64)
        comp_fn = np.zeros_like(threshold_grid, dtype=np.float64)

        with torch.no_grad():
            vbar = tqdm(
                val_loader,
                desc=f"[{tag}] val epoch {epoch}/{args.epochs}",
                leave=False,
                disable=args.no_progress_bar,
            )
            for features, labels, mask in vbar:
                features = features.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
                if mask.sum() < 1.0:
                    continue

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(features)
                    bce = _masked_bce_logits(logits, labels, mask) if args.bce_weight > 0 else 0.0
                    dice = _masked_dice_loss(logits, labels, mask) if args.dice_weight > 0 else 0.0
                    tversky = (
                        _masked_tversky_loss(
                            logits, labels, mask, args.tversky_alpha, args.tversky_beta
                        )
                        if args.tversky_weight > 0
                        else 0.0
                    )
                    loss = (
                        args.bce_weight * bce
                        + args.dice_weight * dice
                        + args.tversky_weight * tversky
                    )
                if torch.isfinite(loss):
                    val_loss_sum += float(loss.detach().cpu())
                    n_val_batches += 1

                probs = torch.sigmoid(logits).float().cpu().numpy()
                labels_np = labels.cpu().numpy()
                mask_np = mask.cpu().numpy()
                for i, thr in enumerate(threshold_grid):
                    tp, fp, fn = _confusion_components(probs, labels_np, mask_np, float(thr))
                    comp_tp[i] += tp
                    comp_fp[i] += fp
                    comp_fn[i] += fn

        val_loss = val_loss_sum / max(1, n_val_batches)
        ious = comp_tp / (comp_tp + comp_fp + comp_fn + 1e-8)
        best_idx = int(np.argmax(ious))
        cur_threshold = float(threshold_grid[best_idx])
        precision, recall, f1, iou = _metrics_from_components(
            comp_tp[best_idx], comp_fp[best_idx], comp_fn[best_idx]
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_precision": float(precision),
                "val_recall": float(recall),
                "val_f1": float(f1),
                "val_iou": float(iou),
                "best_threshold": cur_threshold,
            }
        )

        logger.info(
            "[%s] epoch %d/%d train_loss=%.4f val_loss=%.4f val_iou=%.4f val_f1=%.4f "
            "P=%.3f R=%.3f thr=%.2f",
            tag,
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            iou,
            f1,
            precision,
            recall,
            cur_threshold,
        )

        if iou > best_iou + 1e-6:
            best_iou = iou
            best_epoch = epoch
            best_threshold = cur_threshold
            patience_ctr = 0
            torch.save(model.state_dict(), best_model_out)
            logger.info("[%s] new best val_iou=%.4f thr=%.2f -> %s", tag, iou, cur_threshold, best_model_out)
        else:
            patience_ctr += 1
            if args.early_stop_patience > 0 and patience_ctr >= args.early_stop_patience:
                logger.info(
                    "[%s] early stopping at epoch %d (best epoch=%d, best val_iou=%.4f thr=%.2f)",
                    tag,
                    epoch,
                    best_epoch,
                    best_iou,
                    best_threshold,
                )
                break

        # Always save history each epoch so we can monitor live.
        _save_history_csv(history, history_csv)

    torch.save(model.state_dict(), last_model_out)
    _save_history_csv(history, history_csv)
    _save_history_plot(history, history_plot, title=f"U-Net {tag}")

    meta = {
        "tag": tag,
        "train_region": args.train_region,
        "val_region": args.val_region,
        "train_tiles": sorted(train_tiles),
        "val_tiles": sorted(val_tiles),
        "in_channels": int(in_channels),
        "base_channels": int(args.base_channels),
        "patch_size": int(args.patch_size),
        "aef_downsample": int(args.aef_downsample),
        "best_epoch": int(best_epoch),
        "best_val_iou": float(best_iou),
        "best_threshold": float(best_threshold),
        "label_strategy": args.label_strategy,
        "loss_weights": {
            "bce": args.bce_weight,
            "dice": args.dice_weight,
            "tversky": args.tversky_weight,
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
        },
        "best_model_path": str(best_model_out),
        "last_model_path": str(last_model_out),
    }
    meta_json.write_text(json.dumps(meta, indent=2))
    logger.info("[%s] saved meta -> %s", tag, meta_json)


if __name__ == "__main__":
    main()
