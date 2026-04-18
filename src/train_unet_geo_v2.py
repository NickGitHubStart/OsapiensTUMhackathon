"""Improved geo-split U-Net training with history/plots and early stopping.

This is a new training entrypoint and does not change existing training scripts.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import deque
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from rasterio.enums import Resampling
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
    label_tile_ids,
)
from src.enhanced_labels import load_tile_labels_enhanced
from src.geo_validation import resolve_train_val_tiles
from src.train_unet import UNetSmall, _masked_bce_logits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class _TileCache:
    def __init__(self, max_items: int = 16) -> None:
        self.max_items = max_items
        self._data: dict[tuple[Path, int, str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._order: deque[tuple[Path, int, str, str]] = deque()

    def get(self, key: tuple[Path, int, str, str]) -> tuple[np.ndarray, np.ndarray] | None:
        return self._data.get(key)

    def set(self, key: tuple[Path, int, str, str], value: tuple[np.ndarray, np.ndarray]) -> None:
        if key in self._data:
            return
        if len(self._order) >= self.max_items:
            old_key = self._order.popleft()
            self._data.pop(old_key, None)
        self._data[key] = value
        self._order.append(key)


class AEFPatchDatasetGeoV2(Dataset):
    def __init__(
        self,
        aef_paths: list[Path],
        data_dir: Path,
        patch_size: int,
        samples_per_epoch: int,
        seed: int,
        min_labeled_frac: float,
        max_tries: int,
        forest_mask_dir: Path | None,
        label_strategy: str,
        aug_flip_rotate_prob: float,
        aug_noise_std: float,
        aug_dropout_prob: float,
        aug_dropout_frac: float,
        aug_scale_min: float,
        aug_scale_max: float,
        aef_downsample: int,
    ) -> None:
        self.aef_paths = aef_paths
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.min_labeled_frac = min_labeled_frac
        self.max_tries = max_tries
        self.forest_mask_dir = forest_mask_dir
        self.label_strategy = label_strategy
        self.aug_flip_rotate_prob = aug_flip_rotate_prob
        self.aug_noise_std = aug_noise_std
        self.aug_dropout_prob = aug_dropout_prob
        self.aug_dropout_frac = aug_dropout_frac
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aef_downsample = max(1, int(aef_downsample))
        self._cache = _TileCache(max_items=16)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_tile(self, aef_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        tile_id, year_str = aef_path.stem.rsplit("_", 1)
        year = int(year_str)
        cache_key = (aef_path, year, self.label_strategy, str(self.forest_mask_dir or ""))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with rasterio.open(aef_path) as src:
            if self.aef_downsample > 1:
                out_h = max(1, src.height // self.aef_downsample)
                out_w = max(1, src.width // self.aef_downsample)
                # Nearest avoids introducing NaNs from nodata interpolation on embeddings.
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
            self.data_dir,
            tile_id,
            ref_profile,
            year=year,
            forest_mask_dir=self.forest_mask_dir,
            label_strategy=self.label_strategy,
        )
        if labels is None:
            return None

        label_mask = build_label_mask(labels)
        if not np.isfinite(aef).all():
            aef = np.nan_to_num(aef, nan=0.0, posinf=0.0, neginf=0.0)
        self._cache.set(cache_key, (aef, label_mask))
        return aef, label_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + idx)

        for _ in range(self.max_tries):
            aef_path = self.aef_paths[int(rng.integers(0, len(self.aef_paths)))]
            loaded = self._load_tile(aef_path)
            if loaded is None:
                continue

            aef, label_mask = loaded
            _, height, width = aef.shape
            if height < self.patch_size or width < self.patch_size:
                continue

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

            valid = label_patch >= 0
            if valid.mean() < self.min_labeled_frac:
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

        raise RuntimeError("Failed to sample a valid patch; try lowering min_labeled_frac.")


def _masked_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs * mask
    targets = targets * mask
    intersection = (probs * targets).sum()
    denom = probs.sum() + targets.sum()
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice


def _collect_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> tuple[float, float, float, float]:
    preds = (torch.sigmoid(logits) >= threshold).float()
    valid = mask > 0.5
    labels_b = labels > 0.5
    preds_b = preds > 0.5

    tp = ((preds_b & labels_b) & valid).sum().item()
    fp = ((preds_b & ~labels_b) & valid).sum().item()
    fn = ((~preds_b & labels_b) & valid).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return precision, recall, f1, iou


def _save_history_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["epoch", "train_loss", "val_loss", "val_precision", "val_recall", "val_f1", "val_iou"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_history_plot(rows: list[dict[str, float]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    val_f1 = [float(r["val_f1"]) for r in rows]
    val_iou = [float(r["val_iou"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train loss")
    axes[0].plot(epochs, val_loss, label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("U-Net loss curves")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="val F1 (class 1)")
    axes[1].plot(epochs, val_iou, label="val IoU (class 1)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Validation segmentation metrics")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Improved U-Net on AEF with fixed train/val tile split and tracked metrics."
    )
    parser.add_argument("--data-dir", default="./data/makeathon-challenge")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=3000)
    parser.add_argument("--val-samples", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-labeled-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bce-weight", type=float, default=0.7)
    parser.add_argument("--dice-weight", type=float, default=0.3)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--model-out", default="./artifacts/unet_aef_geo_v2_last.pt")
    parser.add_argument("--best-model-out", default="./artifacts/unet_aef_geo_v2_best.pt")
    parser.add_argument("--history-csv", default="./artifacts/unet_aef_geo_v2_history.csv")
    parser.add_argument("--history-plot", default="./artifacts/unet_aef_geo_v2_history.png")
    parser.add_argument("--geojson-path", type=Path, default=None)
    parser.add_argument("--val-region", type=str, default="americas")
    parser.add_argument("--val-tiles", type=str, default="")
    parser.add_argument(
        "--label-strategy",
        choices=("multi_source", "glads2_radd"),
        default="multi_source",
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
        default=1,
        help="Read AEF at lower resolution (e.g. 4 or 8) for much faster training.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping (0 disables). Helps prevent NaNs from exploding updates.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bars",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.bce_weight < 0 or args.dice_weight < 0:
        raise ValueError("bce-weight and dice-weight must be >= 0")
    if args.bce_weight + args.dice_weight == 0:
        raise ValueError("At least one of bce-weight or dice-weight must be > 0")

    data_dir = Path(args.data_dir)
    aef_dir = data_dir / "aef-embeddings" / "train"
    model_out = Path(args.model_out)
    best_model_out = Path(args.best_model_out)
    history_csv = Path(args.history_csv)
    history_plot = Path(args.history_plot)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    best_model_out.parent.mkdir(parents=True, exist_ok=True)

    label_tiles = label_tile_ids(data_dir / "labels" / "train")
    geojson = args.geojson_path or (data_dir / "metadata" / "train_tiles.geojson")
    val_tiles_arg: set[str] | None = None
    if args.val_tiles.strip():
        val_tiles_arg = {t.strip() for t in args.val_tiles.split(",") if t.strip()}

    train_tile_ids, val_tile_ids, split_desc = resolve_train_val_tiles(
        label_tiles,
        val_region=args.val_region,
        val_tiles=val_tiles_arg,
        geojson_path=geojson,
        use_random_split=False,
        random_split_fraction=0.2,
        seed=args.seed,
    )
    logger.info(
        "Split (%s): train_tiles=%d val_tiles=%d",
        split_desc,
        len(train_tile_ids),
        len(val_tile_ids),
    )
    logger.info("Val tiles: %s", sorted(val_tile_ids))

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for aef_path in iter_aef_files(aef_dir):
        tile_id, year_str = aef_path.stem.rsplit("_", 1)
        year = int(year_str)
        if year < 2020 or tile_id not in label_tiles:
            continue
        if tile_id in val_tile_ids:
            val_paths.append(aef_path)
        elif tile_id in train_tile_ids:
            train_paths.append(aef_path)

    if not train_paths or not val_paths:
        raise RuntimeError("Need non-empty train_paths and val_paths; adjust val-region or val-tiles.")

    train_ds = AEFPatchDatasetGeoV2(
        aef_paths=train_paths,
        data_dir=data_dir,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        min_labeled_frac=args.min_labeled_frac,
        max_tries=50,
        forest_mask_dir=args.forest_mask_dir,
        label_strategy=args.label_strategy,
        aug_flip_rotate_prob=args.aug_flip_rotate_prob,
        aug_noise_std=args.aug_noise_std,
        aug_dropout_prob=args.aug_dropout_prob,
        aug_dropout_frac=args.aug_dropout_frac,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
        aef_downsample=args.aef_downsample,
    )
    val_ds = AEFPatchDatasetGeoV2(
        aef_paths=val_paths,
        data_dir=data_dir,
        patch_size=args.patch_size,
        samples_per_epoch=args.val_samples,
        seed=args.seed + 123,
        min_labeled_frac=args.min_labeled_frac,
        max_tries=50,
        forest_mask_dir=args.forest_mask_dir,
        label_strategy=args.label_strategy,
        aug_flip_rotate_prob=0.0,
        aug_noise_std=0.0,
        aug_dropout_prob=0.0,
        aug_dropout_frac=args.aug_dropout_frac,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
        aef_downsample=args.aef_downsample,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    sample_aef, _, _ = train_ds[0]
    in_channels = sample_aef.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0
        bad_train_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"train epoch {epoch}/{args.epochs}",
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
            logits = model(features)
            bce = _masked_bce_logits(logits, labels, mask)
            dice = _masked_dice_loss(logits, labels, mask)
            loss = args.bce_weight * bce + args.dice_weight * dice
            if not torch.isfinite(loss):
                bad_train_batches += 1
                continue
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu())
            n_train_batches += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        train_loss = train_loss_sum / max(1, n_train_batches)
        if bad_train_batches:
            logger.warning("Skipped %d bad/empty train batches this epoch", bad_train_batches)

        model.eval()
        val_loss_sum = 0.0
        val_prec_sum = 0.0
        val_rec_sum = 0.0
        val_f1_sum = 0.0
        val_iou_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            vbar = tqdm(
                val_loader,
                desc=f"val epoch {epoch}/{args.epochs}",
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

                logits = model(features)
                bce = _masked_bce_logits(logits, labels, mask)
                dice = _masked_dice_loss(logits, labels, mask)
                loss = args.bce_weight * bce + args.dice_weight * dice
                if not torch.isfinite(loss):
                    continue
                val_loss_sum += float(loss.detach().cpu())

                p, r, f1, iou = _collect_metrics(logits, labels, mask, args.threshold)
                val_prec_sum += p
                val_rec_sum += r
                val_f1_sum += f1
                val_iou_sum += iou
                n_val_batches += 1
                vbar.set_postfix(loss=float(loss.detach().cpu()))

        val_loss = val_loss_sum / max(1, n_val_batches)
        val_precision = val_prec_sum / max(1, n_val_batches)
        val_recall = val_rec_sum / max(1, n_val_batches)
        val_f1 = val_f1_sum / max(1, n_val_batches)
        val_iou = val_iou_sum / max(1, n_val_batches)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
                "val_f1": float(val_f1),
                "val_iou": float(val_iou),
            }
        )

        logger.info(
            "Epoch %d/%d - train_loss %.4f - val_loss %.4f - val_f1 %.4f - val_iou %.4f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_f1,
            val_iou,
        )

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), best_model_out)
            logger.info("New best model at epoch %d (val_f1=%.4f) -> %s", epoch, val_f1, best_model_out)
        else:
            patience_ctr += 1
            if args.early_stop_patience > 0 and patience_ctr >= args.early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (best epoch=%d, best val_f1=%.4f)",
                    epoch,
                    best_epoch,
                    best_f1,
                )
                break

    torch.save(model.state_dict(), model_out)
    logger.info("Saved last model to %s", model_out)
    _save_history_csv(history, history_csv)
    logger.info("Saved training history to %s", history_csv)
    _save_history_plot(history, history_plot)
    logger.info("Saved training plot to %s", history_plot)


if __name__ == "__main__":
    main()
