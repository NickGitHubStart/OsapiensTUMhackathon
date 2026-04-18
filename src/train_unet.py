"""Train a small U-Net on AlphaEarth embeddings using patch sampling."""

from __future__ import annotations

import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from src.data_utils import build_label_mask, iter_aef_files, label_tile_ids, load_tile_labels

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class _TileCache:
    def __init__(self, max_items: int = 4) -> None:
        self.max_items = max_items
        self._data: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self._order: deque[Path] = deque()

    def get(self, key: Path) -> tuple[np.ndarray, np.ndarray] | None:
        return self._data.get(key)

    def set(self, key: Path, value: tuple[np.ndarray, np.ndarray]) -> None:
        if key in self._data:
            return
        if len(self._order) >= self.max_items:
            old_key = self._order.popleft()
            self._data.pop(old_key, None)
        self._data[key] = value
        self._order.append(key)


class AEFPatchDataset(Dataset):
    def __init__(
        self,
        aef_paths: list[Path],
        data_dir: Path,
        patch_size: int,
        samples_per_epoch: int,
        seed: int,
        min_labeled_frac: float,
        max_tries: int,
    ) -> None:
        self.aef_paths = aef_paths
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.min_labeled_frac = min_labeled_frac
        self.max_tries = max_tries
        self._cache = _TileCache(max_items=4)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_tile(self, aef_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        cached = self._cache.get(aef_path)
        if cached is not None:
            return cached

        tile_id, _ = aef_path.stem.rsplit("_", 1)
        with rasterio.open(aef_path) as src:
            aef = src.read().astype(np.float32)
            ref_profile = src.profile

        labels = load_tile_labels(self.data_dir, tile_id, ref_profile)
        if labels is None:
            return None

        label_mask = build_label_mask(labels)
        self._cache.set(aef_path, (aef, label_mask))
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

            y = int(rng.integers(0, height - self.patch_size + 1))
            x = int(rng.integers(0, width - self.patch_size + 1))

            label_patch = label_mask[y : y + self.patch_size, x : x + self.patch_size]
            valid = label_patch >= 0
            if valid.mean() < self.min_labeled_frac:
                continue

            features = aef[:, y : y + self.patch_size, x : x + self.patch_size]
            labels = label_patch.astype(np.float32)
            labels[labels < 0] = 0.0
            mask = valid.astype(np.float32)

            return (
                torch.from_numpy(features),
                torch.from_numpy(labels[None, ...]),
                torch.from_numpy(mask[None, ...]),
            )

        raise RuntimeError("Failed to sample a valid patch; try lowering min_labeled_frac.")


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()

        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.bottleneck = self._block(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._block(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))

        dec2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.out(dec1)


def _masked_bce_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small U-Net on AlphaEarth embeddings using patch sampling."
    )
    parser.add_argument(
        "--data-dir",
        default="./data/makeathon-challenge",
        help="Path to the downloaded dataset root",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Square patch size in pixels",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=2000,
        help="Number of random patches per epoch",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=400,
        help="Number of validation patches per epoch",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--min-labeled-frac",
        type=float,
        default=0.1,
        help="Minimum labeled fraction inside a patch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed",
    )
    parser.add_argument(
        "--model-out",
        default="./artifacts/unet_aef.pt",
        help="Where to save the trained model",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    aef_dir = data_dir / "aef-embeddings" / "train"
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    label_tiles = label_tile_ids(data_dir / "labels" / "train")
    logger.info("Label tiles available: %d", len(label_tiles))

    aef_paths: list[Path] = []
    for aef_path in iter_aef_files(aef_dir):
        tile_id, year_str = aef_path.stem.rsplit("_", 1)
        year = int(year_str)
        if year < 2020:
            continue
        if tile_id not in label_tiles:
            continue
        aef_paths.append(aef_path)

    if not aef_paths:
        raise RuntimeError("No AEF tiles with labels were found.")

    train_ds = AEFPatchDataset(
        aef_paths=aef_paths,
        data_dir=data_dir,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        min_labeled_frac=args.min_labeled_frac,
        max_tries=50,
    )
    val_ds = AEFPatchDataset(
        aef_paths=aef_paths,
        data_dir=data_dir,
        patch_size=args.patch_size,
        samples_per_epoch=args.val_samples,
        seed=args.seed + 123,
        min_labeled_frac=args.min_labeled_frac,
        max_tries=50,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    sample_aef, _, _ = train_ds[0]
    in_channels = sample_aef.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for features, labels, mask in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = _masked_bce_logits(logits, labels, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels, mask in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                logits = model(features)
                loss = _masked_bce_logits(logits, labels, mask)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        logger.info("Epoch %d/%d - train %.4f - val %.4f", epoch, args.epochs, train_loss, val_loss)

    torch.save(model.state_dict(), model_out)
    logger.info("Saved model to %s", model_out)


if __name__ == "__main__":
    main()
