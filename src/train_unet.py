"""Train a small U-Net on AlphaEarth embeddings using patch sampling."""

from __future__ import annotations

import argparse
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class TileLabels:
    positive: np.ndarray
    negative: np.ndarray


def _reproject_to_match(src_path: Path, ref_profile: dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.uint16)
        reproject(
            source=src.read(1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
        )
        return dst


def _load_tile_labels(data_dir: Path, tile_id: str, ref_profile: dict) -> TileLabels | None:
    labels_dir = data_dir / "labels" / "train"

    glads2_alert = labels_dir / "glads2" / f"glads2_{tile_id}_alert.tif"
    glads2_date = labels_dir / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    radd_labels = labels_dir / "radd" / f"radd_{tile_id}_labels.tif"

    if not (glads2_alert.exists() and glads2_date.exists() and radd_labels.exists()):
        logger.warning("Missing labels for %s, skipping", tile_id)
        return None

    glads2_alert_r = _reproject_to_match(glads2_alert, ref_profile)
    glads2_date_r = _reproject_to_match(glads2_date, ref_profile)
    radd_r = _reproject_to_match(radd_labels, ref_profile)

    glads2_date = np.datetime64("2019-01-01") + glads2_date_r.astype("timedelta64[D]")
    glads2_pos = (glads2_alert_r >= 2) & (glads2_date >= np.datetime64("2020-01-01"))

    radd_conf = radd_r // 10000
    radd_days = radd_r % 10000
    radd_date = np.datetime64("2014-12-31") + radd_days.astype("timedelta64[D]")
    radd_pos = (radd_conf >= 2) & (radd_date >= np.datetime64("2020-01-01"))

    positive = glads2_pos & radd_pos
    negative = (glads2_alert_r == 0) & (radd_r == 0)

    return TileLabels(positive=positive, negative=negative)


def _iter_aef_files(aef_dir: Path) -> Iterable[Path]:
    yield from sorted(aef_dir.glob("*.tiff"))


def _label_tile_ids(labels_dir: Path) -> set[str]:
    glads2_alert = {
        p.name.replace("glads2_", "").replace("_alert.tif", "")
        for p in (labels_dir / "glads2").glob("glads2_*_alert.tif")
    }
    glads2_date = {
        p.name.replace("glads2_", "").replace("_alertDate.tif", "")
        for p in (labels_dir / "glads2").glob("glads2_*_alertDate.tif")
    }
    radd = {
        p.name.replace("radd_", "").replace("_labels.tif", "")
        for p in (labels_dir / "radd").glob("radd_*_labels.tif")
    }
    return glads2_alert & glads2_date & radd


def _build_label_mask(labels: TileLabels) -> np.ndarray:
    mask = np.full(labels.positive.shape, -1, dtype=np.int8)
    mask[labels.negative] = 0
    mask[labels.positive] = 1
    return mask


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

        labels = _load_tile_labels(self.data_dir, tile_id, ref_profile)
        if labels is None:
            return None

        label_mask = _build_label_mask(labels)
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

    label_tiles = _label_tile_ids(data_dir / "labels" / "train")
    logger.info("Label tiles available: %d", len(label_tiles))

    aef_paths: list[Path] = []
    for aef_path in _iter_aef_files(aef_dir):
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
