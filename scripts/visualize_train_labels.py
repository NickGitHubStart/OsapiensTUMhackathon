"""Show Sentinel-2 RGB next to the weak training label mask used in this repo.

The mask is **not** "forest vs non-forest" from AlphaEarth. It is built from
GLAD-S2 + RADD (and optionally GLAD-L / forest mask) the same way as
``src.data_utils.load_tile_labels`` / ``src.enhanced_labels.load_tile_labels_enhanced``:

- ``1`` = weak **deforestation** supervision (positive)
- ``0`` = weak **no alert** supervision (negative)
- ``-1`` = neither (pixels typically ignored by training)

Run from repo root, after downloading data::

    python -m scripts.visualize_train_labels --data-dir ./data/makeathon-challenge

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio

# Allow ``python scripts/visualize_train_labels.py`` without installing package
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_utils import build_label_mask, label_tile_ids, load_tile_labels
from src.enhanced_labels import load_tile_labels_enhanced


def _percentile_norm(band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    band = band.astype(np.float32)
    valid = np.isfinite(band)
    if not np.any(valid):
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(band[valid], (p_low, p_high))
    if hi <= lo:
        return np.zeros_like(band, dtype=np.float32)
    out = (band - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _pick_s2_scene(tile_id: str, s2_dir: Path, scene_path: Path | None) -> Path:
    if scene_path is not None:
        return scene_path
    folder = s2_dir / f"{tile_id}__s2_l2a"
    scenes = sorted(folder.glob("*.tif"))
    if not scenes:
        raise FileNotFoundError(f"No Sentinel-2 scenes under {folder}")
    return scenes[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot S2 RGB + weak deforestation training label mask."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/makeathon-challenge"),
        help="Dataset root (contains sentinel-2/, labels/, …)",
    )
    parser.add_argument(
        "--tile-id",
        type=str,
        default="",
        help="Tile id (e.g. 18NWG_6_6). Default: first tile that has train labels.",
    )
    parser.add_argument(
        "--s2-path",
        type=Path,
        default=None,
        help="Optional explicit path to one Sentinel-2 monthly GeoTIFF.",
    )
    parser.add_argument(
        "--label-strategy",
        choices=("glads2_radd", "multi_source"),
        default="glads2_radd",
        help="Same strategies as training helpers (multi_source uses GLAD-L if present).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Year for GLAD-L rasters when label-strategy=multi_source.",
    )
    parser.add_argument(
        "--forest-mask-dir",
        type=Path,
        default=None,
        help="Optional directory with per-tile forest-in-2020 masks (see README / enhanced_labels).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If set, write figure to this path instead of opening a window.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    labels_dir = data_dir / "labels" / "train"
    s2_dir = data_dir / "sentinel-2" / "train"

    tile_id = args.tile_id.strip()
    if not tile_id:
        ids = sorted(label_tile_ids(labels_dir))
        if not ids:
            raise SystemExit(f"No train labels found under {labels_dir}")
        tile_id = ids[0]
        print(f"Using tile-id={tile_id!r} (first with GLAD-S2 + RADD files)")

    s2_path = _pick_s2_scene(tile_id, s2_dir, args.s2_path)

    with rasterio.open(s2_path) as src:
        ref_profile = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
        }
        red = src.read(4).astype(np.float32)
        green = src.read(3).astype(np.float32)
        blue = src.read(2).astype(np.float32)

    rgb = np.stack(
        [_percentile_norm(red), _percentile_norm(green), _percentile_norm(blue)],
        axis=-1,
    )

    if args.label_strategy == "glads2_radd":
        labels = load_tile_labels(data_dir, tile_id, ref_profile)
    else:
        labels = load_tile_labels_enhanced(
            data_dir,
            tile_id,
            ref_profile,
            year=args.year,
            forest_mask_dir=args.forest_mask_dir,
            label_strategy="multi_source",
        )

    if labels is None:
        raise SystemExit(f"No label rasters for tile {tile_id!r}")

    mask = build_label_mask(labels)

    cmap = mcolors.ListedColormap(["#2d2d2d", "#4c72b0", "#c44e52"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Sentinel-2 RGB\n{tile_id} — {s2_path.name}", fontsize=10)
    axes[0].axis("off")

    im = axes[1].imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title(
        "Weak training mask (same rules as src/data_utils)\n"
        "red = deforestation supervision, blue = no-alert supervision, dark = unused",
        fontsize=10,
    )
    axes[1].axis("off")
    cbar = fig.colorbar(
        im,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        ticks=[-1, 0, 1],
    )
    cbar.ax.set_yticklabels(["unused (-1)", "negative (0)", "positive (1)"])

    fig.suptitle(
        f"Label strategy: {args.label_strategy}"
        + (f", year={args.year}" if args.label_strategy == "multi_source" else ""),
        fontsize=11,
    )
    fig.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
