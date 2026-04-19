"""
Post-process existing baseline4 caches to add/overwrite per-pixel weights using new consensus confidence rules.
Usage: python postprocess_add_weights.py --cache-dir ... --data-dir ...
"""
import argparse
from pathlib import Path
import numpy as np
import logging
from src.build_cache_baseline4 import _load_consensus_labels
import rasterio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Add/overwrite weights in baseline4 caches.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    data_dir = Path(args.data_dir)
    split = args.split

    npz_files = sorted(cache_dir.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} cache files in {cache_dir}")

    for npz_path in npz_files:
        tile_id = npz_path.stem
        # Load cache
        arr = np.load(npz_path)
        label = arr["label"]
        # Try to get ref_profile from features shape
        height, width = label.shape
        # Build a fake ref_profile for _load_consensus_labels
        # (Assume all tiles have the same CRS/transform; not used for confidence math)
        ref_profile = {"height": height, "width": width, "crs": None, "transform": rasterio.Affine.identity()}
        result = _load_consensus_labels(data_dir, tile_id, ref_profile)
        if result is None:
            logger.warning(f"No label/confidence for {tile_id}, skipping")
            continue
        _, weight = result
        # Overwrite/add weight
        out_dict = dict(arr)
        out_dict["weight"] = weight.astype(np.float32)
        np.savez_compressed(npz_path, **out_dict)
        logger.info(f"Updated weights for {tile_id}")

if __name__ == "__main__":
    main()
