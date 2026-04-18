"""Predict deforestation within a polygon and compare with consensus labels."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely import wkt

from src.data_utils import build_label_mask, load_tile_labels

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def _predict_pixel_xgb(aef: np.ndarray, model) -> np.ndarray:
	channels, height, width = aef.shape
	flat = aef.reshape(channels, height * width).transpose(1, 0)
	proba = model.predict_proba(flat)[:, 1]
	return proba.reshape(height, width)


def _predict_patch_xgb(
	aef: np.ndarray, model, patch_size: int, stride: int
) -> np.ndarray:
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


def _predict_unet(
	aef: np.ndarray, model_path: Path, patch_size: int, stride: int
) -> np.ndarray:
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


def _load_polygon(
	polygon_geojson: str | None,
	polygon_wkt: str | None,
	polygon_crs: str,
) -> gpd.GeoDataFrame:
	if polygon_geojson:
		gdf = gpd.read_file(polygon_geojson)
		if gdf.crs is None:
			gdf = gdf.set_crs(polygon_crs)
		return gdf

	if polygon_wkt:
		geom = wkt.loads(polygon_wkt)
		return gpd.GeoDataFrame({"geometry": [geom]}, crs=polygon_crs)

	raise ValueError("Provide --polygon-geojson or --polygon-wkt")


def _resolve_tile_id(data_dir: Path, polygon: gpd.GeoDataFrame) -> str:
	for name in ["train_tiles.geojson", "test_tiles.geojson"]:
		path = data_dir / "metadata" / name
		if not path.exists():
			continue
		tiles = gpd.read_file(path)
		if tiles.crs is None:
			tiles = tiles.set_crs("EPSG:4326")
		poly = polygon.to_crs(tiles.crs)
		hits = tiles[tiles.intersects(poly.geometry.iloc[0])]
		if not hits.empty:
			return str(hits.iloc[0]["name"])
	raise FileNotFoundError("Polygon does not intersect any tile in metadata.")


def _rasterize_polygon(polygon: gpd.GeoDataFrame, ref_profile: dict) -> np.ndarray:
	poly = polygon.to_crs(ref_profile["crs"])
	return rasterize(
		[(poly.geometry.iloc[0], 1)],
		out_shape=(ref_profile["height"], ref_profile["width"]),
		transform=ref_profile["transform"],
		fill=0,
		dtype="uint8",
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Predict deforestation within a polygon and compare with consensus labels."
	)
	parser.add_argument("--data-dir", default="/content/drive/MyDrive/makeathon-challenge")
	parser.add_argument("--model-path", required=True)
	parser.add_argument(
		"--model-type",
		choices=["aef_xgb", "patch_xgb", "unet"],
		default="aef_xgb",
	)
	parser.add_argument("--tile-id", default="")
	parser.add_argument("--year", type=int, default=2020)
	parser.add_argument("--month", type=int, default=0)
	parser.add_argument("--polygon-geojson", default="")
	parser.add_argument("--polygon-wkt", default="")
	parser.add_argument("--polygon-crs", default="EPSG:4326")
	parser.add_argument("--patch-size", type=int, default=64)
	parser.add_argument("--stride", type=int, default=64)
	parser.add_argument("--threshold", type=float, default=0.5)
	parser.add_argument("--out-dir", default="./outputs")

	args = parser.parse_args()
	data_dir = Path(args.data_dir)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	polygon = _load_polygon(
		args.polygon_geojson or None,
		args.polygon_wkt or None,
		args.polygon_crs,
	)

	tile_id = args.tile_id or _resolve_tile_id(data_dir, polygon)
	if args.month:
		logger.info(
			"Month=%s is ignored for AEF models; using year=%s",
			args.month,
			args.year,
		)

	aef_path = data_dir / "aef-embeddings" / "train" / f"{tile_id}_{args.year}.tiff"
	if not aef_path.exists():
		aef_path = data_dir / "aef-embeddings" / "test" / f"{tile_id}_{args.year}.tiff"
	if not aef_path.exists():
		raise FileNotFoundError(f"AEF file not found for {tile_id} {args.year}")

	with rasterio.open(aef_path) as src:
		aef = src.read().astype(np.float32)
		ref_profile = src.profile

	polygon_mask = _rasterize_polygon(polygon, ref_profile)

	if args.model_type == "aef_xgb":
		model = joblib.load(args.model_path)
		proba = _predict_pixel_xgb(aef, model)
	elif args.model_type == "patch_xgb":
		model = joblib.load(args.model_path)
		proba = _predict_patch_xgb(aef, model, args.patch_size, args.stride)
	else:
		proba = _predict_unet(aef, Path(args.model_path), args.patch_size, args.stride)

	pred_mask = (proba > args.threshold).astype(np.uint8)
	pred_mask[polygon_mask == 0] = 0

	labels = load_tile_labels(data_dir, tile_id, ref_profile)
	if labels is None:
		raise FileNotFoundError("Labels not found for tile; cannot build GLAD/RADD mask.")

	gt_mask = build_label_mask(labels)
	gt_mask = gt_mask.copy()
	gt_mask[polygon_mask == 0] = -1

	rgb = _make_rgb(aef)

	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	axes[0].imshow(rgb)
	axes[0].set_title(f"AEF RGB | {tile_id} {args.year}")
	axes[0].axis("off")

	axes[1].imshow(rgb)
	axes[1].imshow(gt_mask, cmap="gray", vmin=-1, vmax=1, alpha=0.55)
	axes[1].set_title("GLAD/RADD Consensus (pos/neg)")
	axes[1].axis("off")

	axes[2].imshow(rgb)
	axes[2].imshow(pred_mask, cmap="Reds", alpha=0.45)
	axes[2].set_title("Prediction Overlay")
	axes[2].axis("off")

	fig.tight_layout()
	out_path = out_dir / f"{tile_id}_{args.year}_polygon_compare.png"
	fig.savefig(out_path, dpi=200)
	logger.info("Saved comparison to %s", out_path)


if __name__ == "__main__":
	main()
