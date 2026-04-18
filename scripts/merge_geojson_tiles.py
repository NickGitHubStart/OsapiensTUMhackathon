"""Merge per-tile GeoJSONs into a single submission file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED_TYPES = {"Polygon", "MultiPolygon"}


def _clean_properties(props: dict | None, keep_time_step: bool) -> dict:
    if not isinstance(props, dict):
        props = {}
    if keep_time_step and "time_step" in props:
        return {"time_step": props["time_step"]}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-tile GeoJSONs into a submission FeatureCollection.")
    parser.add_argument("--in-dir", required=True, help="Directory with per-tile GeoJSON files.")
    parser.add_argument(
        "--pattern",
        default="pred_*.geojson",
        help="Glob pattern for per-tile GeoJSON files (default: pred_*.geojson).",
    )
    parser.add_argument("--out-file", default="submission.geojson", help="Output GeoJSON path.")
    parser.add_argument(
        "--keep-time-step",
        action="store_true",
        help="Preserve properties.time_step if present.",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    tile_files = sorted(in_dir.glob(args.pattern))
    if not tile_files:
        raise SystemExit(f"No files found in {in_dir} matching {args.pattern}")

    merged = {"type": "FeatureCollection", "features": []}

    for tile_geojson in tile_files:
        with tile_geojson.open("r", encoding="utf-8") as handle:
            gj = json.load(handle)

        for feature in gj.get("features", []):
            geometry = feature.get("geometry") or {}
            if geometry.get("type") not in ALLOWED_TYPES:
                continue
            feature["properties"] = _clean_properties(feature.get("properties"), args.keep_time_step)
            merged["features"].append(feature)

    assert merged["type"] == "FeatureCollection"
    assert all(
        feature.get("geometry", {}).get("type") in ALLOWED_TYPES
        for feature in merged["features"]
    )
    assert all(isinstance(feature.get("properties"), dict) for feature in merged["features"])

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle)

    print(f"Merged {len(merged['features'])} features from {len(tile_files)} tiles")


if __name__ == "__main__":
    main()
