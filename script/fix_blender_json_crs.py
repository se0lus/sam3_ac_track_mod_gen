"""One-time CRS migration: convert blender JSON geo_xy from native CRS to WGS84.

Fixes the coordinate mismatch where geo_xy was stored in the GeoTIFF's native
CRS (e.g. EPSG:32649 UTM) but downstream code expects WGS84 (EPSG:4326).

Also recomputes points_xyz (Blender tileset-local coordinates) from the
corrected WGS84 geo_xy using the tileset transform.

Usage:
    python fix_blender_json_crs.py --geotiff <path> --tiles-dir <path> --output-dir output
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("fix_blender_json_crs")

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)


def _is_likely_geographic(geo_xy: list) -> bool:
    """Heuristic: if all coordinates are in [-180,360] x [-90,90], likely WGS84."""
    for pt in geo_xy[:20]:  # sample first 20 points
        if len(pt) < 2:
            continue
        x, y = float(pt[0]), float(pt[1])
        if not (-360 <= x <= 360 and -90 <= y <= 90):
            return False
    return True


def convert_all_blender_jsons(
    geotiff_path: str,
    tiles_dir: str,
    output_dir: str,
    *,
    dry_run: bool = False,
) -> int:
    """Convert all *_merged_blender.json files from native CRS to WGS84.

    Returns the number of files converted.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from geo_sam3_blender_utils import get_tileset_transform, geo_points_to_blender_xyz

    # --- 1. Check GeoTIFF CRS ---
    with rasterio.open(geotiff_path) as ds:
        if ds.crs.is_geographic:
            log.info("GeoTIFF CRS is already geographic (%s), no conversion needed.", ds.crs)
            return 0
        src_crs = ds.crs
        native_bounds = {
            "left": ds.bounds.left, "right": ds.bounds.right,
            "top": ds.bounds.top, "bottom": ds.bounds.bottom,
        }

    log.info("Source CRS: %s (projected → need conversion to WGS84)", src_crs)

    # --- 2. Get correct tileset transform using WGS84 center ---
    # Convert GeoTIFF center to WGS84 for sample_geo_xy
    cx_native = (native_bounds["left"] + native_bounds["right"]) / 2
    cy_native = (native_bounds["top"] + native_bounds["bottom"]) / 2
    lons, lats = warp_transform(src_crs, "EPSG:4326", [cx_native], [cy_native])
    sample_geo_xy = (lons[0], lats[0])
    log.info("WGS84 center: lon=%.6f, lat=%.6f", sample_geo_xy[0], sample_geo_xy[1])

    tf_info = get_tileset_transform(tiles_dir, sample_geo_xy=sample_geo_xy, frame_mode="auto")
    log.info("Tileset transform: mode=%s, source=%s", tf_info.effective_mode, tf_info.tf_source)
    log.info("Origin: lon=%.6f, lat=%.6f, h=%.2f", tf_info.origin_lon, tf_info.origin_lat, tf_info.origin_h)

    # --- 3. Find all blender JSONs (skip backup dirs) ---
    json_files = []
    for root, dirs, files in os.walk(output_dir):
        # Prune backup directories from traversal
        dirs[:] = [d for d in dirs if not d.startswith("_backup")]
        for fname in files:
            if fname.endswith("_blender.json"):
                json_files.append(os.path.join(root, fname))

    log.info("Found %d *_blender.json files in %s", len(json_files), output_dir)

    # --- 4. Convert each file ---
    converted = 0
    skipped_geographic = 0
    skipped_empty = 0

    for jp in sorted(json_files):
        rel_path = os.path.relpath(jp, output_dir)

        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        mesh_groups = data.get("mesh_groups", [])
        if not mesh_groups:
            skipped_empty += 1
            continue

        # Check if geo_xy is already in WGS84 (skip if so)
        sample_mg = next((mg for mg in mesh_groups if mg.get("geo_xy")), None)
        if sample_mg and _is_likely_geographic(sample_mg["geo_xy"]):
            log.info("  SKIP (already WGS84): %s", rel_path)
            skipped_geographic += 1
            continue

        # Convert
        modified = False
        for mg in mesh_groups:
            geo_xy = mg.get("geo_xy", [])
            if not geo_xy:
                continue

            # Convert geo_xy from native CRS to WGS84
            xs = [float(pt[0]) for pt in geo_xy]
            ys = [float(pt[1]) for pt in geo_xy]
            new_lons, new_lats = warp_transform(src_crs, "EPSG:4326", xs, ys)
            new_geo_xy = [[lon, lat] for lon, lat in zip(new_lons, new_lats)]

            # Recompute points_xyz with correct WGS84 coordinates
            new_xyz = geo_points_to_blender_xyz(new_geo_xy, tf_info, z_mode="zero")

            mg["geo_xy"] = new_geo_xy
            mg["points_xyz"] = new_xyz
            modified = True

        if modified:
            # Update origin/frame metadata
            data["origin"] = {
                "ecef": list(tf_info.origin_ecef),
                "lonlat": [tf_info.origin_lon, tf_info.origin_lat],
                "h": tf_info.origin_h,
                "source": tf_info.origin_src,
            }
            data["frame"] = {
                "mode": tf_info.effective_mode,
                "tileset_transform_source": tf_info.tf_source,
            }

            if not dry_run:
                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            converted += 1
            # Show a sample coordinate for verification
            sample = mesh_groups[0]["geo_xy"][0] if mesh_groups[0].get("geo_xy") else None
            sample_xyz = mesh_groups[0]["points_xyz"][0] if mesh_groups[0].get("points_xyz") else None
            log.info(
                "  CONVERTED: %s  (sample geo_xy=%s, xyz=%s)",
                rel_path,
                [round(v, 6) for v in sample] if sample else "?",
                [round(v, 3) for v in sample_xyz] if sample_xyz else "?",
            )

    log.info("=== Summary ===")
    log.info("  Converted: %d files", converted)
    log.info("  Skipped (already WGS84): %d files", skipped_geographic)
    log.info("  Skipped (no mesh_groups): %d files", skipped_empty)
    if dry_run:
        log.info("  (DRY RUN — no files written)")

    return converted


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fix blender JSON CRS: native → WGS84")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF")
    p.add_argument("--tiles-dir", required=True, help="Directory with tileset.json")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    p.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = p.parse_args()

    convert_all_blender_jsons(
        args.geotiff, args.tiles_dir, args.output_dir,
        dry_run=args.dry_run,
    )
