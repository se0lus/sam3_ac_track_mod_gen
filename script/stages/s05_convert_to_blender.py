"""Stage 5: Merge clip masks (image-based) and convert to Blender coordinates.

Instead of converting each clip individually and consolidating, this stage:
  1. Rasterizes all per-clip polygons onto a shared canvas (per tag)
  2. Extracts clean, overlap-free contours via cv2.findContours
  3. Converts the merged geo_xy polygons to Blender local coordinates
  4. Outputs one *_blender.json per tag (ready for Stage 6)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from typing import List

logger = logging.getLogger("sam3_pipeline.s05")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 5: Merge clip masks and convert to Blender input.

    Reads ``*_masks.json`` from ``config.mask_on_clips_dir`` (stage 4 output),
    merges overlapping clip masks via rasterization, then converts merged
    polygons to Blender coordinates and writes per-tag JSON files to
    ``config.blender_clips_dir``.

    Surface tags (sand, grass, road2, road, kerb) use priority compositing
    to eliminate inter-tag gaps. Non-surface tags (trees, building, water)
    use independent per-tag merging.
    """
    logger.info("=== Stage 5: Convert masks to Blender input (image-based merge) ===")

    if not config.tiles_dir:
        raise ValueError("tiles_dir is required for convert_to_blender stage")
    if not config.geotiff_path:
        raise ValueError("geotiff_path is required for convert_to_blender stage")

    mask_dir = config.mask_on_clips_dir
    if not os.path.isdir(mask_dir):
        raise ValueError(
            f"mask_on_clips_dir not found: {mask_dir}. Run stage 4 first."
        )

    output_dir = config.blender_clips_dir

    # Clean old output to avoid mixing stale per-clip files with new merged ones
    if os.path.isdir(output_dir):
        logger.info("Cleaning old output: %s", output_dir)
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Surface tags: priority compositing (low→high priority order)
    composite_priority = [
        {"tag": "sand",  "clip": True,  "stage2_mask": "sand_mask.png"},
        {"tag": "grass", "clip": True,  "stage2_mask": "grass_mask.png"},
        {"tag": "road2", "clip": False, "stage2_masks": ["merged_mask.png", "concrete_mask.png"]},
        {"tag": "road",  "clip": True,  "stage2a": True},
        {"tag": "kerb",  "clip": True},
    ]

    # Independent tags = all sam3_prompts tags minus composite tags minus concrete
    # (concrete is absorbed into road2)
    composite_tags = {c["tag"] for c in composite_priority}
    all_tags = [p["tag"] for p in config.sam3_prompts]
    independent_tags = [t for t in all_tags if t not in composite_tags and t != "concrete"]

    logger.info("Composite surface tags: %s", [c["tag"] for c in composite_priority])
    logger.info("Independent tags: %s", independent_tags)

    # Stage 2, 2a, and 5a directories
    fullmap_mask_dir = config.mask_full_map_dir
    layout_mask_dir = config.stage_dir("track_layouts")
    manual_surface_dir = config.manual_surface_masks_dir

    _merge_and_convert(
        geotiff_path=config.geotiff_path,
        mask_dir=mask_dir,
        tiles_dir=config.tiles_dir,
        output_dir=output_dir,
        tags=independent_tags,
        fullmap_mask_dir=fullmap_mask_dir,
        layout_mask_dir=layout_mask_dir,
        manual_surface_mask_dir=manual_surface_dir,
        composite_priority=composite_priority,
    )
    logger.info("Blender input files written to %s", output_dir)


def _merge_and_convert(
    geotiff_path: str,
    mask_dir: str,
    tiles_dir: str,
    output_dir: str,
    tags: List[str],
    fullmap_mask_dir: str = None,
    layout_mask_dir: str = None,
    manual_surface_mask_dir: str = None,
    composite_priority: List[dict] = None,
) -> None:
    """Merge clip masks via rasterization, then convert to Blender coordinates."""
    from mask_merger import merge_clip_masks
    from geo_sam3_blender_utils import get_tileset_transform, geo_points_to_blender_xyz

    # Step 1: Merge clip masks per tag (with optional priority compositing)
    preview_dir = os.path.join(output_dir, "merge_preview")
    merged = merge_clip_masks(
        geotiff_path=geotiff_path,
        mask_dir=mask_dir,
        tags=tags,
        fullmap_mask_dir=fullmap_mask_dir,
        layout_mask_dir=layout_mask_dir,
        manual_surface_mask_dir=manual_surface_mask_dir,
        composite_priority=composite_priority,
        preview_dir=preview_dir,
    )

    if not merged:
        logger.warning("No merged results produced")
        return

    # Step 2: Get tileset transform info (one-time)
    # Use the first available polygon as sample point for tileset selection
    sample_geo = None
    for groups in merged.values():
        for group in groups:
            for poly in group.get("include", []):
                if poly and len(poly) > 0:
                    sample_geo = (poly[0][0], poly[0][1])
                    break
            if sample_geo:
                break
        if sample_geo:
            break

    tf_info = get_tileset_transform(tiles_dir, sample_geo_xy=sample_geo, frame_mode="auto")

    # Step 3: Convert each tag's pre-triangulated groups to Blender coords.
    # Each group has earcut-triangulated "vertices" + "faces", so the Blender
    # script can create mesh directly without curve fill.
    for tag, groups in merged.items():
        mesh_groups = []
        total_verts = 0
        total_faces = 0

        for group_idx, group in enumerate(groups):
            verts_geo = group.get("vertices")
            faces = group.get("faces")
            if verts_geo is None or faces is None:
                logger.warning("  %s group %d: no triangulation, skipping", tag, group_idx)
                continue

            # Convert all vertices to Blender coordinates
            points_xyz = geo_points_to_blender_xyz(verts_geo, tf_info, z_mode="zero")
            if len(points_xyz) < 3:
                continue

            mesh_groups.append({
                "group_index": group_idx,
                "tag": tag,
                "points_xyz": points_xyz,
                "faces": faces,
                "geo_xy": verts_geo,
            })
            total_verts += len(points_xyz)
            total_faces += len(faces)

        result = {
            "origin": {
                "ecef": list(tf_info.origin_ecef),
                "lonlat": [tf_info.origin_lon, tf_info.origin_lat],
                "h": tf_info.origin_h,
                "source": tf_info.origin_src,
            },
            "frame": {
                "mode": tf_info.effective_mode,
                "tileset_transform_source": tf_info.tf_source,
            },
            "source_tag": tag,
            "mesh_groups": mesh_groups,
        }

        # Save to tag subdirectory (Stage 6 recursively walks for *_blender.json)
        tag_dir = os.path.join(output_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)
        out_path = os.path.join(tag_dir, f"{tag}_merged_blender.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(
            "  %s: %d groups, %d verts, %d faces -> %s",
            tag, len(mesh_groups), total_verts, total_faces, out_path,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Stage 5: Convert masks to Blender input (image-based merge)")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF file")
    p.add_argument("--tiles-dir", required=True, help="Directory with tileset.json")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(
        geotiff_path=args.geotiff,
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
    ).resolve()
    run(config)
