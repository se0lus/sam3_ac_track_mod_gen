"""
Merge tile results into final blend file.

Usage:
    blender --background --python merge_tiles.py -- \\
        --base tiles_loaded.blend \\
        --tiles tile1.blend,tile2.blend,... \\
        --output final_track.blend
"""
from __future__ import annotations

import argparse
import os
import sys

import bpy  # type: ignore[import-not-found]
import bmesh  # type: ignore[import-not-found]


def _get_script_argv():
    """Extract arguments after '--' in sys.argv."""
    try:
        idx = sys.argv.index("--")
        return sys.argv[idx + 1:]
    except ValueError:
        return []


def merge_tile_results(
    base_blend: str,
    tile_blends: list[str],
    output_blend: str
) -> None:
    """Merge all tile blend files into base blend."""
    print(f"Opening base blend: {base_blend}")
    bpy.ops.wm.open_mainfile(filepath=base_blend)

    # Append all collision collections from tiles
    for tile_blend in tile_blends:
        if not os.path.exists(tile_blend):
            print(f"Warning: tile blend not found: {tile_blend}")
            continue

        try:
            print(f"Appending from: {tile_blend}")
            with bpy.data.libraries.load(tile_blend, link=False) as (data_from, data_to):
                cols = [c for c in data_from.collections if c.startswith("collision_")]
                data_to.collections = cols

            # Link appended collections to scene
            for col in data_to.collections:
                if col and col.name not in bpy.context.scene.collection.children:
                    bpy.context.scene.collection.children.link(col)

        except Exception as e:
            print(f"Error appending from {tile_blend}: {e}")
            continue

    # Group collections by base name (collision_kerb, collision_kerb.001 -> collision_kerb)
    from collections import defaultdict
    grouped = defaultdict(list)

    for col in bpy.data.collections:
        if col.name.startswith("collision_"):
            base_name = col.name.split(".")[0]
            grouped[base_name].append(col)

    # Merge objects for each base collection
    print(f"Merging {len(grouped)} collision types...")
    for base_name, cols in grouped.items():
        # Collect all mesh objects from all collections with this base name
        all_meshes = []
        for col in cols:
            all_meshes.extend([obj for obj in col.objects if obj.type == "MESH"])

        if len(all_meshes) <= 1:
            print(f"  {base_name}: {len(all_meshes)} object(s), skipping merge")
            continue

        print(f"  {base_name}: merging {len(all_meshes)} objects from {len(cols)} collection(s)...")

        try:
            # Join all meshes
            bpy.ops.object.select_all(action="DESELECT")
            for obj in all_meshes:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = all_meshes[0]

            bpy.ops.object.join()

            merged = bpy.context.active_object
            merged.name = base_name.replace("collision_", "1").upper() + "_merged"

            print(f"    => {len(merged.data.vertices)} verts, {len(merged.data.polygons)} faces")

        except Exception as e:
            print(f"    Error merging {base_name}: {e}")

    # Clean up empty collections
    print("Cleaning up empty collections...")
    removed = 0
    for col in list(bpy.data.collections):
        if col.name.startswith("collision_") and len(col.objects) == 0:
            bpy.data.collections.remove(col)
            removed += 1
    if removed > 0:
        print(f"  Removed {removed} empty collection(s)")

    # Save output
    print(f"Saving: {output_blend}")
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    print("Merge complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge tile blend files")
    p.add_argument("--base", required=True, help="Base blend file")
    p.add_argument("--tiles", required=True, help="Comma-separated tile blend files")
    p.add_argument("--output", required=True, help="Output blend file")
    args = p.parse_args(_get_script_argv())

    tile_list = [t.strip() for t in args.tiles.split(",") if t.strip()]

    merge_tile_results(args.base, tile_list, args.output)
