"""
One-off script: Extract 1Wall_0 from final_track.blend, reverse-transform
its base vertices back to pixel coordinates, and restore the outer#0 entry
in 06a_manual_walls/walls.json.

Run with Blender:
  blender --background --python tests/restore_wall_from_blend.py

Or from command line (the script auto-invokes Blender if not running inside it).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Paths (absolute)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CY_GIC_OUTPUT = os.path.join(PROJECT_ROOT, "cy_gic_output")

BLEND_FILE = os.path.join(CY_GIC_OUTPUT, "09a_manual_blender", "final_track.blend")
WALLS_JSON = os.path.join(CY_GIC_OUTPUT, "06a_manual_walls", "walls.json")
GEO_META_JSON = os.path.join(CY_GIC_OUTPUT, "06a_manual_walls", "geo_metadata.json")
TILES_DIR = os.path.join(PROJECT_ROOT, "test_images_gic", "b3dm")
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"

WALL_OBJECT_NAME = "1Wall_0"  # the object to extract


def _is_inside_blender() -> bool:
    try:
        import bpy  # type: ignore[import-not-found]
        return True
    except ImportError:
        return False


# ===========================================================================
# Part A: Blender script — extract base vertices of 1Wall_0 to a temp JSON
# ===========================================================================

def blender_extract_wall():
    """Run inside Blender: open blend, extract 1Wall_0 base vertices, write JSON."""
    import bpy  # type: ignore[import-not-found]

    # The blend file is already loaded (--background <blend>)
    obj = bpy.data.objects.get(WALL_OBJECT_NAME)
    if obj is None:
        # Try case-insensitive search
        for o in bpy.data.objects:
            if o.name.lower() == WALL_OBJECT_NAME.lower():
                obj = o
                break
    if obj is None:
        print(f"ERROR: Object '{WALL_OBJECT_NAME}' not found in blend file.")
        print(f"Available objects: {[o.name for o in bpy.data.objects if 'wall' in o.name.lower() or 'WALL' in o.name]}")
        sys.exit(1)

    mesh = obj.data
    world_matrix = obj.matrix_world

    # Collect all vertices in world space
    verts_world = []
    for v in mesh.vertices:
        co_world = world_matrix @ v.co
        verts_world.append((co_world.x, co_world.y, co_world.z))

    if not verts_world:
        print("ERROR: No vertices found")
        sys.exit(1)

    print(f"Total vertices: {len(verts_world)}")
    print(f"Total faces: {len(mesh.polygons)}")

    # The wall mesh was created as quads (4 verts each), then triangulated.
    # Original structure per segment i:
    #   v0 = (p_i_x, wall_bottom, p_i_z)
    #   v1 = (p_{i+1}_x, wall_bottom, p_{i+1}_z)
    #   v2 = (p_{i+1}_x, wall_top, p_{i+1}_z)
    #   v3 = (p_i_x, wall_top, p_i_z)
    # After triangulation: 2 tris per quad.
    # Each segment creates 4 NEW vertices (no sharing between segments).
    #
    # So for N segments: 4*N total vertices, 2*N at bottom, 2*N at top.
    # The bottom vertices come in pairs: (p_i, p_{i+1}) for each segment.
    # The polyline points are the unique XZ positions among bottom vertices.

    # Find the two Y levels
    all_ys = sorted(set(round(v[1], 4) for v in verts_world))
    print(f"Unique Y levels: {all_ys}")

    bottom_y = all_ys[0]
    tolerance = 0.1

    # Collect bottom vertex world coords with their vertex indices
    bottom_verts = []
    for i, v in enumerate(mesh.vertices):
        co_world = world_matrix @ v.co
        if abs(co_world.y - bottom_y) < tolerance:
            bottom_verts.append((i, co_world.x, co_world.y, co_world.z))

    print(f"Bottom Y = {bottom_y}, bottom verts = {len(bottom_verts)}")

    # Group bottom vertices by unique XZ position to find the original polyline points
    from collections import defaultdict
    xz_groups = defaultdict(list)
    for idx, x, y, z in bottom_verts:
        # Round to merge near-identical positions
        key = (round(x, 3), round(z, 3))
        xz_groups[key].append(idx)

    unique_positions = list(xz_groups.keys())
    print(f"Unique XZ positions among bottom verts: {len(unique_positions)}")

    # Now reconstruct order using face connectivity.
    # Each triangulated quad-segment has 2 faces sharing the diagonal edge.
    # The bottom edge of each segment connects two consecutive polyline points.
    # We need to find which bottom edges exist and chain them.

    # Build bottom-vertex index set
    bottom_idx_set = set(bv[0] for bv in bottom_verts)

    # Map vertex index -> XZ key
    idx_to_xz = {}
    for idx, x, y, z in bottom_verts:
        idx_to_xz[idx] = (round(x, 3), round(z, 3))

    # Find edges between bottom vertices (these are the base edges of wall segments)
    # An edge between two bottom verts with DIFFERENT XZ positions = a segment base edge
    adj_xz = defaultdict(set)
    for edge in mesh.edges:
        a, b = edge.vertices
        if a in bottom_idx_set and b in bottom_idx_set:
            ka = idx_to_xz[a]
            kb = idx_to_xz[b]
            if ka != kb:  # Different positions = segment base edge
                adj_xz[ka].add(kb)
                adj_xz[kb].add(ka)

    print(f"XZ adjacency graph: {len(adj_xz)} nodes")
    for k, v in list(adj_xz.items())[:3]:
        print(f"  {k} -> {len(v)} neighbors")

    # Walk the chain to get ordered polyline
    # Find endpoints (degree 1) for open polyline, or any node for closed
    endpoints = [k for k in adj_xz if len(adj_xz[k]) == 1]

    if endpoints:
        start = endpoints[0]
        print(f"Open polyline, starting from endpoint")
    else:
        start = next(iter(adj_xz)) if adj_xz else None
        print(f"Closed polyline")

    if start is None:
        print("ERROR: No adjacency found")
        sys.exit(1)

    ordered_xz = []
    visited = set()
    current = start
    while current is not None:
        visited.add(current)
        ordered_xz.append(current)
        next_node = None
        for nb in adj_xz[current]:
            if nb not in visited:
                next_node = nb
                break
        current = next_node

    print(f"Ordered polyline: {len(ordered_xz)} points")

    # Convert XZ keys to full Blender coords (use bottom_y for Y)
    ordered_pts = [(xz[0], bottom_y, xz[1]) for xz in ordered_xz]

    # Write to temp JSON
    out_path = os.path.join(CY_GIC_OUTPUT, "09a_manual_blender", "_wall0_base_verts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "object_name": obj.name,
            "bottom_y": bottom_y,
            "points_blender_xyz": [[p[0], p[1], p[2]] for p in ordered_pts],
        }, f, indent=2)
    print(f"Wrote {len(ordered_pts)} base points to {out_path}")


# ===========================================================================
# Part B: Pure Python — reverse-transform Blender XYZ → pixel coords
# ===========================================================================

# WGS84 constants
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)
_WGS84_EP2 = (_WGS84_A**2 - _WGS84_B**2) / (_WGS84_B**2)


def geodetic_to_ecef(lon_deg: float, lat_deg: float, h_m: float = 0.0):
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - _WGS84_E2) + h_m) * sin_lat
    return (x, y, z)


def ecef_to_geodetic(x, y, z):
    lon = math.atan2(y, x)
    p = math.sqrt(x*x + y*y)
    if p < 1e-12:
        lat = math.copysign(math.pi/2, z)
        n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * (math.sin(lat)**2))
        h = abs(z) - n * (1.0 - _WGS84_E2)
        return (math.degrees(lon), math.degrees(lat), h)
    theta = math.atan2(z * _WGS84_A, p * _WGS84_B)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    lat = math.atan2(
        z + _WGS84_EP2 * _WGS84_B * sin_t**3,
        p - _WGS84_E2 * _WGS84_A * cos_t**3,
    )
    sin_lat = math.sin(lat)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    h = p / max(1e-12, math.cos(lat)) - n
    return (math.degrees(lon), math.degrees(lat), h)


def _mat4_from_cesium_col_major(tf):
    m = [[0.0]*4 for _ in range(4)]
    for c in range(4):
        for r in range(4):
            m[r][c] = float(tf[c*4 + r])
    return m


def _mat4_mul_point(m, p):
    x, y, z = p
    px = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3]
    py = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3]
    pz = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3]
    pw = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3]
    if abs(pw) > 1e-12:
        inv = 1.0/pw
        return (px*inv, py*inv, pz*inv)
    return (px, py, pz)


def _geo_to_pixel(lon, lat, geo_meta):
    """Reverse of _pixel_to_geo: WGS84 (lon, lat) → pixel (px, py)."""
    w = geo_meta["image_width"]
    h = geo_meta["image_height"]
    corners = geo_meta.get("corners")

    if corners:
        # Bilinear interpolation is hard to invert analytically.
        # Use Newton iteration to find (u, v) such that bilinear(u,v) = (lon, lat).
        tl = corners["top_left"]     # [lat, lon]
        tr = corners["top_right"]
        bl = corners["bottom_left"]
        br = corners["bottom_right"]

        def bilinear(u, v):
            lon_out = (1-u)*(1-v)*tl[1] + u*(1-v)*tr[1] + (1-u)*v*bl[1] + u*v*br[1]
            lat_out = (1-u)*(1-v)*tl[0] + u*(1-v)*tr[0] + (1-u)*v*bl[0] + u*v*br[0]
            return lon_out, lat_out

        # Initial guess using simple rectangle
        bounds = geo_meta.get("bounds", {})
        if bounds:
            u = (lon - bounds["west"]) / (bounds["east"] - bounds["west"])
            v = (bounds["north"] - lat) / (bounds["north"] - bounds["south"])
        else:
            u, v = 0.5, 0.5

        # Newton iterations
        for _ in range(20):
            lon_est, lat_est = bilinear(u, v)
            dlon = lon - lon_est
            dlat = lat - lat_est
            if abs(dlon) < 1e-12 and abs(dlat) < 1e-12:
                break
            # Jacobian
            dlon_du = -(1-v)*tl[1] + (1-v)*tr[1] - v*bl[1] + v*br[1]
            dlon_dv = -(1-u)*tl[1] - u*tr[1] + (1-u)*bl[1] + u*br[1]
            dlat_du = -(1-v)*tl[0] + (1-v)*tr[0] - v*bl[0] + v*br[0]
            dlat_dv = -(1-u)*tl[0] - u*tr[0] + (1-u)*bl[0] + u*br[0]
            det = dlon_du*dlat_dv - dlon_dv*dlat_du
            if abs(det) < 1e-20:
                break
            du = (dlon*dlat_dv - dlat*dlon_dv) / det
            dv = (dlat*dlon_du - dlon*dlat_du) / det
            u += du
            v += dv

        px = u * w
        py = v * h
    else:
        bounds = geo_meta["bounds"]
        px = (lon - bounds["west"]) / (bounds["east"] - bounds["west"]) * w
        py = (bounds["north"] - lat) / (bounds["north"] - bounds["south"]) * h

    return px, py


def _get_forward_transform(tiles_dir):
    """Get the forward tileset transform matrix (local → ECEF)."""
    for root, _, files in os.walk(tiles_dir):
        for name in files:
            if name.lower() == "tileset.json":
                p = os.path.join(root, name)
                with open(p, "r", encoding="utf-8") as f:
                    ts = json.load(f)
                r = ts.get("root", {})
                tf = r.get("transform")
                if isinstance(tf, list) and len(tf) == 16:
                    return _mat4_from_cesium_col_major([float(x) for x in tf])
    raise RuntimeError(f"No tileset.json with root.transform found in {tiles_dir}")


def reverse_transform_and_update():
    """Read extracted Blender base verts, reverse-transform to pixels, update walls.json."""

    verts_json = os.path.join(CY_GIC_OUTPUT, "09a_manual_blender", "_wall0_base_verts.json")
    if not os.path.exists(verts_json):
        print(f"ERROR: {verts_json} not found. Run Blender extraction first.")
        sys.exit(1)

    with open(verts_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    blender_pts = data["points_blender_xyz"]
    print(f"Loaded {len(blender_pts)} base points from Blender extraction")

    # Load geo metadata
    with open(GEO_META_JSON, "r", encoding="utf-8") as f:
        geo_meta = json.load(f)

    # Get forward transform (tileset local → ECEF)
    forward_tf = _get_forward_transform(TILES_DIR)
    print(f"Forward transform loaded from tileset")

    # Reverse: Blender (bx, by, bz) → local (lx=bx, ly=bz, lz=0)
    #          → forward_tf * local → ECEF → WGS84 → pixel
    pixel_points = []
    for pt in blender_pts:
        bx, by, bz = pt[0], pt[1], pt[2]

        # Undo axis remap: Blender (X,Y,Z) = (local_x, local_z, local_y)
        # So: local_x = bx, local_y = bz, local_z = by
        # But local_z was zeroed in forward pass (z_mode="zero"),
        # and by = wall_bottom (not the original lz).
        # We set lz=0 since the forward pass used geodetic_to_ecef(lon, lat, 0.0)
        # which makes lz ≈ 0 in the local frame.
        local_pt = (bx, bz, 0.0)

        # Forward transform: local → ECEF
        ecef = _mat4_mul_point(forward_tf, local_pt)

        # ECEF → WGS84
        lon, lat, h = ecef_to_geodetic(*ecef)

        # WGS84 → pixel
        px, py = _geo_to_pixel(lon, lat, geo_meta)
        pixel_points.append([round(px), round(py)])

    print(f"\nReverse-transformed {len(pixel_points)} points to pixel coordinates:")
    for i, pt in enumerate(pixel_points[:5]):
        print(f"  [{i}] px={pt[0]}, py={pt[1]}")
    if len(pixel_points) > 5:
        print(f"  ... ({len(pixel_points) - 5} more)")

    # Validate: points should be within image bounds
    img_w = geo_meta["image_width"]
    img_h = geo_meta["image_height"]
    out_of_bounds = sum(1 for p in pixel_points if p[0] < 0 or p[0] >= img_w or p[1] < 0 or p[1] >= img_h)
    if out_of_bounds > 0:
        print(f"WARNING: {out_of_bounds} points out of image bounds ({img_w}x{img_h})")

    # Backup walls.json
    import shutil
    backup = WALLS_JSON + ".bak_restore"
    shutil.copy2(WALLS_JSON, backup)
    print(f"\nBacked up walls.json to {backup}")

    # Update walls.json: replace walls[0] (outer#0) points
    with open(WALLS_JSON, "r", encoding="utf-8") as f:
        walls_data = json.load(f)

    old_wall = walls_data["walls"][0]
    print(f"\nOriginal outer#0: type={old_wall['type']}, {len(old_wall['points'])} points")

    walls_data["walls"][0]["points"] = pixel_points
    # Keep type and closed as-is

    with open(WALLS_JSON, "w", encoding="utf-8") as f:
        json.dump(walls_data, f, indent=2, ensure_ascii=False)

    print(f"Updated outer#0: type={old_wall['type']}, {len(pixel_points)} points")
    print(f"\nDone! walls.json updated successfully.")

    # Cleanup temp file
    os.remove(verts_json)
    print(f"Cleaned up temp file: {verts_json}")


# ===========================================================================
# Main dispatcher
# ===========================================================================

if __name__ == "__main__":
    if _is_inside_blender():
        # Phase 1: Running inside Blender — extract vertices
        blender_extract_wall()
    else:
        # Phase 1: Launch Blender to extract vertices
        print("=" * 60)
        print("Phase 1: Extracting 1Wall_0 base vertices via Blender")
        print("=" * 60)

        cmd = [
            BLENDER_EXE,
            BLEND_FILE,
            "--background",
            "--python", os.path.abspath(__file__),
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-500:])
        if result.returncode != 0:
            print(f"Blender exited with code {result.returncode}")
            sys.exit(1)

        # Phase 2: Reverse transform and update walls.json
        print("\n" + "=" * 60)
        print("Phase 2: Reverse-transforming to pixel coords & updating walls.json")
        print("=" * 60)

        reverse_transform_and_update()
