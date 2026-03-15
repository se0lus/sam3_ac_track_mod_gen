"""Fix kerb4 self-intersection by re-simplifying and re-triangulating."""
import json
import os
import sys
import numpy as np

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_script_dir, "script"))

def check_self_intersection(contour):
    """Check if contour has self-intersections."""
    n = len(contour)
    if n < 4:
        return False

    for i in range(n):
        p1 = np.array(contour[i][:2])
        p2 = np.array(contour[(i + 1) % n][:2])

        for j in range(i + 2, n):
            if j == (i + n - 1) % n:
                continue

            p3 = np.array(contour[j][:2])
            p4 = np.array(contour[(j + 1) % n][:2])

            d1 = p2 - p1
            d2 = p4 - p3
            d3 = p3 - p1

            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) < 1e-10:
                continue

            t1 = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
            t2 = (d3[0] * d1[1] - d3[1] * d1[0]) / cross

            if 0 < t1 < 1 and 0 < t2 < 1:
                return True

    return False

def simplify_contour(contour, epsilon):
    """Simplify contour using Douglas-Peucker."""
    from rdp import rdp
    return rdp(contour, epsilon=epsilon)

def retriangulate(geo_xy):
    """Re-triangulate contour using earcut."""
    import mapbox_earcut

    coords = np.array([[p[0], p[1]] for p in geo_xy], dtype=np.float64)
    rings = np.array([len(coords)], dtype=np.uint32)

    try:
        tri_indices = mapbox_earcut.triangulate_float64(coords, rings)
    except Exception as e:
        print(f"  Triangulation failed: {e}")
        return None

    if len(tri_indices) == 0:
        return None

    # Reverse winding for correct normals
    faces = []
    for i in range(0, len(tri_indices), 3):
        faces.append([int(tri_indices[i]), int(tri_indices[i + 2]), int(tri_indices[i + 1])])

    return faces

json_path = r"C:\Users\CY\Documents\Codes\sam3_track_seg\output_shajing\05a_manual_surface_masks\kerb\kerb_merged_blender.json"

print("=== Fixing kerb4 self-intersection ===\n")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

mg4 = data["mesh_groups"][4]
geo_xy = mg4["geo_xy"]

print(f"Original contour: {len(geo_xy)} vertices")

# Check if self-intersecting
if not check_self_intersection(geo_xy):
    print("✓ No self-intersection, nothing to fix")
    sys.exit(0)

print("❌ Self-intersection detected, fixing...")

# Try re-simplification with smaller epsilon
epsilon = 0.5  # Start with smaller epsilon
fixed_contour = None

for attempt in range(5):
    try:
        simplified = simplify_contour(geo_xy, epsilon)
        if len(simplified) < 3:
            epsilon = epsilon / 2
            continue

        if not check_self_intersection(simplified):
            fixed_contour = simplified
            print(f"✓ Fixed with epsilon={epsilon:.3f}, {len(simplified)} vertices")
            break

        epsilon = epsilon / 2
    except Exception as e:
        print(f"  Attempt {attempt + 1} failed: {e}")
        epsilon = epsilon / 2

if fixed_contour is None:
    print("❌ Could not fix self-intersection")
    sys.exit(1)

# Re-triangulate
print("Re-triangulating...")
new_faces = retriangulate(fixed_contour)

if new_faces is None:
    print("❌ Triangulation failed")
    sys.exit(1)

print(f"✓ Generated {len(new_faces)} triangles")

# Update mesh_group 4
mg4["geo_xy"] = fixed_contour
mg4["faces"] = new_faces
# points_xyz stays the same (will be regenerated from geo_xy in stage8)

# Save backup
backup_path = json_path.replace(".json", "_backup.json")
with open(backup_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
print(f"Backup saved: {backup_path}")

# Save fixed version
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"\n✓ Fixed kerb4 saved to: {json_path}")
print("Now re-run Stage 8 to apply the fix")
