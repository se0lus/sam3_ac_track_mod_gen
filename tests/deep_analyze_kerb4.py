"""Analyze kerb4 contour without external dependencies."""
import json
import numpy as np

def check_self_intersection_simple(contour):
    """Simple self-intersection check using line segment crossings."""
    n = len(contour)
    intersections = []

    for i in range(n):
        p1 = np.array(contour[i][:2])
        p2 = np.array(contour[(i + 1) % n][:2])

        for j in range(i + 2, n):
            if j == (i - 1) % n:  # Skip adjacent segments
                continue

            p3 = np.array(contour[j][:2])
            p4 = np.array(contour[(j + 1) % n][:2])

            # Check if segments (p1,p2) and (p3,p4) intersect
            d1 = p2 - p1
            d2 = p4 - p3
            d3 = p3 - p1

            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) < 1e-10:  # Parallel
                continue

            t1 = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
            t2 = (d3[0] * d1[1] - d3[1] * d1[0]) / cross

            if 0 < t1 < 1 and 0 < t2 < 1:
                intersections.append((i, j))

    return intersections

json_path = r"C:\Users\CY\Documents\Codes\sam3_track_seg\output_shajing\08_blender_polygons\gap_filled\kerb\kerb_merged_blender.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

mg4 = data["mesh_groups"][4]
geo_xy = mg4.get("geo_xy", [])
points_xyz = np.array(mg4["points_xyz"])
faces = np.array(mg4["faces"])

print(f"=== Kerb4 Deep Analysis ===\n")
print(f"Contour: {len(geo_xy)} vertices")
print(f"Mesh: {len(points_xyz)} vertices, {len(faces)} triangles\n")

# Check 1: Duplicate points
print("[1/3] Checking for duplicate consecutive points...")
dups = 0
for i in range(len(geo_xy)):
    p1 = np.array(geo_xy[i][:2])
    p2 = np.array(geo_xy[(i + 1) % len(geo_xy)][:2])
    if np.linalg.norm(p1 - p2) < 1e-9:
        dups += 1

print(f"  {'✓' if dups == 0 else '⚠️'}  {dups} duplicate points")

# Check 2: Self-intersection
print("\n[2/3] Checking contour self-intersection...")
intersections = check_self_intersection_simple(geo_xy)
if intersections:
    print(f"  ❌ Found {len(intersections)} self-intersections!")
    for i, j in intersections[:5]:
        print(f"    Segments {i}-{i+1} and {j}-{j+1} intersect")
else:
    print(f"  ✓ No self-intersections")

# Check 3: Triangle quality
print("\n[3/3] Analyzing triangle quality...")
areas = []
for face in faces:
    p0, p1, p2 = points_xyz[face]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = np.cross(v1, v2)
    area = np.linalg.norm(cross) / 2
    areas.append(area)

areas = np.array(areas)
print(f"  Total area: {areas.sum():.6f}")
print(f"  Min area: {areas.min():.6f}")
print(f"  Max area: {areas.max():.6f}")
print(f"  Mean area: {areas.mean():.6f}")

tiny = np.sum(areas < 0.001)
if tiny > 0:
    print(f"  ⚠️  {tiny} tiny triangles (area < 0.001)")

print(f"\n{'='*50}")
if intersections:
    print("❌ PROBLEM FOUND: Contour has self-intersections")
    print("   This causes earcut to produce invalid triangulation")
    print("   Solution: Fix contour extraction or simplification")
else:
    print("✓ Contour is valid, triangulation should be correct")
