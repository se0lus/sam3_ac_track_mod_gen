"""Analyze kerb4 contour and triangulation quality."""
import json
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.validation import explain_validity

def analyze_kerb4_contour():
    """Analyze the original contour before triangulation."""
    json_path = r"C:\Users\CY\Documents\Codes\sam3_track_seg\output_shajing\08_blender_polygons\gap_filled\kerb\kerb_merged_blender.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mg4 = data["mesh_groups"][4]
    geo_xy = mg4.get("geo_xy", [])  # Original contour in geo coordinates

    if not geo_xy:
        print("ERROR: No geo_xy data found")
        return

    print(f"=== Kerb4 Contour Analysis ===\n")
    print(f"Contour vertices: {len(geo_xy)}")

    # Check for duplicate consecutive points
    duplicates = []
    for i in range(len(geo_xy)):
        p1 = geo_xy[i]
        p2 = geo_xy[(i + 1) % len(geo_xy)]
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if dist < 1e-9:
            duplicates.append(i)

    if duplicates:
        print(f"⚠️  Found {len(duplicates)} duplicate consecutive points")
    else:
        print(f"✓ No duplicate consecutive points")

    # Use shapely to validate polygon
    try:
        poly = Polygon([(p[0], p[1]) for p in geo_xy])
        is_valid = poly.is_valid
        print(f"\nShapely validation: {'✓ Valid' if is_valid else '❌ Invalid'}")

        if not is_valid:
            print(f"  Reason: {explain_validity(poly)}")

        print(f"  Area: {poly.area:.6f}")
        print(f"  Is simple: {poly.is_simple}")
        print(f"  Is closed: {poly.is_closed if hasattr(poly, 'is_closed') else 'N/A'}")

    except Exception as e:
        print(f"❌ Shapely validation failed: {e}")

    # Check triangulation coverage
    points_xyz = np.array(mg4["points_xyz"])
    faces = np.array(mg4["faces"])

    # Compute total triangle area
    total_area = 0
    for face in faces:
        p0, p1, p2 = points_xyz[face]
        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.cross(v1, v2)
        area = np.linalg.norm(cross) / 2
        total_area += area

    print(f"\nTriangulation:")
    print(f"  Total triangle area: {total_area:.6f}")
    print(f"  Number of triangles: {len(faces)}")
    print(f"  Avg triangle area: {total_area / len(faces):.6f}")

    # Check if any triangles are inverted or overlapping
    print(f"\nTriangle quality check:")
    small_triangles = 0
    for i, face in enumerate(faces):
        p0, p1, p2 = points_xyz[face]
        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.cross(v1, v2)
        area = np.linalg.norm(cross) / 2

        if area < 0.001:
            small_triangles += 1

    if small_triangles > 0:
        print(f"  ⚠️  {small_triangles} very small triangles (area < 0.001)")
    else:
        print(f"  ✓ All triangles have reasonable size")

if __name__ == "__main__":
    analyze_kerb4_contour()
