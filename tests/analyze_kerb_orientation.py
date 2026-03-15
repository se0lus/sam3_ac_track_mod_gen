"""Deep analysis of kerb_4 mesh to find crossing/flipped triangles."""
import json
import os

def cross_product_2d(o, a, b):
    """2D cross product for orientation test."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def triangle_area_3d(p0, p1, p2):
    """Calculate signed area of triangle (using cross product)."""
    v1 = [p1[i] - p0[i] for i in range(3)]
    v2 = [p2[i] - p0[i] for i in range(3)]
    # Cross product
    nx = v1[1] * v2[2] - v1[2] * v2[1]
    ny = v1[2] * v2[0] - v1[0] * v2[2]
    nz = v1[0] * v2[1] - v1[1] * v2[0]
    # Return z-component (assuming XZ plane, Y is up/down)
    return ny

def analyze_kerb_4():
    kerb_json = r"C:\Users\CY\Documents\Codes\sam3_track_seg\output_shajing\08_blender_polygons\gap_filled\kerb\kerb_merged_blender.json"

    with open(kerb_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    mg4 = data["mesh_groups"][4]
    points = mg4["points_xyz"]
    faces = mg4["faces"]

    print(f"Analyzing mesh_group 4: {len(points)} verts, {len(faces)} faces\n")

    # Check triangle orientations
    areas = []
    for i, face in enumerate(faces):
        p0, p1, p2 = [points[face[j]] for j in range(3)]
        area = triangle_area_3d(p0, p1, p2)
        areas.append((i, area))

    positive = sum(1 for _, a in areas if a > 0)
    negative = sum(1 for _, a in areas if a < 0)

    print(f"Triangle orientations:")
    print(f"  Positive (CCW): {positive}")
    print(f"  Negative (CW):  {negative}")

    if positive > 0 and negative > 0:
        print(f"  ⚠️  MIXED ORIENTATIONS - this causes rendering artifacts!")
        print(f"\n  Flipped triangles:")
        minority_sign = -1 if positive > negative else 1
        flipped = [(i, a) for i, a in areas if (a > 0) == (minority_sign > 0)]
        for i, a in flipped[:10]:
            print(f"    Face {i}: area={a:.6f}")

    # Check for very small triangles
    tiny = [(i, abs(a)) for i, a in areas if abs(a) < 0.001]
    if tiny:
        print(f"\n  ⚠️  {len(tiny)} very small triangles (area < 0.001):")
        for i, a in tiny[:5]:
            print(f"    Face {i}: area={a:.6f}")

    # Export problematic faces
    if positive > 0 and negative > 0:
        debug_dir = os.path.dirname(kerb_json).replace("gap_filled\\kerb", "kerb_debug")
        os.makedirs(debug_dir, exist_ok=True)

        report_path = os.path.join(debug_dir, "orientation_issues.txt")
        with open(report_path, "w") as f:
            f.write(f"Mixed orientations detected:\n")
            f.write(f"  Positive: {positive}\n")
            f.write(f"  Negative: {negative}\n\n")
            f.write(f"Flipped faces:\n")
            for i, a in flipped:
                f.write(f"  Face {i}: area={a:.6f}\n")

        print(f"\nReport saved: {report_path}")
        return False

    print("\n✓ All triangles have consistent orientation")
    return True

if __name__ == "__main__":
    analyze_kerb_4()
