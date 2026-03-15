"""Test script to diagnose kerb_4 mesh construction issues."""
import json
import os
import sys

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_script_dir, "script"))

def analyze_mesh_group(mesh_group, group_idx):
    """Analyze a single mesh group for potential issues."""
    points = mesh_group.get("points_xyz", [])
    faces = mesh_group.get("faces", [])

    print(f"\n=== Mesh Group {group_idx} ===")
    print(f"Vertices: {len(points)}")
    print(f"Faces: {len(faces)}")

    # Check for degenerate triangles
    degenerate = []
    for i, face in enumerate(faces):
        if len(face) != 3:
            print(f"  WARNING: Face {i} has {len(face)} vertices (expected 3)")
            continue
        if face[0] == face[1] or face[1] == face[2] or face[0] == face[2]:
            degenerate.append(i)

    if degenerate:
        print(f"  WARNING: {len(degenerate)} degenerate triangles (duplicate vertices)")
        print(f"    First few: {degenerate[:5]}")

    # Check for out-of-bounds indices
    max_idx = len(points) - 1
    invalid = []
    for i, face in enumerate(faces):
        for v_idx in face:
            if v_idx < 0 or v_idx > max_idx:
                invalid.append((i, v_idx))
                break

    if invalid:
        print(f"  ERROR: {len(invalid)} faces with invalid vertex indices")
        print(f"    First few: {invalid[:5]}")

    # Compute bounding box
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        print(f"  Bounding box:")
        print(f"    X: [{min(xs):.2f}, {max(xs):.2f}]")
        print(f"    Y: [{min(ys):.2f}, {max(ys):.2f}]")
        print(f"    Z: [{min(zs):.2f}, {max(zs):.2f}]")

    return len(degenerate) == 0 and len(invalid) == 0


def main():
    # Find kerb JSON
    kerb_json = r"C:\Users\CY\Documents\Codes\sam3_track_seg\output_shajing\08_blender_polygons\gap_filled\kerb\kerb_merged_blender.json"

    if not os.path.isfile(kerb_json):
        print(f"ERROR: File not found: {kerb_json}")
        return

    with open(kerb_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    mesh_groups = data.get("mesh_groups", [])
    print(f"Total mesh groups: {len(mesh_groups)}")

    if len(mesh_groups) <= 4:
        print(f"ERROR: mesh_groups[4] does not exist (only {len(mesh_groups)} groups)")
        return

    # Analyze mesh_group 4
    mg4 = mesh_groups[4]
    is_valid = analyze_mesh_group(mg4, 4)

    if is_valid:
        print("\n✓ Mesh group 4 appears structurally valid")
        print("  Issue may be in Blender mesh construction or visualization")
    else:
        print("\n✗ Mesh group 4 has structural issues")
        print("  Check earcut triangulation or contour extraction")

    # Export for external visualization
    export_path = kerb_json.replace(".json", "_group4_debug.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump({"mesh_group_4": mg4}, f, indent=2)
    print(f"\nExported mesh_group 4 to: {export_path}")


if __name__ == "__main__":
    main()
