"""Standalone kerb4 geometry validator and fixer (no Blender dependency)."""
import json
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

def load_kerb4_data():
    """Load kerb4 mesh data."""
    json_path = r"C:\Users\CY\Documents\Codes\sam3_track_seg\output_shajing\08_blender_polygons\gap_filled\kerb\kerb_merged_blender.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mg4 = data["mesh_groups"][4]
    points = np.array(mg4["points_xyz"])  # (N, 3)
    faces = np.array(mg4["faces"])  # (M, 3)

    return points, faces

def check_face_validity(points, faces):
    """Check for invalid face indices and degenerate triangles."""
    issues = []
    n_verts = len(points)

    for i, face in enumerate(faces):
        # Check bounds
        if np.any(face < 0) or np.any(face >= n_verts):
            issues.append(f"Face {i}: invalid index {face}")
            continue

        # Check degenerate (duplicate vertices)
        if face[0] == face[1] or face[1] == face[2] or face[0] == face[2]:
            issues.append(f"Face {i}: degenerate (duplicate vertices)")
            continue

        # Check zero area
        p0, p1, p2 = points[face]
        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.cross(v1, v2)
        area = np.linalg.norm(cross) / 2
        if area < 1e-6:
            issues.append(f"Face {i}: zero area ({area:.2e})")

    return issues

def check_triangle_intersections(points, faces):
    """Check for self-intersecting triangles (simplified 2D check in XZ plane)."""
    # Project to XZ plane (Y is vertical in Blender)
    points_2d = points[:, [0, 2]]  # (N, 2) - X and Z coordinates

    intersections = []
    n_faces = len(faces)

    # Only check a sample to avoid O(n²) explosion
    sample_size = min(100, n_faces)
    sample_indices = np.random.choice(n_faces, sample_size, replace=False)

    for i in sample_indices:
        tri_i = points_2d[faces[i]]
        bbox_i = [tri_i.min(axis=0), tri_i.max(axis=0)]

        for j in range(i + 1, n_faces):
            tri_j = points_2d[faces[j]]
            bbox_j = [tri_j.min(axis=0), tri_j.max(axis=0)]

            # Quick bbox rejection
            if (bbox_i[1][0] < bbox_j[0][0] or bbox_j[1][0] < bbox_i[0][0] or
                bbox_i[1][1] < bbox_j[0][1] or bbox_j[1][1] < bbox_i[0][1]):
                continue

            # Detailed check would go here (skipped for performance)
            # Just flag potential overlaps
            intersections.append((i, j))

    return intersections[:10]  # Return first 10

def visualize_mesh_2d(points, faces, output_path):
    """Visualize mesh in 2D (XZ plane)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Project to XZ
    x = points[:, 0]
    z = points[:, 2]

    # Plot 1: Wireframe
    ax1.set_title("Wireframe (XZ plane)")
    ax1.set_aspect('equal')
    for face in faces:
        tri = points[face][:, [0, 2]]  # XZ coords
        tri_closed = np.vstack([tri, tri[0]])  # Close the triangle
        ax1.plot(tri_closed[:, 0], tri_closed[:, 1], 'b-', linewidth=0.5, alpha=0.6)
    ax1.scatter(x, z, c='red', s=1, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Filled triangles with color by normal direction
    ax2.set_title("Triangles colored by normal Y-component")
    ax2.set_aspect('equal')

    for face in faces:
        p0, p1, p2 = points[face]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        normal_y = normal[1]  # Y component

        # Color: blue if Y>0 (up), red if Y<0 (down)
        color = 'blue' if normal_y > 0 else 'red'
        alpha = min(0.5, abs(normal_y) / 10)

        tri = points[face][:, [0, 2]]
        triangle = plt.Polygon(tri, facecolor=color, edgecolor='black',
                              linewidth=0.3, alpha=alpha)
        ax2.add_patch(triangle)

    ax2.set_xlim(x.min() - 1, x.max() + 1)
    ax2.set_ylim(z.min() - 1, z.max() + 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved: {output_path}")
    plt.close()

def main():
    print("=== Kerb4 Geometry Validation ===\n")

    points, faces = load_kerb4_data()
    print(f"Loaded: {len(points)} vertices, {len(faces)} faces\n")

    # Check 1: Face validity
    print("[1/4] Checking face validity...")
    issues = check_face_validity(points, faces)
    if issues:
        print(f"  ❌ Found {len(issues)} issues:")
        for issue in issues[:5]:
            print(f"    {issue}")
    else:
        print(f"  ✓ All faces valid")

    # Check 2: Triangle orientations
    print("\n[2/4] Checking triangle orientations...")
    normals_y = []
    for face in faces:
        p0, p1, p2 = points[face]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        normals_y.append(normal[1])

    normals_y = np.array(normals_y)
    n_up = np.sum(normals_y > 0)
    n_down = np.sum(normals_y < 0)
    print(f"  Normals pointing up (Y>0): {n_up}")
    print(f"  Normals pointing down (Y<0): {n_down}")

    if n_up > 0 and n_down > 0:
        print(f"  ⚠️  Mixed orientations detected!")
    elif n_down > n_up:
        print(f"  ⚠️  Most normals point down (wrong direction)")
    else:
        print(f"  ✓ Orientations consistent")

    # Check 3: Potential intersections
    print("\n[3/4] Checking for potential triangle overlaps...")
    intersections = check_triangle_intersections(points, faces)
    if intersections:
        print(f"  ⚠️  Found {len(intersections)} potential overlaps (sample)")
    else:
        print(f"  ✓ No obvious overlaps detected")

    # Check 4: Visualize
    print("\n[4/4] Generating visualization...")
    output_dir = r"C:\Users\CY\Documents\Codes\sam3_track_seg\output_shajing\08_blender_polygons\kerb_debug"
    os.makedirs(output_dir, exist_ok=True)
    viz_path = os.path.join(output_dir, "kerb4_analysis.png")
    visualize_mesh_2d(points, faces, viz_path)

    print(f"\n=== Analysis Complete ===")
    print(f"Check visualization: {viz_path}")

if __name__ == "__main__":
    main()
