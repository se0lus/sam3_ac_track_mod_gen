"""Complete debug workflow for kerb_4 mesh issues."""
import json
import os
import subprocess
import sys

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_script_dir, "script"))

from pipeline_config import PipelineConfig

def main():
    # Load config
    config = PipelineConfig(
        output_dir=r"C:\Users\CY\Documents\Codes\sam3_track_seg\output_shajing"
    ).resolve()

    kerb_json = os.path.join(
        config.stage_dir("blender_polygons"),
        "gap_filled", "kerb", "kerb_merged_blender.json"
    )

    if not os.path.isfile(kerb_json):
        print(f"ERROR: {kerb_json} not found")
        return

    # Create debug output directory
    debug_dir = os.path.join(config.stage_dir("blender_polygons"), "kerb_debug")
    os.makedirs(debug_dir, exist_ok=True)

    print("=== Step 1: Analyze mesh data ===")
    with open(kerb_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    mesh_groups = data.get("mesh_groups", [])
    if len(mesh_groups) <= 4:
        print(f"ERROR: mesh_groups[4] not found")
        return

    mg4 = mesh_groups[4]
    points = mg4["points_xyz"]
    faces = mg4["faces"]

    print(f"Vertices: {len(points)}, Faces: {len(faces)}")

    # Check for issues
    issues = []
    for i, face in enumerate(faces):
        if len(face) != 3:
            issues.append(f"Face {i}: not triangle")
        elif face[0] == face[1] or face[1] == face[2] or face[0] == face[2]:
            issues.append(f"Face {i}: degenerate")

    if issues:
        print(f"Found {len(issues)} data issues")
        with open(os.path.join(debug_dir, "data_issues.txt"), "w") as f:
            f.write("\n".join(issues))

    print("\n=== Step 2: Visualize in Blender ===")
    blender_script = os.path.join(_script_dir, "blender_scripts", "debug_kerb_mesh.py")
    cmd = [
        config.blender_exe,
        "--background",
        "--python", blender_script,
        "--",
        kerb_json,
        debug_dir
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Blender failed")
        print(result.stderr)
        return

    print(result.stdout)

    print(f"\n=== Debug complete ===")
    print(f"Results in: {debug_dir}")
    print(f"  - kerb_4_solid.png")
    print(f"  - kerb_4_wireframe.png")
    if issues:
        print(f"  - data_issues.txt ({len(issues)} issues)")

if __name__ == "__main__":
    main()
