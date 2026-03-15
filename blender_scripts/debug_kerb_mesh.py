"""Blender script to visualize and debug kerb mesh_group 4."""
import json
import os
import sys

import bpy
import bmesh

# Parse arguments
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

if len(argv) < 2:
    print("Usage: blender --background --python debug_kerb_mesh.py -- <json_path> <output_dir>")
    sys.exit(1)

json_path = argv[0]
output_dir = argv[1]
os.makedirs(output_dir, exist_ok=True)

# Load data
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

mesh_groups = data.get("mesh_groups", [])
if len(mesh_groups) <= 4:
    print(f"ERROR: mesh_groups[4] not found (only {len(mesh_groups)} groups)")
    sys.exit(1)

mg4 = mesh_groups[4]
points = mg4["points_xyz"]
faces = mg4["faces"]

print(f"Mesh group 4: {len(points)} vertices, {len(faces)} faces")

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create mesh
mesh = bpy.data.meshes.new("kerb_4_mesh")
verts = [(p[0], p[1], p[2]) for p in points]
mesh.from_pydata(verts, [], faces)
mesh.update()

obj = bpy.data.objects.new("kerb_4", mesh)
bpy.context.collection.objects.link(obj)

# Check for issues using bmesh
bm = bmesh.new()
bm.from_mesh(mesh)
bm.faces.ensure_lookup_table()

issues = []
for i, face in enumerate(bm.faces):
    if len(face.verts) != 3:
        issues.append(f"Face {i}: not a triangle ({len(face.verts)} verts)")
    # Check for zero-area faces
    if face.calc_area() < 1e-6:
        issues.append(f"Face {i}: degenerate (area={face.calc_area():.2e})")

bm.free()

if issues:
    print(f"\nFound {len(issues)} mesh issues:")
    for issue in issues[:10]:
        print(f"  {issue}")
    with open(os.path.join(output_dir, "mesh_issues.txt"), "w") as f:
        f.write("\n".join(issues))

# Setup scene for rendering
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

# Add camera
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

# Position camera to frame object
bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
center = sum((v for v in bbox), start=bpy.context.scene.cursor.location.copy()) / len(bbox)
size = max((max(v[i] for v in bbox) - min(v[i] for v in bbox)) for i in range(3))

cam.location = (center.x + size * 1.5, center.y - size * 1.5, center.z + size)
cam.rotation_euler = (1.1, 0, 0.785)

# Add light
light_data = bpy.data.lights.new("Light", 'SUN')
light_data.energy = 2.0
light = bpy.data.objects.new("Light", light_data)
bpy.context.collection.objects.link(light)
light.location = (center.x + size, center.y - size, center.z + size * 2)

# Add material
mat = bpy.data.materials.new("KerbMat")
mat.use_nodes = True
obj.data.materials.append(mat)

# Render solid view
scene.render.filepath = os.path.join(output_dir, "kerb_4_solid.png")
bpy.ops.render.render(write_still=True)
print(f"Rendered: {scene.render.filepath}")

# Render wireframe
obj.display_type = 'WIRE'
scene.render.filepath = os.path.join(output_dir, "kerb_4_wireframe.png")
bpy.ops.render.render(write_still=True)
print(f"Rendered: {scene.render.filepath}")

print(f"\nDebug complete. Check {output_dir} for results.")

