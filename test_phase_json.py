#!/usr/bin/env python3
"""Test three-phase boundary extraction with JSON output."""
import sys
import os
sys.path.insert(0, 'script')

from pipeline_config import PipelineConfig
import numpy as np

# Load config
config = PipelineConfig(
    geotiff_path='test_images_shajing/result.tif',
    tiles_dir='test_images_shajing/b3dm',
    output_dir='output_shajing',
).resolve()

# Create test composite
composite = np.array([
    [1, 1, 2, 2],
    [1, 1, 2, 2],
    [3, 3, 3, 3],
], dtype=np.uint8)

bounds = {"left": 0.0, "right": 4.0, "top": 3.0, "bottom": 0.0}
w, h = 4, 3

# Test three-phase extraction with debug output
from pixel_corner_contour_extractor import phase1_extract, phase2_simplify, phase3_mesh

print("Testing three-phase extraction with JSON output...")
debug_dir = "output_shajing/test_phase_json"
os.makedirs(debug_dir, exist_ok=True)

graph = phase1_extract(composite, bounds, w, h, output_path=f"{debug_dir}/boundary_graph.json")
print(f"✓ Phase 1: {len(graph['chains'])} chains, {len(graph['junctions'])} junctions")

simplified = phase2_simplify(graph, epsilon=0.5, output_path=f"{debug_dir}/boundary_graph_simplified.json")
print(f"✓ Phase 2: Simplified")

meshes = phase3_mesh(simplified)
print(f"✓ Phase 3: {len(meshes)} labels with geometry")

print(f"\nJSON files written to: {debug_dir}/")
print(f"  - boundary_graph.json")
print(f"  - boundary_graph_simplified.json")
