#!/usr/bin/env python3
"""Test debug_boolean parameter in parallel mode."""
import sys
sys.path.insert(0, 'script')

from pipeline_config import PipelineConfig

config = PipelineConfig(
    geotiff_path='test_images_gic/result.tif',
    tiles_dir='test_images_gic/b3dm',
    output_dir='output',
).resolve()

print(f"s9_debug_boolean: {config.s9_debug_boolean}")
print(f"s9_parallel_surfaces: {config.s9_parallel_surfaces}")
print(f"s9_road_kerb_method: {config.s9_road_kerb_method}")

# Test args_dict construction
args_dict = {
    "debug_boolean": config.s9_debug_boolean,
    "road_kerb_bool": config.s9_road_kerb_method == "bool",
}

print(f"\nargs_dict: {args_dict}")
print(f"\nIf debug_boolean is True, --debug-boolean will be added to Blender command")
