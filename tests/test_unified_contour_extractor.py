"""
Test unified contour extractor with a simple synthetic composite
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script'))

import numpy as np


def test_basic_extraction():
    """Test with a simple 2-tag composite"""
    # Create a simple composite: tag 1 on left, tag 2 on right
    composite = np.zeros((100, 100), dtype=np.uint8)
    composite[:, :50] = 1  # tag 1 (sand)
    composite[:, 50:] = 2  # tag 2 (grass)

    bounds = {'x_min': 0, 'y_min': 0, 'x_max': 100, 'y_max': 100}

    # Import here to check if dependencies are available
    try:
        from unified_contour_extractor import extract_all_contours_unified
    except ImportError as e:
        print(f"Cannot import: {e}")
        print("This test requires cv2, numpy, and other dependencies.")
        print("Please set up the conda environment first: setup_env.bat")
        return False

    result = extract_all_contours_unified(
        composite, bounds, 100, 100,
        simplify_epsilon=1.0,
        min_contour_area=10,
    )

    print(f"Extracted {len(result)} tags")
    for tag_id, groups in result.items():
        print(f"  Tag {tag_id}: {len(groups)} groups")
        for i, group in enumerate(groups):
            print(f"    Group {i}: {len(group['vertices'])} vertices, {len(group['faces'])} faces")

    # Basic validation
    assert 1 in result, "Tag 1 should be extracted"
    assert 2 in result, "Tag 2 should be extracted"
    assert len(result[1]) > 0, "Tag 1 should have at least one group"
    assert len(result[2]) > 0, "Tag 2 should have at least one group"

    print("\n✓ Basic extraction test passed")
    return True


if __name__ == "__main__":
    success = test_basic_extraction()
    sys.exit(0 if success else 1)
