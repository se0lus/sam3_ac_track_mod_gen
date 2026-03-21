"""Quick test for shared boundary extraction with synthetic data."""

import numpy as np
import cv2
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from shared_boundary_extractor import (
    extract_all_shared_boundaries,
    visualize_shared_boundaries,
    generate_boundary_report,
)


def create_test_composite():
    """Create a simple test composite with 3 adjacent regions."""
    composite = np.zeros((200, 300), dtype=np.uint8)

    # Region 1 (road): left third
    composite[:, :100] = 1

    # Region 2 (kerb): middle third
    composite[:, 100:200] = 2

    # Region 3 (grass): right third
    composite[:, 200:] = 3

    return composite


def main():
    print("Creating test composite...")
    composite = create_test_composite()

    # Mock bounds
    bounds = {
        "left": 0.0,
        "right": 1.0,
        "top": 1.0,
        "bottom": 0.0,
    }

    canvas_w, canvas_h = composite.shape[1], composite.shape[0]

    print(f"Composite size: {canvas_w}x{canvas_h}")
    print(f"Unique tags: {np.unique(composite)}")

    # Extract boundaries
    print("\nExtracting shared boundaries...")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")

    library = extract_all_shared_boundaries(
        composite, bounds, canvas_w, canvas_h,
        simplify_epsilon=1.0,
        min_chain_length=3,
        logger=logger,
    )

    print(f"\nExtracted {len(library.boundaries)} boundaries")

    # Generate outputs
    output_dir = os.path.join(_script_dir, "..", "output", "test_shared_boundary")
    os.makedirs(output_dir, exist_ok=True)

    # Save composite
    composite_vis = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    composite_vis[composite == 1] = (100, 100, 255)  # road - blue
    composite_vis[composite == 2] = (255, 200, 100)  # kerb - orange
    composite_vis[composite == 3] = (100, 255, 100)  # grass - green
    cv2.imwrite(os.path.join(output_dir, "composite.png"), composite_vis)

    # Visualize boundaries
    vis_path = os.path.join(output_dir, "boundaries.png")
    visualize_shared_boundaries(composite, library, vis_path)
    print(f"Saved visualization: {vis_path}")

    # Generate report
    report_path = os.path.join(output_dir, "report.txt")
    generate_boundary_report(library, report_path)
    print(f"Saved report: {report_path}")

    # Print summary
    print("\n=== Summary ===")
    for tag_id in [1, 2, 3]:
        boundaries = library.get_boundaries_for_tag(tag_id)
        print(f"Tag {tag_id}: {len(boundaries)} boundaries")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
