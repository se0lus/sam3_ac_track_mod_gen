"""Tests for shared boundary extraction."""

import numpy as np
import pytest
import sys
import os

# Add script directory to path
_test_dir = os.path.dirname(os.path.abspath(__file__))
_script_dir = os.path.join(os.path.dirname(_test_dir), "script")
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from shared_boundary_extractor import (
    detect_boundary_pixels,
    trace_boundary_chains,
    extract_all_shared_boundaries,
    SharedBoundaryLibrary,
)


def test_detect_boundary_pixels():
    """Test boundary pixel detection between adjacent regions."""
    # Create simple composite: 2 regions side by side
    composite = np.zeros((10, 10), dtype=np.uint8)
    composite[:, :5] = 1  # Left half = tag 1
    composite[:, 5:] = 2  # Right half = tag 2

    boundaries = detect_boundary_pixels(composite)

    # Should have one boundary pair (1, 2)
    assert (1, 2) in boundaries
    assert len(boundaries) == 1

    # Boundary should be at x=4 and x=5
    pixels = boundaries[(1, 2)]
    assert len(pixels) > 0
    for x, y in pixels:
        assert x in [4, 5]


def test_trace_boundary_chains():
    """Test boundary chain tracing."""
    # Create a simple vertical boundary
    pixels = {(5, y) for y in range(10)}

    chains = trace_boundary_chains(pixels, min_chain_length=3)

    # Should have one chain
    assert len(chains) >= 1
    # Chain should contain most of the pixels
    assert len(chains[0]) >= 3


def test_extract_all_shared_boundaries():
    """Test full boundary extraction pipeline."""
    # Create composite with 2 adjacent regions
    composite = np.zeros((100, 100), dtype=np.uint8)
    composite[:, :50] = 1
    composite[:, 50:] = 2

    bounds = {
        "min_lon": 0.0,
        "max_lon": 1.0,
        "min_lat": 0.0,
        "max_lat": 1.0,
    }

    library = extract_all_shared_boundaries(
        composite, bounds, 100, 100,
        simplify_epsilon=1.0,
        min_chain_length=3,
    )

    # Should extract at least one boundary
    assert len(library.boundaries) > 0

    # Boundary should be between tags 1 and 2
    boundary = library.boundaries[0]
    assert boundary.tag_pair == (1, 2)
    assert len(boundary.geo_coords) >= 2


def test_boundary_library_lookup():
    """Test boundary library tag-based lookup."""
    library = SharedBoundaryLibrary()

    # Add mock boundaries
    from shared_boundary_extractor import SharedBoundary
    b1 = SharedBoundary((1, 2), [], [(0, 0), (1, 1)], 10.0)
    b2 = SharedBoundary((2, 3), [], [(1, 1), (2, 2)], 10.0)

    library.add(b1)
    library.add(b2)

    # Tag 1 should find boundary with tag 2
    boundaries_1 = library.get_boundaries_for_tag(1)
    assert len(boundaries_1) == 1
    assert boundaries_1[0].tag_pair == (1, 2)

    # Tag 2 should find both boundaries
    boundaries_2 = library.get_boundaries_for_tag(2)
    assert len(boundaries_2) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
