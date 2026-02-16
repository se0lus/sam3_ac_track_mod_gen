"""
Visualization helpers for AI-generated track data (walls, game objects).

Pure Python -- uses matplotlib, no Blender dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless use
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def _ensure_mpl() -> None:
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. pip install matplotlib")


# ---------------------------------------------------------------------------
# Wall visualization
# ---------------------------------------------------------------------------

def visualize_walls(
    image_path: Union[str, Path],
    walls_json: Dict[str, Any],
    output_path: Union[str, Path],
    *,
    figsize: Tuple[int, int] = (12, 12),
    dpi: int = 150,
) -> str:
    """Overlay wall polygons on the track map image and save the result.

    Args:
        image_path: Path to the original track image.
        walls_json: The wall JSON dict (with ``walls`` key).
        output_path: Where to save the output image.
        figsize: Matplotlib figure size.
        dpi: Output DPI.

    Returns:
        Absolute path of the saved image.
    """
    _ensure_mpl()

    img = Image.open(str(image_path)).convert("RGB")
    img_array = np.array(img)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_array)

    _WALL_STYLES = {
        "outer":        ("lime",        2.5, "Outer wall"),
        "tree":         ("darkgreen",   2.0, "Tree"),
        "building":     ("purple",      2.0, "Building"),
        "water":        ("deepskyblue", 2.0, "Water"),
        "separation":   ("cyan",        2.0, "Separation wall"),  # legacy
        "inner":        ("red",         2.0, "Inner wall"),       # legacy
        "barrier":      ("orange",      2.0, "Barrier"),          # legacy
        "tire_barrier": ("orange",      2.0, "Tire barrier"),     # legacy
        "guardrail":    ("yellow",      2.0, "Guardrail"),        # legacy
    }

    walls = walls_json.get("walls") or []
    drawn_types: set = set()

    for wall in walls:
        pts = wall.get("points") or []
        if len(pts) < 2:
            continue
        wtype = wall.get("type", "inner")
        closed = wall.get("closed", True)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if closed:
            xs.append(xs[0])
            ys.append(ys[0])

        color, lw, label = _WALL_STYLES.get(wtype, ("red", 2.0, wtype))
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.85,
                label=label if wtype not in drawn_types else None)
        drawn_types.add(wtype)

    ax.set_title("Virtual Walls")
    ax.axis("off")
    if drawn_types:
        ax.legend(loc="upper right", fontsize=10)

    out = str(Path(output_path).resolve())
    fig.savefig(out, bbox_inches="tight", dpi=dpi, pad_inches=0.1)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Game object visualization
# ---------------------------------------------------------------------------

_OBJECT_COLORS: Dict[str, str] = {
    "hotlap_start": "magenta",
    "pit": "cyan",
    "start": "yellow",
    "timing_left": "orange",
    "timing_right": "orangered",
}


def visualize_game_objects(
    image_path: Union[str, Path],
    objects_json: Dict[str, Any],
    output_path: Union[str, Path],
    *,
    figsize: Tuple[int, int] = (12, 12),
    dpi: int = 150,
) -> str:
    """Overlay game-object positions on the track map and save the result.

    Args:
        image_path: Path to the original track image.
        objects_json: The game objects JSON dict (with ``objects`` key).
        output_path: Where to save the output image.
        figsize: Matplotlib figure size.
        dpi: Output DPI.

    Returns:
        Absolute path of the saved image.
    """
    _ensure_mpl()

    img = Image.open(str(image_path)).convert("RGB")
    img_array = np.array(img)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_array)

    objects = objects_json.get("objects") or []
    drawn_types: set = set()

    for obj in objects:
        pos = obj.get("position")
        if not pos or len(pos) < 2:
            continue
        x, y = float(pos[0]), float(pos[1])
        otype = obj.get("type", "unknown")
        name = obj.get("name", "")
        color = _OBJECT_COLORS.get(otype, "white")
        label = otype if otype not in drawn_types else None
        drawn_types.add(otype)

        ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor="black",
                markeredgewidth=0.8, label=label)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5))

        # Draw orientation arrow if available
        orient = obj.get("orientation_z")
        if orient and len(orient) >= 2:
            dx, dy = float(orient[0]) * 20, float(orient[1]) * 20
            ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    direction = objects_json.get("track_direction", "")
    ax.set_title(f"Game Objects (direction: {direction})")
    ax.axis("off")
    if drawn_types:
        ax.legend(loc="upper right", fontsize=9)

    out = str(Path(output_path).resolve())
    fig.savefig(out, bbox_inches="tight", dpi=dpi, pad_inches=0.1)
    plt.close(fig)
    return out
