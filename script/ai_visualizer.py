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
    centerline: Optional[List] = None,
    bends: Optional[List[Dict[str, Any]]] = None,
    figsize: Tuple[int, int] = (12, 12),
    dpi: int = 150,
) -> str:
    """Overlay game-object positions on the track map and save the result.

    Args:
        image_path: Path to the original track image.
        objects_json: The game objects JSON dict (with ``objects`` key).
        output_path: Where to save the output image.
        centerline: Optional list of [x, y] points for centerline overlay.
        bends: Optional list of bend dicts (start_idx, end_idx, exit_idx, total_angle).
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

    # --- Centerline + bend highlighting ---
    cl_arr = None
    if centerline and len(centerline) > 1:
        cl_arr = np.array(centerline)
        # Draw full centerline as thin white dashed line (background)
        ax.plot(cl_arr[:, 0], cl_arr[:, 1], color="white", linewidth=1.2,
                linestyle="--", alpha=0.5, zorder=2, label="Centerline")

        # Highlight bend regions on centerline
        if bends:
            _draw_bend_regions(ax, cl_arr, bends)

    # --- Timing section lines (connect L/R pairs) ---
    objects = objects_json.get("objects") or []
    _draw_timing_sections(ax, objects)

    # --- Game object markers ---
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
                markeredgewidth=0.8, label=label, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5),
                    zorder=6)

        # Draw orientation arrow
        orient = obj.get("orientation_z")
        if orient and len(orient) >= 2:
            dx, dy = float(orient[0]) * 20, float(orient[1]) * 20
            ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                        zorder=5)

    direction = objects_json.get("track_direction", "")
    ax.set_title(f"Game Objects (direction: {direction})")
    ax.axis("off")
    if drawn_types or cl_arr is not None:
        ax.legend(loc="upper right", fontsize=9)

    out = str(Path(output_path).resolve())
    fig.savefig(out, bbox_inches="tight", dpi=dpi, pad_inches=0.1)
    plt.close(fig)
    return out


# Bend region colors — cycle through for multiple bends
_BEND_COLORS = ["#ff4444", "#ff8800", "#ffcc00", "#44ff44", "#4488ff", "#cc44ff"]


def _draw_bend_regions(
    ax: Any,
    centerline: np.ndarray,
    bends: List[Dict[str, Any]],
) -> None:
    """Highlight bend segments on the centerline with colored thick lines."""
    n = len(centerline)
    has_label = False

    for i, bend in enumerate(bends):
        s = bend.get("start_idx", 0)
        e = bend.get("end_idx", 0)
        exit_idx = bend.get("exit_idx", e)
        color = _BEND_COLORS[i % len(_BEND_COLORS)]

        # Build index range (handle wrap-around for closed loops)
        if e >= s:
            idxs = list(range(s, e + 1))
        else:
            idxs = list(range(s, n)) + list(range(0, e + 1))

        if len(idxs) < 2:
            continue

        seg = centerline[idxs]
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=3.5, alpha=0.8,
                solid_capstyle="round", zorder=3,
                label="Bends" if not has_label else None)
        has_label = True

        # Mark bend exit with a triangle
        if 0 <= exit_idx < n:
            ex, ey = centerline[exit_idx]
            ax.plot(ex, ey, "v", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4)

        # Label bend at peak — use turn_label (T1, T2, ...) if available
        peak_idx = bend.get("peak_idx", (s + e) // 2)
        if 0 <= peak_idx < n:
            px, py = centerline[peak_idx]
            angle_deg = round(np.degrees(bend.get("total_angle", 0)))
            label_text = bend.get("turn_label", f"B{i}")
            ax.annotate(f"{label_text} ({angle_deg}\u00b0)", (px, py),
                        textcoords="offset points", xytext=(0, -12),
                        fontsize=7, color="white", ha="center",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.7),
                        zorder=6)


def _draw_timing_sections(
    ax: Any,
    objects: List[Dict[str, Any]],
) -> None:
    """Draw lines connecting each AC_TIME_N_L / AC_TIME_N_R pair."""
    # Group timing objects by index N
    timing_pairs: Dict[str, Dict[str, Any]] = {}
    for obj in objects:
        name = obj.get("name", "")
        if not name.startswith("AC_TIME_"):
            continue
        # Parse AC_TIME_N_L or AC_TIME_N_R
        parts = name.replace("AC_TIME_", "").rsplit("_", 1)
        if len(parts) != 2:
            continue
        idx, side = parts[0], parts[1]
        timing_pairs.setdefault(idx, {})[side] = obj

    has_label = False
    for idx in sorted(timing_pairs.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        pair = timing_pairs[idx]
        left = pair.get("L")
        right = pair.get("R")
        if not left or not right:
            continue
        lpos = left.get("position", [])
        rpos = right.get("position", [])
        if len(lpos) < 2 or len(rpos) < 2:
            continue

        ax.plot([lpos[0], rpos[0]], [lpos[1], rpos[1]],
                color="orange", linewidth=2, alpha=0.8, linestyle="-",
                zorder=4, label="Timing sections" if not has_label else None)
        has_label = True
