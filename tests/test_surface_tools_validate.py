"""
Offline validation of collision surface tool outputs.

Reads ``report.json`` + OBJ files produced by ``test_surface_tools.py``
and runs automated checks.  No Blender dependency — pure Python.

Usage::

    python tests/test_surface_tools_validate.py \
        --report output/test_surface_tools/report.json

Checks:
  1. Tool A: collision_road exists with verts > 0, faces > 0
  2. Tool A: collision_kerb exists (or SKIP if no kerb masks)
  3. Tool B: at least one of collision_grass / collision_sand / collision_road2
  4. All collision meshes have y_range > 0.5m (not flat — projected to terrain)
  5. OBJ files parse correctly, vertex count matches report.json
  6. No degenerate faces (area > epsilon)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# OBJ parser
# ---------------------------------------------------------------------------

@dataclass
class OBJMesh:
    """Minimal parsed OBJ data."""
    verts: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[list[int]] = field(default_factory=list)  # 0-indexed


def parse_obj(filepath: str) -> OBJMesh:
    """Parse a Wavefront OBJ file, extracting vertices and faces."""
    mesh = OBJMesh()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                mesh.verts.append((
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                ))
            elif line.startswith("f "):
                parts = line.split()[1:]
                # OBJ indices are 1-based; may include /vt/vn notation
                indices = []
                for p in parts:
                    idx_str = p.split("/")[0]
                    indices.append(int(idx_str) - 1)
                mesh.faces.append(indices)
    return mesh


# ---------------------------------------------------------------------------
# Triangle area (3D)
# ---------------------------------------------------------------------------

def _triangle_area_3d(
    v0: tuple[float, float, float],
    v1: tuple[float, float, float],
    v2: tuple[float, float, float],
) -> float:
    """Area of a triangle defined by three 3D points (cross product method)."""
    ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
    bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _entries_for_collection(results: list[dict], col_name: str) -> list[dict]:
    return [r for r in results if r["collection"] == col_name]


def check_tool_a_road(results: list[dict]) -> CheckResult:
    """Tool A: collision_road must exist with verts > 0, faces > 0."""
    entries = _entries_for_collection(results, "collision_road")
    if not entries:
        return CheckResult("ToolA_road_exists", False,
                           "No objects in collision_road")
    for e in entries:
        if e["verts"] <= 0 or e["faces"] <= 0:
            return CheckResult("ToolA_road_exists", False,
                               f"{e['object']}: verts={e['verts']}, "
                               f"faces={e['faces']}")
    total_v = sum(e["verts"] for e in entries)
    total_f = sum(e["faces"] for e in entries)
    return CheckResult("ToolA_road_exists", True,
                       f"{len(entries)} object(s), "
                       f"{total_v} verts, {total_f} faces")


def check_tool_a_kerb(results: list[dict]) -> CheckResult:
    """Tool A: collision_kerb should exist (SKIP if no kerb masks)."""
    entries = _entries_for_collection(results, "collision_kerb")
    if not entries:
        # Kerb is optional — depends on whether kerb masks exist
        return CheckResult("ToolA_kerb_exists", True,
                           "SKIP — no kerb objects (kerb masks may not exist)")
    for e in entries:
        if e["verts"] <= 0 or e["faces"] <= 0:
            return CheckResult("ToolA_kerb_exists", False,
                               f"{e['object']}: verts={e['verts']}, "
                               f"faces={e['faces']}")
    total_v = sum(e["verts"] for e in entries)
    return CheckResult("ToolA_kerb_exists", True,
                       f"{len(entries)} object(s), {total_v} verts")


def check_tool_b_any(results: list[dict]) -> CheckResult:
    """Tool B: at least one of grass/sand/road2 must have objects."""
    boolean_cols = ["collision_grass", "collision_sand", "collision_road2"]
    found = {}
    for col_name in boolean_cols:
        entries = _entries_for_collection(results, col_name)
        if entries:
            found[col_name] = len(entries)

    if not found:
        return CheckResult("ToolB_any_exists", False,
                           "No objects in any of: " + ", ".join(boolean_cols))
    detail = ", ".join(f"{k}: {v} obj(s)" for k, v in found.items())
    return CheckResult("ToolB_any_exists", True, detail)


def check_y_range(results: list[dict], min_y_range: float = 0.5) -> CheckResult:
    """All collision meshes should have y_range > min_y_range (not flat)."""
    flat = []
    for e in results:
        if e["y_range"] < min_y_range:
            flat.append(f"{e['object']}(y_range={e['y_range']:.3f})")
    if flat:
        return CheckResult("y_range_check", False,
                           f"{len(flat)} flat mesh(es): " + ", ".join(flat[:5]))
    return CheckResult("y_range_check", True,
                       f"All {len(results)} mesh(es) have y_range >= "
                       f"{min_y_range}m")


def check_obj_integrity(
    results: list[dict],
    output_dir: str,
) -> CheckResult:
    """OBJ files parse correctly; vertex count matches report.json."""
    errors = []
    checked = 0
    for e in results:
        obj_path = os.path.join(output_dir, e["collection"],
                                f"{e['object']}.obj")
        if not os.path.isfile(obj_path):
            errors.append(f"{e['object']}: OBJ file missing")
            continue

        try:
            mesh = parse_obj(obj_path)
        except Exception as exc:
            errors.append(f"{e['object']}: parse error — {exc}")
            continue

        if len(mesh.verts) != e["verts"]:
            errors.append(
                f"{e['object']}: verts mismatch "
                f"(OBJ={len(mesh.verts)}, report={e['verts']})")
            continue

        checked += 1

    if errors:
        return CheckResult("obj_integrity", False,
                           "; ".join(errors[:5]))
    return CheckResult("obj_integrity", True,
                       f"{checked} OBJ file(s) verified")


def check_degenerate_faces(
    results: list[dict],
    output_dir: str,
    area_epsilon: float = 1e-10,
) -> CheckResult:
    """No degenerate faces (area > epsilon)."""
    problems = []
    total_faces_checked = 0

    for e in results:
        obj_path = os.path.join(output_dir, e["collection"],
                                f"{e['object']}.obj")
        if not os.path.isfile(obj_path):
            continue

        try:
            mesh = parse_obj(obj_path)
        except Exception:
            continue

        degen_count = 0
        for face in mesh.faces:
            if len(face) < 3:
                degen_count += 1
                continue
            # Check triangles within the face (fan triangulation)
            for i in range(1, len(face) - 1):
                v0 = mesh.verts[face[0]]
                v1 = mesh.verts[face[i]]
                v2 = mesh.verts[face[i + 1]]
                area = _triangle_area_3d(v0, v1, v2)
                total_faces_checked += 1
                if area < area_epsilon:
                    degen_count += 1

        if degen_count > 0:
            problems.append(f"{e['object']}: {degen_count} degenerate")

    if problems:
        return CheckResult("no_degenerate_faces", False,
                           "; ".join(problems[:5]))
    return CheckResult("no_degenerate_faces", True,
                       f"{total_faces_checked} triangles checked, none degenerate")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Validate collision surface tool outputs")
    p.add_argument("--report", required=True,
                    help="Path to report.json from test_surface_tools.py")
    args = p.parse_args()

    report_path = os.path.abspath(args.report)
    if not os.path.isfile(report_path):
        print(f"FATAL: report file not found: {report_path}")
        return 1

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    results = report.get("results", [])
    output_dir = os.path.dirname(report_path)

    print(f"Loaded report: {len(results)} collision object(s)")
    print(f"  Tool A result: {report.get('tool_a_result', '?')} "
          f"({report.get('tool_a_elapsed_s', '?')}s)")
    print(f"  Tool B result: {report.get('tool_b_result', '?')} "
          f"({report.get('tool_b_elapsed_s', '?')}s)")
    print()

    # Run all checks
    checks = [
        check_tool_a_road(results),
        check_tool_a_kerb(results),
        check_tool_b_any(results),
        check_y_range(results),
        check_obj_integrity(results, output_dir),
        check_degenerate_faces(results, output_dir),
    ]

    # Print results
    all_pass = True
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        if not c.passed:
            all_pass = False
        print(f"  [{status}] {c.name}: {c.detail}")

    print()
    if all_pass:
        print("=== ALL CHECKS PASSED ===")
        return 0
    else:
        n_fail = sum(1 for c in checks if not c.passed)
        print(f"=== {n_fail} CHECK(S) FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
