"""Stage 10: Model export — clean, split, rename, batch, and export FBX + KN5.

Reads ``09_result/final_track.blend`` and exports split models as FBX files
suitable for Assetto Corsa.  Runs Blender in ``--background`` mode.
Optionally converts FBX → KN5 via ``ksEditorAT.exe`` from ac_tools_cmd.

Tile levels are auto-detected from ``L{N}`` collections in the .blend file,
so no base_level/target_level parameters are needed.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from typing import List, Optional

logger = logging.getLogger("sam3_pipeline.s10")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


# ---------------------------------------------------------------------------
# ksEditorAT auto-setup + FBX→KN5 conversion
# ---------------------------------------------------------------------------
_KSEDITOR_RELEASE_URL = "https://github.com/leBluem/ac_tools_cmd/releases/download/0.9.5/ackn5.7z"


def _download_and_extract_7z(url: str, dest_dir: str) -> None:
    """Download a .7z archive and extract it to *dest_dir*."""
    os.makedirs(dest_dir, exist_ok=True)
    tmp_path = os.path.join(tempfile.gettempdir(), "ackn5.7z")
    try:
        logger.info("Downloading %s ...", url)
        urllib.request.urlretrieve(url, tmp_path)
        logger.info("Downloaded %.1f MB", os.path.getsize(tmp_path) / 1e6)

        # Strategy 1: py7zr (pure Python)
        try:
            import py7zr
        except ImportError:
            logger.info("py7zr not installed, installing...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "py7zr"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            import py7zr

        with py7zr.SevenZipFile(tmp_path, mode="r") as z:
            z.extractall(path=dest_dir)
        logger.info("Extracted to %s", dest_dir)
    except Exception as e:
        # Strategy 2: system 7z.exe
        logger.info("py7zr failed (%s), trying system 7z...", e)
        exe_7z = _find_system_7z()
        if not exe_7z:
            raise RuntimeError(
                f"Cannot extract 7z archive: py7zr failed and 7z.exe not found. "
                f"Please install py7zr (pip install py7zr) or 7-Zip."
            ) from e
        subprocess.check_call(
            [exe_7z, "x", "-y", f"-o{dest_dir}", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        logger.info("Extracted to %s (via 7z.exe)", dest_dir)
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _find_system_7z() -> Optional[str]:
    """Search for 7z.exe on PATH and common install locations."""
    # Check PATH
    if shutil.which("7z"):
        return shutil.which("7z")
    # Common Windows locations
    for candidate in [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _ensure_kseditor(config: PipelineConfig, project_root: str) -> Optional[str]:
    """Locate or download ``ksEditorAt.exe``.

    The ac_tools_cmd release extracts to: ``ackn5/ksEditorAT/ksEditorAt.exe``

    Search order:
    1. ``config.s10_kseditor_exe`` (user-configured path)
    2. ``{project_root}/ac_toolbox/ac_tools_cmd/ackn5/ksEditorAT/ksEditorAt.exe``
    3. Auto-download from GitHub Releases
    """
    # 1. User-configured path
    if config.s10_kseditor_exe and os.path.isfile(config.s10_kseditor_exe):
        return config.s10_kseditor_exe

    # 2. Convention path (7z extracts to ackn5/ksEditorAT/)
    default_path = os.path.join(
        project_root, "ac_toolbox", "ac_tools_cmd",
        "ackn5", "ksEditorAT", "ksEditorAt.exe",
    )
    if os.path.isfile(default_path):
        return default_path

    # 3. Auto-download
    logger.info("ksEditorAt not found, downloading from GitHub...")
    dest_dir = os.path.join(project_root, "ac_toolbox", "ac_tools_cmd")
    try:
        _download_and_extract_7z(_KSEDITOR_RELEASE_URL, dest_dir)
    except Exception as e:
        logger.error("Failed to download ac_tools_cmd: %s", e)
        return None

    return default_path if os.path.isfile(default_path) else None


def _find_kn5fixbyini(kseditor_exe: str) -> Optional[str]:
    """Locate ``kn5FixByINI.exe`` relative to ksEditorAt.exe.

    The tool lives at ``ackn5/kn5FixByINI/x64/kn5FixByINI.exe`` (prefer x64)
    or ``ackn5/kn5FixByINI/kn5FixByINI.exe``.
    """
    # ackn5/ksEditorAT/ksEditorAt.exe → ackn5/
    ackn5_dir = os.path.dirname(os.path.dirname(kseditor_exe))
    for subpath in [
        os.path.join("kn5FixByINI", "x64", "kn5FixByINI.exe"),
        os.path.join("kn5FixByINI", "kn5FixByINI.exe"),
    ]:
        candidate = os.path.join(ackn5_dir, subpath)
        if os.path.isfile(candidate):
            return candidate
    return None


def _convert_fbx_to_kn5(kseditor_exe: str, export_dir: str) -> List[str]:
    """Convert all FBX files in *export_dir* to KN5 using ksEditorAt.

    Workflow per FBX (matching ``fbx2kn5_track.cmd``):
    1. ``ksEditorAt.exe kn5track {batch}.kn5 {batch}.fbx``
    2. ``kn5FixByINI.exe {batch}.kn5`` (post-fix based on INI)

    cwd is set to *export_dir* so the tool finds ``texture/`` and ``.fbx.ini``.
    Returns list of successfully created KN5 file paths.
    """
    fbx_files = sorted(glob.glob(os.path.join(export_dir, "*.fbx")))
    if not fbx_files:
        logger.warning("No FBX files found in %s", export_dir)
        return []

    kn5fix_exe = _find_kn5fixbyini(kseditor_exe)
    if kn5fix_exe:
        logger.info("kn5FixByINI: %s", kn5fix_exe)
    else:
        logger.info("kn5FixByINI not found, skipping post-fix step")

    kn5_files: List[str] = []
    for fbx_path in fbx_files:
        # Skip very small FBX files (likely empty game object batches)
        fbx_size = os.path.getsize(fbx_path)
        if fbx_size < 1024:
            logger.info("Skipping tiny FBX (%d bytes): %s",
                        fbx_size, os.path.basename(fbx_path))
            continue

        basename = os.path.splitext(os.path.basename(fbx_path))[0]
        kn5_name = basename + ".kn5"
        kn5_path = os.path.join(export_dir, kn5_name)

        cmd = [kseditor_exe, "kn5track", kn5_path, fbx_path]
        logger.info("KN5 converting: %s -> %s (%.1f MB)",
                    os.path.basename(fbx_path), kn5_name, fbx_size / 1e6)

        try:
            result = subprocess.run(
                cmd, cwd=export_dir, capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0 and os.path.isfile(kn5_path):
                size_mb = os.path.getsize(kn5_path) / 1e6
                logger.info("  OK: %s (%.1f MB)", kn5_name, size_mb)
                kn5_files.append(kn5_path)

                # Post-fix with kn5FixByINI
                if kn5fix_exe:
                    try:
                        fix_result = subprocess.run(
                            [kn5fix_exe, kn5_path],
                            cwd=os.path.dirname(kn5fix_exe),
                            capture_output=True, text=True, timeout=60,
                        )
                        if fix_result.returncode == 0:
                            logger.info("  kn5FixByINI OK: %s", kn5_name)
                        else:
                            logger.warning("  kn5FixByINI failed (rc=%d): %s",
                                           fix_result.returncode, fix_result.stderr[:200])
                    except Exception as e:
                        logger.warning("  kn5FixByINI error: %s", e)
            else:
                logger.warning(
                    "  FAILED: %s (rc=%d)\n  stdout: %s\n  stderr: %s",
                    kn5_name, result.returncode,
                    result.stdout[:500] if result.stdout else "",
                    result.stderr[:500] if result.stderr else "",
                )
        except subprocess.TimeoutExpired:
            logger.warning("  TIMEOUT: %s (>300s)", kn5_name)
        except Exception as e:
            logger.warning("  ERROR: %s -- %s", kn5_name, e)

    return kn5_files


def run(config: PipelineConfig) -> None:
    """Execute Stage 10: Model export.

    Reads:
    - ``config.blender_result_dir / final_track.blend`` from stage 9

    Writes to ``config.export_dir`` (``output/10_model_export/``).
    """
    logger.info("=== Stage 10: Model export ===")

    if not config.blender_exe:
        raise ValueError("blender_exe is required for model_export stage")

    # Locate input blend file from 09_result junction
    blend_input = os.path.join(config.blender_result_dir, "final_track.blend")
    if not os.path.isfile(blend_input):
        raise FileNotFoundError(
            f"Input blend file not found: {blend_input}\n"
            "Run Stage 9 (blender_automate) first."
        )

    os.makedirs(config.export_dir, exist_ok=True)

    blender_script = os.path.join(
        _script_dir, "..", "blender_scripts", "blender_export.py",
    )
    blender_script = os.path.abspath(blender_script)

    # Build Blender command (no base-level/target-level — auto-detected)
    cmd = [
        config.blender_exe, "--background",
        "--python", blender_script,
        "--",
        "--blend-input", blend_input,
        "--output-dir", config.export_dir,
        "--tiles-dir", config.tiles_dir,
        "--max-vertices", str(config.s10_max_vertices),
        "--max-batch-mb", str(config.s10_max_batch_mb),
        "--fbx-scale", str(config.s10_fbx_scale),
    ]

    # Find centerline.json from 07_result
    go_result = config.game_objects_result_dir
    if not os.path.isdir(go_result):
        go_result = os.path.dirname(config.game_objects_json)
    centerline_json = ""
    for candidate in glob.glob(os.path.join(go_result, "*", "centerline.json")):
        centerline_json = candidate
        break
    if not centerline_json:
        candidate = os.path.join(go_result, "centerline.json")
        if os.path.isfile(candidate):
            centerline_json = candidate
    if centerline_json:
        cmd.extend(["--centerline-json", centerline_json])
        logger.info("Using centerline: %s", centerline_json)
    else:
        logger.warning("centerline.json not found — road splitting will use XZ bisection")

    # Find geo_metadata.json (same search as s09)
    walls_result = config.walls_result_dir
    if not os.path.isdir(walls_result):
        walls_result = os.path.dirname(config.walls_json)
    geo_metadata = ""
    for candidate_dir in [walls_result, go_result,
                          os.path.dirname(config.walls_json),
                          os.path.dirname(config.game_objects_json)]:
        candidate = os.path.join(candidate_dir, "geo_metadata.json")
        if os.path.isfile(candidate):
            geo_metadata = candidate
            break
    if geo_metadata:
        cmd.extend(["--geo-metadata", geo_metadata])
        logger.info("Using geo metadata: %s", geo_metadata)
    else:
        logger.warning("geo_metadata.json not found — centerline coordinate conversion may fail")

    # INI material parameters
    cmd.extend(["--ks-ambient", str(config.s10_ks_ambient)])
    cmd.extend(["--ks-diffuse", str(config.s10_ks_diffuse)])
    cmd.extend(["--ks-emissive", str(config.s10_ks_emissive)])

    logger.info("Running Blender export: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Copy textures from 09_result/texture/ to 10_model_export/texture/
    src_texture_dir = os.path.join(config.blender_result_dir, "texture")
    dst_texture_dir = os.path.join(config.export_dir, "texture")
    if os.path.isdir(src_texture_dir):
        shutil.copytree(src_texture_dir, dst_texture_dir, dirs_exist_ok=True)
        n_textures = len([f for f in os.listdir(dst_texture_dir) if os.path.isfile(os.path.join(dst_texture_dir, f))])
        logger.info("Copied %d textures to %s", n_textures, dst_texture_dir)
    else:
        logger.warning("No texture directory found at %s", src_texture_dir)
        os.makedirs(dst_texture_dir, exist_ok=True)

    # Copy NULL.dds for collision materials
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    null_dds_src = os.path.join(project_root, "test_images_shajing", "kn5_example", "texture", "NULL.dds")
    if os.path.isfile(null_dds_src):
        shutil.copy2(null_dds_src, os.path.join(dst_texture_dir, "NULL.dds"))
        logger.info("Copied NULL.dds to %s", dst_texture_dir)

    # KN5 conversion (FBX → KN5 via ksEditorAT)
    kseditor = _ensure_kseditor(config, project_root)
    if kseditor:
        logger.info("Using ksEditorAT: %s", kseditor)
        kn5_files = _convert_fbx_to_kn5(kseditor, config.export_dir)
        logger.info("Converted %d KN5 files", len(kn5_files))
    else:
        logger.warning("ksEditorAT not found, skipping KN5 conversion")

    logger.info("=== Stage 10 complete: %s ===", config.export_dir)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 10: Model export")
    parser.add_argument("--output-dir", default="output", help="Output base directory")
    parser.add_argument("--tiles-dir", default="", help="Directory with tileset.json")
    parser.add_argument("--blender-exe", default="", help="Path to Blender executable")
    parser.add_argument("--max-vertices", type=int, default=0, help="Max vertices per mesh")
    parser.add_argument("--max-batch-mb", type=int, default=0, help="Max FBX batch size MB")
    parser.add_argument("--fbx-scale", type=float, default=0.0, help="FBX export scale")
    parser.add_argument("--ks-ambient", type=float, default=-1.0, help="ksAmbient for visible materials")
    parser.add_argument("--ks-diffuse", type=float, default=-1.0, help="ksDiffuse for visible materials")
    parser.add_argument("--ks-emissive", type=float, default=-1.0, help="ksEmissive for visible materials")
    parser.add_argument("--kseditor-exe", default="", help="Path to ksEditorAT.exe")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
    ).resolve()
    if args.blender_exe:
        config.blender_exe = args.blender_exe
    if args.max_vertices > 0:
        config.s10_max_vertices = args.max_vertices
    if args.max_batch_mb > 0:
        config.s10_max_batch_mb = args.max_batch_mb
    if args.fbx_scale > 0:
        config.s10_fbx_scale = args.fbx_scale
    if args.ks_ambient >= 0:
        config.s10_ks_ambient = args.ks_ambient
    if args.ks_diffuse >= 0:
        config.s10_ks_diffuse = args.ks_diffuse
    if args.ks_emissive >= 0:
        config.s10_ks_emissive = args.ks_emissive
    if args.kseditor_exe:
        config.s10_kseditor_exe = args.kseditor_exe
    run(config)


if __name__ == "__main__":
    main()
