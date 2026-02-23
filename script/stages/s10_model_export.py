"""Stage 10: Model export — clean, split, rename, batch, and export FBX + KN5.

Reads ``09_result/final_track.blend`` and exports split models as FBX files
suitable for Assetto Corsa.  Runs Blender in ``--background`` mode.
Optionally converts FBX → KN5 via ``ksEditorAt.exe`` (KsEditorAt v6).

Tile levels are auto-detected from ``L{N}`` collections in the .blend file,
so no base_level/target_level parameters are needed.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
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
from progress import report_progress

# Progress range allocation — based on measured timings:
#   Blender export ~9% of total, KN5 conversion ~91%
_PCT_BLENDER_START = 2
_PCT_BLENDER_END = 10
_PCT_KN5_START = 12
_PCT_KN5_END = 98
_RE_PROGRESS = re.compile(r"@@PROGRESS@@\s+(\d+)\s*(.*)")


# ---------------------------------------------------------------------------
# KsEditorAt v6 auto-setup + FBX→KN5 conversion
# ---------------------------------------------------------------------------
_KSEDITOR_YANDEX_KEY = "https://yadi.sk/d/RuPNrcCurCm4Z"
_KSEDITOR_SHA256 = "1058657ad6922715e5f7c060e2800dd004d68ce135dd2073cedc076b14d4d4b7"


def _download_kseditor_v6(dest_dir: str) -> None:
    """Download KsEditorAt v6 via Yandex Disk public API and extract."""
    import hashlib
    import json
    import zipfile

    # 1. Get direct download link from Yandex Disk public API
    api_url = (
        "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        f"?public_key={_KSEDITOR_YANDEX_KEY}"
    )
    logger.info("Fetching KsEditorAt v6 download link...")
    with urllib.request.urlopen(api_url) as resp:
        href = json.loads(resp.read())["href"]

    # 2. Download ZIP
    tmp_zip = os.path.join(tempfile.gettempdir(), "KsEditorAt_v6.zip")
    try:
        logger.info("Downloading KsEditorAt v6...")
        urllib.request.urlretrieve(href, tmp_zip)
        logger.info("Downloaded %.1f MB", os.path.getsize(tmp_zip) / 1e6)

        # 3. Verify SHA256
        h = hashlib.sha256()
        with open(tmp_zip, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != _KSEDITOR_SHA256:
            raise RuntimeError(
                f"KsEditorAt v6 SHA256 mismatch: expected {_KSEDITOR_SHA256}, "
                f"got {h.hexdigest()}"
            )

        # 4. Extract
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(dest_dir)
        logger.info("Extracted KsEditorAt v6 to %s", dest_dir)
    finally:
        if os.path.isfile(tmp_zip):
            try:
                os.remove(tmp_zip)
            except OSError:
                pass


def _find_exe_in_dir(base: str, name: str) -> Optional[str]:
    """Search for *name* in *base* and its immediate subdirectories."""
    if not os.path.isdir(base):
        return None
    # Direct child
    candidate = os.path.join(base, name)
    if os.path.isfile(candidate):
        return candidate
    # One level of subdirectories
    for entry in os.listdir(base):
        candidate = os.path.join(base, entry, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _ensure_kseditor(config: PipelineConfig, project_root: str) -> Optional[str]:
    """Locate or auto-download ``ksEditorAt.exe`` (KsEditorAt v6).

    Search order:
    1. ``config.s10_kseditor_exe`` (user-configured path)
    2. ``{project_root}/ac_toolbox/ksEditorAt/`` and one level of subdirs
    3. Auto-download from Yandex Disk → extract to convention path → search
    """
    # 1. User-configured path
    if config.s10_kseditor_exe and os.path.isfile(config.s10_kseditor_exe):
        return config.s10_kseditor_exe

    # 2. Convention path
    base = os.path.join(project_root, "ac_toolbox", "ksEditorAt")
    exe = _find_exe_in_dir(base, "ksEditorAt.exe")
    if exe:
        return exe

    # 3. Auto-download
    logger.info("ksEditorAt not found, downloading KsEditorAt v6...")
    try:
        _download_kseditor_v6(base)
    except Exception as e:
        logger.error("Failed to download KsEditorAt v6: %s", e)
        return None

    return _find_exe_in_dir(base, "ksEditorAt.exe")


_AC_TOOLS_CMD_URL = "https://github.com/leBluem/ac_tools_cmd/releases/download/0.9.5/ackn5.7z"


def _find_7z() -> Optional[str]:
    """Locate 7z.exe on PATH or common install locations."""
    found = shutil.which("7z")
    if found:
        return found
    for candidate in [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _download_and_extract_7z(url: str, dest_dir: str) -> None:
    """Download a .7z archive and extract it to *dest_dir* using system 7z."""
    exe_7z = _find_7z()
    if not exe_7z:
        raise RuntimeError(
            "7z.exe not found. Please install 7-Zip "
            "(winget install 7zip.7zip) to extract ac_tools_cmd."
        )
    os.makedirs(dest_dir, exist_ok=True)
    tmp_path = os.path.join(tempfile.gettempdir(), "ackn5.7z")
    try:
        logger.info("Downloading %s ...", url)
        urllib.request.urlretrieve(url, tmp_path)
        logger.info("Downloaded %.1f MB", os.path.getsize(tmp_path) / 1e6)
        subprocess.check_call(
            [exe_7z, "x", "-y", f"-o{dest_dir}", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        logger.info("Extracted to %s", dest_dir)
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _search_kn5fixbyini(search_dirs: list) -> Optional[str]:
    """Search for kn5FixByINI.exe in a list of directories."""
    for base in search_dirs:
        for subpath in [
            os.path.join("kn5FixByINI", "x64", "kn5FixByINI.exe"),
            os.path.join("kn5FixByINI", "kn5FixByINI.exe"),
            "kn5FixByINI.exe",
        ]:
            candidate = os.path.join(base, subpath)
            if os.path.isfile(candidate):
                return candidate
    return None


def _find_kn5fixbyini(kseditor_exe: str, project_root: str) -> Optional[str]:
    """Locate or download ``kn5FixByINI.exe``.

    kn5FixByINI is NOT bundled with KsEditorAt v6 — it comes from
    ac_tools_cmd (GitHub).  Search order:
    1. Same directory as ksEditorAt.exe and its parent
    2. ``ac_toolbox/ac_tools_cmd/ackn5/kn5FixByINI/``
    3. Auto-download ac_tools_cmd from GitHub → extract → search
    """
    search_dirs = [
        os.path.dirname(kseditor_exe),
        os.path.dirname(os.path.dirname(kseditor_exe)),
    ]
    ackn5_dir = os.path.join(project_root, "ac_toolbox", "ac_tools_cmd", "ackn5")
    if os.path.isdir(ackn5_dir):
        search_dirs.append(ackn5_dir)

    exe = _search_kn5fixbyini(search_dirs)
    if exe:
        return exe

    # Auto-download ac_tools_cmd for kn5FixByINI
    logger.info("kn5FixByINI not found, downloading ac_tools_cmd...")
    dest_dir = os.path.join(project_root, "ac_toolbox", "ac_tools_cmd")
    try:
        _download_and_extract_7z(_AC_TOOLS_CMD_URL, dest_dir)
    except Exception as e:
        logger.warning("Failed to download ac_tools_cmd for kn5FixByINI: %s", e)
        return None

    # Search again after download
    if os.path.isdir(ackn5_dir):
        search_dirs.append(ackn5_dir)
    return _search_kn5fixbyini(search_dirs)


def _convert_fbx_to_kn5(kseditor_exe: str, export_dir: str,
                        project_root: str) -> List[str]:
    """Convert all FBX files in *export_dir* to KN5 using ksEditorAt.

    Workflow per FBX (matching ``fbx2kn5_track.cmd``):
    1. ``ksEditorAt.exe kn5track {batch}.kn5 {batch}.fbx``
    2. ``kn5FixByINI.exe {batch}.kn5`` (post-fix based on INI)

    cwd is set to *export_dir* so the tool finds ``texture/`` and ``.fbx.ini``.
    Progress is reported proportionally to FBX file sizes.
    Returns list of successfully created KN5 file paths.
    """
    fbx_files = sorted(glob.glob(os.path.join(export_dir, "*.fbx")))
    if not fbx_files:
        logger.warning("No FBX files found in %s", export_dir)
        return []

    kn5fix_exe = _find_kn5fixbyini(kseditor_exe, project_root)
    if kn5fix_exe:
        logger.info("kn5FixByINI: %s", kn5fix_exe)
    else:
        logger.info("kn5FixByINI not found, skipping post-fix step")

    # Pre-scan: collect (path, size) for files that will actually be converted
    convert_list: list[tuple[str, int]] = []
    for fbx_path in fbx_files:
        fbx_size = os.path.getsize(fbx_path)
        if fbx_size < 1024:
            logger.info("Skipping tiny FBX (%d bytes): %s",
                        fbx_size, os.path.basename(fbx_path))
            continue
        convert_list.append((fbx_path, fbx_size))

    total_bytes = sum(sz for _, sz in convert_list)
    processed_bytes = 0

    kn5_files: List[str] = []
    for fbx_path, fbx_size in convert_list:
        basename = os.path.splitext(os.path.basename(fbx_path))[0]
        kn5_name = basename + ".kn5"
        kn5_path = os.path.join(export_dir, kn5_name)

        cmd = [kseditor_exe, "kn5track", kn5_path, fbx_path]
        logger.info("KN5 converting: %s -> %s (%.1f MB)",
                    os.path.basename(fbx_path), kn5_name, fbx_size / 1e6)

        # Report progress before conversion starts
        frac = processed_bytes / total_bytes if total_bytes else 0
        pct = int(_PCT_KN5_START + frac * (_PCT_KN5_END - _PCT_KN5_START))
        report_progress(pct, f"KN5 {os.path.basename(fbx_path)}")

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

        processed_bytes += fbx_size

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

    report_progress(_PCT_BLENDER_START, "Blender export starting")
    logger.info("Running Blender export: %s", " ".join(cmd))

    # Run Blender with stdout interception to remap progress lines
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip("\n\r")
        m = _RE_PROGRESS.search(line)
        if m:
            # Remap Blender's 0-100 → _PCT_BLENDER_START.._PCT_BLENDER_END
            raw_pct = int(m.group(1))
            frac = max(0.0, min(1.0, raw_pct / 100))
            mapped = int(_PCT_BLENDER_START
                         + frac * (_PCT_BLENDER_END - _PCT_BLENDER_START))
            report_progress(mapped, m.group(2) or "Blender export")
        else:
            # Pass through non-progress output for logging
            if line:
                logger.info("[blender] %s", line)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    report_progress(_PCT_BLENDER_END, "Copying textures")

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

    # KN5 conversion (FBX → KN5 via ksEditorAt)
    report_progress(_PCT_KN5_START, "Preparing KN5 conversion")
    kseditor = _ensure_kseditor(config, project_root)
    if kseditor:
        logger.info("Using ksEditorAt: %s", kseditor)
        kn5_files = _convert_fbx_to_kn5(kseditor, config.export_dir, project_root)
        logger.info("Converted %d KN5 files", len(kn5_files))
    else:
        logger.warning("ksEditorAt not found, skipping KN5 conversion")

    report_progress(100, "Stage 10 complete")
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
