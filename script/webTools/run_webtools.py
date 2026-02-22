import argparse
import atexit
import json
import math
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time as _time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


def repo_root_from_here() -> str:
    # script/webTools -> script -> repo root
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def find_free_port(host: str, start_port: int, max_tries: int) -> int:
    for i in range(max_tries):
        port = start_port + i
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
            return port
        except OSError:
            continue
        finally:
            try:
                s.close()
            except Exception:
                pass
    raise RuntimeError("No free port found.")


# ---------------------------------------------------------------------------
# Output paths — absolute, resolved from repo root at import time.
# Server chdir's to the analyzer directory for static file serving,
# so all data paths must be absolute.
# ---------------------------------------------------------------------------
_REPO_ROOT = repo_root_from_here()
# --- Direct stage output directories ---
_WALLS_JSON = os.path.join(_REPO_ROOT, "output", "06_ai_walls", "walls.json")
_MANUAL_WALLS_JSON = os.path.join(_REPO_ROOT, "output", "06a_manual_walls", "walls.json")
_MANUAL_WALLS_DIR = os.path.join(_REPO_ROOT, "output", "06a_manual_walls")
_GEO_META_JSON = os.path.join(_REPO_ROOT, "output", "06_ai_walls", "geo_metadata.json")
_GAME_OBJECTS_JSON = os.path.join(_REPO_ROOT, "output", "07_ai_game_objects", "game_objects.json")
_GAME_OBJECTS_GEO_META_JSON = os.path.join(_REPO_ROOT, "output", "07_ai_game_objects", "geo_metadata.json")
_CENTERLINE_JSON = os.path.join(_REPO_ROOT, "output", "07_ai_game_objects", "centerline.json")
_LAYOUTS_DIR = os.path.join(_REPO_ROOT, "output", "02a_track_layouts")
_LAYOUTS_JSON = os.path.join(_REPO_ROOT, "output", "02a_track_layouts", "layouts.json")
_MASK_FULL_MAP_DIR = os.path.join(_REPO_ROOT, "output", "02_mask_full_map")
_GAME_OBJECTS_DIR = os.path.join(_REPO_ROOT, "output", "07_ai_game_objects")
_MANUAL_GAME_OBJECTS_DIR = os.path.join(_REPO_ROOT, "output", "07a_manual_game_objects")
_MANUAL_SURFACE_MASKS_DIR = os.path.join(_REPO_ROOT, "output", "05a_manual_surface_masks")
_MANUAL_SURFACE_MASKS_EDIT_DIR = os.path.join(_REPO_ROOT, "output", "05a_manual_surface_masks", "masks")
_STAGE5_PREVIEW_DIR = os.path.join(_REPO_ROOT, "output", "05_merge_segments", "merge_preview")
_TILES_DIR = os.path.join(_REPO_ROOT, "test_images_shajing", "map")
_CONFIG_JSON = os.path.join(_REPO_ROOT, "output", "webtools_config.json")

# --- Result junction directories (downstream stages read from these) ---
_02_RESULT_DIR = os.path.join(_REPO_ROOT, "output", "02_result")
_05_RESULT_DIR = os.path.join(_REPO_ROOT, "output", "05_result")
_06_RESULT_DIR = os.path.join(_REPO_ROOT, "output", "06_result")
_07_RESULT_DIR = os.path.join(_REPO_ROOT, "output", "07_result")


def _result_or_fallback(result_dir: str, fallback_dir: str) -> str:
    """Return the result junction dir if it exists, otherwise fallback."""
    return result_dir if os.path.isdir(result_dir) else fallback_dir


def _ensure_wgs84_bounds(geo_data: dict) -> dict:
    """If geo_data['bounds'] looks like projected coords (not lat/lng), convert to WGS84.

    Detects projected CRS by checking if any bound value exceeds +-180.
    Reads the source CRS from result_masks.json to perform accurate conversion.
    Also adds 'corners' for bilinear interpolation (eliminates UTM grid convergence error).
    """
    bounds = geo_data.get("bounds")
    if not bounds:
        return geo_data

    has_corners = "corners" in geo_data
    vals = [abs(bounds.get(k, 0)) for k in ("north", "south", "east", "west")]
    if all(v <= 180 for v in vals) and has_corners:
        return geo_data  # already WGS84 with corners

    # Read CRS from result_masks.json
    crs_str = None
    raw_bounds = None
    for candidate in [
        os.path.join(_MASK_FULL_MAP_DIR, "result_masks.json"),
        os.path.join(_02_RESULT_DIR, "result_masks.json"),
    ]:
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    rm = json.load(f)
                geo = rm.get("meta", {}).get("geo", {})
                crs_str = geo.get("crs")
                raw_bounds = geo.get("bounds")
                if crs_str:
                    break
            except Exception:
                pass

    if not crs_str:
        # Add axis-aligned corners if missing
        if not has_corners:
            geo_data = dict(geo_data)
            geo_data["corners"] = {
                "top_left": [bounds["north"], bounds["west"]],
                "top_right": [bounds["north"], bounds["east"]],
                "bottom_left": [bounds["south"], bounds["west"]],
                "bottom_right": [bounds["south"], bounds["east"]],
            }
        return geo_data

    try:
        script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from pipeline_config import bounds_to_wgs84
        # Use raw UTM bounds from result_masks.json (not the converted ones)
        src = raw_bounds or {"left": bounds["west"], "bottom": bounds["south"],
                             "right": bounds["east"], "top": bounds["north"]}
        wgs84 = bounds_to_wgs84(
            crs_str,
            src.get("left", 0), src.get("bottom", 0),
            src.get("right", 0), src.get("top", 0),
        )
        geo_data = dict(geo_data)
        geo_data["bounds"] = {k: wgs84[k] for k in ("north", "south", "east", "west")}
        geo_data["corners"] = wgs84.get("corners")
    except Exception as e:
        print(f"[webtools] WARNING: WGS84 conversion failed: {e}")

    return geo_data


# ---------------------------------------------------------------------------
# Surface mask tile cache (lazy-loaded in-memory numpy arrays)
# ---------------------------------------------------------------------------
_surface_mask_cache = {}    # tag -> numpy.ndarray (H, W, uint8)
_surface_meta_cache = None
_cache_lock = threading.Lock()


def _get_surface_meta():
    global _surface_meta_cache
    if _surface_meta_cache is None:
        p = os.path.join(_MANUAL_SURFACE_MASKS_EDIT_DIR, "surface_masks.json")
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                _surface_meta_cache = json.load(f)
    return _surface_meta_cache


def _get_surface_mask(tag):
    """Load full mask into memory cache, return numpy array or None."""
    with _cache_lock:
        if tag in _surface_mask_cache:
            return _surface_mask_cache[tag]
    import cv2
    manual = os.path.join(_MANUAL_SURFACE_MASKS_EDIT_DIR, f"{tag}_mask.png")
    stage5 = os.path.join(_STAGE5_PREVIEW_DIR, f"{tag}_merged.png")
    path = manual if os.path.isfile(manual) else stage5
    if not os.path.isfile(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    with _cache_lock:
        _surface_mask_cache[tag] = mask
    return mask


def _patch_surface_tile(tag, col, row, tile_png_bytes):
    """Decode tile PNG and patch into the cached full mask."""
    import cv2
    import numpy as np
    meta = _get_surface_meta()
    if not meta:
        return False
    ts = meta.get("tile_size", 512)
    mask = _get_surface_mask(tag)
    if mask is None:
        return False
    arr = np.frombuffer(tile_png_bytes, dtype=np.uint8)
    tile = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if tile is None:
        return False
    y0, x0 = row * ts, col * ts
    h = min(ts, mask.shape[0] - y0)
    w = min(ts, mask.shape[1] - x0)
    if h <= 0 or w <= 0:
        return False
    with _cache_lock:
        mask[y0:y0 + h, x0:x0 + w] = tile[:h, :w]
    return True


def _flush_surface_masks():
    """Write all cached masks to disk."""
    import cv2
    os.makedirs(_MANUAL_SURFACE_MASKS_EDIT_DIR, exist_ok=True)
    with _cache_lock:
        tags = list(_surface_mask_cache.keys())
        for tag in tags:
            path = os.path.join(_MANUAL_SURFACE_MASKS_EDIT_DIR, f"{tag}_mask.png")
            cv2.imwrite(path, _surface_mask_cache[tag])
    return len(tags)


def _safe_layout_name(name: str) -> str:
    """Convert layout name to filesystem-safe filename."""
    return re.sub(r'[^\w\-]', '_', name).strip('_') or "unnamed"


def _layout_read_path(layout_name: str, filename: str) -> str:
    """Read from 07_result junction (points to 07 or 07a)."""
    safe = _safe_layout_name(layout_name)
    result_dir = _result_or_fallback(_07_RESULT_DIR, _GAME_OBJECTS_DIR)
    p = os.path.join(result_dir, safe, filename)
    if os.path.isfile(p):
        return p
    # Fallback: try direct directories
    for base in [_MANUAL_GAME_OBJECTS_DIR, _GAME_OBJECTS_DIR]:
        p = os.path.join(base, safe, filename)
        if os.path.isfile(p):
            return p
    return os.path.join(_GAME_OBJECTS_DIR, safe, filename)


def _layout_write_path(layout_name: str, filename: str) -> str:
    """Write always goes to 7a."""
    safe = _safe_layout_name(layout_name)
    p = os.path.join(_MANUAL_GAME_OBJECTS_DIR, safe, filename)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _auto_merge_manual() -> None:
    """After any write to 7a, re-merge all layouts into top-level game_objects.json."""
    if not os.path.isdir(_MANUAL_GAME_OBJECTS_DIR):
        return
    all_objects = []
    layout_names = []
    for entry in sorted(os.listdir(_MANUAL_GAME_OBJECTS_DIR)):
        go_path = os.path.join(_MANUAL_GAME_OBJECTS_DIR, entry, "game_objects.json")
        if os.path.isfile(go_path):
            with open(go_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            layout_name = data.get("layout_name", entry)
            layout_names.append(layout_name)
            for obj in data.get("objects", []):
                obj["_layout"] = layout_name
                all_objects.append(obj)

    merged = {
        "track_direction": "clockwise",
        "layouts": layout_names,
        "objects": all_objects,
    }
    merged_path = os.path.join(_MANUAL_GAME_OBJECTS_DIR, "game_objects.json")
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def _modelscale_image_path() -> str:
    """Find modelscale image in 02_result (or stage 2 output)."""
    search_dir = _result_or_fallback(_02_RESULT_DIR, _MASK_FULL_MAP_DIR)
    if os.path.isdir(search_dir):
        for f in os.listdir(search_dir):
            if f.endswith("_modelscale.png"):
                return os.path.join(search_dir, f)
    return ""


# ---------------------------------------------------------------------------
# Pipeline stage metadata (for Dashboard)
# ---------------------------------------------------------------------------
PIPELINE_STAGE_META = [
    {"id": "prep",               "num": "0",  "name": "准备",       "type": "config",
     "desc": "配置 GeoTIFF、3D Tiles、输出目录等基本参数", "output_dir": ""},
    {"id": "b3dm_convert",       "num": "1",  "name": "B3DM转换",   "type": "auto",
     "desc": "将 3D Tiles (B3DM) 转换为 GLB 格式", "output_dir": "01_b3dm_convert"},
    {"id": "mask_full_map",      "num": "2",  "name": "全图分割",   "type": "auto",
     "desc": "SAM3 语义分割全图（8类标签）+ VLM 高分辨率图 + 中心孔洞修复", "output_dir": "02_mask_full_map"},
    {"id": "track_layouts",      "num": "2a", "name": "赛道布局",   "type": "manual",
     "desc": "赛道布局管理（多布局支持）", "output_dir": "02a_track_layouts",
     "editor": "layout_editor.html"},
    {"id": "clip_full_map",      "num": "3",  "name": "全图裁剪",   "type": "auto",
     "desc": "将全图智能裁剪为瓦片 (clips)", "output_dir": "03_clip_full_map"},
    {"id": "mask_on_clips",      "num": "4",  "name": "瓦片分割",   "type": "auto",
     "desc": "逐瓦片精细分割（含回退提示词）", "output_dir": "04_mask_on_clips"},
    {"id": "merge_segments", "num": "5",  "name": "合并分割", "type": "auto",
     "desc": "多瓦片分割结果合并 + 地理坐标 → Blender 坐标转换", "output_dir": "05_merge_segments"},
    {"id": "manual_surface_masks","num": "5a", "name": "表面编辑",   "type": "manual",
     "desc": "手动编辑表面 mask（可选）", "output_dir": "05a_manual_surface_masks",
     "editor": "surface_editor.html"},
    {"id": "ai_walls",           "num": "6",  "name": "围墙生成",   "type": "auto",
     "desc": "程序化围墙生成（SAM3 洪水填充，无 LLM 依赖）", "output_dir": "06_ai_walls"},
    {"id": "manual_walls",       "num": "6a", "name": "围墙编辑",   "type": "manual",
     "desc": "手动编辑围墙边界（可选）", "output_dir": "06a_manual_walls",
     "editor": "wall_editor.html"},
    {"id": "ai_game_objects",    "num": "7",  "name": "游戏对象",   "type": "auto",
     "desc": "混合游戏对象生成（VLM 布局对象 + 程序化计时点）", "output_dir": "07_ai_game_objects"},
    {"id": "manual_game_objects","num": "7a", "name": "对象编辑",   "type": "manual",
     "desc": "手动编辑游戏对象位置/朝向（可选）", "output_dir": "07a_manual_game_objects",
     "editor": "gameobjects_editor.html"},
    {"id": "blender_polygons",   "num": "8",  "name": "多边形生成", "type": "auto",
     "desc": "Blender 批处理生成 2D Curve + Mesh", "output_dir": "08_blender_polygons"},
    {"id": "blender_automate",   "num": "9",  "name": "Blender集成", "type": "auto",
     "desc": "Blender 无头自动化（加载瓦片 → 精炼 → 表面提取 → 导入 → 保存）",
     "output_dir": "09_blender_automate"},
    {"id": "manual_blender",    "num": "9a", "name": "Blender编辑", "type": "manual",
     "desc": "手动编辑 Blender 文件（在 Blender 中直接打开编辑）",
     "output_dir": "09a_manual_blender"},
    {"id": "model_export",      "num": "10", "name": "模型导出",     "type": "auto",
     "desc": "清理 → 拆分超大网格 → 碰撞命名 → 分批 FBX 导出（Assetto Corsa）",
     "output_dir": "10_model_export"},
    {"id": "track_packaging",   "num": "11", "name": "赛道打包",    "type": "auto",
     "desc": "组装最终 AC 赛道 Mod 文件夹（KN5 + INI + UI + 地图）",
     "output_dir": "11_track_packaging"},
]

# Map auto stage ids to sam3_track_gen stage names
_STAGE_ID_TO_PIPELINE = {
    "b3dm_convert": "b3dm_convert",
    "mask_full_map": "mask_full_map",
    "clip_full_map": "clip_full_map",
    "mask_on_clips": "mask_on_clips",
    "merge_segments": "merge_segments",
    "blender_polygons": "blender_polygons",
    "ai_walls": "ai_walls",
    "ai_game_objects": "ai_game_objects",
    "blender_automate": "blender_automate",
    "model_export": "model_export",
    "track_packaging": "track_packaging",
}


# ---------------------------------------------------------------------------
# PipelineRunner — background pipeline execution with SSE streaming
# ---------------------------------------------------------------------------
class PipelineRunner:
    """Manages background execution of pipeline stages.

    Uses a LOG FILE instead of stdout pipe to capture subprocess output.
    This eliminates the Windows pipe-buffer deadlock entirely: the subprocess
    writes to a file (never blocks), and a tail-reader thread reads the file
    and broadcasts to SSE clients.
    """

    def __init__(self):
        self.process = None
        self.current_stage = None
        self.stages_requested = []
        self.log_lines = []
        self.sse_clients = []          # list of queue.Queue
        self._lock = threading.Lock()
        self._stage_status = {}        # stage_id -> "not_started" | "running" | "completed" | "error"
        self._log_file_handle = None   # file handle for subprocess stdout
        self._stop_event = threading.Event()

    def run_stages(self, stage_ids, config_dict):
        """Launch pipeline stages in a background process."""
        with self._lock:
            if self.process and self.process.poll() is None:
                raise RuntimeError("Pipeline already running")

        output_base = config_dict.get("output_dir", os.path.join(_REPO_ROOT, "output"))

        # Map stage_ids to pipeline stage names
        pipeline_stages = []
        for sid in stage_ids:
            mapped = _STAGE_ID_TO_PIPELINE.get(sid)
            if mapped and mapped not in pipeline_stages:
                pipeline_stages.append(mapped)

        if not pipeline_stages:
            raise ValueError("No valid auto stages to run")

        cmd = [sys.executable, os.path.join(_REPO_ROOT, "script", "sam3_track_gen.py")]
        for s in pipeline_stages:
            cmd.extend(["--stage", s])
        cmd.extend(["--output-dir", config_dict.get("output_dir", "output")])
        if config_dict.get("geotiff_path"):
            cmd.extend(["--geotiff", config_dict["geotiff_path"]])
        if config_dict.get("tiles_dir"):
            cmd.extend(["--tiles-dir", config_dict["tiles_dir"]])
        if config_dict.get("inpaint_model"):
            cmd.extend(["--inpaint-model", config_dict["inpaint_model"]])
        if config_dict.get("max_workers"):
            cmd.extend(["--max-workers", str(config_dict["max_workers"])])
        if config_dict.get("road_mask_offset_px") is not None:
            cmd.extend(["--road-mask-offset", str(config_dict["road_mask_offset_px"])])

        # Stage 5 options
        if config_dict.get("s5_road_gap_close_m") is not None:
            cmd.extend(["--s5-road-gap-close", str(config_dict["s5_road_gap_close_m"])])
        if config_dict.get("s5_kerb_narrow_max_width_m") is not None:
            cmd.extend(["--s5-kerb-narrow-width", str(config_dict["s5_kerb_narrow_max_width_m"])])
        if config_dict.get("s5_kerb_narrow_adjacency_m") is not None:
            cmd.extend(["--s5-kerb-narrow-adjacency", str(config_dict["s5_kerb_narrow_adjacency_m"])])

        # Stage 8 options
        if config_dict.get("s8_generate_curves"):
            cmd.append("--generate-curves")
        if config_dict.get("s8_gap_fill_enabled") is False:
            cmd.append("--no-gap-fill")
        if config_dict.get("s8_gap_fill_threshold_m"):
            cmd.extend(["--gap-fill-threshold", str(config_dict["s8_gap_fill_threshold_m"])])
        if config_dict.get("s8_gap_fill_default_tag"):
            cmd.extend(["--gap-fill-default-tag", config_dict["s8_gap_fill_default_tag"]])

        # Stage 9 options
        if config_dict.get("s9_base_level"):
            cmd.extend(["--base-level", str(config_dict["s9_base_level"])])
        if config_dict.get("s9_target_level"):
            cmd.extend(["--target-level", str(config_dict["s9_target_level"])])
        if not config_dict.get("s9_import_walls", True):
            cmd.append("--no-walls")
        if not config_dict.get("s9_import_game_objects", True):
            cmd.append("--no-game-objects")
        if not config_dict.get("s9_extract_surfaces", True):
            cmd.append("--no-surfaces")
        if not config_dict.get("s9_convert_textures", True):
            cmd.append("--no-textures")
        if not config_dict.get("s9_background", True):
            cmd.append("--no-background")
        refine_tags = config_dict.get("s9_refine_tags", ["road"])
        if refine_tags:
            cmd.extend(["--refine-tags", ",".join(refine_tags)])
        if config_dict.get("s9_edge_simplify"):
            cmd.extend(["--edge-simplify", str(config_dict["s9_edge_simplify"])])
        if config_dict.get("s9_mesh_simplify"):
            cmd.append("--mesh-simplify")
            if config_dict.get("s9_mesh_weld_distance"):
                cmd.extend(["--mesh-weld-distance", str(config_dict["s9_mesh_weld_distance"])])
            if config_dict.get("s9_mesh_decimate_ratio"):
                cmd.extend(["--mesh-decimate-ratio", str(config_dict["s9_mesh_decimate_ratio"])])
        for dtag in ["road", "kerb", "grass", "sand", "road2"]:
            val = config_dict.get(f"s9_density_{dtag}")
            if val:
                cmd.extend([f"--density-{dtag}", str(val)])

        # Stage 10 options
        if config_dict.get("s10_max_vertices"):
            cmd.extend(["--max-vertices", str(config_dict["s10_max_vertices"])])
        if config_dict.get("s10_max_batch_mb"):
            cmd.extend(["--max-batch-mb", str(config_dict["s10_max_batch_mb"])])
        if config_dict.get("s10_fbx_scale"):
            cmd.extend(["--fbx-scale", str(config_dict["s10_fbx_scale"])])
        if config_dict.get("s10_ks_ambient") is not None:
            cmd.extend(["--ks-ambient", str(config_dict["s10_ks_ambient"])])
        if config_dict.get("s10_ks_diffuse") is not None:
            cmd.extend(["--ks-diffuse", str(config_dict["s10_ks_diffuse"])])
        if config_dict.get("s10_ks_emissive") is not None:
            cmd.extend(["--ks-emissive", str(config_dict["s10_ks_emissive"])])
        if config_dict.get("s10_kseditor_exe"):
            cmd.extend(["--kseditor-exe", config_dict["s10_kseditor_exe"]])

        # Stage 11 options
        if config_dict.get("s11_track_name"):
            cmd.extend(["--track-name", config_dict["s11_track_name"]])
        if config_dict.get("s11_track_author"):
            cmd.extend(["--track-author", config_dict["s11_track_author"]])
        if config_dict.get("s11_track_country"):
            cmd.extend(["--track-country", config_dict["s11_track_country"]])
        if config_dict.get("s11_track_city"):
            cmd.extend(["--track-city", config_dict["s11_track_city"]])
        if config_dict.get("s11_track_tags"):
            tags = config_dict["s11_track_tags"]
            if isinstance(tags, list):
                tags = ",".join(tags)
            cmd.extend(["--track-tags", tags])
        if config_dict.get("s11_track_year"):
            cmd.extend(["--track-year", str(config_dict["s11_track_year"])])
        if config_dict.get("s11_pitboxes"):
            cmd.extend(["--pitboxes", str(config_dict["s11_pitboxes"])])
        if config_dict.get("track_description"):
            cmd.extend(["--track-description", config_dict["track_description"]])
        if config_dict.get("s11_track_url"):
            cmd.extend(["--track-url", config_dict["s11_track_url"]])
        if config_dict.get("s11_layout_display_names"):
            cmd.extend(["--layout-display-names", config_dict["s11_layout_display_names"]])
        if config_dict.get("s11_llm_description") is False:
            cmd.append("--no-llm-description")
        if config_dict.get("s11_llm_preview") is False:
            cmd.append("--no-llm-preview")

        # Set up result junctions before running
        try:
            script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            from pipeline_config import PipelineConfig, _is_junction, _remove_junction_safe
            pc = PipelineConfig(
                geotiff_path=config_dict.get("geotiff_path", ""),
                tiles_dir=config_dict.get("tiles_dir", ""),
                output_dir=config_dict.get("output_dir", os.path.join(_REPO_ROOT, "output")),
            ).resolve()
            manual_cfg = config_dict.get("manual_stages", {})
            # Also load from webtools_config.json
            wt_cfg = _load_webtools_config()
            saved_manual = wt_cfg.get("manual_stages", {})
            saved_manual.update(manual_cfg)
            pc.setup_result_junctions(saved_manual)
        except Exception as e:
            print(f"[pipeline] WARNING: Failed to set up junctions: {e}")

        # Clean output directories for stages being re-run (best-effort)
        for sid in stage_ids:
            meta = next((m for m in PIPELINE_STAGE_META if m["id"] == sid), None)
            if meta and meta.get("output_dir"):
                stage_out = os.path.join(output_base, meta["output_dir"])
                # Check if it's a junction — never rmtree a junction
                try:
                    from pipeline_config import _is_junction, _remove_junction_safe
                    if _is_junction(stage_out):
                        continue  # don't clean junction targets
                except ImportError:
                    pass
                if os.path.isdir(stage_out):
                    try:
                        shutil.rmtree(stage_out)
                    except Exception:
                        # Directory locked — clear contents best-effort
                        for root, dirs, files in os.walk(stage_out, topdown=False):
                            for f in files:
                                try:
                                    os.remove(os.path.join(root, f))
                                except Exception:
                                    pass
                            for d in dirs:
                                try:
                                    os.rmdir(os.path.join(root, d))
                                except Exception:
                                    pass
                os.makedirs(stage_out, exist_ok=True)

        # Build stage ID list for status tracking
        all_stage_ids = []
        for ps in pipeline_stages:
            # Reverse-map pipeline name to stage_id (they happen to be the same)
            sid = next((k for k, v in _STAGE_ID_TO_PIPELINE.items() if v == ps), ps)
            if sid not in all_stage_ids:
                all_stage_ids.append(sid)

        with self._lock:
            self.log_lines = []
            self.stages_requested = all_stage_ids
            for sid in all_stage_ids:
                self._stage_status[sid] = "pending"

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        # On Windows, prevent child process from opening a console window
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        # Write subprocess output to a LOG FILE instead of a pipe.
        # This eliminates the Windows pipe-buffer deadlock entirely:
        # the subprocess writes to a file (never blocks), and a tail-reader
        # thread reads the file and broadcasts to SSE clients.
        log_path = os.path.join(output_base, "pipeline.log")
        self._log_file_handle = open(log_path, "w", encoding="utf-8")
        self._stop_event.clear()

        self.process = subprocess.Popen(
            cmd, stdout=self._log_file_handle, stderr=subprocess.STDOUT,
            cwd=_REPO_ROOT, env=env, **kwargs,
        )
        # Mark first stage as "running" immediately (initialization phase)
        with self._lock:
            if stage_ids:
                self.current_stage = stage_ids[0]
                self._stage_status[stage_ids[0]] = "running"
        threading.Thread(target=self._tail_log, args=(log_path,), daemon=True).start()
        self._broadcast("pipeline_start", {"stages": stage_ids})

    def _tail_log(self, log_path):
        """Tail the log file and broadcast new lines to SSE clients.

        The subprocess writes to this file directly (no pipe involved).
        This thread reads new lines and broadcasts them.  Even if this
        thread crashes, the subprocess is completely unaffected.
        """
        import time

        proc = self.process
        log_batch = []
        last_flush = time.monotonic()
        FLUSH_INTERVAL = 0.1

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                while not self._stop_event.is_set():
                    line = f.readline()
                    if line:
                        line = line.rstrip()
                        if not line:
                            continue
                        with self._lock:
                            self.log_lines.append(line)

                        is_stage_line = "=== Stage" in line

                        if is_stage_line:
                            if log_batch:
                                self._safe_broadcast("log", "\n".join(log_batch))
                                log_batch = []
                                last_flush = time.monotonic()
                            self._safe_broadcast("log", line)
                            for sid, pname in _STAGE_ID_TO_PIPELINE.items():
                                if pname in line:
                                    if self.current_stage and self.current_stage != sid:
                                        self._stage_status[self.current_stage] = "completed"
                                        self._safe_broadcast("stage_complete", {"stage": self.current_stage})
                                    self.current_stage = sid
                                    self._stage_status[sid] = "running"
                                    self._safe_broadcast("stage_start", {"stage": sid})
                                    break
                        else:
                            log_batch.append(line)
                            now = time.monotonic()
                            if now - last_flush >= FLUSH_INTERVAL or len(log_batch) >= 100:
                                self._safe_broadcast("log", "\n".join(log_batch))
                                log_batch = []
                                last_flush = now
                    else:
                        # No new content — check if process is still alive
                        if proc.poll() is not None:
                            # Process exited; do one final read pass
                            remaining = f.read()
                            if remaining:
                                for rline in remaining.splitlines():
                                    rline = rline.rstrip()
                                    if rline:
                                        log_batch.append(rline)
                            break
                        time.sleep(0.05)  # brief sleep, then retry

            # Flush remaining batch
            if log_batch:
                self._safe_broadcast("log", "\n".join(log_batch))
        except Exception:
            pass
        finally:
            rc = proc.wait()
            # Close the log file handle
            try:
                if self._log_file_handle:
                    self._log_file_handle.close()
                    self._log_file_handle = None
            except Exception:
                pass
            with self._lock:
                if self.current_stage and self._stage_status.get(self.current_stage) == "running":
                    self._stage_status[self.current_stage] = "completed" if rc == 0 else "error"
                if rc == 0:
                    for sid in self.stages_requested:
                        if self._stage_status.get(sid) == "pending":
                            self._stage_status[sid] = "completed"
            self._safe_broadcast("pipeline_done", {"returncode": rc})
            self.current_stage = None
            if self.process is proc:
                self.process = None

    def _safe_broadcast(self, event_type, data):
        """Broadcast that never raises."""
        try:
            self._broadcast(event_type, data)
        except Exception:
            pass

    def stop(self):
        self._stop_event.set()
        proc = self.process
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            self._safe_broadcast("pipeline_stop", {})
        # Close log file handle
        try:
            if self._log_file_handle:
                self._log_file_handle.close()
                self._log_file_handle = None
        except Exception:
            pass

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def get_status(self):
        """Return status dict for all stages."""
        result = {}
        for meta in PIPELINE_STAGE_META:
            sid = meta["id"]
            if sid in self._stage_status:
                result[sid] = self._stage_status[sid]
            else:
                # Check if output directory has content
                od = meta.get("output_dir")
                if od:
                    full = os.path.join(_REPO_ROOT, "output", od)
                    if os.path.isdir(full) and os.listdir(full):
                        result[sid] = "completed"
                    else:
                        result[sid] = "not_started"
                else:
                    result[sid] = "not_started"
        return result

    def add_sse_client(self):
        q = queue.Queue()
        with self._lock:
            self.sse_clients.append(q)
        return q

    def remove_sse_client(self, q):
        with self._lock:
            if q in self.sse_clients:
                self.sse_clients.remove(q)

    def _broadcast(self, event_type, data):
        msg = json.dumps({"type": event_type, "data": data})
        with self._lock:
            dead = []
            for q in self.sse_clients:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self.sse_clients.remove(q)


# Global runner instance
_pipeline_runner = PipelineRunner()
atexit.register(_pipeline_runner.stop)  # kill subprocess on server exit


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------
def _load_webtools_config():
    if os.path.isfile(_CONFIG_JSON):
        with open(_CONFIG_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "geotiff_path": os.path.join(_REPO_ROOT, "test_images_shajing", "result.tif"),
        "tiles_dir": os.path.join(_REPO_ROOT, "test_images_shajing", "b3dm"),
        "map_tiles_dir": os.path.join(_REPO_ROOT, "test_images_shajing", "map"),
        "output_dir": os.path.join(_REPO_ROOT, "output"),
        "blender_exe": r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        "track_direction": "clockwise",
        "inpaint_model": "gemini-2.5-flash-image",
        "gemini_api_key": "***REDACTED_GEMINI_KEY***",
    }


def _save_webtools_config(cfg):
    os.makedirs(os.path.dirname(_CONFIG_JSON), exist_ok=True)
    with open(_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


class ApiHandler(SimpleHTTPRequestHandler):
    """Extends static file serving with JSON API endpoints for editors."""

    def send_error(self, code, message=None, explain=None):
        """Override to handle non-ASCII error messages (Windows CJK errors)."""
        if message:
            message = message.encode("ascii", "replace").decode("ascii")
        if explain:
            explain = explain.encode("ascii", "replace").decode("ascii")
        super().send_error(code, message, explain)

    def do_GET(self):
        # --- Existing endpoints ---
        if self.path == "/api/walls":
            # Read from 07_result junction
            result_walls = os.path.join(_result_or_fallback(_06_RESULT_DIR, os.path.dirname(_WALLS_JSON)), "walls.json")
            self._serve_json_file(result_walls)
        elif self.path == "/api/geo_metadata":
            # Read from 06_result, fallback to 06_ai_walls, then 02_mask_full_map
            result_meta = os.path.join(_result_or_fallback(_06_RESULT_DIR, os.path.dirname(_GEO_META_JSON)), "geo_metadata.json")
            if not os.path.isfile(result_meta):
                result_meta = os.path.join(_MASK_FULL_MAP_DIR, "geo_metadata.json")
            self._serve_geo_metadata(result_meta)
        elif self.path == "/api/game_objects":
            # Read from 08_result junction
            result_go = os.path.join(_result_or_fallback(_07_RESULT_DIR, _GAME_OBJECTS_DIR), "game_objects.json")
            self._serve_json_file(result_go)
        elif self.path == "/api/game_objects/geo_metadata":
            # Read from 07_result, then 06_result, then 02
            result_go_dir = _result_or_fallback(_07_RESULT_DIR, _GAME_OBJECTS_DIR)
            go_meta = os.path.join(result_go_dir, "geo_metadata.json")
            if not os.path.isfile(go_meta):
                result_w_dir = _result_or_fallback(_06_RESULT_DIR, os.path.dirname(_GEO_META_JSON))
                go_meta = os.path.join(result_w_dir, "geo_metadata.json")
            if not os.path.isfile(go_meta):
                go_meta = os.path.join(_MASK_FULL_MAP_DIR, "geo_metadata.json")
            self._serve_geo_metadata(go_meta)
        elif self.path == "/api/centerline":
            # Read from 08_result
            result_go_dir = _result_or_fallback(_07_RESULT_DIR, _GAME_OBJECTS_DIR)
            self._serve_json_file(os.path.join(result_go_dir, "centerline.json"))

        # --- Track layouts (read from 02_result) ---
        elif self.path == "/api/track_layouts":
            result_dir = _result_or_fallback(_02_RESULT_DIR, _MASK_FULL_MAP_DIR)
            layouts_path = os.path.join(result_dir, "layouts.json")
            # Fallback to old location for backward compat
            if not os.path.isfile(layouts_path):
                layouts_path = _LAYOUTS_JSON
            self._serve_json_file(layouts_path)

        elif self.path.startswith("/api/layout_mask/"):
            name = self.path[len("/api/layout_mask/"):]
            name = _safe_layout_name(name)
            # Read from 02_result, fallback to 02a
            result_dir = _result_or_fallback(_02_RESULT_DIR, _LAYOUTS_DIR)
            mask_path = os.path.join(result_dir, f"{name}.png")
            if not os.path.isfile(mask_path):
                mask_path = os.path.join(_LAYOUTS_DIR, f"{name}.png")
            self._serve_binary_file(mask_path, "image/png")

        elif self.path.startswith("/api/mask_overlay/"):
            tag = self.path[len("/api/mask_overlay/"):]
            tag = re.sub(r'[^\w]', '', tag)
            result_dir = _result_or_fallback(_02_RESULT_DIR, _MASK_FULL_MAP_DIR)
            mask_path = os.path.join(result_dir, f"{tag}_mask.png")
            self._serve_binary_file(mask_path, "image/png")

        elif self.path == "/api/modelscale_geo_metadata":
            # Geo metadata from stage 2 output (matches modelscale image dims).
            # 02_result may point to 02a (no geo_metadata), so always fall back
            # to the authoritative 02_mask_full_map directory.
            result_dir = _result_or_fallback(_02_RESULT_DIR, _MASK_FULL_MAP_DIR)
            meta_path = os.path.join(result_dir, "geo_metadata.json")
            if not os.path.isfile(meta_path):
                meta_path = os.path.join(_MASK_FULL_MAP_DIR, "geo_metadata.json")
            self._serve_geo_metadata(meta_path)

        elif self.path == "/api/modelscale_image":
            # Find modelscale image in 02_result or stage 2 output
            search_dir = _result_or_fallback(_02_RESULT_DIR, _MASK_FULL_MAP_DIR)
            found = None
            if os.path.isdir(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith("_modelscale.png"):
                        found = os.path.join(search_dir, f)
                        break
            if found:
                self._serve_binary_file(found, "image/png")
            else:
                self.send_error(404, "No modelscale image found")

        # --- Surface masks (5a manual editing) ---
        elif self.path == "/api/surface_masks":
            sf_json = os.path.join(_MANUAL_SURFACE_MASKS_EDIT_DIR, "surface_masks.json")
            self._serve_json_file(sf_json)

        elif self.path.startswith("/api/surface_tile_index/"):
            tag = self.path.split("?", 1)[0][len("/api/surface_tile_index/"):]
            tag = re.sub(r'[^\w]', '', tag)
            self._serve_surface_tile_index(tag)

        elif self.path.startswith("/api/surface_mask/"):
            tag = self.path[len("/api/surface_mask/"):]
            tag = re.sub(r'[^\w]', '', tag)
            # Priority: 5a masks/ > Stage 5 merge_preview
            manual_path = os.path.join(_MANUAL_SURFACE_MASKS_EDIT_DIR, f"{tag}_mask.png")
            stage5_path = os.path.join(_STAGE5_PREVIEW_DIR, f"{tag}_merged.png")
            if os.path.isfile(manual_path):
                self._serve_binary_file(manual_path, "image/png")
            elif os.path.isfile(stage5_path):
                self._serve_binary_file(stage5_path, "image/png")
            else:
                self.send_error(404, f"No mask for tag '{tag}'")

        # --- Surface tiles (tile-based editing) ---
        elif self.path.startswith("/api/surface_tile/"):
            # Strip query string (?t=...) for cache-busting support
            tile_path = self.path.split("?", 1)[0]
            parts = tile_path[len("/api/surface_tile/"):].split("/")
            if len(parts) == 2:
                tag = re.sub(r'[^\w]', '', parts[0])
                m = re.match(r'(\d+)_(\d+)\.png$', parts[1])
                if m:
                    self._serve_surface_tile(tag, int(m.group(1)), int(m.group(2)))
                else:
                    self.send_error(400, "Bad tile coords")
            else:
                self.send_error(400, "Bad surface tile path")

        # --- Per-layout centerline & game objects (7a > 7 priority) ---
        elif self.path.startswith("/api/layout_centerline/"):
            name = self.path[len("/api/layout_centerline/"):]
            cl_path = _layout_read_path(name, "centerline.json")
            self._serve_json_file(cl_path)

        elif self.path.startswith("/api/layout_game_objects/"):
            name = self.path[len("/api/layout_game_objects/"):]
            go_path = _layout_read_path(name, "game_objects.json")
            self._serve_json_file(go_path)

        # --- Map tiles proxy (with overzoom fallback) ---
        elif self.path.startswith("/tiles/"):
            parts = self.path[len("/tiles/"):].rstrip("/").split("/")
            if len(parts) == 3:
                cfg = _load_webtools_config()
                tiles_dir = cfg.get("map_tiles_dir", _TILES_DIR)
                try:
                    z, x = int(parts[0]), int(parts[1])
                    y_str = parts[2].replace(".png", "")
                    y = int(y_str)
                except ValueError:
                    self.send_error(400, "Bad tile coordinates")
                    return
                self._serve_tile(tiles_dir, z, x, y)
            else:
                self.send_error(400, "Bad tile path — expect /tiles/{z}/{x}/{y}.png")

        # --- Pipeline API ---
        elif self.path == "/api/pipeline/stages":
            self._serve_pipeline_stages()
        elif self.path == "/api/pipeline/status":
            self._serve_pipeline_status()
        elif self.path == "/api/pipeline/config":
            self._serve_pipeline_config()
        elif self.path.startswith("/api/files/list"):
            self._serve_file_list()
        elif self.path.startswith("/api/files/preview"):
            self._serve_file_preview()
        elif self.path == "/api/sse/pipeline":
            self._serve_sse()
        elif self.path == "/api/pipeline/manual_stages":
            self._serve_manual_stages()

        else:
            super().do_GET()

    def end_headers(self):
        """Add no-cache for static dev assets (.js/.css/.html)."""
        if self.path and any(self.path.endswith(ext) for ext in (".js", ".css", ".html")):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_POST(self):
        if self.path == "/api/walls":
            self._save_walls()
        elif self.path == "/api/game_objects":
            self._save_game_objects()

        # --- Track layouts ---
        elif self.path == "/api/track_layouts":
            self._save_track_layouts()

        elif self.path.startswith("/api/layout_mask/"):
            name = self.path[len("/api/layout_mask/"):]
            self._save_layout_mask(name)

        # --- Surface masks (5a) ---
        elif self.path.startswith("/api/surface_mask/"):
            tag = self.path[len("/api/surface_mask/"):]
            self._save_surface_mask(tag)

        elif self.path.startswith("/api/surface_tile/"):
            parts = self.path[len("/api/surface_tile/"):].split("/")
            if len(parts) == 2:
                tag = re.sub(r'[^\w]', '', parts[0])
                m = re.match(r'(\d+)_(\d+)$', parts[1])
                if m:
                    self._handle_save_surface_tile(tag, int(m.group(1)), int(m.group(2)))
                else:
                    self.send_error(400, "Bad tile coords")
            else:
                self.send_error(400, "Bad surface tile path")

        elif self.path == "/api/surface_flush":
            self._handle_surface_flush()

        # --- Per-layout centerline & game objects (always write to 7a) ---
        elif self.path.startswith("/api/layout_centerline/"):
            name = self.path[len("/api/layout_centerline/"):]
            cl_path = _layout_write_path(name, "centerline.json")
            self._save_json_body(cl_path)

        elif self.path.startswith("/api/layout_game_objects/"):
            name = self.path[len("/api/layout_game_objects/"):]
            go_path = _layout_write_path(name, "game_objects.json")
            self._save_json_body(go_path)
            _auto_merge_manual()

        elif self.path == "/api/centerline/regenerate":
            self._regenerate_centerline()

        elif self.path == "/api/vlm_objects/regenerate":
            self._regenerate_vlm_objects()

        elif self.path == "/api/regenerate_from_start":
            self._regenerate_from_start_point()

        # --- Pipeline control ---
        elif self.path == "/api/pipeline/run":
            self._handle_pipeline_run()
        elif self.path == "/api/pipeline/stop":
            self._handle_pipeline_stop()
        elif self.path == "/api/pipeline/config":
            self._handle_pipeline_config_save()
        elif self.path == "/api/pipeline/manual_stages":
            self._handle_manual_stage_toggle()

        else:
            self.send_error(404, "Not Found")

    # -- helpers --

    def _serve_json_file(self, path: str):
        if not os.path.isfile(path):
            self.send_error(404, f"File not found: {path}")
            return
        try:
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def _serve_geo_metadata(self, path: str):
        """Serve geo_metadata.json, converting projected bounds to WGS84 if needed."""
        if not os.path.isfile(path):
            self.send_error(404, f"File not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                geo_data = json.load(f)
            geo_data = _ensure_wgs84_bounds(geo_data)
            data = json.dumps(geo_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def _serve_tile(self, tiles_dir: str, z: int, x: int, y: int):
        """Serve a map tile with overzoom fallback.

        If the tile at (z, x, y) doesn't exist, walk up parent zoom levels,
        crop the relevant sub-region from the nearest parent tile, scale it up
        to 256x256, and serve it.
        """
        tile_path = os.path.join(tiles_dir, str(z), str(x), f"{y}.png")
        if os.path.isfile(tile_path):
            self._serve_binary_file(tile_path, "image/png")
            return

        # Walk up zoom levels to find a parent tile
        cur_z, cur_x, cur_y = z, x, y
        while cur_z > 0:
            cur_z -= 1
            cur_x //= 2
            cur_y //= 2
            parent_path = os.path.join(
                tiles_dir, str(cur_z), str(cur_x), f"{cur_y}.png"
            )
            if os.path.isfile(parent_path):
                self._serve_overzoom_tile(parent_path, z, x, y, cur_z)
                return

        self.send_error(404, f"No tile at z={z} x={x} y={y}")

    def _serve_overzoom_tile(
        self, parent_path: str, z: int, x: int, y: int, parent_z: int
    ):
        """Crop and upscale a sub-region of a parent tile."""
        import io
        from PIL import Image

        try:
            img = Image.open(parent_path)
            w, h = img.size  # typically 256×256

            levels_up = z - parent_z
            scale = 2 ** levels_up

            # Which sub-region of the parent does (z, x, y) correspond to?
            parent_x = x >> levels_up
            parent_y = y >> levels_up
            sub_x = x - parent_x * scale
            sub_y = y - parent_y * scale

            region_w = w / scale
            region_h = h / scale
            left = sub_x * region_w
            top = sub_y * region_h

            cropped = img.crop((
                int(left), int(top),
                int(left + region_w), int(top + region_h),
            ))
            resized = cropped.resize((w, h), Image.LANCZOS)

            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            data = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, f"Overzoom tile error: {e}")

    def _serve_binary_file(self, path: str, content_type: str):
        if not os.path.isfile(path):
            self.send_error(404, f"File not found: {path}")
            return
        try:
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def _send_json_ok(self, extra=None):
        resp_data = {"ok": True}
        if extra:
            resp_data.update(extra)
        resp = json.dumps(resp_data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _save_walls(self):
        """Save walls JSON — always writes to 6a (manual), preserving 6 (AI)."""
        try:
            body = self._read_body()
            data = json.loads(body)

            if "walls" not in data or not isinstance(data["walls"], list):
                self.send_error(400, "Invalid walls JSON: missing 'walls' array")
                return

            for i, w in enumerate(data["walls"]):
                if "type" not in w or "points" not in w:
                    self.send_error(400, f"Wall {i} missing 'type' or 'points'")
                    return

            os.makedirs(_MANUAL_WALLS_DIR, exist_ok=True)

            if os.path.isfile(_MANUAL_WALLS_JSON):
                shutil.copy2(_MANUAL_WALLS_JSON, _MANUAL_WALLS_JSON + ".bak")

            with open(_MANUAL_WALLS_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Copy geo_metadata.json to 6a if not present
            src_geo = _GEO_META_JSON
            dst_geo = os.path.join(_MANUAL_WALLS_DIR, "geo_metadata.json")
            if os.path.isfile(src_geo) and not os.path.isfile(dst_geo):
                shutil.copy2(src_geo, dst_geo)

            self._send_json_ok()
        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            self.send_error(500, str(e))

    def _save_game_objects(self):
        try:
            body = self._read_body()
            data = json.loads(body)

            if "objects" not in data or not isinstance(data["objects"], list):
                self.send_error(400, "Invalid game objects JSON: missing 'objects' array")
                return

            for i, obj in enumerate(data["objects"]):
                if "name" not in obj or "position" not in obj:
                    self.send_error(400, f"Object {i} missing 'name' or 'position'")
                    return

            if os.path.isfile(_GAME_OBJECTS_JSON):
                shutil.copy2(_GAME_OBJECTS_JSON, _GAME_OBJECTS_JSON + ".bak")

            os.makedirs(os.path.dirname(_GAME_OBJECTS_JSON), exist_ok=True)
            with open(_GAME_OBJECTS_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._send_json_ok()
        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            self.send_error(500, str(e))

    def _save_track_layouts(self):
        """Save layouts.json metadata."""
        try:
            body = self._read_body()
            data = json.loads(body)

            if "layouts" not in data or not isinstance(data["layouts"], list):
                self.send_error(400, "Invalid layouts JSON: missing 'layouts' array")
                return

            os.makedirs(_LAYOUTS_DIR, exist_ok=True)

            if os.path.isfile(_LAYOUTS_JSON):
                shutil.copy2(_LAYOUTS_JSON, _LAYOUTS_JSON + ".bak")

            with open(_LAYOUTS_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._send_json_ok()
        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            self.send_error(500, str(e))

    def _save_layout_mask(self, name: str):
        """Save layout mask as PNG binary."""
        try:
            safe = _safe_layout_name(name)
            body = self._read_body()

            if len(body) < 8:
                self.send_error(400, "Empty or too small PNG data")
                return

            os.makedirs(_LAYOUTS_DIR, exist_ok=True)
            mask_path = os.path.join(_LAYOUTS_DIR, f"{safe}.png")

            with open(mask_path, "wb") as f:
                f.write(body)

            self._send_json_ok({"path": mask_path, "size": len(body)})
        except Exception as e:
            self.send_error(500, str(e))

    def _save_surface_mask(self, tag: str):
        """Save surface mask PNG to 05a directory."""
        try:
            tag = re.sub(r'[^\w]', '', tag)
            body = self._read_body()

            if len(body) < 8:
                self.send_error(400, "Empty or too small PNG data")
                return

            os.makedirs(_MANUAL_SURFACE_MASKS_EDIT_DIR, exist_ok=True)
            mask_path = os.path.join(_MANUAL_SURFACE_MASKS_EDIT_DIR, f"{tag}_mask.png")

            with open(mask_path, "wb") as f:
                f.write(body)

            self._send_json_ok({"path": mask_path, "size": len(body)})
        except Exception as e:
            self.send_error(500, str(e))

    def _serve_surface_tile_index(self, tag):
        """Return list of non-empty tile coordinates for a tag.

        Scans the full mask and returns {"tiles": ["0_3", "1_3", ...]}
        containing only tiles that have at least one non-zero pixel.
        """
        import numpy as np
        try:
            meta = _get_surface_meta()
            if not meta:
                self.send_error(404, "No surface meta")
                return
            ts = meta.get("tile_size", 512)
            cols = meta.get("grid_cols", 0)
            rows = meta.get("grid_rows", 0)
            mask = _get_surface_mask(tag)
            non_empty = []
            if mask is not None:
                for r in range(rows):
                    for c in range(cols):
                        y0, x0 = r * ts, c * ts
                        y1 = min(y0 + ts, mask.shape[0])
                        x1 = min(x0 + ts, mask.shape[1])
                        if y0 < mask.shape[0] and x0 < mask.shape[1]:
                            if np.any(mask[y0:y1, x0:x1] > 0):
                                non_empty.append(f"{c}_{r}")
            resp = json.dumps({"tiles": non_empty}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(resp)
        except Exception as e:
            self.send_error(500, str(e))

    def _serve_surface_tile(self, tag, col, row):
        """Extract and serve a 512x512 tile from the full mask."""
        import cv2
        import numpy as np
        try:
            meta = _get_surface_meta()
            if not meta:
                self.send_error(404, "No surface meta")
                return
            ts = meta.get("tile_size", 512)
            mask = _get_surface_mask(tag)
            if mask is None:
                tile = np.zeros((ts, ts), dtype=np.uint8)
            else:
                y0, x0 = row * ts, col * ts
                y1 = min(y0 + ts, mask.shape[0])
                x1 = min(x0 + ts, mask.shape[1])
                tile = np.zeros((ts, ts), dtype=np.uint8)
                if y0 < mask.shape[0] and x0 < mask.shape[1]:
                    tile[:y1 - y0, :x1 - x0] = mask[y0:y1, x0:x1]
            _, buf = cv2.imencode(".png", tile)
            data = buf.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def _handle_save_surface_tile(self, tag, col, row):
        """Receive a tile PNG and patch into the cached mask."""
        try:
            body = self._read_body()
            if _patch_surface_tile(tag, col, row, body):
                self._send_json_ok()
            else:
                self.send_error(400, "Failed to patch tile")
        except Exception as e:
            self.send_error(500, str(e))

    def _handle_surface_flush(self):
        """Flush all cached surface masks to disk, then reconvert to blender JSON."""
        try:
            count = _flush_surface_masks()

            # After flushing masks, reconvert to blender JSON
            reconvert_ok = False
            try:
                script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                from stages.s05a_manual_surface_masks import reconvert_masks
                from pipeline_config import PipelineConfig
                cfg = _load_webtools_config()
                pc = PipelineConfig(
                    geotiff_path=cfg.get("geotiff_path", ""),
                    tiles_dir=cfg.get("tiles_dir", ""),
                    output_dir=cfg.get("output_dir", os.path.join(_REPO_ROOT, "output")),
                ).resolve()
                reconvert_masks(pc)
                reconvert_ok = True
                # Invalidate mask cache — reconvert wrote composited masks
                # back to disk, so the cache (holding raw pre-composite data)
                # is now stale.
                with _cache_lock:
                    _surface_mask_cache.clear()
            except Exception as e:
                print(f"[surface_flush] WARNING: reconvert failed: {e}")
                import traceback
                traceback.print_exc()

            self._send_json_ok({
                "flushed": count,
                "reconverted": reconvert_ok,
                "composited": reconvert_ok,
            })
        except Exception as e:
            self.send_error(500, str(e))

    def _save_json_body(self, path: str):
        """Generic: save JSON body to a file."""
        try:
            body = self._read_body()
            data = json.loads(body)

            os.makedirs(os.path.dirname(path), exist_ok=True)

            if os.path.isfile(path):
                shutil.copy2(path, path + ".bak")

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._send_json_ok()
        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            self.send_error(500, str(e))

    def _regenerate_centerline(self):
        """Recompute bends + timing from an edited centerline (server-side Python)."""
        try:
            body = self._read_body()
            data = json.loads(body)

            centerline_points = data.get("centerline", [])
            layout_name = data.get("layout_name", "")
            track_direction = data.get("track_direction", "clockwise")

            if not centerline_points or len(centerline_points) < 10:
                self.send_error(400, "Centerline too short (need >= 10 points)")
                return

            # Load the layout mask for track width measurement
            import numpy as np
            mask = None

            result_02 = _result_or_fallback(_02_RESULT_DIR, _MASK_FULL_MAP_DIR)
            if layout_name:
                safe = _safe_layout_name(layout_name)
                mask_path = os.path.join(result_02, f"{safe}.png")
                if not os.path.isfile(mask_path):
                    mask_path = os.path.join(_LAYOUTS_DIR, f"{safe}.png")
                if os.path.isfile(mask_path):
                    from PIL import Image
                    mask_img = Image.open(mask_path).convert("L")
                    mask = np.array(mask_img)

            if mask is None:
                # Fall back to road mask or merged mask
                for candidate in [
                    os.path.join(result_02, "road_mask.png"),
                    os.path.join(result_02, "merged_mask.png"),
                ]:
                    if os.path.isfile(candidate):
                        from PIL import Image
                        mask_img = Image.open(candidate).convert("L")
                        mask = np.array(mask_img)
                        break

            if mask is None:
                self.send_error(400, "No road mask found for track width measurement")
                return

            # Add script dir to path for imports
            script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)

            from road_centerline import regenerate_from_centerline

            # Check if we have a time0_idx (from VLM-generated timing)
            time0_idx = data.get("time0_idx")
            if time0_idx is not None:
                time0_idx = int(time0_idx)

            # Compute pixel_size_m from geo metadata
            pixel_size_m = 0.3  # default
            geo_meta_path = os.path.join(result_02, "result_masks.json")
            if os.path.isfile(geo_meta_path):
                try:
                    from ai_game_objects import _compute_pixel_size_m
                    pixel_size_m = _compute_pixel_size_m(geo_meta_path, mask.shape)
                except Exception:
                    pass

            result = regenerate_from_centerline(
                centerline_points, mask, track_direction,
                time0_idx=time0_idx, pixel_size_m=pixel_size_m,
            )

            resp = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))

    def _regenerate_vlm_objects(self):
        """Regenerate VLM objects with per-type support.

        Request body:
          layout_name: str
          object_type: "hotlap" | "pit" | "start" | "timing_0" | "all"  (default: "all")
          pit_count: int (optional)
          start_count: int (optional)
          track_direction: str
        """
        try:
            body = self._read_body()
            data = json.loads(body)

            layout_name = data.get("layout_name", "")
            object_type = data.get("object_type", "all")
            pit_count = data.get("pit_count")
            start_count = data.get("start_count")
            track_direction = data.get("track_direction", "clockwise")

            if pit_count is not None:
                pit_count = int(pit_count)
            if start_count is not None:
                start_count = int(start_count)

            # Find image and mask
            image_path = _modelscale_image_path()
            if not image_path:
                self.send_error(400, "No modelscale image found in stage 2 output")
                return

            result_02 = _result_or_fallback(_02_RESULT_DIR, _MASK_FULL_MAP_DIR)
            mask_path = None
            if layout_name:
                safe = _safe_layout_name(layout_name)
                candidate = os.path.join(result_02, f"{safe}.png")
                if not os.path.isfile(candidate):
                    candidate = os.path.join(_LAYOUTS_DIR, f"{safe}.png")
                if os.path.isfile(candidate):
                    mask_path = candidate

            # Add script dir to path
            script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)

            from pipeline_config import PipelineConfig
            config = PipelineConfig().resolve()

            # Load validation masks
            from ai_game_objects import ValidationMasks, generate_single_type_vlm, generate_all_vlm_sequential
            geo_meta_path = os.path.join(result_02, "result_masks.json")
            masks = None
            if mask_path and os.path.isfile(geo_meta_path):
                try:
                    masks = ValidationMasks.load(mask_path, result_02, geo_meta_path)
                except Exception:
                    pass

            if object_type == "all":
                # Full sequential generation
                vlm_result = generate_all_vlm_sequential(
                    image_path, mask_path, track_direction, masks,
                    pit_count=pit_count or 8,
                    start_count=start_count or 8,
                    api_key=config.gemini_api_key,
                    model_name=config.gemini_model,
                )

                # Handle timing_0: snap to centerline + regenerate timing
                all_objects = vlm_result["hotlap"] + vlm_result["pits"] + vlm_result["starts"]
                timing_objects = self._process_timing0(
                    vlm_result.get("timing_0_raw", []),
                    layout_name, track_direction, masks,
                )
                all_objects.extend(timing_objects)

                resp_data = {
                    "objects": all_objects,
                    "validation": vlm_result.get("validation", {}),
                }

            elif object_type == "timing_0":
                # VLM generates timing_0 → snap → regenerate all timing
                objs, val = generate_single_type_vlm(
                    "timing_0", image_path, mask_path, track_direction, masks,
                    api_key=config.gemini_api_key,
                    model_name=config.gemini_model,
                )
                timing_objects = self._process_timing0(
                    objs, layout_name, track_direction, masks,
                )
                resp_data = {
                    "objects": timing_objects,
                    "validation": {"timing_0": val},
                    "replace_type": "timing",  # signal frontend to replace all timing
                }

            else:
                # Single type: hotlap, pit, or start
                objs, val = generate_single_type_vlm(
                    object_type, image_path, mask_path, track_direction, masks,
                    pit_count=pit_count or 8,
                    start_count=start_count or 8,
                    api_key=config.gemini_api_key,
                    model_name=config.gemini_model,
                )
                resp_data = {
                    "objects": objs,
                    "validation": {object_type: val},
                }

            resp = json.dumps(resp_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))

    def _process_timing0(self, timing0_raw, layout_name, track_direction, masks):
        """Snap TIME_0 to centerline and regenerate all timing objects."""
        import numpy as np
        from road_centerline import (
            snap_to_centerline, generate_timing_points_from_time0,
            compute_curvature, detect_composite_bends,
        )

        if not timing0_raw:
            return []

        # Load centerline for this layout
        cl_path = _layout_read_path(layout_name, "centerline.json")
        if not os.path.isfile(cl_path):
            return []

        with open(cl_path, "r", encoding="utf-8") as f:
            cl_data = json.load(f)

        centerline_pts = cl_data.get("centerline", [])
        if len(centerline_pts) < 10:
            return []

        centerline = np.array(centerline_pts, dtype=np.float64)
        curvature = compute_curvature(centerline)
        bends = detect_composite_bends(centerline, curvature)

        # Snap TIME_0 to centerline
        time0_pos = timing0_raw[0]["position"]
        time0_idx, _ = snap_to_centerline(time0_pos, centerline)

        # Load layout mask for track width measurement
        mask = None
        result_02 = _result_or_fallback(_02_RESULT_DIR, _MASK_FULL_MAP_DIR)
        if layout_name:
            safe = _safe_layout_name(layout_name)
            mask_file = os.path.join(result_02, f"{safe}.png")
            if not os.path.isfile(mask_file):
                mask_file = os.path.join(_LAYOUTS_DIR, f"{safe}.png")
            if os.path.isfile(mask_file):
                from PIL import Image
                mask = np.array(Image.open(mask_file).convert("L"))

        if mask is None:
            return []

        pixel_size_m = masks.pixel_size_m if masks else 0.3

        timing_objs, labeled_bends = generate_timing_points_from_time0(
            mask, centerline, curvature, bends, time0_idx,
            track_direction, pixel_size_m,
        )

        # Update centerline.json with new time0_idx and labeled bends
        cl_data["bends"] = labeled_bends
        cl_data["time0_idx"] = time0_idx
        cl_write_path = _layout_write_path(layout_name, "centerline.json")
        with open(cl_write_path, "w", encoding="utf-8") as f:
            json.dump(cl_data, f, indent=2, ensure_ascii=False)

        return timing_objs

    def _regenerate_from_start_point(self):
        """Regenerate all objects from a manually-set start point.

        Request:
          start_position: [x, y]  -- pixel coords of start/finish point
          layout_name: str
          pit_count: int | null
          start_count: int | null

        Direction is auto-computed from centerline tangent (no user input needed).

        Logic:
          1. Snap start_position to centerline -> time0_idx, direction from tangent
          2. Find the last bend exit before time0_idx -> hotlap_start position
          3. Regenerate all timing points from time0_idx
          4. Call VLM for pits + starts with start_point_hint
          5. Merge all objects -> return
        """
        try:
            # --- sys.path setup (must be before local imports) ---
            script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)

            import numpy as np
            from road_centerline import (
                snap_to_centerline, generate_timing_points_from_time0,
                compute_curvature, detect_composite_bends,
            )
            from pipeline_config import PipelineConfig
            from ai_game_objects import ValidationMasks, generate_single_type_vlm

            body = self._read_body()
            data = json.loads(body)

            layout_name = data.get("layout_name", "")
            start_position = data.get("start_position")
            if not start_position or len(start_position) != 2:
                self.send_error(400, "start_position [x, y] is required")
                return
            pit_count = data.get("pit_count")
            start_count = data.get("start_count")

            if pit_count is not None:
                pit_count = int(pit_count)
            if start_count is not None:
                start_count = int(start_count)

            # --- Load centerline ---
            cl_path = _layout_read_path(layout_name, "centerline.json")
            if not os.path.isfile(cl_path):
                self.send_error(400, f"No centerline found for layout '{layout_name}'")
                return

            with open(cl_path, "r", encoding="utf-8") as f:
                cl_data = json.load(f)

            centerline_pts = cl_data.get("centerline", [])
            if len(centerline_pts) < 10:
                self.send_error(400, "Centerline too short (need >= 10 points)")
                return

            track_direction = cl_data.get("track_direction", "clockwise")
            centerline = np.array(centerline_pts, dtype=np.float64)
            curvature = compute_curvature(centerline)
            bends = detect_composite_bends(centerline, curvature)

            # 1. Snap start_position to centerline -> time0_idx
            time0_idx, snapped_pos = snap_to_centerline(start_position, centerline)

            # --- Determine driving direction along indices ---
            n = len(centerline)
            x, y = centerline[:, 0], centerline[:, 1]
            signed_area = 0.5 * (
                np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
                + x[-1] * y[0] - x[0] * y[-1]
            )
            centerline_is_cw = signed_area > 0
            track_is_cw = (track_direction == "clockwise")
            driving_follows_index = (centerline_is_cw == track_is_cw)

            # Get orientation at time0_idx from centerline tangent
            i_back = (time0_idx - 3) % n
            i_fwd = (time0_idx + 3) % n
            tangent = centerline[i_fwd] - centerline[i_back]
            tlen = float(np.linalg.norm(tangent))
            if tlen > 1e-6:
                tangent = tangent / tlen
            if not driving_follows_index:
                tangent = -tangent
            time0_orient = [round(float(tangent[0]), 4), round(float(tangent[1]), 4)]

            # 2. Find the last bend exit BEFORE time0_idx (in driving direction)
            #    -> that's where hotlap_start goes
            hotlap_idx = None
            if bends:
                best_dist = None
                for bend in bends:
                    exit_idx = bend["exit_idx"]
                    if driving_follows_index:
                        dist = (time0_idx - exit_idx) % n
                    else:
                        dist = (exit_idx - time0_idx) % n
                    if dist > 0 and (best_dist is None or dist < best_dist):
                        best_dist = dist
                        hotlap_idx = exit_idx

            if hotlap_idx is None:
                # No bends found -- place hotlap a bit before time0
                offset = max(10, n // 20)
                if driving_follows_index:
                    hotlap_idx = (time0_idx - offset) % n
                else:
                    hotlap_idx = (time0_idx + offset) % n

            hotlap_pos = centerline[hotlap_idx].tolist()
            hb = (hotlap_idx - 3) % n
            hf = (hotlap_idx + 3) % n
            h_tang = centerline[hf] - centerline[hb]
            h_tlen = float(np.linalg.norm(h_tang))
            if h_tlen > 1e-6:
                h_tang = h_tang / h_tlen
            if not driving_follows_index:
                h_tang = -h_tang
            hotlap_orient = [round(float(h_tang[0]), 4), round(float(h_tang[1]), 4)]

            hotlap_obj = {
                "name": "AC_HOTLAP_START_0",
                "position": [round(hotlap_pos[0]), round(hotlap_pos[1])],
                "orientation_z": hotlap_orient,
                "type": "hotlap_start",
            }

            # 3. Load mask and generate timing from time0_idx
            result_02 = _result_or_fallback(_02_RESULT_DIR, _MASK_FULL_MAP_DIR)
            safe = _safe_layout_name(layout_name)
            mask_path = None
            mask = None
            for base in [result_02, _LAYOUTS_DIR]:
                candidate = os.path.join(base, f"{safe}.png")
                if os.path.isfile(candidate):
                    mask_path = candidate
                    break

            config = PipelineConfig().resolve()

            geo_meta_path = os.path.join(result_02, "result_masks.json")
            masks = None
            if mask_path and os.path.isfile(geo_meta_path):
                try:
                    masks = ValidationMasks.load(mask_path, result_02, geo_meta_path)
                except Exception:
                    pass

            pixel_size_m = masks.pixel_size_m if masks else 0.3

            if mask_path:
                from PIL import Image as PILImage
                mask = np.array(PILImage.open(mask_path).convert("L"))

            timing_objs = []
            labeled_bends = []
            if mask is not None:
                timing_objs, labeled_bends = generate_timing_points_from_time0(
                    mask, centerline, curvature, bends, time0_idx,
                    track_direction, pixel_size_m,
                )

            # 4. Update centerline.json with time0_idx and labeled bends
            cl_data["bends"] = labeled_bends
            cl_data["time0_idx"] = time0_idx
            cl_write_path = _layout_write_path(layout_name, "centerline.json")
            with open(cl_write_path, "w", encoding="utf-8") as f:
                json.dump(cl_data, f, indent=2, ensure_ascii=False)

            # 5. Call VLM for pits + starts with start_point_hint
            image_path = _modelscale_image_path()
            start_point_hint = {
                "position": snapped_pos,
                "direction": time0_orient,
            }

            validation_results = {}
            pit_objs = []
            start_objs = []

            if image_path:
                try:
                    pit_objs, pit_val = generate_single_type_vlm(
                        "pit", image_path, mask_path, track_direction, masks,
                        pit_count=pit_count or 8,
                        start_count=start_count or 8,
                        api_key=config.gemini_api_key,
                        model_name=config.gemini_model,
                        start_point_hint=start_point_hint,
                    )
                    validation_results["pit"] = pit_val
                except Exception as e:
                    print(f"[SetStart] VLM pit generation failed: {e}")

                try:
                    start_objs, start_val = generate_single_type_vlm(
                        "start", image_path, mask_path, track_direction, masks,
                        pit_count=pit_count or 8,
                        start_count=start_count or 8,
                        api_key=config.gemini_api_key,
                        model_name=config.gemini_model,
                        start_point_hint=start_point_hint,
                    )
                    validation_results["start"] = start_val
                except Exception as e:
                    print(f"[SetStart] VLM start generation failed: {e}")

            # 6. Merge all objects
            all_objects = [hotlap_obj] + timing_objs + pit_objs + start_objs

            # 7. Save game_objects.json to 7a
            go_data = {
                "layout_name": layout_name,
                "track_direction": track_direction,
                "objects": all_objects,
            }
            go_path = _layout_write_path(layout_name, "game_objects.json")
            with open(go_path, "w", encoding="utf-8") as f:
                json.dump(go_data, f, indent=2, ensure_ascii=False)
            _auto_merge_manual()

            # 8. Return response
            resp_data = {
                "objects": all_objects,
                "validation": validation_results,
            }

            resp = json.dumps(resp_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))

    # ------------------------------------------------------------------
    # Pipeline API handlers
    # ------------------------------------------------------------------

    def _serve_pipeline_stages(self):
        resp = json.dumps(PIPELINE_STAGE_META).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(resp)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(resp)

    def _serve_pipeline_status(self):
        status = _pipeline_runner.get_status()
        status["_running"] = _pipeline_runner.is_running()
        resp = json.dumps(status).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(resp)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(resp)

    def _serve_pipeline_config(self):
        cfg = _load_webtools_config()
        # Add path validity checks
        cfg["_valid"] = {
            "geotiff_path": os.path.isfile(cfg.get("geotiff_path", "")),
            "tiles_dir": os.path.isdir(cfg.get("tiles_dir", "")),
            "map_tiles_dir": os.path.isdir(cfg.get("map_tiles_dir", "")),
            "output_dir": os.path.isdir(cfg.get("output_dir", "")),
            "blender_exe": os.path.isfile(cfg.get("blender_exe", "")),
        }
        resp = json.dumps(cfg).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(resp)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(resp)

    def _handle_pipeline_config_save(self):
        try:
            body = self._read_body()
            cfg = json.loads(body)
            _save_webtools_config(cfg)
            self._send_json_ok()
        except Exception as e:
            self.send_error(500, str(e))

    def _handle_pipeline_run(self):
        try:
            body = self._read_body()
            data = json.loads(body)
            stages = data.get("stages", [])
            cfg = _load_webtools_config()
            cfg.update(data.get("config", {}))
            _pipeline_runner.run_stages(stages, cfg)
            self._send_json_ok()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.send_error(500, str(e))

    def _handle_pipeline_stop(self):
        try:
            _pipeline_runner.stop()
            self._send_json_ok()
        except Exception as e:
            self.send_error(500, str(e))

    # Manual stage IDs and their output directories
    _MANUAL_STAGE_DIRS = {
        "track_layouts":      "02a_track_layouts",
        "manual_surface_masks": "05a_manual_surface_masks",
        "manual_walls":       "06a_manual_walls",
        "manual_game_objects": "07a_manual_game_objects",
        "manual_blender":     "09a_manual_blender",
    }

    def _serve_manual_stages(self):
        """GET /api/pipeline/manual_stages — return enabled state for each manual stage."""
        cfg = _load_webtools_config()
        enabled_map = cfg.get("manual_stages", {})
        result = {}
        for sid, odir in self._MANUAL_STAGE_DIRS.items():
            full = os.path.join(_REPO_ROOT, "output", odir)
            has_data = os.path.isdir(full) and bool(os.listdir(full))
            # Enabled if explicitly set, or auto-detected via existing data
            explicitly = enabled_map.get(sid)
            result[sid] = {
                "enabled": explicitly if explicitly is not None else has_data,
                "has_data": has_data,
                "output_dir": odir,
            }
        resp = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(resp)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(resp)

    def _handle_manual_stage_toggle(self):
        """POST /api/pipeline/manual_stages — toggle or reset a manual stage.
        Body: { "stage_id": "manual_walls", "enabled": true }
              { "stage_id": "manual_walls", "enabled": false, "reset": true }
        """
        try:
            body = self._read_body()
            data = json.loads(body)
            sid = data.get("stage_id", "")
            enabled = data.get("enabled", False)
            reset = data.get("reset", False)

            if sid not in self._MANUAL_STAGE_DIRS:
                self.send_error(400, f"Unknown manual stage: {sid}")
                return

            odir = self._MANUAL_STAGE_DIRS[sid]
            full = os.path.join(_REPO_ROOT, "output", odir)

            # Reset: remove all files in the manual output directory
            if reset and os.path.isdir(full):
                shutil.rmtree(full)
                # Clear in-memory caches for surface masks
                if sid == "manual_surface_masks":
                    global _surface_mask_cache, _surface_meta_cache
                    with _cache_lock:
                        _surface_mask_cache.clear()
                    _surface_meta_cache = None

            cfg = _load_webtools_config()
            if "manual_stages" not in cfg:
                cfg["manual_stages"] = {}
            cfg["manual_stages"][sid] = enabled
            _save_webtools_config(cfg)

            # If enabling, ensure the output directory exists
            if enabled:
                os.makedirs(full, exist_ok=True)

            # After reset, re-initialise from upstream stage data
            if reset:
                self._reinit_manual_stage(sid, cfg)

            # Update result junctions to reflect the change
            try:
                script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                from pipeline_config import PipelineConfig
                pc = PipelineConfig(
                    output_dir=cfg.get("output_dir", os.path.join(_REPO_ROOT, "output")),
                ).resolve()
                pc.setup_result_junctions(cfg.get("manual_stages", {}))
            except Exception as e:
                print(f"[pipeline] WARNING: Failed to update junctions: {e}")

            self._send_json_ok({"stage_id": sid, "enabled": enabled, "reset": reset})
        except Exception as e:
            self.send_error(500, str(e))

    def _reinit_manual_stage(self, sid, cfg):
        """Re-initialise a manual stage directory from its upstream stage data."""
        try:
            script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)

            from pipeline_config import PipelineConfig
            pc = PipelineConfig(
                geotiff_path=cfg.get("geotiff_path", ""),
                tiles_dir=cfg.get("tiles_dir", ""),
                output_dir=cfg.get("output_dir", os.path.join(_REPO_ROOT, "output")),
            ).resolve()

            if sid == "manual_surface_masks":
                from stages.s05a_manual_surface_masks import run as run_s05a
                run_s05a(pc)
                print("[pipeline] Re-initialised manual_surface_masks from stage 5")
            elif sid == "track_layouts":
                from stages.s02a_track_layouts import run as run_s02a
                run_s02a(pc)
                print("[pipeline] Re-initialised track_layouts from stage 2")
            elif sid == "manual_walls":
                # Create 06a dir and copy from 06
                src = pc.stage_dir("ai_walls")
                dst = pc.stage_dir("manual_walls")
                if os.path.isdir(src):
                    os.makedirs(dst, exist_ok=True)
                    for f in ["walls.json", "geo_metadata.json"]:
                        s = os.path.join(src, f)
                        d = os.path.join(dst, f)
                        if os.path.isfile(s) and not os.path.isfile(d):
                            import shutil as _sh
                            _sh.copy2(s, d)
                    print("[pipeline] Re-initialised manual_walls from stage 6")
            elif sid == "manual_game_objects":
                from stages.s07a_manual_game_objects import run as run_s07a
                run_s07a(pc)
                print("[pipeline] Re-initialised manual_game_objects from stage 7")
            elif sid == "manual_blender":
                from stages.s09a_manual_blender import run as run_s09a
                run_s09a(pc)
                print("[pipeline] Re-initialised manual_blender from stage 9")
        except Exception as e:
            print(f"[pipeline] WARNING: Failed to re-init {sid}: {e}")

    def _serve_file_list(self):
        """GET /api/files/list?path=relative/path — list files in output dir."""
        try:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            rel = params.get("path", [""])[0]
            cfg = _load_webtools_config()
            base = cfg.get("output_dir", os.path.join(_REPO_ROOT, "output"))
            target = os.path.normpath(os.path.join(base, rel))
            # Security: prevent path traversal outside output dir
            if not target.startswith(os.path.normpath(base)):
                self.send_error(403, "Access denied")
                return
            if not os.path.isdir(target):
                self.send_error(404, f"Directory not found: {rel}")
                return
            items = []
            names = sorted(os.listdir(target))
            # Directories first (unlimited), then files (capped at 200)
            dirs_list = []
            files_list = []
            for name in names:
                full = os.path.join(target, name)
                if os.path.isdir(full):
                    dirs_list.append({"name": name, "is_dir": True})
                else:
                    sz = 0
                    try:
                        sz = os.path.getsize(full)
                    except OSError:
                        pass
                    files_list.append({"name": name, "is_dir": False, "size": sz})
            items = dirs_list + files_list[:200]
            if len(files_list) > 200:
                items.append({"name": f"... ({len(files_list)} files total)", "is_dir": False, "size": 0, "_truncated": True})
            resp = json.dumps(items).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(resp)
        except Exception as e:
            self.send_error(500, str(e))

    def _serve_file_preview(self):
        """GET /api/files/preview?path=relative/path — serve file for preview."""
        try:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            rel = params.get("path", [""])[0]
            cfg = _load_webtools_config()
            base = cfg.get("output_dir", os.path.join(_REPO_ROOT, "output"))
            target = os.path.normpath(os.path.join(base, rel))
            if not target.startswith(os.path.normpath(base)):
                self.send_error(403, "Access denied")
                return
            if not os.path.isfile(target):
                self.send_error(404, f"File not found: {rel}")
                return
            ext = os.path.splitext(target)[1].lower()
            ct_map = {
                ".json": "application/json; charset=utf-8",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".svg": "image/svg+xml",
                ".txt": "text/plain; charset=utf-8",
                ".csv": "text/csv; charset=utf-8",
                ".ini": "text/plain; charset=utf-8",
                ".blend": "application/octet-stream",
            }
            ct = ct_map.get(ext, "application/octet-stream")
            with open(target, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def _serve_sse(self):
        """SSE endpoint for pipeline log streaming."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = _pipeline_runner.add_sse_client()
        try:
            # Send buffered log lines (last 200 only to avoid flooding)
            with _pipeline_runner._lock:
                tail = _pipeline_runner.log_lines[-200:]
                for line in tail:
                    msg = json.dumps({"type": "log", "data": line})
                    self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                self.wfile.flush()

            while True:
                try:
                    msg = q.get(timeout=30)
                    self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                    # Drain any additional queued messages before flushing
                    while not q.empty():
                        try:
                            msg = q.get_nowait()
                            self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                        except queue.Empty:
                            break
                    self.wfile.flush()
                except queue.Empty:
                    # Send keepalive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            _pipeline_runner.remove_sse_client(q)

    def log_message(self, format, *args):
        # Quieter logging — only log API calls and pipeline routes
        try:
            msg = str(args[0]) if args else ""
            if "/api/" in msg or "/tiles/" in msg:
                super().log_message(format, *args)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="SAM3 赛道生成工具 — Web Dashboard + 编辑器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tries", type=int, default=50)
    ap.add_argument(
        "--page",
        default="dashboard.html",
        help="Page to open: dashboard.html, analyzer.html, wall_editor.html, "
             "objects_editor.html, layout_editor.html, centerline_editor.html, "
             "gameobjects_editor.html, or surface_editor.html",
    )
    args = ap.parse_args()

    # Serve static files from the webTools directory
    webtools_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(webtools_dir)

    port = find_free_port(args.host, args.port, args.max_tries)
    url = f"http://{args.host}:{port}/{args.page}"

    httpd = ThreadingHTTPServer((args.host, port), ApiHandler)

    print(f"Python: {sys.executable}")
    print(f"Repo root: {_REPO_ROOT}")
    print(f"WebTools dir: {webtools_dir}")
    print(f"Tiles dir: {_TILES_DIR}")
    print(f"Serving: {url}")
    print("Press Ctrl+C to stop.")

    try:
        webbrowser.open(url, new=1, autoraise=True)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Terminate any running pipeline subprocess to avoid orphan processes
        # that hold Windows file locks and prevent future re-runs.
        _pipeline_runner.stop()
        try:
            httpd.server_close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
