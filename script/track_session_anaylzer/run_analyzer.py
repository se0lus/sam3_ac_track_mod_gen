import argparse
import json
import math
import os
import re
import shutil
import socket
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def repo_root_from_here() -> str:
    # script/track_session_anaylzer -> script -> repo root
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
_WALLS_JSON = os.path.join(_REPO_ROOT, "output", "07_ai_walls", "walls.json")
_MANUAL_WALLS_JSON = os.path.join(_REPO_ROOT, "output", "07a_manual_walls", "walls.json")
_MANUAL_WALLS_DIR = os.path.join(_REPO_ROOT, "output", "07a_manual_walls")
_GEO_META_JSON = os.path.join(_REPO_ROOT, "output", "07_ai_walls", "geo_metadata.json")
_GAME_OBJECTS_JSON = os.path.join(_REPO_ROOT, "output", "08_ai_game_objects", "game_objects.json")
_GAME_OBJECTS_GEO_META_JSON = os.path.join(_REPO_ROOT, "output", "08_ai_game_objects", "geo_metadata.json")
_CENTERLINE_JSON = os.path.join(_REPO_ROOT, "output", "08_ai_game_objects", "centerline.json")
_LAYOUTS_DIR = os.path.join(_REPO_ROOT, "output", "02a_track_layouts")
_LAYOUTS_JSON = os.path.join(_REPO_ROOT, "output", "02a_track_layouts", "layouts.json")
_MASK_FULL_MAP_DIR = os.path.join(_REPO_ROOT, "output", "02_mask_full_map")
_GAME_OBJECTS_DIR = os.path.join(_REPO_ROOT, "output", "08_ai_game_objects")
_MANUAL_GAME_OBJECTS_DIR = os.path.join(_REPO_ROOT, "output", "08a_manual_game_objects")
_MANUAL_SURFACE_MASKS_DIR = os.path.join(_REPO_ROOT, "output", "05a_manual_surface_masks")
_STAGE5_PREVIEW_DIR = os.path.join(_REPO_ROOT, "output", "05_convert_to_blender", "merge_preview")


# ---------------------------------------------------------------------------
# Surface mask tile cache (lazy-loaded in-memory numpy arrays)
# ---------------------------------------------------------------------------
_surface_mask_cache = {}    # tag -> numpy.ndarray (H, W, uint8)
_surface_meta_cache = None
_cache_lock = threading.Lock()


def _get_surface_meta():
    global _surface_meta_cache
    if _surface_meta_cache is None:
        p = os.path.join(_MANUAL_SURFACE_MASKS_DIR, "surface_masks.json")
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
    manual = os.path.join(_MANUAL_SURFACE_MASKS_DIR, f"{tag}_mask.png")
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
    os.makedirs(_MANUAL_SURFACE_MASKS_DIR, exist_ok=True)
    with _cache_lock:
        tags = list(_surface_mask_cache.keys())
        for tag in tags:
            path = os.path.join(_MANUAL_SURFACE_MASKS_DIR, f"{tag}_mask.png")
            cv2.imwrite(path, _surface_mask_cache[tag])
    return len(tags)


def _safe_layout_name(name: str) -> str:
    """Convert layout name to filesystem-safe filename."""
    return re.sub(r'[^\w\-]', '_', name).strip('_') or "unnamed"


def _layout_read_path(layout_name: str, filename: str) -> str:
    """Read priority: 8a > 8."""
    safe = _safe_layout_name(layout_name)
    for base in [_MANUAL_GAME_OBJECTS_DIR, _GAME_OBJECTS_DIR]:
        p = os.path.join(base, safe, filename)
        if os.path.isfile(p):
            return p
    return os.path.join(_GAME_OBJECTS_DIR, safe, filename)


def _layout_write_path(layout_name: str, filename: str) -> str:
    """Write always goes to 8a."""
    safe = _safe_layout_name(layout_name)
    p = os.path.join(_MANUAL_GAME_OBJECTS_DIR, safe, filename)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _auto_merge_manual() -> None:
    """After any write to 8a, re-merge all layouts into top-level game_objects.json."""
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
    """Find modelscale image in stage 2 output."""
    if os.path.isdir(_MASK_FULL_MAP_DIR):
        for f in os.listdir(_MASK_FULL_MAP_DIR):
            if f.endswith("_modelscale.png"):
                return os.path.join(_MASK_FULL_MAP_DIR, f)
    return ""


class ApiHandler(SimpleHTTPRequestHandler):
    """Extends static file serving with JSON API endpoints for editors."""

    def do_GET(self):
        # --- Existing endpoints ---
        if self.path == "/api/walls":
            # 7a manual > 7 auto
            if os.path.isfile(_MANUAL_WALLS_JSON):
                self._serve_json_file(_MANUAL_WALLS_JSON)
            else:
                self._serve_json_file(_WALLS_JSON)
        elif self.path == "/api/geo_metadata":
            self._serve_json_file(_GEO_META_JSON)
        elif self.path == "/api/game_objects":
            self._serve_json_file(_GAME_OBJECTS_JSON)
        elif self.path == "/api/game_objects/geo_metadata":
            # 8a > 8 > stage 7
            manual_meta = os.path.join(_MANUAL_GAME_OBJECTS_DIR, "geo_metadata.json")
            if os.path.isfile(manual_meta):
                self._serve_json_file(manual_meta)
            elif os.path.isfile(_GAME_OBJECTS_GEO_META_JSON):
                self._serve_json_file(_GAME_OBJECTS_GEO_META_JSON)
            else:
                self._serve_json_file(_GEO_META_JSON)
        elif self.path == "/api/centerline":
            self._serve_json_file(_CENTERLINE_JSON)

        # --- Track layouts ---
        elif self.path == "/api/track_layouts":
            self._serve_json_file(_LAYOUTS_JSON)

        elif self.path.startswith("/api/layout_mask/"):
            name = self.path[len("/api/layout_mask/"):]
            name = _safe_layout_name(name)
            mask_path = os.path.join(_LAYOUTS_DIR, f"{name}.png")
            self._serve_binary_file(mask_path, "image/png")

        elif self.path.startswith("/api/mask_overlay/"):
            tag = self.path[len("/api/mask_overlay/"):]
            tag = re.sub(r'[^\w]', '', tag)
            mask_path = os.path.join(_MASK_FULL_MAP_DIR, f"{tag}_mask.png")
            self._serve_binary_file(mask_path, "image/png")

        elif self.path == "/api/modelscale_image":
            # Find modelscale image in stage 2 output
            found = None
            if os.path.isdir(_MASK_FULL_MAP_DIR):
                for f in os.listdir(_MASK_FULL_MAP_DIR):
                    if f.endswith("_modelscale.png"):
                        found = os.path.join(_MASK_FULL_MAP_DIR, f)
                        break
            if found:
                self._serve_binary_file(found, "image/png")
            else:
                self.send_error(404, "No modelscale image found")

        # --- Surface masks (5a manual editing) ---
        elif self.path == "/api/surface_masks":
            sf_json = os.path.join(_MANUAL_SURFACE_MASKS_DIR, "surface_masks.json")
            self._serve_json_file(sf_json)

        elif self.path.startswith("/api/surface_mask/"):
            tag = self.path[len("/api/surface_mask/"):]
            tag = re.sub(r'[^\w]', '', tag)
            # Priority: 5a manual > Stage 5 merge_preview
            manual_path = os.path.join(_MANUAL_SURFACE_MASKS_DIR, f"{tag}_mask.png")
            stage5_path = os.path.join(_STAGE5_PREVIEW_DIR, f"{tag}_merged.png")
            if os.path.isfile(manual_path):
                self._serve_binary_file(manual_path, "image/png")
            elif os.path.isfile(stage5_path):
                self._serve_binary_file(stage5_path, "image/png")
            else:
                self.send_error(404, f"No mask for tag '{tag}'")

        # --- Surface tiles (tile-based editing) ---
        elif self.path.startswith("/api/surface_tile/"):
            parts = self.path[len("/api/surface_tile/"):].split("/")
            if len(parts) == 2:
                tag = re.sub(r'[^\w]', '', parts[0])
                m = re.match(r'(\d+)_(\d+)\.png$', parts[1])
                if m:
                    self._serve_surface_tile(tag, int(m.group(1)), int(m.group(2)))
                else:
                    self.send_error(400, "Bad tile coords")
            else:
                self.send_error(400, "Bad surface tile path")

        # --- Per-layout centerline & game objects (8a > 8 priority) ---
        elif self.path.startswith("/api/layout_centerline/"):
            name = self.path[len("/api/layout_centerline/"):]
            cl_path = _layout_read_path(name, "centerline.json")
            self._serve_json_file(cl_path)

        elif self.path.startswith("/api/layout_game_objects/"):
            name = self.path[len("/api/layout_game_objects/"):]
            go_path = _layout_read_path(name, "game_objects.json")
            self._serve_json_file(go_path)

        else:
            super().do_GET()

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

        # --- Per-layout centerline & game objects (always write to 8a) ---
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
        """Save walls JSON — always writes to 7a (manual), preserving 7 (AI)."""
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

            # Copy geo_metadata.json to 7a if not present
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

            os.makedirs(_MANUAL_SURFACE_MASKS_DIR, exist_ok=True)
            mask_path = os.path.join(_MANUAL_SURFACE_MASKS_DIR, f"{tag}_mask.png")

            with open(mask_path, "wb") as f:
                f.write(body)

            self._send_json_ok({"path": mask_path, "size": len(body)})
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
        """Flush all cached surface masks to disk."""
        try:
            count = _flush_surface_masks()
            self._send_json_ok({"flushed": count})
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

            if layout_name:
                safe = _safe_layout_name(layout_name)
                mask_path = os.path.join(_LAYOUTS_DIR, f"{safe}.png")
                if os.path.isfile(mask_path):
                    from PIL import Image
                    mask_img = Image.open(mask_path).convert("L")
                    mask = np.array(mask_img)

            if mask is None:
                # Fall back to road mask or merged mask
                for candidate in [
                    os.path.join(_MASK_FULL_MAP_DIR, "road_mask.png"),
                    os.path.join(_MASK_FULL_MAP_DIR, "merged_mask.png"),
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
            geo_meta_path = os.path.join(_MASK_FULL_MAP_DIR, "result_masks.json")
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

            mask_path = None
            if layout_name:
                safe = _safe_layout_name(layout_name)
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
            geo_meta_path = os.path.join(_MASK_FULL_MAP_DIR, "result_masks.json")
            masks = None
            if mask_path and os.path.isfile(geo_meta_path):
                try:
                    masks = ValidationMasks.load(mask_path, _MASK_FULL_MAP_DIR, geo_meta_path)
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
        if layout_name:
            safe = _safe_layout_name(layout_name)
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

    def log_message(self, format, *args):
        # Quieter logging — only log API calls
        try:
            msg = str(args[0]) if args else ""
            if "/api/" in msg:
                super().log_message(format, *args)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Local track session analyzer web server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tries", type=int, default=50)
    ap.add_argument(
        "--page",
        default="index.html",
        help="Page to open: index.html, wall_editor.html, objects_editor.html, "
             "layout_editor.html, centerline_editor.html, gameobjects_editor.html, "
             "or surface_editor.html",
    )
    args = ap.parse_args()

    # Serve static files from the analyzer directory so URLs are clean:
    #   http://localhost:8000/wall_editor.html  (not /script/track_session_anaylzer/...)
    analyzer_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(analyzer_dir)

    port = find_free_port(args.host, args.port, args.max_tries)
    url = f"http://{args.host}:{port}/{args.page}"

    httpd = ThreadingHTTPServer((args.host, port), ApiHandler)

    print(f"Python: {sys.executable}")
    print(f"Repo root: {_REPO_ROOT}")
    print(f"Analyzer dir: {analyzer_dir}")
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
        try:
            httpd.server_close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
