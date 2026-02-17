import argparse
import json
import os
import re
import shutil
import socket
import sys
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
# Output paths (relative to repo root, resolved at startup)
# ---------------------------------------------------------------------------
_WALLS_JSON = os.path.join("output", "07_ai_walls", "walls.json")
_GEO_META_JSON = os.path.join("output", "07_ai_walls", "geo_metadata.json")
_GAME_OBJECTS_JSON = os.path.join("output", "08_ai_game_objects", "game_objects.json")
_GAME_OBJECTS_GEO_META_JSON = os.path.join("output", "08_ai_game_objects", "geo_metadata.json")
_CENTERLINE_JSON = os.path.join("output", "08_ai_game_objects", "centerline.json")
_LAYOUTS_DIR = os.path.join("output", "02a_track_layouts")
_LAYOUTS_JSON = os.path.join("output", "02a_track_layouts", "layouts.json")
_MASK_FULL_MAP_DIR = os.path.join("output", "02_mask_full_map")
_GAME_OBJECTS_DIR = os.path.join("output", "08_ai_game_objects")
_MANUAL_GAME_OBJECTS_DIR = os.path.join("output", "08a_manual_game_objects")


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

            if os.path.isfile(_WALLS_JSON):
                shutil.copy2(_WALLS_JSON, _WALLS_JSON + ".bak")

            with open(_WALLS_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

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
             "layout_editor.html, centerline_editor.html, or gameobjects_editor.html",
    )
    args = ap.parse_args()

    repo_root = repo_root_from_here()
    os.chdir(repo_root)

    port = find_free_port(args.host, args.port, args.max_tries)
    url = f"http://{args.host}:{port}/script/track_session_anaylzer/{args.page}"

    httpd = ThreadingHTTPServer((args.host, port), ApiHandler)

    print(f"Python: {sys.executable}")
    print(f"Repo root: {repo_root}")
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
