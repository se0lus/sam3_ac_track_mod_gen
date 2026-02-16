import argparse
import json
import os
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
# API-aware handler
# ---------------------------------------------------------------------------
_WALLS_JSON = os.path.join("output", "07_ai_walls", "walls.json")
_GEO_META_JSON = os.path.join("output", "07_ai_walls", "geo_metadata.json")


class ApiHandler(SimpleHTTPRequestHandler):
    """Extends static file serving with JSON API endpoints for wall editor."""

    def do_GET(self):
        if self.path == "/api/walls":
            self._serve_json_file(_WALLS_JSON)
        elif self.path == "/api/geo_metadata":
            self._serve_json_file(_GEO_META_JSON)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/walls":
            self._save_walls()
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

    def _save_walls(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body)

            # Basic validation
            if "walls" not in data or not isinstance(data["walls"], list):
                self.send_error(400, "Invalid walls JSON: missing 'walls' array")
                return

            for i, w in enumerate(data["walls"]):
                if "type" not in w or "points" not in w:
                    self.send_error(400, f"Wall {i} missing 'type' or 'points'")
                    return

            # Backup existing file
            if os.path.isfile(_WALLS_JSON):
                shutil.copy2(_WALLS_JSON, _WALLS_JSON + ".bak")

            # Save
            with open(_WALLS_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            resp = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        # Quieter logging — only log API calls
        if "/api/" in (args[0] if args else ""):
            super().log_message(format, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local track session analyzer web server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tries", type=int, default=50)
    ap.add_argument(
        "--page",
        default="index.html",
        help="Page to open: index.html or wall_editor.html",
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
