import argparse
import os
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Local track session analyzer web server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tries", type=int, default=50)
    args = ap.parse_args()

    repo_root = repo_root_from_here()
    os.chdir(repo_root)

    port = find_free_port(args.host, args.port, args.max_tries)
    url = f"http://{args.host}:{port}/script/track_session_anaylzer/index.html"

    # Python 3.7+ SimpleHTTPRequestHandler supports 'directory' arg (3.7+ via init kw)
    handler = SimpleHTTPRequestHandler
    httpd = ThreadingHTTPServer((args.host, port), handler)

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


