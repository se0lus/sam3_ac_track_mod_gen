"""Backward-compatible wrapper — launches webTools server with analyzer page."""
import os
import sys
import subprocess

webtools = os.path.join(os.path.dirname(__file__), "..", "webTools", "run_webtools.py")
sys.exit(subprocess.call([sys.executable, webtools, "--page", "analyzer.html"] + sys.argv[1:]))
