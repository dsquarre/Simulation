#!/usr/bin/env python3
"""Farmland Tactics - Interactive Backend Server with On-Demand Docker & Optimal Solvers.

Serves the visualizer web app and provides REST API endpoints:
- GET /api/instance?seed=<int> : Dynamically generates instance & DP optimal schedule
- GET /api/luna?seed=<int>     : Returns cached Luna trajectory, or runs Docker Compose on-demand

Usage:
    python3 play_gui.py
    python3 play_gui.py --port 8080
"""

import argparse
import glob
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser

DEFAULT_PORT = 8080
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import harness instance generator
sys.path.insert(0, WORKSPACE_DIR)
try:
    from harness.crop_planning import generate_crop_planning_instance
except Exception as e:
    print(f"Warning: Could not import harness.crop_planning: {e}")
    generate_crop_planning_instance = None


def find_luna_trajectory(seed):
    """Scan logs for a completed 10-season run by gpt-6-luna or model for this seed."""
    runs = sorted(glob.glob(os.path.join(WORKSPACE_DIR, "logs/*/episode.jsonl")), reverse=True)
    for p in runs:
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
                if not lines:
                    continue
                first_meta = lines[0]
                if first_meta.get("seed") == seed and first_meta.get("agent") in ["gpt-6-luna", "model"]:
                    sched = []
                    for item in lines:
                        for act in item.get("parsed_actions", []):
                            if act.get("tool") == "plant_season":
                                res = item.get("tool_result")
                                out = res[0] if isinstance(res, list) and res else str(res)
                                if "error:" not in str(out).lower():
                                    raw = act.get("args", {}).get("allocation", "{}")
                                    alloc = json.loads(raw) if isinstance(raw, str) else raw
                                    sched.append(alloc)
                    if len(sched) == 10:
                        return sched, p
        except Exception:
            continue
    return None, None


def run_docker_luna(seed):
    """Execute docker compose up for gpt-6-luna on the given seed and extract moves."""
    print(f"🤖 [Docker Runner] Launching GPT-6 Luna container for Seed {seed}...")
    env = os.environ.copy()
    env["SEED"] = str(seed)
    env["TASK_ID"] = "crop_planning"
    env["MODE"] = "full"

    cmd = [
        "docker", "compose", "up",
        "--abort-on-container-exit",
        "--exit-code-from", "harness"
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=WORKSPACE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300
        )
        print(f"🤖 [Docker Runner] Docker finished with exit code {proc.returncode}")
    except subprocess.TimeoutExpired:
        return None, "Docker execution timed out after 300 seconds."
    except Exception as e:
        return None, f"Docker execution error: {e}"

    # Search for newly produced episode.jsonl
    sched, log_file = find_luna_trajectory(seed)
    if sched:
        return sched, None

    # Parse stderr/stdout error message
    err_snippet = "\n".join(proc.stdout.strip().splitlines()[-10:])
    return None, f"Failed to extract 10-season schedule. Output snippet:\n{err_snippet}"


class FarmlandRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving web app and internal REST APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WORKSPACE_DIR, **kwargs)

    def log_message(self, format, *args):
        # Suppress verbose 200 GET logs
        return

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/api/instance":
            self.handle_api_instance(params)
        elif path == "/api/luna":
            self.handle_api_luna(params)
        else:
            # Default static file serving (visualizer.html)
            super().do_GET()

    def handle_api_instance(self, params):
        seed_str = params.get("seed", ["0"])[0]
        try:
            seed = int(seed_str)
        except ValueError:
            seed = 0

        if generate_crop_planning_instance:
            try:
                inst = generate_crop_planning_instance(seed)
                self.send_json_response(200, inst)
                return
            except Exception as e:
                self.send_json_response(500, {"error": str(e)})
                return
        self.send_json_response(500, {"error": "generate_crop_planning_instance not available"})

    def handle_api_luna(self, params):
        seed_str = params.get("seed", ["0"])[0]
        try:
            seed = int(seed_str)
        except ValueError:
            seed = 0

        # Check cached logs first
        sched, log_file = find_luna_trajectory(seed)
        if sched:
            self.send_json_response(200, {
                "seed": seed,
                "schedule": sched,
                "cached": True,
                "source": log_file
            })
            return

        # Not cached: Run Docker Compose on-demand
        print(f"🌐 [API] Received on-demand request for GPT Luna on Seed {seed}. Starting Docker...")
        sched, err = run_docker_luna(seed)
        if sched:
            self.send_json_response(200, {
                "seed": seed,
                "schedule": sched,
                "cached": False,
                "source": "live_docker_run"
            })
        else:
            self.send_json_response(500, {
                "error": err or "Failed to run Docker harness for Luna",
                "seed": seed
            })

    def send_json_response(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def start_server(port):
    with socketserver.TCPServer(("", port), FarmlandRequestHandler) as httpd:
        httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Launch Farmland Tactics Web Game")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to serve on (default: 8080)")
    args = parser.parse_args()

    port = args.port
    url = f"http://localhost:{port}/visualizer.html"

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_running = (s.connect_ex(('127.0.0.1', port)) == 0)
    s.close()

    if not is_running:
        t = threading.Thread(target=start_server, args=(port,), daemon=True)
        t.start()
        time.sleep(0.4)
        print(f"🌾 Farmland Tactics server started on http://localhost:{port}")
    else:
        print(f"🌾 Farmland Tactics server already listening on http://localhost:{port}")

    print(f"🚀 Opening game in browser: {url}")
    webbrowser.open(url)
    print("Press Ctrl+C to exit.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting Farmland Tactics. Goodbye!")


if __name__ == "__main__":
    main()
