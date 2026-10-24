#!/usr/bin/env python3
"""Interactive local tester for the Box Packing environment (No Docker required).

Usage:
    python3 play.py [SEED]
Example:
    python3 play.py 42
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

seed = sys.argv[1] if len(sys.argv) > 1 else "42"
task_id = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TASK_ID", "box_packing")
port = "8008"
ws_dir = Path = tempfile.mkdtemp(prefix="play_")

env = dict(os.environ)
env["PORT"] = port
env["WORKSPACE_DIR"] = ws_dir
env["SANDBOX_URL"] = f"http://127.0.0.1:{port}"
env["TASK_ID"] = task_id
env["SEED"] = str(seed)
env["MODE"] = env.get("MODE", "full")
env["MODEL"] = env.get("MODEL", "mock/solver")

print(f"Starting sandbox server on :{port} (workspace: {ws_dir})...")
server = subprocess.Popen([sys.executable, "-m", "sandbox.server"], env=env)

try:
    # Wait for sandbox server to be ready
    import requests
    for _ in range(30):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).ok:
                break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("Sandbox failed to start")

    print(f"\n--- Running Box Packing with SEED={seed} ---\n")
    # Run harness with interactive terminal
    res = subprocess.run([sys.executable, "-m", "harness.run"], env=env)
    sys.exit(res.returncode)

finally:
    server.terminate()
    server.wait()
    shutil.rmtree(ws_dir, ignore_errors=True)
