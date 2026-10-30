#!/usr/bin/env python3
"""Build the static web version of the game into web/ (for Netlify or any static host).

    python3 build_web.py            # seeds 0-999
    python3 build_web.py --seeds 200

web/index.html is visualizer.html with STATIC_SITE set, so the page reads seeds/<n>.json
instead of calling play_gui.py's /api/instance, and the AI button shows how to run the
harness locally. Everything a player could need is in those files: future prices and the
optimal plan are visible to anyone who opens the browser's developer tools, so scores on
the website are for fun, not benchmark results.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from harness.crop_planning import generate_crop_planning_instance  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=1000, help="build seeds 0..N-1 (default 1000)")
    parser.add_argument("--out", default="web")
    args = parser.parse_args()

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "seeds").mkdir(parents=True)

    page = (ROOT / "visualizer.html").read_text(encoding="utf-8")
    flag = (f"<script>window.STATIC_SITE = true; window.STATIC_MAX_SEED = {args.seeds - 1};</script>\n"
            "</head>")
    assert page.count("</head>") == 1
    (out / "index.html").write_text(page.replace("</head>", flag), encoding="utf-8")

    for seed in range(args.seeds):
        inst = generate_crop_planning_instance(seed)
        (out / "seeds" / f"{seed}.json").write_text(json.dumps(inst, separators=(",", ":")))
        if seed % 100 == 99:
            print(f"  {seed + 1}/{args.seeds} seeds", flush=True)
    print(f"built {out}: index.html + {args.seeds} seeds. Deploy by dragging the folder onto "
          "https://app.netlify.com/drop")


if __name__ == "__main__":
    main()
