"""``python -m mbo_utilities.gui.app [path] [--frames N]``."""

from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser(description="the mbo app host")
parser.add_argument("path", nargs="?", default=None)
parser.add_argument(
    "--frames", type=int, default=0, help="draw N frames offscreen and exit"
)
args = parser.parse_args()

if args.frames > 0:
    os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "1"

from mbo_utilities.gui.app import run_app  # noqa: E402

run_app(args.path, frames=args.frames)
