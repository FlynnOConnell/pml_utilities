"""Extract per-ROI kymograph traces from every linescan unit in a .mesc file.

Thin wrapper around ``mbo linescan`` (``mbo_utilities.cli``), kept for the
original invocation:

    python scripts/extract_todd_linescan.py FILE.mesc [-o OUT_ROOT] [--unit MUnit_n]

With no ``-o`` results go to ``rois_linescan/<MUnit_n>/`` beside the file;
pass ``-o`` to keep a raw data folder untouched. ``mbo linescan --help`` lists
the rest.
"""

import sys

from mbo_utilities.cli import main

if __name__ == "__main__":
    main(["linescan", *sys.argv[1:]])
