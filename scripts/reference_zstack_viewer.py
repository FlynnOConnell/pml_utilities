"""Reference line-scan + Z-stack viewer; moved into the package.

The viewer is ``mbo_utilities.gui.linescan_viewer``: ``mbo scan.mesc`` opens
it when the unit picked is an AOD line scan, ``mbo linescan scan.mesc --view``
does the same without the dialog. This shim keeps the old script path working.
"""

from mbo_utilities.gui.linescan_viewer import main

if __name__ == "__main__":
    main()
