"""Canonical colormap list shared by GUI widgets.

Both the summary-image widget and the video save dialog start from
DEFAULT_COLORMAPS and dynamically prepend the active fastplotlib cmap when
it falls outside this set, so any cmap-library name (gnuplot2, nipy_spectral,
etc.) still works after a Sync.
"""

from __future__ import annotations

from cmap import Colormap

DEFAULT_COLORMAPS: tuple[str, ...] = (
    "viridis",
    "magma",
    "inferno",
    "plasma",
    "cividis",
    "gray",
    "turbo",
    "hot",
    "gnuplot2",
)

DEFAULT_COLORMAP: str = "viridis"


def listed_name(cmap) -> str:
    """The name a selector lists a ``cmap.Colormap`` under: the short one in
    ``DEFAULT_COLORMAPS`` when it is one of them, else the catalog's own
    (``gnuplot:gnuplot2``). A graphic's ``cmap`` is the object, and ``str``
    of it is its repr, which names no colormap.
    """
    # by name: == compares the color stops and raises when their counts differ
    return next(
        (name for name in DEFAULT_COLORMAPS if Colormap(name).name == cmap.name),
        cmap.name,
    )
