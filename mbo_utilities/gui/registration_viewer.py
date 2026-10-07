"""A masknmf plane's movies side by side on one time slider.

Raw, the alignment copy registration estimated its shifts on, registered and
denoised: whichever of them the plane folder holds, in that order. ``python -m
mbo_utilities.gui.registration_viewer <plane dir>`` opens it in its own window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mbo_utilities.arrays.compression import CompressedMovieArray
from mbo_utilities.arrays.suite2p import Suite2pArray
from mbo_utilities.masknmf.params import ALIGN_FILE, PMD_FILE


def registration_movies(plane_dir: Path | str) -> dict[str, object]:
    """The plane's movies by panel name, each squeezed to ``(T, Y, X)``."""
    plane_dir = Path(plane_dir)
    movies = {}
    if (plane_dir / "data_raw.bin").exists():
        movies["raw"] = Suite2pArray(plane_dir, use_raw=True).squeeze()
    if (plane_dir / ALIGN_FILE).exists():
        movies["alignment copy"] = CompressedMovieArray(plane_dir / ALIGN_FILE).squeeze()
    if (plane_dir / "data.bin").exists():
        movies["registered"] = Suite2pArray(plane_dir).squeeze()
    if (plane_dir / PMD_FILE).exists():
        movies["denoised"] = CompressedMovieArray(plane_dir / PMD_FILE).squeeze()
    if not movies:
        raise FileNotFoundError(f"no masknmf movies in {plane_dir}")
    return movies


def registration_viewer(plane_dir: Path | str, figure_kwargs: dict | None = None):
    """An ``MboNDViewer`` over ``registration_movies(plane_dir)``, one row; call ``show()``."""
    from mbo_utilities.gui._ndviewer import MboNDViewer

    movies = registration_movies(plane_dir)
    return MboNDViewer(
        data=list(movies.values()),
        names=list(movies),
        figure_shape=(1, len(movies)),
        figure_kwargs=figure_kwargs or {"size": (400 * len(movies), 520)},
        cmap="gnuplot2",
    )


if __name__ == "__main__":
    import fastplotlib as fpl

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plane_dir", type=Path)
    viewer = registration_viewer(parser.parse_args().plane_dir)
    viewer.show()
    fpl.loop.run()
