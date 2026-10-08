"""A masknmf run's movies in a 2 x 2 grid on one time slider.

Top: raw and registered. Bottom: the same two as PMD, the alignment copy
registration estimated its shifts on and that copy with the shifts applied,
to compare the denoised versions with the originals. ``python -m
mbo_utilities.gui.registration_viewer <run folder>`` opens it in its own window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mbo_utilities.arrays.compression import CompressedMovieArray
from mbo_utilities.arrays.masknmf_run import MasknmfRunArray
from mbo_utilities.masknmf.params import ALIGN_FILE


def registration_movies(run: Path | str) -> dict[str, object]:
    """The run's movies by panel name, each ``(T, Y, X)``: raw, registered,
    raw (pmd), registered (pmd), whichever the run has.
    """
    run = Path(run)
    movie = MasknmfRunArray(run)
    registered = movie.shifts is not None
    aligned = (run / ALIGN_FILE).exists()
    movies = {"raw": movie.raw}
    if registered:
        movies["registered"] = movie.squeeze()
    if aligned:
        movies["raw (pmd)"] = CompressedMovieArray(run / ALIGN_FILE).squeeze()
    if aligned and registered:
        movies["registered (pmd)"] = MasknmfRunArray(run, on_alignment_copy=True).squeeze()
    return movies


def registration_viewer(run: Path | str, figure_kwargs: dict | None = None):
    """An ``MboNDViewer`` over ``registration_movies(run)``, two per row; call ``show()``."""
    from mbo_utilities.gui._ndviewer import MboNDViewer

    movies = registration_movies(run)
    rows = (len(movies) + 1) // 2
    return MboNDViewer(
        data=list(movies.values()),
        names=list(movies),
        figure_shape=(rows, min(len(movies), 2)),
        figure_kwargs=figure_kwargs or {"size": (900, 420 * rows + 120)},
        cmap="gray",
    )


if __name__ == "__main__":
    import fastplotlib as fpl

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    viewer = registration_viewer(parser.parse_args().run)
    viewer.show()
    fpl.loop.run()
