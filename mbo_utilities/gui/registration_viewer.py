"""A masknmf run's movies side by side on one time slider.

Raw, the alignment copy registration estimated its shifts on, registered and
denoised: whichever of them the run folder holds, in that order. ``python -m
mbo_utilities.gui.registration_viewer <run folder>`` opens it in its own window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mbo_utilities.arrays.compression import CompressedMovieArray, has_compressed_movie
from mbo_utilities.arrays.masknmf_run import RESULTS_FILE, MasknmfRunArray
from mbo_utilities.masknmf.params import ALIGN_FILE


def registration_movies(run: Path | str) -> dict[str, object]:
    """The run's movies by panel name, each ``(T, Y, X)``."""
    run = Path(run)
    movie = MasknmfRunArray(run)
    movies = {"raw": movie.raw}
    if (run / ALIGN_FILE).exists():
        movies["alignment copy"] = CompressedMovieArray(run / ALIGN_FILE).squeeze()
    if movie.shifts is not None:
        movies["registered"] = movie.squeeze()
    if has_compressed_movie(run / RESULTS_FILE):
        movies["denoised"] = CompressedMovieArray(run / RESULTS_FILE).squeeze()
    return movies


def registration_viewer(run: Path | str, figure_kwargs: dict | None = None):
    """An ``MboNDViewer`` over ``registration_movies(run)``, one row; call ``show()``."""
    from mbo_utilities.gui._ndviewer import MboNDViewer

    movies = registration_movies(run)
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
    parser.add_argument("run", type=Path)
    viewer = registration_viewer(parser.parse_args().run)
    viewer.show()
    fpl.loop.run()
