"""Synthetic apps that exercise every surface an app can draw on."""

from __future__ import annotations

from typing import Any

import numpy as np

from mbo_utilities import imread
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.app._host import AppHost
from mbo_utilities.gui.app.demo.image_viewer import ImageViewerApp
from mbo_utilities.gui.app.demo.movie import MovieApp
from mbo_utilities.gui.app.demo.traces import TracesApp

# orbit radius as a fraction of the frame, period in frames, blob width in pixels
BLOBS = ((0.30, 90.0, 9.0), (0.18, 55.0, 6.0), (0.38, 140.0, 12.0))

__all__ = [
    "ImageViewerApp",
    "MovieApp",
    "TracesApp",
    "demo_apps",
    "demo_host",
    "movie_data",
]


def demo_apps() -> list[App]:
    return [MovieApp(), TracesApp(), ImageViewerApp()]


def demo_host(data: Any = None, size: tuple[int, int] = (1400, 900)) -> AppHost:
    """A plain figure with two slots, the movie and traces mounted on them.

    No viewer: this is the host's slot machinery on its own, what a scene
    app sees when it is granted a subplot.
    """
    import fastplotlib as fpl

    figure = fpl.Figure(shape=(1, 2), names=[["scene", "traces"]], size=size)
    host = AppHost(figure, data=imread(movie_data() if data is None else data))
    host.register(*demo_apps())
    host.mount("movie", 0)
    host.mount("traces", 1)
    return host


def movie_data(nt: int = 240, ny: int = 128, nx: int = 128) -> np.ndarray:
    """Three gaussian blobs orbiting in noise, as (T, Y, X) float32."""
    rng = np.random.default_rng(0)
    t = np.arange(nt, dtype=np.float32)
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float32)
    movie = rng.normal(0.0, 0.05, (nt, ny, nx)).astype(np.float32)
    for k, (radius, period, sigma) in enumerate(BLOBS):
        cy = ny * (0.5 + radius * np.sin(2 * np.pi * t / period + k))
        cx = nx * (0.5 + radius * np.cos(2 * np.pi * t / period + k))
        for i in range(nt):
            movie[i] += np.exp(
                -(((y - cy[i]) ** 2 + (x - cx[i]) ** 2) / (2 * sigma**2))
            )
    return movie
