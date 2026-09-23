"""Synthetic apps that exercise every surface an app can draw on."""

from __future__ import annotations

import numpy as np

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.app.demo.image_viewer import ImageViewerApp
from mbo_utilities.gui.app.demo.movie import MovieApp
from mbo_utilities.gui.app.demo.traces import TracesApp

# orbit radius as a fraction of the frame, period in frames, blob width in pixels
BLOBS = ((0.30, 90.0, 9.0), (0.18, 55.0, 6.0), (0.38, 140.0, 12.0))

__all__ = ["ImageViewerApp", "MovieApp", "TracesApp", "demo_apps", "movie_data"]


def demo_apps() -> list[App]:
    return [MovieApp(), TracesApp(), ImageViewerApp()]


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
