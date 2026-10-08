"""Registration-Denoising Quality Control: the movies of a pre-registration denoising run.

``mbo run/results.hdf5`` lands here when the file is a ``run_reg_denoise``
result (``is_reg_denoise``), and the MaskNMF widget's View movies opens a run
folder here in its own process (``python -m
mbo_utilities.gui.reg_denoise_viewer <results.hdf5 | run folder>``). The
movies share one time slider and one camera; masknmf's ``TracePlot`` is docked
above them and linked to the same time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mbo_utilities import log
from mbo_utilities.gpu import compute_gpu
from mbo_utilities.masknmf.reg_denoise import RegDenoiseRun, on_frames

logger = log.get("gui.reg_denoise_viewer")

TITLE = "Registration-Denoising Quality Control"
RTMC_PANEL = "RTMC (um)"
SHIFT_PANEL = "masknmf shift (px)"
FOV_PANEL = "FOV mean"
ROI_PANEL = "ROI mean"
MOVIE_COLORS = {
    "raw": (0.6, 0.6, 0.6),
    "registered": (0.3, 0.55, 1.0),
    "pmd(raw)": (1.0, 0.6, 0.2),
    "pmd(registered)": (0.3, 0.85, 0.4),
}
AXIS_COLORS = {"X": (1.0, 0.35, 0.35), "Y": (0.35, 0.8, 0.35), "Z": (0.4, 0.6, 1.0)}


class RegDenoiseViewer:
    """raw | registered over pmd(raw) | pmd(registered), with traces above.

    Trace panels: the source's RTMC curves (µm, when it has them), masknmf's
    shifts (px), each movie's whole-frame mean, and each movie's mean inside
    the rectangle drawn on any of them (the four rectangles move together).
    """

    def __init__(
        self,
        path: Path | str,
        raw_path: Path | str | None = None,
        device: str | None = None,
    ):
        import fastplotlib as fpl
        from masknmf.visualization.imgui import TracePlot, resolve_time_reference

        if device is None:
            device = "cpu"
            if compute_gpu()["backend"] == "cuda":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"opening {TITLE} for {path} on {device}")
        self.run = RegDenoiseRun.open(path, raw_path=raw_path, device=device)
        movies = self.run.movies
        times = self.run.times
        ref_range, timings = resolve_time_reference(len(times), times)
        names = list(movies)

        self.ndw = fpl.NDWidget(
            ref_range,
            shape=(2, 2) if len(names) == 4 else (1, len(names)),
            names=names,
            controller_ids=[tuple(names)],
            size=(1200, 1100),
            canvas_kwargs={"title": TITLE},
        )
        self.images = {}
        for name, movie in movies.items():
            nd = self.ndw[name].add_nd_image(
                movie,
                ["time", "m", "n"],
                ["m", "n"],
                slider_maps={"time": timings},
                name=name,
            )
            nd.graphic.cmap = "gray"
            self.images[name] = nd.graphic
        for subplot in self.ndw.figure:
            subplot.tooltip.enabled = False

        rtmc = on_frames(self.run.rtmc, times)
        shifts = on_frames(self.run.shifts, times)
        panels = (
            ((RTMC_PANEL,) if rtmc else ())
            + ((SHIFT_PANEL,) if shifts else ())
            + (FOV_PANEL, ROI_PANEL)
        )
        self.traces = TracePlot(panels, len(times), timings, autofit=False)
        self.traces.dock(self.ndw.figure, size=560, title="traces")
        self.traces.link(self.ndw.indices)
        if rtmc:
            self.traces.set(
                RTMC_PANEL,
                [(f"RTMC {k}", v, AXIS_COLORS.get(k[0])) for k, v in rtmc.items()],
            )
        if shifts:
            self.traces.set(
                SHIFT_PANEL,
                [(f"masknmf {k}", v, AXIS_COLORS.get(k[0])) for k, v in shifts.items()],
            )
        self.traces.set(
            FOV_PANEL,
            [(n, m.mean(axis=(1, 2)), MOVIE_COLORS[n]) for n, m in movies.items()],
        )

        ny, nx = self.run.raw.shape[1:]
        start = [nx / 4, 3 * nx / 4, ny / 4, 3 * ny / 4]
        self.selectors = []
        for graphic in self.images.values():
            selector = graphic.add_rectangle_selector(
                selection=start,
                edge_thickness=1,
                edge_color="w",
                vertex_size=3.0,
                vertex_color="cyan",
            )
            selector.add_event_handler(self._on_selection, "selection")
            self.selectors.append(selector)
        self._syncing = False
        self.roi = None
        self.set_roi(start)

    def set_roi(self, selection) -> None:
        """Plot each movie's mean inside ``(xmin, xmax, ymin, ymax)`` in pixels."""
        ny, nx = self.run.raw.shape[1:]
        x0, x1, y0, y1 = (int(round(v)) for v in selection)
        x0, y0 = min(max(x0, 0), nx - 1), min(max(y0, 0), ny - 1)
        x1, y1 = max(min(x1, nx), x0 + 1), max(min(y1, ny), y0 + 1)
        if self.roi == (x0, x1, y0, y1):
            return
        self.roi = (x0, x1, y0, y1)
        self.traces.set(
            ROI_PANEL,
            [
                (n, m[:, y0:y1, x0:x1].mean(axis=(1, 2)), MOVIE_COLORS[n])
                for n, m in self.run.movies.items()
            ],
            fit=True,
        )

    def _on_selection(self, ev) -> None:
        if self._syncing:
            return
        value = ev.info["value"]
        self._syncing = True
        for selector in self.selectors:
            if selector is not ev.graphic:
                selector.selection = value
        self._syncing = False
        self.set_roi(value)

    def show(self):
        return self.ndw.show()


if __name__ == "__main__":
    import fastplotlib as fpl

    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("path", type=Path, help="a reg-denoise results.hdf5 or a MaskNMF run folder")
    RegDenoiseViewer(parser.parse_args().path).show()
    fpl.loop.run()
