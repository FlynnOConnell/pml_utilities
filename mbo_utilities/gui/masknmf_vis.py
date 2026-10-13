"""masknmf's own viewers on a demixing result.

``mbo run/demixing_results.hdf5``, or a masknmf run folder's ``results.hdf5``,
lands here instead of the Studio viewer.
Which of masknmf's viewers opens is decided before launch (``--vis`` or the
prompt in ``mbo view``); the viewer windows are masknmf's, untouched. Torch
and masknmf are required; the device follows the compute-GPU policy in
``mbo_utilities.gpu``.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays.masknmf_run import is_masknmf_run, run_config, run_raw_movie
from mbo_utilities.gpu import compute_gpu
from mbo_utilities.masknmf.asap7 import (
    is_asap7_run,
    run_frame_times,
    run_provenance,
    run_registration,
    run_rtmc,
)
from mbo_utilities.masknmf.asap7 import run_raw_movie as asap7_raw_movie

logger = log.get("gui.masknmf_vis")

KINDS = ("demixing", "classification")
RTMC_PANEL = "RTMC (um)"
AXIS_COLORS = {"X": (0.4, 0.7, 1.0), "Y": (1.0, 0.6, 0.2), "Z": (0.6, 0.9, 0.5)}


def run_files(path: Path | str) -> dict[str, Path | None]:
    """The result and the plane binary beside it, ``None`` when absent.

    ``raw`` and ``ops`` are the plane binary and its ``ops.npy`` the MaskNMF
    pipeline writes beside ``demixing_results.hdf5``.
    """
    p = Path(path)
    found = {
        "demixing": p,
        "raw": p.parent / "data_raw.bin",
        "ops": p.parent / "ops.npy",
    }
    return {k: (v if v.is_file() else None) for k, v in found.items()}


class MasknmfViewers:
    """masknmf's viewers on one demixing result, built on demand.

    ``open(kind)`` builds the viewer once and shows it. The demixing viewer
    takes the raw movie as an extra panel when there is one (the recording a
    run folder read, the plane binary beside the result, else ``raw_path``),
    the registration a run's ``results.hdf5`` holds, and the shifts of
    ``motion_correction_path`` (or the motion export beside the result) as a
    trace panel. An ``mbo asap7`` run folder gives the inverted raw movie, its
    ``registration.hdf5`` shifts, and the scan's RTMC as a panel of its own.
    """

    def __init__(
        self,
        path: Path | str,
        device: str | None = None,
        raw_path: Path | str | None = None,
        motion_correction_path: Path | str | None = None,
    ):
        self.files = run_files(path)
        self.path = self.files["demixing"]
        self.raw_path = None if raw_path is None else Path(raw_path)
        self.motion_correction_path = (
            None if motion_correction_path is None else Path(motion_correction_path)
        )
        self._vis: dict[str, object] = {}
        self._rtmc_plot = None
        self.asap7 = is_asap7_run(self.path.parent)
        if self.asap7 and self.motion_correction_path is None:
            self.motion_correction_path = run_registration(self.path.parent)
        with h5py.File(self.path, "r") as f:
            prov = (
                json.loads(f.attrs["mbo_provenance"])
                if "mbo_provenance" in f.attrs
                else {}
            )
            self._nframes = int(f["DemixingResults"]["shape"][0])
        self._fs = float(prov["fs"]) if prov.get("fs") else None
        if self._fs is None and self.asap7:
            self._fs = float(run_provenance(self.path.parent)["fs"])
        if self._fs is None and is_masknmf_run(self.path.parent):
            self._fs = run_config(self.path.parent)["inputs"]["movie"].get("fs")
        if device is None:
            device = "cpu"
            if compute_gpu()["backend"] == "cuda":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    @property
    def timings(self) -> np.ndarray | None:
        return np.arange(self._nframes) / self._fs if self._fs else None

    def _raw_movie(self):
        """The raw movie: the recording a masknmf run folder read, else the
        plane binary beside the result, else ``raw_path``, else None.
        """
        if self.asap7:
            return asap7_raw_movie(self.path.parent)
        if is_masknmf_run(self.path.parent):
            return run_raw_movie(self.path.parent)
        if self.files["raw"] is not None and self.files["ops"] is not None:
            ops = np.load(self.files["ops"], allow_pickle=True).item()
            ly, lx = int(ops["Ly"]), int(ops["Lx"])
            nframes = self.files["raw"].stat().st_size // (ly * lx * 2)
            return np.memmap(
                self.files["raw"], dtype=np.int16, mode="r", shape=(nframes, ly, lx)
            )
        if self.raw_path is None:
            return None
        from mbo_utilities.reader import imread

        raw = imread(self.raw_path).squeeze()
        if raw.ndim != 3:
            raise ValueError(
                f"{self.raw_path.name} is not a single-plane movie: shape {raw.shape}"
            )
        return raw

    def _dock_rtmc(self, vis):
        """The scan's RTMC, sampled at the run's frames, docked under the viewer and following its frame."""
        from masknmf.visualization.imgui import TracePlot

        rtmc = run_rtmc(self.path.parent)
        if rtmc is None:
            click.echo("no RTMC: the scan's real-time motion correction never moved")
            return None
        values = rtmc.at(run_frame_times(self.path.parent))
        plot = TracePlot((RTMC_PANEL,), self._nframes, self.timings, autofit=False)
        plot.set(
            RTMC_PANEL,
            [(f"RTMC {k}", v, AXIS_COLORS.get(k[0])) for k, v in values.items()],
            fit=True,
        )
        plot.dock(vis.fov_widget.figure, size=220, title="RTMC")
        plot.link(vis.reference_index)
        return plot

    def open(self, kind: str):
        """Show masknmf's ``kind`` viewer, building it on first use."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        vis = self._vis.get(kind)
        if vis is not None:
            return vis.show()
        from masknmf.visualization import ClassificationVis, SingleSessionDemixingVis

        import masknmf

        logger.info(f"opening masknmf {kind} viewer for {self.path} on {self.device}")
        if kind == "demixing":
            results = masknmf.DemixingResults.from_hdf5(
                str(self.path), device=self.device
            )
            vis = SingleSessionDemixingVis(
                results,
                frame_timings=self.timings,
                device=self.device,
                results_path=self.path,
                raw=self._raw_movie(),
                shifts=self.motion_correction_path,
            )
            if vis.raw is None:
                click.echo(
                    "no raw movie: `mbo view ... --raw <movie>` adds the raw panel"
                )
            if vis.shifts is None:
                click.echo(
                    "no motion shifts: `mbo view ... --motion-correction <hdf5>` adds the shift traces"
                )
            if self.asap7:
                self._rtmc_plot = self._dock_rtmc(vis)
        else:
            vis = ClassificationVis.from_masknmf([str(self.path)])
        self._vis[kind] = vis
        return vis.show()
