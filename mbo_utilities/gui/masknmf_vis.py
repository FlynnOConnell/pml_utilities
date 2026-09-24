"""masknmf's own viewers on a demixing result.

``mbo run/demixing_results.hdf5`` lands here instead of the Studio viewer.
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
from mbo_utilities.gpu import compute_gpu

logger = log.get("gui.masknmf_vis")

KINDS = ("demixing", "classification")


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
    takes the raw movie as an extra panel when there is one (the plane binary
    beside the result, else ``raw_path``) and the shifts of
    ``motion_correction_path`` (or the motion export beside the result) as a
    trace panel.
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
        with h5py.File(self.path, "r") as f:
            prov = (
                json.loads(f.attrs["mbo_provenance"])
                if "mbo_provenance" in f.attrs
                else {}
            )
            self._nframes = int(f["DemixingResults"]["shape"][0])
        self._fs = float(prov["fs"]) if prov.get("fs") else None
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
        """The raw movie: the plane binary beside the result, else ``raw_path``, else None."""
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
        else:
            vis = ClassificationVis.from_masknmf([str(self.path)])
        self._vis[kind] = vis
        return vis.show()
