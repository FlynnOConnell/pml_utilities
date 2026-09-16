"""masknmf's own viewers on a demixing result, with a bar to switch between them.

``mbo run/demixing_results.hdf5`` lands here instead of the Studio viewer:
the file opens in masknmf's demixing viewer, and a bar at the top of every
viewer window opens the compression and classification viewers of the same
run. Torch and masknmf are required; the device follows the compute-GPU
policy in ``mbo_utilities.gpu``.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from imgui_bundle import imgui, portable_file_dialogs as pfd

from mbo_utilities import log
from mbo_utilities.gpu import compute_gpu
from mbo_utilities.masknmf.params import MOCO_FILE, PMD_FILE

logger = log.get("gui.masknmf_vis")

KINDS = ("demixing", "compression", "classification")
_BAR_HEIGHT = 44


def run_files(path: Path | str) -> dict[str, Path | None]:
    """The stage files beside a demixing result, ``None`` when absent.

    ``compression`` and ``motion`` are the PMD and motion-correction exports
    of the same folder; ``raw`` and ``ops`` are the plane binary and its
    ``ops.npy`` the MaskNMF pipeline writes there.
    """
    p = Path(path)
    folder = p.parent
    found = {
        "demixing": p,
        "compression": folder / PMD_FILE,
        "motion": folder / MOCO_FILE,
        "raw": folder / "data_raw.bin",
        "ops": folder / "ops.npy",
    }
    return {k: (v if v.is_file() else None) for k, v in found.items()}


class MasknmfViewers:
    """masknmf's viewers on one demixing result, built on demand.

    ``open(kind)`` builds the viewer once, hangs the switcher bar on its
    figure and shows it. The compression viewer needs the run's PMD export
    and a raw movie: the plane binary when the folder has one, otherwise a
    file picked from the bar.
    """

    def __init__(self, path: Path | str, device: str | None = None):
        self.files = run_files(path)
        self.path = self.files["demixing"]
        self._vis: dict[str, object] = {}
        self._dialog = None
        self._raw_path: Path | None = None
        self._error: str | None = None
        with h5py.File(self.path, "r") as f:
            prov = json.loads(f.attrs["mbo_provenance"]) if "mbo_provenance" in f.attrs else {}
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

    def open(self, kind: str):
        """Show masknmf's ``kind`` viewer, building it on first use."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        vis = self._vis.get(kind)
        if vis is not None:
            return vis.show()
        import masknmf
        from masknmf.visualization import ClassificationVis, CompressionVis, SingleSessionDemixingVis

        logger.info(f"opening masknmf {kind} viewer for {self.path} on {self.device}")
        if kind == "demixing":
            results = masknmf.DemixingResults.from_hdf5(str(self.path), device=self.device)
            vis = SingleSessionDemixingVis(
                results, frame_timings=self.timings, device=self.device, source_path=self.path
            )
            figure = vis.fov_widget.figure
        elif kind == "classification":
            vis = ClassificationVis.from_masknmf([str(self.path)])
            figure = vis.figure
        else:
            if self.files["compression"] is None:
                raise FileNotFoundError(f"no {PMD_FILE} beside {self.path.name}")
            if self.files["raw"] is not None and self.files["ops"] is not None:
                ops = np.load(self.files["ops"], allow_pickle=True).item()
                ly, lx = int(ops["Ly"]), int(ops["Lx"])
                nframes = self.files["raw"].stat().st_size // (ly * lx * 2)
                raw = np.memmap(self.files["raw"], dtype=np.int16, mode="r", shape=(nframes, ly, lx))
            elif self._raw_path is not None:
                from mbo_utilities.reader import imread

                raw = imread(self._raw_path).squeeze()
                if raw.ndim != 3:
                    raise ValueError(f"{self._raw_path.name} is not a single-plane movie: shape {raw.shape}")
            else:
                raise FileNotFoundError(
                    f"no data_raw.bin beside {self.path.name}; pick the raw movie from the bar"
                )
            pmd = masknmf.PMDArray.from_hdf5(str(self.files["compression"]))
            moco = raw
            if self.files["motion"] is not None:
                import torch

                with h5py.File(self.files["motion"], "r") as f:
                    names = [n for n in ("PiecewiseRigidRegistrationArray", "RigidRegistrationArray") if n in f]
                if names:
                    moco = getattr(masknmf, names[0]).from_hdf5(str(self.files["motion"]), input_movie=raw)
                    moco.output_device = torch.device(self.device)
            vis = CompressionVis(moco, pmd, device=self.device, frame_timings=self.timings)
            figure = vis.ndw_videos.figure
        figure.add_imgui_window(self.draw_bar, location="top", size=_BAR_HEIGHT, title="masknmf")
        self._vis[kind] = vis
        return vis.show()

    def draw_bar(self) -> None:
        """The switcher: one button per viewer, a raw-movie picker for compression."""
        if self._dialog is not None and self._dialog.ready():
            picked = self._dialog.result()
            self._dialog = None
            if picked:
                self._raw_path = Path(picked[0])
                self._vis.pop("compression", None)
                self._error = None
        needs_raw = self.files["raw"] is None and self._raw_path is None
        for i, kind in enumerate(KINDS):
            if i:
                imgui.same_line()
            disabled = (kind == "compression" and (self.files["compression"] is None or needs_raw)) or (
                self._dialog is not None
            )
            imgui.begin_disabled(disabled)
            if imgui.button(kind.capitalize()):
                self._error = None
                try:
                    self.open(kind)
                except Exception as e:
                    self._error = str(e)
                    logger.exception(f"masknmf {kind} viewer failed: {e}")
            imgui.end_disabled()
            if kind == "compression" and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                if self.files["compression"] is None:
                    imgui.set_tooltip(f"needs {PMD_FILE} beside the result")
                elif needs_raw:
                    imgui.set_tooltip("needs the raw movie: pick it with the button on the right")
                elif self._raw_path is not None:
                    imgui.set_tooltip(f"raw movie: {self._raw_path}")
        if self.files["raw"] is None and self.files["compression"] is not None:
            imgui.same_line()
            imgui.begin_disabled(self._dialog is not None)
            if imgui.button("Raw movie..."):
                self._dialog = pfd.open_file("Raw movie for the compression viewer", str(self.path.parent))
            imgui.end_disabled()
        imgui.same_line()
        imgui.text_disabled(f"{self.path.name} · {self.device}")
        if self._error:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), self._error)
