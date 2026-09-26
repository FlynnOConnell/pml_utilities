"""The preview window's surface, for the widgets that were written against it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mbo_utilities import log
from mbo_utilities.arrays import FrameAveragedView, ScanImageArray, TiffArray
from mbo_utilities.gui._dialogs import (
    _try_hydrate_s2p_from_binary,
    outdir_from_fpath,
    suite2p_output_dir,
)
from mbo_utilities.gui._save_as import SaveAs
from mbo_utilities.gui.app.apps.viewer import PROJECTIONS
from mbo_utilities.gui.manual_roi import detach_roi_widget
from mbo_utilities.gui.widgets.pipelines._base import Suite2pState
from mbo_utilities.lazy_array import base_array
from mbo_utilities.preferences import get_last_dir

if TYPE_CHECKING:
    from mbo_utilities.gui.app._host import AppHost

logger = log.get("gui.app")


class WindowContext(Suite2pState):
    """What the preview window's widgets read as their ``parent``, served from the host.

    The pipeline widgets, the manual ROI widget and the panel widgets were
    written against the preview window, so they read the open data through
    its attribute names (``image_widget``, ``fpath``, ``_custom_metadata``,
    ``nz``, ...) and keep state on it (the ``_s2p_*`` fields, the parked ROI
    store). This class is that surface in one place: every window value is
    read off the host when asked, and the widget state starts where the
    preview window started it and starts over with each dataset. When the
    preview window is deleted those widgets move to plain names and this
    class goes with them; until then nothing else in the app speaks that
    vocabulary.
    """

    def __init__(self, host: AppHost):
        super().__init__()
        self.host = host
        self.logger = logger
        # the Run tab's save-as fallback for suite2p's output folder
        self.save_as = SaveAs()
        # the manual ROI widget while it is on, and what it left when turned off
        self.manual_roi = None
        self._manual_roi_store = None
        self._manual_roi_runs = None
        # a line scan's per-line traces while the ROI widget is on
        self.linescan_traces = None
        self._selected_planes = None
        self.reset()

    def reset(self) -> None:
        """Start the per-dataset widget state over for the host's open data."""
        self._s2p_frame_average = self.frame_average
        self._masknmf_frame_average = self.frame_average
        self._register_z = False
        self._axial_max_frames = 200
        self._axial_max_reg_xy = 30
        self._s2p_outdir = outdir_from_fpath(self.fpath) or str(
            get_last_dir("suite2p_output") or ""
        )
        self._s2p_outdir = suite2p_output_dir(self.fpath) or self._s2p_outdir
        if self.fpath:
            _try_hydrate_s2p_from_binary(self, self.fpath)

    def close(self) -> None:
        """Release what the pipeline and ROI widgets hold: windows, threads, files."""
        if self.linescan_traces is not None:
            self.linescan_traces.close()
            self.linescan_traces = None
        detach_roi_widget(self)

    def sync_manual_roi(self, enabled: bool) -> None:
        """Turn manual ROI labeling on or off now, as a pipeline asks before it
        reads ``manual_roi`` back.
        """
        app = self.host.apps["manual_roi"]
        app.open = enabled
        app.frame(self.host)

    @property
    def reference_view(self):
        """The MESc app's reference-image opener, for the Traces panel's button."""
        app = self.host.apps.get("mesc")
        if app is None or not app.available(self.host):
            return None
        return app.open_reference

    @property
    def run(self):
        """The Process tab: the pipeline widgets it built and the one selected."""
        return self.host.apps["run"]

    @property
    def image_widget(self):
        return self.host.viewer

    @property
    def top_strip(self):
        return self.host.strip

    @property
    def playhead(self):
        return self.host.playhead

    @property
    def fpath(self) -> str | None:
        source = self.host.data.source_path
        return None if source is None else str(source)

    @property
    def metadata_edits(self):
        return self.host.metadata_edits

    @property
    def _custom_metadata(self) -> dict:
        return self.host.metadata_edits.values

    @property
    def _bold_font(self):
        return self.host.bold_font

    @property
    def nz(self) -> int:
        return self.host.data.shape[2]

    @property
    def nc(self) -> int:
        return self.host.data.shape[1]

    @property
    def frame_average(self) -> int:
        data = self.host.data
        return data.factor if isinstance(data, FrameAveragedView) else 1

    @property
    def window_size(self) -> int:
        funcs = self.host.viewer.window_funcs or {}
        return next(iter(funcs.values()), (None, 1))[1]

    @property
    def proj(self) -> str:
        funcs = self.host.viewer.window_funcs or {}
        func = next(iter(funcs.values()), (None, 1))[0]
        return {v: k for k, v in PROJECTIONS.items()}.get(func, "mean")

    @property
    def _source(self):
        data = self.host.data
        return data.source if isinstance(data, FrameAveragedView) else data

    @property
    def is_mbo_scan(self) -> bool:
        return isinstance(base_array(self.host.data), (ScanImageArray, TiffArray))

    @property
    def has_raster_scan_support(self) -> bool:
        return hasattr(self._source, "phase_correction")

    @property
    def border(self) -> int:
        return self._source.border

    @border.setter
    def border(self, value: int) -> None:
        self._source.border = value

    @property
    def max_offset(self) -> int:
        return self._source.max_offset

    @max_offset.setter
    def max_offset(self, value: int) -> None:
        self._source.max_offset = value

    @property
    def current_offset(self) -> list[float]:
        host = self.host
        offset = self._source.get_offset_at(host.frame, host.channel, host.zplane)
        return [float(offset or 0.0)]
