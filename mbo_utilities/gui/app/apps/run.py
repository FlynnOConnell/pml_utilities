"""The Process tab: the registered pipelines, run on the open data."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities import log
from mbo_utilities.arrays import FrameAveragedView, ScanImageArray, TiffArray
from mbo_utilities.gui._dialogs import (
    _try_hydrate_s2p_from_binary,
    outdir_from_fpath,
    suite2p_output_dir,
)
from mbo_utilities.gui._save_as import SaveAs
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets.pipelines import (
    cleanup_pipelines,
    draw_run_tab,
    start_preload,
)
from mbo_utilities.gui.widgets.pipelines._base import Suite2pState
from mbo_utilities.lazy_array import base_array
from mbo_utilities.preferences import get_last_dir

logger = log.get("gui.app.run")


class RunContext(Suite2pState):
    """What the pipeline widgets read as their ``parent``, served from the host.

    The widgets in ``gui/widgets/pipelines`` were written against the
    preview window, so they read the open data through its attribute names
    (``image_widget``, ``fpath``, ``_custom_metadata``, ``nz``, ...) and keep
    their own state on it (the ``_s2p_*`` fields, the axial registration
    settings, the Run tab's pipeline instances). This class is that surface
    in one place: every window value is read off the host when asked, and
    the pipeline state starts where the preview window started it. When the
    preview window is deleted the pipelines move to plain names and this
    class goes with it.
    """

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.logger = logger
        # the Run tab's save-as fallback for suite2p's output folder
        self.save_as = SaveAs()
        self.manual_roi = None
        self._register_z = False
        # compute_axial_shifts defaults
        self._axial_max_frames = 200
        self._axial_max_reg_xy = 30
        self._selected_planes = None
        self.reset()

    def reset(self) -> None:
        """Start the per-dataset pipeline state over for the host's open data."""
        self._s2p_frame_average = self.frame_average
        self._masknmf_frame_average = self.frame_average
        self._s2p_outdir = outdir_from_fpath(self.fpath) or str(
            get_last_dir("suite2p_output") or ""
        )
        self._s2p_outdir = suite2p_output_dir(self.fpath) or self._s2p_outdir
        # applies_to is asked again for the new array
        self._pipeline_applies_cache = None
        if self.fpath:
            _try_hydrate_s2p_from_binary(self, self.fpath)

    @property
    def image_widget(self):
        return self.host.viewer

    @property
    def fpath(self) -> str | None:
        source = self.host.data.source_path
        return None if source is None else str(source)

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
    def is_mbo_scan(self) -> bool:
        return isinstance(base_array(self.host.data), (ScanImageArray, TiffArray))

    @property
    def _source(self):
        data = self.host.data
        return data.source if isinstance(data, FrameAveragedView) else data

    @property
    def has_raster_scan_support(self) -> bool:
        return hasattr(self._source, "phase_correction")

    @property
    def border(self) -> int:
        return getattr(self._source, "border", 3)

    @property
    def max_offset(self) -> int:
        return getattr(self._source, "max_offset", 3)

    @property
    def current_offset(self) -> list[float]:
        host = self.host
        lookup = getattr(self._source, "get_offset_at", None)
        offset = lookup and lookup(host.frame, host.channel, host.zplane)
        return [float(offset or 0.0)]


class RunApp(App):
    """The pipelines that apply to the open data, each with its settings and run button.

    The pipeline widgets draw themselves against a ``RunContext``; which one
    is selected and each widget's state survive opening other data, the
    per-dataset defaults are started over.
    """

    id = "run"
    title = "Process"
    dock = "right"
    order = 2
    size = 380

    def __init__(self):
        super().__init__()
        self.context: RunContext | None = None
        start_preload()

    def available(self, host) -> bool:
        return host.viewer is not None

    def data_changed(self, host) -> None:
        if self.context is not None:
            self.context.reset()

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        if self.context is None:
            self.context = RunContext(host)
        draw_run_tab(self.context)

    def close(self) -> None:
        if self.context is not None:
            cleanup_pipelines(self.context)
