"""The Studio's right-hand widget: the viewer controls, the menus and popups, the Run tab and the Signal Quality stats."""

import functools
import importlib.util
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

# rendercanvas picks glfw unless the qt backend is chosen before fastplotlib imports
if importlib.util.find_spec("PyQt6") is not None and os.environ.get(
    "RENDERCANVAS_BACKEND", "qt"
).lower() in ("qt", "pyqt6"):
    os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
    import PyQt6  # noqa: F401
    from PyQt6.QtWidgets import QSlider

    # suite2p's gui reads QSlider.NoTicks, which PyQt6 moved
    if not hasattr(QSlider, "NoTicks"):
        QSlider.NoTicks = QSlider.TickPosition.NoTicks


import cv2
import imgui_bundle
import numpy as np
from imgui_bundle import hello_imgui, imgui, imgui_ctx, implot

from mbo_utilities import log
from mbo_utilities.arrays import ScanImageArray
from mbo_utilities.arrays._base import temporal_mean
from mbo_utilities.arrays.features import PhaseCorrectionFeature
from mbo_utilities.gui._availability import HAS_SUITE2P
from mbo_utilities.gui._dialogs import check_file_dialogs
from mbo_utilities.gui._edge_window import EdgeWindow
from mbo_utilities.gui._help_viewer import draw_help_popup
from mbo_utilities.gui._imgui_helpers import PopupAutoSize, fit_width, push_font_safe
from mbo_utilities.gui._keyboard import (
    handle_keyboard_shortcuts,
    rebind_space_to_playback,
)
from mbo_utilities.gui._metadata_editor import draw_metadata_popup
from mbo_utilities.gui._options_popup import draw_options_popup
from mbo_utilities.gui._popups import draw_process_console_popup, draw_tools_popups
from mbo_utilities.gui._save_as import draw_saveas_popup
from mbo_utilities.gui._stats import (
    compute_zstats,
    draw_stats_section,
    hydrate_zstats,
    refresh_zstats,
    signal_quality_enabled,
)
from mbo_utilities.gui.playhead import Playhead, TimeAxis
from mbo_utilities.gui.slice import Slice, viewer_positions
from mbo_utilities.gui.widgets import draw_all_widgets, get_supported_widgets
from mbo_utilities.gui.widgets.gui_logger import GuiLogger, GuiLogHandler
from mbo_utilities.gui.widgets.imgui_debug import draw_imgui_debug_windows
from mbo_utilities.gui.widgets.menu_bar import draw_keybinds_popup, draw_menu_bar
from mbo_utilities.gui.widgets.notebooks import draw_notebooks_window
from mbo_utilities.gui.widgets.progress_bar import start_output_capture
from mbo_utilities.gui.widgets.style_editor import (
    apply_saved_style,
    draw_style_editor_window,
)
from mbo_utilities.lazy_array import base_array
from mbo_utilities.preferences import get_gpu_index, get_last_dir, get_mbo_dirs
from mbo_utilities.reader import MBO_AVAILABLE_FTYPES

if TYPE_CHECKING:
    from mbo_utilities.gui._ndviewer import MboNDViewer

__all__ = ["PreviewDataWidget"]


_PLANE_DIR_RE = re.compile(r"^plane\d+$", re.IGNORECASE)


def _first_existing(fpath) -> Path | None:
    if isinstance(fpath, (list, tuple)):
        fpath = fpath[0] if fpath else None
    if fpath is None or not Path(fpath).exists():
        return None
    return Path(fpath)


def _outdir_from_fpath(fpath) -> str | None:
    """The Run tab's default output dir: the folder ``fpath`` is, or the one its file is in."""
    p = _first_existing(fpath)
    if p is None:
        return None
    return str(p if p.is_dir() else p.parent)


def _derive_suite2p_output_dir(fpath) -> str | None:
    """The suite2p root above ``fpath``, the parent of its ``plane*/`` dirs, or None when it is not inside one."""
    p = _first_existing(fpath)
    if p is None:
        return None
    parent = p.parent if p.is_file() else p
    if _PLANE_DIR_RE.match(parent.name):
        return str(parent.parent)
    if any(
        child.is_dir() and _PLANE_DIR_RE.match(child.name) for child in parent.iterdir()
    ):
        return str(parent)
    return None


def _base_5d(arr):
    """Peel ``_ScrubTimingProxy``, ``_SqueezeSingletonDims`` and a ``FrameAveragedView`` back to the 5D array."""
    from mbo_utilities.arrays import FrameAveragedView
    from mbo_utilities.gui.run_gui import _ScrubTimingProxy, _SqueezeSingletonDims

    while True:
        if isinstance(arr, _ScrubTimingProxy):
            arr = arr._wrapped
        elif isinstance(arr, (_SqueezeSingletonDims, FrameAveragedView)):
            arr = arr._arr
        else:
            return arr


def _mean_key(arr) -> tuple:
    """The read state a temporal mean depends on besides the array itself."""
    pc = getattr(arr, "phase_correction", None)
    return getattr(arr, "fix_phase", None), getattr(pc, "effective_shift", None)


def _spatial_func(
    frame,
    mean_img: np.ndarray | None,
    sigma: float | None,
    subtract: bool,
    invert: bool,
) -> np.ndarray:
    """Mean subtraction, inversion about the mean (``2 * mean - frame``, for negative-going indicators) and a gaussian blur.

    With both, the frame is ``mean - frame``. The frame is copied to float32
    first, so the source is never written.
    """
    # with no slider dims fastplotlib passes the lazy array itself
    result = np.asarray(frame)
    # a windowed projection over a 3D block, or a view that reshaped Y/X, has no matching mean plane
    if mean_img is not None and result.shape == mean_img.shape:
        result = result.astype(np.float32) - mean_img
        if invert:
            result = -result
        if not subtract:
            result += mean_img
    if sigma is not None and sigma > 0 and result.ndim == 2:
        ksize = int(sigma * 6) | 1
        result = cv2.GaussianBlur(result, (ksize, ksize), sigma)
    return result


ZSTATS_PANEL_HEIGHT = 156  # the Signal Quality plot on the top strip


class PreviewDataWidget(EdgeWindow):
    """The Studio's controls on one edge of ``iw``'s figure: projections, contrast, scan phase, Save As, the Run tab, Signal Quality.

    ``threading_enabled`` starts the z-stats in the background on construction.
    """

    manual_roi = (
        None  # the ROI widget while the Widgets-menu toggle is on (sync_manual_roi)
    )
    top_strip = (
        None  # the figure's top edge: the menu row and the registered full-width panels
    )
    _frame_average = 1  # the slider handler runs before _reset_per_data_state sets it

    def __init__(
        self,
        iw: "MboNDViewer",
        fpath: str | None | list = None,
        threading_enabled: bool = True,
        size: int | None = None,
        location: Literal["bottom", "right"] = "right",
        title: str
        | None = None,  # fastplotlib draws a full-width title box for a titled edge window
        show_title: bool = False,
        movable: bool = False,
        resizable: bool = False,
        scrollable: bool = False,
        auto_resize: bool = True,
        window_flags: int | None = None,
        **kwargs,
    ):
        flags = (
            (imgui.WindowFlags_.no_title_bar if not show_title else 0)
            | (imgui.WindowFlags_.no_move if not movable else 0)
            | (imgui.WindowFlags_.no_resize if not resizable else 0)
            | (imgui.WindowFlags_.no_scrollbar if not scrollable else 0)
            | (imgui.WindowFlags_.no_scroll_with_mouse if not scrollable else 0)
            | (imgui.WindowFlags_.always_auto_resize if auto_resize else 0)
            | (window_flags or 0)
        )
        super().__init__(
            figure=iw.figure,
            size=250 if size is None else size,
            location=location,
            title=title,
            window_flags=flags,
        )

        # the figure's FPS overlay builds a second ImguiRenderer whose __init__ leaves its own context
        # current; fonts pushed into that atlas never draw and kill the render loop
        imgui.set_current_context(iw.figure.imgui_renderer.imgui_context)

        self._init_logging()
        self._init_suite2p()
        if implot.get_current_context() is None:
            implot.create_context()

        # the style editor's saved style wins over the shipped opaque theme
        from mbo_utilities.gui._imgui_helpers import style_imgui_opaque

        style_imgui_opaque()
        apply_saved_style()
        self._init_fonts()

        self.kwargs = kwargs
        self.fpath = fpath if fpath else getattr(iw, "fpath", None)
        self.image_widget = iw
        self.num_graphics = len(self.image_widget.graphics)
        self.shape = self.image_widget.data[0].shape
        # one handler on the sliders feeds the slice (channel, z-plane) and the playhead (time)
        self.slice = Slice()
        self.playhead = Playhead()
        iw.ndwidget.indices.add_event_handler(self._on_indices)
        self._on_indices(None)

        from mbo_utilities.arrays import TiffArray

        first_arr = self.image_widget.data[0]
        underlying = getattr(first_arr, "_arr", first_arr)
        self.is_mbo_scan = isinstance(underlying, (ScanImageArray, TiffArray))
        self.logger.debug(
            f"Data type: {type(first_arr).__name__}, is_mbo_scan: {self.is_mbo_scan}"
        )

        self._init_state()
        self.slice.add_event_handler(self._on_slice, "slice")
        self._init_zstats()
        self._init_saveas_state()
        self._init_viewer()

        if threading_enabled:
            self.start_zstats()

    def start_zstats(self) -> None:
        """Load cached z-stats and compute the rest, when Signal Quality is on."""
        if not signal_quality_enabled():
            return
        hydrated = hydrate_zstats(self)
        pending = [
            i
            for i in range(self.num_graphics)
            if not hydrated[i] and not self._zstats_running[i]
        ]
        if pending:
            self.logger.debug(
                f"Starting zstats computation for {len(pending)} array(s)..."
            )
            for i in pending:
                self._zstats_running[i] = True
            compute_zstats(self, only=pending)

    def _init_logging(self):
        self.debug_panel = GuiLogger()
        gui_handler = GuiLogHandler(self.debug_panel)
        gui_handler.setFormatter(logging.Formatter("%(message)s"))
        gui_handler.setLevel(logging.DEBUG)
        log.attach(gui_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s - %(name)s - %(message)s")
        )
        console_handler.setLevel(logging.DEBUG)

        log.set_global_level(logging.DEBUG if log.debug_enabled() else logging.INFO)

        log.attach(console_handler)
        self.logger = log.get("gui")
        self.logger.debug("Logger initialized.")
        start_output_capture()

    def _init_suite2p(self):
        # the settings dataclasses import suite2p, which contends for the GIL during fastplotlib's
        # first paint and freezes the window for seconds; the properties below build them on first use
        self._s2p = None
        self._s2p_db = None
        self._s2p_extras = None
        self._s2p_folder_dialog = None
        self._install_error = False
        self._current_pipeline = None
        self._lsp_missing_warned = False
        self._s2p_fix_phase = True
        self._s2p_use_fft = True
        self._s2p_frame_average = 1
        self._s2p_background = True
        self._s2p_diameter_lock = True
        self._s2p_slicing_open = False
        self._s2p_last_fpath = None
        self._s2p_last_max_tp = None
        self._s2p_last_num_planes = None
        self._s2p_last_num_channels = None
        self._s2p_tp_selection = ""
        self._s2p_tp_error = ""
        self._s2p_tp_parsed = None
        self._s2p_z_start = self._s2p_z_stop = self._s2p_z_step = 1
        self._s2p_c_start = self._s2p_c_stop = self._s2p_c_step = 1
        self._files_popup_sizer = None
        self._data_options_sizer = None
        self._pipe_settings_sizer = None
        self._pipe_settings_just_opened = False
        self._force_pipe_settings = False
        self._force_pipe_legend = False
        self._cellpose_cpu_warned = False
        self._classifier_dialog = None

    @property
    def s2p(self):
        """Suite2p processing settings (upstream schema)."""
        if self._s2p is None and HAS_SUITE2P:
            from mbo_utilities.gui.widgets.pipelines.settings import Suite2pSettings
            from mbo_utilities.preferences import get_s2p_torch_device

            self._s2p = Suite2pSettings()
            # a loaded dataset's settings.npy still overrides this (_try_hydrate_s2p)
            self._s2p.torch_device = get_s2p_torch_device()
        return self._s2p

    @s2p.setter
    def s2p(self, value):
        self._s2p = value

    @property
    def s2p_db(self):
        """Suite2p input/output db (paths, plane counts) — upstream schema."""
        if self._s2p_db is None and HAS_SUITE2P:
            from mbo_utilities.gui.widgets.pipelines.settings import Suite2pDB

            self._s2p_db = Suite2pDB()
        return self._s2p_db

    @s2p_db.setter
    def s2p_db(self, value):
        self._s2p_db = value

    @property
    def s2p_extras(self):
        """Mbo-only suite2p helper fields (dff_*, accept_all_cells, etc.)."""
        if self._s2p_extras is None and HAS_SUITE2P:
            from mbo_utilities.gui.widgets.pipelines.settings import MboSuite2pExtras

            self._s2p_extras = MboSuite2pExtras()
        return self._s2p_extras

    @s2p_extras.setter
    def s2p_extras(self, value):
        self._s2p_extras = value

    def _init_fonts(self):
        io = imgui.get_io()
        # with keyboard nav on, Space and Enter activate whatever button holds nav focus
        io.config_flags &= ~imgui.ConfigFlags_.nav_enable_keyboard

        fd_settings_dir = (
            Path(get_mbo_dirs()["imgui"])
            .joinpath("assets", "app_settings", "preview_settings.ini")
            .expanduser()
            .resolve()
        )
        io.set_ini_filename(str(fd_settings_dir))

        _fonts_dir = Path(imgui_bundle.__file__).parent.joinpath(
            "assets", "fonts", "Roboto"
        )
        sans_serif_font = str(_fonts_dir / "Roboto-Regular.ttf")
        self._default_imgui_font = io.fonts.add_font_from_file_ttf(
            sans_serif_font, 14, imgui.ImFontConfig()
        )
        # the pipeline-settings popup marks _IMPORTANT_FIELDS in bold when the file is there
        bold_path = _fonts_dir / "Roboto-Bold.ttf"
        if bold_path.is_file():
            self._bold_font = io.fonts.add_font_from_file_ttf(
                str(bold_path), 14, imgui.ImFontConfig()
            )
        else:
            self._bold_font = None
        push_font_safe(self._default_imgui_font)

    def _init_state(self):
        self.image_widget._sliders_ui._loop = True

        arr = self.image_widget.data[0]
        dims = getattr(arr, "dims", None)
        dims_lower = tuple(d.lower() for d in dims) if dims else None
        if dims_lower is not None and "z" in dims_lower:
            z_idx = dims_lower.index("z")
            self.nz = self.shape[z_idx]
        elif dims_lower is not None and any(
            d in dims_lower for d in ("z-planes", "z-slices")
        ):
            for d in ("z-planes", "z-slices"):
                if d in dims_lower:
                    z_idx = dims_lower.index(d)
                    self.nz = self.shape[z_idx]
                    break
        elif dims_lower is not None and "volumes" in dims_lower:
            if hasattr(arr, "num_slices"):
                self.nz = arr.num_slices
            else:
                vol_idx = dims_lower.index("volumes")
                if (
                    len(self.shape) >= 2
                    and self.shape[vol_idx] <= 1
                    and vol_idx + 1 < len(self.shape)
                ):
                    self.nz = self.shape[vol_idx + 1]
                else:
                    self.nz = self.shape[vol_idx]
        elif len(self.shape) >= 4 and (
            dims_lower is None
            or not any(d in dims_lower for d in ("channel", "channels", "c"))
        ):
            # a squeezed TCZYX array with no Z reports ("T", "C", "Y", "X"), so shape[1] would be the channels
            self.nz = self.shape[1]
        else:
            self.nz = 1

        if dims_lower is not None and "channel" in dims_lower:
            c_idx = dims_lower.index("channel")
            self.nc = self.shape[c_idx]
        elif dims_lower is not None and "channels" in dims_lower:
            c_idx = dims_lower.index("channels")
            self.nc = self.shape[c_idx]
        elif dims_lower is not None and "c" in dims_lower:
            c_idx = dims_lower.index("c")
            self.nc = self.shape[c_idx]
        elif hasattr(arr, "num_color_channels"):
            self.nc = arr.num_color_channels
        else:
            self.nc = 1

        self.n_views = 1  # isoview cameras
        if dims_lower is not None and "cm" in dims_lower:
            cm_idx = dims_lower.index("cm")
            self.n_views = self.shape[cm_idx]

        self.logger.debug(
            f"Detected nz={self.nz}, nc={self.nc}, n_views={self.n_views} from dims={dims}"
        )

        self._mean_jobs: dict[int, list] = {}
        self._mean_ready = False
        # every per-data control is reset there, on launch and on every File > Open
        from mbo_utilities.gui._dialogs import _reset_per_data_state

        _reset_per_data_state(self)

        self._register_z = False
        self._register_z_progress = 0.0
        self._register_z_done = False
        self._register_z_running = False
        self._register_z_current_msg = ""
        self._register_z_complete_time = 0.0
        self._axial_max_frames = (
            200  # compute_axial_shifts defaults, set in the Save As options
        )
        self._axial_max_reg_xy = 30

        self._selected_array = 0
        self._selected_planes = None

        self.show_scope_window = False
        self.show_metadata_viewer = False
        self._diagnostics_widget = None

        self._file_dialog = None
        self._folder_dialog = None

        self._custom_metadata = {}
        self._iso_crop_window_open = False
        self.linescan_traces = None
        self._last_frame_t = None
        self._show_iso_crop_window = False
        self._iso_crop_cache_key = None
        self._iso_crop_shapes = {}
        self._iso_crop_pending = {}  # the crop editor's unsaved drags, keyed (tile, camera); tile is None when not tiled
        self._iso_crop_proj_index = {}
        self._iso_crop_timepoints = []
        self._iso_crop_current_tp = 0
        self._iso_crop_gpu = {}
        self._iso_crop_gpu_lru = []  # keyed (view, timepoint)
        self._iso_crop_vmin = 0.0
        self._iso_crop_vmax = 1.0
        self._show_iso_seg_window = False
        self._iso_seg_window_open = False
        self._iso_seg_snapshot = None  # the segmentation params as they were when the window opened, for Cancel
        self._iso_seg_cache_key = None
        self._iso_seg_proj_index = {}
        self._iso_seg_timepoints = []
        self._iso_seg_current_tp = 0
        self._iso_seg_proj_cache = {}  # (view, tp) -> projection float32
        self._iso_seg_filtered_cache = {}  # (view, tp, sigma, kernel) -> filtered uint16
        self._iso_seg_filtered_lru = []
        self._iso_seg_gpu = {}  # view -> _GpuRGBA, reuploaded on compose
        self._iso_seg_compose_key = {}  # view -> what the texture was composed from
        self._iso_seg_vmin = 0.0
        self._iso_seg_vmax = 1.0
        self._iso_seg_dragging = False
        self._iso_seg_shown = {}  # view -> the last mask and level, held while a slider drags
        self._show_iso_dp_window = False
        self._iso_dp_window_open = False
        self._iso_dp_snapshot = None
        self._iso_dp_cache_key = None
        self._iso_dp_proj_index = {}
        self._iso_dp_timepoints = []
        self._iso_dp_current_tp = 0
        self._iso_dp_proj_cache = {}
        self._iso_dp_mask_cache = {}  # (view, tp, background, kernel, factor) -> (mask, t_abs, t_rel)
        self._iso_dp_mask_lru = []
        self._iso_dp_gpu = {}
        self._iso_dp_compose_key = {}
        self._iso_dp_vmin = 0.0
        self._iso_dp_vmax = 1.0
        self._iso_dp_dragging = False
        self._iso_dp_shown = {}  # view -> the last median mask, held while a slider drags
        self._pipeline_instances = {}
        self._selected_pipeline_name = None
        self._pipeline_applies_cache = None
        self._force_run_tab = False
        self._masknmf_frame_average = 1
        self._show_process_console = False
        self._process_console_open = False
        self._process_console_focus = False
        self._process_console_size = imgui.ImVec2(500, 350)
        self._process_console_content_h = 0.0
        self._process_console_grow_to = None
        self._proc_log_fixed_h = 0.0
        self._proc_log_count = 0
        self._sys_gpu_devices = []
        self._sys_cpu_pct = None
        self._sys_gpu_last_refresh = 0.0
        self._sys_mem = None
        self._sys_mem_last_refresh = 0.0
        self._options_gpu_idx = get_gpu_index()
        self._show_keybinds_popup = False
        self._keybinds_popup_actually_open = False
        self._keybinds_sizer = PopupAutoSize("Keybinds", auto_resize=False)
        self._show_help_popup = False
        self._help_selected_doc = 0
        self._help_select_doc = None
        self._help_sizer = PopupAutoSize("Help##HelpViewer", auto_resize=False)
        self._custom_key = ""
        self._custom_value = ""

        self._widgets = get_supported_widgets(self)

    def _init_zstats(self):
        """One slot per array, keyed by the breakout combo (``()`` without one); ``_zstats_spec[i]`` maps the sliders back to a combo."""
        self._zstats = [{} for _ in range(self.num_graphics)]
        self._zstats_means = [{} for _ in range(self.num_graphics)]
        self._zstats_mean_scalar = [{} for _ in range(self.num_graphics)]
        self._zstats_spec = [None] * self.num_graphics
        self._zstats_done = [False] * self.num_graphics
        self._zstats_running = [False] * self.num_graphics
        self._zstats_progress = [0.0] * self.num_graphics
        self._zstats_current_z = [0] * self.num_graphics
        self._zstats_z_indices = [None] * self.num_graphics
        self._stats_axis_pref = (
            "z"  # the series axis when both z-planes and timepoints exist
        )

    def _init_saveas_state(self):
        self._ext = ".tiff"
        self._ext_idx = MBO_AVAILABLE_FTYPES.index(".tiff")
        self._overwrite = True
        self._debug = False
        self._saveas_chunk_mb = 100

        self._zarr_sharded = True
        self._zarr_ome = True
        self._zarr_compression_level = 1
        self._zarr_pyramid = False
        self._zarr_pyramid_max_layers = 4
        self._zarr_pyramid_method = "median"
        self._h5_dataset_name = (
            "mov"  # suite2p and lbm_suite2p_python read this dataset
        )

        self._saveas_popup_open = False
        self._saveas_done = False
        self._saveas_running = False
        self._saveas_progress = 0.0
        self._saveas_current_index = 0
        self._saveas_complete_time = 0.0
        self._show_metadata_popup = False
        self._metadata_sizer = PopupAutoSize(
            "Set Metadata##MetadataPopup", auto_resize=False
        )
        self._meta_inputs = {}
        self.manual_roi = None

        save_as_dir = get_last_dir("save_as")
        self._saveas_outdir = str(save_as_dir) if save_as_dir else ""
        s2p_output_dir = get_last_dir("suite2p_output")
        # a re-run lands beside the source, or back in the suite2p root the data was opened from
        self._s2p_outdir = (
            _derive_suite2p_output_dir(self.fpath)
            or _outdir_from_fpath(self.fpath)
            or (str(s2p_output_dir) if s2p_output_dir else "")
        )
        from mbo_utilities.gui._dialogs import _try_hydrate_s2p_from_binary

        # File > Open hydrates in load_new_data; this covers `mbo path/to/data.bin`
        _try_hydrate_s2p_from_binary(self, self.fpath)

        self._saveas_folder_dialog = None
        self._saveas_selected_roi = set()
        self._saveas_rois = False
        self._saveas_output_suffix = ""
        self._saveas_background = True
        self._saveas_fix_phase = True
        self._saveas_use_fft = True

        self._saveas_video_fps = 30
        self._saveas_video_speed_factor = 1.0
        self._saveas_video_auto = True
        self._saveas_video_vmin = 0.0
        self._saveas_video_vmax = 1000.0
        self._saveas_video_vmin_pct = 1.0
        self._saveas_video_vmax_pct = 99.5
        self._saveas_video_temporal_smooth = 0
        self._saveas_video_temporal_mode_idx = 0
        self._saveas_video_spatial_smooth = 0.0  # 0 mean, 1 max, 2 std
        self._saveas_video_gamma = 1.0
        from mbo_utilities.gui._colormaps import DEFAULT_COLORMAP, DEFAULT_COLORMAPS

        self._saveas_video_cmaps: list[str] = list(DEFAULT_COLORMAPS)
        self._saveas_video_cmap_idx: int = self._saveas_video_cmaps.index(
            DEFAULT_COLORMAP
        )
        self._saveas_video_quality_idx = (
            2  # 0 preview, 1 high, 2 visually lossless, 3 lossless
        )
        self._saveas_video_codec_idx = 0
        self._saveas_video_mean_subtract = False  # libx264
        self._saveas_video_time_overlay = False
        self._saveas_video_scalebar = False
        self._saveas_video_upscale = 0

    # auto
    def _init_top_strip(self):
        """Claim the figure's top edge for the menu row and the full-width panels (the Traces plot, Signal Quality)."""
        from mbo_utilities.gui._top_strip import TopStrip

        self.top_strip = TopStrip(
            self.image_widget.figure, draw_menu=lambda: draw_menu_bar(self)
        )

    def _sync_top_panels(self) -> None:
        """Register / drop the top panels this widget owns, per frame."""
        from mbo_utilities.gui._top_strip import TopPanel
        from mbo_utilities.gui.widgets.widget_toggles import widget_enabled

        want = widget_enabled("signal_quality") and any(self._zstats_done)
        if want and not self.top_strip.has("zstats"):
            self.top_strip.register(
                TopPanel(
                    "zstats",
                    "Signal Quality",
                    self.draw_stats_plot,
                    height=ZSTATS_PANEL_HEIGHT,
                    priority=20,
                )
            )
        elif not want:
            self.top_strip.unregister("zstats")

    def _init_viewer(self):
        from mbo_utilities.gui.viewers import get_viewer_class

        self._init_top_strip()
        viewer_cls = get_viewer_class(self.image_widget.data[0])
        self._viewer = viewer_cls(self.image_widget, self.fpath, parent=self)
        self.logger.debug(f"Viewer: {self._viewer.name}")
        self.set_context_info()
        self._update_window_funcs()
        rebind_space_to_playback(self)
        self._seed_playback_fps()
        from mbo_utilities.gui.widgets.pipelines.isoview import (
            maybe_spawn_raw_projections,
        )

        maybe_spawn_raw_projections(self)
        from mbo_utilities.gui.widgets.widget_toggles import widget_enabled

        self.sync_manual_roi(widget_enabled("manual_roi"))

    def sync_manual_roi(self, enabled: bool) -> None:
        """Build or tear down the ROI widget to match the Widgets-menu toggle.

        It attaches an overlay to the subplot, puts its Traces panel on the top
        strip and fills the ROIs and Traces tabs, so it exists only while on.
        """
        from mbo_utilities.gui.linescan_viewer import attach_standard_traces
        from mbo_utilities.gui.manual_roi import attach_roi_widget, detach_roi_widget

        if enabled == (self.manual_roi is not None):
            return
        if (
            self.linescan_traces is not None
        ):  # a line-scan unit's per-ROI traces live on the Traces tab
            self.linescan_traces.close()
            self.linescan_traces = None
        if enabled:
            attach_roi_widget(self)
            attach_standard_traces(self)  # adopts the store the last detach parked
        else:
            detach_roi_widget(self)

    def _seed_playback_fps(self):
        """Seed the playback rate from the array's ``fs``; ``seed_fps`` clamps to the slider's range and never overrides a typed fps."""
        from mbo_utilities.gui._keyboard import _get_sliders_ui

        sliders = _get_sliders_ui(self)
        if sliders is not None and hasattr(sliders, "seed_fps"):
            sliders.seed_fps("t", self.image_widget.data[0].fs)

    def set_context_info(self):
        """Put the dataset's name in the window title."""
        if self.fpath is None:
            return
        from mbo_utilities import __version__

        name = (
            Path(self.fpath[0]).parent.name
            if isinstance(self.fpath, list)
            else Path(self.fpath).name
        )
        try:
            runner_params = hello_imgui.get_runner_params()
        except RuntimeError:  # no hello_imgui runner in an offscreen or notebook canvas
            return
        runner_params.app_shallow_settings.window_title = (
            f"The Studio v{__version__} - {name}"
        )

    def _get_data_arrays(self) -> list:
        return list(self.image_widget.data)

    @property
    def current_offset(self) -> list[float]:
        """The scan-phase offset of the frame on screen, per graphic.

        ``arr.offset`` is the offset of whatever was read last, and the
        histogram and z-stats workers read in the background, so the display
        uses the per-(t, c, z) cache instead; 0.0 before the frame is read.
        """
        offsets = []
        for arr in self._get_data_arrays():
            getter = getattr(arr, "get_offset_at", None)
            value = getter(self.slice.t, self.slice.c, self.slice.z) if getter else None
            offsets.append(0.0 if value is None else float(value))
        return offsets

    def _phase_features(self) -> list[PhaseCorrectionFeature]:
        """The scan-phase feature of every array on screen that has one (raw ScanImage, ``PhaseCorrectedView``)."""
        features = (
            getattr(arr, "phase_correction", None) for arr in self._get_data_arrays()
        )
        return [pc for pc in features if isinstance(pc, PhaseCorrectionFeature)]

    def _set_phase(self, field: str, value) -> None:
        self.logger.debug(f"Setting {field} to {value}.")
        for pc in self._phase_features():
            setattr(pc, field, value)
        self._refresh_image_widget()

    @property
    def has_raster_scan_support(self) -> bool:
        return bool(self._phase_features())

    @property
    def fix_phase(self) -> bool:
        features = self._phase_features()
        return features[0].enabled if features else False

    @fix_phase.setter
    def fix_phase(self, value: bool):
        self._set_phase("enabled", value)

    @property
    def use_fft(self) -> bool:
        features = self._phase_features()
        return features[0].use_fft if features else False

    @use_fft.setter
    def use_fft(self, value: bool):
        self._set_phase("use_fft", value)

    @property
    def border(self) -> int:
        """Edge pixels left out of the phase correlation."""
        features = self._phase_features()
        return features[0].border if features else 3

    @border.setter
    def border(self, value: int):
        self._set_phase("border", value)

    @property
    def max_offset(self) -> int:
        features = self._phase_features()
        return features[0].max_offset if features else 3

    @max_offset.setter
    def max_offset(self, value: int):
        self._set_phase("max_offset", value)

    @property
    def gaussian_sigma(self) -> float:
        """Blur sigma in px; 0 is off."""
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, value: float):
        self._gaussian_sigma = max(0.0, value)
        self._rebuild_spatial_func()
        self._refresh_image_widget()

    @property
    def proj(self) -> str:
        """The window projection: mean, max or std."""
        return self._proj

    @proj.setter
    def proj(self, value: str):
        if value != self._proj:
            self.logger.debug(f"Projection changed: {self._proj} -> {value}")
            self._proj = value
            self._update_window_funcs()

    @property
    def mean_subtraction(self) -> bool:
        return self._mean_subtraction

    @mean_subtraction.setter
    def mean_subtraction(self, value: bool):
        if value != self._mean_subtraction:
            self._mean_subtraction = value
            self._update_mean_subtraction()
            self.image_widget.reset_vmin_vmax_frame()

    @property
    def invert_deflection(self) -> bool:
        """Each frame flipped about the mean image, display only."""
        return self._invert_deflection

    @invert_deflection.setter
    def invert_deflection(self, value: bool):
        if value != self._invert_deflection:
            self._invert_deflection = value
            self._update_mean_subtraction()
            self.image_widget.reset_vmin_vmax_frame()

    def mean_image_progress(self) -> float | None:
        """Progress of the slowest mean image still computing, or None."""
        jobs = list(self._mean_jobs.values())
        return min(job[1] for job in jobs) if jobs else None

    def _mean_image(self, i: int) -> np.ndarray | None:
        """Graphic ``i``'s per-pixel mean over T as ``(C, Z, Y, X)``, or None.

        The first call for an array starts `temporal_mean` in a background
        thread and answers None; `update` rebuilds the spatial function when
        it lands. Keyed by the array and its scan-phase state, so a unit swap
        or a phase toggle recomputes.
        """
        base = _base_5d(self.image_widget.data[i])
        entry = self._mean_images.get(i)
        if entry is not None and entry[0] is base and entry[1] == _mean_key(base):
            return entry[2]
        running = self._mean_jobs.get(i)
        if running is not None and running[0] is base:
            return None
        job = [base, 0.0]
        self._mean_jobs[i] = job
        threading.Thread(
            target=self._compute_mean_image, args=(i, job), daemon=True
        ).start()
        return None

    def _compute_mean_image(self, i: int, job: list) -> None:
        base = job[0]
        key = _mean_key(base)
        t0 = time.perf_counter()
        try:
            mean = temporal_mean(
                base, progress_callback=functools.partial(job.__setitem__, 1)
            )
            self.logger.info(
                f"mean image {mean.shape} in {time.perf_counter() - t0:.1f}s"
            )
        except Exception:
            self.logger.exception("mean image failed")
            mean = None
        running = self._mean_jobs.get(i)
        # a job for data opened since supersedes this one
        if running is not None and running is not job:
            return
        self._mean_images[i] = (base, key, mean)
        self._mean_jobs.pop(i, None)
        self._mean_ready = True

    def _on_indices(self, _indices) -> None:
        """The viewer's sliders moved: the slice and the playhead follow."""
        positions, roles = viewer_positions(self.image_widget)
        self.slice.move(positions, roles, source=self)
        t_name = next((name for name, role in roles.items() if role == "t"), None)
        if t_name is not None:
            self.playhead.seek(self.time_axis().seconds(positions[t_name]), source=self)

    def time_axis(self) -> TimeAxis:
        """The viewer's T slider on the playhead's clock: frames at the binning shown."""
        arr = base_array(self.image_widget.data[0])
        return TimeAxis.sampled(arr.fs, self._frame_average or 1)

    def _on_slice(self, _event) -> None:
        """Another channel or z-plane is on screen: its mean image and contrast follow."""
        if self._mean_subtraction or self._invert_deflection:
            self._update_mean_subtraction()
        if self._auto_contrast_on_z and self.image_widget:
            self.image_widget.reset_vmin_vmax_frame()

    @property
    def auto_contrast_on_z(self) -> bool:
        return self._auto_contrast_on_z

    @auto_contrast_on_z.setter
    def auto_contrast_on_z(self, value: bool):
        self._auto_contrast_on_z = value

    @property
    def frame_average(self) -> int:
        """Source frames averaged into each displayed frame; 1 is off.

        The array itself is wrapped, so the ROI traces, the pipelines and the
        writers see the averaged frames too, the way ``fix_phase`` is baked in.
        The window functions, by contrast, project for display only.
        """
        return self._frame_average

    @frame_average.setter
    def frame_average(self, value: int):
        value = max(1, int(value))
        source = self._frame_average_source or _base_5d(self.image_widget.data[0])
        nt = int(source.shape[0]) if len(source.shape) == 5 else 1
        value = min(value, max(nt, 1))
        if value == self._frame_average:
            return
        self._apply_frame_average(source, value)

    def _sync_frame_average_options(self, factor: int) -> None:
        """The Save As, suite2p and masknmf options default to the viewer's binning; each can still change it per run."""
        self._saveas_frame_average = factor
        self._s2p_frame_average = factor
        self._masknmf_frame_average = factor

    def _apply_frame_average(self, source, factor: int) -> None:
        """Swap the viewer onto a ``FrameAveragedView`` of ``source``, or back off it.

        T changes, so this is File > Open's swap (window funcs dropped, indices
        reset, z-stats recomputed, playback re-seeded), except that the ROI
        store survives: masks are (Z, Y, X) and averaging only touches T.
        """
        from mbo_utilities.arrays import average_frames
        from mbo_utilities.gui.manual_roi import attach_roi_widget, detach_roi_widget
        from mbo_utilities.gui.run_gui import _squeeze_for_viewer

        iw = self.image_widget
        wrapped = _squeeze_for_viewer(average_frames(source, factor))

        roi_was_on = self.manual_roi is not None
        if roi_was_on:
            detach_roi_widget(self)
        # keeps the store and parked runs
        self._window_size = 1
        # the window funcs and the spatial closure are bound to the old t-rank
        self._rebuild_spatial_func()

        iw.data[0] = wrapped
        if iw.n_sliders > 0:
            iw.indices = [0] * iw.n_sliders

        self._frame_average = factor
        self._frame_average_source = source if factor > 1 else None
        self._sync_frame_average_options(factor)
        self.shape = wrapped.shape
        if len(self.shape) == 5:
            self.nc, self.nz = self.shape[1], self.shape[2]
        elif len(self.shape) == 4:
            self.nc, self.nz = 1, self.shape[1]
        else:
            self.nc = self.nz = 1
        self.set_context_info()
        self._refresh_widgets()
        self._rebuild_spatial_func()
        # the swap cleared the spatial func; averaging leaves the mean unchanged
        self._seed_playback_fps()
        self.refresh_zstats()

        if roi_was_on:
            attach_roi_widget(self)
        self.logger.info(
            f"frame averaging {'off' if factor == 1 else factor}: shape {self.shape}"
        )

    @property
    def window_size(self) -> int:
        """Frames in the temporal projection window."""
        return self._window_size

    @window_size.setter
    def window_size(self, value: int):
        nt = self.shape[0] if len(self.shape) > 0 else 1
        value = max(1, min(value, nt))
        if value == self._window_size:
            return
        self._window_size = value
        self.logger.debug(f"Window size set to {value}.")
        self._update_window_funcs()

    def _refresh_image_widget(self):
        """Redraw the frame on screen."""
        self.image_widget.indices = list(self.image_widget.indices)

    def _refresh_widgets(self):
        self._widgets = get_supported_widgets(self)

    def _update_mean_subtraction(self):
        """Rebuild the spatial function; the contrast is left alone so a z-plane change honours auto_contrast_on_z."""
        self._rebuild_spatial_func()
        self._refresh_image_widget()

    def _rebuild_spatial_func(self):
        sigma = self.gaussian_sigma if self.gaussian_sigma > 0 else None
        uses_mean = self._mean_subtraction or self._invert_deflection

        mean_imgs = []
        for i in range(self.num_graphics):
            mean = self._mean_image(i) if uses_mean else None
            if mean is not None:
                c, z = self.slice.c, self.slice.z
                mean = mean[min(c, mean.shape[0] - 1), min(z, mean.shape[1] - 1)]
            mean_imgs.append(mean)

        if sigma is None and all(m is None for m in mean_imgs):
            self.image_widget.spatial_func = None
            return

        self.image_widget.spatial_func = [
            functools.partial(
                _spatial_func,
                mean_img=mean,
                sigma=sigma,
                subtract=self._mean_subtraction,
                invert=self._invert_deflection,
            )
            for mean in mean_imgs
        ]

    def _update_window_funcs(self):
        """Set the viewer's ``window_funcs`` from the projection and window size; sizes are made odd so the window centres on the frame."""
        iw = self.image_widget
        if "t" not in (iw.slider_dims or ()):
            return
        size = int(self._window_size)
        if size <= 1:
            iw.window_funcs = None
            return
        proj_funcs = {"mean": np.mean, "max": np.max, "std": np.std}
        iw.window_funcs = {"t": (proj_funcs.get(self._proj, np.mean), max(3, size | 1))}

    def gui_progress_callback(self, frac, meta=None):
        """Progress from a save (``meta`` the plane index) or an axial registration (``meta`` the message)."""
        if isinstance(meta, (int, np.integer)):
            self._saveas_progress = frac
            self._saveas_current_index = meta
            self._saveas_done = frac >= 1.0
            if frac >= 1.0:
                self._saveas_running = False
                self._saveas_complete_time = time.time()
                self.logger.info("Save complete")
        elif isinstance(meta, str):
            self._register_z_progress = frac
            self._register_z_current_msg = meta
            self._register_z_done = frac >= 1.0
            if frac >= 1.0:
                self._register_z_running = False
                self._register_z_complete_time = time.time()

    def _clear_stale_progress(self):
        """Drop a finished progress bar five seconds after it completed."""
        now = time.time()
        if self._saveas_done and now - self._saveas_complete_time > 5.0:
            self._saveas_done = False
            self._saveas_progress = 0.0
        if self._register_z_done and now - self._register_z_complete_time > 5.0:
            self._register_z_done = False
            self._register_z_progress = 0.0
            self._register_z_current_msg = None

    def draw(self):
        """Keyboard shortcuts and the global popups, then the edge window."""
        handle_keyboard_shortcuts(self)
        check_file_dialogs(self)
        draw_tools_popups(self)
        draw_saveas_popup(self)
        draw_metadata_popup(self)
        draw_process_console_popup(self)
        draw_keybinds_popup(self)
        draw_help_popup(self)
        draw_options_popup(self)
        draw_imgui_debug_windows(self)
        draw_style_editor_window(self)
        draw_notebooks_window(self)
        try:
            # the isoview windows need immvision, which some imgui_bundle builds leave out
            from mbo_utilities.gui.widgets.isoview_crop import draw_window as draw_crop
            from mbo_utilities.gui.widgets.isoview_deadpixel import (
                draw_window as draw_deadpixel,
            )
            from mbo_utilities.gui.widgets.isoview_segment import (
                draw_window as draw_segment,
            )
        except ImportError:
            pass
        else:
            draw_crop(self)
            draw_segment(self)
            draw_deadpixel(self)

        super().draw()

    def update(self):
        """The per-frame callback: the widget's tabs, the top panels and the pipeline polls."""
        t0 = time.perf_counter()
        prev = self._last_frame_t
        # a gap well over the 16 ms frame interval means the main thread was blocked between
        # frames (a worker holding the GIL, the canvas swap); it is the signal behind "the gui froze"
        gap_ms = (t0 - prev) * 1000.0 if prev is not None else 0.0
        self._last_frame_t = t0

        from mbo_utilities.gui.widgets.pipelines.isoview import (
            maybe_refresh_raw_projections,
        )

        maybe_refresh_raw_projections(self)
        for pipeline in list(self._pipeline_instances.values()):
            pipeline.poll()
        self._sync_top_panels()
        t1 = time.perf_counter()
        with fit_width():
            # imgui wraps per window, so each tab's child re-applies the wrap
            self._viewer.draw()
        t2 = time.perf_counter()
        menu_ms = (t1 - t0) * 1000
        draw_ms = (t2 - t1) * 1000
        if gap_ms > 100 or menu_ms > 50 or draw_ms > 50:
            self.logger.debug(
                f"SLOW FRAME: gap={gap_ms:.0f}ms menu={menu_ms:.1f}ms draw={draw_ms:.1f}ms"
            )

        if self._mean_ready:
            self._mean_ready = False
            if self._mean_subtraction or self._invert_deflection:
                self._update_mean_subtraction()
                self.image_widget.reset_vmin_vmax_frame()

    def draw_stats_section(self):
        """The Signal Quality tab: the metric table; the plot is the top strip's panel."""
        draw_stats_section(self, plot=False)

    def draw_stats_plot(self):
        """The Signal Quality top panel: the plot at the canvas's full width."""
        draw_stats_section(self, table=False)

    def draw_preview_section(self):
        imgui.dummy(imgui.ImVec2(0, 5))
        with imgui_ctx.begin_child(
            "##PreviewChild", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            with fit_width():
                draw_all_widgets(self, self._widgets)

    def compute_zstats(self):
        compute_zstats(self)

    def refresh_zstats(self):
        refresh_zstats(self)

    def cleanup(self):
        from mbo_utilities.gui.widgets import cleanup_all_widgets
        from mbo_utilities.gui.widgets.pipelines import cleanup_pipelines

        cleanup_pipelines(self)
        cleanup_all_widgets(self._widgets)
        self.sync_manual_roi(False)
        if self.top_strip is not None:
            self.top_strip.close()
            self.top_strip = None
        self._file_dialog = None
        self._folder_dialog = None
        self._s2p_folder_dialog = None
        self.logger.debug("GUI cleanup complete")
