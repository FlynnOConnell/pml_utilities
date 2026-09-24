"""
Save As dialog and worker functions.

This module contains the Save As popup dialog for exporting data
to different file formats with various options.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import hello_imgui, imgui
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities import log
from mbo_utilities.arrays import _sanitize_suffix
from mbo_utilities.arrays.features import (
    TAG_REGISTRY,
    DimensionTag,
    parse_timepoint_selection,
)
from mbo_utilities.gui._colormaps import DEFAULT_COLORMAP, DEFAULT_COLORMAPS
from mbo_utilities.gui._files import NATIVE_DIALOGS, no_dialog_hint
from mbo_utilities.gui._imgui_helpers import (
    PopupAutoSize,
    checkbox_with_tooltip,
    set_tooltip,
)
from mbo_utilities.gui._selection_ui import (
    draw_frame_average_input,
    draw_selection_table,
    resolve_dim_labels,
)
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.reader import MBO_AVAILABLE_FTYPES, imread, widget_reader_kwargs
from mbo_utilities.writer import imwrite

logger = log.get("gui.save_as")


def _get_array_features(src: SaveSource) -> dict[str, bool]:
    """Which save options the shown array supports.

    ``phase_correction`` bidirectional scan-phase correction, ``z_registration``
    axial registration of a multi-plane ScanImage recording, ``multi_roi``
    ROIs saved one folder each, ``frame_average`` temporal binning (counted
    on the un-binned source, since the viewer may already be binned down).
    """
    data = src.viewer.data[0]
    return {
        "phase_correction": hasattr(data, "phase_correction"),
        "z_registration": src.planes > 1 and src.scanimage,
        "multi_roi": getattr(data, "num_rois", 1) > 1,
        "frame_averaging": bool(getattr(data, "can_average", False)),
        "frame_average": src.source_frames > 1,
    }


class SaveAs:
    """The Save As dialog's settings, its selection and the running save's progress.

    Settings keep their value between opens; the selection rows follow the
    shown array's dimensions and reset when another file opens.
    """

    def __init__(self):
        self.open_requested = False
        self.modal_open = False
        self.options_requested = False
        self.options_reset = False
        self.sizer = PopupAutoSize("Save As")
        self.folder_dialog = None
        last = get_last_dir("save_as")
        self.outdir = str(last) if last else ""
        self.ext = ".tiff"
        self.ext_idx = MBO_AVAILABLE_FTYPES.index(".tiff")
        # suite2p and lbm_suite2p_python read "mov"
        self.h5_dataset = "mov"
        self.overwrite = True
        self.debug = False
        self.chunk_mb = 100
        self.background = True
        self.zarr_sharded = True
        self.zarr_ome = True
        self.zarr_level = 1
        self.zarr_pyramid = False
        self.zarr_pyramid_levels = 4
        self.zarr_pyramid_method = "median"
        # scan-phase correction for the written file, apart from the display
        self.fix_phase = True
        self.use_fft = True
        self.frame_average = 1
        self.register_z = False
        # compute_axial_shifts defaults
        self.axial_max_frames = 200
        self.axial_max_shift = 30
        self.split_rois = False
        self.selected_rois: set[int] = set()
        self.output_suffix = ""
        self.planes: set[int] | None = None
        self.channels: set[int] = {0}
        self.total = 0
        self.progress = 0.0
        self.current_index = 0
        self.done = False
        self.running = False
        self.complete_time = 0.0
        self.register_progress = 0.0
        self.register_done = False
        self.register_running = False
        self.register_msg = ""
        self.register_complete_time = 0.0
        self.video_fps = 30
        self.video_speed_factor = 1.0
        self.video_auto = True
        self.video_vmin = 0.0
        self.video_vmax = 1000.0
        self.video_vmin_pct = 1.0
        self.video_vmax_pct = 99.5
        self.video_temporal_smooth = 0
        self.video_temporal_mode_idx = 0
        self.video_spatial_smooth = 0.0
        self.video_gamma = 1.0
        self.video_cmaps: list[str] = list(DEFAULT_COLORMAPS)
        self.video_cmap_idx = self.video_cmaps.index(DEFAULT_COLORMAP)
        # preview, high, visually lossless, lossless
        self.video_quality_idx = 2
        self.video_codec_idx = 0
        self.video_mean_subtract = False
        self.video_time_overlay = False
        self.video_scalebar = False
        # 0 lifts the short side to 480 px
        self.video_upscale = 0

    def progress_callback(self, frac: float, meta=None) -> None:
        """Progress from imwrite: an int is the plane being written, a str axial registration."""
        if isinstance(meta, (int, np.integer)):
            self.progress = frac
            self.current_index = meta
            self.done = frac >= 1.0
            if self.done:
                self.running = False
                self.complete_time = time.time()
                logger.info("Save complete")
        elif isinstance(meta, str):
            self.register_progress = frac
            self.register_msg = meta
            self.register_done = frac >= 1.0
            if self.register_done:
                self.register_running = False
                self.register_complete_time = time.time()

    def clear_stale_progress(self, delay: float = 5.0) -> None:
        """Drop a finished save's or registration's progress ``delay`` s after it ended."""
        now = time.time()
        if self.done and now - self.complete_time > delay:
            self.done = False
            self.progress = 0.0
        if self.register_done and now - self.register_complete_time > delay:
            self.register_done = False
            self.register_progress = 0.0
            self.register_msg = ""


@dataclass
class SaveSource:
    """What Save As writes from, read off the window that opened it.

    ``viewer`` shows the array (its first data array is what is saved, its
    first graphic's contrast and colormap seed the video options). The
    display settings seed the video options too; ``metadata`` is what the
    user typed over the array's own.
    """

    viewer: Any
    fpath: Any
    planes: int = 1
    scanimage: bool = False
    frame_average: int = 1
    source_frames: int = 1
    projection: str = "mean"
    window: int = 1
    sigma: float = 0.0
    mean_subtraction: bool = False
    zstats: Any = None
    metadata: dict = field(default_factory=dict)


def _save_as_worker(path, **imwrite_kwargs):
    """Background worker for saving data to disk."""
    # Don't pass roi to imread - let it load all ROIs
    # Then imwrite will handle splitting/filtering based on roi parameter
    data = imread(path)

    # read-time features (fix_phase, use_fft, border, max_offset,
    # mean_subtraction, frame_average) ride along in imwrite_kwargs; imwrite
    # applies them to the array before writing
    imwrite(data, **imwrite_kwargs)


def draw_saveas_popup(sa: SaveAs, src: SaveSource):
    """Draw the Save As popup dialog."""
    just_opened = False
    if sa.open_requested:
        sa.sizer.before_open()
        imgui.open_popup("Save As")
        sa.open_requested = False
        # reset modal open state when reopening popup
        sa.modal_open = True
        just_opened = True

    if not hasattr(sa, "tp_error"):
        sa.tp_error = ""
    if not hasattr(sa, "tp_parsed"):
        sa.tp_parsed = None

    # modal_open is a bool, so we handle the 'X' button manually
    # by checking the second return value of begin_popup_modal.
    # PopupAutoSize lets imgui resize the window to fit content every
    # frame, so the dialog grows/shrinks as the user switches extensions
    # (.h5 reveals the H5 dataset name field, etc.).
    opened, visible = imgui.begin_popup_modal(
        "Save As",
        p_open=sa.modal_open,
        flags=sa.sizer.flags(imgui.WindowFlags_.no_saved_settings),
    )

    if opened:
        if not visible:
            # user closed via X button or Escape
            sa.modal_open = False
            imgui.close_current_popup()
            imgui.end_popup()
            return
    else:
        # If not opened, and we didn't just try to open it, ensure state is synced
        if not just_opened:
            sa.modal_open = False
        return

    # If we are here, popup is open and visible
    if opened:
        sa.modal_open = True

        if imgui.begin_tab_bar("SaveAsTabBar"):
            # === Save tab ===
            if imgui.begin_tab_item("Save")[0]:
                imgui.dummy(imgui.ImVec2(0, 5))

                # === PATH SECTION ===
                imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Output")
                imgui.dummy(imgui.ImVec2(0, 5))

                imgui.set_next_item_width(hello_imgui.em_size(25))

                # Directory
                current_dir_str = sa.outdir or ""
                changed, new_str = imgui.input_text("Save Dir", current_dir_str)
                if changed:
                    sa.outdir = new_str

                imgui.same_line()
                if not NATIVE_DIALOGS:
                    imgui.begin_disabled()
                if imgui.button("Browse"):
                    default_dir = sa.outdir or str(
                        get_last_dir("save_as") or Path.home()
                    )
                    sa.folder_dialog = pfd.select_folder(
                        "Select output folder", default_dir
                    )
                if not NATIVE_DIALOGS:
                    imgui.end_disabled()
                    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                        imgui.set_tooltip(no_dialog_hint())

                # Check if async folder dialog has a result
                if sa.folder_dialog is not None and sa.folder_dialog.ready():
                    result = sa.folder_dialog.result()
                    if result:
                        sa.outdir = str(result)
                        set_last_dir("save_as", result)
                    sa.folder_dialog = None

                # Extension
                imgui.set_next_item_width(hello_imgui.em_size(25))
                # combo bounds-check — if the persisted index points past the
                # filtered list (e.g. user previously selected an ext whose
                # backing pkg is no longer installed), clamp to .tiff.
                if sa.ext_idx >= len(MBO_AVAILABLE_FTYPES):
                    sa.ext_idx = MBO_AVAILABLE_FTYPES.index(".tiff")
                _, sa.ext_idx = imgui.combo("Ext", sa.ext_idx, MBO_AVAILABLE_FTYPES)
                sa.ext = MBO_AVAILABLE_FTYPES[sa.ext_idx]

                # H5-specific: let the user pick the internal dataset name.
                # only meaningful when saving .h5. uses the same 2-arg
                # input_text idiom as the rest of the dialog — passing a
                # third positional arg gets interpreted as imgui flags,
                # not a buffer size, which silently mangles the value.
                if sa.ext == ".h5":
                    imgui.set_next_item_width(hello_imgui.em_size(25))
                    _, sa.h5_dataset = imgui.input_text(
                        "H5 dataset name", sa.h5_dataset
                    )
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(
                            "Path to the dataset inside the .h5 file. "
                            "'mov' is the H5Array auto-detect default; "
                            "use whatever your downstream tool expects."
                        )

                imgui.spacing()
                imgui.separator()

                # === SELECTION SECTION ===
                _draw_selection_section(sa, src)

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                # === SAVE/OPTIONS BUTTONS ===
                _draw_save_button(sa, src)

                # Options popup (opened by button in _draw_save_button)
                _draw_options_popup(sa, src)

                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end_popup()


def _draw_options_popup(sa: SaveAs, src: SaveSource):
    """Draw the options popup for advanced save settings."""
    # track when options popup is about to open (for resetting defaults)
    if sa.options_requested:
        imgui.open_popup("Save Options")
        sa.options_requested = False
        # mark that we need to reset defaults on next frame when popup actually opens
        sa.options_reset = True

    # explicit size + centered position. imgui's default popup placement
    # anchors at the click site, which puts the popup near the bottom of the
    # screen when the Options button is at the bottom of the Save As dialog
    # — and tall content (video options) then gets clipped. centering on the
    # viewport sidesteps both problems.
    is_video = sa.ext == ".mp4"
    viewport = imgui.get_main_viewport()
    default_h = min(720 if is_video else 420, int(viewport.size.y * 0.9))
    center = imgui.ImVec2(
        viewport.pos.x + viewport.size.x * 0.5,
        viewport.pos.y + viewport.size.y * 0.5,
    )
    imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_size(imgui.ImVec2(420, default_h), imgui.Cond_.appearing)
    if imgui.begin_popup("Save Options"):
        # reset defaults on first frame popup is actually visible
        if sa.options_reset:
            sa.fix_phase = True
            sa.options_reset = False
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Options")
        imgui.dummy(imgui.ImVec2(0, 5))

        # Get available features for current data
        features = _get_array_features(src)

        _, sa.background = imgui.checkbox("Run in background", sa.background)
        set_tooltip(
            "Run save operation as a separate process that continues after closing the GUI. "
            "Progress will be logged to a file in the output directory."
        )

        sa.overwrite = checkbox_with_tooltip(
            "Overwrite", sa.overwrite, "Replace any existing output files."
        )

        # Z-registration: show disabled with reason if unavailable
        if not features.get("z_registration", False):
            imgui.begin_disabled()
        _changed, _reg_value = imgui.checkbox(
            "Register Z-Planes Axially",
            sa.register_z if features.get("z_registration") else False,
        )
        if features.get("z_registration") and _changed:
            sa.register_z = _reg_value
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            if src.planes <= 1:
                imgui.text_unformatted(
                    "Requires multi-plane (4D) data with more than one z-plane."
                )
            else:
                imgui.text_unformatted(
                    "Compute per-plane rigid shifts via phase correlation and "
                    "apply them on write. GPU (cupy) is used automatically if "
                    "available, otherwise CPU."
                )
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        if not features.get("z_registration", False):
            imgui.end_disabled()

        if features.get("z_registration", False) and sa.register_z:
            imgui.indent(16)
            imgui.set_next_item_width(80)
            _changed, _val = imgui.input_int("Max frames", sa.axial_max_frames, 50, 100)
            if _changed:
                sa.axial_max_frames = max(10, _val)
            set_tooltip(
                "Frames subsampled (evenly spaced) for the time-mean used to\n"
                "compute plane shifts. 200 is usually plenty."
            )

            imgui.set_next_item_width(80)
            _changed, _val = imgui.input_int(
                "Max shift (px)", sa.axial_max_shift, 10, 50
            )
            if _changed:
                sa.axial_max_shift = max(1, _val)
            set_tooltip("Max shift search radius in pixels. Default 150.")
            imgui.unindent(16)

        # Phase correction: only show if data supports it
        # uses separate _saveas_* settings (default True) instead of display settings
        if features.get("phase_correction", False):
            fix_phase_changed, fix_phase_value = imgui.checkbox(
                "Fix Scan Phase", sa.fix_phase
            )
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                imgui.text_unformatted("Correct for bi-directional scan phase offsets.")
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()
            if fix_phase_changed:
                sa.fix_phase = fix_phase_value

            use_fft_changed, use_fft_value = imgui.checkbox(
                "Subpixel Phase Correction", sa.use_fft
            )
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                imgui.text_unformatted(
                    "Use FFT-based subpixel registration (slower, more precise)."
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()
            if use_fft_changed:
                sa.use_fft = use_fft_value

        if features.get("frame_average", False):
            sa.frame_average = draw_frame_average_input(
                "Frame Average##saveas",
                sa.frame_average,
                viewer_factor=src.frame_average,
                max_frames=src.source_frames,
            )

        sa.debug = checkbox_with_tooltip(
            "Debug",
            sa.debug,
            "Print additional information to the terminal during process.",
        )

        imgui.spacing()
        imgui.text("Chunk Size (MB)")
        set_tooltip(
            "The size of the chunk, in MB, to read and write at a time. Larger chunks may be faster but use more memory.",
        )

        imgui.set_next_item_width(hello_imgui.em_size(15))
        _, sa.chunk_mb = imgui.drag_int(
            "##chunk_size_mb_mb",
            sa.chunk_mb,
            v_speed=1,
            v_min=1,
            v_max=1024,
        )

        # Format-specific options
        if sa.ext in (".zarr",):
            imgui.spacing()
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Zarr Options")
            imgui.dummy(imgui.ImVec2(0, 5))

            _, sa.zarr_sharded = imgui.checkbox("Sharded", sa.zarr_sharded)
            set_tooltip(
                "Use sharding to group multiple chunks into single files (100 frames/shard). "
                "Improves read/write performance for large datasets by reducing filesystem overhead.",
            )

            _, sa.zarr_ome = imgui.checkbox("OME-Zarr", sa.zarr_ome)
            set_tooltip(
                "Write OME-NGFF v0.5 metadata for compatibility with OME-Zarr viewers "
                "(napari, vizarr, etc). Includes multiscales, axes, and coordinate transforms.",
            )

            imgui.text("Compression Level")
            set_tooltip(
                "GZip compression level (0-9). Higher = smaller files, slower write. "
                "Level 1 is fast with decent compression. Level 0 disables compression.",
            )
            imgui.set_next_item_width(hello_imgui.em_size(10))
            _, sa.zarr_level = imgui.slider_int("##zarr_level", sa.zarr_level, 0, 9)

            imgui.spacing()
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.8, 1.0, 1.0), "Pyramid (Multi-resolution)"
            )
            imgui.dummy(imgui.ImVec2(0, 3))

            _, sa.zarr_pyramid = imgui.checkbox("Generate Pyramid", sa.zarr_pyramid)
            set_tooltip(
                "Write multi-resolution OME-Zarr pyramid.",
            )

            if sa.zarr_pyramid:
                imgui.text("Max Levels")
                set_tooltip(
                    "Levels beyond full-res. Factors set by voxel anisotropy.",
                )
                imgui.set_next_item_width(hello_imgui.em_size(8))
                _, sa.zarr_pyramid_levels = imgui.slider_int(
                    "##pyramid_levels", sa.zarr_pyramid_levels, 1, 6
                )

                imgui.text("Method")
                set_tooltip(
                    "median/mode = webknossos (intensity/labels). "
                    "mean/nearest/gaussian also available.",
                )
                methods = ["median", "mode", "mean", "nearest", "gaussian"]
                imgui.set_next_item_width(hello_imgui.em_size(10))
                if imgui.begin_combo("##pyramid_method", sa.zarr_pyramid_method):
                    for method in methods:
                        selected = method == sa.zarr_pyramid_method
                        if imgui.selectable(method, selected)[0]:
                            sa.zarr_pyramid_method = method
                        if selected:
                            imgui.set_item_default_focus()
                    imgui.end_combo()

        if sa.ext == ".mp4":
            _draw_video_options(sa, src)

        imgui.spacing()
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


_VIDEO_CODECS = ["libx264", "mpeg4"]
_VIDEO_QUALITY_PRESETS = ["preview", "high", "visually lossless", "lossless"]
_VIDEO_TEMPORAL_MODES = ["mean", "max", "std"]


def _preview_fps(src: SaveSource) -> float | None:
    """The shown array's sampling rate, when it has one."""
    fs = src.viewer.data[0].fs
    return float(fs) if fs and fs > 0 else None


def _preview_cmap(src: SaveSource) -> str | None:
    cmap = src.viewer.graphics[0].cmap
    return None if cmap is None else cmap.name.split(":")[-1]


def _preview_vmin_vmax(src: SaveSource) -> tuple[float, float]:
    graphic = src.viewer.graphics[0]
    return float(graphic.vmin), float(graphic.vmax)


def _sync_video_options_from_preview(sa: SaveAs, src: SaveSource) -> None:
    """Pull every video-export field from the live preview state in one shot."""
    synced: list[str] = []

    fps = _preview_fps(src)
    if fps is not None:
        sa.video_fps = max(1, int(round(fps)))
        synced.append(f"fps={sa.video_fps}")

    vv = _preview_vmin_vmax(src)
    sa.video_vmin, sa.video_vmax = vv
    sa.video_auto = False
    synced.append(f"vmin/vmax=[{vv[0]:.1f}, {vv[1]:.1f}]")

    cmap = _preview_cmap(src)
    if cmap is not None:
        if cmap not in sa.video_cmaps:
            sa.video_cmaps = [cmap, *sa.video_cmaps]
        sa.video_cmap_idx = sa.video_cmaps.index(cmap)
        synced.append(f"cmap={cmap!r}")

    sa.video_mean_subtract = src.mean_subtraction
    synced.append(f"mean_subtract={src.mean_subtraction}")

    sa.video_temporal_mode_idx = _VIDEO_TEMPORAL_MODES.index(src.projection)
    synced.append(f"temporal_mode={src.projection!r}")
    sa.video_temporal_smooth = src.window
    synced.append(f"window={src.window}")
    sa.video_spatial_smooth = float(src.sigma)
    synced.append(f"sigma={src.sigma:.2f}")

    if synced:
        logger.info("Synced from preview: " + ", ".join(synced))
    else:
        logger.warning(
            "Sync from preview found no usable values (fs not set, no graphics, etc.)"
        )


def _write_mean_subtract_stack(sa: SaveAs, src: SaveSource) -> str | None:
    """If mean-subtract is enabled and zstats are computed, write a (C, Z, Y, X)
    npy file and return its path. Returns None if not applicable / unavailable.

    ``zstats.means[i]`` is a dict keyed by the breakout-combo tuple; this
    writer picks the combo currently shown in the stats tab (which follows
    the sliders) so the saved stack matches what the user sees in the GUI.
    """
    if not sa.video_mean_subtract:
        return None
    means = src.zstats.means
    done = src.zstats.done
    if not means or not done:
        return None
    import tempfile

    import numpy as np

    from mbo_utilities.gui._stats import current_breakout_key

    stacks = []
    for i in range(len(means)):
        if i >= len(done) or not done[i]:
            return None
        slot = means[i]
        if not isinstance(slot, dict) or not slot:
            return None
        arr = slot.get(current_breakout_key(src.zstats, i))
        if arr is None:
            arr = slot.get(())
        if arr is None:
            arr = next(iter(slot.values()))
        if arr is None:
            return None
        stacks.append(np.asarray(arr, dtype=np.float32))
    if not stacks:
        return None
    stack = np.stack(stacks, axis=0)  # (C, Z, Y, X)
    fd, path = tempfile.mkstemp(prefix="mbo_meansub_", suffix=".npy")
    import os

    os.close(fd)
    np.save(path, stack)
    return path


def _draw_video_options(sa: SaveAs, src: SaveSource):
    """Video-specific options shown when ext is .mp4."""
    imgui.spacing()
    imgui.separator()
    imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Video Options")
    imgui.same_line()
    if imgui.button("Sync from preview"):
        _sync_video_options_from_preview(sa, src)
    set_tooltip(
        "Pull current values from the preview widget: fps (metadata.fs), "
        "vmin/vmax (histogram), colormap, mean-subtract toggle, "
        "temporal projection mode, window size, and gaussian sigma."
    )
    imgui.dummy(imgui.ImVec2(0, 5))

    imgui.set_next_item_width(hello_imgui.em_size(10))
    _, sa.video_fps = imgui.input_int("FPS", sa.video_fps, step=1, step_fast=10)
    sa.video_fps = max(1, min(int(sa.video_fps), 240))
    set_tooltip("Base frame rate of the recording. Type a value or use +/-.")

    imgui.set_next_item_width(hello_imgui.em_size(10))
    _, sa.video_speed_factor = imgui.input_float(
        "Speed",
        sa.video_speed_factor,
        step=0.1,
        step_fast=1.0,
        format="%.2f",
    )
    sa.video_speed_factor = max(0.1, min(float(sa.video_speed_factor), 100.0))
    set_tooltip(
        "Playback speed multiplier. Type any value (e.g. 10, 10.5). "
        "Output FPS = FPS * Speed."
    )

    imgui.spacing()
    _, sa.video_auto = imgui.checkbox("Auto contrast", sa.video_auto)
    set_tooltip("Use percentile-based intensity scaling instead of explicit vmin/vmax.")

    if sa.video_auto:
        imgui.set_next_item_width(hello_imgui.em_size(10))
        _, sa.video_vmin_pct = imgui.input_float(
            "vmin %",
            sa.video_vmin_pct,
            step=0.1,
            step_fast=1.0,
            format="%.1f",
        )
        sa.video_vmin_pct = max(0.0, min(sa.video_vmin_pct, 10.0))
        set_tooltip("Percentile for auto vmin. Lower = darker blacks.")

        imgui.set_next_item_width(hello_imgui.em_size(10))
        _, sa.video_vmax_pct = imgui.input_float(
            "vmax %",
            sa.video_vmax_pct,
            step=0.1,
            step_fast=1.0,
            format="%.1f",
        )
        sa.video_vmax_pct = max(90.0, min(sa.video_vmax_pct, 100.0))
        set_tooltip("Percentile for auto vmax. Lower = brighter highlights.")
    else:
        imgui.set_next_item_width(hello_imgui.em_size(12))
        _, sa.video_vmin = imgui.input_float(
            "vmin",
            sa.video_vmin,
            step=10.0,
            step_fast=100.0,
            format="%.1f",
        )
        set_tooltip("Min intensity value (clipped to black).")

        imgui.set_next_item_width(hello_imgui.em_size(12))
        _, sa.video_vmax = imgui.input_float(
            "vmax",
            sa.video_vmax,
            step=10.0,
            step_fast=100.0,
            format="%.1f",
        )
        set_tooltip("Max intensity value (clipped to white).")

    imgui.spacing()
    _, sa.video_mean_subtract = imgui.checkbox("Mean subtract", sa.video_mean_subtract)
    set_tooltip(
        "Subtract the per-(channel, plane) mean image from every frame before "
        "contrast scaling. Requires z-stats to be computed."
    )

    _, sa.video_time_overlay = imgui.checkbox("Time overlay", sa.video_time_overlay)
    set_tooltip(
        "Draw a clock in the top-left of every frame showing recording time "
        "(frame_index / fps). The clock always reflects real recording seconds, "
        "so a sped-up clip ticks faster on screen — useful for showing how fast "
        "playback is relative to the source."
    )

    _, sa.video_scalebar = imgui.checkbox("Scalebar", sa.video_scalebar)
    set_tooltip(
        "Draw a scalebar at exactly 10% of frame width in the bottom-left, "
        "labeled with the matching micron length. Multiply the label by 10 "
        "to read the full-frame width. Pixel size is taken from arr.dx."
    )

    imgui.set_next_item_width(hello_imgui.em_size(8))
    _, sa.video_upscale = imgui.input_int("Upscale", sa.video_upscale)
    sa.video_upscale = max(0, sa.video_upscale)
    set_tooltip(
        "Integer magnification of every frame, applied before the scalebar and\n"
        "clock are drawn so their text is rasterised at the output size instead\n"
        "of being smeared across a handful of pixels.\n"
        "\n"
        "Each source pixel becomes an NxN block (nearest-neighbour): nothing is\n"
        "interpolated and no detail is invented, so 'lossless' stays exact.\n"
        "\n"
        "0 = auto: lifts the short side to 480px (long side capped at 2048px).\n"
        "Frames already that big are left alone. Set 1 to disable."
    )

    imgui.set_next_item_width(hello_imgui.em_size(8))
    _, sa.video_temporal_mode_idx = imgui.combo(
        "Temporal mode",
        sa.video_temporal_mode_idx,
        _VIDEO_TEMPORAL_MODES,
    )
    set_tooltip(
        "Rolling-window aggregation: mean smooths flicker, max emphasizes "
        "transient activity, std highlights variance."
    )

    imgui.set_next_item_width(hello_imgui.em_size(10))
    _, sa.video_temporal_smooth = imgui.input_int(
        "Window", sa.video_temporal_smooth, step=1, step_fast=10
    )
    sa.video_temporal_smooth = max(0, min(int(sa.video_temporal_smooth), 500))
    set_tooltip("Rolling-window size in frames. 0 = off; combined with Temporal mode.")

    imgui.set_next_item_width(hello_imgui.em_size(10))
    _, sa.video_spatial_smooth = imgui.input_float(
        "Spatial smooth",
        sa.video_spatial_smooth,
        step=0.1,
        step_fast=1.0,
        format="%.2f",
    )
    sa.video_spatial_smooth = max(0.0, min(float(sa.video_spatial_smooth), 20.0))
    set_tooltip("Gaussian blur sigma in pixels. 0 = off.")

    imgui.set_next_item_width(hello_imgui.em_size(10))
    _, sa.video_gamma = imgui.input_float(
        "Gamma",
        sa.video_gamma,
        step=0.05,
        step_fast=0.5,
        format="%.2f",
    )
    sa.video_gamma = max(0.1, min(float(sa.video_gamma), 5.0))
    set_tooltip("Gamma correction. <1 brightens midtones, >1 darkens them.")

    imgui.spacing()
    imgui.set_next_item_width(hello_imgui.em_size(14))
    _, sa.video_cmap_idx = imgui.combo("Colormap", sa.video_cmap_idx, sa.video_cmaps)
    set_tooltip(
        "Colormap (cmap-library / fastplotlib names). 'gray' writes grayscale. "
        "Click 'Sync from preview' to adopt the active fpl cmap if it's not in this list."
    )

    imgui.set_next_item_width(hello_imgui.em_size(14))
    _, sa.video_quality_idx = imgui.combo(
        "Quality", sa.video_quality_idx, _VIDEO_QUALITY_PRESETS
    )
    set_tooltip(
        "Quality preset (libx264), fastest/smallest at the top:\n"
        "  preview            crf 23, veryfast, yuvj420p  (rough check)\n"
        "  high               crf 17, medium,   yuvj420p  (slides, talks)\n"
        "  visually lossless  crf  1, slow,     yuvj420p  (max error 8/255)\n"
        "  lossless           qp   0, veryslow, yuvj444p  (bit-exact)\n"
        "\n"
        "The top three are standard High profile and open anywhere.\n"
        "'lossless' is High 4:4:4 Predictive: VLC, mpv, ImageJ and ffmpeg\n"
        "play it, but Chrome, Windows Photos and PowerPoint do not. Pick\n"
        "'visually lossless' if it has to open in a browser or a deck.\n"
        "Colormapped movies keep exact color only at 'lossless' (4:4:4).\n"
        "mpeg4 maps these to -qscale:v 8/4/2/1."
    )

    imgui.set_next_item_width(hello_imgui.em_size(12))
    _, sa.video_codec_idx = imgui.combo("Codec", sa.video_codec_idx, _VIDEO_CODECS)
    set_tooltip("Video codec. libx264 is best for browser playback.")


def _draw_selection_section(sa: SaveAs, src: SaveSource):
    """Draw the selection section with text input for dimension slicing and output path."""
    imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Selection")
    imgui.same_line()
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(
            "Select which data to save using start:stop:step notation.\n\n"
            "Format: start:stop or start:stop:step\n"
            "  start = first index (1-based)\n"
            "  stop = last index (inclusive)\n"
            "  step = interval (default 1)\n\n"
            "To exclude frames, add comma + exclude range:\n"
            "  1:100,50:60 = frames 1-100 excluding 50-60\n\n"
            "Examples:\n"
            "  1:100 = frames 1-100\n"
            "  1:100:2 = every other frame (1,3,5...99)\n"
            "  1:1000,200:300 = 1-1000 excluding 200-300"
        )
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()
    imgui.dummy(imgui.ImVec2(0, 5))

    # get data dimensions - store reference once
    data = None
    max_frames = 1000
    num_planes = 1
    num_channels = 1
    try:
        data = src.viewer.data[0]
        max_frames = int(getattr(data, "num_timepoints", None) or data.shape[0])
        # get actual z-planes
        if hasattr(data, "num_planes"):
            num_planes = data.num_planes
        else:
            num_planes = 1
        # C axis: num_views for IsoView (cameras), else color channels
        if hasattr(data, "num_views"):
            num_channels = data.num_views
        elif hasattr(data, "num_color_channels"):
            num_channels = data.num_color_channels
        elif hasattr(data, "_num_color_channels"):
            num_channels = data._num_color_channels
        else:
            num_channels = 1
    except Exception as e:
        hello_imgui.log(
            hello_imgui.LogLevel.error, f"Could not read data dimensions: {e}"
        )

    # track file path to reset state when file changes
    current_fpath = src.fpath
    current_fpath = str(current_fpath) if current_fpath else ""
    file_changed = False
    if not hasattr(sa, "last_fpath"):
        sa.last_fpath = current_fpath
    elif sa.last_fpath != current_fpath:
        file_changed = True
        sa.last_fpath = current_fpath

    # reset selection state when file changes
    if file_changed:
        sa.tp_selection = f"1:{max_frames}"
        sa.tp_error = ""
        sa.tp_parsed = None
        sa.last_max_tp = max_frames
        sa.z_start = 1
        sa.z_stop = num_planes
        sa.z_step = 1
        sa.last_num_planes = num_planes
        sa.c_start = 1
        sa.c_stop = num_channels
        sa.c_step = 1
        sa.last_num_channels = num_channels

    # initialize timepoint selection state (new single text input)
    if not hasattr(sa, "tp_selection"):
        sa.tp_selection = f"1:{max_frames}"
    if not hasattr(sa, "tp_error"):
        sa.tp_error = ""
    if not hasattr(sa, "tp_parsed"):
        sa.tp_parsed = None
    if not hasattr(sa, "last_max_tp"):
        sa.last_max_tp = max_frames
    elif sa.last_max_tp != max_frames:
        # update default selection when data changes
        sa.last_max_tp = max_frames
        sa.tp_selection = f"1:{max_frames}"
        sa.tp_parsed = None
        sa.tp_error = ""

    # initialize z-plane selection state
    if not hasattr(sa, "z_start"):
        sa.z_start = 1
    if not hasattr(sa, "z_stop"):
        sa.z_stop = num_planes
    if not hasattr(sa, "z_step"):
        sa.z_step = 1
    if not hasattr(sa, "last_num_planes"):
        sa.last_num_planes = num_planes
    elif sa.last_num_planes != num_planes:
        sa.last_num_planes = num_planes
        sa.z_start = 1
        sa.z_stop = num_planes
        sa.z_step = 1

    # initialize channel selection state
    if not hasattr(sa, "c_start"):
        sa.c_start = 1
    if not hasattr(sa, "c_stop"):
        sa.c_stop = num_channels
    if not hasattr(sa, "c_step"):
        sa.c_step = 1
    if not hasattr(sa, "last_num_channels"):
        sa.last_num_channels = num_channels
    elif sa.last_num_channels != num_channels:
        sa.last_num_channels = num_channels
        sa.c_start = 1
        sa.c_stop = num_channels
        sa.c_step = 1

    # draw selection table using shared component (includes suffix row).
    # row labels track the loaded array's slider names (isoview Tile/Cam/
    # Zplane, etc.); the selected indices are unaffected.
    tp_label, z_label, c_label = resolve_dim_labels(src.viewer)
    tp_parsed, z_start, z_stop, z_step, c_start, c_stop, c_step = draw_selection_table(
        sa,
        max_frames,
        num_planes,
        tp_attr="tp",
        z_attr="z",
        id_suffix="_saveas",
        suffix_attr="output_suffix",
        num_channels=num_channels,
        c_attr="c",
        tp_label=tp_label,
        z_label=z_label,
        c_label=c_label,
    )

    # update legacy _selected_planes for compatibility
    if num_planes > 1:
        selected_planes = list(range(z_start, z_stop + 1, z_step))
        sa.planes = set(p - 1 for p in selected_planes)
    else:
        sa.planes = {0}

    # track selected channels
    if num_channels > 1:
        selected_channels = list(range(c_start, c_stop + 1, c_step))
        sa.channels = set(c - 1 for c in selected_channels)
    else:
        sa.channels = {0}

    # parse selection if not already done (initial load)
    if sa.tp_parsed is None and not sa.tp_error:
        try:
            sa.tp_parsed = parse_timepoint_selection(sa.tp_selection, max_frames)
        except ValueError as e:
            sa.tp_error = str(e)

    imgui.spacing()

    # build filename preview
    ext = getattr(sa, "ext", ".tiff").lstrip(".")
    is_video = f".{ext}" == ".mp4"
    tags = []

    # timepoint tag from parsed selection
    tp_parsed = sa.tp_parsed
    t_tag = None
    if tp_parsed:
        final_indices = tp_parsed.final_indices
        if final_indices:
            # convert 0-based back to 1-based for display
            tp_start_1 = final_indices[0] + 1
            tp_stop_1 = final_indices[-1] + 1
            # detect step from indices
            if len(final_indices) > 1:
                tp_step = final_indices[1] - final_indices[0]
            else:
                tp_step = 1

            if tp_start_1 == tp_stop_1:
                t_tag = DimensionTag(
                    TAG_REGISTRY["T"], start=tp_start_1, stop=None, step=1
                )
            else:
                t_tag = DimensionTag(
                    TAG_REGISTRY["T"],
                    start=tp_start_1,
                    stop=tp_stop_1,
                    step=tp_step if tp_step != 1 else 1,
                )
        n_frames = tp_parsed.count
    else:
        n_frames = 0

    z_start = getattr(sa, "z_start", 1)
    z_stop = getattr(sa, "z_stop", 1)
    z_step = getattr(sa, "z_step", 1)

    if is_video:
        # video writes one file per (z, channel). Preview shows the first
        # such file (z_start, c_start) with order zplane_ch_tp.
        z_tag = DimensionTag(TAG_REGISTRY["Z"], start=z_start, stop=None, step=1)
        c_tag = DimensionTag(TAG_REGISTRY["C"], start=c_start, stop=None, step=1)
        tags.append(z_tag)
        tags.append(c_tag)
        if t_tag is not None:
            tags.append(t_tag)
    else:
        # other formats: T, C (multi-channel only), Z (multi-plane only)
        if t_tag is not None:
            tags.append(t_tag)
        if num_channels > 1:
            if c_start == c_stop:
                c_tag = DimensionTag(
                    TAG_REGISTRY["C"], start=c_start, stop=None, step=1
                )
            else:
                c_tag = DimensionTag(
                    TAG_REGISTRY["C"],
                    start=c_start,
                    stop=c_stop,
                    step=c_step if c_step != 1 else 1,
                )
            tags.append(c_tag)
        if num_planes > 1:
            if z_start == z_stop:
                z_tag = DimensionTag(
                    TAG_REGISTRY["Z"], start=z_start, stop=None, step=1
                )
            else:
                z_tag = DimensionTag(
                    TAG_REGISTRY["Z"],
                    start=z_start,
                    stop=z_stop,
                    step=z_step if z_step != 1 else 1,
                )
            tags.append(z_tag)

    # build filename
    suffix = getattr(sa, "output_suffix", "")
    sanitized_suffix = _sanitize_suffix(suffix).lstrip("_") if suffix else ""
    if is_video and not sanitized_suffix:
        sanitized_suffix = "movie"
    if tags:
        dim_parts = "_".join(tag.to_string() for tag in tags)
        if sanitized_suffix:
            filename = f"{dim_parts}_{sanitized_suffix}.{ext}"
        else:
            filename = f"{dim_parts}.{ext}"
    else:
        filename = f"{sanitized_suffix}.{ext}" if sanitized_suffix else f"output.{ext}"

    # calculate output info
    n_planes_out = len(range(z_start, z_stop + 1, z_step)) if num_planes > 1 else 1
    n_channels_out = len(range(c_start, c_stop + 1, c_step)) if num_channels > 1 else 1

    # get image dimensions and dtype for size estimate
    Ly, Lx = 512, 512
    dtype_size = 2  # default assume uint16
    if data is not None:
        try:
            # try shape directly (Y, X are last two dims)
            if hasattr(data, "shape") and len(data.shape) >= 2:
                Ly, Lx = data.shape[-2], data.shape[-1]
            # fallback to metadata if shape is lazy/not available
            elif hasattr(data, "metadata") and isinstance(data.metadata, dict):
                meta = data.metadata
                Ly = (
                    meta.get("Ly")
                    or meta.get("height")
                    or meta.get("frame_height")
                    or 512
                )
                Lx = (
                    meta.get("Lx")
                    or meta.get("width")
                    or meta.get("frame_width")
                    or 512
                )
            if hasattr(data, "dtype"):
                dtype_size = data.dtype.itemsize
        except Exception:
            pass  # keep defaults

    # estimate file size (raw data size, compression varies)
    if is_video:
        # video: one (T, Y, X) file per (z, c). codec compresses heavily;
        # raw uncompressed size is an upper-bound proxy.
        raw_bytes = n_frames * Ly * Lx * dtype_size
    elif num_channels > 1:
        raw_bytes = n_frames * n_channels_out * Ly * Lx * dtype_size
    else:
        raw_bytes = n_frames * n_planes_out * Ly * Lx * dtype_size
    if raw_bytes >= 1e9:
        size_str = f"~{raw_bytes / 1e9:.1f} GB"
    elif raw_bytes >= 1e6:
        size_str = f"~{raw_bytes / 1e6:.0f} MB"
    else:
        size_str = f"~{raw_bytes / 1e3:.0f} KB"

    n_video_files = n_planes_out * n_channels_out if is_video else 0
    if is_video and n_video_files > 1:
        size_str = f"{size_str} per file (uncompressed) × {n_video_files} files"
    elif is_video:
        size_str = f"{size_str} (uncompressed)"

    # output shape string
    if is_video:
        shape_str = f"({n_frames}, {Ly}, {Lx})"
        dims_str = "TYX per file"
    elif num_planes > 1:
        shape_str = f"({n_frames}, {n_planes_out}, {Ly}, {Lx})"
        dims_str = "TZYX"
    elif num_channels > 1:
        shape_str = f"({n_frames}, {n_channels_out}, {Ly}, {Lx})"
        dims_str = "TCYX"
    else:
        shape_str = f"({n_frames}, {Ly}, {Lx})"
        dims_str = "TYX"

    imgui.spacing()

    # output preview table
    table_flags = (
        imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
    )
    if imgui.begin_table("output_preview", 2, table_flags):
        imgui.table_setup_column(
            "label", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(6)
        )
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)

        # filename row
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Filename")
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.6, 0.9, 0.6, 1.0), filename)
        outdir = getattr(sa, "outdir", "")
        if imgui.is_item_hovered() and outdir:
            imgui.begin_tooltip()
            display_path = str(Path(outdir)).replace("\\", "/")
            imgui.text_unformatted(f"{display_path}/{filename}")
            imgui.end_tooltip()

        # size row
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Size")
        imgui.table_next_column()
        imgui.text(size_str)

        # shape row
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Shape")
        imgui.table_next_column()
        imgui.text(f"{shape_str} {dims_str}")

        imgui.end_table()


def _draw_save_button(sa: SaveAs, src: SaveSource):
    """Draw the save/cancel buttons and handle save logic."""
    no_planes = sa.planes is not None and len(sa.planes) == 0
    no_valid_tp = sa.tp_error or sa.tp_parsed is None

    # set by _draw_selection_section
    num_channels = getattr(sa, "last_num_channels", 1)

    averaged = sa.frame_average
    if averaged > 1:
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.8, 0.6, 1.0),
            f"Frame average: {averaged} frames per saved frame",
        )
        set_tooltip(
            "The output is temporally binned by this factor (change it under"
            " Options). Timepoints above are counted in averaged frames."
        )

    if no_planes:
        imgui.begin_disabled()
        imgui.button("Save", imgui.ImVec2(100, 0))
        imgui.end_disabled()
        imgui.same_line()
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.4, 0.4, 1.0), "Select at least one z-plane"
        )
    elif no_valid_tp:
        imgui.begin_disabled()
        imgui.button("Save", imgui.ImVec2(100, 0))
        imgui.end_disabled()
        imgui.same_line()
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.4, 0.4, 1.0), "Invalid timepoint selection"
        )
    elif imgui.button("Save", imgui.ImVec2(100, 0)):
        if not sa.outdir:
            last_dir = get_last_dir("save_as") or Path().home()
            sa.outdir = str(last_dir)
        try:
            save_planes = [p + 1 for p in sa.planes]
            save_channels = [c + 1 for c in sa.channels]

            # Validate that at least one plane is selected
            if not save_planes:
                logger.error("No z-planes selected! Please select at least one plane.")
            else:
                sa.total = len(save_planes)
                if sa.split_rois:
                    if not sa.selected_rois or len(sa.selected_rois) == 0:
                        # Get mROI count from data array (ScanImage-specific)
                        try:
                            mroi_count = src.viewer.data[0].num_rois
                        except Exception:
                            mroi_count = 1
                        sa.selected_rois = set(range(mroi_count))
                    # Convert 0-indexed UI values to 1-indexed ROI values for ScanImageArray
                    rois = sorted([r + 1 for r in sa.selected_rois])
                else:
                    rois = None

                outdir = Path(sa.outdir).expanduser()
                if not outdir.exists():
                    outdir.mkdir(parents=True, exist_ok=True)

                # build frames list from parsed timepoint selection
                tp_parsed = sa.tp_parsed
                try:
                    _d = src.viewer.data[0]
                    max_timepoints = int(
                        getattr(_d, "num_timepoints", None) or _d.shape[0]
                    )
                except (IndexError, AttributeError):
                    max_timepoints = 1000

                # get final frame indices (0-based)
                final_indices_0 = (
                    tp_parsed.final_indices
                    if tp_parsed
                    else list(range(max_timepoints))
                )

                # check if selecting all frames (None means all)
                if len(final_indices_0) == max_timepoints and final_indices_0 == list(
                    range(max_timepoints)
                ):
                    frames = None
                else:
                    # convert to 1-based for imwrite
                    frames = [i + 1 for i in final_indices_0]

                # Build metadata overrides dict from custom metadata
                metadata_overrides = dict(src.metadata)

                # add timepoint selection metadata (include/exclude info)
                if tp_parsed:
                    tp_meta = tp_parsed.to_metadata()
                    metadata_overrides["timepoint_selection"] = tp_meta

                # add channel selection metadata
                if len(save_channels) < num_channels:
                    metadata_overrides["num_color_channels"] = len(save_channels)
                    metadata_overrides["channel_selection"] = save_channels

                # Determine output_suffix: only use custom suffix for multi-ROI stitched data
                output_suffix = None
                if rois is None:
                    # Stitching all ROIs - use custom suffix (or default "_stitched")
                    output_suffix = sa.output_suffix

                # determine roi_mode based on whether splitting ROIs
                from mbo_utilities.metadata import RoiMode

                roi_mode = RoiMode.separate if rois else RoiMode.concat_y

                save_kwargs = {
                    "path": src.fpath,
                    "outpath": sa.outdir,
                    "planes": save_planes,
                    "channels": save_channels
                    if len(save_channels) < num_channels
                    else None,
                    "frames": frames,
                    "roi": rois,
                    "roi_mode": roi_mode,
                    "overwrite": sa.overwrite,
                    "debug": sa.debug,
                    "ext": sa.ext,
                    "target_chunk_mb": sa.chunk_mb,
                    # scan-phase correction settings (separate from display)
                    "fix_phase": sa.fix_phase,
                    "use_fft": sa.use_fft,
                    "border": getattr(src.viewer.data[0], "border", 3),
                    "max_offset": getattr(src.viewer.data[0], "max_offset", 3),
                    # temporal binning, seeded from the viewer's "Apply to dataset"
                    "frame_average": sa.frame_average,
                    "register_z": sa.register_z,
                    "max_frames": sa.axial_max_frames,
                    "max_reg_xy": sa.axial_max_shift,
                    "mean_subtraction": src.mean_subtraction,
                    "progress_callback": sa.progress_callback,
                    # metadata overrides
                    "metadata": metadata_overrides if metadata_overrides else None,
                    # filename suffix
                    "output_suffix": output_suffix,
                }
                # Add zarr-specific options if saving to zarr
                if sa.ext == ".zarr":
                    save_kwargs["sharded"] = sa.zarr_sharded
                    save_kwargs["ome"] = sa.zarr_ome
                    save_kwargs["level"] = sa.zarr_level
                    save_kwargs["pyramid"] = sa.zarr_pyramid
                    if sa.zarr_pyramid:
                        save_kwargs["pyramid_max_layers"] = sa.zarr_pyramid_levels
                        save_kwargs["pyramid_method"] = sa.zarr_pyramid_method

                # H5-specific: dataset name (default "mov" for suite2p compat)
                if sa.ext == ".h5":
                    save_kwargs["dataset_name"] = sa.h5_dataset

                # Video options (.mp4)
                if sa.ext == ".mp4":
                    cmap_name = sa.video_cmaps[sa.video_cmap_idx]
                    save_kwargs["fps"] = sa.video_fps
                    save_kwargs["speed_factor"] = sa.video_speed_factor
                    if sa.video_auto:
                        save_kwargs["vmin"] = None
                        save_kwargs["vmax"] = None
                    else:
                        save_kwargs["vmin"] = sa.video_vmin
                        save_kwargs["vmax"] = sa.video_vmax
                    save_kwargs["vmin_percentile"] = sa.video_vmin_pct
                    save_kwargs["vmax_percentile"] = sa.video_vmax_pct
                    save_kwargs["temporal_smooth"] = sa.video_temporal_smooth
                    save_kwargs["spatial_smooth"] = sa.video_spatial_smooth
                    save_kwargs["gamma"] = sa.video_gamma
                    save_kwargs["cmap"] = cmap_name
                    save_kwargs["quality"] = _VIDEO_QUALITY_PRESETS[
                        sa.video_quality_idx
                    ]
                    save_kwargs["codec"] = _VIDEO_CODECS[sa.video_codec_idx]
                    save_kwargs["temporal_mode"] = _VIDEO_TEMPORAL_MODES[
                        sa.video_temporal_mode_idx
                    ]
                    save_kwargs["time_overlay"] = sa.video_time_overlay
                    save_kwargs["scalebar"] = sa.video_scalebar
                    save_kwargs["upscale"] = sa.video_upscale or None
                    mean_sub_path = _write_mean_subtract_stack(sa, src)
                    if mean_sub_path:
                        save_kwargs["mean_subtract_path"] = mean_sub_path
                    elif sa.video_mean_subtract:
                        logger.warning(
                            "Mean subtract enabled but z-stats not computed — skipping."
                        )

                n_frames = len(frames) if frames else max_timepoints
                # build frames message from parsed selection
                if tp_parsed and tp_parsed.exclude_str:
                    frames_msg = f"{n_frames} frames ({tp_parsed.include_str} excl. {tp_parsed.exclude_str})"
                elif tp_parsed:
                    frames_msg = f"{n_frames} frames ({tp_parsed.include_str})"
                else:
                    frames_msg = "all frames"
                num_rois = getattr(src.viewer.data[0], "num_rois", 1)
                if rois:
                    roi_msg = f", ROIs {rois}"
                elif num_rois > 1:
                    roi_msg = f", {roi_mode.description}"
                else:
                    roi_msg = ""
                channels_msg = (
                    f", channels {save_channels}"
                    if len(save_channels) < num_channels
                    else ""
                )
                avg_msg = (
                    f", {sa.frame_average} frames averaged"
                    if sa.frame_average > 1
                    else ""
                )
                logger.info(
                    f"Saving planes {save_planes}{channels_msg} ({frames_msg}){roi_msg}{avg_msg}"
                )
                logger.info(f"Saving to {sa.outdir} as {sa.ext}")

                # check if running as background process
                if sa.background:
                    # spawn as detached subprocess via process manager
                    pm = get_process_manager()
                    src = Path(src.fpath) if src.fpath else None
                    input_path = str(src) if src else ""
                    fname = (src.name or "data") if src else "data"
                    is_video = sa.ext == ".mp4"
                    video_kwargs = {}
                    if is_video:
                        cmap_name = sa.video_cmaps[sa.video_cmap_idx]
                        video_kwargs = {
                            "fps": sa.video_fps,
                            "speed_factor": sa.video_speed_factor,
                            "vmin": None if sa.video_auto else sa.video_vmin,
                            "vmax": None if sa.video_auto else sa.video_vmax,
                            "vmin_percentile": sa.video_vmin_pct,
                            "vmax_percentile": sa.video_vmax_pct,
                            "temporal_smooth": sa.video_temporal_smooth,
                            "spatial_smooth": sa.video_spatial_smooth,
                            "gamma": sa.video_gamma,
                            "cmap": cmap_name,
                            "quality": _VIDEO_QUALITY_PRESETS[sa.video_quality_idx],
                            "codec": _VIDEO_CODECS[sa.video_codec_idx],
                            "temporal_mode": _VIDEO_TEMPORAL_MODES[
                                sa.video_temporal_mode_idx
                            ],
                            "time_overlay": sa.video_time_overlay,
                            "scalebar": sa.video_scalebar,
                            "upscale": sa.video_upscale or None,
                        }
                        mean_sub_path = _write_mean_subtract_stack(sa, src)
                        if mean_sub_path:
                            video_kwargs["mean_subtract_path"] = mean_sub_path
                        elif sa.video_mean_subtract:
                            logger.warning(
                                "Mean subtract enabled but z-stats not computed — skipping."
                            )

                    worker_args = {
                        "input_path": input_path,
                        # multi-dataset sources (.mesc) need the selector so
                        # the worker re-opens the unit on screen, not unit 0
                        "reader_kwargs": widget_reader_kwargs(src.viewer),
                        "output_path": str(sa.outdir),
                        "ext": sa.ext,
                        "planes": save_planes,
                        "channels": save_channels
                        if len(save_channels) < num_channels
                        else None,
                        "frames": frames,
                        "rois": rois,
                        "fix_phase": sa.fix_phase,
                        "use_fft": sa.use_fft,
                        "frame_average": sa.frame_average,
                        "register_z": sa.register_z,
                        "max_frames": sa.axial_max_frames,
                        "max_reg_xy": sa.axial_max_shift,
                        "metadata": metadata_overrides if metadata_overrides else {},
                        "kwargs": {
                            "sharded": sa.zarr_sharded if sa.ext == ".zarr" else False,
                            "ome": sa.zarr_ome if sa.ext == ".zarr" else False,
                            "output_suffix": output_suffix,
                            "pyramid": sa.zarr_pyramid if sa.ext == ".zarr" else False,
                            "pyramid_max_layers": sa.zarr_pyramid_levels
                            if sa.ext == ".zarr" and sa.zarr_pyramid
                            else 4,
                            "pyramid_method": sa.zarr_pyramid_method
                            if sa.ext == ".zarr" and sa.zarr_pyramid
                            else "median",
                            # h5: dataset name inside the .h5 file (default "mov")
                            **(
                                {"dataset_name": sa.h5_dataset}
                                if sa.ext == ".h5"
                                else {}
                            ),
                            **video_kwargs,
                        },
                    }
                    pid = pm.spawn(
                        task_type="save_as",
                        args=worker_args,
                        description=f"Saving {fname} to {sa.ext}",
                        output_path=str(sa.outdir),
                    )
                    if pid:
                        logger.info(f"Started background save process (PID {pid})")
                        logger.info("You can close the GUI - the save will continue.")
                    else:
                        logger.error("Failed to start background process")
                else:
                    # run in foreground thread (existing behavior)
                    sa.progress = 0.0
                    sa.done = False
                    sa.running = True
                    logger.info("Starting save operation...")
                    # Also reset register_z progress if enabled
                    if sa.register_z:
                        sa.register_progress = 0.0
                        sa.register_done = False
                        sa.register_running = True
                        sa.register_msg = "Starting..."
                    threading.Thread(
                        target=_save_as_worker, kwargs=save_kwargs, daemon=True
                    ).start()
            sa.modal_open = False
            imgui.close_current_popup()
        except Exception as e:
            logger.info(f"Error saving data: {e}")
            sa.modal_open = False
            imgui.close_current_popup()

    imgui.same_line()
    if imgui.button("Options", imgui.ImVec2(80, 0)):
        sa.options_requested = True

    imgui.same_line()
    if imgui.button("Cancel"):
        sa.modal_open = False
        imgui.close_current_popup()
