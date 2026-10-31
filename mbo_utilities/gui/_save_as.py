"""The Save As dialog: selection, format options and the write it starts, in a thread or a worker process."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import hello_imgui, imgui
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities.arrays import _sanitize_suffix
from mbo_utilities.arrays.features import (
    TAG_REGISTRY,
    DimensionTag,
    parse_timepoint_selection,
)
from mbo_utilities.gui._colormaps import listed_name
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
    source_timepoints,
)
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.gui.widgets.progress_bar import reset_progress_state
from mbo_utilities.metadata import RoiMode
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.reader import MBO_AVAILABLE_FTYPES, imread, widget_reader_kwargs
from mbo_utilities.writer import imwrite


def _get_array_features(widget: Any) -> dict[str, bool]:
    """Which options the dialog offers for the array on screen: scan-phase correction, axial registration, mROI splitting, piezo frame averaging, temporal binning."""
    data = widget.image_widget.data[0]
    return {
        "phase_correction": hasattr(data, "phase_correction"),
        "z_registration": widget.nz > 1 and widget.is_mbo_scan,
        "multi_roi": getattr(data, "num_rois", 1) > 1,
        "frame_averaging": getattr(data, "can_average", False),
        # the viewer may already be binned down to one frame, so the source is counted
        "frame_average": source_timepoints(widget) > 1,
    }


def _display_timepoints(parent: Any) -> int:
    """T of the array on screen, after any binning the viewer applied."""
    data = parent.image_widget.data[0]
    return int(getattr(data, "num_timepoints", None) or data.shape[0])


def _save_as_worker(path, **imwrite_kwargs):
    """The foreground save: every ROI is read and imwrite splits them and applies the read features (``fix_phase``, ``frame_average``, ...)."""
    imwrite(imread(path), **imwrite_kwargs)


def _init_state(parent: Any) -> None:
    """The dialog's own state on the host, created on first draw; the selection attributes are read back by ``draw_selection_table``."""
    if hasattr(parent, "_saveas_sizer"):
        return
    parent._saveas_sizer = PopupAutoSize("Save As")
    parent._saveas_modal_open = True
    parent._saveas_options_open = False
    parent._saveas_options_needs_reset = False
    parent._saveas_last_fpath = None
    parent._saveas_tp_selection = None
    parent._saveas_tp_error = ""
    parent._saveas_tp_parsed = None
    parent._saveas_last_max_tp = None
    parent._saveas_last_num_planes = None
    parent._saveas_last_num_channels = None
    parent._selected_channels = {0}


def draw_saveas_popup(parent: Any):
    """Draw the Save As modal; ``parent._saveas_popup_open`` opens it."""
    _init_state(parent)
    just_opened = False
    if parent._saveas_popup_open:
        parent._saveas_sizer.before_open()
        imgui.open_popup("Save As")
        parent._saveas_popup_open = False
        parent._saveas_modal_open = True
        just_opened = True

    # PopupAutoSize lets the window refit every frame, so the dialog grows when .h5 reveals its field
    opened, visible = imgui.begin_popup_modal(
        "Save As",
        p_open=parent._saveas_modal_open,
        flags=parent._saveas_sizer.flags(imgui.WindowFlags_.no_saved_settings),
    )
    if not opened:
        if not just_opened:
            parent._saveas_modal_open = False
        return
    if not visible:
        parent._saveas_modal_open = False
        imgui.close_current_popup()
        imgui.end_popup()
        return

    parent._saveas_modal_open = True
    if imgui.begin_tab_bar("SaveAsTabBar"):
        if imgui.begin_tab_item("Save")[0]:
            imgui.dummy(imgui.ImVec2(0, 5))
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Output")
            imgui.dummy(imgui.ImVec2(0, 5))

            imgui.set_next_item_width(hello_imgui.em_size(25))
            current_dir_str = parent._saveas_outdir or ""
            changed, new_str = imgui.input_text("Save Dir", current_dir_str)
            if changed:
                parent._saveas_outdir = new_str

            imgui.same_line()
            if not NATIVE_DIALOGS:
                imgui.begin_disabled()
            if imgui.button("Browse"):
                default_dir = parent._saveas_outdir or str(
                    get_last_dir("save_as") or Path.home()
                )
                parent._saveas_folder_dialog = pfd.select_folder(
                    "Select output folder", default_dir
                )
            if not NATIVE_DIALOGS:
                imgui.end_disabled()
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                    imgui.set_tooltip(no_dialog_hint())

            if (
                parent._saveas_folder_dialog is not None
                and parent._saveas_folder_dialog.ready()
            ):
                result = parent._saveas_folder_dialog.result()
                if result:
                    parent._saveas_outdir = str(result)
                    set_last_dir("save_as", result)
                parent._saveas_folder_dialog = None

            imgui.set_next_item_width(hello_imgui.em_size(25))
            # a persisted index can point past the list when the ext's package was uninstalled
            if parent._ext_idx >= len(MBO_AVAILABLE_FTYPES):
                parent._ext_idx = MBO_AVAILABLE_FTYPES.index(".tiff")
            _, parent._ext_idx = imgui.combo(
                "Ext", parent._ext_idx, MBO_AVAILABLE_FTYPES
            )
            parent._ext = MBO_AVAILABLE_FTYPES[parent._ext_idx]

            if parent._ext == ".h5":
                imgui.set_next_item_width(hello_imgui.em_size(25))
                _, parent._h5_dataset_name = imgui.input_text(
                    "H5 dataset name", parent._h5_dataset_name
                )
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Path to the dataset inside the .h5 file. "
                        "'mov' is the H5Array auto-detect default; "
                        "use whatever your downstream tool expects."
                    )

            imgui.spacing()
            imgui.separator()
            _draw_selection_section(parent)
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            _draw_save_button(parent)
            _draw_options_popup(parent)

            imgui.end_tab_item()

        imgui.end_tab_bar()

    imgui.end_popup()


def _draw_options_popup(parent: Any):
    """The Save Options popup, centered on the viewport so the tall video options are not clipped where the button sits."""
    if parent._saveas_options_open:
        imgui.open_popup("Save Options")
        parent._saveas_options_open = False
        parent._saveas_options_needs_reset = True

    is_video = parent._ext == ".mp4"
    viewport = imgui.get_main_viewport()
    default_h = min(720 if is_video else 420, int(viewport.size.y * 0.9))
    center = imgui.ImVec2(
        viewport.pos.x + viewport.size.x * 0.5,
        viewport.pos.y + viewport.size.y * 0.5,
    )
    imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_size(imgui.ImVec2(420, default_h), imgui.Cond_.appearing)
    if imgui.begin_popup("Save Options"):
        if parent._saveas_options_needs_reset:
            parent._saveas_fix_phase = True
            parent._saveas_options_needs_reset = False
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Options")
        imgui.dummy(imgui.ImVec2(0, 5))

        features = _get_array_features(parent)
        _, parent._saveas_background = imgui.checkbox(
            "Run in background", parent._saveas_background
        )
        set_tooltip(
            "Run save operation as a separate process that continues after closing the GUI. "
            "Progress will be logged to a file in the output directory."
        )

        parent._overwrite = checkbox_with_tooltip(
            "Overwrite", parent._overwrite, "Replace any existing output files."
        )

        if not features["z_registration"]:
            imgui.begin_disabled()
        _changed, _reg_value = imgui.checkbox(
            "Register Z-Planes Axially",
            parent._register_z if features["z_registration"] else False,
        )
        if features["z_registration"] and _changed:
            parent._register_z = _reg_value
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            if parent.nz <= 1:
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
        if not features["z_registration"]:
            imgui.end_disabled()

        if features["z_registration"] and parent._register_z:
            imgui.indent(16)
            imgui.set_next_item_width(80)
            _changed, _val = imgui.input_int(
                "Max frames", parent._axial_max_frames, 50, 100
            )
            if _changed:
                parent._axial_max_frames = max(10, _val)
            set_tooltip(
                "Frames subsampled (evenly spaced) for the time-mean used to\n"
                "compute plane shifts. 200 is usually plenty."
            )

            imgui.set_next_item_width(80)
            _changed, _val = imgui.input_int(
                "Max shift (px)", parent._axial_max_reg_xy, 10, 50
            )
            if _changed:
                parent._axial_max_reg_xy = max(1, _val)
            set_tooltip("Max shift search radius in pixels. Default 150.")
            imgui.unindent(16)

        # the save has its own phase settings, apart from the display's
        if features["phase_correction"]:
            fix_phase_changed, fix_phase_value = imgui.checkbox(
                "Fix Scan Phase", parent._saveas_fix_phase
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
                parent._saveas_fix_phase = fix_phase_value

            use_fft_changed, use_fft_value = imgui.checkbox(
                "Subpixel Phase Correction", parent._saveas_use_fft
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
                parent._saveas_use_fft = use_fft_value

        if features["frame_average"]:
            parent._saveas_frame_average = draw_frame_average_input(
                "Frame Average##saveas",
                parent._saveas_frame_average,
                viewer_factor=parent.frame_average,
                max_frames=source_timepoints(parent),
            )

        parent._debug = checkbox_with_tooltip(
            "Debug",
            parent._debug,
            "Print additional information to the terminal during process.",
        )

        imgui.spacing()
        imgui.text("Chunk Size (MB)")
        set_tooltip(
            "The size of the chunk, in MB, to read and write at a time. Larger chunks may be faster but use more memory.",
        )

        imgui.set_next_item_width(hello_imgui.em_size(15))
        _, parent._saveas_chunk_mb = imgui.drag_int(
            "##chunk_size_mb_mb",
            parent._saveas_chunk_mb,
            v_speed=1,
            v_min=1,
            v_max=1024,
        )

        if parent._ext == ".zarr":
            imgui.spacing()
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Zarr Options")
            imgui.dummy(imgui.ImVec2(0, 5))

            _, parent._zarr_sharded = imgui.checkbox("Sharded", parent._zarr_sharded)
            set_tooltip(
                "Use sharding to group multiple chunks into single files (100 frames/shard). "
                "Improves read/write performance for large datasets by reducing filesystem overhead.",
            )

            _, parent._zarr_ome = imgui.checkbox("OME-Zarr", parent._zarr_ome)
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
            _, parent._zarr_compression_level = imgui.slider_int(
                "##zarr_level", parent._zarr_compression_level, 0, 9
            )

            imgui.spacing()
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.8, 1.0, 1.0), "Pyramid (Multi-resolution)"
            )
            imgui.dummy(imgui.ImVec2(0, 3))

            _, parent._zarr_pyramid = imgui.checkbox(
                "Generate Pyramid", parent._zarr_pyramid
            )
            set_tooltip(
                "Write multi-resolution OME-Zarr pyramid.",
            )

            if parent._zarr_pyramid:
                imgui.text("Max Levels")
                set_tooltip(
                    "Levels beyond full-res. Factors set by voxel anisotropy.",
                )
                imgui.set_next_item_width(hello_imgui.em_size(8))
                _, parent._zarr_pyramid_max_layers = imgui.slider_int(
                    "##pyramid_levels", parent._zarr_pyramid_max_layers, 1, 6
                )

                imgui.text("Method")
                set_tooltip(
                    "median/mode = webknossos (intensity/labels). "
                    "mean/nearest/gaussian also available.",
                )
                methods = ["median", "mode", "mean", "nearest", "gaussian"]
                imgui.set_next_item_width(hello_imgui.em_size(10))
                if imgui.begin_combo("##pyramid_method", parent._zarr_pyramid_method):
                    for method in methods:
                        selected = method == parent._zarr_pyramid_method
                        if imgui.selectable(method, selected)[0]:
                            parent._zarr_pyramid_method = method
                        if selected:
                            imgui.set_item_default_focus()
                    imgui.end_combo()

        if parent._ext == ".mp4":
            _draw_video_options(parent)

        imgui.spacing()
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


_VIDEO_CODECS = ["libx264", "mpeg4"]
_VIDEO_QUALITY_PRESETS = ["preview", "high", "visually lossless", "lossless"]
_VIDEO_TEMPORAL_MODES = ["mean", "max", "std"]


def _sync_video_options_from_preview(parent: Any) -> None:
    """Set every video-export field from the preview: fs, the graphic's contrast and colormap, mean subtraction, projection, window and blur."""
    synced: list[str] = []
    graphic = parent.image_widget.graphics[0]

    fs = getattr(parent.image_widget.data[0], "fs", None)
    if fs:
        parent._saveas_video_fps = max(1, int(round(float(fs))))
        synced.append(f"fps={parent._saveas_video_fps}")

    parent._saveas_video_vmin = float(graphic.vmin)
    parent._saveas_video_vmax = float(graphic.vmax)
    parent._saveas_video_auto = False
    synced.append(f"vmin/vmax=[{graphic.vmin:.1f}, {graphic.vmax:.1f}]")

    if graphic.cmap is not None:  # None for an rgb image
        cmap = listed_name(graphic.cmap)
        if cmap not in parent._saveas_video_cmaps:
            parent._saveas_video_cmaps = [cmap, *parent._saveas_video_cmaps]
        parent._saveas_video_cmap_idx = parent._saveas_video_cmaps.index(cmap)
        synced.append(f"cmap={cmap!r}")

    parent._saveas_video_mean_subtract = parent.mean_subtraction
    synced.append(f"mean_subtract={parent.mean_subtraction}")

    if parent.proj in _VIDEO_TEMPORAL_MODES:
        parent._saveas_video_temporal_mode_idx = _VIDEO_TEMPORAL_MODES.index(
            parent.proj
        )
        synced.append(f"temporal_mode={parent.proj!r}")

    if parent.window_size > 0:
        parent._saveas_video_temporal_smooth = parent.window_size
        synced.append(f"window={parent.window_size}")

    parent._saveas_video_spatial_smooth = float(parent.gaussian_sigma)
    synced.append(f"sigma={parent.gaussian_sigma:.2f}")
    parent.logger.info("Synced from preview: " + ", ".join(synced))


def _write_mean_subtract_stack(parent: Any) -> str | None:
    """The viewer's (C, Z, Y, X) mean image in a temp .npy for the video writer; None when off or still computing."""
    if not parent._saveas_video_mean_subtract:
        return None
    mean = parent._mean_image(0)
    if mean is None:
        return None
    fd, path = tempfile.mkstemp(prefix="mbo_meansub_", suffix=".npy")
    os.close(fd)
    np.save(path, mean)
    return path


def _video_kwargs(parent: Any) -> dict:
    """The .mp4 writer's kwargs from the dialog's video options."""
    kwargs = {
        "fps": parent._saveas_video_fps,
        "speed_factor": parent._saveas_video_speed_factor,
        "vmin": None if parent._saveas_video_auto else parent._saveas_video_vmin,
        "vmax": None if parent._saveas_video_auto else parent._saveas_video_vmax,
        "vmin_percentile": parent._saveas_video_vmin_pct,
        "vmax_percentile": parent._saveas_video_vmax_pct,
        "temporal_smooth": parent._saveas_video_temporal_smooth,
        "spatial_smooth": parent._saveas_video_spatial_smooth,
        "gamma": parent._saveas_video_gamma,
        "cmap": parent._saveas_video_cmaps[parent._saveas_video_cmap_idx],
        "quality": _VIDEO_QUALITY_PRESETS[parent._saveas_video_quality_idx],
        "codec": _VIDEO_CODECS[parent._saveas_video_codec_idx],
        "temporal_mode": _VIDEO_TEMPORAL_MODES[parent._saveas_video_temporal_mode_idx],
        "time_overlay": parent._saveas_video_time_overlay,
        "scalebar": parent._saveas_video_scalebar,
        "upscale": parent._saveas_video_upscale or None,
    }
    mean_sub_path = _write_mean_subtract_stack(parent)
    if mean_sub_path:
        kwargs["mean_subtract_path"] = mean_sub_path
    elif parent._saveas_video_mean_subtract:
        parent.logger.warning(
            "Mean subtract on but the mean image is still computing; skipping."
        )
    return kwargs


def _draw_video_options(parent: Any):
    imgui.spacing()
    imgui.separator()
    imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Video Options")
    imgui.same_line()
    if imgui.button("Sync from preview"):
        _sync_video_options_from_preview(parent)
    set_tooltip(
        "Pull current values from the preview widget: fps (metadata.fs), "
        "vmin/vmax (histogram), colormap, mean-subtract toggle, "
        "temporal projection mode, window size, and gaussian sigma."
    )
    imgui.dummy(imgui.ImVec2(0, 5))

    imgui.set_next_item_width(hello_imgui.em_size(10))
    _, parent._saveas_video_fps = imgui.input_int(
        "FPS", parent._saveas_video_fps, step=1, step_fast=10
    )
    parent._saveas_video_fps = max(1, min(int(parent._saveas_video_fps), 240))
    set_tooltip("Base frame rate of the recording. Type a value or use +/-.")

    imgui.set_next_item_width(hello_imgui.em_size(10))
    _, parent._saveas_video_speed_factor = imgui.input_float(
        "Speed",
        parent._saveas_video_speed_factor,
        step=0.1,
        step_fast=1.0,
        format="%.2f",
    )
    parent._saveas_video_speed_factor = max(
        0.1, min(float(parent._saveas_video_speed_factor), 100.0)
    )
    set_tooltip(
        "Playback speed multiplier. Type any value (e.g. 10, 10.5). "
        "Output FPS = FPS * Speed."
    )

    imgui.spacing()
    _, parent._saveas_video_auto = imgui.checkbox(
        "Auto contrast", parent._saveas_video_auto
    )
    set_tooltip("Use percentile-based intensity scaling instead of explicit vmin/vmax.")

    if parent._saveas_video_auto:
        imgui.set_next_item_width(hello_imgui.em_size(10))
        _, parent._saveas_video_vmin_pct = imgui.input_float(
            "vmin %",
            parent._saveas_video_vmin_pct,
            step=0.1,
            step_fast=1.0,
            format="%.1f",
        )
        parent._saveas_video_vmin_pct = max(
            0.0, min(parent._saveas_video_vmin_pct, 10.0)
        )
        set_tooltip("Percentile for auto vmin. Lower = darker blacks.")

        imgui.set_next_item_width(hello_imgui.em_size(10))
        _, parent._saveas_video_vmax_pct = imgui.input_float(
            "vmax %",
            parent._saveas_video_vmax_pct,
            step=0.1,
            step_fast=1.0,
            format="%.1f",
        )
        parent._saveas_video_vmax_pct = max(
            90.0, min(parent._saveas_video_vmax_pct, 100.0)
        )
        set_tooltip("Percentile for auto vmax. Lower = brighter highlights.")
    else:
        imgui.set_next_item_width(hello_imgui.em_size(12))
        _, parent._saveas_video_vmin = imgui.input_float(
            "vmin",
            parent._saveas_video_vmin,
            step=10.0,
            step_fast=100.0,
            format="%.1f",
        )
        set_tooltip("Min intensity value (clipped to black).")

        imgui.set_next_item_width(hello_imgui.em_size(12))
        _, parent._saveas_video_vmax = imgui.input_float(
            "vmax",
            parent._saveas_video_vmax,
            step=10.0,
            step_fast=100.0,
            format="%.1f",
        )
        set_tooltip("Max intensity value (clipped to white).")

    imgui.spacing()
    changed, parent._saveas_video_mean_subtract = imgui.checkbox(
        "Mean subtract", parent._saveas_video_mean_subtract
    )
    if changed and parent._saveas_video_mean_subtract:
        parent._mean_image(0)
    set_tooltip(
        "Subtract each pixel's mean over the recording from every frame before "
        "contrast scaling (the viewer's Mean Subtraction image)."
    )

    _, parent._saveas_video_time_overlay = imgui.checkbox(
        "Time overlay", parent._saveas_video_time_overlay
    )
    set_tooltip(
        "Draw a clock in the top-left of every frame showing recording time "
        "(frame_index / fps). The clock always reflects real recording seconds, "
        "so a sped-up clip ticks faster on screen — useful for showing how fast "
        "playback is relative to the source."
    )

    _, parent._saveas_video_scalebar = imgui.checkbox(
        "Scalebar", parent._saveas_video_scalebar
    )
    set_tooltip(
        "Draw a scalebar at exactly 10% of frame width in the bottom-left, "
        "labeled with the matching micron length. Multiply the label by 10 "
        "to read the full-frame width. Pixel size is taken from arr.dx."
    )

    imgui.set_next_item_width(hello_imgui.em_size(8))
    _, parent._saveas_video_upscale = imgui.input_int(
        "Upscale", parent._saveas_video_upscale
    )
    parent._saveas_video_upscale = max(0, parent._saveas_video_upscale)
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
    _, parent._saveas_video_temporal_mode_idx = imgui.combo(
        "Temporal mode",
        parent._saveas_video_temporal_mode_idx,
        _VIDEO_TEMPORAL_MODES,
    )
    set_tooltip(
        "Rolling-window aggregation: mean smooths flicker, max emphasizes "
        "transient activity, std highlights variance."
    )

    imgui.set_next_item_width(hello_imgui.em_size(10))
    _, parent._saveas_video_temporal_smooth = imgui.input_int(
        "Window", parent._saveas_video_temporal_smooth, step=1, step_fast=10
    )
    parent._saveas_video_temporal_smooth = max(
        0, min(int(parent._saveas_video_temporal_smooth), 500)
    )
    set_tooltip("Rolling-window size in frames. 0 = off; combined with Temporal mode.")

    imgui.set_next_item_width(hello_imgui.em_size(10))
    _, parent._saveas_video_spatial_smooth = imgui.input_float(
        "Spatial smooth",
        parent._saveas_video_spatial_smooth,
        step=0.1,
        step_fast=1.0,
        format="%.2f",
    )
    parent._saveas_video_spatial_smooth = max(
        0.0, min(float(parent._saveas_video_spatial_smooth), 20.0)
    )
    set_tooltip("Gaussian blur sigma in pixels. 0 = off.")

    imgui.set_next_item_width(hello_imgui.em_size(10))
    _, parent._saveas_video_gamma = imgui.input_float(
        "Gamma",
        parent._saveas_video_gamma,
        step=0.05,
        step_fast=0.5,
        format="%.2f",
    )
    parent._saveas_video_gamma = max(0.1, min(float(parent._saveas_video_gamma), 5.0))
    set_tooltip("Gamma correction. <1 brightens midtones, >1 darkens them.")

    imgui.spacing()
    imgui.set_next_item_width(hello_imgui.em_size(14))
    _, parent._saveas_video_cmap_idx = imgui.combo(
        "Colormap", parent._saveas_video_cmap_idx, parent._saveas_video_cmaps
    )
    set_tooltip(
        "Colormap (cmap-library / fastplotlib names). 'gray' writes grayscale. "
        "Click 'Sync from preview' to adopt the active fpl cmap if it's not in this list."
    )

    imgui.set_next_item_width(hello_imgui.em_size(14))
    _, parent._saveas_video_quality_idx = imgui.combo(
        "Quality", parent._saveas_video_quality_idx, _VIDEO_QUALITY_PRESETS
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
    _, parent._saveas_video_codec_idx = imgui.combo(
        "Codec", parent._saveas_video_codec_idx, _VIDEO_CODECS
    )
    set_tooltip("Video codec. libx264 is best for browser playback.")


def _draw_selection_section(parent: Any):
    """The T, Z and C selection rows, and a preview of the filename, size and shape they produce."""
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

    data = parent.image_widget.data[0]
    max_frames = _display_timepoints(parent)
    num_planes = parent.nz
    num_channels = parent.nc

    # a new file, or a changed extent, puts every selection back to the full range
    current_fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
    current_fpath = str(current_fpath) if current_fpath else ""
    file_changed = parent._saveas_last_fpath != current_fpath
    parent._saveas_last_fpath = current_fpath
    if file_changed or parent._saveas_last_max_tp != max_frames:
        parent._saveas_tp_selection = f"1:{max_frames}"
        parent._saveas_tp_error = ""
        parent._saveas_tp_parsed = None
        parent._saveas_last_max_tp = max_frames
    if file_changed or parent._saveas_last_num_planes != num_planes:
        parent._saveas_last_num_planes = num_planes
    if file_changed or parent._saveas_last_num_channels != num_channels:
        parent._saveas_last_num_channels = num_channels

    # row labels follow the array's slider names (isoview Tile / Cam / Zplane); the indices do not
    tp_label, z_label, c_label = resolve_dim_labels(parent)
    tp_parsed, z_start, z_stop, z_step, c_start, c_stop, c_step = draw_selection_table(
        parent,
        max_frames,
        num_planes,
        tp_attr="_saveas_tp",
        z_attr="_saveas_z",
        id_suffix="_saveas",
        suffix_attr="_saveas_output_suffix",
        num_channels=num_channels,
        c_attr="_saveas_c",
        tp_label=tp_label,
        z_label=z_label,
        c_label=c_label,
    )

    if num_planes > 1:
        parent._selected_planes = set(range(z_start - 1, z_stop, z_step))
    else:
        parent._selected_planes = {0}
    if num_channels > 1:
        parent._selected_channels = set(range(c_start - 1, c_stop, c_step))
    else:
        parent._selected_channels = {0}

    if parent._saveas_tp_parsed is None and not parent._saveas_tp_error:
        try:
            parent._saveas_tp_parsed = parse_timepoint_selection(
                parent._saveas_tp_selection, max_frames
            )
        except ValueError as e:
            parent._saveas_tp_error = str(e)

    imgui.spacing()

    ext = parent._ext.lstrip(".")
    is_video = parent._ext == ".mp4"
    tp_parsed = parent._saveas_tp_parsed
    t_tag = None
    n_frames = 0
    if tp_parsed:
        n_frames = tp_parsed.count
        final_indices = tp_parsed.final_indices
        if final_indices:
            tp_step = (
                final_indices[1] - final_indices[0] if len(final_indices) > 1 else 1
            )
            t_tag = _range_tag(
                "T", final_indices[0] + 1, final_indices[-1] + 1, tp_step
            )

    if is_video:
        # one file per (z, channel); the preview names the first, in zplane_ch_tp order
        tags = [
            _range_tag("Z", z_start, z_start, 1),
            _range_tag("C", c_start, c_start, 1),
        ]
        if t_tag is not None:
            tags.append(t_tag)
    else:
        tags = [t_tag] if t_tag is not None else []
        if num_channels > 1:
            tags.append(_range_tag("C", c_start, c_stop, c_step))
        if num_planes > 1:
            tags.append(_range_tag("Z", z_start, z_stop, z_step))

    suffix = parent._saveas_output_suffix
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

    n_planes_out = len(parent._selected_planes)
    n_channels_out = len(parent._selected_channels)
    Ly, Lx = data.shape[-2:]
    dtype_size = np.dtype(data.dtype).itemsize

    # the raw size; a codec or compressor only brings it down
    if is_video:
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

    table_flags = (
        imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
    )
    if imgui.begin_table("output_preview", 2, table_flags):
        imgui.table_setup_column(
            "label", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(6)
        )
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)

        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Filename")
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.6, 0.9, 0.6, 1.0), filename)
        outdir = parent._saveas_outdir
        if imgui.is_item_hovered() and outdir:
            imgui.begin_tooltip()
            display_path = str(Path(outdir)).replace("\\", "/")
            imgui.text_unformatted(f"{display_path}/{filename}")
            imgui.end_tooltip()

        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Size")
        imgui.table_next_column()
        imgui.text(size_str)

        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Shape")
        imgui.table_next_column()
        imgui.text(f"{shape_str} {dims_str}")

        imgui.end_table()


def _range_tag(dim: str, start: int, stop: int, step: int) -> DimensionTag:
    """The 1-based filename tag for ``start:stop:step`` of ``dim``; a single index has no stop."""
    if start == stop:
        return DimensionTag(TAG_REGISTRY[dim], start=start, stop=None, step=1)
    return DimensionTag(TAG_REGISTRY[dim], start=start, stop=stop, step=step)


def _draw_save_button(parent: Any):
    """The Save, Options and Cancel buttons; Save starts the write and closes the dialog."""
    no_planes = (
        parent._selected_planes is not None and len(parent._selected_planes) == 0
    )
    no_valid_tp = parent._saveas_tp_error or parent._saveas_tp_parsed is None
    num_channels = parent._saveas_last_num_channels
    averaged = parent._saveas_frame_average
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
        if not parent._saveas_outdir:
            last_dir = get_last_dir("save_as") or Path().home()
            parent._saveas_outdir = str(last_dir)
        try:
            save_planes = [p + 1 for p in parent._selected_planes]
            save_channels = [c + 1 for c in parent._selected_channels]
            if not save_planes:
                parent.logger.error(
                    "No z-planes selected! Please select at least one plane."
                )
            else:
                rois = None
                if parent._saveas_rois:
                    if not parent._saveas_selected_roi:
                        mroi_count = getattr(parent.image_widget.data[0], "num_rois", 1)
                        parent._saveas_selected_roi = set(range(mroi_count))
                    rois = sorted(r + 1 for r in parent._saveas_selected_roi)

                Path(parent._saveas_outdir).expanduser().mkdir(
                    parents=True, exist_ok=True
                )

                tp_parsed = parent._saveas_tp_parsed
                max_timepoints = _display_timepoints(parent)
                final_indices_0 = (
                    tp_parsed.final_indices
                    if tp_parsed
                    else list(range(max_timepoints))
                )
                if final_indices_0 == list(range(max_timepoints)):
                    frames = None  # every frame
                else:
                    frames = [i + 1 for i in final_indices_0]

                metadata_overrides = dict(parent._custom_metadata)
                if tp_parsed:
                    metadata_overrides["timepoint_selection"] = tp_parsed.to_metadata()
                if len(save_channels) < num_channels:
                    metadata_overrides["num_color_channels"] = len(save_channels)
                    metadata_overrides["channel_selection"] = save_channels

                # the suffix names a stitched output; split ROIs are named by their roiNN
                output_suffix = parent._saveas_output_suffix if rois is None else None
                roi_mode = RoiMode.separate if rois else RoiMode.concat_y

                save_kwargs = {
                    "path": parent.fpath,
                    "outpath": parent._saveas_outdir,
                    "planes": save_planes,
                    "channels": save_channels
                    if len(save_channels) < num_channels
                    else None,
                    "frames": frames,
                    "roi": rois,
                    "roi_mode": roi_mode,
                    "overwrite": parent._overwrite,
                    "debug": parent._debug,
                    "ext": parent._ext,
                    "target_chunk_mb": parent._saveas_chunk_mb,
                    "fix_phase": parent._saveas_fix_phase,
                    "use_fft": parent._saveas_use_fft,
                    "border": parent.border,
                    "max_offset": parent.max_offset,
                    "frame_average": parent._saveas_frame_average,
                    "register_z": parent._register_z,
                    "max_frames": parent._axial_max_frames,
                    "max_reg_xy": parent._axial_max_reg_xy,
                    "mean_subtraction": parent.mean_subtraction,
                    "progress_callback": parent.gui_progress_callback,
                    "metadata": metadata_overrides if metadata_overrides else None,
                    "output_suffix": output_suffix,
                }
                if parent._ext == ".zarr":
                    save_kwargs["sharded"] = parent._zarr_sharded
                    save_kwargs["ome"] = parent._zarr_ome
                    save_kwargs["level"] = parent._zarr_compression_level
                    save_kwargs["pyramid"] = parent._zarr_pyramid
                    if parent._zarr_pyramid:
                        save_kwargs["pyramid_max_layers"] = (
                            parent._zarr_pyramid_max_layers
                        )
                        save_kwargs["pyramid_method"] = parent._zarr_pyramid_method

                if parent._ext == ".h5":
                    save_kwargs["dataset_name"] = parent._h5_dataset_name
                video_kwargs = _video_kwargs(parent) if parent._ext == ".mp4" else {}
                save_kwargs.update(video_kwargs)

                n_frames = len(frames) if frames else max_timepoints
                if tp_parsed and tp_parsed.exclude_str:
                    frames_msg = f"{n_frames} frames ({tp_parsed.include_str} excl. {tp_parsed.exclude_str})"
                elif tp_parsed:
                    frames_msg = f"{n_frames} frames ({tp_parsed.include_str})"
                else:
                    frames_msg = "all frames"
                num_rois = getattr(parent.image_widget.data[0], "num_rois", 1)
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
                    f", {parent._saveas_frame_average} frames averaged"
                    if parent._saveas_frame_average > 1
                    else ""
                )
                parent.logger.info(
                    f"Saving planes {save_planes}{channels_msg} ({frames_msg}){roi_msg}{avg_msg}"
                )
                parent.logger.info(
                    f"Saving to {parent._saveas_outdir} as {parent._ext}"
                )

                if parent._saveas_background:
                    pm = get_process_manager()
                    src = Path(parent.fpath) if parent.fpath else None
                    input_path = str(src) if src else ""
                    fname = (src.name or "data") if src else "data"
                    worker_args = {
                        "input_path": input_path,
                        # a .mesc worker re-opens the unit on screen, not unit 0
                        "reader_kwargs": widget_reader_kwargs(parent.image_widget),
                        "output_path": str(parent._saveas_outdir),
                        "ext": parent._ext,
                        "planes": save_planes,
                        "channels": save_channels
                        if len(save_channels) < num_channels
                        else None,
                        "frames": frames,
                        "rois": rois,
                        "fix_phase": parent._saveas_fix_phase,
                        "use_fft": parent._saveas_use_fft,
                        "frame_average": parent._saveas_frame_average,
                        "register_z": parent._register_z,
                        "max_frames": parent._axial_max_frames,
                        "max_reg_xy": parent._axial_max_reg_xy,
                        "metadata": metadata_overrides if metadata_overrides else {},
                        "kwargs": {
                            "sharded": parent._zarr_sharded
                            if parent._ext == ".zarr"
                            else False,
                            "ome": parent._zarr_ome
                            if parent._ext == ".zarr"
                            else False,
                            "output_suffix": output_suffix,
                            "pyramid": parent._zarr_pyramid
                            if parent._ext == ".zarr"
                            else False,
                            "pyramid_max_layers": parent._zarr_pyramid_max_layers
                            if parent._ext == ".zarr" and parent._zarr_pyramid
                            else 4,
                            "pyramid_method": parent._zarr_pyramid_method
                            if parent._ext == ".zarr" and parent._zarr_pyramid
                            else "median",
                            **(
                                {"dataset_name": parent._h5_dataset_name}
                                if parent._ext == ".h5"
                                else {}
                            ),
                            **video_kwargs,
                        },
                    }
                    pid = pm.spawn(
                        task_type="save_as",
                        args=worker_args,
                        description=f"Saving {fname} to {parent._ext}",
                        output_path=str(parent._saveas_outdir),
                    )
                    if pid:
                        parent.logger.info(
                            f"Started background save process (PID {pid})"
                        )
                        parent.logger.info(
                            "You can close the GUI - the save will continue."
                        )
                    else:
                        parent.logger.error("Failed to start background process")
                else:
                    reset_progress_state("saveas")
                    parent._saveas_progress = 0.0
                    parent._saveas_done = False
                    parent._saveas_running = True
                    parent.logger.info("Starting save operation...")
                    if parent._register_z:
                        reset_progress_state("register_z")
                        parent._register_z_progress = 0.0
                        parent._register_z_done = False
                        parent._register_z_running = True
                        parent._register_z_current_msg = "Starting..."
                    threading.Thread(
                        target=_save_as_worker, kwargs=save_kwargs, daemon=True
                    ).start()
            parent._saveas_modal_open = False
            imgui.close_current_popup()
        except Exception as e:
            parent.logger.exception(f"Error saving data: {e}")
            parent._saveas_modal_open = False
            imgui.close_current_popup()

    imgui.same_line()
    if imgui.button("Options", imgui.ImVec2(80, 0)):
        parent._saveas_options_open = True

    imgui.same_line()
    if imgui.button("Cancel"):
        parent._saveas_modal_open = False
        imgui.close_current_popup()
