"""
Pollen calibration viewer for LBM beamlet calibration.

This viewer handles pollen calibration data (ZCYX) and provides:
- Info panel showing beamlet count, cavities, z-step, pixel size
- Automatic background calibration (detection + analysis)
- Manual calibration mode: click through beamlets in the viewer
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from imgui_bundle import imgui, implot
from scipy.ndimage import uniform_filter1d

from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.metadata import get_param
from mbo_utilities.metadata.scanimage import (
    get_saved_channel_ports,
    get_z_step_size,
)

from . import BaseViewer

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["PollenCalibrationViewer"]


def get_cavity_indices(metadata: dict, nc: int) -> dict:
    """Get cavity A and cavity B channel indices from LBM metadata."""
    from mbo_utilities.metadata.scanimage import is_lbm_stack

    result = {
        "cavity_a": [],
        "cavity_b": [],
        "is_lbm": False,
        "num_cavities": 1,
    }

    if not is_lbm_stack(metadata):
        half = nc // 2
        result["cavity_a"] = list(range(half))
        result["cavity_b"] = list(range(half, nc))
        return result

    result["is_lbm"] = True
    ai_sources = get_saved_channel_ports(metadata)

    if not ai_sources:
        half = nc // 2
        result["cavity_a"] = list(range(half))
        result["cavity_b"] = list(range(half, nc))
        return result

    sorted_sources = sorted(ai_sources.keys())

    if len(sorted_sources) >= 1:
        cavity_a_channels = ai_sources.get(sorted_sources[0], [])
        result["cavity_a"] = sorted(
            [ch - 1 if ch > 0 else ch for ch in cavity_a_channels]
        )

    if len(sorted_sources) >= 2:
        cavity_b_channels = ai_sources.get(sorted_sources[1], [])
        result["cavity_b"] = sorted(
            [ch - 1 if ch > 0 else ch for ch in cavity_b_channels]
        )
        result["num_cavities"] = 2

    return result


class PollenCalibrationViewer(BaseViewer):
    """
    Viewer for pollen calibration data (ZCYX).

    This viewer is specialized for LBM beamlet calibration using
    pollen grain stacks. It provides:
    - Automatic bead detection and calibration
    - Interactive manual calibration mode
    - Results visualization and comparison
    """

    name = "Pollen Calibration Viewer"

    DEFAULT_ORDER_30 = [  # the 30-beamlet cavity's beam order
        0,
        4,
        5,
        6,
        7,
        8,
        1,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        2,
        16,
        17,
        18,
        19,
        20,
        21,
        3,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
    ]

    def __init__(
        self,
        image_widget: ImageWidget,
        fpath: str | list[str],
        parent=None,
        **kwargs,
    ):
        super().__init__(image_widget, fpath, parent=parent, **kwargs)

        self._z_step_um = 1.0
        self._pixel_size_um = 1.0
        self._cavity_info = None
        self._beam_order = None

        self._status = "Initializing..."
        self._progress = 0.0
        self._processing = False
        self._done = False
        self._error = None
        self._initialized = False

        self._results_auto = None
        self._results_manual = None
        self._saved_images: list[Path] = []
        self._show_figures_popup = False
        self._figures_popup_mode = "auto"
        self._calibration_data_auto = None
        self._calibration_data_manual = None

        self._manual_mode = False
        self._manual_channel_idx = 0  # index into the beam order
        self._manual_positions = []  # clicked (x, y) per beamlet
        self._manual_z_indices = []  # best z per clicked position
        self._click_handler = None
        self._subplot = None
        self._vol = None  # the volume in memory while manual mode runs
        self._max_projections = (
            None  # (C, Y, X) shown in place of the volume during manual mode
        )
        self._original_metadata = None
        self._original_data_array = None

        self._loaded_external = None
        self._existing_h5_files = []

        self._pointer_down_pos = (
            None  # (x, y) at pointer_down; a move past _drag_threshold is a pan
        )
        self._drag_threshold = 5.0

    @property
    def data(self):
        if self.parent is not None:
            return self.parent.image_widget.data if self.parent.image_widget else None
        return self.image_widget.data if self.image_widget else None

    @property
    def logger(self):
        if self.parent is not None:
            return self.parent.logger
        import logging

        return logging.getLogger("mbo_utilities")

    def _init_from_data(self):
        """Z step, pixel size, cavity info and beam order from the loaded array's metadata."""
        arr = self._get_array()
        if arr is None:
            return
        metadata = arr.metadata

        z_step = get_z_step_size(metadata)
        if z_step is not None:
            self._z_step_um = z_step
        else:
            self._z_step_um = get_param(metadata, "dz", default=1.0)

        pixel_res = get_param(metadata, "pixel_resolution", default=None)
        if pixel_res is not None:
            self._pixel_size_um = float(np.ravel(pixel_res)[0])
        else:
            fov_um = get_param(metadata, "fov_um", default=None)
            if fov_um is not None and arr.ndim >= 2:
                nx = arr.shape[-1]
                self._pixel_size_um = float(np.ravel(fov_um)[0]) / nx
            elif arr.ndim >= 2:
                # a 600 um field at zoom 1 is the last resort
                zoom = get_param(metadata, "zoom_factor", default=1.0)
                nx = arr.shape[-1]
                self._pixel_size_um = 600.0 / zoom / nx
                self.logger.warning(
                    "pixel_resolution not in metadata, using default FOV=600um"
                )

        nc = getattr(arr, "num_channels", arr.shape[1] if arr.ndim >= 2 else 1)
        self._cavity_info = get_cavity_indices(metadata, nc)

        if nc == 30:
            self._beam_order = self.DEFAULT_ORDER_30.copy()
        else:
            self._beam_order = list(range(nc))

        self._initialized = True

    def _get_array(self):
        data = self.data
        return data[0] if data else None

    def _get_fpath(self):
        parent_fpath = self.parent.fpath if self.parent is not None else self.fpath
        if isinstance(parent_fpath, (list, tuple)):
            parent_fpath = parent_fpath[0] if parent_fpath else None
        return Path(parent_fpath) if parent_fpath else None

    @property
    def num_beamlets(self) -> int:
        """The beamlet count: the array's ``num_beamlets``, else C of a (Z, C, Y, X) volume."""
        arr = self._get_array()
        if arr is None:
            return 0
        if hasattr(arr, "num_beamlets"):
            return arr.num_beamlets
        if hasattr(arr, "num_channels"):
            return arr.num_channels
        if arr.ndim == 4:
            return arr.shape[1]
        return 1

    def draw(self) -> None:
        self._draw_calibration_ui()
        if self._show_figures_popup:
            self._draw_figures_popup()

    def _draw_calibration_ui(self) -> None:
        if not self._initialized:
            self._init_from_data()
            if self._initialized and not self._processing and not self._done:
                self._start_auto_calibration()

        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))

        imgui.dummy(imgui.ImVec2(0, 5))
        self._draw_info_section()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if self._manual_mode:
            self._draw_manual_mode()
        else:
            self._draw_load_previous_section()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            self._draw_auto_status()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            self._draw_manual_button()

        imgui.pop_style_var()
        imgui.pop_style_var()

    def _draw_info_section(self):
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Pollen Calibration")
        imgui.spacing()

        arr = self._get_array()
        if arr is not None:
            imgui.text(f"Shape: {arr.shape}")
            set_tooltip("(Z-piezo, Channels, Y, X)", show_mark=False)
            imgui.text(f"Z-step: {self._z_step_um:.2f} um")
            set_tooltip(
                "Piezo z-step size between each slice (from stackZStepSize)",
                show_mark=False,
            )
            imgui.text(f"Pixel: {self._pixel_size_um:.3f} um")
            set_tooltip(
                "Pixel size in microns (from metadata pixel_resolution)",
                show_mark=False,
            )
        else:
            imgui.text_disabled("No data loaded")

    def _draw_auto_status(self):
        if self._processing:
            imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.8, 1.0), "Processing...")
            imgui.text(self._status)
            imgui.progress_bar(self._progress, imgui.ImVec2(-1, 0))
            imgui.spacing()
        elif self._error:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {self._error}"
            )
            imgui.spacing()

        has_auto = self._results_auto is not None
        has_manual = self._results_manual is not None
        if has_auto or has_manual:
            imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.9, 1.0), "Results")
            imgui.spacing()
            if imgui.button("Open Folder"):
                self._open_output_folder("manual" if has_manual else "auto")
            imgui.same_line()
            if imgui.button("View Figures"):
                self._open_all_graphs("auto" if has_auto else "manual")

        elif not self._processing and not self._error:
            imgui.text_disabled("Waiting...")

    def _draw_load_previous_section(self):
        """Load the newest ``*_pollen.h5`` beside the data once, and name it."""
        if not self._existing_h5_files:
            self._scan_existing_h5_files()
        if self._existing_h5_files and not self._loaded_external:
            self._load_h5_file(str(self._existing_h5_files[0]))
        if self._loaded_external:
            mode = self._loaded_external.get("calibration_mode", "previous")
            imgui.text_colored(imgui.ImVec4(0.6, 0.8, 0.6, 1.0), f"Loaded: {mode}")
            h5_path = self._loaded_external.get("h5_path", "")
            if h5_path:
                name = Path(h5_path).name
                imgui.text_disabled(name[:30] + "..." if len(name) > 30 else name)

    def _scan_existing_h5_files(self):
        """The ``*_pollen.h5`` files beside the data, newest first."""
        self._existing_h5_files = []
        fpath = self._get_fpath()
        if fpath is None or not fpath.parent.exists():
            return
        self._existing_h5_files = sorted(
            fpath.parent.glob("*_pollen.h5"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def _load_h5_file(self, path: str):
        from pollen._pollen_analysis import extract_calibration_summary

        summary = extract_calibration_summary(path)
        if summary:
            self._loaded_external = summary
            self._loaded_external["h5_path"] = path
            self.logger.info(f"Loaded calibration from: {path}")
        else:
            self.logger.error(f"Failed to load calibration from: {path}")

    def _draw_manual_button(self):
        avail_w = imgui.get_content_region_avail().x
        text = "Interactive Calibration"
        text_w = imgui.calc_text_size(text).x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - text_w - 20) / 2)
        imgui.text_colored(imgui.ImVec4(0.8, 0.6, 0.2, 1.0), text)
        set_tooltip(
            "Manual pollen calibration mode.\n\n"
            "Click on the same pollen bead in each beamlet.\n"
            "The viewer will show one beamlet at a time.\n\n"
            "Navigation:\n"
            "  - Drag to pan the view\n"
            "  - Scroll wheel to zoom\n"
            "  - Single click (no drag) to mark bead position"
        )

        imgui.spacing()

        btn_w = 180
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - btn_w) / 2)
        if imgui.button("Start", imgui.ImVec2(btn_w, 0)):
            self._start_manual_mode()

    def _draw_manual_mode(self):
        imgui.text_colored(imgui.ImVec4(0.2, 0.8, 0.4, 1.0), "Manual Calibration")
        imgui.spacing()

        nc = (
            self._max_projections.shape[0]
            if self._max_projections is not None
            else self.num_beamlets
        )
        current = self._manual_channel_idx
        channel = (
            self._beam_order[current] if current < len(self._beam_order) else current
        )
        num_marked = len(self._manual_positions)

        imgui.text(f"Beamlet {current + 1}/{nc} (ch {channel})")
        imgui.text(f"Marked: {num_marked}")
        progress = num_marked / nc if nc > 0 else 0
        imgui.progress_bar(progress, imgui.ImVec2(-1, 0), f"{num_marked}/{nc}")
        imgui.spacing()

        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Click on pollen bead")
        imgui.spacing()

        imgui.button("Prev") and self._manual_prev()
        imgui.same_line()
        imgui.button("Skip") and self._manual_skip()
        imgui.same_line()
        imgui.button("Next") and self._manual_next()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if num_marked > 0:
            if num_marked >= nc:
                imgui.text_colored(imgui.ImVec4(0.3, 1.0, 0.3, 1.0), "All done!")
            imgui.button("Finish") and self._finish_manual_calibration()
            imgui.spacing()

        imgui.button("Cancel") and self._cancel_manual_mode()

    def _start_manual_mode(self):
        """Load the volume, show its per-channel max projections in the viewer and arm the click handler."""
        arr = self._get_array()
        if arr is None:
            self.logger.error("No data for manual calibration")
            return

        self._manual_mode = True
        self._manual_channel_idx = 0
        self._manual_positions = []
        self._manual_z_indices = []
        self._original_data_array = arr
        self._original_metadata = arr.metadata

        self.logger.info("Loading volume for manual calibration...")
        self._vol = np.asarray(arr[:]).astype(np.float32)  # (Z piezo, C beamlets, Y, X)
        self._vol -= self._vol.mean()
        nz, nc, _ny, _nx = self._vol.shape
        self.logger.info(f"Volume shape: {self._vol.shape} (Z, C, Y, X)")
        self._max_projections = self._vol.max(axis=0)  # (C, Y, X)
        self.image_widget.data[0] = self._max_projections
        if self.image_widget.n_sliders > 0:
            self.image_widget.indices = [0]
        self._show_beamlet(0)
        self._setup_click_handler()
        self.logger.info("Manual calibration started. Click on pollen beads.")

    def _setup_click_handler(self):
        if self.image_widget is None or self._click_handler is not None:
            return
        self._subplot = self.image_widget.figure[0, 0]
        self._subplot.renderer.add_event_handler(self._on_pointer_down, "pointer_down")
        self._subplot.renderer.add_event_handler(self._on_click, "click")
        self._click_handler = self._on_click

    def _on_pointer_down(self, ev):
        if self._manual_mode:
            self._pointer_down_pos = (ev.x, ev.y)

    def _on_click(self, ev):
        """Mark the bead under a click; a pointer that moved past the drag threshold was a pan."""
        if not self._manual_mode:
            return
        down = self._pointer_down_pos
        self._pointer_down_pos = None
        if down is not None and (
            abs(ev.x - down[0]) > self._drag_threshold
            or abs(ev.y - down[1]) > self._drag_threshold
        ):
            return
        world_pos = self._subplot.map_screen_to_world((ev.x, ev.y))
        if world_pos is not None:
            self._handle_click(world_pos[0], world_pos[1])

    def _handle_click(self, x, y):
        """Record the clicked bead position and its brightest z for the current beamlet, then advance."""
        if not self._manual_mode or self._vol is None:
            return

        _nz, nc, ny, nx = self._vol.shape
        current = self._manual_channel_idx
        if current >= nc:
            self.logger.info("All beamlets already marked. Click 'Finish'.")
            return

        channel = (
            self._beam_order[current] if current < len(self._beam_order) else current
        )

        x = max(0, min(nx - 1, x))
        y = max(0, min(ny - 1, y))
        self.logger.info(f"Beamlet {current + 1}: clicked at ({x:.1f}, {y:.1f})")

        ix, iy = round(x), round(y)
        patch_size = 10
        y0 = max(0, iy - patch_size)
        y1 = min(ny, iy + patch_size + 1)
        x0 = max(0, ix - patch_size)
        x1 = min(nx, ix + patch_size + 1)

        patch = self._vol[:, channel, y0:y1, x0:x1]
        smoothed = uniform_filter1d(patch.max(axis=(1, 2)), size=3, mode="nearest")
        best_z = int(np.argmax(smoothed))

        if current < len(self._manual_positions):
            self._manual_positions[current] = (x, y)
            self._manual_z_indices[current] = best_z
        else:
            self._manual_positions.append((x, y))
            self._manual_z_indices.append(best_z)

        if current < nc - 1:
            self._manual_channel_idx = current + 1
            self._show_beamlet(self._manual_channel_idx)
        else:
            self.logger.info("All beamlets marked! Starting calibration...")
            self._finish_manual_calibration()

    def _show_beamlet(self, idx):
        """Put beamlet ``idx`` (in beam order) of the max projections on the viewer's one slider."""
        if self._max_projections is None or self.image_widget is None:
            return
        nc = self._max_projections.shape[0]
        channel = self._beam_order[idx] if idx < len(self._beam_order) else idx
        if channel >= nc:
            self.logger.error(f"Channel {channel} out of range ({nc} channels)")
            return
        if self.image_widget.n_sliders > 0:
            self.image_widget.indices = [channel]

    def _manual_prev(self):
        if self._manual_channel_idx > 0:
            self._manual_channel_idx -= 1
            self._show_beamlet(self._manual_channel_idx)

    def _manual_next(self):
        nc = self.num_beamlets
        if self._manual_channel_idx < nc - 1:
            self._manual_channel_idx += 1
            self._show_beamlet(self._manual_channel_idx)

    def _manual_skip(self):
        """Mark the current beamlet at the frame's centre and move on."""
        if self._vol is None:
            return

        _nz, _nc, ny, nx = self._vol.shape
        current = self._manual_channel_idx
        channel = (
            self._beam_order[current] if current < len(self._beam_order) else current
        )
        x, y = nx / 2, ny / 2
        patch_size = 10
        iy, ix = int(ny / 2), int(nx / 2)
        y0 = max(0, iy - patch_size)
        y1 = min(ny, iy + patch_size + 1)
        x0 = max(0, ix - patch_size)
        x1 = min(nx, ix + patch_size + 1)

        patch = self._vol[:, channel, y0:y1, x0:x1]
        smoothed = uniform_filter1d(patch.max(axis=(1, 2)), size=3, mode="nearest")
        best_z = int(np.argmax(smoothed))

        if current < len(self._manual_positions):
            self._manual_positions[current] = (x, y)
            self._manual_z_indices[current] = best_z
        else:
            self._manual_positions.append((x, y))
            self._manual_z_indices.append(best_z)

        self.logger.info(f"Beamlet {current + 1}: skipped (using center)")
        self._manual_next()

    def _cancel_manual_mode(self):
        self._manual_mode = False
        self._manual_positions = []
        self._manual_z_indices = []
        self._vol = None
        self._max_projections = None
        self._restore_original_view()
        self._original_metadata = None
        self._original_data_array = None
        self.logger.info("Manual calibration cancelled")

    def _restore_original_view(self):
        """Put the lazy array back in place of the max projections, sliders at 0."""
        if self.image_widget is None:
            return
        arr = (
            self._original_data_array
            if self._original_data_array is not None
            else self._get_array()
        )
        if arr is None:
            return
        self.image_widget.data[0] = arr
        if self.image_widget.n_sliders > 0:
            self.image_widget.indices = [0] * self.image_widget.n_sliders
        self.image_widget.figure[0, 0].auto_scale()
        self.logger.info("Restored original data view")

    def _finish_manual_calibration(self):
        """Run the calibration on the marked positions in a background thread."""
        self._manual_mode = False

        positions = self._manual_positions
        z_indices = self._manual_z_indices

        if len(positions) < self.num_beamlets:
            self.logger.warning(
                f"Only {len(positions)} positions marked, expected {self.num_beamlets}"
            )

        self.logger.info(
            f"Running calibration with {len(positions)} marked positions..."
        )
        self._restore_original_view()
        self._max_projections = None
        self._processing = True
        self._status = "Running calibration..."

        threading.Thread(
            target=self._run_calibration_with_positions,
            args=(positions, z_indices),
            daemon=True,
        ).start()

    def _run_calibration_with_positions(self, positions, z_indices):
        try:
            arr = self._get_array()
            if arr is None:
                raise ValueError("No data")

            fpath = self._get_fpath()
            if fpath is None:
                fpath = Path("calibration")

            vol = (
                self._vol
                if self._vol is not None
                else np.asarray(arr[:]).astype(np.float32)
            )
            if self._vol is None:
                vol -= vol.mean()

            nz, nc, ny, nx = vol.shape

            from pollen._pollen_analysis import (
                analyze_power_vs_z,
                analyze_z_positions,
                calibrate_xy,
                correct_scan_phase,
                fit_exp_decay,
                plot_beamlet_grid,
                plot_z_spacing,
            )

            self._progress = 0.2
            # the viewer may still hold the max-projection ndarray, which has no metadata
            metadata = (
                self._original_metadata
                if self._original_metadata
                else getattr(arr, "metadata", {})
            )
            vol, _ = correct_scan_phase(
                vol, fpath, self._z_step_um, metadata, mode="manual"
            )

            self._progress = 0.3
            plot_beamlet_grid(vol, self._beam_order, fpath, mode="manual")

            self._progress = 0.4
            Iz, III = self._extract_traces(vol, positions, z_indices)
            xs = np.array([p[0] for p in positions])
            ys = np.array([p[1] for p in positions])

            self._progress = 0.5
            z_peaks, pp = analyze_power_vs_z(
                Iz, fpath, self._z_step_um, self._beam_order, nc, mode="manual"
            )

            self._progress = 0.6
            analyze_z_positions(
                z_peaks, self._beam_order, fpath, self._cavity_info, mode="manual"
            )

            self._progress = 0.7
            fit_exp_decay(
                z_peaks,
                self._beam_order,
                fpath,
                pp,
                self._cavity_info,
                self._z_step_um,
                nz,
                mode="manual",
            )

            self._progress = 0.8
            plot_z_spacing(z_peaks, self._beam_order, fpath, mode="manual")

            self._progress = 0.9
            dx = dy = self._pixel_size_um
            calibrate_xy(
                xs, ys, III, fpath, dx, dy, nx, ny, self._cavity_info, mode="manual"
            )

            self._progress = 1.0
            self._done = True
            self._results_manual = {
                "output_dir": str(fpath.parent),
                "h5_file": str(fpath.with_name(fpath.stem + "_pollen.h5")),
                "mode": "manual",
            }
            self.logger.info(
                f"Manual calibration complete! Results saved to {fpath.parent}"
            )

        except Exception as e:
            self.logger.exception(f"Calibration failed: {e}")
            self._error = str(e)
        finally:
            self._processing = False
            self._vol = None
            self._original_metadata = None
            self._original_data_array = None

    def _start_auto_calibration(self):
        """Load a previous ``*_pollen.h5`` beside the data, else calibrate in a background thread."""
        self._scan_existing_h5_files()
        if self._existing_h5_files:
            self._load_h5_file(str(self._existing_h5_files[0]))
            self._done = True
            self._status = "Loaded previous results"
            self.logger.info(
                "Found previous calibration results, skipping auto calibration"
            )
            return

        self._processing = True
        self._progress = 0.0
        self._done = False
        self._error = None
        self._results = None
        self._status = "Starting..."

        threading.Thread(target=self._auto_calibration_worker, daemon=True).start()

    def _auto_calibration_worker(self):
        try:
            arr = self._get_array()
            if arr is None:
                raise ValueError("No data loaded")

            fpath = self._get_fpath()
            if fpath is None:
                fpath = Path("calibration")

            self._status = "Loading data..."
            self._progress = 0.1
            vol = np.asarray(arr[:]).astype(np.float32)
            vol -= vol.mean()
            nz, nc, ny, nx = vol.shape

            self._status = "Detecting beads..."
            self._progress = 0.2
            positions, z_indices = self._detect_beads(vol)
            self.logger.info(f"Detected {len(positions)} bead positions")

            self._status = "Running calibration..."
            self._progress = 0.4

            from pollen._pollen_analysis import (
                analyze_power_vs_z,
                analyze_z_positions,
                calibrate_xy,
                correct_scan_phase,
                fit_exp_decay,
                plot_beamlet_grid,
                plot_z_spacing,
            )

            vol, _ = correct_scan_phase(
                vol, fpath, self._z_step_um, arr.metadata, mode="auto"
            )
            self._progress = 0.5

            plot_beamlet_grid(vol, self._beam_order, fpath, mode="auto")
            self._progress = 0.6

            Iz, III = self._extract_traces(vol, positions, z_indices)
            xs = np.array([p[0] for p in positions])
            ys = np.array([p[1] for p in positions])

            z_peaks, pp = analyze_power_vs_z(
                Iz, fpath, self._z_step_um, self._beam_order, nc, mode="auto"
            )
            self._progress = 0.7

            analyze_z_positions(
                z_peaks, self._beam_order, fpath, self._cavity_info, mode="auto"
            )

            fit_exp_decay(
                z_peaks,
                self._beam_order,
                fpath,
                pp,
                self._cavity_info,
                self._z_step_um,
                nz,
                mode="auto",
            )
            self._progress = 0.8

            plot_z_spacing(z_peaks, self._beam_order, fpath, mode="auto")
            self._progress = 0.9

            dx = dy = self._pixel_size_um
            calibrate_xy(
                xs, ys, III, fpath, dx, dy, nx, ny, self._cavity_info, mode="auto"
            )

            self._progress = 1.0
            self._done = True
            self._results_auto = {
                "output_dir": str(fpath.parent),
                "h5_file": str(fpath.with_name(fpath.stem + "_pollen.h5")),
                "mode": "auto",
            }
            self.logger.info(
                f"Auto calibration complete! Results saved to {fpath.parent}"
            )

        except Exception as e:
            self.logger.exception(f"Auto calibration failed: {e}")
            self._error = str(e)
        finally:
            self._processing = False

    def _detect_beads(self, vol):
        """Bead positions and best z per beamlet, in beam order: the brightest bead of the first beamlet, tracked into the others by normalized cross-correlation of its template."""
        from scipy.signal import correlate2d

        _nz, nc, ny, nx = vol.shape
        positions = []
        z_indices = []
        patch_size = 10
        template_size = 25
        max_projs = vol.max(axis=0)  # (nc, ny, nx)

        first_channel = self._beam_order[0] if len(self._beam_order) > 0 else 0
        ref_img = max_projs[first_channel]
        threshold = np.percentile(ref_img, 90)
        mask = ref_img > threshold
        if mask.sum() > 0:
            yy, xx = np.where(mask)
            weights = ref_img[mask]
            ref_cx = np.average(xx, weights=weights)
            ref_cy = np.average(yy, weights=weights)
        else:
            peak_idx = np.argmax(ref_img)
            ref_cy, ref_cx = np.unravel_index(peak_idx, ref_img.shape)

        ix, iy = round(ref_cx), round(ref_cy)
        t_y0 = max(0, iy - template_size)
        t_y1 = min(ny, iy + template_size + 1)
        t_x0 = max(0, ix - template_size)
        t_x1 = min(nx, ix + template_size + 1)
        template = ref_img[t_y0:t_y1, t_x0:t_x1].copy()
        template = template - template.mean()
        template_std = template.std()
        if template_std > 0:
            template = template / template_std

        for idx in range(nc):
            channel = self._beam_order[idx] if idx < len(self._beam_order) else idx
            img = max_projs[channel]

            if idx == 0:
                cx, cy = ref_cx, ref_cy
            else:
                img_norm = img - img.mean()
                img_std = img.std()
                if img_std > 0:
                    img_norm = img_norm / img_std
                corr = correlate2d(img_norm, template, mode="same")
                peak_idx = np.argmax(corr)
                cy, cx = np.unravel_index(peak_idx, corr.shape)
                p_y0 = max(0, cy - 5)
                p_y1 = min(ny, cy + 6)
                p_x0 = max(0, cx - 5)
                p_x1 = min(nx, cx + 6)
                peak_region = corr[p_y0:p_y1, p_x0:p_x1]

                if peak_region.max() > 0:
                    peak_region = peak_region - peak_region.min()
                    if peak_region.sum() > 0:
                        yy, xx = np.mgrid[
                            0 : peak_region.shape[0], 0 : peak_region.shape[1]
                        ]
                        cx = p_x0 + np.average(xx, weights=peak_region)
                        cy = p_y0 + np.average(yy, weights=peak_region)

            positions.append((float(cx), float(cy)))

            ix, iy = round(cx), round(cy)
            y0 = max(0, iy - patch_size)
            y1 = min(ny, iy + patch_size + 1)
            x0 = max(0, ix - patch_size)
            x1 = min(nx, ix + patch_size + 1)

            patch = vol[:, channel, y0:y1, x0:x1]
            smoothed = uniform_filter1d(patch.max(axis=(1, 2)), size=3, mode="nearest")
            best_z = int(np.argmax(smoothed))
            z_indices.append(best_z)

        return positions, z_indices

    def _extract_traces(self, vol, positions, z_indices):
        """Per beamlet, the smoothed patch maximum along z (``Iz``) and the patch at its best z (``III``, padded to one size)."""
        _nz, _nc, ny, nx = vol.shape
        Iz = []
        III = []
        patch_size = 10

        for idx, (x, y) in enumerate(positions):
            channel = self._beam_order[idx] if idx < len(self._beam_order) else idx
            ix, iy = round(x), round(y)

            y0 = max(0, iy - patch_size)
            y1 = min(ny, iy + patch_size + 1)
            x0 = max(0, ix - patch_size)
            x1 = min(nx, ix + patch_size + 1)

            patch = vol[:, channel, y0:y1, x0:x1]
            smoothed = uniform_filter1d(patch, size=3, axis=1, mode="nearest")
            smoothed = uniform_filter1d(smoothed, size=3, axis=2, mode="nearest")
            trace = smoothed.max(axis=(1, 2))
            Iz.append(trace)

            best_z = z_indices[idx] if idx < len(z_indices) else 0
            III.append(vol[best_z, channel, y0:y1, x0:x1])

        Iz = np.vstack(Iz) if Iz else np.zeros((0, vol.shape[0]))

        if III:
            max_h = max(im.shape[0] for im in III)
            max_w = max(im.shape[1] for im in III)
            pads = [
                np.pad(
                    im,
                    ((0, max_h - im.shape[0]), (0, max_w - im.shape[1])),
                    mode="constant",
                )
                for im in III
            ]
            III = np.stack(pads, axis=-1)
        else:
            III = np.zeros((2 * patch_size + 1, 2 * patch_size + 1, 0))

        return Iz, III

    def _scan_saved_images(self, mode: str = "auto"):
        """The ``pollen_<mode>_*`` figures in the run's output folder, by name."""
        self._saved_images = []
        results = self._results_auto if mode == "auto" else self._results_manual
        if not results:
            return

        out_dir = results.get("output_dir", "")
        if not out_dir:
            return

        out_path = Path(out_dir)
        if not out_path.exists():
            return

        prefix = f"pollen_{mode}_"
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
            for img in out_path.glob(ext):
                if img.name.startswith(prefix):
                    self._saved_images.append(img)
        self._saved_images.sort(key=lambda p: p.name)

    def _open_image(self, path: Path):
        """Open an image in the system's default viewer."""
        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                import os

                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as e:
            self.logger.exception(f"Failed to open image: {e}")

    def _open_output_folder(self, mode: str = "auto"):
        results = self._results_auto if mode == "auto" else self._results_manual
        if not results:
            return

        out_dir = results.get("output_dir", "")
        if out_dir:
            self._open_output_folder_path(out_dir)

    def _open_output_folder_path(self, folder_path: str):
        """Open a folder in the system's file browser."""
        import subprocess
        import sys

        if not folder_path:
            return

        try:
            if sys.platform == "win32":
                import os

                os.startfile(folder_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", folder_path], check=False)
            else:
                subprocess.run(["xdg-open", folder_path], check=False)
        except Exception as e:
            self.logger.exception(f"Failed to open folder: {e}")

    def _open_all_graphs(self, mode: str = "auto"):
        self._figures_popup_mode = mode
        self._show_figures_popup = True

    def _load_calibration_data(self, mode: str):
        """The calibration summary of ``mode``'s h5 (the run's, else the loaded file's), read once."""
        from pollen._pollen_analysis import extract_calibration_summary

        if mode == "auto" and self._calibration_data_auto is not None:
            return self._calibration_data_auto
        if mode == "manual" and self._calibration_data_manual is not None:
            return self._calibration_data_manual

        results = self._results_auto if mode == "auto" else self._results_manual
        if not results:
            if self._loaded_external and self._loaded_external.get("h5_path"):
                h5_path = self._loaded_external["h5_path"]
            else:
                return None
        else:
            h5_path = results.get("h5_file", "")

        if not h5_path:
            return None

        data = extract_calibration_summary(h5_path, mode=mode)
        if data:
            if mode == "auto":
                self._calibration_data_auto = data
            else:
                self._calibration_data_manual = data

        return data

    def _draw_figures_popup(self):
        """Draw popup window with calibration plots using implot."""
        io = imgui.get_io()
        screen_w, screen_h = io.display_size.x, io.display_size.y
        win_w, win_h = min(600, screen_w * 0.5), min(550, screen_h * 0.7)

        imgui.set_next_window_pos(
            imgui.ImVec2((screen_w - win_w) / 2, (screen_h - win_h) / 2),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(win_w, win_h), imgui.Cond_.first_use_ever
        )

        flags = imgui.WindowFlags_.no_collapse
        expanded, opened = imgui.begin("Calibration Results", True, flags)
        if not opened:
            self._show_figures_popup = False
            imgui.end()
            return

        if expanded:
            data_auto = self._load_calibration_data("auto")
            data_manual = self._load_calibration_data("manual")

            has_auto = data_auto is not None and "xs_um" in data_auto
            has_manual = data_manual is not None and "xs_um" in data_manual

            if not has_auto and not has_manual:
                imgui.text_disabled("No calibration data available")
                if imgui.button("Open Saved Figures"):
                    self._scan_saved_images(self._figures_popup_mode)
                    for img_path in self._saved_images:
                        self._open_image(img_path)
                imgui.end()
                return

            color_auto = imgui.ImVec4(0.0, 0.75, 1.0, 1.0)  # pollen_analysis.py's cyan
            color_manual = imgui.ImVec4(0.4, 1.0, 0.4, 1.0)  # and its green

            imgui.text("Legend:")
            imgui.same_line()
            imgui.text_colored(color_auto, "Auto")
            imgui.same_line()
            imgui.text_colored(color_manual, "Manual")
            imgui.spacing()

            plot_h = 220
            if implot.begin_plot("XY Positions (normalized)", imgui.ImVec2(-1, plot_h)):
                implot.setup_axes("dX (um)", "dY (um)")
                implot.setup_axis_limits(implot.ImAxis_.x1, -50, 50, implot.Cond_.once)
                implot.setup_axis_limits(implot.ImAxis_.y1, -50, 50, implot.Cond_.once)

                if has_auto:
                    xs_auto = np.asarray(data_auto["xs_um"], dtype=np.float64)
                    ys_auto = np.asarray(data_auto["ys_um"], dtype=np.float64)
                    xs_auto = xs_auto - xs_auto.mean()
                    ys_auto = ys_auto - ys_auto.mean()
                    implot.plot_scatter(
                        "Auto",
                        xs_auto,
                        ys_auto,
                        implot.Spec(
                            marker=implot.Marker_.circle.value,
                            marker_size=6.0,
                            marker_fill_color=color_auto,
                        ),
                    )

                if has_manual:
                    xs_manual = np.asarray(data_manual["xs_um"], dtype=np.float64)
                    ys_manual = np.asarray(data_manual["ys_um"], dtype=np.float64)
                    xs_manual = xs_manual - xs_manual.mean()
                    ys_manual = ys_manual - ys_manual.mean()
                    implot.plot_scatter(
                        "Manual",
                        xs_manual,
                        ys_manual,
                        implot.Spec(
                            marker=implot.Marker_.square.value,
                            marker_size=6.0,
                            marker_fill_color=color_manual,
                        ),
                    )

                implot.end_plot()

            imgui.spacing()

            if implot.begin_plot("XY Offsets (relative)", imgui.ImVec2(-1, plot_h)):
                implot.setup_axes("Beam #", "Offset (um)")
                n_beams = 0
                if has_auto and "diffx" in data_auto:
                    n_beams = max(n_beams, len(data_auto["diffx"]))
                if has_manual and "diffx" in data_manual:
                    n_beams = max(n_beams, len(data_manual["diffx"]))

                if n_beams > 0:
                    beam_nums = np.arange(1, n_beams + 1, dtype=np.float64)
                    bar_width = 0.35

                    if has_auto and "diffx" in data_auto:
                        diffx_auto = np.asarray(data_auto["diffx"], dtype=np.float64)
                        diffx_auto = diffx_auto - diffx_auto[0]
                        implot.plot_bars(
                            "dX Auto",
                            beam_nums - bar_width / 2,
                            diffx_auto,
                            bar_width,
                            implot.Spec(fill_color=color_auto),
                        )

                    if has_manual and "diffx" in data_manual:
                        diffx_manual = np.asarray(
                            data_manual["diffx"], dtype=np.float64
                        )
                        diffx_manual = diffx_manual - diffx_manual[0]
                        implot.plot_bars(
                            "dX Manual",
                            beam_nums + bar_width / 2,
                            diffx_manual,
                            bar_width,
                            implot.Spec(fill_color=color_manual),
                        )

                implot.end_plot()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.9, 1.0), "Summary:")
            if has_auto:
                rms_dx = data_auto.get("rms_dx")
                rms_dy = data_auto.get("rms_dy")
                if rms_dx is not None and rms_dy is not None:
                    imgui.text_colored(
                        color_auto, f"  Auto RMS: dX={rms_dx:.2f}um, dY={rms_dy:.2f}um"
                    )

            if has_manual:
                rms_dx = data_manual.get("rms_dx")
                rms_dy = data_manual.get("rms_dy")
                if rms_dx is not None and rms_dy is not None:
                    imgui.text_colored(
                        color_manual,
                        f"  Manual RMS: dX={rms_dx:.2f}um, dY={rms_dy:.2f}um",
                    )

            imgui.spacing()

            if imgui.button("Open Output Folder"):
                if self._results_auto:
                    self._open_output_folder("auto")
                elif self._results_manual:
                    self._open_output_folder("manual")
                elif self._loaded_external and self._loaded_external.get("h5_path"):
                    h5_path = Path(self._loaded_external["h5_path"])
                    if h5_path.parent.exists():
                        self._open_output_folder_path(str(h5_path.parent))

        imgui.end()

    def on_data_loaded(self) -> None:
        """Reinitialize when new data is loaded."""
        self._init_from_data()
        self._done = False
        self._error = None
        self._results_auto = None
        self._results_manual = None
        self._manual_mode = False
        self._saved_images = []
        self._show_figures_popup = False
        self._calibration_data_auto = None
        self._calibration_data_manual = None

        if self._initialized and not self._processing:
            self._start_auto_calibration()

    def cleanup(self) -> None:
        """Clean up resources when viewer closes."""
        self._vol = None
        self._max_projections = None
        self._original_metadata = None
        self._original_data_array = None
        self._manual_mode = False
        self._pointer_down_pos = None
        self._calibration_data_auto = None
        self._calibration_data_manual = None
        super().cleanup()
