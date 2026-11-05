"""
suite2p pipeline widget.

combines processing configuration with a button to view trace quality statistics
in a separate popup window.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui

from mbo_utilities.gui._availability import HAS_SUITE2P
from mbo_utilities.gui._imgui_helpers import PopupAutoSize, set_tooltip
from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.lazy_array import base_array
from mbo_utilities.preferences import set_last_dir

_HAS_LSP: bool | None = None


def _check_lsp_available() -> bool:
    """Check if lbm_suite2p_python and suite2p are both available."""
    global _HAS_LSP
    if _HAS_LSP is None:
        import importlib.util

        _HAS_LSP = (
            importlib.util.find_spec("lbm_suite2p_python") is not None
            and importlib.util.find_spec("suite2p") is not None
        )
    return _HAS_LSP


def _patch_pyqt6_slider():
    """Give QSlider the ``NoTicks`` name suite2p's gui reads; PyQt6 moved it onto ``TickPosition``."""
    from PyQt6.QtWidgets import QSlider

    if not hasattr(QSlider, "NoTicks"):
        QSlider.NoTicks = QSlider.TickPosition.NoTicks


class Suite2pPipelineWidget(PipelineWidget):
    """suite2p processing and results widget."""

    name = "Suite2p"
    install_command = "uv pip install --no-deps lbm-suite2p-python && uv pip install suite2p rastermap torch torchvision"
    extracts_traces = True

    @property
    def is_available(self) -> bool:
        """Check availability lazily to avoid slow imports at module load."""
        return HAS_SUITE2P and _check_lsp_available()

    @classmethod
    def extract_traces(cls, movie: Any, labels: Any) -> dict | None:
        """Run suite2p's own extractor over hand-drawn masks.

        Builds the `stat` dicts suite2p's mask builder expects straight
        from a label image, so the ROIs skip detection and go into the
        same weighted extraction — which also gives the surrounding
        neuropil trace that a plain mask average cannot.
        """
        import torch
        from suite2p.extraction.extract import extract_traces
        from suite2p.extraction.masks import create_masks

        labels = np.asarray(labels)
        n_rois = int(labels.max())
        if n_rois == 0:
            return None
        ly, lx = labels.shape

        stats = []
        for i in range(1, n_rois + 1):
            ypix, xpix = np.nonzero(labels == i)
            npix = int(ypix.size)
            stats.append(
                {
                    "ypix": ypix.astype(np.int32),
                    "xpix": xpix.astype(np.int32),
                    "lam": np.ones(npix, np.float32),
                    "npix": npix,
                    "med": [int(np.median(ypix)), int(np.median(xpix))],
                    "radius": float(np.sqrt(npix / np.pi)),
                    "overlap": np.zeros(npix, bool),
                }
            )

        cell_masks, neuropil_masks = create_masks(stats, ly, lx)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f_cell, f_neuropil = extract_traces(
            movie, cell_masks, neuropil_masks, device=device
        )
        return {"F": f_cell, "Fneu": f_neuropil}

    def __init__(self, parent: Any):
        super().__init__(parent)

        self._diagnostics_widget = None
        self._show_diagnostics_popup = False
        self._diagnostics_popup_open = False
        self._diagnostics_unit = None
        self._file_dialog = None

        self._grid_search_widget = None
        self._show_grid_search_popup = False
        self._grid_search_popup_open = False
        self._grid_search_dialog = None

        self._external_gui_window = None
        self._external_gui_type = None
        self._last_suite2p_ichosen = None
        self._last_poll_time = 0.0
        self._poll_interval = 0.1

        self._gui_choice = 0
        self._diagnostics_sizer = PopupAutoSize(
            "Trace Quality Statistics", auto_resize=False
        )
        self._grid_search_sizer = PopupAutoSize(
            "Grid Search Results", auto_resize=False
        )

    def _get_diagnostics_widget(self):
        """Get diagnostics widget (lazy init)."""
        if self._diagnostics_widget is None:
            from mbo_utilities.gui.widgets.diagnostics import DiagnosticsWidget

            self._diagnostics_widget = DiagnosticsWidget()
        return self._diagnostics_widget

    def _get_grid_search_widget(self):
        """Get grid search widget (lazy init)."""
        if self._grid_search_widget is None:
            from mbo_utilities.gui.widgets.grid_search import GridSearchViewer

            self._grid_search_widget = GridSearchViewer()
        return self._grid_search_widget

    def _check_suite2p_gui(self) -> bool:
        """Check if suite2p GUI is available (requires rastermap)."""
        import importlib.util

        return (
            importlib.util.find_spec("suite2p.gui") is not None
            and importlib.util.find_spec("rastermap") is not None
        )

    def _check_cellpose_gui(self) -> bool:
        """Check if cellpose GUI is available."""
        import importlib.util

        return importlib.util.find_spec("cellpose.gui") is not None

    def draw_config(self) -> None:
        """Draw suite2p configuration ui."""
        from mbo_utilities.gui.widgets.pipelines.settings import draw_section_suite2p

        imgui.spacing()

        self.parent._current_pipeline = "suite2p"
        draw_section_suite2p(self.parent)

        self._poll_suite2p_selection()

        self._draw_diagnostics_popup()
        self._draw_grid_search_popup()

    def _poll_suite2p_selection(self):
        """Poll suite2p window for selection changes."""
        if self._external_gui_type != "suite2p" or self._external_gui_window is None:
            return

        current_time = time.time()
        if current_time - self._last_poll_time < self._poll_interval:
            return
        self._last_poll_time = current_time

        try:
            if not self._external_gui_window.isVisible():
                self._external_gui_window = None
                self._external_gui_type = None
                self._last_suite2p_ichosen = None
                return
        except RuntimeError:
            self._external_gui_window = None
            self._external_gui_type = None
            self._last_suite2p_ichosen = None
            return

        if (
            not hasattr(self._external_gui_window, "loaded")
            or not self._external_gui_window.loaded
        ):
            return

        ichosen = getattr(self._external_gui_window, "ichosen", None)

        if ichosen != self._last_suite2p_ichosen:
            self._last_suite2p_ichosen = ichosen
            self._on_suite2p_cell_selected(ichosen)

    def _on_suite2p_cell_selected(self, cell_idx: int):
        """Handle cell selection in suite2p GUI.

        Parameters
        ----------
        cell_idx : int
            Index of the selected cell in suite2p
        """
        if cell_idx is None:
            return

        diag = self._get_diagnostics_widget()
        visible = diag.visible_indices
        if len(visible) > 0:
            matches = np.where(visible == cell_idx)[0]
            if len(matches) > 0:
                diag.selected_roi = int(matches[0])

    def _screen_unit(self):
        """The results unit of the plane on screen, when the data shown is a
        run's output with a plane dir to write curation back to; else None.
        """
        host = self.parent
        on_screen = host.slice
        data = host.image_widget.data
        results = base_array(data[0]).results if data else None
        if results is None:
            return None
        return next(
            (
                u
                for u in results.units.values()
                if u.member_kind == "pixel"
                and u.attrs.get("plane_dir")
                and u.attrs.get("z") == on_screen.z
            ),
            None,
        )

    def _draw_diagnostics_popup(self):
        """Draw the diagnostics popup window if open."""
        unit = self._screen_unit()
        if unit is not None:
            if imgui.button("Diagnostics of the plane on screen"):
                self._show_diagnostics_popup = True
                self._diagnostics_unit = None
            set_tooltip(
                "The ROIs, accept flags and traces the run found on the z-plane "
                "shown; follows the Z slider",
                show_mark=False,
            )
            if (
                self._diagnostics_popup_open or self._show_diagnostics_popup
            ) and self._diagnostics_unit is not unit:
                self._get_diagnostics_widget().load_unit(
                    unit, Path(unit.attrs["plane_dir"])
                )
                self._diagnostics_unit = unit
        if self._file_dialog is not None and self._file_dialog.ready():
            result = self._file_dialog.result()
            if result and len(result) > 0:
                stat_path = Path(result[0])
                set_last_dir("suite2p_stat", stat_path)
                if stat_path.name == "stat.npy" and stat_path.exists():
                    self._get_diagnostics_widget().load_results(stat_path.parent)
                    self._show_diagnostics_popup = True
                    self._open_external_gui(stat_path)
            self._file_dialog = None

        if self._show_diagnostics_popup:
            self._diagnostics_popup_open = True
            self._diagnostics_sizer.before_open()
            imgui.open_popup("Trace Quality Statistics")
            self._show_diagnostics_popup = False

        viewport = imgui.get_main_viewport()
        popup_width = min(1200, viewport.size.x * 0.9)
        popup_height = min(800, viewport.size.y * 0.85)
        imgui.set_next_window_size(
            imgui.ImVec2(popup_width, popup_height), imgui.Cond_.first_use_ever
        )

        opened, visible = imgui.begin_popup_modal(
            "Trace Quality Statistics",
            p_open=True if self._diagnostics_popup_open else None,
            flags=imgui.WindowFlags_.no_saved_settings,
        )

        if opened:
            if not visible:
                self._diagnostics_popup_open = False
                imgui.close_current_popup()
            else:
                try:
                    self._get_diagnostics_widget().draw()
                except Exception as e:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {e}")

                imgui.spacing()
                imgui.separator()
                if imgui.button("Close", imgui.ImVec2(100, 0)):
                    self._diagnostics_popup_open = False
                    imgui.close_current_popup()

            imgui.end_popup()

    def _open_external_gui(self, statfile: Path):
        """Open external GUI (suite2p or cellpose) based on user choice.

        Parameters
        ----------
        statfile : Path
            Path to stat.npy file
        """
        if self._gui_choice == 0:
            if self._check_suite2p_gui():
                self._open_suite2p_gui(statfile)
            else:
                pass
        elif self._gui_choice == 1:
            if self._check_cellpose_gui():
                self._open_cellpose_gui(statfile)
            else:
                pass

    def _open_suite2p_gui(self, statfile: Path):
        """Open suite2p GUI with the given stat.npy file."""
        try:
            _patch_pyqt6_slider()
            from PyQt6.QtCore import QRect
            from PyQt6.QtWidgets import QApplication
            from suite2p.gui.gui2p import MainWindow as Suite2pMainWindow

            self._external_gui_window = Suite2pMainWindow(statfile=str(statfile))
            self._external_gui_type = "suite2p"

            screen = QApplication.primaryScreen()
            if screen:
                screen_geom = screen.availableGeometry()
                screen_x = screen_geom.x()
                screen_y = screen_geom.y()
                screen_w = screen_geom.width()
                screen_h = screen_geom.height()
                half_width = screen_w // 2
                margin_top = 30
                margin_bottom = 10
                win_height = screen_h - margin_top - margin_bottom

                self._external_gui_window.setGeometry(
                    QRect(
                        screen_x + half_width,
                        screen_y + margin_top,
                        half_width,
                        win_height,
                    )
                )
                self._external_gui_window.setMinimumSize(400, 300)
                self._reposition_mbo_window(
                    screen_x, screen_y + margin_top, half_width, win_height
                )

            self._external_gui_window.show()
            self._external_gui_window.showNormal()
            self._last_suite2p_ichosen = None
        except Exception:
            self.parent.logger.exception("suite2p gui failed to open")

    def _open_cellpose_gui(self, statfile: Path):
        """Open cellpose GUI with Suite2p results.

        Converts stat.npy to a cellpose-compatible _seg.npy file, then opens
        the cellpose GUI using cellpose's native loading mechanism.
        Edits made in the GUI can be saved and will persist.
        """
        try:
            from cellpose.gui import io as cellpose_io
            from cellpose.gui.gui import MainW
            from lbm_suite2p_python.conversion import ensure_cellpose_format
            from PyQt6.QtCore import QRect
            from PyQt6.QtWidgets import QApplication

            from mbo_utilities.analysis import _patch_qt_checkbox

            _patch_qt_checkbox()

            plane_dir = statfile.parent

            seg_file = ensure_cellpose_format(plane_dir)

            self._external_gui_window = MainW()
            self._external_gui_type = "cellpose"

            cellpose_io._load_seg(
                self._external_gui_window, filename=str(seg_file), load_3D=False
            )

            screen = QApplication.primaryScreen()
            if screen:
                screen_geom = screen.availableGeometry()
                screen_x = screen_geom.x()
                screen_y = screen_geom.y()
                screen_w = screen_geom.width()
                screen_h = screen_geom.height()
                half_width = screen_w // 2
                margin_top = 30
                margin_bottom = 10
                win_height = screen_h - margin_top - margin_bottom

                self._external_gui_window.setGeometry(
                    QRect(
                        screen_x + half_width,
                        screen_y + margin_top,
                        half_width,
                        win_height,
                    )
                )
                self._reposition_mbo_window(
                    screen_x, screen_y + margin_top, half_width, win_height
                )

            self._external_gui_window.show()
        except Exception:
            self.parent.logger.exception("cellpose gui failed to open")

    def _reposition_mbo_window(self, x: int, y: int, width: int, height: int):
        """Reposition the MBO window to the specified geometry.

        Uses the Qt canvas from fastplotlib's ImageWidget to find and
        reposition the parent window.

        Parameters
        ----------
        x, y : int
            Window position
        width, height : int
            Window size
        """
        from PyQt6.QtCore import QRect

        canvas = self.parent.image_widget.figure.canvas
        if not hasattr(canvas, "isWindow"):  # not a qt canvas
            return
        window = canvas.window()
        if window:
            window.setGeometry(QRect(x, y, width, height))
            return
        widget = canvas
        while widget is not None:
            if widget.isWindow():
                widget.setGeometry(QRect(x, y, width, height))
                return
            widget = widget.parent()

    def _draw_grid_search_popup(self):
        """Draw the grid search viewer popup window if open."""
        if self._grid_search_dialog is not None and self._grid_search_dialog.ready():
            result = self._grid_search_dialog.result()
            if result:
                set_last_dir("grid_search", result)
                self._get_grid_search_widget().load_results(Path(result))
                self._show_grid_search_popup = True
            self._grid_search_dialog = None

        if self._show_grid_search_popup:
            self._grid_search_popup_open = True
            self._grid_search_sizer.before_open()
            imgui.open_popup("Grid Search Results")
            self._show_grid_search_popup = False

        viewport = imgui.get_main_viewport()
        popup_width = min(1200, viewport.size.x * 0.9)
        popup_height = min(800, viewport.size.y * 0.85)
        imgui.set_next_window_size(
            imgui.ImVec2(popup_width, popup_height), imgui.Cond_.first_use_ever
        )

        opened, visible = imgui.begin_popup_modal(
            "Grid Search Results",
            p_open=True if self._grid_search_popup_open else None,
            flags=imgui.WindowFlags_.no_saved_settings,
        )

        if opened:
            if not visible:
                self._grid_search_popup_open = False
                imgui.close_current_popup()
            else:
                try:
                    self._get_grid_search_widget().draw()
                except Exception as e:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {e}")

                imgui.spacing()
                imgui.separator()
                if imgui.button("Close", imgui.ImVec2(100, 0)):
                    self._grid_search_popup_open = False
                    imgui.close_current_popup()

            imgui.end_popup()

    def cleanup(self):
        """Clean up resources when widget is destroyed.

        Should be called when the parent GUI is closing to ensure
        proper cleanup of Qt windows and other resources.
        """
        if self._suite2p_window is not None:
            try:
                self._suite2p_window.close()
            except (RuntimeError, AttributeError):
                pass
            self._suite2p_window = None
            self._last_suite2p_ichosen = None

        self._file_dialog = None
        self._grid_search_dialog = None
