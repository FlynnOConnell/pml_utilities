"""
ROI Diagnostics Widget for Suite2p Results.

Provides ROI filtering and visualization:
- dF/F trace with adjustable baseline (median or percentile)
- Metric histograms with threshold sliders for filtering
- Updates iscell based on filter thresholds
- Bi-directional sync with Suite2p GUI via file watching
"""

import shutil
import time
from pathlib import Path

import numpy as np
from imgui_bundle import imgui, implot
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities.gui._imgui_helpers import PopupAutoSize
from mbo_utilities.preferences import get_last_dir, set_last_dir


class DiagnosticsWidget:
    """ROI diagnostics viewer for suite2p results.

    Supports bi-directional synchronization with Suite2p GUI:
    - Changes here auto-save to iscell.npy
    - External changes to iscell.npy are detected and reloaded
    """

    def __init__(self):
        self.stat = None
        self.iscell = None
        self.iscell_original = None
        self.F = None
        self.Fneu = None
        self.ops = None
        self.loaded_path = None
        self._save_path = None

        self.selected_roi = 0
        self.show_only_cells = True

        self._baseline_method = 0
        self._baseline_percentile = 10.0
        self._neuropil_coeff = 0.7

        self._snr = None
        self._dff = None
        self._skewness = None
        self._activity = None
        self._shot_noise = None

        self._snr_min = 0.0
        self._snr_max = 100.0
        self._shot_noise_min = 0.0
        self._shot_noise_max = 1000.0
        self._skewness_min = -10.0
        self._skewness_max = 10.0
        self._activity_min = 0.0
        self._activity_max = 1.0

        self._filter_snr_min = 0.0
        self._filter_shot_noise_max = 1000.0
        self._filter_skewness_min = 0.0
        self._filter_activity_min = 0.0

        self._trace_height_frac = 0.30
        self._min_plot_height = 60

        self._show_stats_popup = False
        self._stats_popup_open = False
        self._stats_sizer = PopupAutoSize("ROI Statistics", auto_resize=False)

        self._auto_save = True
        self._watch_for_changes = True
        self._last_iscell_mtime = None
        self._last_check_time = 0
        self._check_interval = 1.0
        self._last_save_time = 0
        self._save_cooldown = (
            0.5  # NOTE: our own save would otherwise be reloaded as an external change
        )
        self._save_status_msg = ""
        self._save_status_time = 0

        self._training_dir = None
        self._exported_datasets = []

    def load_unit(self, unit, plane_dir: Path) -> None:
        """Show one pixel unit of a run's results (``results.ResultUnit``):
        its ROIs, accept flags and traces, with ``ops.npy`` from the plane
        dir it was read from, where ``iscell.npy`` is written back.
        """
        from mbo_utilities.file_io import has_plane_ops, load_plane_ops

        plane_dir = Path(plane_dir)
        self.ops = load_plane_ops(plane_dir) if has_plane_ops(plane_dir) else None
        ly, lx = unit.image_shape
        stat = np.empty(unit.n_rois, dtype=object)
        weights = unit.weights if unit.weights is not None else [None] * unit.n_rois
        for k, (members, lam) in enumerate(zip(unit.members, weights, strict=True)):
            ypix, xpix = np.divmod(np.asarray(members, np.int64).ravel(), lx)
            stat[k] = {
                "ypix": ypix.astype(np.int32),
                "xpix": xpix.astype(np.int32),
                "lam": np.ones(ypix.size, np.float32)
                if lam is None
                else np.asarray(lam, np.float32),
                "med": (float(np.median(ypix)), float(np.median(xpix)))
                if ypix.size
                else (0.0, 0.0),
                "npix": int(ypix.size),
            }
        self.stat = stat
        self.iscell = (
            None if unit.iscell is None else np.asarray(unit.iscell, np.float32).copy()
        )
        self.iscell_original = None if self.iscell is None else self.iscell.copy()
        self.F = unit.traces.get("raw")
        self.Fneu = unit.traces.get("neuropil")
        self.loaded_path = plane_dir
        self._save_path = plane_dir
        iscell_path = plane_dir / "iscell.npy"
        self._last_iscell_mtime = (
            iscell_path.stat().st_mtime if iscell_path.exists() else None
        )
        self._compute_metrics()
        self._update_filter_ranges()
        self.selected_roi = 0

    def load_results(self, plane_dir: Path):
        """Load suite2p results from a plane directory."""
        from mbo_utilities.file_io import has_plane_ops, load_npy, load_plane_ops

        plane_dir = Path(plane_dir)

        if has_plane_ops(plane_dir):
            self.ops = load_plane_ops(plane_dir)
            save_path = Path(self.ops.get("save_path", plane_dir))
            if not save_path.exists():
                save_path = plane_dir
        else:
            self.ops = None
            save_path = plane_dir

        stat_path = save_path / "stat.npy"
        iscell_path = save_path / "iscell.npy"
        f_path = save_path / "F.npy"
        fneu_path = save_path / "Fneu.npy"

        if stat_path.exists():
            self.stat = load_npy(stat_path)
        else:
            self.stat = None

        if iscell_path.exists():
            self.iscell = load_npy(iscell_path)
            if self.iscell is not None:
                self.iscell_original = self.iscell.copy()
        else:
            self.iscell = None
            self.iscell_original = None

        if f_path.exists():
            self.F = load_npy(f_path)
        else:
            self.F = None

        if fneu_path.exists():
            self.Fneu = load_npy(fneu_path)
        else:
            self.Fneu = None

        self.loaded_path = plane_dir
        self._save_path = save_path

        iscell_path = save_path / "iscell.npy"
        if iscell_path.exists():
            self._last_iscell_mtime = iscell_path.stat().st_mtime
        else:
            self._last_iscell_mtime = None

        self._compute_metrics()
        self._update_filter_ranges()
        self.selected_roi = 0

    def _compute_metrics(self):
        """Compute quality metrics for all ROIs."""
        if self.F is None or self.stat is None:
            return

        self._recompute_dff()

        n_rois = len(self.F)

        noise = np.std(self._dff, axis=1)
        signal = np.max(self._dff, axis=1) - np.min(self._dff, axis=1)
        self._snr = np.where(noise > 0, signal / noise, 0)

        self._skewness = np.zeros(n_rois)
        self._activity = np.zeros(n_rois)
        self._shot_noise = np.zeros(n_rois)

        for i, s in enumerate(self.stat):
            self._skewness[i] = s.get("skew", 0)
            self._activity[i] = np.sum(self._dff[i] > 0.5) / len(self._dff[i])
            self._shot_noise[i] = (
                s.get("std", np.std(self.F[i])) if self.F is not None else 0
            )

    def _recompute_dff(self):
        """Recompute dF/F with current baseline settings."""
        if self.F is None:
            return

        if self.Fneu is not None:
            F_corrected = self.F - self._neuropil_coeff * self.Fneu
        else:
            F_corrected = self.F

        if self._baseline_method == 0:
            baseline = np.percentile(
                F_corrected, self._baseline_percentile, axis=1, keepdims=True
            )
        else:
            baseline = np.median(F_corrected, axis=1, keepdims=True)

        baseline = np.maximum(baseline, 1.0)
        self._dff = (F_corrected - baseline) / baseline

    def _update_filter_ranges(self):
        """Update filter slider ranges based on data."""
        if self._snr is not None:
            self._snr_min = float(np.min(self._snr))
            self._snr_max = float(np.max(self._snr))
            self._filter_snr_min = self._snr_min
        if self._shot_noise is not None:
            self._shot_noise_min = float(np.min(self._shot_noise))
            self._shot_noise_max = float(np.max(self._shot_noise))
            self._filter_shot_noise_max = self._shot_noise_max
        if self._skewness is not None:
            self._skewness_min = float(np.min(self._skewness))
            self._skewness_max = float(np.max(self._skewness))
            self._filter_skewness_min = self._skewness_min
        if self._activity is not None:
            self._activity_min = float(np.min(self._activity))
            self._activity_max = float(np.max(self._activity))
            self._filter_activity_min = self._activity_min

    def _apply_filters(self):
        """Apply current filter thresholds to iscell."""
        if self.iscell is None or self.iscell_original is None:
            return

        n_rois = len(self.stat)
        changed = False
        for i in range(n_rois):
            passes = True
            if self._snr is not None and self._snr[i] < self._filter_snr_min:
                passes = False
            if (
                self._shot_noise is not None
                and self._shot_noise[i] > self._filter_shot_noise_max
            ):
                passes = False
            if (
                self._skewness is not None
                and self._skewness[i] < self._filter_skewness_min
            ):
                passes = False
            if (
                self._activity is not None
                and self._activity[i] < self._filter_activity_min
            ):
                passes = False

            orig_prob = (
                self.iscell_original[i, 0]
                if self.iscell_original.ndim > 1
                else self.iscell_original[i]
            )
            if orig_prob > 0.5:
                if self.iscell.ndim > 1:
                    new_val = orig_prob if passes else 0.0
                    if self.iscell[i, 0] != new_val:
                        self.iscell[i, 0] = new_val
                        changed = True
                else:
                    new_val = orig_prob if passes else 0.0
                    if self.iscell[i] != new_val:
                        self.iscell[i] = new_val
                        changed = True

        if changed and self._auto_save:
            self._save_iscell()

    def _check_for_external_changes(self):
        """Check if iscell.npy was modified externally (e.g., by Suite2p GUI)."""
        if not self._watch_for_changes or self._save_path is None:
            return

        current_time = time.time()

        if current_time - self._last_check_time < self._check_interval:
            return
        self._last_check_time = current_time

        if current_time - self._last_save_time < self._save_cooldown:
            return

        iscell_path = self._save_path / "iscell.npy"
        if not iscell_path.exists():
            return

        current_mtime = iscell_path.stat().st_mtime
        if (
            self._last_iscell_mtime is not None
            and current_mtime > self._last_iscell_mtime
        ):
            self._reload_iscell()

    def _reload_iscell(self):
        """Reload iscell.npy from disk (called when external changes detected)."""
        from mbo_utilities.file_io import load_npy

        iscell_path = self._save_path / "iscell.npy"
        if not iscell_path.exists():
            return

        try:
            new_iscell = load_npy(iscell_path)
        except (OSError, ValueError):  # NOTE: suite2p may still be writing the file
            return
        if new_iscell is not None:
            self.iscell = new_iscell
            self._last_iscell_mtime = iscell_path.stat().st_mtime

    @property
    def n_rois(self):
        """Number of ROIs."""
        return len(self.stat) if self.stat is not None else 0

    @property
    def cell_indices(self):
        """Indices of ROIs classified as cells."""
        if self.iscell is None:
            return np.arange(self.n_rois)
        if self.iscell.ndim == 2:
            return np.where(self.iscell[:, 0] > 0.5)[0]
        return np.where(self.iscell[:] > 0.5)[0]

    @property
    def visible_indices(self):
        """Indices of currently visible ROIs."""
        if self.show_only_cells:
            return self.cell_indices
        return np.arange(self.n_rois)

    def draw(self):
        """Draw the diagnostics widget."""
        if self.stat is None:
            self._draw_load_ui()
            return

        self._check_for_external_changes()

        avail = imgui.get_content_region_avail()
        left_width = min(280, avail.x * 0.28)

        if imgui.begin_child(
            "DiagLeft", imgui.ImVec2(left_width, 0), imgui.ChildFlags_.borders
        ):
            self._draw_controls()
            imgui.separator()
            self._draw_baseline_settings()
            imgui.separator()
            self._draw_sync_settings()
        imgui.end_child()

        imgui.same_line()

        if imgui.begin_child(
            "DiagRight", imgui.ImVec2(0, 0), imgui.ChildFlags_.borders
        ):
            self._draw_right_panel()
        imgui.end_child()

        self._draw_stats_popup()

    def _draw_load_ui(self):
        """Draw UI for loading results."""
        if imgui.button("Load suite2p results"):
            default_dir = str(get_last_dir("suite2p_diagnostics") or Path.home())
            result = pfd.select_folder("Select plane folder", default_dir)
            if result and result.result():
                try:
                    set_last_dir("suite2p_diagnostics", result.result())
                    self.load_results(Path(result.result()))
                except Exception as e:
                    imgui.text_colored(imgui.ImVec4(1, 0, 0, 1), f"Error: {e}")

    def _draw_controls(self):
        """Draw ROI navigation controls."""
        imgui.text(f"Path: {self.loaded_path.name if self.loaded_path else 'None'}")

        n_cells = len(self.cell_indices)
        n_orig = (
            len(np.where(self.iscell_original[:, 0] > 0.5)[0])
            if self.iscell_original is not None and self.iscell_original.ndim > 1
            else n_cells
        )
        imgui.text(f"ROIs: {self.n_rois} | Cells: {n_cells}/{n_orig}")

        imgui.spacing()

        visible = self.visible_indices
        if len(visible) > 0:
            if self.selected_roi >= len(visible):
                self.selected_roi = 0

            imgui.set_next_item_width(120)
            changed, new_val = imgui.slider_int(
                "ROI", self.selected_roi, 0, len(visible) - 1
            )
            if changed:
                self.selected_roi = new_val

            imgui.same_line()
            if imgui.button("<"):
                self.selected_roi = max(0, self.selected_roi - 1)
            imgui.same_line()
            if imgui.button(">"):
                self.selected_roi = min(len(visible) - 1, self.selected_roi + 1)

            roi_idx = visible[self.selected_roi]
            imgui.text(f"ROI {roi_idx}")
            if self.iscell is not None:
                prob = (
                    self.iscell[roi_idx, 0]
                    if self.iscell.ndim > 1
                    else self.iscell[roi_idx]
                )
                color = (
                    imgui.ImVec4(0.2, 1.0, 0.2, 1.0)
                    if prob > 0.5
                    else imgui.ImVec4(1.0, 0.5, 0.2, 1.0)
                )
                imgui.same_line()
                imgui.text_colored(color, f"p={prob:.2f}")

        imgui.spacing()
        _, self.show_only_cells = imgui.checkbox(
            "Show only cells", self.show_only_cells
        )

        if imgui.button("View Stats..."):
            self._show_stats_popup = True

        imgui.same_line()
        if imgui.button("Reset Filters"):
            self._update_filter_ranges()
            if self.iscell_original is not None:
                self.iscell = self.iscell_original.copy()

        imgui.same_line()
        if imgui.button("Save iscell"):
            self._save_iscell()

    def _draw_baseline_settings(self):
        """Draw dF/F baseline configuration."""
        imgui.text("dF/F Baseline")

        imgui.set_next_item_width(100)
        changed_method, self._baseline_method = imgui.combo(
            "Method", self._baseline_method, ["Percentile", "Median"]
        )

        if self._baseline_method == 0:
            imgui.set_next_item_width(100)
            changed_pct, self._baseline_percentile = imgui.slider_float(
                "Percentile", self._baseline_percentile, 1.0, 50.0, "%.0f%%"
            )
        else:
            changed_pct = False

        imgui.set_next_item_width(100)
        changed_coeff, self._neuropil_coeff = imgui.slider_float(
            "Fneu coeff", self._neuropil_coeff, 0.0, 1.0, "%.2f"
        )

        if changed_method or changed_pct or changed_coeff:
            self._recompute_dff()
            if self._dff is not None:
                noise = np.std(self._dff, axis=1)
                signal = np.max(self._dff, axis=1) - np.min(self._dff, axis=1)
                self._snr = np.where(noise > 0, signal / noise, 0)
                for i in range(len(self.stat)):
                    self._activity[i] = np.sum(self._dff[i] > 0.5) / len(self._dff[i])
                self._update_filter_ranges()

    def _draw_sync_settings(self):
        """Draw Suite2p GUI synchronization settings."""
        imgui.text("Suite2p Sync")

        _, self._auto_save = imgui.checkbox("Auto-save", self._auto_save)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Auto-save iscell.npy when filters change")

        _, self._watch_for_changes = imgui.checkbox(
            "Watch file", self._watch_for_changes
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Reload when iscell.npy is modified externally (e.g., by Suite2p GUI)"
            )

        if self._save_status_msg and time.time() - self._save_status_time < 3.0:
            if "error" in self._save_status_msg.lower():
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.3, 0.3, 1.0), self._save_status_msg
                )
            else:
                imgui.text_colored(
                    imgui.ImVec4(0.3, 1.0, 0.3, 1.0), self._save_status_msg
                )
        elif self._auto_save:
            imgui.text_colored(imgui.ImVec4(0.5, 0.7, 0.5, 1.0), "Auto-save ON")

        imgui.text_wrapped("Suite2p: File > Load to refresh")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "iscell.npy is saved when filters change.\n\n"
                "To see changes in Suite2p GUI:\n"
                "  1. File > Load results (or drag stat.npy)\n"
                "  2. Or press 'r' to reload in some versions\n\n"
                "Suite2p GUI doesn't auto-refresh from disk."
            )

        imgui.separator()
        imgui.text("Classifier Training")

        if imgui.button("Export Training"):
            self._export_for_training()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "export current iscell + stat features for classifier training"
            )

        imgui.same_line()
        if imgui.button("Mark Curated"):
            self._mark_as_curated()
        if imgui.is_item_hovered():
            imgui.set_tooltip("copy iscell.npy and stat.npy to training folder")

        if self._training_dir:
            imgui.text_colored(
                imgui.ImVec4(0.5, 0.8, 0.5, 1.0), f"Training: {self._training_dir.name}"
            )

        if len(self._exported_datasets) > 0:
            imgui.text(f"Datasets: {len(self._exported_datasets)}")

            if imgui.button("Build Classifier"):
                self._build_classifier()
            if imgui.is_item_hovered():
                imgui.set_tooltip("build classifier from all exported datasets")

            imgui.same_line()
            if imgui.button("Clear"):
                self._exported_datasets = []

    def _draw_right_panel(self):
        """Draw right panel with trace and histograms."""
        avail = imgui.get_content_region_avail()
        trace_height = max(self._min_plot_height, avail.y * self._trace_height_frac)

        if imgui.begin_child(
            "TraceSection", imgui.ImVec2(-1, trace_height), imgui.ChildFlags_.none
        ):
            self._draw_dff_trace()
        imgui.end_child()

        imgui.separator()

        if imgui.begin_child(
            "HistSection", imgui.ImVec2(-1, 0), imgui.ChildFlags_.none
        ):
            self._draw_filter_histograms()
        imgui.end_child()

    def _draw_dff_trace(self):
        """Draw dF/F trace for selected ROI."""
        visible = self.visible_indices
        if len(visible) == 0 or self._dff is None:
            imgui.text("No data")
            return

        roi_idx = visible[self.selected_roi]

        imgui.text(f"dF/F - ROI {roi_idx}")
        imgui.same_line()
        if self._snr is not None:
            imgui.text(f"| SNR:{self._snr[roi_idx]:.1f}")
            imgui.same_line()
        if self._skewness is not None:
            imgui.text(f"Skew:{self._skewness[roi_idx]:.2f}")
            imgui.same_line()
        if self._activity is not None:
            imgui.text(f"Act:{self._activity[roi_idx]:.1%}")

        dff_trace = self._dff[roi_idx].astype(np.float64)
        n_frames = len(dff_trace)
        xs = np.arange(n_frames, dtype=np.float64)

        avail = imgui.get_content_region_avail()
        plot_height = max(60, avail.y - 5)

        if implot.begin_plot("##dFF", imgui.ImVec2(-1, plot_height)):
            implot.setup_axes("Frame", "dF/F")
            implot.plot_line("dF/F", xs, dff_trace)
            implot.end_plot()

    def _draw_filter_histograms(self):
        """Draw histograms with filter sliders."""
        if self._snr is None:
            return

        imgui.text("Filter Thresholds (adjust to modify cell classification)")

        avail = imgui.get_content_region_avail()
        hist_height = max(80, (avail.y - 60) / 4)

        filters_changed = False

        filters_changed |= self._draw_histogram_with_slider(
            "SNR",
            self._snr,
            hist_height,
            "_filter_snr_min",
            self._snr_min,
            self._snr_max,
            is_min_filter=True,
        )

        filters_changed |= self._draw_histogram_with_slider(
            "Shot Noise",
            self._shot_noise,
            hist_height,
            "_filter_shot_noise_max",
            self._shot_noise_min,
            self._shot_noise_max,
            is_min_filter=False,
        )

        filters_changed |= self._draw_histogram_with_slider(
            "Skewness",
            self._skewness,
            hist_height,
            "_filter_skewness_min",
            self._skewness_min,
            self._skewness_max,
            is_min_filter=True,
        )

        filters_changed |= self._draw_histogram_with_slider(
            "Activity %",
            self._activity * 100,
            hist_height,
            "_filter_activity_min",
            self._activity_min * 100,
            self._activity_max * 100,
            is_min_filter=True,
            scale=100,
        )

        if filters_changed:
            self._apply_filters()

    def _draw_histogram_with_slider(
        self,
        label: str,
        data: np.ndarray,
        height: float,
        filter_attr: str,
        data_min: float,
        data_max: float,
        is_min_filter: bool,
        scale: float = 1.0,
    ) -> bool:
        """Draw a histogram with an integrated threshold slider.

        Returns True if filter value changed.
        """
        if data is None or len(data) == 0:
            return False

        current_val = getattr(self, filter_attr)
        if scale != 1.0:
            current_val = current_val * scale

        n_bins = 50
        hist_data = data * scale if scale != 1.0 else data
        hist_min = data_min * scale if scale != 1.0 else data_min
        hist_max = data_max * scale if scale != 1.0 else data_max

        if hist_max <= hist_min:
            hist_max = hist_min + 1

        counts, edges = np.histogram(hist_data, bins=n_bins, range=(hist_min, hist_max))
        bin_centers = (edges[:-1] + edges[1:]) / 2

        avail_width = imgui.get_content_region_avail().x
        content_width = max(200, avail_width - 8)

        filter_type = "Min" if is_min_filter else "Max"
        imgui.text(f"{label} ({filter_type}: {current_val:.2f})")

        if implot.begin_plot(
            f"##{label}",
            imgui.ImVec2(content_width, height - 25),
            implot.Flags_.no_legend,
        ):
            implot.setup_axes(
                "",
                "",
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
                implot.AxisFlags_.auto_fit,
            )

            bar_width = (hist_max - hist_min) / n_bins * 0.9
            implot.plot_bars(
                label,
                bin_centers.astype(np.float64),
                counts.astype(np.float64),
                bar_width,
            )

            threshold_x = np.array([current_val, current_val], dtype=np.float64)
            threshold_y = np.array([0, np.max(counts) * 1.1], dtype=np.float64)
            implot.plot_line(
                "threshold",
                threshold_x,
                threshold_y,
                implot.Spec(line_color=imgui.ImVec4(1.0, 0.3, 0.3, 1.0)),
            )

            if is_min_filter:
                shade_x = np.array(
                    [hist_min, current_val, current_val, hist_min], dtype=np.float64
                )
            else:
                shade_x = np.array(
                    [current_val, hist_max, hist_max, current_val], dtype=np.float64
                )
            shade_y = np.array(
                [0, 0, np.max(counts) * 1.1, np.max(counts) * 1.1], dtype=np.float64
            )
            implot.plot_shaded(
                "excluded",
                shade_x[:2],
                shade_y[:2],
                shade_y[2:4],
                implot.Spec(fill_color=imgui.ImVec4(1.0, 0.3, 0.3, 0.2)),
            )

            implot.end_plot()

        imgui.set_next_item_width(content_width)
        slider_label = f"##{label}_slider"
        changed, new_val = imgui.slider_float(
            slider_label, current_val, hist_min, hist_max, "%.2f"
        )

        if changed:
            if scale != 1.0:
                new_val = new_val / scale
            setattr(self, filter_attr, new_val)
            return True

        imgui.spacing()
        return False

    def _save_iscell(self):
        """Save modified iscell.npy to disk."""
        if self.iscell is None:
            return

        save_path = self._save_path
        if save_path is None:
            return

        iscell_path = save_path / "iscell.npy"
        try:
            np.save(iscell_path, self.iscell)
            self._last_save_time = time.time()
            self._last_iscell_mtime = iscell_path.stat().st_mtime
            self._save_status_msg = "Saved! Reload in Suite2p"
            self._save_status_time = time.time()
        except Exception as e:
            self._save_status_msg = f"Save error: {e}"
            self._save_status_time = time.time()

    def _export_for_training(self):
        """Export current iscell + stat for classifier training."""
        if self.stat is None or self.iscell is None:
            self._save_status_msg = "No data loaded"
            self._save_status_time = time.time()
            return

        save_path = self._save_path
        if save_path is None:
            return

        training_dir = save_path / "classifier_training"
        training_dir.mkdir(exist_ok=True)

        keys = [
            "npix_norm",
            "compact",
            "skew",
        ]  # suite2p's default classifier features
        n_rois = len(self.stat)

        stats = np.zeros((n_rois, len(keys)), dtype=np.float32)
        for i, s in enumerate(self.stat):
            for j, k in enumerate(keys):
                stats[i, j] = s.get(k, 0.0)

        if self.iscell.ndim == 2:
            iscell_labels = self.iscell[:, 0].astype(np.float32)
        else:
            iscell_labels = self.iscell.astype(np.float32)

        training_data = {
            "stats": stats,
            "iscell": iscell_labels,
            "keys": keys,
            "source_path": str(save_path),
        }

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_path = training_dir / f"training_{timestamp}.npy"
        np.save(export_path, training_data)

        self._training_dir = training_dir
        self._exported_datasets.append(export_path)
        self._save_status_msg = f"Exported: {export_path.name}"
        self._save_status_time = time.time()
        print(f"Training data exported to: {export_path}")

    def _mark_as_curated(self):
        """Mark current iscell.npy as manually curated (copy to training folder)."""
        save_path = self._save_path
        if save_path is None:
            return

        training_dir = save_path / "classifier_training"
        training_dir.mkdir(exist_ok=True)

        for fname in ["iscell.npy", "stat.npy"]:
            src = save_path / fname
            if src.exists():
                dst = training_dir / fname
                shutil.copy(src, dst)

        self._training_dir = training_dir
        self._save_status_msg = "Marked as curated"
        self._save_status_time = time.time()
        print(f"Curated data saved to: {training_dir}")

    def _build_classifier(self):
        """Build classifier from exported training datasets."""
        if len(self._exported_datasets) == 0:
            self._save_status_msg = "No training data exported"
            self._save_status_time = time.time()
            return

        all_stats = []
        all_iscell = []
        keys = None

        for path in self._exported_datasets:
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=True).item()
            all_stats.append(data["stats"])
            all_iscell.append(data["iscell"])
            if keys is None:
                keys = data["keys"]

        if len(all_stats) == 0:
            self._save_status_msg = "No valid training data found"
            self._save_status_time = time.time()
            return

        combined_stats = np.vstack(all_stats)
        combined_iscell = np.concatenate(all_iscell)

        save_path = self._save_path
        if save_path is None:
            return

        classifier_path = save_path / "custom_classifier.npy"
        model = {
            "stats": combined_stats,
            "iscell": combined_iscell,
            "keys": keys,
        }
        np.save(classifier_path, model)

        n_cells = int(combined_iscell.sum())
        n_total = len(combined_iscell)
        self._save_status_msg = f"Classifier: {n_cells}/{n_total} cells"
        self._save_status_time = time.time()
        print(f"Classifier saved to: {classifier_path}")
        print(f"Training data: {n_cells} cells, {n_total - n_cells} non-cells")

    def _draw_stats_popup(self):
        """Draw ROI statistics popup."""
        if self._show_stats_popup:
            self._stats_popup_open = True
            self._stats_sizer.before_open()
            imgui.open_popup("ROI Statistics")
            self._show_stats_popup = False

        imgui.set_next_window_size(imgui.ImVec2(400, 450), imgui.Cond_.first_use_ever)

        opened, visible = imgui.begin_popup_modal(
            "ROI Statistics",
            p_open=True if self._stats_popup_open else None,
            flags=imgui.WindowFlags_.no_saved_settings,
        )

        if opened:
            if not visible:
                self._stats_popup_open = False
                imgui.close_current_popup()
            else:
                visible_indices = self.visible_indices
                if len(visible_indices) == 0:
                    imgui.text("No ROIs selected")
                else:
                    roi_idx = visible_indices[self.selected_roi]
                    s = self.stat[roi_idx]

                    imgui.text(f"ROI {roi_idx}")
                    imgui.separator()

                    if imgui.begin_child(
                        "StatsScroll", imgui.ImVec2(0, -35), imgui.ChildFlags_.borders
                    ):
                        if imgui.collapsing_header(
                            "Classification", imgui.TreeNodeFlags_.default_open
                        ):
                            if self.iscell is not None:
                                prob = (
                                    self.iscell[roi_idx, 0]
                                    if self.iscell.ndim > 1
                                    else self.iscell[roi_idx]
                                )
                                orig_prob = (
                                    self.iscell_original[roi_idx, 0]
                                    if self.iscell_original.ndim > 1
                                    else self.iscell_original[roi_idx]
                                )
                                color = (
                                    imgui.ImVec4(0.2, 1.0, 0.2, 1.0)
                                    if prob > 0.5
                                    else imgui.ImVec4(1.0, 0.5, 0.2, 1.0)
                                )
                                imgui.text("Current probability:")
                                imgui.same_line()
                                imgui.text_colored(color, f"{prob:.4f}")
                                imgui.text(f"Original probability: {orig_prob:.4f}")

                        if imgui.collapsing_header(
                            "Morphology", imgui.TreeNodeFlags_.default_open
                        ):
                            ypix = s.get("ypix", [])
                            xpix = s.get("xpix", [])
                            imgui.text(f"Pixels: {len(ypix)}")
                            if len(ypix) > 0:
                                imgui.text(
                                    f"Center: ({np.mean(xpix):.1f}, {np.mean(ypix):.1f})"
                                )
                            imgui.text(f"Radius: {s.get('radius', 0):.2f}")
                            imgui.text(f"Aspect ratio: {s.get('aspect_ratio', 0):.2f}")

                        if imgui.collapsing_header(
                            "Signal Metrics", imgui.TreeNodeFlags_.default_open
                        ):
                            if self._snr is not None:
                                imgui.text(f"SNR: {self._snr[roi_idx]:.4f}")
                            if self._shot_noise is not None:
                                imgui.text(
                                    f"Shot noise: {self._shot_noise[roi_idx]:.2f}"
                                )
                            if self._skewness is not None:
                                imgui.text(f"Skewness: {self._skewness[roi_idx]:.4f}")
                            if self._activity is not None:
                                imgui.text(f"Activity: {self._activity[roi_idx]:.2%}")
                            if self._dff is not None:
                                dff = self._dff[roi_idx]
                                imgui.text(
                                    f"dF/F range: [{np.min(dff):.3f}, {np.max(dff):.3f}]"
                                )
                                imgui.text(f"dF/F std: {np.std(dff):.4f}")

                        if imgui.collapsing_header(
                            "Classifier Features", imgui.TreeNodeFlags_.default_open
                        ):
                            npix_norm = s.get("npix_norm", None)
                            compact = s.get("compact", None)
                            skew_val = s.get("skew", None)

                            if npix_norm is not None:
                                imgui.text(f"npix_norm: {npix_norm:.4f}")
                            else:
                                imgui.text_disabled("npix_norm: not computed")

                            if compact is not None:
                                imgui.text(f"compact: {compact:.4f}")
                            else:
                                imgui.text_disabled("compact: not computed")

                            if skew_val is not None:
                                imgui.text(f"skew: {skew_val:.4f}")
                            else:
                                imgui.text_disabled("skew: not computed")

                            imgui.spacing()
                            imgui.text_wrapped(
                                "these 3 features are used by suite2p's default classifier"
                            )

                        if imgui.collapsing_header("stat.npy values"):
                            for key, value in sorted(s.items()):
                                if key in ("ypix", "xpix", "lam"):
                                    imgui.text_disabled(f"{key}: [{len(value)} values]")
                                elif isinstance(
                                    value, (int, float, np.integer, np.floating)
                                ):
                                    if isinstance(value, float):
                                        imgui.text(f"{key}: {value:.4f}")
                                    else:
                                        imgui.text(f"{key}: {value}")
                                elif isinstance(value, np.ndarray) and value.size <= 10:
                                    imgui.text(f"{key}: {value}")
                                else:
                                    imgui.text_disabled(
                                        f"{key}: [shape {getattr(value, 'shape', 'N/A')}]"
                                    )

                    imgui.end_child()

                imgui.spacing()
                if imgui.button("Close", imgui.ImVec2(100, 0)):
                    self._stats_popup_open = False
                    imgui.close_current_popup()

            imgui.end_popup()
