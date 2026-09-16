"""Demixing tab: the ROIs of an open masknmf demixing result.

A run writes one result file per channel and pass
(``calcium_spine_demixing.hdf5``, ``glutamate_spine_demixing.hdf5``) or one
``demixing_results.hdf5`` per plane. The tab switches between them without
leaving the viewer, lists every ROI with its class label and trace, and
paints footprints over the image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui, imgui_ctx, implot

from mbo_utilities.arrays.demixing import (
    VIEW_LABELS,
    VIEWS,
    DemixingArray,
    list_demixing_results,
)
from mbo_utilities.gui._imgui_helpers import set_tooltip, style_seaborn_dark
from mbo_utilities.gui.widgets._base import Widget

_ACCENT = imgui.ImVec4(0.8, 0.8, 0.2, 1.0)
_ERROR = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
OVERLAY_NAME = "demixing_footprints"

# (header, stretch weight)
ROI_COLUMNS = (
    ("roi", 0.6),
    ("label", 1.0),
    ("cell", 0.5),
    ("peak", 0.9),
    ("mean", 0.9),
)

# rgb per class label, cycling; the selected ROI is drawn in _SELECTED
_PALETTE = np.array(
    [
        [80, 200, 255],
        [255, 120, 140],
        [120, 255, 120],
        [220, 140, 255],
        [255, 170, 60],
        [90, 230, 210],
    ],
    dtype=np.uint8,
)
_SELECTED = np.array([255, 255, 90], dtype=np.uint8)


def demixing_array_of(obj):
    """The `DemixingArray` behind a viewer array, or None.

    Peels the display wrappers (`_SqueezeSingletonDims`, `_ScrubTimingProxy`,
    …), each of which exposes its source as ``_arr``.
    """
    for _ in range(8):
        if isinstance(obj, DemixingArray):
            return obj
        nxt = getattr(obj, "_arr", None)
        if nxt is None or nxt is obj:
            return None
        obj = nxt
    return None


def roi_rows(arr: DemixingArray) -> list[tuple[tuple[str, ...], tuple]]:
    """One table row per ROI.

    The cell texts and the sort keys, both in ROI_COLUMNS order (numbers sort
    as numbers).
    """
    traces = arr.traces
    peaks = traces.max(axis=0)
    means = traces.mean(axis=0)
    labels = arr.roi_labels
    rows = []
    for k in range(arr.num_rois):
        cell = bool(arr.iscell[k])
        cells = (
            str(k),
            labels[k],
            "yes" if cell else "no",
            f"{peaks[k]:.3g}",
            f"{means[k]:.3g}",
        )
        rows.append((cells, (k, labels[k], cell, float(peaks[k]), float(means[k]))))
    return rows


def footprint_rgba(arr: DemixingArray, rois, selected: int | None) -> np.ndarray:
    """An ``(Y, X, 4)`` uint8 overlay.

    ``rois`` are coloured by class label, the selected ROI on top in the
    highlight colour.
    """
    ny, nx = arr.shape[3:]
    rgba = np.zeros((ny * nx, 4), dtype=np.uint8)
    footprints = arr.footprints
    order = [k for k in rois if k != selected] + ([selected] if selected is not None else [])
    for k in order:
        col = footprints.getcol(k)
        if col.nnz == 0:
            continue
        weight = col.data / col.data.max()
        color = _SELECTED if k == selected else _PALETTE[int(arr.class_labels[k]) % len(_PALETTE)]
        rgba[col.indices, :3] = color
        rgba[col.indices, 3] = (60 + 160 * weight).astype(np.uint8)
    return rgba.reshape(ny, nx, 4)


class DemixingTabWidget(Widget):
    """The Demixing tab.

    Channel switch, ROI table with the selected trace, and footprints painted
    over the image.
    """

    name = "Demixing"
    tab_label = "Demixing"
    placement = "tab"
    toggle_key = "demixing"
    priority = 16

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._cache: dict[Path, DemixingArray] = {}
        self._siblings: list[dict] = []
        self._siblings_for: Path | None = None
        self._rows: list = []
        self._rows_for: DemixingArray | None = None
        self._selected: int | None = None
        self._show_all = False
        self._sort = (0, True)
        self._error: str | None = None
        self._overlay = None
        self._overlay_shape: tuple[int, int] | None = None

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        iw = getattr(parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        return bool(len(data)) and demixing_array_of(data[0]) is not None

    @property
    def _array(self) -> DemixingArray | None:
        iw = getattr(self.parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        return demixing_array_of(data[0]) if len(data) else None

    def _switch(self, entry: dict) -> None:
        """Show another result file of the run in the viewer."""
        from mbo_utilities.gui._dialogs import swap_viewer_array

        self._error = None
        path = Path(entry["path"])
        try:
            arr = self._cache.get(path)
            if arr is None:
                arr = DemixingArray(path)
                self._cache[path] = arr
            self._drop_overlay()
            self._selected = None
            swap_viewer_array(self.parent, arr, title=entry["label"])
            self.parent.logger.info(f"demixing result: {path}  shape={arr.shape}")
        except Exception as e:
            self._error = str(e)
            self.parent.logger.exception(f"demixing switch failed: {e}")

    def _paint(self, arr: DemixingArray) -> None:
        """Repaint the footprint overlay for the current selection."""
        figure = getattr(self.parent.image_widget, "figure", None)
        if figure is None:
            return
        subplot = figure[0, 0]
        ny, nx = arr.shape[3:]
        if self._overlay is not None and self._overlay_shape != (ny, nx):
            self._drop_overlay()
        if self._overlay is None:
            self._overlay_shape = (ny, nx)
            self._overlay = subplot.add_image(
                np.zeros((ny, nx, 4), np.uint8),
                name=OVERLAY_NAME,
                alpha_mode="blend",
                offset=(0, 0, 1),
            )
            # literal RGBA bytes: auto-ranging off the all-zero start saturates them
            self._overlay.vmin, self._overlay.vmax = 0, 255
            for tile in self._overlay.world_object.children:
                tile.material.pick_write = False
        rois = range(arr.num_rois) if self._show_all else ()
        self._overlay.data = footprint_rgba(arr, rois, self._selected)
        self._overlay.visible = self._show_all or self._selected is not None

    def _drop_overlay(self) -> None:
        if self._overlay is None:
            return
        try:
            self.parent.image_widget.figure[0, 0].delete_graphic(self._overlay)
        except (KeyError, ValueError, AttributeError):
            pass
        self._overlay = None

    def draw(self) -> None:
        with imgui_ctx.begin_child(
            "##DemixingContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            arr = self._array
            if arr is None:
                imgui.text_disabled("No demixing result is open.")
                return
            path = arr.filenames[0]
            self._cache.setdefault(path, arr)
            if self._siblings_for != path.parent:
                self._siblings = list_demixing_results(path)
                self._siblings_for = path.parent
            if self._rows_for is not arr:
                self._rows = roi_rows(arr)
                self._rows_for = arr

            nt, _, _, ny, nx = arr.shape
            imgui.text_colored(_ACCENT, path.name)
            imgui.same_line(0, 12)
            detail = f"{arr.num_rois} ROIs · {nt} timepoints · {ny} x {nx} px"
            fs = arr.metadata.get("fs")
            if fs:
                detail += f" · {fs:.2f} Hz"
            imgui.text_disabled(detail)

            labels = [e["label"] for e in self._siblings]
            current = next(
                (i for i, e in enumerate(self._siblings) if Path(e["path"]) == path), 0
            )
            if labels:
                imgui.set_next_item_width(imgui.get_content_region_avail().x * 0.5)
                changed, new_idx = imgui.combo("Channel", current, labels)
                set_tooltip(
                    "Result files of this run: one per channel and pass "
                    "(calcium / glutamate, spine / global activity), or one per "
                    "plane. Switching keeps the viewer open."
                )
                if changed and new_idx != current:
                    self._switch(self._siblings[new_idx])
                    return  # the array under us changed; redraw next frame

            imgui.text_disabled(
                "View slider: " + " · ".join(f"{i} {v}" for i, v in enumerate(VIEWS))
            )
            set_tooltip("\n".join(f"{i} {v}: {VIEW_LABELS[v]}" for i, v in enumerate(VIEWS)))

            changed, self._show_all = imgui.checkbox("All footprints", self._show_all)
            set_tooltip("Paint every ROI's footprint over the image, coloured by class label.")
            if changed:
                self._paint(arr)
            imgui.same_line(0, 12)
            if imgui.button("Curation GUI"):
                from mbo_utilities.gui.run_gui import _open_curation_gui

                self._error = None
                try:
                    _open_curation_gui(path)
                except Exception as e:
                    self._error = str(e)
                    self.parent.logger.exception(f"curation GUI failed: {e}")
            set_tooltip(
                "masknmf's accept / reject and class-label window for this file. "
                "Labels save to <file>.labels.hdf5 beside it."
            )
            if self._error:
                imgui.text_colored(_ERROR, "Failed")
                set_tooltip(self._error)

            k = self._selected
            if k is not None and implot.begin_plot("##demixing_trace", imgui.ImVec2(-1, 150)):
                try:
                    style_seaborn_dark()
                    implot.setup_axes(
                        "time (s)" if fs else "timepoint",
                        "c",
                        implot.AxisFlags_.auto_fit.value,
                        implot.AxisFlags_.auto_fit.value,
                    )
                    trace = np.asarray(arr.traces[:, k], dtype=np.float64)
                    implot.plot_line(f"ROI {k}", trace, xscale=1.0 / fs if fs else 1.0)
                finally:
                    implot.end_plot()

            flags = (
                imgui.TableFlags_.sortable | imgui.TableFlags_.row_bg
                | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.scroll_y
                | imgui.TableFlags_.resizable | imgui.TableFlags_.sizing_stretch_prop
            )
            avail = imgui.get_content_region_avail()
            if not imgui.begin_table(
                "##demixing_rois", len(ROI_COLUMNS), flags, imgui.ImVec2(0, avail.y)
            ):
                return
            imgui.table_setup_scroll_freeze(0, 1)
            for i, (name, weight) in enumerate(ROI_COLUMNS):
                column_flags = imgui.TableColumnFlags_.width_stretch
                if i == 0:
                    column_flags |= imgui.TableColumnFlags_.default_sort
                imgui.table_setup_column(name, column_flags, weight)
            imgui.table_headers_row()
            set_tooltip("Click a row to plot its trace and paint its footprint", show_mark=False)
            specs = imgui.table_get_sort_specs()
            if specs is not None and specs.specs_dirty:
                if specs.specs_count > 0:
                    self._sort = (
                        int(specs.specs.column_index),
                        specs.specs.sort_direction == imgui.SortDirection.ascending,
                    )
                specs.specs_dirty = False
            column, ascending = self._sort
            rows = sorted(self._rows, key=lambda row: row[1][column], reverse=not ascending)
            picked = None
            clipper = imgui.ListClipper()
            clipper.begin(len(rows))
            while clipper.step():
                for r in range(clipper.display_start, clipper.display_end):
                    cells, keys = rows[r]
                    imgui.table_next_row()
                    imgui.table_next_column()
                    clicked, _ = imgui.selectable(
                        f"{cells[0]}##roi_{keys[0]}",
                        keys[0] == self._selected,
                        imgui.SelectableFlags_.span_all_columns,
                    )
                    if clicked:
                        picked = keys[0]
                    for i in range(1, len(cells)):
                        if imgui.table_next_column():
                            imgui.text(cells[i])
            imgui.end_table()

            if picked is not None:
                # clicking the selected row clears the selection
                self._selected = None if picked == self._selected else picked
                self._paint(arr)

    def cleanup(self) -> None:
        self._drop_overlay()
        for arr in self._cache.values():
            arr.close()
        self._cache = {}
