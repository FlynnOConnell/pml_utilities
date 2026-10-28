"""The ROIs tab: labels, on-disk and exchange rows, the Curation and ROIs sub-tabs."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
from imgui_bundle import (
    icons_fontawesome_6 as fa,
)
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd
from masknmf.visualization.imgui import (
    button_colors,
    draw_help_buttons,
    draw_range_filter,
    draw_roi_table,
    draw_switch,
    grid,
    help_buttons_width,
    help_mark,
    right_aligned_text,
    section,
    switch_width,
    tooltip,
)
from masknmf.visualization.imgui import THEME as MTHEME
from masknmf.visualization.rois import MARKED_COLOR

from mbo_utilities.annotation import (
    ENGINES,
    UNLABELED,
)
from mbo_utilities.annotation.imagej import export_roiset, export_traces, import_roiset
from mbo_utilities.gui._theme import THEME, em, label_button, to_vec4
from mbo_utilities.gui.imgui import (
    UNLABEL_ALL,
    draw_label_editor,
)
from mbo_utilities.roi_workflow import (
    roi_folder,
)

from mbo_utilities.gui.roi._shared import (
    _CAPTIONS,
    COLOR_BY,
    COLORMAPS,
    DEFAULT_RUN_TAG,
    ENGINE_HELP,
    ROI_KEYS,
    RUN_WHERE,
)


class CurationMixin:
    """The ROIs tab: labels, on-disk and exchange rows, the Curation and ROIs sub-tabs."""

    def draw_rois(self):
        """The ROIs tab: masknmf's Tools panel. The Full FOV button and the
        guide and keybinds buttons, then the Curation and ROIs tabs, each a
        child that scrolls on its own.
        """
        self._note_hover()
        if imgui.button(f"{fa.ICON_FA_IMAGE} Full FOV"):
            self.open_full_fov()
        tooltip(
            "the frame on screen and the ROI labels at full size: zoom, colormap, contrast, pixel values"
        )
        buttons_w = help_buttons_width("ROI Guide")
        imgui.same_line()
        if imgui.get_content_region_avail().x < buttons_w + em(0.6):
            imgui.new_line()
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - buttons_w
        )
        guide, self.keybinds_open = draw_help_buttons(
            False, self.keybinds_open, "ROI Guide"
        )
        if guide:
            self.open_guide()
        if imgui.begin_tab_bar("##roi_tools"):
            if imgui.begin_tab_item("Curation")[0]:
                imgui.begin_child("##roi_curation_tab")
                self._draw_curation()
                imgui.end_child()
                imgui.end_tab_item()
            if imgui.begin_tab_item("ROIs")[0]:
                imgui.begin_child("##roi_table_tab")
                self.draw_tab()
                imgui.end_child()
                imgui.end_tab_item()
            imgui.end_tab_bar()

    def _draw_labels(self, g):
        section("LABELS")
        g.row("new label")
        self.new_label, changed = draw_label_editor(
            self.classes, self.new_label, "_roi"
        )
        if changed:
            self._sync_store_from_classes()
            self._resync()
            self.refresh_overlay()
            self._autosave()
        if self.n_rois:
            done = int((self.classes.labels[: self.n_rois] >= 0).sum())
            g.row("labeled")
            imgui.text_colored(
                to_vec4(THEME.ok if done == self.n_rois else THEME.warn),
                f"{done} / {self.n_rois}",
            )
            help_mark("drawn ROIs with a label; u jumps to the next one without")
        if not self.classes.names:
            return
        g.row("classes")
        imgui.new_line()
        picked = self._draw_label_columns()
        if picked == UNLABEL_ALL:
            self.unlabel_all()
        elif picked is not None:
            self.assign_class(picked)

    def _draw_label_columns(self):
        """The unlabel actions pinned across the top, then one button per
        class, split into columns filled evenly.

        Class buttons carry two columns of their own, the count then the
        name, each left-aligned, so the names line up down the column
        instead of drifting with however wide the counts happen to be.
        """
        if not self.classes.names:
            return None
        picked = self._draw_unlabel_row()
        n = len(self.classes.names)
        gap, hint = em(0.8), em(2.0)
        # as many columns as the width holds at a readable button, three at most
        ncols = max(
            1,
            min(n, 3, int((imgui.get_content_region_avail().x + gap) // (em(8) + gap))),
        )
        per_col = -(-n // ncols)
        col_w = max(
            (imgui.get_content_region_avail().x - gap * (ncols - 1)) / ncols, em(5)
        )
        size = imgui.ImVec2(max(col_w - hint - em(0.7), em(3.0)), 0)
        count_w = max(
            imgui.calc_text_size(self._count_text(i)).x for i in range(n)
        ) + em(0.5)
        for c0 in range(0, n, per_col):
            if c0:
                imgui.same_line(0, gap)
            imgui.begin_group()
            for i in range(c0, min(c0 + per_col, n)):
                with label_button(self.classes.color(i)):
                    if imgui.button(f"##lab{i}", size):
                        picked = i
                self._draw_label_button_text(i, count_w)
                if i < 9:
                    imgui.same_line(0, 4)
                    imgui.text_disabled(f"({i + 1})")
            imgui.end_group()
        return picked

    def _count_text(self, i: int) -> str:
        """The count column of a class button. "n=3", not "(3)": the (1-9)
        that follows the button is the keybind, and a bare count beside it
        read as one too.
        """
        return f"n={self.classes.count(i)}"

    def _draw_label_button_text(self, i: int, count_w: float) -> None:
        """Paint one class button's two text columns over the button just
        drawn. imgui centres a button's own label, which left the names
        starting at a different x on every row.
        """
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        pad = imgui.get_style().frame_padding.x
        y = lo.y + (hi.y - lo.y - imgui.get_text_line_height()) * 0.5
        draw = imgui.get_window_draw_list()
        color = imgui.get_color_u32(imgui.Col_.text)
        # a long name is clipped to its button rather than bleeding into the
        # column beside it
        draw.push_clip_rect(
            imgui.ImVec2(lo.x, lo.y), imgui.ImVec2(hi.x - 1, hi.y), True
        )
        draw.add_text(imgui.ImVec2(lo.x + pad, y), color, self._count_text(i))
        draw.add_text(
            imgui.ImVec2(lo.x + pad + count_w, y), color, self.classes.names[i]
        )
        draw.pop_clip_rect()

    def _draw_unlabel_row(self):
        """Unlabel at the top left, unlabel all at the top right: two small
        buttons that stay put however many classes there are.
        """
        picked = None
        x0 = imgui.get_cursor_pos_x()
        avail = imgui.get_content_region_avail().x
        if imgui.small_button("unlabel##_roi"):
            picked = UNLABELED
        tooltip("Clear the selected ROI's label (0)")
        pad = imgui.get_style().frame_padding.x * 2
        right_w = imgui.calc_text_size("unlabel all").x + pad
        imgui.same_line()
        imgui.set_cursor_pos_x(max(x0 + avail - right_w, imgui.get_cursor_pos_x()))
        with button_colors(MTHEME.danger, MTHEME.danger_hover):
            if imgui.small_button("unlabel all##_roi"):
                picked = UNLABEL_ALL
        tooltip("Clear every label on this plane")
        return picked

    def _status_message(self) -> tuple[tuple, str]:
        if self._save_error is not None:
            return THEME.err, self._save_error
        if self._run_error is not None:
            return THEME.err, self._run_error
        active = self.manager.active
        if active:
            verbs = {"extract": "extract", "demix": "masknmf"}
            names = ", ".join(f"{verbs.get(r.kind, r.kind)} {r.tag}" for r in active)
            return THEME.warn, f"{len(active)} running: {names}"
        return THEME.text_dim, self.status

    def _draw_save_controls(self, g):
        """The on disk row: autosave, and the folder Save and autosave write to."""
        g.row("on disk")
        imgui.begin_disabled(self.fpath is None)
        changed, on = imgui.checkbox("autosave##roi_autosave", self.autosave)
        if changed:
            self.set_autosave(on)
        tooltip(
            "Write ROIs and loaded runs as they change. Off: nothing is "
            "written until Save. Applies to every dataset."
        )
        imgui.same_line(0, g.gap)
        if imgui.button(f"{fa.ICON_FA_FOLDER_OPEN}##roi_folder"):
            start = str(self.root or roi_folder(self.fpath))
            self._folder_dialog = pfd.select_folder("Save ROIs under", start)
        tooltip(
            "Pick a folder to save ROIs and runs under, one folder per dataset "
            "inside it, instead of beside the data"
        )
        if self.root is not None:
            imgui.same_line(0, g.gap)
            if imgui.button("beside data##roi_folder"):
                self.set_folder(None)
            tooltip("Save beside the data again")
        imgui.end_disabled()
        self._draw_exchange_buttons(g)
        if self.fpath is None:
            imgui.text_disabled("in memory only: no data path")
            return
        imgui.text_disabled(f"to {self.folder}")
        if not imgui.is_item_hovered():
            return
        imgui.begin_tooltip()
        imgui.text(f"rois zarr: {self._save_target()}")
        imgui.text(
            "runs:      <stem>.<tag>.<stamp>.manual.zarr beside it, listed in the rois zarr"
        )
        imgui.text_colored(
            to_vec4(THEME.code),
            "from mbo_utilities.annotation import RoiFile\n"
            f'rois = RoiFile.load(r"{self._save_target()}")\n'
            "rois.store.rois            # per-ROI pixels, weights, plane, class, note, uid\n"
            "rois.store.label_image(z)  # (Y, X); 0 = bg, ROI i = i + 1\n"
            "rois.traces                # the quick traces and pixel averages",
        )
        imgui.end_tooltip()

    def _draw_exchange_buttons(self, g):
        """Fiji and CSV exchange: export the drawn ROIs as a RoiSet.zip,
        import one, export the trace table's rows as a CSV.
        """
        g.row("exchange")
        start = str(self.folder if self.fpath is not None else Path.home())
        busy = self._file_dialog is not None
        imgui.begin_disabled(self.n_rois == 0 or busy)
        if imgui.button("export RoiSet##roi_export_roiset"):
            self._file_dialog = (
                "export_roiset",
                pfd.save_file(
                    "Export ROIs for Fiji", f"{start}/RoiSet.zip", ["RoiSet", "*.zip"]
                ),
            )
        tooltip(
            "Every drawn ROI as a RoiSet.zip Fiji's ROI Manager opens: its "
            "polygon as drawn, on its z-plane and channel"
        )
        imgui.end_disabled()
        imgui.same_line(0, g.gap)
        imgui.begin_disabled(busy)
        if imgui.button("import RoiSet##roi_import_roiset"):
            self._file_dialog = (
                "import_roiset",
                pfd.open_file(
                    "Import ROIs from Fiji", start, ["RoiSet", "*.zip *.roi"]
                ),
            )
        tooltip(
            "Add the ROIs of a RoiSet.zip (or one .roi), each on the plane it names"
        )
        imgui.end_disabled()
        imgui.same_line(0, g.gap)
        imgui.begin_disabled(not self.has_traces() or busy)
        if imgui.button("export traces##roi_export_traces"):
            self._file_dialog = (
                "export_traces",
                pfd.save_file("Export traces", f"{start}/traces.csv", ["CSV", "*.csv"]),
            )
        tooltip(
            "The trace table's rows as one CSV: time (or frame), then one column per row"
        )
        imgui.end_disabled()

    def _poll_folder_dialog(self):
        dialog = self._folder_dialog
        if dialog is not None and dialog.ready():
            self._folder_dialog = None
            picked = dialog.result()
            if picked:
                self.set_folder(picked)
        if self._file_dialog is None or not self._file_dialog[1].ready():
            return
        kind, dialog = self._file_dialog
        self._file_dialog = None
        picked = dialog.result()
        if isinstance(picked, list):
            picked = picked[0] if picked else ""
        if picked:
            self.exchange(kind, picked)

    def exchange(self, kind: str, path) -> None:
        """Run one exchange action on ``path``: ``export_roiset``,
        ``import_roiset`` (adds the ROIs, one undo step) or ``export_traces``
        (the rows the trace table lists). Errors land in the status row.
        """
        path = Path(path)
        try:
            if kind == "export_roiset":
                export_roiset(self.store, path)
                self.status = f"exported {self.n_rois} ROIs to {path}"
            elif kind == "import_roiset":
                self._push_undo()
                added = import_roiset(self.store, path)
                if not added:
                    self._undo.pop()
                self.status = f"imported {len(added)} ROIs from {path.name}"
                if added:
                    self.select_roi(added[-1])
            elif kind == "export_traces":
                rows = [self.traces.get(k) for k in self._sorted_trace_rows()]
                export_traces([t for t in rows if t is not None], path, fs=self.fs())
                self.status = f"exported {len(rows)} traces to {path}"
            else:
                raise ValueError(f"unknown exchange {kind!r}")
        except (OSError, ValueError) as e:
            self._run_error = f"{kind.replace('_', ' ')} failed: {e}"

    def draw_tab(self):
        """Masknmf's Signals tab for ROIs: the filter, then the drawn, algo
        and pixel-average rows in one sortable table, the selection under it.
        """
        if self.order.range_column is not None:
            self._draw_filter()
        footer = imgui.get_frame_height_with_spacing() * 2.5
        if imgui.begin_child("##roi_table", imgui.ImVec2(0, -footer)):
            if self.rows or self.pixels:
                columns = self.columns
                formatters = {
                    name: partial(self._format_cell, name) for name in columns[1:]
                }
                self.scroll_to_selection = draw_roi_table(
                    self.order,
                    columns,
                    formatters,
                    self.scroll_to_selection,
                    table_id="manual_rois",
                    hidden={"ok", "peak"} - ({"ok"} if self.derived else set()),
                    cursor=self._selection_pair() is not None,
                    on_select=self._table_select,
                    is_grouped=self._row_grouped,
                    on_ctrl_select=self._table_ctrl,
                    on_shift_select=self._table_shift,
                    row_color=self._table_color,
                    prefix_rows=[
                        (("px", *pid), f"px {pid[1]},{pid[2]}") for pid in self.pixels
                    ],
                )
            else:
                imgui.text_disabled("no ROIs yet: draw (a) a region and add it (r)")
        imgui.end_child()
        imgui.separator()
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(self._selection_status())
        imgui.pop_text_wrap_pos()

    def open_guide(self):
        """The guide (h): this tool's page of the app's help viewer."""
        from mbo_utilities.gui._help_viewer import ROI_DOC, docs_for

        host = self.host
        if host is None:
            return
        names = [filename for _title, filename in docs_for(host)]
        if ROI_DOC in names:
            host._help_selected_doc = names.index(ROI_DOC)
        host._show_help_popup = True

    def _draw_curation(self):
        """Masknmf's Curation tab: OVERLAY, SELECTION, then this tool's LABELS and RUN."""
        g = grid(_CAPTIONS)
        # sliders: seven tenths of the panel, or what their row has left before the (?) mark
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        slider_w = min(0.7 * imgui.get_window_width(), right - g.cell_x[0] - g.mark_w)
        self._draw_overlay(g, slider_w)
        self._draw_selection()
        self._draw_labels(g)
        self._draw_run(g)
        imgui.spacing()
        imgui.push_text_wrap_pos(0)
        if self.drawing:
            imgui.text_disabled("click to add points; click the first point to close")
        else:
            color, text = self._status_message()
            imgui.text_colored(to_vec4(color), text)
        imgui.pop_text_wrap_pos()
        imgui.separator()
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(self._selection_status())
        imgui.pop_text_wrap_pos()

    def _draw_overlay(self, g, slider_w: float):
        section("OVERLAY")
        rows = (
            ("masks", "show_masks", "opacity", "every ROI's mask at this opacity (m)"),
            (
                "sel masks",
                "show_selected_masks",
                "selected_opacity",
                "the selected and grouped masks at this opacity, with a white rim",
            ),
            (
                "contours",
                "show_contours",
                "contour_opacity",
                "every other ROI's contour at this opacity (c; o rings them)",
            ),
            (
                "sel contours",
                "show_selected_contours",
                "selected_contour_opacity",
                "the selected and grouped contours, in their mask's color at this opacity",
            ),
        )
        for caption, show, opacity, tip in rows:
            changed, on = imgui.checkbox(caption, getattr(self, show))
            if changed:
                setattr(self, show, on)
                self.refresh_overlay()
            g.cell(0)
            imgui.set_next_item_width(slider_w)
            changed, value = imgui.slider_float(
                f"##{opacity}", getattr(self, opacity), 0.05, 1.0, "%.2f"
            )
            if changed:
                setattr(self, opacity, value)
                self.refresh_overlay()
            help_mark(tip)
            if caption == "masks":
                g.row("weighting")
                flipped, self.masks_by_peak = draw_switch(
                    "roi_weighting",
                    self.masks_by_peak,
                    "own peak",
                    "signal peak",
                    self.show_masks,
                )
                if flipped:
                    self.refresh_overlay()
                help_mark(
                    "own peak: every mask solid; signal peak: faint where the ROI's trace is weak",
                    g.cell_x[0] + slider_w + em(0.3),
                )
        g.row("color by")
        imgui.set_next_item_width(g.w)
        changed, index = imgui.combo(
            "##color_by", COLOR_BY.index(self.color_by), list(COLOR_BY)
        )
        if changed:
            self.set_color_by(COLOR_BY[index])
        g.cell(1)
        imgui.begin_disabled(self.color_by == COLOR_BY[0])
        imgui.set_next_item_width(g.w)
        changed, index = imgui.combo(
            "##color_cmap", COLORMAPS.index(self.color_cmap), list(COLORMAPS)
        )
        if changed:
            self.set_color_by(self.color_by, COLORMAPS[index])
        imgui.end_disabled()
        help_mark(
            "color the drawn ROIs by their class, z-plane or channel (a color per level), their area or the peak of "
            "their traces (a gradient); roi id: class / hue colors"
        )
        g.row("traces")
        buttons = (
            (
                fa.ICON_FA_ARROWS_LEFT_RIGHT_TO_LINE,
                self.trace_plot is not None and self.trace_plot.follow,
                "trace_follow",
                "Center: keep the current frame in the middle of the traces as the movie plays or the slider moves, "
                "the zoom kept; near either end of the recording the view stops at that end (t)",
            ),
            (
                fa.ICON_FA_EYE_DROPPER,
                self.pixel_traces,
                "pixel_traces",
                "Quick pixel trace: click an empty pixel to add the movie's 5x5 average there to the plot, grouped "
                "with whatever is shown, and to the top of the ROIs table; delete drops it (p)",
            ),
            (
                fa.ICON_FA_CHART_LINE,
                self.show_traces,
                "show_traces",
                "Show selected traces: plot whatever is selected, every trace of one ROI, or one line per group "
                "member. Off, selecting only highlights, however big the selection",
            ),
            (
                fa.ICON_FA_PENCIL,
                self.auto_trace,
                "auto_trace",
                "Trace on draw: trace every ROI the moment it is added, its mean where the RUN section points it",
            ),
        )
        for i, (icon, on, action, tip) in enumerate(buttons):
            if i:
                imgui.same_line(0, g.gap)
            with button_colors(MTHEME.accent, MTHEME.accent, (0.05, 0.05, 0.05), on=on):
                clicked = imgui.button(
                    f"{icon}##roi_traces_{i}", imgui.ImVec2(em(3.2), 0)
                )
            tooltip(tip)
            if not clicked:
                continue
            if action == "trace_follow" and self.trace_plot is not None:
                self.trace_plot.follow = not self.trace_plot.follow
            elif action == "pixel_traces":
                self.set_pixel_traces(not self.pixel_traces)
            elif action == "show_traces":
                self.show_traces = not self.show_traces
                self._plot_lines_key = None
            elif action == "auto_trace":
                self.auto_trace = not self.auto_trace
        self._draw_save_controls(g)

    def _draw_selection(self):
        """Masknmf's SELECTION: one centered row of icon buttons, the region's side switch under it."""
        section("SELECTION")
        drawing = self.region_selector is not None
        settled = drawing and not self._drawing()
        pair = self._selection_pair()
        derived = [p for p in self.buffer if p[0] >= 0] or (
            [pair] if pair is not None and pair[0] >= 0 else []
        )
        nothing = pair is None and self.active_pixel is None and not self.buffer
        unmark = (
            bool(derived)
            and self.selected < 0
            and self.active_pixel is None
            and all(k in self.derived[si].discarded for si, k in derived)
        )
        gap, avail = em(0.6), imgui.get_content_region_avail().x
        w = min((avail - 6 * gap) / 7, em(3.2))
        size = imgui.ImVec2(w, imgui.get_frame_height() * 1.2)
        imgui.dummy(imgui.ImVec2(0, em(0.4)))
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + max((avail - 7 * w - 6 * gap) / 2, 0)
        )
        with button_colors(
            MTHEME.accent, MTHEME.accent, (0.05, 0.05, 0.05), on=self.follow
        ):
            if imgui.button(f"{fa.ICON_FA_LOCATION_CROSSHAIRS}##roi_center", size):
                self.toggle_follow()
        tooltip(
            "Center: the image on the selected ROI, following it as the selection moves; labeling then advances (f)"
        )
        imgui.same_line(0, gap)
        with button_colors(
            MTHEME.emphasis, MTHEME.emphasis_hover, (0.05, 0.05, 0.05), on=drawing
        ):
            if imgui.button(f"{fa.ICON_FA_DRAW_POLYGON}##roi_draw", size):
                self._start_region()
        tooltip(
            "Draw is on: click again or esc to drop the region, the selection stays (a)"
            if drawing
            else "Draw: a polygon on the image selects every ROI in view whose center is on the switch's side of it, "
            "live as it is drawn and dragged; Add ROI keeps it (a)"
        )
        imgui.same_line(0, gap)
        imgui.begin_disabled(not settled)
        if imgui.button(f"{fa.ICON_FA_PLUS}##roi_add", size):
            self._commit_region()
        imgui.end_disabled()
        tooltip(
            "Add ROI: keep the drawn region as a drawn ROI on the slice on screen (r)"
            + ("" if settled else "; draw one first")
        )
        imgui.same_line(0, gap)
        imgui.begin_disabled(nothing)
        with button_colors(MTHEME.danger, MTHEME.danger_hover, on=not unmark):
            if imgui.button(f"{fa.ICON_FA_TRASH}##roi_delete", size):
                self.delete_selected()
        imgui.end_disabled()
        key = ROI_KEYS["delete"].label
        tooltip(
            f"Unmark: the {len(derived)} selected algo ROI(s) stay ({key})"
            if unmark
            else f"Delete: the selected drawn ROI, or mark the {len(derived)} selected algo ROI(s) for deletion; a "
            f"pixel average is dropped ({key})"
        )
        imgui.same_line(0, gap)
        promotable = (
            self.selected_derived is not None
            and self.promoted_index(*self.selected_derived) is None
        )
        imgui.begin_disabled(not promotable)
        if imgui.button(f"{fa.ICON_FA_ARROW_UP}##roi_promote", size):
            self.promote_derived(*self.selected_derived)
        imgui.end_disabled()
        tooltip(
            "Promote: copy the selected algo ROI into the drawn ROIs, then step to the next (y)"
        )
        imgui.same_line(0, gap)
        imgui.begin_disabled(self.selected_derived is None)
        if imgui.button(f"{fa.ICON_FA_CHECK}##roi_accept", size):
            self.set_accepted(*self.selected_derived)
        imgui.end_disabled()
        tooltip("Accept / reject the selected algo ROI, kept with the ROIs (x)")
        imgui.same_line(0, gap)
        if imgui.button(f"{fa.ICON_FA_FLOPPY_DISK}##roi_save", size):
            self.save()
        tooltip(
            "Save: the drawn ROIs are kept in memory only; there is no data path to save beside"
            if self.fpath is None
            else f"Save: the drawn ROIs and loaded runs to {self.folder} now"
            + ("; they autosave there after every change" if self.autosave else "")
        )
        imgui.dummy(imgui.ImVec2(0, em(0.4)))
        row_w = (
            switch_width("inside", "outside") + em(0.3) + imgui.calc_text_size("(?)").x
        )
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x()
            + max((imgui.get_content_region_avail().x - row_w) / 2, 0)
        )
        flipped, self.region_outside = draw_switch(
            "roi_region", self.region_outside, "inside", "outside", drawing
        )
        if flipped:
            self._region_key = None
        help_mark(
            "which side of the region is selected, lit while it drives the selection"
        )
        if self.selected >= 0 and len(self.buffer) <= 1:
            imgui.set_next_item_width(-1)
            changed, self._note_buf = imgui.input_text_with_hint(
                "##roi_note", "note", self._note_buf
            )
            if changed:
                self.store.set_note(self.selected, self._note_buf)
            if imgui.is_item_deactivated_after_edit():
                self._autosave()

    def _draw_run(self, g):
        """RUN: the engine and where it reads, the run buttons, the tag."""
        section("RUN")
        g.row("engine")
        imgui.set_next_item_width(g.w)
        changed, index = imgui.combo(
            "##roi_engine", ENGINES.index(self.engine), list(ENGINES)
        )
        if changed:
            self.engine = ENGINES[index]
        g.cell(1)
        imgui.set_next_item_width(g.w)
        where = self.run_where if self.run_where in RUN_WHERE else RUN_WHERE[0]
        changed, index = imgui.combo(
            "##roi_where", RUN_WHERE.index(where), ["as drawn", "on screen"]
        )
        if changed:
            self.run_where = RUN_WHERE[index]
        help_mark(f"{ENGINE_HELP[self.engine]}\nReads each mask {self._where_label()}.")
        g.row("run")
        indices = self.selection_indices()
        listed = self.listed_drawn()
        no_path = self.fpath is None
        w = imgui.get_frame_height() * 1.6
        imgui.begin_disabled(no_path or not indices)
        if imgui.button(f"{fa.ICON_FA_PLAY}##roi_run_sel", imgui.ImVec2(w, 0)):
            self.run_selection(indices)
        imgui.end_disabled()
        tooltip(
            "no data path to write beside"
            if no_path
            else f"Run the {len(indices)} selected ROI(s) through {self.engine} -> "
            f"<stem>.{self.effective_tag}.<stamp>.manual.zarr (shift+t)"
            if indices
            else "select a drawn ROI first"
        )
        imgui.same_line(0, g.gap)
        imgui.begin_disabled(no_path or not listed)
        if imgui.button(f"{fa.ICON_FA_FORWARD}##roi_run_all", imgui.ImVec2(w, 0)):
            self.run_in_view()
        imgui.end_disabled()
        tooltip(
            f"Run all {len(listed)} drawn ROIs the table lists through {self.engine}"
        )
        imgui.same_line(0, g.gap)
        why = "no drawn ROIs listed" if not listed else self.trace_disabled(listed[0])
        imgui.begin_disabled(why is not None)
        if imgui.button(f"{fa.ICON_FA_CHART_LINE}##roi_trace_all", imgui.ImVec2(w, 0)):
            self.trace_in_view()
        imgui.end_disabled()
        tooltip(
            why
            or f"Quick trace all {len(listed)} listed ROIs: their means ({self._where_label()})"
        )
        imgui.same_line(0, g.gap)
        marked = sum(len(s.discarded) for s in self.derived)
        right_aligned_text(f"{self.n_rois} drawn, {marked} marked")
        g.row("options")
        imgui.set_next_item_width(g.w)
        _changed, self.run_tag = imgui.input_text_with_hint(
            "##roi_run_tag", DEFAULT_RUN_TAG, self.run_tag
        )
        help_mark(
            "the run's tag: it names the run file, <stem>.<tag>.<stamp>.manual.zarr, "
            "in the ROI folder"
        )

    def _selection_status(self) -> str:
        """Masknmf's footer: what is grouped or selected, else how to start."""
        if len(self.buffer) > 1 or self.pixel_group:
            drawn = sorted(k for si, k in self.buffer if si < 0)
            algo = [self._row_index[p] for p in self.buffer if p[0] >= 0]
            shown = f"{len(drawn)} drawn" if len(drawn) > 12 else f"drawn {drawn}"
            pixels = [f"{r},{c}" for _z, r, c in self.pixel_group]
            return f"{len(self.buffer) + len(pixels)} grouped: {shown}, algo rows {algo}, pixel avgs {pixels}"
        if self.selected >= 0:
            return f"ROI {self.selected} selected"
        if self.selected_derived is not None:
            si, k = self.selected_derived
            s = self.derived[si]
            return f"{s.name} row {k} selected" + (
                " (marked for deletion)" if k in s.discarded else ""
            )
        return "click an ROI to see its traces; draw (a) a region and add it (r) for a new one"

    def _table_color(self, item):
        if isinstance(item, tuple):
            return self._group_colors().get(item[1:], MARKED_COLOR)
        return self._row_rgb(item)

    def _filter_pairs(self) -> set:
        """The rows on the filter switch's side of the range."""
        order = self.order
        values = np.asarray(order.columns[order.range_column], np.float64)
        inside = (values >= order.range_limits[0]) & (values <= order.range_limits[1])
        side = np.isfinite(values) & ~inside if self.filter_outside else inside
        return {self.rows[int(row)] for row in np.flatnonzero(side)}

    def _draw_filter(self):
        """Masknmf's filter over the table: its column, apply with its side
        switch, the range, then this tool's label, source and slice filters.
        """
        order = self.order
        g = grid(_CAPTIONS)
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        slider_w = min(0.7 * imgui.get_window_width(), right - g.cell_x[0] - g.mark_w)
        g.row("filter")
        imgui.set_next_item_width(g.w)
        glow = self.filter_select
        if glow:
            for color, value in (
                (imgui.Col_.frame_bg, MTHEME.emphasis),
                (imgui.Col_.frame_bg_hovered, MTHEME.emphasis_hover),
                (imgui.Col_.button, MTHEME.emphasis),
                (imgui.Col_.button_hovered, MTHEME.emphasis_hover),
                (imgui.Col_.text, (0.05, 0.05, 0.05)),
            ):
                imgui.push_style_color(color, to_vec4(value))
        opened = imgui.begin_combo("##roi_range_column", order.range_column)
        if glow:
            imgui.pop_style_color(5)
        picked = None
        if opened:
            for name in order.columns:
                if imgui.selectable(name, name == order.range_column)[0]:
                    picked = name
            imgui.end_combo()
        changed = False
        if picked is not None and picked != order.range_column:
            order.set_range_column(picked)
            changed = True
            self.filter_select = False
        tooltip(
            "Filter column: the range below spans it (del is 0 or 1); picking one puts the range back at its full span"
        )
        imgui.same_line(0, g.gap)
        inner = imgui.get_style().item_inner_spacing.x
        need = (
            imgui.get_frame_height()
            + inner
            + imgui.calc_text_size("apply").x
            + g.gap
            + switch_width("inside", "outside")
            + g.mark_w
        )
        if imgui.get_content_region_avail().x < need:
            # a narrow panel: apply and its switch on a line of their own, under the combo
            imgui.new_line()
            imgui.set_cursor_pos_x(g.cell_x[0])
        if self.filter_select and set(self.buffer) != self._filter_pairs():
            self.filter_select = False
        toggled, self.filter_select = imgui.checkbox("apply", self.filter_select)
        if toggled and self.filter_select:
            self._drop_region()
        tooltip(
            "Apply to selection: the rows on the switch's side of the range become the group, as ctrl+a does for "
            "the table, and follow the lines as they move; editing the selection by hand switches this off"
        )
        imgui.same_line(0, g.gap)
        flipped, self.filter_outside = draw_switch(
            "roi_filter", self.filter_outside, "inside", "outside", self.filter_select
        )
        help_mark(
            "which side of the range the filter selects, lit while it drives the selection"
        )
        imgui.same_line(0, g.gap)
        right_aligned_text(f"{len(order.order)} / {order.n_items}")
        g.row("range")
        moved = draw_range_filter(order, "_roi", slider_w)
        tooltip(
            "Range: drag a line to move it, double-click for the full span; the table shows what is inside"
        )
        changed |= moved
        g.row("label")
        names = ["all", "unlabeled", *self.classes.names]
        imgui.set_next_item_width(g.w)
        picked, index = imgui.combo("##roi_label_filter", order.filter_label + 2, names)
        if picked:
            order.filter_label = index - 2
            changed = True
        g.cell(1)
        sources = ["all", "drawn", *(s.name for s in self.derived)]
        current = 0 if order.source is None else order.source + 1
        imgui.set_next_item_width(g.w)
        picked, index = imgui.combo(
            "##roi_source_filter", min(current, len(sources) - 1), sources
        )
        if picked:
            order.source = None if index == 0 else index - 1
            changed = True
        help_mark(
            "only the rows with this label, and from this source: drawn by hand, or a loaded run"
        )
        if self.store.nz > 1:
            g.row("slice")
            on = order.plane is not None
            picked, on = imgui.checkbox(
                f"this plane ({self._plane_label(self.z)} of {self.store.nz})", on
            )
            if picked:
                order.plane = self.z if on else None
                changed = True
        if changed:
            order.rebuild()
        if self.filter_select and (toggled or moved or flipped or changed):
            pairs = self._filter_pairs()
            self.buffer = [
                self.rows[int(row)]
                for row in order.order
                if self.rows[int(row)] in pairs
            ]
            self._refresh_group_view()
        imgui.dummy(imgui.ImVec2(0, em(0.2)))
