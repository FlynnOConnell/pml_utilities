"""The Traces tab's table of trace rows."""

from __future__ import annotations

from functools import partial

import numpy as np
from imgui_bundle import (
    icons_fontawesome_6 as fa,
)
from imgui_bundle import imgui
from masknmf.visualization.imgui import (
    button_colors,
    draw_roi_table,
    tooltip,
)
from masknmf.visualization.imgui import THEME as MTHEME

from mbo_utilities.gui._theme import em

from mbo_utilities.gui.roi._shared import (
    PIXEL_RADIUS,
    TRACE_COLUMNS,
)


class TraceTableMixin:
    """The Traces tab's table of trace rows."""

    def _sorted_trace_rows(self) -> list[tuple]:
        """Trace-table keys in the order the table shows them, so stepping
        with the arrows walks what the user sees.
        """
        self._sync_trace_order(self._trace_rows())
        return [self._trace_keys[int(i)] for i in self.trace_order.order]

    def _trace_rows(self) -> list[tuple]:
        """The table's rows, in insertion order: every trace key, or only the
        ones placed on the slice on screen (drawn ROIs, loaded components,
        scanned lines and pixel averages read on the channel and z-plane
        shown; a results file's rows have no slice and always show).
        """
        keys = self.traces.keys
        if not self.traces_this_slice or self.store.nz <= 1:
            return keys
        z, c = self.slice.z, self.slice.c
        out = []
        for key in keys:
            trace = self.traces.get(key)
            placed = trace is not None and (
                trace.stands_for_roi
                or trace.source == "pixel"
                or "line" in trace.extra
                or self._set_by_name(trace.source) is not None
            )
            if not placed or (trace.z == z and trace.c == c):
                out.append(key)
        return out

    def _trace_cells(self, key) -> tuple:
        """The text of one row's roi / z / c / engine / source columns."""
        trace = self.traces.get(key)
        _v, shown = self._trace_shown(key)
        if trace is None:
            return (shown, "", "", "", "")
        # a results file's rows have no slice to name
        placed = (
            trace.stands_for_roi
            or trace.source == "pixel"
            or "line" in trace.extra
            or self._set_by_name(trace.source) is not None
        )
        if not placed:
            return (shown, "", "", trace.engine, trace.source)
        return (
            shown,
            f"{trace.z + 1}",
            f"{trace.c}",
            trace.engine,
            trace.source + self._binning_tag(key),
        )

    def _trace_stat(self, key) -> tuple[int, float, float, float]:
        """``(frames, mean, peak, snr)`` of the displayed trace,
        cached until the trace sets change; snr is peak over baseline in
        robust sd units.
        """
        got = self._trace_stats.get(key)
        if got is None:
            y = self._display(key)
            f = y if y is not None else np.zeros(0, np.float32)
            if f.size:
                med = float(np.median(f))
                mad = float(np.median(np.abs(f - med)))
                peak = float(f.max())
                snr = (peak - med) / (1.4826 * mad) if mad > 0 else 0.0
                got = (int(f.size), float(f.mean()), peak, snr)
            else:
                got = (0, 0.0, 0.0, 0.0)
            self._trace_stats[key] = got
        return got

    def delete_trace_row(self, key) -> None:
        """Delete one trace row. A drawn ROI keeps its mask (and its other
        rows); an algo component is its trace, so it is marked for deletion.
        """
        self.trace_sel.discard(key)
        pair = self._key_to_pair(key)
        if pair is not None and pair[0] >= 0:
            self.mark([pair], True)
            return
        if self.traces.remove(key) is not None:
            self.status = "deleted trace"

    def draw_trace_table(self, keys=None, table_id: str = "##trace_table"):
        """The Traces tab: every collected trace in masknmf's ROI table;
        click plots one, ctrl+click several, shift+click a range. ``keys``
        narrows the rows to those (the Process tab shows the ROIs it is
        about to run).
        """
        self._note_hover()
        rows = self._trace_rows()
        if keys is not None:
            wanted = set(keys)
            rows = [key for key in rows if key in wanted]
        if not rows:
            imgui.text_disabled(
                "No traces yet. Draw an ROI with trace on draw, run the selection (shift+t), or load a run."
            )
            return
        self._sync_trace_order(rows)
        imgui.text_disabled(f"{len(rows)} traces · {len(self.trace_sel)} picked")
        imgui.same_line(0, em(0.8))
        if imgui.small_button(f"{fa.ICON_FA_LIST_CHECK}##plot_all"):
            self.trace_sel = set(rows)
            self.trace_picked = True
            self._plot_lines_key = None
        tooltip("Plot every listed trace")
        imgui.same_line(0, em(0.4))
        if imgui.small_button(f"{fa.ICON_FA_ERASER}##unplot"):
            self._sync_trace_sel()
        tooltip(
            "Unpick every trace: the plot goes back to the selection; the rows stay"
        )
        imgui.same_line(0, em(0.4))
        picked = [key for key in rows if key in self.trace_sel]
        imgui.begin_disabled(not picked)
        with button_colors(MTHEME.danger, MTHEME.danger_hover):
            delete = imgui.small_button(f"{fa.ICON_FA_TRASH}##delete_picked")
        imgui.end_disabled()
        tooltip(
            f"Delete the {len(picked)} picked trace(s) (a drawn ROI keeps its mask; an algo component is marked for "
            "deletion)"
        )
        if self.store.nz > 1:
            imgui.same_line(0, em(0.8))
            changed, every = imgui.checkbox("all slices", not self.traces_this_slice)
            if changed:
                self.traces_this_slice = not every
            tooltip(
                "List the rows of every channel and z-plane, not only the slice on screen"
            )
        columns = (
            "id",
            *(self._trace_header(name) for name, _sortable, _hidden in TRACE_COLUMNS),
        )
        hidden = {
            self._trace_header(name) for name, _sortable, hide in TRACE_COLUMNS if hide
        }
        if self.store.axis_size("z") <= 1:
            hidden.add(self._trace_header("z"))
        formatters = {
            self._trace_header(name): partial(self._trace_cell, name)
            for name, _sortable, _hidden in TRACE_COLUMNS
        }
        draw_roi_table(
            self.trace_order,
            columns,
            formatters,
            False,
            table_id=table_id,
            hidden=hidden,
            cursor=False,
            on_select=self._trace_table_select,
            is_grouped=lambda i: self._trace_keys[i] in self.trace_sel,
            on_ctrl_select=self._trace_table_ctrl,
            on_shift_select=self._trace_table_shift,
            row_color=lambda i: self._trace_color(self._trace_keys[i]),
            row_label=lambda i: self._trace_shown(self._trace_keys[i])[1],
        )
        picked_one = [key for key in rows if key in self.trace_sel]
        if len(picked_one) == 1:
            trace = self.traces.get(picked_one[0])
            pos = self._line_position(trace) if trace is not None else {}
            if pos.get("start_um") is not None:
                (x0, y0), (x1, y1) = pos["start_um"][:2], pos["end_um"][:2]
                imgui.text_disabled(
                    f"line ({x0:.0f}, {y0:.0f}) -> ({x1:.0f}, {y1:.0f}) um, {pos['length_um']:.1f} um long"
                    + (
                        f", {pos['sample_um']:.2f} um per sample"
                        if pos.get("sample_um")
                        else ""
                    )
                )
        if delete:
            for key in picked:
                self.delete_trace_row(key)
            self.status = f"deleted {len(picked)} traces"

    def _row_peaks(self) -> np.ndarray:
        """The peak of each row's displayed traces, NaN for a row without any."""
        peaks = np.full(len(self.rows), np.nan)
        for row, (si, k) in enumerate(self.rows):
            # mid store event the rows can still name an ROI the store just dropped
            if si < 0 and k >= self.n_rois or si >= len(self.derived):
                continue
            if si < 0:
                keys = [t.key for t in self.traces.for_roi(self.store.rois[k].uid)]
            else:
                keys = self._member_keys(si, k)
            values = [self._trace_stat(key)[2] for key in keys if key in self.traces]
            if values:
                peaks[row] = max(values)
        return peaks

    def _member_keys(self, si: int, k: int) -> list[tuple]:
        """Trace keys of algo row ``k`` of set ``si``: its own row, else its promoted ROI's."""
        s = self.derived[si]
        key = ("member", s.name, k)
        if key in self.traces:
            return [key]
        index = self._promoted.get((s.name, k))
        if index is None:
            return []
        return [t.key for t in self.traces.for_roi(self.store.rois[index].uid)]

    def _format_cell(self, name: str, item) -> str:
        """One cell of the ROIs table; ``item`` is a row, or a pixel average's key."""
        if isinstance(item, tuple):
            trace = self.traces.get(self.pixels.get(item))
            peak = "" if trace is None else f"{self._trace_stat(trace.key)[2]:.3g}"
            side = 2 * PIXEL_RADIUS + 1
            return {"source": "pixel", "area": f"{side * side}", "peak": peak}.get(
                name, ""
            )
        si, k = self.rows[item]
        if name == "label":
            return self.classes.name_of(item) if self.classes.labels[item] >= 0 else ""
        if name == "source":
            # the algorithm, not the run name: the source filter lists run names
            if si < 0:
                return "drawn"
            s = self.derived[si]
            algo = s.algo or s.name
            return f"{algo} · promoted" if (s.name, k) in self._promoted else algo
        if name == "z":
            z = self.store.rois[k].plane if si < 0 else self.derived[si].z
            return self._plane_label(z)
        if name == "ok":
            return "" if si < 0 else ("yes" if self.derived[si].accepted[k] else "no")
        if name == "del":
            return "x" if self.order.columns["del"][item] else ""
        value = self.order.columns[name][item]
        if np.isnan(value):
            return ""
        return f"{int(value)}" if name == "area" else f"{float(value):.3g}"

    def _sync_trace_order(self, keys: list[tuple]) -> None:
        """The trace table's order over ``keys``, its sort kept when they change."""
        if keys == self._trace_keys and self.trace_order.n_items == len(keys):
            return
        self._trace_keys = list(keys)
        stats = [self._trace_stat(key) for key in keys]
        cells = [self._trace_cells(key) for key in keys]
        columns = {
            self._trace_header("z"): np.array(
                [float(c[1]) if c[1] else np.nan for c in cells]
            ),
            self._trace_header("c"): np.array(
                [float(c[2]) if c[2] else np.nan for c in cells]
            ),
            "frames": np.array([s[0] for s in stats], np.float64),
            "peak": np.array([s[2] for s in stats], np.float64),
        }
        order = self.trace_order
        order.columns = columns
        order.n_items = len(keys)
        if order.sort_by not in columns:
            order.sort_by = None
        order.rebuild()

    def _trace_header(self, name: str) -> str:
        """A trace-table column's header: the data's own name for the z and c axes (``ROI``, ``Channel``)."""
        return self.axis_label(name) if name in ("z", "c") else name

    def _trace_cell(self, name: str, item: int) -> str:
        key = self._trace_keys[item]
        shown, z_text, c_text, engine, source = self._trace_cells(key)
        if name == "z":
            return z_text
        if name == "c":
            return c_text
        if name == "source":
            return source
        if name == "engine":
            return engine
        n, _mean, peak, _snr = self._trace_stat(key)
        return f"{n}" if name == "frames" else f"{peak:.3g}"

    def _trace_table_select(self, item: int):
        self.select_trace(self._trace_keys[item])

    def _trace_table_ctrl(self, item: int):
        self.toggle_trace(self._trace_keys[item])

    def _trace_table_shift(self, item: int):
        """Every row between the last picked one and ``item``, in the order shown."""
        view = [int(i) for i in self.trace_order.order]
        picked = [
            view.index(i)
            for i, key in enumerate(self._trace_keys)
            if key in self.trace_sel and i in view
        ]
        if item not in view:
            return
        to = view.index(item)
        start = picked[-1] if picked else to
        for pos in range(min(start, to), max(start, to) + 1):
            self.trace_sel.add(self._trace_keys[view[pos]])
        self.trace_picked = True
        self._plot_lines_key = None
