"""Selection, the group buffer, labeling, marks, delete and undo."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from mbo_utilities.annotation import (
    UNLABELED,
)

from mbo_utilities.gui.roi._shared import (
    _cleared_note,
    _UNDO_DEPTH,
)


class SelectionMixin:
    """Selection, the group buffer, labeling, marks, delete and undo."""

    def _pick(self, row: int, col: int, mods: frozenset = frozenset()):
        """Masknmf's click: a drawn ROI under the cursor, else an algo one.
        Ctrl toggles it in the group, shift adds it; a plain click selects it
        alone, or deselects it when it is already the selection. An empty
        pixel adds its 5x5 average with pixel traces on, else deselects.
        """
        try:
            hit: tuple[int, int] | None = None
            index = self.store.roi_at(self.z, row, col)
            if index >= 0:
                hit = (-1, index)
            elif 0 <= row < self.ny and 0 <= col < self.nx:
                for si, s in enumerate(self.derived):
                    if not s.visible or s.z != self.z:
                        continue
                    k = int(s.pick_map[row, col])
                    if k >= 0:
                        hit = (si, k)
                        break
            ctrl = bool(mods & {"Ctrl", "Control"})
            if hit is not None:
                if ctrl:
                    self.buffer_toggle(*hit)
                elif "Shift" in mods:
                    self.buffer_add(*hit)
                elif self._selection_pair() == hit and not self.buffer:
                    self._snapshot()
                    self.select_roi(-1)
                else:
                    self.buffer_clear()
                    self.select_pair(*hit)
                return
            if self.pixel_traces and 0 <= row < self.ny and 0 <= col < self.nx:
                self.add_pixel(row, col, toggle=ctrl)
                return
            if self.buffer or self.pixel_group or self._selection_pair() is not None:
                self._snapshot()
            self.pixel_group = []
            self.active_pixel = None
            self.buffer_clear()
            self.select_roi(-1)
        except Exception as e:  # noqa: BLE001 - surfaced in the status row
            self.logger.exception("pick failed")
            self.status = f"pick failed: {type(e).__name__}: {e}"

    def _row_grouped(self, item) -> bool:
        if isinstance(item, tuple):
            return item in self.pixel_group
        return 0 <= item < len(self.rows) and self.rows[item] in self.buffer

    def _table_select(self, item):
        """A plain table click: the row alone, or nothing when it already is the selection."""
        if isinstance(item, tuple):
            pid = item[1:]
            self.buffer_clear()
            self.select_roi(-1)
            self.pixel_group = [pid]
            self.active_pixel = pid
            self._refresh_group_view()
            return
        if self.rows[item] == self._selection_pair() and not self.buffer:
            self._snapshot()
            self.select_roi(-1)
            return
        self.buffer_clear()
        self.select_row(item)

    def _table_ctrl(self, item):
        if isinstance(item, tuple):
            pid = item[1:]
            if pid in self.pixel_group:
                self.pixel_group.remove(pid)
                self.active_pixel = None
            else:
                self._seed_buffer()
                self.pixel_group.append(pid)
                self.active_pixel = pid
            self._refresh_group_view()
        elif 0 <= item < len(self.rows):
            self.buffer_toggle(*self.rows[item])

    def select_row(self, row: int | None):
        """Route a table-row selection to the right kind."""
        if row is None or not 0 <= row < len(self.rows):
            self.select_roi(-1)
            return
        self.select_pair(*self.rows[row])

    def select_roi(self, index: int | None):
        """Select drawn ROI ``index``; anything out of range clears the
        selection (and any derived one).

        Selecting an ROI on another plane jumps the z slider to it; one with
        a trace shows it in the trace plot.
        """
        self.selected_derived = None
        self.active_pixel = None
        self.selected = index if index is not None and 0 <= index < self.n_rois else -1
        self.scroll_to_selection = True
        if self.selected < 0:
            self._note_buf = ""
            self.status = f"{self.n_rois} ROIs"
        else:
            record = self.store.rois[self.selected]
            self._note_buf = record.note
            cleared = self.order.reveal(self.selected)
            self.status = f"ROI {self.selected}: {record.area} px" + _cleared_note(
                cleared
            )
            if record.plane != self.z:
                self._goto_plane(record.plane)
            if self.traces.for_roi(record.uid):
                self.trace_uid = record.uid
            if self.follow:
                self._center_on(*self._feather(self.selected)[:2])
        self._sync_trace_sel()
        self.refresh_overlay()

    def select_derived(self, si: int, k: int):
        """Select component ``k`` of derived set ``si`` (clears any drawn
        selection; jumps z to the set's plane).
        """
        if not (0 <= si < len(self.derived) and 0 <= k < len(self.derived[si])):
            self.select_roi(-1)
            return
        self.selected = -1
        self.active_pixel = None
        self._note_buf = ""
        self.selected_derived = (si, k)
        self.scroll_to_selection = True
        s = self.derived[si]
        row = self._row_index.get((si, k))
        cleared = self.order.reveal(row) if row is not None else []
        ypix, xpix, _lam = s.footprints[k]
        tail = " · promoted" if (s.name, k) in self._promoted else ""
        tail += " · marked for deletion" if k in s.discarded else ""
        self.status = f"{s.name} row {k}: {ypix.size} px{tail}" + _cleared_note(cleared)
        if s.z != self.z:
            self._goto_plane(s.z)
        if self.follow:
            self._center_on(ypix, xpix)
        self._sync_trace_sel()
        self.refresh_overlay()

    def _seed_buffer(self):
        """A first ctrl / shift click keeps the current selection grouped,
        so 'select one, ctrl+click another' makes a group of two.
        """
        if self.buffer:
            return
        pair = self._selection_pair()
        if pair is not None:
            self.buffer.append(pair)

    def buffer_add(self, si: int, k: int):
        """Add one row to the group and make it the shown one."""
        self._seed_buffer()
        if (si, k) not in self.buffer:
            self.buffer.append((si, k))
        self._after_buffer_change(si, k)

    def buffer_toggle(self, si: int, k: int):
        """Ctrl+click: flip one row's group membership; the cursor moves to another member."""
        self._seed_buffer()
        if (si, k) in self.buffer:
            self.buffer.remove((si, k))
            if self._selection_pair() == (si, k):
                if self.buffer:
                    self.select_pair(*self.buffer[-1])
                else:
                    self.select_roi(-1)
            self._refresh_group_view()
            self.status = f"{len(self.buffer)} in group"
        else:
            self.buffer.append((si, k))
            self._after_buffer_change(si, k)

    def buffer_extend_to(self, item: int):
        """Shift+click in the table: group every row between the cursor and
        ``item``, in the order the table shows.
        """
        hits = np.flatnonzero(self.order.order == item)
        if not len(hits):
            return
        self._seed_buffer()
        a, b = sorted((self.order.pos, int(hits[0])))
        for pos in range(a, b + 1):
            pair = self.rows[int(self.order.order[pos])]
            if pair not in self.buffer:
                self.buffer.append(pair)
        self._after_buffer_change(*self.rows[item])

    def buffer_clear(self):
        if self.buffer or self.pixel_group:
            self.buffer = []
            self.pixel_group = []
            self.filter_select = False
            self._refresh_group_view()

    def _after_buffer_change(self, si: int, k: int):
        n = len(self.buffer)
        self.filter_select = (
            self.filter_select and set(self.buffer) == self._filter_pairs()
        )
        self.select_pair(si, k)
        if n > 1:
            self.status = (
                f"{n} in group · their traces share the plot, labels apply to all"
            )

    def _refresh_group_view(self):
        self._plot_lines_key = None
        self.refresh_overlay()

    def set_group_color(self, rgb: tuple[float, float, float] | None):
        """Give every grouped ROI (or just the selection) an explicit
        display color; None reverts to the class / hue colors.
        """
        targets = list(self.buffer)
        if not targets and self._selection_pair() is not None:
            targets = [self._selection_pair()]
        if not targets:
            return
        rgb255 = None if rgb is None else tuple(int(round(float(v) * 255)) for v in rgb)
        self.store.block_events(True)
        try:
            for si, k in targets:
                if si < 0:
                    self.store.set_color(k, rgb255)
                elif rgb is None:
                    self.derived[si].colors.pop(k, None)
                else:
                    self.derived[si].colors[k] = tuple(float(v) for v in rgb)
        finally:
            self.store.block_events(False)
        self._refresh_group_view()
        self._autosave()
        self._autosave()
        self.status = (
            f"colored {len(targets)} ROI(s)"
            if rgb is not None
            else f"reset {len(targets)} color(s)"
        )

    def step(self, delta: int):
        """Up / down: the next or previous row of the ROIs table; the image,
        the table and the trace plot land on the same ROI.
        """
        if self.order.step(delta):
            self.buffer_clear()
            self.select_row(self.order.current)

    def step_trace(self, delta: int) -> bool:
        """Move to the next / previous row of the trace table, in the order
        the table shows, and select the ROI behind it.
        """
        rows = self._sorted_trace_rows()
        if not rows:
            return False
        at = next((i for i, key in enumerate(rows) if key in self.trace_sel), None)
        if at is None:
            pos = 0 if delta > 0 else len(rows) - 1
        else:
            pos = int(np.clip(at + delta, 0, len(rows) - 1))
        self.select_trace(rows[pos])
        return True

    def select_trace(self, key):
        """Plot just this trace and select the ROI behind it."""
        self.trace_sel = {key}
        self.trace_picked = True
        self._plot_lines_key = None
        pair = self._key_to_pair(key)
        if pair is None:
            return
        si, k = pair
        if si < 0:
            self.trace_uid = self.store.rois[k].uid
        self.select_pair(si, k)
        self.trace_sel = {key}
        self.trace_picked = True

    def toggle_trace(self, key):
        """Add / remove one trace from the plotted set (ctrl+click)."""
        (self.trace_sel.discard if key in self.trace_sel else self.trace_sel.add)(key)
        self.trace_picked = True
        self._plot_lines_key = None

    def next_unlabeled(self, inclusive: bool = False):
        if self.order.next_unlabeled(inclusive):
            self.select_row(self.order.current)

    def _select_next_derived(
        self,
        start_pos: int,
        skip_promoted: bool = False,
        unlabeled_only: bool = False,
    ):
        """Select the next derived row in view at or after ``start_pos``.

        ``unlabeled_only`` walks past rows that already carry a label and
        wraps once, so labeling a run's components in follow mode lands on
        each one that still needs a label rather than on whatever row
        happens to come next.
        """
        n = len(self.order.order)
        if not n:
            return
        start = max(start_pos, 0)
        span = (
            [(start + i) % n for i in range(n)] if unlabeled_only else range(start, n)
        )
        for pos in span:
            row = int(self.order.order[pos])
            si, k = self.rows[row]
            if si < 0:
                continue
            if skip_promoted and (self.derived[si].name, k) in self._promoted:
                continue
            if unlabeled_only and int(self.classes.labels[row]) != UNLABELED:
                continue
            self.select_row(row)
            return

    def delete_roi(self, index: int, record: bool = True):
        """Drop one drawn ROI; the indices above it shift and the traces of
        every other ROI survive (they are keyed by uid). Ctrl+z brings it back.
        """
        if not 0 <= index < self.n_rois:
            return
        if record:
            self._push_undo()
        # the store's event resyncs the rows, so the group is renumbered first
        self.buffer = [
            (si, k - (1 if si < 0 and k > index else 0))
            for si, k in self.buffer
            if not (si < 0 and k == index)
        ]
        self.store.delete_roi(index)
        self.select_roi(min(index, self.n_rois - 1))
        self.status = f"deleted ROI {index}"

    def delete_selected(self):
        """Delete (d): the active drawn ROI, else the active pixel average,
        else mark the selected algo ROIs for deletion (unmark when all are);
        marking one steps to the next row. A group of drawn ROIs is never
        deleted at once.
        """
        if self.selected >= 0:
            self.delete_roi(self.selected)
            return
        if self.active_pixel is not None:
            self._snapshot()
            self.drop_pixel(self.active_pixel)
            self.refresh_overlay()
            return
        pairs = [pair for pair in self.buffer if pair[0] >= 0]
        if not pairs and self.selected_derived is not None:
            pairs = [self.selected_derived]
        if not pairs:
            return
        on = not all(k in self.derived[si].discarded for si, k in pairs)
        follows = None
        if on and len(pairs) == 1:
            row = self._row_index.get(pairs[0])
            view = [int(r) for r in self.order.order]
            if row in view[:-1]:
                follows = view[view.index(row) + 1]
        self.mark(pairs, on)
        if follows is not None:
            self.buffer_clear()
            self.select_row(follows)

    def clear(self):
        """Delete every drawn ROI (loaded runs and their rows stay)."""
        self.buffer = [pair for pair in self.buffer if pair[0] >= 0]
        self.store.clear()
        self.select_roi(-1)
        self.status = "cleared"

    def assign_class(self, class_index: int):
        """Give the selected ROI - drawn or derived - a class label;
        UNLABELED (-1) clears it. With a group of two or more (ctrl / shift
        click) the label lands on every member.
        """
        if len(self.buffer) > 1:
            self.store.block_events(True)
            try:
                for si, k in self.buffer:
                    if si < 0:
                        self.store.set_class(k, class_index)
                    elif class_index == UNLABELED:
                        self.derived[si].classes.pop(k, None)
                    else:
                        self.derived[si].classes[k] = int(class_index)
            finally:
                self.store.block_events(False)
            self._resync()
            name = (
                "unlabeled"
                if class_index == UNLABELED
                else self.classes.names[class_index]
            )
            self.status = f"{len(self.buffer)} ROIs -> {name}"
            self._refresh_group_view()
            self._autosave()
            self._autosave()
            return
        if self.selected >= 0:
            self.store.set_class(self.selected, class_index)
            # False when a label filter dropped the row as it was labeled -
            # the cursor is then already on the next candidate
            listed = self.order.goto(self.selected)
            self.status = f"ROI {self.selected}: {self.classes.name_of(self.selected)}"
            if self.follow and class_index != UNLABELED:
                self.next_unlabeled(inclusive=not listed)
            return
        if self.selected_derived is None:
            return
        si, k = self.selected_derived
        s = self.derived[si]
        if class_index == UNLABELED:
            s.classes.pop(k, None)
        else:
            s.classes[k] = int(class_index)
        self._resync()
        row = self._row_index.get((si, k))
        if row is not None:
            self.order.goto(row)
            self.status = f"{s.name} row {k}: {self.classes.name_of(row)}"
        self._autosave()
        if self.follow and class_index != UNLABELED:
            # the next algo row that still needs a label, not just the next
            # one: labeling used to land on rows already labeled
            self._select_next_derived(self.order.pos + 1, unlabeled_only=True)

    label_selected = assign_class

    def unlabel_all(self):
        self.store.block_events(True)
        try:
            for i in range(self.n_rois):
                self.store.set_class(i, UNLABELED)
        finally:
            self.store.block_events(False)
        self.classes.assign(range(self.n_rois), UNLABELED)
        self.order.rebuild()
        self.refresh_overlay()
        self.status = f"cleared {self.n_rois} labels"
        self._autosave()

    def discard_derived(self, si: int, k: int, advance: bool = False):
        """Mark one algo row for deletion (n); ``advance`` steps to the next row."""
        self.mark([(si, k)], True)
        if advance:
            self._select_next_derived(self.order.pos + 1)

    def undiscard_derived(self, si: int, k: int):
        self.mark([(si, k)], False)

    def _selection_pair(self) -> tuple[int, int] | None:
        if self.selected >= 0:
            return (-1, self.selected)
        return self.selected_derived

    def select_pair(self, si: int, k: int):
        """Select one row by its ``(set, row)`` pair; ``si`` -1 is a drawn ROI."""
        if si < 0:
            self.select_roi(k)
        else:
            self.select_derived(si, k)

    def _table_shift(self, item):
        if isinstance(item, tuple):
            pid = item[1:]
            self._seed_buffer()
            if pid not in self.pixel_group:
                self.pixel_group.append(pid)
            self.active_pixel = pid
            self._refresh_group_view()
        else:
            self.buffer_extend_to(item)

    def mark(self, pairs, on: bool, record: bool = True):
        """Mark algo rows for deletion, or unmark them: they stay listed,
        pinned at the top of the table and red on the image, until unmarked.
        """
        pairs = [(int(si), int(k)) for si, k in pairs if si >= 0]
        if not pairs:
            return
        if record:
            self._push_undo()
        for si, k in pairs:
            if on:
                self.derived[si].discarded.add(k)
            else:
                self.derived[si].discarded.discard(k)
        self._resync()
        self.refresh_overlay()
        self._autosave()
        self.status = f"{'marked' if on else 'unmarked'} {len(pairs)} algo ROI(s)"

    def deselect(self):
        """Esc: drop the selection, the group and the pixel averages."""
        if self.buffer or self.pixels or self._selection_pair() is not None:
            self._snapshot()
        for pid in list(self.pixels):
            self.drop_pixel(pid)
        self.buffer = []
        self.pixel_group = []
        self.filter_select = False
        self.select_roi(-1)

    def select_all(self):
        """Ctrl+a: group every row the table shows."""
        rows = [int(r) for r in self.order.order]
        if not rows:
            return
        self.buffer = [self.rows[r] for r in rows]
        self.select_row(rows[-1])
        self._refresh_group_view()
        self.status = f"{len(rows)} in group"

    def _key_of(self, pair):
        """A row's identity for undo: a drawn ROI by uid, an algo row by its pair."""
        if pair is None:
            return None
        si, k = pair
        return ("uid", self.store.rois[k].uid) if si < 0 else (si, k)

    def _pair_of(self, key):
        """The row an undo key names now, or None when it is gone."""
        if key is None:
            return None
        if key[0] == "uid":
            index = self.store.uid_index(key[1])
            return None if index is None else (-1, index)
        return key if key in self._row_index else None

    def _selection_state(self) -> dict:
        return {
            "selected": self._key_of(self._selection_pair()),
            "buffer": [self._key_of(pair) for pair in self.buffer],
            "pixel_group": list(self.pixel_group),
            "pixels": OrderedDict(self.pixels),
            "active_pixel": self.active_pixel,
        }

    def _push_undo(self):
        """Remember the ROIs, their traces, the marks and the selection as
        they are now, for ctrl+z (before the step that changes them).
        """
        self._undo.append(
            {
                "store": self.store.snapshot(),
                "traces": [t for t in self.traces if t.stands_for_roi],
                "marks": {str(s.path): set(s.discarded) for s in self.derived},
                "selection": self._selection_state(),
            }
        )
        del self._undo[:-_UNDO_DEPTH]

    def _snapshot(self):
        """Push the selection for ctrl+z (masknmf's snapshot before a deselect)."""
        self._push_undo()

    def undo(self):
        """Ctrl+z: back one step; a deleted ROI comes back with its traces, a drawn one goes."""
        if not self._undo:
            return
        step = self._undo.pop()
        self._drop_region()
        self.store.restore(step["store"])
        live = {r.uid for r in self.store.rois}
        for trace in step["traces"]:
            if trace.uid in live and trace.key not in self.traces:
                self.traces.add(trace)
        for s in self.derived:
            marks = step["marks"].get(str(s.path))
            if marks is not None:
                s.discarded = set(marks)
        self._resync()
        state = step["selection"]
        for pid in list(self.pixels):
            if pid not in state["pixels"]:
                self.drop_pixel(pid)
        self.pixels = OrderedDict(
            (pid, key)
            for pid, key in state["pixels"].items()
            if key in self.traces or pid in self.pixels
        )
        self.pixel_group = [pid for pid in state["pixel_group"] if pid in self.pixels]
        self.active_pixel = (
            state["active_pixel"] if state["active_pixel"] in self.pixels else None
        )
        self.buffer = [
            pair
            for pair in (self._pair_of(key) for key in state["buffer"])
            if pair is not None
        ]
        pair = self._pair_of(state["selected"])
        if pair is None:
            self.select_roi(-1)
        else:
            self.select_pair(*pair)
        self._refresh_group_view()
        self.status = f"undone, {len(self._undo)} more"
