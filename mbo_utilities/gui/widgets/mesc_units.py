"""MESc unit selector: switch which MUnit of a ``.mesc`` file is displayed.

A ``.mesc`` holds one measurement unit per scan the operator ran — a z-stack,
a ribbon time series, a snapshot — and they are unrelated recordings with
different shapes. The launch picker chooses the first one to open; this widget
switches between them without leaving the viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities.arrays.mesc_geometry import zstack_contents
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.widgets._base import Widget

_ACCENT = imgui.ImVec4(0.8, 0.8, 0.2, 1.0)
_ERROR = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)

# (header, hidden by default); columns fit their content and the table scrolls
# sideways; right-click a header to show or hide one
UNIT_COLUMNS = (
    ("session", False),
    ("unit", False),
    ("modality", False),
    ("layout", True),
    ("ROIs", False),
    ("links", False),
    ("T", False),
    ("C", False),
    ("Z", False),
    ("Y", False),
    ("X", False),
    ("fs", False),
    ("duration", False),
    ("start", False),
    ("comment", False),
)
LINKS_COLUMN = next(i for i, (name, _hidden) in enumerate(UNIT_COLUMNS) if name == "links")


def mesc_array_of(obj):
    """The `MescArray` behind a viewer array, or None.

    Peels the display wrappers (`_SqueezeSingletonDims`, `_ScrubTimingProxy`,
    …), each of which exposes its source as ``_arr``.
    """
    from mbo_utilities.arrays.mesc import MescArray

    for _ in range(8):
        if isinstance(obj, MescArray):
            return obj
        nxt = getattr(obj, "_arr", None)
        if nxt is None or nxt is obj:
            return None
        obj = nxt
    return None


def display_wrap(arr):
    """Wrap a reader the way the launch path does before handing it to the viewer."""
    from mbo_utilities.gui.run_gui import _ScrubTimingProxy, _SqueezeSingletonDims

    shape = getattr(arr, "shape", ())
    if len(shape) == 5 and any(shape[i] == 1 for i in range(3)):
        arr = _SqueezeSingletonDims(arr)
    return _ScrubTimingProxy(arr)


def _shape_text(shape) -> str:
    t, c, z, y, x = shape
    return f"{t}T x {c}C x {z}Z  ·  {y} x {x} px"


class MescUnitsWidget(Widget):
    """Combo bar to switch which MUnit of the open ``.mesc`` is displayed."""

    name = "MESc Units"
    priority = 4
    toggle_key = "preview.mesc_units"

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._error: str | None = None

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        iw = getattr(parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        return bool(len(data)) and mesc_array_of(data[0]) is not None

    # -- state -----------------------------------------------------------

    @property
    def _mesc(self):
        iw = getattr(self.parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        return mesc_array_of(data[0]) if len(data) else None

    def _cache(self) -> dict:
        """Open units, kept on the parent so a widget rebuild doesn't drop them."""
        cache = getattr(self.parent, "_mesc_unit_cache", None)
        if cache is None:
            cache = {}
            self.parent._mesc_unit_cache = cache
        return cache

    def _open_unit(self, path, key: str):
        """The `MescArray` for one unit, opening it on first use."""
        cache = self._cache()
        arr = cache.get(key)
        if arr is None:
            from mbo_utilities.arrays.mesc import MescArray

            arr = MescArray(path, unit=key)
            cache[key] = arr
        return arr

    # -- swapping --------------------------------------------------------

    def _install(self, arr) -> None:
        """Show `arr` in the viewer, re-deriving every per-dataset display state."""
        from mbo_utilities.gui._dialogs import swap_viewer_array

        unit = arr.unit_key.rsplit("/", 1)[-1]
        swap_viewer_array(
            self.parent, arr, title=f"{Path(arr.filenames[0]).stem[:16]} · {unit}"
        )
        parent = self.parent

        # the Manual ROI panel caches tdim/zdim/cdim and its mask store's
        # (ny, nx) from whichever unit was live when it was built; each unit
        # is an unrelated recording, so that has to be re-derived per swap.
        manual_roi = getattr(parent, "manual_roi", None)
        if manual_roi is not None:
            manual_roi.rebind()

        # the Traces tab is bound to the old unit's slider dims (ROI may not
        # exist on the new one at all); tear it down and let it re-derive
        # itself, same as when the viewer first opens
        traces = getattr(parent, "linescan_traces", None)
        if traces is not None:
            traces.close()
            parent.linescan_traces = None
        from mbo_utilities.gui.linescan_viewer import attach_standard_traces

        try:
            attach_standard_traces(parent)
        except Exception:
            parent.logger.warning("line-scan traces tab unavailable", exc_info=True)
        parent.logger.info(f"MESc unit: {arr.unit_key}  shape={arr.shape}")

    def _switch(self, arr) -> None:
        self._error = None
        try:
            self._install(arr)
        except Exception as e:
            self._error = str(e)
            self.parent.logger.exception(f"MESc unit switch failed: {e}")

    # -- ui --------------------------------------------------------------

    def draw(self) -> None:
        mesc = self._mesc
        if mesc is None:
            return
        units = mesc.units
        # the unit opened at launch belongs in the cache too, so switching
        # away and back reuses it instead of opening the file a second time
        self._cache().setdefault(mesc.unit_key, mesc)

        imgui.spacing()
        imgui.text_colored(_ACCENT, "MESc Units")
        imgui.spacing()

        # `--roi 0` fans the ROIs of one unit across several subplots. Swapping
        # units there would replace only the first one and leave the rest
        # showing the old unit, so the selector stands down and says why.
        if len(self.parent.image_widget.data) > 1:
            imgui.text_disabled(f"{mesc.unit_key.rsplit('/', 1)[-1]} · split ROIs")
            imgui.text_disabled("Reopen without --roi to switch units.")
            return

        labels = [
            f"{u['munit']} · {u['modality_name']}"
            + (f" · {u['role']}" if u.get("role") and u["role"] != "measurement" else "")
            + (f" · {u['start_time'][:10]}" if u.get("start_time") else "")
            for u in units
        ]
        current = next(
            (i for i, u in enumerate(units) if u["key"] == mesc.unit_key), 0
        )

        imgui.set_next_item_width(imgui.get_content_region_avail().x * 0.9)
        changed, new_idx = imgui.combo("##mesc_unit", current, labels)
        set_tooltip(
            "Measurement unit to display. Each MUnit is one scan from this "
            "session — a z-stack, a time series, a snapshot — with its own "
            "shape and acquisition settings."
        )
        if changed and new_idx != current:
            unit = units[new_idx]
            try:
                arr = self._open_unit(mesc.filenames[0], unit["key"])
            except Exception as e:
                self._error = str(e)
                self.parent.logger.exception(f"cannot open {unit['key']}: {e}")
            else:
                self._switch(arr)
                return  # the array under us changed; redraw next frame

        info = units[current]
        imgui.text_disabled(_shape_text(info["shape"]))
        detail = f"{info['kind']} layout"
        fs = mesc.metadata.get("fs")
        if fs:
            detail += f"  ·  {fs:.1f} Hz"
        dur = info.get("duration_s")
        if dur:
            detail += f"  ·  {dur:.0f} s"
        if info["start_time"]:
            detail += f"  ·  {info['start_time'][:16].replace('T', ' ')}"
        imgui.text_disabled(detail)
        if info["comment"]:
            imgui.text_disabled(info["comment"][:48])
            set_tooltip(info["comment"])

        if self._error:
            imgui.text_colored(_ERROR, "Unit switch failed")
            set_tooltip(self._error)

    def cleanup(self) -> None:
        for arr in (getattr(self.parent, "_mesc_unit_cache", None) or {}).values():
            try:
                arr.close()
            except Exception:
                pass
        self.parent._mesc_unit_cache = None


def unit_links(info: dict, contains: list[str] | tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """The units one unit is paired with, as ``(role, key)`` pairs: a scan's
    snapshot (``drawn on``) and RTMC stream, a snapshot's scans (``background
    of``), a stream's scan (``RTMC of``) and a Z-stack's scans (``holds``:
    ``contains``, from ``mesc_geometry.zstack_contents``)."""
    links = []
    if info.get("background_unit"):
        links.append(("drawn on", info["background_unit"]))
    if info.get("rtmc_unit"):
        links.append(("RTMC stream", info["rtmc_unit"]))
    links += [("background of", k) for k in info.get("scans", ())]
    links += [("RTMC of", k) for k in info.get("rtmc_of", ())]
    links += [("holds", k) for k in contains]
    return links


def unit_row(info: dict, contains: list[str] | tuple[str, ...] = ()) -> tuple[tuple[str, ...], tuple]:
    """One table row per `list_mesc_units` entry: the cell texts and the sort
    keys, both in UNIT_COLUMNS order (numbers sort as numbers). The ``links``
    cell is how many units this one is paired with (:func:`unit_links`); the
    tab draws it as a button that lists them."""
    t, c, z, y, x = info["shape"]
    fs = info.get("fs")
    dur = info.get("duration_s")
    start = (info.get("start_time") or "")[:19].replace("T", " ")
    comment = " / ".join(info.get("comment", "").splitlines())
    n = info.get("n_outlines", 0)
    nouns = {"line": ("line", "lines"), "patch": ("patch", "patches")}
    rois = f"{n} {nouns[info['outline_kind']][n != 1]}" if n else "-"
    links = len(unit_links(info, contains))
    cells = (
        info["session"],
        info["munit"],
        info["modality_name"],
        info["kind"],
        rois,
        str(links) if links else "-",
        str(t),
        str(c),
        str(z),
        str(y),
        str(x),
        f"{fs:.1f} Hz" if fs else "-",
        f"{dur:.0f} s" if dur else "-",
        start,
        comment,
    )
    keys = (
        info["session"],
        info["index"],
        info["modality_name"],
        info["kind"],
        n,
        links,
        t,
        c,
        z,
        y,
        x,
        fs or 0.0,
        dur or 0.0,
        start,
        comment.lower(),
    )
    return cells, keys


class MescTabWidget(Widget):
    """The MESc tab: every measurement unit in the open file, with its
    comment; the displayed unit is highlighted and clicking a row shows it."""

    name = "MESc"
    tab_label = "MESc"
    placement = "tab"
    toggle_key = "mesc"
    priority = 15

    def __init__(self, parent: Any):
        super().__init__(parent)
        # the combo widget owns the unit cache and the viewer swap
        self._units = MescUnitsWidget(parent)
        self._sort = (0, True)

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return MescUnitsWidget.is_supported(parent)

    def _contains(self, mesc) -> dict[str, list[str]]:
        """Each Z-stack's scans, placed once per file and kept on the parent."""
        path = str(mesc.filenames[0])
        cached = getattr(self.parent, "_mesc_zstack_contents", None)
        if cached is None or cached[0] != path:
            try:
                contents = zstack_contents(path, mesc.units)
            except Exception:
                self.parent.logger.warning(
                    f"{Path(path).name}: cannot place its scans on its Z-stacks", exc_info=True
                )
                contents = {}
            cached = (path, contents)
            self.parent._mesc_zstack_contents = cached
        return cached[1]

    def draw(self) -> None:
        with imgui_ctx.begin_child(
            "##MescContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            mesc = self._units._mesc
            if mesc is None:
                imgui.text_disabled("No .mesc file is open.")
                return
            units = mesc.units
            contains = self._contains(mesc)
            self._units._cache().setdefault(mesc.unit_key, mesc)
            split = len(self.parent.image_widget.data) > 1

            imgui.text_colored(_ACCENT, Path(mesc.filenames[0]).name)
            imgui.same_line(0, 12)
            imgui.text_disabled(
                f"{len(units)} units · showing {mesc.unit_key.rsplit('/', 1)[-1]}"
            )
            if split:
                imgui.text_disabled("Split ROIs: reopen without --roi to switch units.")
            else:
                imgui.text_disabled("Click a row to display that unit.")
            if self._units._error:
                imgui.text_colored(_ERROR, "Unit switch failed")
                set_tooltip(self._units._error)

            flags = (
                imgui.TableFlags_.sortable | imgui.TableFlags_.row_bg
                | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.scroll_y
                | imgui.TableFlags_.scroll_x | imgui.TableFlags_.resizable
                | imgui.TableFlags_.hideable | imgui.TableFlags_.sizing_fixed_fit
            )
            avail = imgui.get_content_region_avail()
            # a new table id whenever the columns change: imgui restores a
            # saved layout by column index, so the old widths would land on
            # the wrong columns
            if not imgui.begin_table(
                "##mesc_units_v2", len(UNIT_COLUMNS), flags, imgui.ImVec2(0, avail.y)
            ):
                return
            # the header row and the session and unit columns stay put while scrolling
            imgui.table_setup_scroll_freeze(2, 1)
            for i, (name, hidden) in enumerate(UNIT_COLUMNS):
                column_flags = imgui.TableColumnFlags_.width_fixed
                if i == 0:
                    column_flags |= imgui.TableColumnFlags_.default_sort
                if hidden:
                    column_flags |= imgui.TableColumnFlags_.default_hide
                imgui.table_setup_column(name, column_flags)
            imgui.table_headers_row()
            set_tooltip("Right-click a header to show or hide columns", show_mark=False)
            specs = imgui.table_get_sort_specs()
            if specs is not None and specs.specs_dirty:
                if specs.specs_count > 0:
                    self._sort = (
                        int(specs.specs.column_index),
                        specs.specs.sort_direction == imgui.SortDirection.ascending,
                    )
                specs.specs_dirty = False
            column, ascending = self._sort
            rows = sorted(
                ((*unit_row(u, contains.get(u["key"], ())), u) for u in units),
                key=lambda row: row[1][column],
                reverse=not ascending,
            )
            by_key = {u["key"]: u for u in units}
            picked = None
            last = len(UNIT_COLUMNS) - 1
            for cells, _keys, info in rows:
                imgui.table_next_row()
                imgui.table_next_column()
                clicked, _ = imgui.selectable(
                    f"{cells[0]}##unit_{info['key']}",
                    info["key"] == mesc.unit_key,
                    # the row spans the links button's column too; without
                    # overlap it takes the hover and the button never sees a click
                    imgui.SelectableFlags_.span_all_columns | imgui.SelectableFlags_.allow_overlap,
                )
                if clicked and not split and info["key"] != mesc.unit_key:
                    picked = info
                for i in range(1, len(cells)):
                    if not imgui.table_next_column():
                        continue
                    if i != LINKS_COLUMN:
                        imgui.text(cells[i])
                        if i == last and info["comment"] and imgui.is_item_hovered():
                            imgui.set_tooltip(info["comment"])
                        continue
                    links = unit_links(info, contains.get(info["key"], ()))
                    if not links:
                        imgui.text_disabled("-")
                        continue
                    if imgui.small_button(f"{cells[i]}##links_{info['key']}"):
                        imgui.open_popup(f"##links_{info['key']}")
                    set_tooltip("The units this one is paired with; click one to display it", show_mark=False)
                    if imgui.begin_popup(f"##links_{info['key']}"):
                        for role, key in links:
                            other = by_key.get(key)
                            text = f"{role}  {key}" + (f"  ·  {other['modality_name']}" if other else "")
                            if imgui.selectable(f"{text}##{info['key']}", False)[0] and other is not None and not split:
                                picked = other
                        imgui.end_popup()
            imgui.end_table()

            if picked is None:
                return
            # swapping rebuilds the widget list; finish the frame on the old one
            try:
                arr = self._units._open_unit(mesc.filenames[0], picked["key"])
            except Exception as e:
                self._units._error = str(e)
                self.parent.logger.exception(f"cannot open {picked['key']}: {e}")
            else:
                self._units._switch(arr)

