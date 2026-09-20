"""MESc widgets: the MESc tab and the ROI overlay.

A ``.mesc`` holds one measurement unit per scan the operator ran — a z-stack,
a ribbon time series, a snapshot — and they are unrelated recordings with
different shapes. The launch picker chooses the first one to open; the MESc
tab lists every unit and switches the viewer between them in place, keeping
the units it opened.

The MESc GUI draws every ROI of a scan on the raster snapshot it was placed
on (the unit's ``BackgroundImagePath``), whatever its depth. The ROI overlay
does the same when that snapshot, or a Z-stack containing the ROIs, is the
displayed unit: the lines and patches are drawn the moment the unit is on
screen, from ``arrays.mesc_geometry.image_overlays`` in the colours MESc
used, solid on the snapshot they were placed on and on the Z-stack slice
they sit on, faint on any other slice. The Image tab's ROI Overlay panel
hides them for the session; the MESc tab's ``depth`` column says where each
scan's ROIs sit. The overlay
exists only when the file pairs ROIs with the image by metadata alone, so a
drawn line is one MESc recorded for exactly this picture.

The tab's Outer view button opens ``gui.mesc_outer_view``: the shown unit's
quick projections and the snapshot and Z-stacks it was scanned on, its ROIs
drawn, in a popup the top strip's hook redraws every frame.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui, imgui_ctx

from mbo_utilities import log
from mbo_utilities.arrays.mesc_geometry import image_overlays, zstack_contents
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.mesc_outer_view import (
    GHOST_ALPHA,
    GHOST_THICKNESS,
    ON_THICKNESS,
    OuterView,
    current_z,
    on_plane,
    z_slider,
)
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.widget_toggles import widget_enabled

logger = log.get("gui.mesc_units")

_ACCENT = imgui.ImVec4(0.8, 0.8, 0.2, 1.0)
_ERROR = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
# in front of the image (0) and the manual-ROI masks (1 .. 1.75)
OVERLAY_Z = 2.0

# (header, hidden by default); columns fit their content and the table scrolls
# sideways; right-click a header to show or hide one
UNIT_COLUMNS = (
    ("session", False),
    ("unit", False),
    ("modality", False),
    ("layout", True),
    ("ROIs", False),
    ("depth", False),
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
DEPTH_COLUMN = next(i for i, (name, _hidden) in enumerate(UNIT_COLUMNS) if name == "depth")
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


def overlay_records(parent: Any) -> list[dict]:
    """:func:`image_overlays` for the displayed unit, cached on the parent per unit."""
    mesc = mesc_array_of(parent.image_widget.data[0])
    if mesc is None or mesc.modality not in (1, 2):
        return []
    key = (str(mesc.filenames[0]), mesc.unit_key)
    cached = getattr(parent, "_mesc_overlay_records", None)
    if cached is None or cached[0] != key:
        cached = (key, image_overlays(mesc.filenames[0], mesc.unit_key, mesc.units))
        parent._mesc_overlay_records = cached
    return cached[1]


def close_overlay(parent: Any) -> None:
    """Drop the live overlay graphics, if any; a unit swap or file load calls this."""
    overlay = getattr(parent, "_mesc_overlay", None)
    if overlay is not None:
        overlay.close()
    parent._mesc_overlay = None


class MescOverlay:
    """The line graphics of one image unit's ROI outlines on the viewer's subplot.

    Kept on the parent (``parent._mesc_overlay``) so it outlives the widget
    rebuild every unit swap triggers.
    """

    def __init__(self, iw, unit_key: str, records: list[dict], show_ghosts: bool):
        from mbo_utilities.annotation.store import CLASS_COLORS

        self.iw = iw
        self.unit_key = unit_key
        self.records = records
        self.show_ghosts = show_ghosts
        self.subplot = iw.figure[0, 0]
        self.zdim = z_slider(iw.dim_names)
        units = sorted({r["unit"] for r in records})
        self.colors = np.array(
            [
                (*(r["color"] or CLASS_COLORS[units.index(r["unit"]) % len(CLASS_COLORS)])[:3], 1.0)
                for r in records
            ],
            dtype=np.float32,
        )
        self.lines = self.subplot.add_line_collection(
            [r["pixels"].astype(np.float32) for r in records],
            colors=self.colors,
            thickness=ON_THICKNESS,
            name="mesc_roi_overlay",
            offset=(0, 0, OVERLAY_Z),
        )
        if self.zdim is not None:
            iw.ndwidget.indices.add_event_handler(self._on_indices)
        self.refresh()

    def _on_indices(self, _indices) -> None:
        self.refresh()

    def refresh(self) -> None:
        z = current_z(self.iw)
        for record, graphic, color in zip(self.records, self.lines.graphics, self.colors):
            on = on_plane(record, z)
            graphic.visible = on or self.show_ghosts
            graphic.colors = (*color[:3], 1.0 if on else GHOST_ALPHA)
            graphic.thickness = ON_THICKNESS if on else GHOST_THICKNESS

    def close(self) -> None:
        if self.zdim is not None:
            self.iw.ndwidget.indices.remove_event_handler(self._on_indices)
        self.subplot.delete_graphic(self.lines)


class MescOverlayWidget(Widget):
    """Checkboxes for the ROIs MESc recorded for the displayed snapshot or Z-stack.

    The graphics are built when the widget is, so a unit opens with its ROIs
    drawn whichever tab is on screen; the panel only hides them.
    """

    name = "ROI Overlay"
    priority = 5
    toggle_key = "preview.mesc_overlay"

    def __init__(self, parent: Any):
        super().__init__(parent)
        self.sync()

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        iw = getattr(parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        # `--roi 0` fans one unit over several subplots; the overlay draws on one
        return len(data) == 1 and bool(overlay_records(parent))

    def sync(self) -> None:
        """Build or drop the graphics to match the displayed unit, the
        Overlay ROIs checkbox and the Widgets menu."""
        parent = self.parent
        mesc = mesc_array_of(parent.image_widget.data[0])
        overlay = getattr(parent, "_mesc_overlay", None)
        if overlay is not None and overlay.unit_key != mesc.unit_key:
            close_overlay(parent)
            overlay = None
        wanted = widget_enabled(self.toggle_key) and bool(getattr(parent, "_mesc_overlay_on", True))
        if wanted and overlay is None:
            parent._mesc_overlay = MescOverlay(
                parent.image_widget,
                mesc.unit_key,
                overlay_records(parent),
                bool(getattr(parent, "_mesc_overlay_ghosts", True)),
            )
        elif not wanted and overlay is not None:
            close_overlay(parent)

    def draw(self) -> None:
        parent = self.parent
        imgui.spacing()
        imgui.text_colored(_ACCENT, "ROI Overlay")
        imgui.spacing()
        changed, wanted = imgui.checkbox(
            "Overlay ROIs", bool(getattr(parent, "_mesc_overlay_on", True))
        )
        set_tooltip(
            "Draw the lines and patches MESc scanned on this image, in the "
            "colours it used. Solid: placed on this snapshot, or on the slice shown "
            "of a Z-stack. Faint: on another slice. The MESc tab's depth column "
            "says where each scan's ROIs sit."
        )
        if changed:
            parent._mesc_overlay_on = wanted
            self.sync()
        overlay = getattr(parent, "_mesc_overlay", None)
        if overlay is None:
            return
        changed, ghosts = imgui.checkbox("Show off-plane ROIs", overlay.show_ghosts)
        set_tooltip("Keep ROIs scanned at another depth visible, faint.")
        if changed:
            parent._mesc_overlay_ghosts = overlay.show_ghosts = ghosts
            overlay.refresh()

    def cleanup(self) -> None:
        close_overlay(self.parent)
        self.parent._mesc_overlay_records = None


def unit_links(info: dict, contains: list[str] | tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """The units one unit is paired with, as ``(role, key)`` pairs, the role
    worded from this unit's side: a scan's snapshot (the image its ROIs were
    placed on, its ``BackgroundImagePath``) and its RTMC stream (the stage
    shifts recorded while it ran), a snapshot's scans, a stream's scan, and
    a Z-stack's scans (``contains``, from ``mesc_geometry.zstack_contents``:
    every scan whose ROIs fall inside the stack's field)."""
    links = []
    if info.get("background_unit"):
        links.append(("snapshot it was drawn on", info["background_unit"]))
    if info.get("rtmc_unit"):
        links.append(("its RTMC motion stream", info["rtmc_unit"]))
    links += [("scan drawn on it", k) for k in info.get("scans", ())]
    links += [("scan it is the RTMC stream of", k) for k in info.get("rtmc_of", ())]
    links += [("scan inside this Z-stack", k) for k in contains]
    return links


def unit_row(
    info: dict,
    contains: list[str] | tuple[str, ...] = (),
    drawn: list[dict] | tuple[dict, ...] = (),
) -> tuple[tuple[str, ...], tuple]:
    """One table row per `list_mesc_units` entry: the cell texts and the sort
    keys, both in UNIT_COLUMNS order (numbers sort as numbers). The ``links``
    cell is how many units this one is paired with (:func:`unit_links`); the
    tab draws it as a button that lists them. ``drawn`` is this unit's overlay
    records on the displayed image (:func:`overlay_records`); the ``depth``
    cell says where they sit: the slices they are on in a Z-stack (and how
    many were scanned outside it), or their offsets from a snapshot's plane."""
    t, c, z_size, y, x = info["shape"]
    fs = info.get("fs")
    dur = info.get("duration_s")
    start = (info.get("start_time") or "")[:19].replace("T", " ")
    comment = " / ".join(info.get("comment", "").splitlines())
    n = info.get("n_outlines", 0)
    nouns = {"line": ("line", "lines"), "patch": ("patch", "patches")}
    rois = f"{n} {nouns[info['outline_kind']][n != 1]}" if n else "-"
    depth, depth_key = "-", float("inf")
    if drawn and drawn[0]["slice"] is None:
        dzs = sorted(r["dz_um"] for r in drawn)
        depth = f"{dzs[0]:+.1f} um" if dzs[-1] - dzs[0] < 0.05 else f"{dzs[0]:+.1f}..{dzs[-1]:+.1f} um"
        depth_key = dzs[0]
    elif drawn:
        slices = sorted({r["slice"] for r in drawn if r["on_plane"]})
        outside = sum(not r["on_plane"] for r in drawn)
        if not slices:
            depth = "outside"
        elif len(slices) == 1:
            depth = f"slice {slices[0] + 1}"
        else:
            depth = f"slices {slices[0] + 1}-{slices[-1] + 1}"
        if slices and outside:
            depth += f" ({outside} outside)"
        depth_key = float(slices[0]) if slices else float("inf")
    links = len(unit_links(info, contains))
    cells = (
        info["session"],
        info["munit"],
        info["modality_name"],
        info["kind"],
        rois,
        depth,
        str(links) if links else "-",
        str(t),
        str(c),
        str(z_size),
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
        depth_key,
        links,
        t,
        c,
        z_size,
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
    comment; the displayed unit is highlighted and clicking a row shows it.

    The first tab for a ``.mesc``, ahead of Image: the file is a set of
    recordings before it is one picture.
    """

    name = "MESc"
    tab_label = "MESc"
    placement = "tab"
    toggle_key = "mesc"
    priority = 5

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._error: str | None = None
        self._note: str | None = None
        self._sort = (0, True)
        # the Manual ROI state of every unit left for another, by (file, unit)
        self._parked: dict[tuple[str, str], tuple] = {}
        # the outer-view popup, drawn from the strip's hook whatever tab is up;
        # the host offers it wherever a recording's ROIs are on screen
        self._outer: OuterView | None = None
        parent.outer_view = self.open_outer_view
        strip = getattr(parent, "top_strip", None)
        if strip is not None:
            strip.add_hook(self._frame)

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        iw = getattr(parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        return bool(len(data)) and mesc_array_of(data[0]) is not None

    def _frame(self) -> None:
        if self._outer is not None:
            self._outer.draw()

    def open_outer_view(self) -> None:
        """The popup around the shown unit (``mesc_outer_view``); the tab
        says so when there is nothing to show."""
        mesc = self._mesc
        if self._outer is None:
            self._outer = OuterView(self.parent)
        self._note = None
        if not self._outer.open(mesc, partial(self._open_unit, mesc.filenames[0])):
            self._note = (
                f"nothing around {mesc.unit_key.rsplit('/', 1)[-1]}: one frame, "
                "and no snapshot or Z-stack holds it"
            )

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

    def _install(self, arr) -> None:
        """Show `arr` in the viewer, re-deriving every per-dataset display state.

        Each unit is an unrelated recording, so the Manual ROI widget is
        rebuilt for it: the outgoing unit's ROIs, runs and traces are parked
        under its key (``detach_roi_widget``) and the incoming unit's, parked
        earlier or autosaved beside the file under its name, are adopted
        (``attach_roi_widget``); a line-scan unit's own traces follow the
        same way.
        """
        from mbo_utilities.gui._dialogs import swap_viewer_array
        from mbo_utilities.gui.linescan_viewer import attach_standard_traces
        from mbo_utilities.gui.manual_roi import attach_roi_widget, detach_roi_widget

        parent = self.parent
        path = str(arr.filenames[0])
        shown = self._mesc.unit_key
        traces = getattr(parent, "linescan_traces", None)
        if traces is not None:
            traces.close()
            parent.linescan_traces = None
        roi_on = getattr(parent, "manual_roi", None) is not None
        if roi_on:
            detach_roi_widget(parent)
            self._parked[(path, shown)] = (parent._manual_roi_store, parent._manual_roi_runs)
            parent._manual_roi_store, parent._manual_roi_runs = self._parked.get(
                (path, arr.unit_key), (None, None)
            )

        unit = arr.unit_key.rsplit("/", 1)[-1]
        swap_viewer_array(parent, arr, title=f"{Path(path).stem[:16]} · {unit}")

        if roi_on:
            attach_roi_widget(parent)
        try:
            attach_standard_traces(parent)
        except Exception:
            parent.logger.warning("line-scan traces tab unavailable", exc_info=True)
        if self._outer is not None and self._outer.is_open:
            self.open_outer_view()
        parent.logger.info(f"MESc unit: {arr.unit_key}  shape={arr.shape}")

    def _switch(self, info: dict) -> None:
        """Open the unit ``info`` describes and show it; a failure is shown in the tab."""
        mesc = self._mesc
        self._error = None
        try:
            self._install(self._open_unit(mesc.filenames[0], info["key"]))
        except Exception as e:
            self._error = str(e)
            self.parent.logger.exception(f"MESc unit switch to {info['key']} failed: {e}")

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
            mesc = self._mesc
            if mesc is None:
                imgui.text_disabled("No .mesc file is open.")
                return
            units = mesc.units
            contains = self._contains(mesc)
            # the unit opened at launch belongs in the cache too, so switching
            # away and back reuses it instead of opening the file a second time
            self._cache().setdefault(mesc.unit_key, mesc)
            # `--roi 0` fans the ROIs of one unit across several subplots;
            # swapping would replace only the first and strand the rest
            split = len(self.parent.image_widget.data) > 1
            shown = mesc.unit_key.rsplit("/", 1)[-1]
            drawn: dict[str, list[dict]] = {}
            records = overlay_records(self.parent) if not split else []
            for r in records:
                drawn.setdefault(r["unit"], []).append(r)
            z = current_z(self.parent.image_widget) if records else 0

            imgui.text_colored(_ACCENT, Path(mesc.filenames[0]).name)
            imgui.same_line(0, 12)
            imgui.text_disabled(f"{len(units)} units · showing {shown}")
            if imgui.small_button("Outer view##outer"):
                self.open_outer_view()
            set_tooltip(
                "Where this unit sits: its quick mean / max / std over time at the slice "
                "on screen, and the snapshot and Z-stacks it was scanned on with its ROIs "
                "drawn, the slider's ROI thicker.",
                show_mark=False,
            )
            imgui.same_line(0, 12)
            if split:
                imgui.text_disabled("Split ROIs: reopen without --roi to switch units.")
            else:
                imgui.text_disabled("Click a row to display that unit.")
            if self._note:
                imgui.text_disabled(self._note)
            if self._error:
                imgui.text_colored(_ERROR, "Unit switch failed")
                set_tooltip(self._error)

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
                "##mesc_units_v3", len(UNIT_COLUMNS), flags, imgui.ImVec2(0, avail.y)
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
                (
                    (*unit_row(u, contains.get(u["key"], ()), drawn.get(u["key"], ())), u)
                    for u in units
                ),
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
                    if i == DEPTH_COLUMN and drawn.get(info["key"]):
                        rs = drawn[info["key"]]
                        imgui.text(cells[i])
                        stack = rs[0]["slice"] is not None
                        header = (
                            f"{shown} slice {z + 1}: {sum(on_plane(r, z) for r in rs)} of {len(rs)} on it"
                            if stack else f"{shown}: {len(rs)} ROIs placed on it, offsets from its plane"
                        )
                        set_tooltip(
                            "\n".join(
                                [header]
                                + [
                                    f"ROI {r['roi'] + 1}: z {r['z_um']:+.1f} um, {r['dz_um']:+.1f} um off"
                                    + ("" if r["slice"] is None else f", slice {r['slice'] + 1}")
                                    + ("" if r["on_plane"] or not stack else " (outside the stack)")
                                    for r in rs
                                ]
                            ),
                            show_mark=False,
                        )
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

            # swapping rebuilds the panel widgets; finish the frame on the old ones
            if picked is not None:
                self._switch(picked)

    def cleanup(self) -> None:
        strip = getattr(self.parent, "top_strip", None)
        if strip is not None:
            strip.remove_hook(self._frame)
        if self._outer is not None:
            self._outer.cleanup()
            self._outer = None
        self.parent.outer_view = None
        for arr in (getattr(self.parent, "_mesc_unit_cache", None) or {}).values():
            try:
                arr.close()
            except Exception:
                pass
        self.parent._mesc_unit_cache = None
        self._parked.clear()
