"""The MESc app: every recording in a ``.mesc`` file, as a table.

A ``.mesc`` holds one measurement unit per scan the operator ran — a z-stack,
a line scan, a picture — and they are unrelated recordings with different
shapes. The launch picker chooses the first one to open; the MESc tab lists
every unit and switches the viewer between them in place, keeping the units
it opened.

The table has one row per recording. The two units MEScan saves beside a
scan on its own, the picture its lines or patches were drawn on
(``BackgroundImagePath``, role ``background``) and the small reference region
RTMC re-scanned every cycle (``MotionCorrectionImagePath``, role
``motionCorrection``), fold into the scan's row (:func:`companions`) as its
``picture`` and ``RTMC`` cells. ``RTMC`` is yes or no. ``COLUMN_HELP`` puts
each column's meaning on its header, :func:`describe_unit` puts a
recording's on its name, and the ``?`` opens ``assets/docs/mesc.md``.

The ``picture`` cell is one button: it opens ``gui.mesc_reference``, the
picture the scan's lines or patches were drawn on with them drawn, in a popup
the top strip's hook redraws every frame. Everything about where a scan sits
lives in that popup, the Z-stack around it included: the table says only
which picture it was drawn on. The popup's own button displays the image's
unit in the viewer, a Z-stack at the slice the ROIs sit on, which the tab
applies after the popup has drawn (``_pending``) rather than mid-frame.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui, imgui_ctx

from mbo_utilities import log
from mbo_utilities.arrays.mesc import MescArray
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.mesc_reference import ReferenceView, roi_slider
from mbo_utilities.lazy_array import base_array

logger = log.get("gui.app.mesc")

# opens the reference image on the picture or Z-stack the cell names
IMAGE_ICON = fa.ICON_FA_IMAGE

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
    ("picture", False),
    ("RTMC", False),
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
UNIT_COLUMN = next(
    i for i, (name, _hidden) in enumerate(UNIT_COLUMNS) if name == "unit"
)
MODALITY_COLUMN = next(
    i for i, (name, _hidden) in enumerate(UNIT_COLUMNS) if name == "modality"
)
PICTURE_COLUMN = next(
    i for i, (name, _hidden) in enumerate(UNIT_COLUMNS) if name == "picture"
)
RTMC_COLUMN = next(
    i for i, (name, _hidden) in enumerate(UNIT_COLUMNS) if name == "RTMC"
)

# what each column means, on its header
COLUMN_HELP = {
    "session": (
        "MESc groups recordings into sessions. MSession_0 holds what the operator "
        "ran. MEScan saves the picture and the RTMC reference pixels of each scan "
        "into MSession_1 on its own. Units are numbered from 0 in every session, so "
        "MSession_0/MUnit_3 and MSession_1/MUnit_3 are different recordings."
    ),
    "unit": "One recording. Hover a name for what it is. Click the row to display it.",
    "modality": (
        "How the laser moved. A raster scan sweeps the whole field (timeseries, "
        "zstack). An AOD scan visits only the lines or patches the operator drew "
        "(linescan, chessboard, ribbon). Hover a value for details."
    ),
    "layout": (
        "How the reader unpacks the raw pixels: frames (one picture per timepoint), "
        "zstack (one picture per depth), packed (every line's samples stacked into "
        "one tall page), tiled (patches side by side on one page), boxes (patches "
        "cut out of a page), multicube (slices interleaved along time)."
    ),
    "ROIs": (
        "What the AOD scanned: the count of lines (line scan) or patches (chessboard, "
        "ribbon: a patch is one small rectangle) the operator drew on the picture. "
        "Each one is a step of the viewer's ROI slider."
    ),
    "picture": (
        "The raster picture MEScan took just before the scan; the operator drew the "
        "lines or patches on it. Click to see them drawn on it, and on the Z-stack "
        "taken around them when there is one. MESc calls it the background image "
        "and saves it as its own unit, which this table folds into the scan's row."
    ),
    "RTMC": (
        "Real-time motion correction: whether it was on for this recording. yes: "
        "the microscope tracked the tissue while scanning, and if it moved the "
        "scan the X, Y, Z shifts are the MC plot under the traces. no: it was off."
    ),
    "T": "Timepoints.",
    "C": "Colors: the detector channels (UG green, UR red).",
    "Z": "Slices of a Z-stack, or the lines or patches of an AOD scan.",
    "Y": "Rows per frame. A line scan has 1: each cycle records one row of samples per line.",
    "X": "Columns per frame: samples along a line, or pixels across a patch.",
    "fs": "Timepoints per second.",
    "duration": "Recorded length. Shorter than the planned length when the run was stopped early.",
    "start": "Acquisition start, UTC-05:00.",
    "comment": "The comment typed in MESc.",
}

MODALITY_HELP = {
    "timeseries": (
        "A raster scan of the whole field, one frame per timepoint. With one frame it "
        "is a picture."
    ),
    "zstack": "A raster scan repeated at each depth: one frame per slice, no time axis.",
    "linescan": (
        "An AOD scan along straight lines drawn on the picture. Every cycle the "
        "laser runs along each line once and records one row of samples per line, "
        "so each line becomes a (time, samples) image, a kymograph."
    ),
    "multiline": (
        "An AOD line scan with dichroic switching: alternate cycles go to the green "
        "and the red light path, so the two colors are recorded at different "
        "timepoints."
    ),
    "chessboard": (
        "An AOD scan of small squares (patches) drawn on the picture. Every cycle the "
        "laser rasters each square once, so each square is a small movie. MESc packs "
        "the squares side by side on disk; the reader cuts them apart onto the Z "
        "axis."
    ),
    "ribbon_transverse": (
        "An AOD scan of a bent strip drawn along a dendrite, swept across its width "
        "every cycle; each strip is a patch on the Z axis."
    ),
    "ribbon_longitudinal": "An AOD scan of a bent strip swept along its length every cycle.",
    "multicube": "An AOD scan of small volumes, each scanned slice by slice, so Z is real depth.",
}

# the help page the (?) opens, under assets/docs
MESC_DOC = "mesc.md"


def mesc_array_of(obj):
    """The `MescArray` behind a viewer array, or None.

    Peels the display wrappers (`_SqueezeSingletonDims`, `_ScrubTimingProxy`,
    …), each of which exposes its source as ``_arr``.
    """
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


def companions(units: list[dict]) -> dict[str, str]:
    """The units the table folds into a scan's row: every picture
    (``BackgroundImagePath``) and RTMC reference unit
    (``MotionCorrectionImagePath``) a scan names, as ``{companion key: scan
    key}``. A scan named by another scan, or a Z-stack, is never folded.
    """
    out = {}
    for u in units:
        owners = [*u["scans"], *u["rtmc_of"]]
        if owners and u["role"] != "measurement" and u["kind"] == "frames":
            out[u["key"]] = owners[0]
    return out


def rtmc_on(info: dict) -> bool:
    """Whether RTMC was on for a recording: it carries correction curves,
    with samples (it moved the scan) or without (it never had to).
    """
    return bool(info.get("rtmc_armed") or info.get("rtmc"))


def describe_unit(info: dict, by_key: dict[str, dict]) -> str:
    """One recording in plain words, for the unit cell's tooltip: what the
    laser did, how much was recorded, and the picture and RTMC reference
    unit MEScan saved beside it.
    """
    t, c, z, y, x = info["shape"]
    fs = info.get("fs")
    dur = info.get("duration_s")
    colors = f"{c} color" + ("s" if c != 1 else "")
    rate = f"{fs:.0f} times a second" if fs else "at an unknown rate"
    length = f" for {dur:.0f} s" if dur else ""
    name = info["modality_name"]
    n = info.get("n_outlines", 0)
    if info["role"] == "background":
        head = f"The picture the lines or patches were drawn on: one raster frame of {y} x {x} px, {colors}."
    elif info["role"] == "motionCorrection":
        head = (
            "RTMC reference pixels: the small region the microscope re-scanned every "
            f"cycle to measure drift. {t} frames of {y} x {x} px; a line scan's are two "
            "reference lines stored as rows, not a movie of the field."
        )
    elif name in ("linescan", "multiline"):
        head = (
            f"Line scan: {n or z} lines of {x} samples, 1 px wide, {rate}{length}, "
            f"{colors}. Each line is a (time, samples) image."
        )
    elif name in ("chessboard", "ribbon_transverse", "ribbon_longitudinal"):
        head = (
            f"{name.split('_')[0].capitalize()}: {n or z} patches of {y} x {x} px, "
            f"{rate}{length}, {colors}. Each patch is a small movie."
        )
    elif name == "zstack":
        head = f"Z-stack: {z} raster frames of {y} x {x} px, one per depth, {colors}."
    elif t == 1:
        head = f"Picture: one raster frame of {y} x {x} px, {colors}."
    else:
        head = f"Raster movie: {t} frames of {y} x {x} px, {rate}{length}, {colors}."
    lines = [info["key"], head]
    bg = info.get("background_unit")
    if bg:
        lines.append(f"Picture: {bg}" + ("" if bg in by_key else " (not in this file)"))
    ref = info.get("rtmc_unit")
    if rtmc_on(info) or ref:
        lines.append(
            f"RTMC {'on' if rtmc_on(info) else 'off'}"
            + (f"; reference pixels {ref}" if ref else "")
        )
    for role, field in (
        ("Picture of", "scans"),
        ("RTMC reference pixels of", "rtmc_of"),
    ):
        if info.get(field):
            lines.append(f"{role} " + ", ".join(info[field]))
    if info.get("comment"):
        lines.append(f"Comment: {info['comment']}")
    return "\n".join(lines)


def unit_row(info: dict) -> tuple[tuple[str, ...], tuple]:
    """One table row per `list_mesc_units` entry: the cell texts and the sort
    keys, both in UNIT_COLUMNS order (numbers sort as numbers).
    """
    t, c, z_size, y, x = info["shape"]
    fs = info.get("fs")
    dur = info.get("duration_s")
    start = (info.get("start_time") or "")[:19].replace("T", " ")
    comment = " / ".join(info.get("comment", "").splitlines())
    n = info.get("n_outlines", 0)
    nouns = {"line": ("line", "lines"), "patch": ("patch", "patches")}
    rois = f"{n} {nouns[info['outline_kind']][n != 1]}" if n else "-"
    picture = info.get("background_unit")
    picture_text = picture.rsplit("/", 1)[-1] if picture else "-"
    cells = (
        info["session"],
        info["munit"],
        info["modality_name"],
        info["kind"],
        rois,
        picture_text,
        "yes" if rtmc_on(info) else "no",
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
        picture_text,
        int(rtmc_on(info)),
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


class MescApp(App):
    """Every measurement unit in the open ``.mesc`` with its comment; the
    displayed unit is highlighted and clicking a row shows it, and the
    picture button shows where its lines or patches were drawn.

    The first panel for a ``.mesc``, ahead of the viewer's: the file is a
    set of recordings before it is one picture. The units it opened stay
    open across switches and are closed when another file opens. The
    reference popup draws from the top strip's hook whatever panel is up.
    """

    id = "mesc"
    title = "MESc"
    dock = "right"
    order = 0
    size = 380
    start_open = True

    def __init__(self):
        super().__init__()
        self._error: str | None = None
        self._note: str | None = None
        self._sort = (0, True)
        # pictures and RTMC reference units as rows of their own, not cells
        self._show_companions = False
        # the units opened so far, by (file, unit key)
        self._units: dict[tuple[str, str], MescArray] = {}
        # the reference-image popup, drawn from the strip's hook whatever
        # panel is up; the host offers it wherever a recording's ROIs are on screen
        self._reference: ReferenceView | None = None
        # a unit the popup's button asked for, applied once it has drawn
        self._pending: tuple[dict, int | None] | None = None
        self._hooked = False

    def available(self, host) -> bool:
        return self._mesc is not None

    def frame(self, host) -> None:
        if not self._hooked:
            host.strip.add_hook(self._strip_hook)
            self._hooked = True

    def data_changed(self, host) -> None:
        if self._mesc is None:
            self.close()

    def _strip_hook(self) -> None:
        if self._reference is not None:
            self._reference.draw()
        # switching rebuilds the panel widgets, so never inside the popup's draw
        if self._pending is not None:
            info, z = self._pending
            self._pending = None
            self._switch(info, z)

    def _show_reference_unit(self, key: str, z: int | None) -> None:
        """The popup's display button: show that picture or Z-stack next frame."""
        info = next((u for u in self._mesc.units if u["key"] == key), None)
        if info is not None and len(self.host.viewer.data) == 1:
            self._pending = (info, z)

    def open_reference(self) -> None:
        """The popup with the shown unit's lines or patches drawn on the
        picture they were drawn on and the Z-stack around them
        (``mesc_reference``); the tab says so when neither exists.
        """
        mesc = self._mesc
        if self._reference is None:
            self._reference = ReferenceView(
                self.host.viewer, on_show=self._show_reference_unit
            )
        self._note = None
        if not self._reference.open(mesc, partial(self._open_unit, mesc.filenames[0])):
            self._note = f"no picture or Z-stack in this file carries {mesc.unit_key.rsplit('/', 1)[-1]}'s ROIs"

    @property
    def _mesc(self) -> MescArray | None:
        data = None if self.host is None else self.host.data
        arr = None if data is None else base_array(data)
        return arr if isinstance(arr, MescArray) else None

    def _open_unit(self, path, key: str) -> MescArray:
        """The `MescArray` for one unit, opening it on first use."""
        arr = self._units.get((str(path), key))
        if arr is None:
            arr = MescArray(path, unit=key)
            self._units[(str(path), key)] = arr
        return arr

    def _install(self, arr, z: int | None = None) -> None:
        """Show `arr` in the viewer at slice ``z``, re-deriving every
        per-dataset display state through ``_show``.
        """
        path = str(arr.filenames[0])
        unit = arr.unit_key.rsplit("/", 1)[-1]
        self._show(arr, f"{Path(path).stem[:16]} · {unit}")

        # a Z-stack opened from a scan's row lands on the slice its ROIs sit
        # on; the swap reset the sliders, so this follows it
        iw = self.host.viewer
        zdim = roi_slider(iw.dim_names) if z is not None else None
        if zdim is not None:
            iw.indices[zdim] = int(z)
        if self._reference is not None and self._reference.is_open:
            self.open_reference()
        logger.info(f"MESc unit: {arr.unit_key}  shape={arr.shape}")

    def _show(self, arr, title: str) -> None:
        """Open `arr` into the host: the viewer swap, each unit's own ROIs
        and a line scan's traces all follow from ``set_data``.
        """
        self.host.set_data(arr)

    def _switch(self, info: dict, z: int | None = None) -> None:
        """Open the unit ``info`` describes and show it at slice ``z``; a
        failure is shown in the tab.
        """
        mesc = self._mesc
        self._error = None
        try:
            self._install(self._open_unit(mesc.filenames[0], info["key"]), z)
        except Exception as e:
            self._error = str(e)
            logger.exception(
                f"MESc unit switch to {info['key']} failed: {e}"
            )

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        with imgui_ctx.begin_child(
            "##MescContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            mesc = self._mesc
            if mesc is None:
                imgui.text_disabled("No .mesc file is open.")
                return
            units = mesc.units
            by_key = {u["key"]: u for u in units}
            # the unit opened at launch belongs in the cache too, so switching
            # away and back reuses it instead of opening the file a second time
            self._units.setdefault((str(mesc.filenames[0]), mesc.unit_key), mesc)
            # `--roi 0` fans the ROIs of one unit across several subplots;
            # swapping would replace only the first and strand the rest
            split = len(host.viewer.data) > 1
            paired = companions(units)
            folded = {} if self._show_companions else paired
            # a folded picture or reference unit on screen highlights its scan's row
            owner = folded.get(mesc.unit_key)
            highlight = owner or mesc.unit_key
            shown = mesc.unit_key.rsplit("/", 1)[-1]
            if owner is not None:
                what = (
                    "picture"
                    if by_key[owner].get("background_unit") == mesc.unit_key
                    else "RTMC reference pixels"
                )
                shown = f"{shown}, the {what} of {owner.rsplit('/', 1)[-1]}"

            imgui.text_colored(_ACCENT, Path(mesc.filenames[0]).name)
            imgui.same_line(0, 12)
            imgui.text_disabled(
                f"{len(units) - len(folded)} recordings · showing {shown}"
            )
            imgui.same_line(0, 12)
            if imgui.small_button("?##mesc_help"):
                host.apps["help"].show(MESC_DOC)
            set_tooltip(
                "What a .mesc holds: sessions, scans, pictures, RTMC, and how each "
                "column reads. Every column header and most cells carry their own tip.",
                show_mark=False,
            )
            imgui.same_line(0, 12)
            if split:
                imgui.text_disabled("Split ROIs: reopen without --roi to switch units.")
            else:
                imgui.text_disabled("Click a row to display that unit.")
            if paired:
                imgui.same_line(0, 12)
                changed, on = imgui.checkbox(
                    "companion units as rows", self._show_companions
                )
                set_tooltip(
                    "Give each scan's picture and RTMC reference pixels a row of their "
                    "own instead of the picture and RTMC cells of the scan's row.",
                    show_mark=False,
                )
                if changed:
                    self._show_companions = on
            if self._note:
                imgui.text_disabled(self._note)
            if self._error:
                imgui.text_colored(_ERROR, "Unit switch failed")
                set_tooltip(self._error)

            flags = (
                imgui.TableFlags_.sortable
                | imgui.TableFlags_.row_bg
                | imgui.TableFlags_.borders_inner_h
                | imgui.TableFlags_.scroll_y
                | imgui.TableFlags_.scroll_x
                | imgui.TableFlags_.resizable
                | imgui.TableFlags_.hideable
                | imgui.TableFlags_.sizing_fixed_fit
            )
            avail = imgui.get_content_region_avail()
            # a new table id whenever the columns change: imgui restores a
            # saved layout by column index, so the old widths would land on
            # the wrong columns
            if not imgui.begin_table(
                "##mesc_units_v5", len(UNIT_COLUMNS), flags, imgui.ImVec2(0, avail.y)
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
            # the header row by hand, so each header carries its column's meaning
            imgui.table_next_row(imgui.TableRowFlags_.headers)
            for i, (name, _hidden) in enumerate(UNIT_COLUMNS):
                if not imgui.table_set_column_index(i):
                    continue
                imgui.table_header(name)
                set_tooltip(
                    COLUMN_HELP[name]
                    + "\n\nRight-click a header to show or hide columns.",
                    show_mark=False,
                )
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
                ((*unit_row(u), u) for u in units if u["key"] not in folded),
                key=lambda row: row[1][column],
                reverse=not ascending,
            )
            picked = None
            reference = None
            last = len(UNIT_COLUMNS) - 1
            for cells, _keys, info in rows:
                what = {"line": "lines", "patch": "patches"}.get(
                    info.get("outline_kind"), "ROIs"
                )
                imgui.table_next_row()
                imgui.table_next_column()
                clicked, _ = imgui.selectable(
                    f"{cells[0]}##unit_{info['key']}",
                    info["key"] == highlight,
                    # the row spans the buttons' columns too; without overlap
                    # it takes the hover and a button never sees a click
                    imgui.SelectableFlags_.span_all_columns
                    | imgui.SelectableFlags_.allow_overlap,
                )
                if clicked and not split and info["key"] != mesc.unit_key:
                    picked = (info, None)
                for i in range(1, len(cells)):
                    if not imgui.table_next_column():
                        continue
                    if i == UNIT_COLUMN:
                        imgui.text(cells[i])
                        set_tooltip(describe_unit(info, by_key), show_mark=False)
                        continue
                    if i == MODALITY_COLUMN:
                        imgui.text(cells[i])
                        if cells[i] in MODALITY_HELP:
                            set_tooltip(MODALITY_HELP[cells[i]], show_mark=False)
                        continue
                    if i == PICTURE_COLUMN and info.get("background_unit"):
                        bg = info["background_unit"]
                        other = by_key.get(bg)
                        if other is None:
                            imgui.text_disabled(cells[i])
                            set_tooltip(f"{bg} is not in this file.", show_mark=False)
                            continue
                        if (
                            imgui.small_button(
                                f"{IMAGE_ICON} {cells[i]}##pic_{info['key']}"
                            )
                            and not split
                        ):
                            reference = info
                        _t, c, _z, y, x = other["shape"]
                        set_tooltip(
                            f"Show this scan's {what} drawn on {bg}, the picture MEScan took "
                            f"just before it ({y} x {x} px, {c} color{'s' if c != 1 else ''}), "
                            "and on the Z-stack taken around them when there is one.",
                            show_mark=False,
                        )
                        continue
                    if i == RTMC_COLUMN:
                        imgui.text(cells[i])
                        if info.get("rtmc"):
                            tip = (
                                "RTMC was on and moved the scan to follow the tissue. The X, Y, Z "
                                "shifts in microns are the MC plot under the traces."
                            )
                        elif info.get("rtmc_armed"):
                            tip = "RTMC was on; the tissue never moved, so there are no shifts to plot."
                        else:
                            tip = "RTMC was off."
                        ref = info.get("rtmc_unit")
                        if ref:
                            tip += (
                                f"\nThe reference region it re-scanned every cycle is {ref}; "
                                "companion units as rows lists it."
                            )
                        set_tooltip(tip, show_mark=False)
                        continue
                    imgui.text(cells[i])
                    if i == last and info["comment"] and imgui.is_item_hovered():
                        imgui.set_tooltip(info["comment"])
            imgui.end_table()

            # swapping rebuilds the panel widgets; finish the frame on the old ones
            if picked is not None:
                self._switch(*picked)
            if reference is not None:
                if reference["key"] != mesc.unit_key:
                    self._switch(reference)
                self.open_reference()

    def close(self) -> None:
        if self._hooked:
            self.host.strip.remove_hook(self._strip_hook)
            self._hooked = False
        if self._reference is not None:
            self._reference.cleanup()
            self._reference = None
        for arr in self._units.values():
            arr.close()
        self._units.clear()
        self._pending = None
        self._error = self._note = None
