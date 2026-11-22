"""The MESc tab: every recording in a ``.mesc`` file, as a table.

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
which picture it was drawn on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui, imgui_ctx

from mbo_utilities import log
from mbo_utilities.annotation.domains import (
    DomainFile,
    find_domains,
    scan_of,
    write_roi_name,
    write_trim,
)
from mbo_utilities.arrays.mesc import MescArray
from mbo_utilities.gui._availability import HAS_SKAN, HAS_VNOISER
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.launch import launch_window
from mbo_utilities.gui.mesc_reference import ReferenceView
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.pipelines import (
    open_pipeline,
    quick_pipelines,
    shown_name,
)
from mbo_utilities.lazy_array import base_array
from mbo_utilities.preferences import get_roi_folder
from mbo_utilities.reader import imread
from mbo_utilities.results import vnoiser_run_for_mesc

logger = log.get("gui.mesc_units")

# opens the reference image on the picture or Z-stack the cell names
IMAGE_ICON = fa.ICON_FA_IMAGE

_ACCENT = imgui.ImVec4(0.8, 0.8, 0.2, 1.0)
_ERROR = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)

UNIT_COLUMNS = (  # (header, hidden by default); columns fit their content and the table scrolls
    # sideways; right-click a header to show or hide one
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
    ("trim", False),
    ("start", False),
    ("comment", False),
)
COLUMN = {name: i for i, (name, _hidden) in enumerate(UNIT_COLUMNS)}
TRIM_WIDTH_EM = 4.0
NAME_WIDTH_EM = 10.0

COLUMN_HELP = {
    # what each column means, on its header
    "session": (
        "MSession_0: scans the operator ran. MSession_1: the picture and RTMC "
        "reference scan MEScan saved for each. Unit numbers restart in each session."
    ),
    "unit": "One recording. Click the row to show it.",
    "modality": (
        "Scan type: raster (timeseries, zstack) or AOD (linescan, chessboard, "
        "ribbon, multicube)."
    ),
    "layout": (
        "Storage layout of the raw pixels: frames, zstack, packed, tiled, boxes or "
        "multicube."
    ),
    "ROIs": "Lines or patches the AOD scanned. Click the count to name them.",
    "picture": (
        "Raster image the ROIs were drawn on (MESc: background image). Click to see "
        "the ROIs on it."
    ),
    "RTMC": "Real-time motion correction: on (yes) or off (no).",
    "T": "Timepoints.",
    "C": "Detector channels (UG green, UR red).",
    "Z": "Z-stack slices, or the ROIs of an AOD scan.",
    "Y": "Rows per frame; 1 for a line scan.",
    "X": "Columns per frame: samples along a line, or pixels across a patch.",
    "fs": "Timepoints per second.",
    "duration": "Recorded length; shorter than planned when the run was stopped early.",
    "trim": (
        "Seconds skipped at the start of this scan. Grey: the file's value. A "
        "negative value resets to it."
    ),
    "start": "Acquisition start, UTC-05:00.",
    "comment": "The comment typed in MESc.",
}

MODALITY_HELP = {
    "timeseries": "Raster scan of the whole field, one frame per timepoint.",
    "zstack": "Raster scan at each depth: one frame per slice, no time axis.",
    "linescan": "AOD scan along drawn lines; each line is a (time, samples) kymograph.",
    "multiline": (
        "AOD line scan that alternates the green and red light paths between cycles."
    ),
    "chessboard": "AOD scan of drawn square patches; each patch is a small movie on Z.",
    "ribbon_transverse": (
        "AOD scan of a strip along a dendrite, swept across its width; each strip is "
        "a patch on Z."
    ),
    "ribbon_longitudinal": "AOD scan of a strip along a dendrite, swept along its length.",
    "multicube": "AOD scan of small volumes, slice by slice; Z is depth.",
}

MESC_DOC = "mesc.md"  # the help page the (?) opens, under assets/docs


def mesc_array_of(obj):
    """The `MescArray` under a viewer array's wrappers and read-time views (``base_array``), or None."""
    arr = base_array(obj)
    return arr if isinstance(arr, MescArray) else None


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
    rate = f"{fs:.0f} Hz" if fs else "rate unknown"
    length = f", {dur:.0f} s" if dur else ""
    name = info["modality_name"]
    n = info.get("n_outlines", 0)
    if info["role"] == "background":
        head = f"Picture: 1 raster frame, {y} x {x} px, {colors}."
    elif info["role"] == "motionCorrection":
        head = f"RTMC reference scan: {t} frames, {y} x {x} px."
    elif name in ("linescan", "multiline"):
        head = f"Line scan: {n or z} lines x {x} samples, {rate}{length}, {colors}."
    elif name in ("chessboard", "ribbon_transverse", "ribbon_longitudinal"):
        head = (
            f"{name.split('_')[0].capitalize()}: {n or z} patches, {y} x {x} px, "
            f"{rate}{length}, {colors}."
        )
    elif name == "zstack":
        head = f"Z-stack: {z} slices, {y} x {x} px, {colors}."
    elif t == 1:
        head = f"Picture: 1 raster frame, {y} x {x} px, {colors}."
    else:
        head = f"Raster movie: {t} frames, {y} x {x} px, {rate}{length}, {colors}."
    lines = [info["key"], head]
    bg = info.get("background_unit")
    if bg:
        lines.append(f"Picture: {bg}" + ("" if bg in by_key else " (not in this file)"))
    ref = info.get("rtmc_unit")
    if rtmc_on(info) or ref:
        lines.append(
            f"RTMC {'on' if rtmc_on(info) else 'off'}"
            + (f"; reference scan {ref}" if ref else "")
        )
    for role, field in (
        ("Picture of", "scans"),
        ("RTMC reference scan of", "rtmc_of"),
    ):
        if info.get(field):
            lines.append(f"{role} " + ", ".join(info[field]))
    if info.get("comment"):
        lines.append(f"Comment: {info['comment']}")
    return "\n".join(lines)


def unit_row(info: dict, trim_s: float = 0.0) -> tuple[tuple[str, ...], tuple]:
    """One table row per `list_mesc_units` entry: the cell texts and the sort
    keys, both in UNIT_COLUMNS order (numbers sort as numbers). ``trim_s`` is
    the unit's trim in the domain file.
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
        f"{trim_s:g}",
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
        trim_s,
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
        self._show_companions = False
        # pictures and RTMC reference units as rows of their own, not cells
        self._parked: dict[tuple[str, str], tuple] = {}
        # the Manual ROI state of every unit left for another, by (file, unit)
        self._reference: ReferenceView | None = None
        # the reference-image popup, drawn from the strip's hook whatever tab
        # is up; the host offers it wherever a recording's ROIs are on screen
        self._curation: tuple[str, Path | None] | None = None
        self._trims: tuple[Path, int, DomainFile] | None = None
        self._naming: dict | None = None
        # the open file's domain file as last read: (path, mtime, contents)
        # the vnoiser run beside the open file the curation window can take,
        # looked up once per file: (file, PF folder or results file, or None)
        parent.reference_view = self.open_reference
        strip = getattr(parent, "top_strip", None)
        if strip is not None:
            strip.add_hook(self._frame)

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        iw = getattr(parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        return bool(len(data)) and mesc_array_of(data[0]) is not None

    def _frame(self) -> None:
        if self._reference is not None:
            self._reference.draw()

    def open_reference(self) -> None:
        """The popup with the shown unit's lines or patches drawn on the
        picture they were drawn on and every Z-stack it matches
        (``mesc_reference``); the tab says so when there is no picture.
        """
        mesc = self._mesc
        if self._reference is None:
            self._reference = ReferenceView(self.parent)
        self._note = None
        if not self._reference.open(mesc, self._open_unit):
            self._note = (
                f"{mesc.unit_key.rsplit('/', 1)[-1]} has no picture."
            )

    @property
    def _mesc(self):
        iw = getattr(self.parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        return mesc_array_of(data[0]) if len(data) else None

    def _trim_book(self, path) -> DomainFile:
        """The open file's domain file, re-read when it changes on disk (the
        dashboard writes trims too); an empty one when there is none.
        """
        found = find_domains(path, get_roi_folder() or None)
        if found is None:
            return DomainFile()
        stamp = found.stat().st_mtime_ns
        if self._trims is None or self._trims[:2] != (found, stamp):
            self._trims = (found, stamp, DomainFile.read(found))
        return self._trims[2]

    def _draw_trim_cell(self, path, info: dict, book: DomainFile) -> None:
        scan = scan_of(info["key"])
        own = scan in book.scan_trim
        imgui.set_next_item_width(imgui.get_font_size() * TRIM_WIDTH_EM)
        if not own:
            imgui.push_style_color(
                imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
            )
        _, value = imgui.input_float(
            f"##trim_{info['key']}", book.trim_of(scan), 0.0, 0.0, "%g"
        )
        if not own:
            imgui.pop_style_color()
        if imgui.is_item_deactivated_after_edit():
            write_trim(
                path,
                None if value < 0 else value,
                info["key"],
                get_roi_folder() or None,
            )

    def _draw_names(self, path, info: dict, book: DomainFile, units: list) -> None:
        """The ROI names popup: one name per line or patch of ``info``'s scan, written to the domain file as each is entered."""
        n, kind = info["n_outlines"], info["outline_kind"]
        names = book.roi_names(scan_of(info["key"]))
        imgui.text_disabled(f"{info['munit']}: ROIs sharing a name form one domain")
        for k in range(n):
            imgui.align_text_to_frame_padding()
            imgui.text(f"ROI {k}")
            imgui.same_line(imgui.get_font_size() * 4)
            imgui.set_next_item_width(imgui.get_font_size() * NAME_WIDTH_EM)
            _, name = imgui.input_text(f"##roi_name_{k}", names.get(k, f"roi{k}"))
            if not imgui.is_item_deactivated_after_edit():
                continue
            scans = [
                u["key"]
                for u in units
                if (u["session"], u.get("n_outlines"), u.get("outline_kind"))
                == (info["session"], n, kind)
            ]
            self._note = None
            try:
                write_roi_name(
                    path, info["key"], k, name, n, scans, get_roi_folder() or None
                )
            except ValueError as e:
                self._note = str(e)
                continue
            traces = getattr(self.parent, "linescan_traces", None)
            if traces is not None:
                traces.relabel()
            if self._reference is not None:
                self._reference.reload_domains()

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
        arr = cache.get((str(path), key))
        if arr is None:
            arr = imread(path, unit=key)
            cache[(str(path), key)] = arr
        return arr

    def _install(self, arr) -> None:
        """Show `arr` in the viewer, re-deriving every per-dataset display state.

        Each unit is an unrelated recording, so the Manual ROI widget is
        rebuilt for it: the outgoing unit's ROIs, runs and traces are parked
        under its key (``detach_roi_widget``) and the incoming unit's, parked
        earlier or saved under its name, are adopted
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
            self._parked[(path, shown)] = (
                parent._manual_roi_store,
                parent._manual_roi_runs,
            )
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
        if self._reference is not None and self._reference.is_open:
            self.open_reference()
        parent.logger.info(f"MESc unit: {arr.unit_key}  shape={arr.shape}")

    def _switch(self, info: dict) -> None:
        """Open the unit ``info`` describes and show it; a failure is shown in the tab."""
        mesc = self._mesc
        self._error = None
        try:
            self._install(self._open_unit(mesc.filenames[0], info["key"]))
        except Exception as e:
            self._error = str(e)
            self.parent.logger.exception(
                f"MESc unit switch to {info['key']} failed: {e}"
            )

    def draw(self) -> None:
        with imgui_ctx.begin_child(
            "##MescContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            mesc = self._mesc
            if mesc is None:
                imgui.text_disabled("No .mesc file is open.")
                return
            units = mesc.units
            by_key = {u["key"]: u for u in units}
            self._cache().setdefault(mesc.unit_key, mesc)
            # the unit opened at launch belongs in the cache too, so switching
            # away and back reuses it instead of opening the file a second time
            split = len(self.parent.image_widget.data) > 1
            # `--roi 0` fans the ROIs of one unit across several subplots;
            # swapping would replace only the first and strand the rest
            paired = companions(units)
            folded = {} if self._show_companions else paired
            owner = folded.get(mesc.unit_key)
            # a folded picture or reference unit on screen highlights its scan's row
            highlight = owner or mesc.unit_key
            shown = mesc.unit_key.rsplit("/", 1)[-1]
            if owner is not None:
                what = (
                    "picture"
                    if by_key[owner].get("background_unit") == mesc.unit_key
                    else "RTMC reference scan"
                )
                shown = f"{shown}, the {what} of {owner.rsplit('/', 1)[-1]}"

            imgui.text_colored(_ACCENT, Path(mesc.filenames[0]).name)
            imgui.same_line(0, 12)
            imgui.text_disabled(
                f"{len(units) - len(folded)} units · showing {shown}"
            )
            imgui.same_line(0, 12)
            if imgui.small_button("?##mesc_help"):
                self.parent._show_help_popup = True
                self.parent._help_select_doc = MESC_DOC
            set_tooltip(
                "Open the MESc file guide.",
                show_mark=False,
            )
            unit_name = shown_name(self.parent)
            # every pipeline that can set itself to the unit on screen gets a
            # button here; the pipeline decides, the tab names none
            for cls in quick_pipelines(self.parent):
                imgui.same_line(0, 12)
                if imgui.small_button(f"{cls.name} on {unit_name}##quick_{cls.name}"):
                    open_pipeline(self.parent, cls.name, seed=True)
                set_tooltip(
                    f"Open {cls.name} in the Process tab, set to {unit_name} and the "
                    "ROI and channel on screen.",
                    show_mark=False,
                )
            imgui.same_line(0, 12)
            if imgui.small_button("Dashboard##mesc_dashboard"):
                from mbo_utilities.gui.dashboard_app import launch_dashboard

                # its module brings hello_imgui; the window is its own process
                launch_dashboard(mesc.filenames[0], owner or mesc.unit_key)
            set_tooltip(
                f"Open {unit_name}'s ROI traces in the dashboard window (`mbo dashboard`).",
                show_mark=False,
            )
            path = Path(mesc.filenames[0])
            if self._curation is None or self._curation[0] != str(path):
                self._curation = (str(path), vnoiser_run_for_mesc(path))
            if HAS_VNOISER and self._curation[1] is not None:
                imgui.same_line(0, 12)
                if imgui.small_button("Curation##mesc_curate"):
                    from mbo_utilities.gui.curation_viewer import launch_curation_window

                    # its module brings hello_imgui; the window is its own process
                    launch_curation_window(self._curation[1])
                set_tooltip(
                    f"Open {self._curation[1].name} in the curation window (`mbo curate`).",
                    show_mark=False,
                )
            if HAS_SKAN and by_key[mesc.unit_key].get("kind") == "zstack":
                imgui.same_line(0, 12)
                if imgui.small_button("Annotate in 3D##mesc_annotate"):
                    # the window is its own process, through `mbo stack-annotate`
                    launch_window(
                        "mbo_utilities.cli",
                        ["stack-annotate", str(path), "--unit", mesc.unit_key],
                        f"stack_annotate_{path.stem}",
                    )
                set_tooltip(
                    f"Open {unit_name} in the stack annotator (`mbo stack-annotate`): XY, XZ, "
                    "YZ and 3D views, a brush on all three, the trace. Its ROIs are this "
                    "unit's rois zarr; edit them in one window at a time.",
                    show_mark=False,
                )
            imgui.same_line(0, 12)
            if split:
                imgui.text_disabled("Split ROIs: reopen without --roi to switch units.")
            else:
                imgui.text_disabled("Click a row to show it.")
            if paired:
                imgui.same_line(0, 12)
                sessions = ", ".join(sorted({key.split("/")[0] for key in paired}))
                changed, on = imgui.checkbox(
                    f"show {sessions} units##companions", self._show_companions
                )
                set_tooltip(
                    "List the picture and RTMC reference scan MEScan saved for each "
                    f"scan ({sessions}) as rows.",
                    show_mark=False,
                )
                if changed:
                    self._show_companions = on
            book = self._trim_book(path)
            imgui.align_text_to_frame_padding()
            imgui.text("trim start (s)")
            imgui.same_line()
            imgui.set_next_item_width(imgui.get_font_size() * TRIM_WIDTH_EM)
            _, value = imgui.input_float("##trim_file", book.trim_s, 0.0, 0.0, "%g")
            if imgui.is_item_deactivated_after_edit():
                write_trim(path, max(value, 0.0), None, get_roi_folder() or None)
            set_tooltip(
                "Seconds the dashboard and vnoiser skip at the start of every scan "
                "in this file.",
                show_mark=False,
            )
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
            if not imgui.begin_table(
                # a new table id whenever the columns change: imgui restores a
                # saved layout by column index, so the old widths would land on
                # the wrong columns
                "##mesc_units_v6",
                len(UNIT_COLUMNS),
                flags,
                imgui.ImVec2(0, avail.y),
            ):
                return
            imgui.table_setup_scroll_freeze(2, 1)
            # the header row and the session and unit columns stay put while scrolling
            for i, (name, hidden) in enumerate(UNIT_COLUMNS):
                column_flags = imgui.TableColumnFlags_.width_fixed
                if i == 0:
                    column_flags |= (
                        # the row's selectable lives here; hidden, a row takes no clicks
                        imgui.TableColumnFlags_.default_sort
                        | imgui.TableColumnFlags_.no_hide
                    )
                if hidden:
                    column_flags |= imgui.TableColumnFlags_.default_hide
                imgui.table_setup_column(name, column_flags)
            imgui.table_next_row(imgui.TableRowFlags_.headers)
            # the header row by hand, so each header carries its column's meaning
            for i, (name, _hidden) in enumerate(UNIT_COLUMNS):
                if not imgui.table_set_column_index(i):
                    continue
                imgui.table_header(name)
                set_tooltip(COLUMN_HELP[name], show_mark=False)
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
                    (*unit_row(u, book.trim_of(scan_of(u["key"]))), u)
                    for u in units
                    if u["key"] not in folded
                ),
                key=lambda row: row[1][column],
                reverse=not ascending,
            )
            picked = None
            reference = None
            naming = None
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
                    imgui.SelectableFlags_.span_all_columns
                    # the row spans the buttons' columns too; without overlap
                    # it takes the hover and a button never sees a click
                    | imgui.SelectableFlags_.allow_overlap,
                )
                if clicked and not split and info["key"] != mesc.unit_key:
                    picked = info
                for i in range(1, len(cells)):
                    if not imgui.table_next_column():
                        continue
                    if i == COLUMN["unit"]:
                        imgui.text(cells[i])
                        set_tooltip(describe_unit(info, by_key), show_mark=False)
                        continue
                    if i == COLUMN["modality"]:
                        imgui.text(cells[i])
                        if cells[i] in MODALITY_HELP:
                            set_tooltip(MODALITY_HELP[cells[i]], show_mark=False)
                        continue
                    if i == COLUMN["picture"] and info.get("background_unit"):
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
                        set_tooltip(
                            f"Show this scan's {what} on {bg} and on each matching Z-stack.",
                            show_mark=False,
                        )
                        continue
                    if i == COLUMN["ROIs"] and info.get("n_outlines"):
                        if imgui.small_button(f"{cells[i]}##rois_{info['key']}"):
                            naming = info
                        set_tooltip(f"Name this scan's {what}.", show_mark=False)
                        continue
                    if i == COLUMN["trim"]:
                        self._draw_trim_cell(path, info, book)
                        continue
                    if i == COLUMN["RTMC"]:
                        imgui.text(cells[i])
                        if info.get("rtmc"):
                            tip = "RTMC on. Its X, Y, Z shifts (um) are the MC plot under the traces."
                        elif info.get("rtmc_armed"):
                            tip = "RTMC on; it recorded no shifts."
                        else:
                            tip = "RTMC off."
                        ref = info.get("rtmc_unit")
                        if ref:
                            tip += f"\nReference scan: {ref}."
                        set_tooltip(tip, show_mark=False)
                        continue
                    imgui.text(cells[i])
                    if i == last and info["comment"] and imgui.is_item_hovered():
                        imgui.set_tooltip(info["comment"])
            imgui.end_table()

            if naming is not None:
                self._naming = naming
                imgui.open_popup("##roi_names")
            if self._naming is not None and imgui.begin_popup("##roi_names"):
                self._draw_names(path, self._naming, book, units)
                imgui.end_popup()
            if picked is not None:
                # swapping rebuilds the panel widgets; finish the frame on the old ones
                self._switch(picked)
            if reference is not None:
                if reference["key"] != mesc.unit_key:
                    self._switch(reference)
                self.open_reference()

    def cleanup(self) -> None:
        strip = getattr(self.parent, "top_strip", None)
        if strip is not None:
            strip.remove_hook(self._frame)
        if self._reference is not None:
            self._reference.cleanup()
            self._reference = None
        self.parent.reference_view = None
        for arr in (getattr(self.parent, "_mesc_unit_cache", None) or {}).values():
            try:
                arr.close()
            except Exception:
                pass
        self.parent._mesc_unit_cache = None
        self._parked.clear()
