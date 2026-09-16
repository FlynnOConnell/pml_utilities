"""Event curation of voltage traces with vnoiser, inside the viewer.

The curation notebook's dashboard as one viewer panel, over every recording
a data path holds. Scanning a path catalogs each animal, experiment and
scan / domain the pipeline processed and loads them in the background. The
``Curation`` panel on the top strip shows one recording at a time in the
notebook's grid: the trace with its candidates (A) beside the threshold /
slider card (A1 to A4), then the template, focused candidate and PCA (B to
D) under it; the arrows flip through recordings. The Curation tab on the
right bar holds the Decision and Recordings tabs (labels, navigation, the
table, the mode). Every plot reads the focused session, so a
label, a threshold or a flip updates all of them at once. Every rule comes from
``vnoiser.curation`` through :class:`mbo_utilities.vnoiser.CurationSession`:
one per mode and recording, each with its own JSON file, as the notebook's
sections have.
"""

from __future__ import annotations

import logging
from functools import partial
import queue
import threading
import time
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui, imgui_ctx, implot, portable_file_dialogs as pfd

from mbo_utilities.gui import _theme as theme
from mbo_utilities.gui._files import PathPrompt, draw_path_prompt
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui._theme import card, em, section
from mbo_utilities.gui._top_strip import TopPanel, TopStrip
from mbo_utilities.gui.imgui.lines import (
    decimate_minmax, dotted_vline, drag_hline, drag_vline, line, line_plot, vec4, vlines,
)
from mbo_utilities.gui.imgui.panels import draw_keybinds_popup
from mbo_utilities.gui.imgui.scatter import ScatterPlot
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.install import VNOISER_HINT
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.vnoiser import MODES, CurationSession, pf_dir_for_mesc

__all__ = [
    "KEYBINDS",
    "EventCurationWidget",
    "Recording",
    "attach_curation_widget",
    "detach_curation_widget",
    "help_markdown",
]

# the dashboard is two rows: the trace row (A with its cards) over the
# candidate row (B to D); the strip asks for their sum
TIMELINE_HEIGHT = 260
CANDIDATE_HEIGHT = 210
PANEL_HEIGHT = TIMELINE_HEIGHT + CANDIDATE_HEIGHT
# the slider card beside the trace: one column per rule (A1 threshold, A2
# amplitude, A3 PC1, A4 cosine), each a vertical slider with its range
# above and below and a short name and count under it
SLIDER_COL_EM = 4.2
SLIDER_GAP_EM = 0.4
SLIDER_CARD_PAD_EM = 1.4
FILTERS = ("all", "yes", "no", "unlabeled")

TRACE_COLOR = (0.85, 0.85, 0.85, 1.0)
LOWPASS_COLOR = (0.35, 0.60, 0.95, 1.0)
THRESHOLD_COLOR = (0.84, 0.15, 0.24, 1.0)
AUTO_PASS_COLOR = (0.16, 0.62, 0.56, 1.0)
# A3: the PC1 line on the PCA and its slider; A4: the cosine slider
PC1_COLOR = (0.62, 0.45, 0.90, 1.0)
COSINE_COLOR = (0.95, 0.68, 0.25, 1.0)
# drag-tool id of the A3 line on the PCA plot
PC1_LINE_ID = 3
TEMPLATE_COLOR = (1.0, 1.0, 1.0, 1.0)
SNIPPET_COLOR = (0.30, 0.47, 0.66, 1.0)
FOCUS_COLOR = THRESHOLD_COLOR
# the t = 0 marker on the candidate plots: a reference, not a call
ZERO_COLOR = (0.62, 0.62, 0.65, 0.9)
# Box accept / Box reject: the box and the rings on what it holds take the
# colour of the label it will apply
BOX_COLORS = {"yes": (0.14, 0.72, 0.65, 1.0), "no": (0.95, 0.22, 0.30, 1.0)}
# in box mode the plots give right-drag to the box: no zoom-to-box, no menu
BOX_PLOT_FLAGS = implot.Flags_.no_box_select | implot.Flags_.no_menus
# drag-tool id of the box on a plot (the threshold lines are 1 and 2)
BOX_TOOL_ID = 7

KEYBINDS = (
    ("y", "label the focused candidate yes"),
    ("n", "label the focused candidate no"),
    ("backspace", "clear its label"),
    ("\u2190 / \u2192", "previous / next candidate in view"),
    ("\u2191 / \u2193", "previous / next recording"),
    ("click", "focus a candidate on the trace or the PCA plot"),
    ("Box accept / reject", "box mode: right-drag a box on the trace or the PCA, then drag its edges"),
    ("enter", "apply the box"),
    ("esc", "leave box mode"),
    ("drag line", "move the threshold (red) / auto-pass (teal) line on the trace, or the PC1 (purple) line on the PCA"),
    ("scroll", "zoom (shift: x only, alt: y only); drag pans; double-click fits"),
    ("k", "this list"),
)

_MODE_TITLES = {
    "fast": "fast candidates",
    "slow": "slow (<{cutoff:g} Hz) candidates",
    "manual": "manual candidates",
}


def help_markdown() -> str:
    return (
        "## Event Curation\n\n"
        "vnoiser's curation notebook inside the viewer, over every recording "
        "at once. Pick a data path (a `Data` folder, an animal or experiment "
        "folder, a `PF` folder, a raw `.mat` recording, or a line-scan `.mesc` "
        "with a PF folder beside it): every scan / domain the pipeline "
        "processed is listed and loaded.\n\n"
        "### Modes\n\n"
        "- **fast**: thresholds the denoised trace; the template is seeded "
        "from the top 25% highest-amplitude candidates.\n"
        "- **slow**: thresholds a zero-phase low-pass view of the denoised "
        "trace (blue); the template is seeded the same way.\n"
        "- **manual**: no seed template; only Yes events shape it.\n\n"
        "### Panels\n\n"
        "- **Curation** (top): the notebook's dashboard for one recording at "
        "a time; the arrows (or up / down) flip through them. Top row: the "
        "trace with its candidates (drag the red line to change the candidate "
        "threshold, the teal line or the A2 slider to set the auto-pass "
        "amplitude) beside the slider card. A3 is the purple line on the PCA (drag it, or its slider): "
        "every candidate on its passing side auto-passes; the arrow button "
        "under the slider picks the side. A4 is the seed-template cosine at "
        "or above which a candidate auto-passes; below it auto-rejects, so "
        "A4 at the bottom rejects nothing. Bottom row: the "
        "current template, the focused candidate against it, and the candidate "
        "PCA over 400 ms windows. Click a point to focus it. **Box accept** / **Box "
        "reject** start a box mode: right-drag a box on the trace or the PCA, "
        "drag its edges or corners to adjust (the candidates inside are "
        "ringed and counted), then **Apply** or enter labels them all; esc "
        "leaves the mode. Manual labels (boxed ones too) always win over the "
        "A2 / A3 / A4 rules; **Clear all** drops them so the rules apply "
        "again. Drag the strip's grab bar to give the rows more "
        "height; the `keybinds` button lists the keys.\n"
        "- **Curation** tab (right): **Decision** (Yes / No / Clear, the box "
        "modes, the focused event, navigation, the view filter, files) and "
        "**Recordings** (mode, data path, the table).\n\n"
        "Each mode saves to its own `PF/.curation/<mode>_template_curation.json`; "
        "reopening the same scan / domain restores it."
    )


def _mode_title(mode: str, cutoff: float) -> str:
    return _MODE_TITLES[mode].format(cutoff=cutoff)


class UnitTraces:
    """The per-ROI traces of one AOD ROI unit, read the first time a ROI is
    asked for (on the curation worker, never at open): the pipeline's own
    PF traces or an ``F.npy`` when the unit has them, else the
    ``.curation/cache`` copy of an earlier reduction keyed by the file's
    size and mtime, else the reduction itself, cached there for next time."""

    def __init__(self, mesc_path: Path, unit: dict, channel: int, traces_dir=None):
        self.mesc_path = Path(mesc_path)
        self.unit = unit
        self.channel = int(channel)
        self.traces_dir = traces_dir
        self.traces: np.ndarray | None = None

    def load(self) -> np.ndarray:
        """``(rois, samples)``, read or computed once."""
        from mbo_utilities.arrays.mesc import MescArray
        from mbo_utilities.gui.linescan_viewer import saved_roi_traces
        from mbo_utilities.roi_workflow import linescan_roi_means

        if self.traces is not None:
            return self.traces
        arr = MescArray(self.mesc_path, unit=self.unit["key"])
        saved = saved_roi_traces(arr, self.traces_dir)
        if saved is not None:
            self.traces = np.asarray(saved[0])
            return self.traces
        stat = self.mesc_path.stat()
        cache_dir = self.mesc_path.parent / ".curation" / "cache"
        cache = cache_dir / f"{self.mesc_path.stem}_{self.unit['munit']}_ch{self.channel}_traces.npz"
        if cache.exists():
            with np.load(cache) as saved:
                if int(saved["size"]) == stat.st_size and int(saved["mtime_ns"]) == stat.st_mtime_ns:
                    self.traces = np.asarray(saved["traces"])
                    return self.traces
        self.traces = np.asarray(linescan_roi_means(arr, channel=self.channel))
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, traces=self.traces, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        return self.traces

    def roi(self, i: int) -> np.ndarray:
        return self.load()[int(i)]


def raw_linescan_traces(mesc_path, channel: int = 0, traces_dir=None) -> list[dict]:
    """Every AOD ROI unit (line scan, chessboard or ribbon patches) in a
    ``.mesc``: ``[{"key", "munit", "fs", "n_rois", "traces"}, ...]`` where
    ``traces`` is a :class:`UnitTraces`, read when first asked for. Nothing
    is read here beyond the units' metadata, so opening a file with many
    long scans costs nothing until a ROI is clicked."""
    from mbo_utilities.arrays.mesc import ROI_LAYOUTS, list_mesc_units

    mesc_path = Path(mesc_path)
    out = []
    for unit in list_mesc_units(mesc_path):
        if unit.get("kind") not in ROI_LAYOUTS:
            continue
        out.append({
            "key": unit["key"],
            "munit": unit["munit"],
            "fs": float(unit["fs"]),
            "n_rois": int(unit["nrois"]),
            "traces": UnitTraces(mesc_path, unit, channel, traces_dir),
        })
    return out


def curation_source(arr) -> str:
    """What a viewer's array brings to the curation: "pf" for a PF folder
    (:class:`~mbo_utilities.arrays.pf.PfArray`) or an AOD ROI unit whose
    experiment has one, "raw" for an AOD ROI unit without, "" otherwise."""
    from mbo_utilities.arrays.mesc import ROI_LAYOUTS
    from mbo_utilities.arrays.pf import PfArray
    from mbo_utilities.lazy_array import base_array

    arr = base_array(arr)
    if isinstance(arr, PfArray):
        return "pf"
    md = getattr(arr, "metadata", None) or {}
    files = getattr(arr, "filenames", None) or []
    if md.get("mesc_layout") not in ROI_LAYOUTS or not files:
        return ""
    return "pf" if pf_dir_for_mesc(files[0]) is not None else "raw"


@dataclass
class Recording:
    """One curatable recording: what a session opens and which id loads it."""

    rid: str
    label: str
    experiment: str
    source: str
    pre_denoised: bool
    # why the last load failed (the pipeline never wrote this scan / domain)
    error: str = ""


class EventCurationWidget:
    """The curation panels and tab on a ``PreviewDataWidget``-like parent.

    Parameters
    ----------
    parent
        Anything with ``image_widget`` (a figure) and a ``logger``.
    strip : TopStrip, optional
        The figure's shared top strip; a private one is built without.
    data_path : str, optional
        Path to scan at once; None reopens the last one, "" none.
    """

    def __init__(self, parent: Any, strip: TopStrip | None = None, data_path: str | None = None):
        self.parent = parent
        self.logger = getattr(parent, "logger", None) or logging.getLogger(__name__)
        self.figure = parent.image_widget.figure
        self._own_strip = strip is None
        self.strip = TopStrip(self.figure) if self._own_strip else strip
        self.strip.add_hook(self._frame)
        self.strip.register(
            TopPanel("curation", "Curation", self.draw_panel, PANEL_HEIGHT, "curation", 12)
        )

        self.mode = "fast"
        self.slow_cutoff_hz = 40.0
        self.catalog: list[Recording] = []
        self.sessions: dict[tuple[str, str], CurationSession] = {}
        self.current = ""
        # traces handed over in memory (line-scan ROIs), by recording id
        self._trace_sources: dict[str, dict] = {}
        # called with the focused candidate's time (s) whenever it changes
        self.on_focus = None
        self._last_focus = None
        # called with the recording id whenever another recording is focused
        self.on_recording = None
        self._last_recording = None
        # a host may narrow the recordings it shows to those of one scan
        # (the line-scan viewer: the unit on screen); None shows them all
        self.scope = None
        self.data_path = ""
        self.prompt = PathPrompt(
            "Curation data",
            path="",
            action="scan",
            hint="vnoiser Data folder, an animal, experiment or PF folder, a .mat, or a line-scan .mesc",
        )
        self._folder_dialog = None
        self.status = "Set a data path: every recording under it is listed and loaded."

        self._busy: set[tuple[str, str]] = set()
        self._jobs: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        # the job the worker is on: (key, label, started at), for the loading line
        self._active: tuple | None = None
        self.timeline_points = ScatterPlot("##curation_timeline_pts", marker_size=7.0)
        self.pca = ScatterPlot("##curation_pca", marker_size=7.0)
        self.autofit = True
        self.show_keybinds = False
        # Box accept / reject mode: the label a drawn box applies ("yes" /
        # "no"), the rectangle (plot name, corners in plot units, whether
        # the right button is still drawing it) and the candidates it holds
        self.box_mode: str | None = None
        self._box_rect: dict | None = None
        self._box: tuple[str, np.ndarray] | None = None
        self._fit_timeline = False
        self._timeline_key = None
        self._timeline_trace_cache: dict[str, tuple] = {}
        self._panel_keys: dict[str, tuple] = {}
        self._threshold_drag: float | None = None
        self._auto_pass_drag: float | None = None
        self._pc1_drag: float | None = None
        self._slider_pending: dict[str, float] = {}

        self._hovered = False
        self.focus_tab = False
        self._closed = False

        if data_path is None:
            last = get_last_dir("vnoiser")
            data_path = str(last) if last else ""
        if data_path:
            self.scan(data_path)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Take the panels back off the strip."""
        if self._closed:
            return
        self._closed = True
        self._folder_dialog = None
        self.strip.remove_hook(self._frame)
        self.strip.unregister("curation")
        if self._own_strip:
            self.strip.close()

    # ------------------------------------------------------------------
    # catalog and sessions
    # ------------------------------------------------------------------

    @property
    def session(self) -> CurationSession | None:
        """The focused recording's session in the current mode, if loaded."""
        return self.sessions.get((self.mode, self.current))

    @property
    def loading(self) -> bool:
        return bool(self._busy)

    def _ready(self) -> CurationSession | None:
        """The focused session when it can be read: loaded and not mid-load."""
        if (self.mode, self.current) in self._busy:
            return None
        session = self.session
        return session if session is not None and session.loaded else None

    def in_scope(self, rec: Recording) -> bool:
        """Whether the host shows this recording (see ``scope``)."""
        return self.scope is None or bool(self.scope(rec))

    @property
    def shown(self) -> list[Recording]:
        """The catalog narrowed to the host's scope."""
        return [r for r in self.catalog if self.in_scope(r)]

    def recording(self, rid: str) -> Recording | None:
        return next((r for r in self.catalog if r.rid == rid), None)

    @property
    def experiments(self) -> list[str]:
        seen: list[str] = []
        for rec in self.catalog:
            if rec.experiment not in seen:
                seen.append(rec.experiment)
        return seen

    def loaded(self, mode: str | None = None) -> list[tuple[Recording, CurationSession]]:
        """Every loaded recording of ``mode`` (the current one), catalog order."""
        mode = mode or self.mode
        out = []
        for rec in self.shown:
            session = self.sessions.get((mode, rec.rid))
            if session is not None and session.loaded and (mode, rec.rid) not in self._busy:
                out.append((rec, session))
        return out

    def scan(self, path) -> None:
        """Catalog every recording under a path and start loading them."""
        self.sessions.clear()
        self._trace_sources.clear()
        self.catalog = []
        self.current = ""
        path = Path(path).expanduser()
        note = ""
        if path.suffix.lower() == ".mesc":
            # the raw line scan; its processed traces sit in the experiment's
            # PF folder, which is what the curation notebook reads
            pf = pf_dir_for_mesc(path)
            if pf is None:
                self.data_path = ""
                self.status = (
                    f"{path.name} is a raw line scan with no PF folder beside it. Open it "
                    "with `mbo <file>.mesc` and curate its lines there, or point at a "
                    "vnoiser Data / experiment / PF folder."
                )
                return
            note = f" (PF folder of {path.name})"
            path = pf
        self.data_path = str(path)
        try:
            self.catalog = _build_catalog(path, self.logger)
        except Exception as error:
            self.logger.warning("vnoiser cannot open %s", path, exc_info=True)
            self.status = f"cannot open {path}: {error}"
            self.data_path = ""
            return
        set_last_dir("vnoiser", self.data_path)
        self.prompt.path = self.data_path
        if not self.catalog:
            self.status = (
                f"no vnoiser data at {self.data_path}: expected a Data folder "
                "(stan*/…_expt*/PF/denoised_trace_scans.pkl), an animal, experiment or "
                "PF folder, or a .mat recording."
            )
            return
        n_pre = sum(r.pre_denoised for r in self.catalog)
        self.status = (
            f"{len(self.catalog)} recordings in {len(self.experiments)} experiment(s)"
            f"{note}; {n_pre} processed"
        )
        # the first processed experiment loads on its own; raw .mat
        # recordings take minutes each in the denoiser, so they wait for a click
        first = next((r for r in self.shown if r.pre_denoised), None)
        if first is not None:
            self.current = first.rid
            self.load_all(first.experiment)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES or mode == self.mode:
            return
        self.mode = mode
        self._fit_timeline = True
        rec = self.recording(self.current)
        if rec is not None:
            self.load(self.current)
            if rec.pre_denoised:
                self.load_all(rec.experiment)

    def set_slow_cutoff(self, cutoff_hz: float) -> None:
        cutoff_hz = float(cutoff_hz)
        if cutoff_hz <= 0 or cutoff_hz == self.slow_cutoff_hz:
            return
        self.slow_cutoff_hz = cutoff_hz
        for key in [k for k in self.sessions if k[0] == "slow"]:
            self.sessions.pop(key)
        if self.mode == "slow":
            self.set_mode("fast")
            self.set_mode("slow")

    def load(self, rid: str) -> None:
        """Focus recording ``rid``, loading it for the current mode if needed."""
        rec = self.recording(rid)
        if rec is None:
            return
        if rec.error:
            self.status = f"{rec.label}: {rec.error}"
            return
        self.current = rid
        self._fit_timeline = True
        self.pca.refit()
        self._panel_keys.clear()
        key = (self.mode, rid)
        if key in self.sessions or key in self._busy:
            return
        self._enqueue(rec, self.mode)

    def load_all(self, experiment: str | None = None) -> None:
        """Load every processed recording of ``experiment`` (all when None)."""
        for rec in self.shown:
            if experiment is not None and rec.experiment != experiment:
                continue
            if not rec.pre_denoised and rec.rid not in self._trace_sources:
                continue
            key = (self.mode, rec.rid)
            if key not in self.sessions and key not in self._busy:
                self._enqueue(rec, self.mode)
        if not self.current:
            first = next((r for r in self.catalog if (self.mode, r.rid) in self._busy), None)
            if first is not None:
                self.current = first.rid

    def add_trace(self, trace, fs_hz, *, recording_id, label, source_path, curation_dir=None) -> Recording:
        """List a trace held in memory (a line-scan ROI) as a raw recording;
        loading it runs vnoiser's denoiser on the worker unless a cache
        exists. See :meth:`CurationSession.load_trace`."""
        source_path = str(Path(source_path))
        rid = str(recording_id)
        rec = self.recording(rid)
        if rec is None:
            rec = Recording(rid, str(label), Path(source_path).stem, source_path, False)
            self.catalog.append(rec)
        self._trace_sources[rid] = {
            "trace": trace,
            "fs_hz": float(fs_hz),
            "recording_id": rid,
            "label": str(label),
            "source_path": source_path,
            "curation_dir": curation_dir,
        }
        for key in [k for k in self.sessions if k[1] == rid]:
            self.sessions.pop(key)
        return rec

    def load_trace(self, trace, fs_hz, *, recording_id, label, source_path, curation_dir=None) -> None:
        """:meth:`add_trace`, then load it."""
        rec = self.add_trace(
            trace, fs_hz, recording_id=recording_id, label=label,
            source_path=source_path, curation_dir=curation_dir,
        )
        self.load(rec.rid)

    def open_array(self, arr) -> str:
        """Point the curation at the array a viewer shows: every scan of a
        PF folder (the folder itself, or the one beside an AOD ROI unit)
        with the scan on screen selected first, or the raw ROIs of a file
        without one scoped to that unit. Called again for another unit of
        the same folder it only moves the selection.
        Returns :func:`curation_source` of the array."""
        from mbo_utilities.lazy_array import base_array

        arr = base_array(arr)
        kind = curation_source(arr)
        if kind == "pf":
            pf_dir = getattr(arr, "pf_dir", None)
            scan_id = getattr(arr, "scan", None)
            if pf_dir is None:
                pf_dir = pf_dir_for_mesc(arr.filenames[0])
                scan_id = str(arr.unit_key).rsplit("_", 1)[-1]
            self.scope = None
            if self.data_path != str(pf_dir) or not self.catalog:
                self.scan(pf_dir)
            tag = f"scan={scan_id}"
            first = next((r for r in self.shown if tag in r.rid.split("/") and r.pre_denoised), None)
            if first is not None and first.rid != self.current:
                self.load(first.rid)
            elif first is None and self.catalog:
                scans = sorted({part[5:] for r in self.catalog for part in r.rid.split("/") if part.startswith("scan=")})
                self.status = f"scan {scan_id} is not in {pf_dir} (scans {', '.join(scans)}); showing them all"
        elif kind == "raw":
            self.scan_raw_mesc(arr.filenames[0])
            munit = str(arr.unit_key).rsplit("/", 1)[-1]
            self.scope = lambda rec: rec.rid.split("/")[1:2] == [munit]
        return kind

    def scan_raw_mesc(self, mesc_path, channel: int = 0) -> int:
        """Every ROI of every AOD ROI unit of a ``.mesc`` (line scans,
        chessboard or ribbon patches) as a raw recording the denoiser runs
        on when clicked. Returns how many."""
        mesc_path = Path(mesc_path)
        self.sessions.clear()
        self._trace_sources.clear()
        self.catalog = []
        self.current = ""
        n = 0
        # per-unit loaders, so a viewer that already has a unit's traces
        # (the Traces tab's background job) can hand them over
        self.unit_traces = {}
        for unit in raw_linescan_traces(mesc_path, channel):
            self.unit_traces[unit["munit"]] = unit["traces"]
            for i in range(unit["n_rois"]):
                # read on the worker when the recording is clicked
                self.add_trace(
                    partial(unit["traces"].roi, i), unit["fs"],
                    recording_id=f"{mesc_path.stem}/{unit['munit']}/roi={i}",
                    label=f"{unit['munit']} ROI {i}",
                    source_path=mesc_path,
                )
                n += 1
        self.data_path = str(mesc_path)
        self.prompt.path = self.data_path
        self.status = (
            f"{n} raw ROI traces in {mesc_path.name}; no PF folder beside it, so click a "
            "recording to run vnoiser's denoiser on it (minutes the first time, cached after)"
            if n else f"no AOD ROI units in {mesc_path.name}"
        )
        return n

    def _enqueue(self, rec: Recording, mode: str) -> None:
        key = (mode, rec.rid)
        self._busy.add(key)
        source = self._trace_sources.get(rec.rid)
        cutoff = self.slow_cutoff_hz

        def work():
            session = CurationSession(rec.source, mode=mode, slow_cutoff_hz=cutoff)
            if source is not None:
                # a line-scan ROI's trace is a loader until now: the read or
                # reduction happens here, on the worker
                if callable(source["trace"]):
                    source["trace"] = source["trace"]()
                message = session.load_trace(**source)
            else:
                message = session.load(rec.rid)
            return session, message

        self._jobs.put((key, rec.label, work))
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run_jobs, name="vnoiser-load", daemon=True)
            self._worker.start()

    def _run_jobs(self) -> None:
        manager = get_process_manager()
        while True:
            try:
                key, label, work = self._jobs.get(timeout=0.5)
            except queue.Empty:
                return
            job = manager.start_job("vnoiser", f"curation: {label}")
            self._active = (key, label, time.monotonic())
            try:
                session, message = work()
            except Exception as error:
                job.fail(f"{type(error).__name__}: {error}")
                self._results.put((key, None, f"{label}: load failed: {error}"))
            else:
                job.done(message)
                self._results.put((key, session, message))
            finally:
                self._active = None
                self._jobs.task_done()

    def loading_line(self) -> str:
        """What the worker is doing: the job, its elapsed time and the queue."""
        active = self._active
        if active is None:
            return f"{len(self._busy)} queued" if self._busy else ""
        key, label, started = active
        elapsed = int(time.monotonic() - started)
        verb = "denoising" if key[1] in self._trace_sources else "loading"
        line = f"{verb} {label} · {elapsed // 60}:{elapsed % 60:02d}"
        queued = len(self._busy) - 1
        return line + (f" · {queued} queued" if queued > 0 else "")

    def wait(self, timeout: float | None = None) -> None:
        """Block until every queued load finishes and apply them (tests)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._busy:
            self._drain()
            worker = self._worker
            if not self._busy:
                break
            if worker is None or not worker.is_alive():
                if self._jobs.empty():
                    break
            if deadline is not None and time.monotonic() > deadline:
                break
            time.sleep(0.02)
        self._drain()

    def _drain(self) -> None:
        while True:
            try:
                key, session, message = self._results.get_nowait()
            except queue.Empty:
                return
            self._busy.discard(key)
            if session is None:
                self.logger.warning(message)
                self.status = message
                rec = self.recording(key[1])
                if rec is not None:
                    rec.error = message.rsplit(": ", 1)[-1]
                # a recording the pipeline never wrote cannot be shown:
                # move on to the next one rather than sit on an empty panel
                if key == (self.mode, self.current) and self.loadable():
                    self.step_recording(1)
                continue
            self.sessions[key] = session
            if key == (self.mode, self.current):
                self.status = message
                self._fit_timeline = True
                self.pca.refit()
                self._panel_keys.clear()
            elif not self._busy:
                self.status = f"{len(self.loaded())} recordings loaded"

    # ------------------------------------------------------------------
    # per frame
    # ------------------------------------------------------------------

    def _frame(self) -> None:
        self._drain()
        self._poll_folder_dialog()
        self._draw_prompt()
        self._handle_keys()
        self.show_keybinds = draw_keybinds_popup(KEYBINDS, self.show_keybinds, "Curation keybinds")
        self._report_focus()
        self._hovered = False

    def _report_focus(self) -> None:
        session = self._ready()
        if session is None:
            return
        if session.recording_id != self._last_recording:
            self._last_recording = session.recording_id
            if self.on_recording is not None:
                try:
                    self.on_recording(session.recording_id)
                except Exception:
                    self.logger.debug("curation recording callback failed", exc_info=True)
        if not session.n:
            return
        key = (session.mode, session.recording_id, session.current)
        if key == self._last_focus:
            return
        self._last_focus = key
        if self.on_focus is not None:
            try:
                self.on_focus(float(session.times_s[session.current]))
            except Exception:
                self.logger.debug("curation focus callback failed", exc_info=True)

    def _draw_prompt(self) -> None:
        submitted, browse = draw_path_prompt(self.prompt)
        if submitted:
            if Path(submitted).expanduser().exists():
                self.scan(submitted)
                if self.catalog:
                    self.prompt.open = False
                else:
                    self.prompt.status = self.status
            else:
                self.prompt.status = "no such path"
        if browse and self._folder_dialog is None:
            start = self.prompt.path if Path(self.prompt.path or "").exists() else str(Path.home())
            self._folder_dialog = pfd.select_folder("Curation data", start)

    def _poll_folder_dialog(self) -> None:
        dialog = self._folder_dialog
        if dialog is None or not dialog.ready():
            return
        self._folder_dialog = None
        picked = dialog.result()
        if picked:
            self.prompt.path = str(picked)

    def _handle_keys(self) -> None:
        io = imgui.get_io()
        if not self._hovered or io.want_text_input:
            return
        if imgui.is_key_pressed(imgui.Key.k, False):
            self.show_keybinds = not self.show_keybinds
        if imgui.is_key_pressed(imgui.Key.escape, False):
            self.exit_box_mode()
        if self.box_mode is not None and imgui.is_key_pressed(imgui.Key.enter, False):
            self.apply_box()
        if imgui.is_key_pressed(imgui.Key.up_arrow, False):
            self.step_recording(-1)
        if imgui.is_key_pressed(imgui.Key.down_arrow, False):
            self.step_recording(1)
        session = self._ready()
        if session is None or not session.n:
            return
        if imgui.is_key_pressed(imgui.Key.y, False):
            session.set_label("yes")
        if imgui.is_key_pressed(imgui.Key.n, False):
            session.set_label("no")
        if imgui.is_key_pressed(imgui.Key.backspace, False):
            session.set_label("unlabeled")
        if imgui.is_key_pressed(imgui.Key.left_arrow, True):
            session.step(-1)
        if imgui.is_key_pressed(imgui.Key.right_arrow, True):
            session.step(1)

    def _mark_hovered(self) -> None:
        if imgui.is_window_hovered(imgui.HoveredFlags_.root_and_child_windows):
            self._hovered = True

    # ------------------------------------------------------------------
    # flipping through recordings
    # ------------------------------------------------------------------

    def loadable(self) -> list[Recording]:
        """The recordings a flip can land on: processed ones and traces handed over."""
        return [
            r for r in self.shown
            if not r.error and (r.pre_denoised or r.rid in self._trace_sources)
        ]

    def step_recording(self, delta: int) -> None:
        """Focus the previous / next loadable recording, loading it if needed."""
        rows = self.loadable()
        if not rows:
            return
        rids = [r.rid for r in rows]
        if self.current not in rids:
            self.load(rids[0])
            return
        pos = rids.index(self.current)
        self.load(rids[(pos + int(delta)) % len(rids)])

    # ------------------------------------------------------------------
    # top panel: the dashboard for the focused recording
    # ------------------------------------------------------------------

    def draw_panel(self) -> None:
        """The Curation panel: the notebook's dashboard for one recording.
        The trace row (A with the threshold / auto-pass, Decision and
        Navigation cards) sits over the candidate row (B to D); the rows
        share the height the strip gives in the ratio they asked for."""
        self._mark_hovered()
        self._draw_flip_row()
        session = self._ready()
        if session is None:
            imgui.text_disabled(self.status)
            return
        avail = imgui.get_content_region_avail()
        top = max(avail.y * TIMELINE_HEIGHT / PANEL_HEIGHT, em(6))
        with imgui_ctx.begin_child("##curation_row_a", imgui.ImVec2(0, top)):
            self._draw_timeline_row(session)
        with imgui_ctx.begin_child("##curation_row_b", imgui.ImVec2(0, 0)):
            self._draw_candidate_row(session)

    def _draw_timeline_row(self, session: CurationSession) -> None:
        """A: the trace with its candidates, as wide as the row allows, and
        the slider card (A1 to A4) beside it. Decision and Navigation live
        in the controls column (``draw_tab``), not in this row."""
        kind = _mode_title(session.mode, self.slow_cutoff_hz)
        imgui.text_disabled(f"A. {kind}: {session.n} ({len(session.visible)} in view)")
        imgui.same_line(0, 12)
        if imgui.small_button("keybinds"):
            self.show_keybinds = not self.show_keybinds
        set_tooltip("k", show_mark=False)
        avail = imgui.get_content_region_avail()
        n_cols = 4 if session.seeded else 1
        card_w = em(SLIDER_COL_EM) * n_cols + em(SLIDER_GAP_EM) * (n_cols - 1) + em(SLIDER_CARD_PAD_EM)
        plot_w = max(avail.x - card_w - em(0.5), em(10))
        with imgui_ctx.begin_child("##curation_trace", imgui.ImVec2(plot_w, 0)):
            self._draw_timeline(session)
        imgui.same_line(0, em(0.5))
        self._draw_slider_card(session, card_w, avail.y)

    def _draw_flip_row(self) -> None:
        """Previous / next recording, and which one this is."""
        rows = self.loadable()
        rec = self.recording(self.current)
        pos = next((i for i, r in enumerate(rows) if r.rid == self.current), -1)
        if not rows:
            imgui.begin_disabled()
        if imgui.arrow_button("##prev_rec", imgui.Dir.up):
            self.step_recording(-1)
        set_tooltip("previous recording (up)", show_mark=False)
        imgui.same_line(0, em(0.3))
        if imgui.arrow_button("##next_rec", imgui.Dir.down):
            self.step_recording(1)
        set_tooltip("next recording (down)", show_mark=False)
        if not rows:
            imgui.end_disabled()
        imgui.same_line(0, em(0.6))
        label = rec.label if rec is not None else "no recording"
        where = f"  ({pos + 1}/{len(rows)})" if pos >= 0 else ""
        imgui.text(f"{label}{where}")
        if (self.mode, self.current) in self._busy:
            imgui.same_line(0, em(0.6))
            imgui.text_disabled(self.loading_line() or "loading...")

    def _cached_trace(self, name: str, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Min/max-decimated ``(t, y)`` for the timeline trace, cached per
        recording so zooming does not replot the full-resolution array every
        frame (mirrors the notebook's ``downsample_xy`` before plotly)."""
        cached = self._timeline_trace_cache.get(name)
        if cached is None:
            idx, values = decimate_minmax(y, 4000)
            cached = (t[idx.astype(int)], values)
            self._timeline_trace_cache[name] = cached
        return cached

    def _draw_timeline(self, session: CurationSession) -> None:
        key = (session.mode, session.recording_id)
        fit = self._fit_timeline or (self.autofit and key != self._timeline_key)
        if fit or key != self._timeline_key:
            # fit also fires on a slow-cutoff edit, which changes analysis_trace under the same key
            self._timeline_trace_cache = {}
        self._timeline_key = key
        self._fit_timeline = False
        height = max(imgui.get_content_region_avail().y - 2, 60.0)
        flags = BOX_PLOT_FLAGS if self.box_mode is not None else 0
        with line_plot("##curation_timeline", "time (s)", "z", height=height, fit=fit, flags=flags) as ok:
            if not ok:
                return
            t = session.t
            t_plot, denoised_plot = self._cached_trace("denoised", t, session.denoised)
            line("denoised", denoised_plot, x=t_plot, color=TRACE_COLOR, weight=1.0)
            marker_source = session.denoised
            if session.mode == "slow":
                t_lp, lp_plot = self._cached_trace("analysis", t, session.analysis_trace)
                line(
                    f"<{self.slow_cutoff_hz:g} Hz low-pass",
                    lp_plot, x=t_lp, color=LOWPASS_COLOR, weight=1.6,
                )
                marker_source = session.analysis_trace

            shown = self._threshold_drag if self._threshold_drag is not None else session.threshold
            value, held = drag_hline(1, shown, THRESHOLD_COLOR)
            if held:
                self._threshold_drag = value
            elif self._threshold_drag is not None:
                session.set_threshold(self._threshold_drag)
                self._threshold_drag = None

            if session.seeded and session.auto_pass is not None:
                shown = self._auto_pass_drag if self._auto_pass_drag is not None else session.auto_pass
                value, held = drag_hline(2, shown, AUTO_PASS_COLOR)
                if held:
                    self._auto_pass_drag = value
                elif self._auto_pass_drag is not None:
                    session.set_auto_pass(self._auto_pass_drag)
                    self._auto_pass_drag = None

            visible = session.visible
            if not session.n:
                return
            xs = session.times_s[visible]
            ys = np.interp(xs, t, marker_source)
            focused = np.flatnonzero(visible == session.current)
            picked = self.timeline_points.items(
                xs, ys, session.colors(visible),
                focused=int(focused[0]) if focused.size else None,
                label="candidates",
                tooltip=lambda i: f"candidate {visible[i] + 1}: {xs[i]:.3f} s, {session.label(visible[i])}",
            )
            if picked is not None:
                session.select(int(visible[picked]))
            self._box_select("timeline", xs, ys, visible, self.timeline_points)
            vlines("##focus", [session.times_s[session.current]], FOCUS_COLOR, 1.0, legend=False)

    # ------------------------------------------------------------------
    # box mode: Box accept / Box reject draw a rectangle on the trace or
    # the PCA that labels everything inside on Apply
    # ------------------------------------------------------------------

    def set_box_mode(self, mode: str | None) -> None:
        """Enter box mode for ``mode`` ("yes" / "no"); the same mode again
        leaves it. Any box drawn so far is dropped."""
        self.box_mode = None if mode == self.box_mode else mode
        self.clear_box()

    def exit_box_mode(self) -> None:
        self.box_mode = None
        self.clear_box()

    def clear_box(self) -> None:
        self._box_rect = None
        self._box = None

    def _box_select(self, name: str, xs, ys, visible, points: ScatterPlot) -> None:
        """Inside an open plot, in box mode: a right-drag starts the box on
        this plot; once the button is up the box stays as an implot drag
        rectangle whose edges and corners resize it. The candidates inside
        are ringed and remembered for :meth:`apply_box`."""
        if self.box_mode is None:
            return
        color = BOX_COLORS[self.box_mode]
        rect = self._box_rect if self._box_rect is not None and self._box_rect["plot"] == name else None
        mouse = implot.get_plot_mouse_pos()
        if rect is None:
            if not (implot.is_plot_hovered() and imgui.is_mouse_clicked(1)):
                return
            # a new box here drops one drawn on the other plot
            rect = self._box_rect = {
                "plot": name, "x0": float(mouse.x), "y0": float(mouse.y),
                "x1": float(mouse.x), "y1": float(mouse.y), "drawing": True,
            }
            self._box = None
        if rect["drawing"]:
            if imgui.is_mouse_down(1):
                rect["x1"], rect["y1"] = float(mouse.x), float(mouse.y)
            else:
                rect["drawing"] = False
            implot.drag_rect(
                BOX_TOOL_ID, rect["x0"], rect["y0"], rect["x1"], rect["y1"], vec4(color),
                implot.DragToolFlags_.no_inputs,
            )
        else:
            changed, x0, y0, x1, y1, _clicked, _hovered, _held = implot.drag_rect(
                BOX_TOOL_ID, rect["x0"], rect["y0"], rect["x1"], rect["y1"], vec4(color),
                implot.DragToolFlags_.no_fit,
            )
            if changed:
                rect.update(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        lo_x, hi_x = sorted((rect["x0"], rect["x1"]))
        lo_y, hi_y = sorted((rect["y0"], rect["y1"]))
        inside = (xs >= lo_x) & (xs <= hi_x) & (ys >= lo_y) & (ys <= hi_y)
        self._box = (name, np.asarray(visible, dtype=int)[inside])
        points.rings(xs[inside], ys[inside], color, 2.0, "##boxed")

    def boxed(self) -> np.ndarray:
        """Candidate indices inside the box."""
        return np.array([], dtype=int) if self._box is None else self._box[1]

    def apply_box(self, label: str | None = None) -> int:
        """Label every boxed candidate (the mode's label unless given) and
        drop the box; the mode stays on for the next one. Returns how many."""
        label = label or self.box_mode
        session = self._ready()
        idx = self.boxed()
        if session is None or label not in ("yes", "no") or not len(idx):
            return 0
        n = session.set_labels([int(i) for i in idx], label)
        self.clear_box()
        return n

    def _draw_slider_card(self, session, width, height) -> None:
        """A1 to A4 as one line of slider columns: the candidate threshold
        (red, also the red line on the trace); in a seeded mode also the
        auto-pass amplitude (teal, the teal line), the PC1 line of the PCA
        (purple, drawn on panel D; the arrow picks the passing side) and the
        seed-template cosine at or above which a candidate passes (amber)."""
        title = "A1 - A4" if session.seeded else "A1"
        with card("##curation_sliders", title, height, width):
            set_tooltip(
                "Sliders apply on release. A1 threshold: candidates are local "
                "maxima of the trace above it. A2 amplitude: a candidate at or "
                "above it passes whatever its shape. A3 PC1: every candidate on "
                "the passing side of the purple line on the PCA passes (the "
                "arrow flips the side; drag the line on the plot too). A4 "
                "cosine: a candidate whose similarity to the seed template is "
                "at or above it passes, below it is rejected; at the bottom "
                "nothing is rejected. Manual labels always win. All saved per "
                "recording.",
            )
            lo, hi, _step = session.threshold_range
            self._slider_column(
                "a1", "thr", session.threshold, lo, hi, session.set_threshold, THRESHOLD_COLOR,
                f"{session.n} found", "candidate threshold: local maxima of the trace above it",
            )
            if not session.seeded:
                return
            alo, ahi, _step = session.auto_pass_range
            imgui.same_line(0, em(SLIDER_GAP_EM))
            self._slider_column(
                "a2", "amp", session.auto_pass if session.auto_pass is not None else ahi,
                alo, ahi, session.set_auto_pass, AUTO_PASS_COLOR,
                "off" if session.auto_pass is None else f"{session.auto_pass_count()}/{session.n}",
                "auto-pass amplitude: candidates at or above it pass; at the bottom all pass",
            )
            plo, phi, _step = session.auto_pass_pc1_range
            side = session.auto_pass_pc1_side
            imgui.same_line(0, em(SLIDER_GAP_EM))
            self._slider_column(
                "a3", "PC1", session.auto_pass_pc1_shown, plo, phi, session.set_auto_pass_pc1, PC1_COLOR,
                "off" if session.auto_pass_pc1 is None else f"{session.auto_pass_pc1_count()}/{session.n}",
                f"PC1 auto-pass: candidates at or {'above' if side == 'right' else 'below'} the "
                "purple line on the PCA pass; the arrow flips the side",
                side=side, flip=lambda: session.set_auto_pass_pc1_side("left" if side == "right" else "right"),
            )
            clo, chi, _step = session.auto_template_threshold_range
            imgui.same_line(0, em(SLIDER_GAP_EM))
            self._slider_column(
                "a4", "cos", session.auto_template_threshold, clo, chi,
                session.set_auto_template_threshold, COSINE_COLOR,
                f"{session.auto_template_count()}/{session.n}",
                "cosine auto-pass: candidates whose seed-template cosine is at or above it pass, "
                "below it they are rejected",
            )

    def _slider_column(self, key, name, value, lo, hi, apply, color, count, tooltip, *,
                       side=None, flip=None) -> None:
        """One column of the slider card: the range's top, a vertical slider
        that applies on release (a drag does not rebuild the candidates every
        frame), the range's bottom, the rule's short name (``side`` adds the
        PC1 arrow before it; ``flip`` is called when it is clicked) and a
        count. Everything is centred on a SLIDER_COL_EM column."""
        width = em(SLIDER_COL_EM)
        pending = self._slider_pending.get(key)
        shown = pending if pending is not None else float(value)
        line_h = imgui.get_text_line_height_with_spacing()
        track_h = max(imgui.get_content_region_avail().y - 4 * line_h - em(0.4), em(3))
        imgui.begin_group()
        x0 = imgui.get_cursor_pos_x()

        def centred(text: str) -> None:
            imgui.set_cursor_pos_x(x0 + max((width - imgui.calc_text_size(text).x) / 2, 0.0))
            imgui.text_disabled(text)

        centred(f"{hi:.2f}")
        slider_w = em(2.2)
        imgui.set_cursor_pos_x(x0 + (width - slider_w) / 2)
        imgui.push_style_color(imgui.Col_.slider_grab, vec4(color, 0.85))
        imgui.push_style_color(imgui.Col_.slider_grab_active, vec4(color, 1.0))
        changed, shown = imgui.v_slider_float(
            f"##{key}", imgui.ImVec2(slider_w, track_h), shown, float(lo), float(hi), "%.2f",
        )
        imgui.pop_style_color(2)
        if changed:
            self._slider_pending[key] = shown
        if imgui.is_item_deactivated_after_edit():
            self._slider_pending.pop(key, None)
            apply(shown)
        centred(f"{lo:.2f}")
        if side is None:
            centred(name)
        else:
            arrow_w = imgui.get_frame_height()
            name_w = imgui.calc_text_size(name).x
            imgui.set_cursor_pos_x(x0 + max((width - arrow_w - em(0.3) - name_w) / 2, 0.0))
            if imgui.arrow_button(f"##{key}_side", imgui.Dir.right if side == "right" else imgui.Dir.left):
                flip()
            imgui.same_line(0, em(0.3))
            imgui.text_disabled(name)
        set_tooltip(tooltip, show_mark=False)
        centred(count)
        imgui.dummy(imgui.ImVec2(width, 1))
        imgui.end_group()

    def _draw_candidate_row(self, session: CurationSession) -> None:
        """B to D: template, focused candidate, PCA."""
        if not session.n:
            imgui.text_disabled("No candidate events found; lower the threshold.")
            return
        avail = imgui.get_content_region_avail()
        gap = em(0.5)
        width = max((avail.x - 2 * gap) / 3, em(8))
        panels = (
            ("B. Current template", self._draw_template),
            (self._candidate_title(session), self._draw_candidate),
            ("D. Candidate PCA (400 ms)", self._draw_pca),
        )
        for i, (title, draw) in enumerate(panels):
            if i:
                imgui.same_line(0, gap)
            with imgui_ctx.begin_child(f"##curation_panel{i}", imgui.ImVec2(width, 0)):
                imgui.text_colored(theme.to_vec4(theme.THEME.accent), title)
                # the plot takes exactly what is left under the title: a
                # fill-height plot rounds over by a pixel and grows a scrollbar
                draw(session, max(imgui.get_content_region_avail().y - 2, em(3)))

    @staticmethod
    def _candidate_title(session) -> str:
        scores = session.template_scores
        if session.template is None or not len(scores):
            return "C. Focused candidate (cosine n/a)"
        return f"C. Focused candidate (cosine {scores[session.current]:.2f})"

    def _fit_for(self, name: str, key: tuple) -> bool:
        """Whether panel ``name`` should refit: its content changed."""
        if self._panel_keys.get(name) == key:
            return False
        self._panel_keys[name] = key
        return True

    def _draw_template(self, session, height: float) -> None:
        source = session.template_source
        fit = self._fit_for("template", (session.mode, session.recording_id, len(source)))
        with line_plot(
            "##curation_b", "aligned time (ms)", "z", height=height, fit=fit, legend=False,
        ) as ok:
            if not ok:
                return
            if session.template is None:
                implot.plot_text("No Yes events yet", 0.0, 0.0)
                return
            t_ms = session.template_time_ms
            for i in source:
                line(f"##snippet{i}", session.short_snippet(i), x=t_ms, color=SNIPPET_COLOR, alpha=0.18)
            line("template", session.template, x=t_ms, color=TEMPLATE_COLOR, weight=3.0)

    def _draw_candidate(self, session, height: float) -> None:
        i = session.current
        half = session.candidate_window_ms
        t_ms = session.long_time_ms
        y = session.long_snippet(i)
        if self._fit_for("candidate", (session.mode, session.recording_id, i, session.threshold)):
            inside = (t_ms >= -half) & (t_ms <= half) & np.isfinite(y)
            ys = y[inside] if inside.any() else y[np.isfinite(y)]
            if session.template is not None:
                ys = np.concatenate([ys, session.template])
            lo, hi = (float(np.min(ys)), float(np.max(ys))) if ys.size else (-1.0, 1.0)
            pad = 0.05 * (hi - lo or 1.0)
            implot.set_next_axes_limits(-half, half, lo - pad, hi + pad, imgui.Cond_.always)
        with line_plot("##curation_c", "time from event (ms)", "z", height=height, legend=True) as ok:
            if not ok:
                return
            line("candidate", y, x=t_ms, color=SNIPPET_COLOR, weight=1.2)
            if session.template is not None:
                line("template", session.template, x=session.template_time_ms, color=TEMPLATE_COLOR, weight=2.5)
            dotted_vline(0.0, ZERO_COLOR)

    def _draw_pca(self, session, height: float) -> None:
        scores = session.pca_scores
        explained = session.pca_explained * 100.0
        visible = session.visible
        focused = np.flatnonzero(visible == session.current)
        if self._fit_for("pca", (session.mode, session.recording_id, session.n)):
            self.pca.refit()
        picked = self.pca.draw(
            scores[visible, 0], scores[visible, 1], session.colors(visible),
            x_label=f"PC1 ({explained[0]:.1f}%)",
            y_label=f"PC2 ({explained[1]:.1f}%)",
            height=height,
            focused=int(focused[0]) if focused.size else None,
            tooltip=lambda k: (
                f"event {visible[k] + 1}: {session.label(visible[k])}\n"
                f"PC1 {scores[visible[k], 0]:.2f}  PC2 {scores[visible[k], 1]:.2f}"
            ),
            inside=lambda: self._pca_inside(session, scores, visible),
            flags=BOX_PLOT_FLAGS if self.box_mode is not None else 0,
        )
        if picked is not None:
            session.select(int(visible[picked]))

    def _pca_inside(self, session, scores, visible) -> None:
        self._box_select("pca", scores[visible, 0], scores[visible, 1], visible, self.pca)
        self._draw_pc1_line(session)

    def _draw_pc1_line(self, session) -> None:
        """The A3 line on the PCA: drag it to set the PC1 auto-pass, applied
        on release like the trace's lines; the label names the passing side."""
        if not session.seeded:
            return
        shown = self._pc1_drag if self._pc1_drag is not None else session.auto_pass_pc1_shown
        value, held = drag_vline(PC1_LINE_ID, shown, PC1_COLOR)
        if held:
            self._pc1_drag = value
        elif self._pc1_drag is not None:
            session.set_auto_pass_pc1(self._pc1_drag)
            self._pc1_drag = None
        left = session.auto_pass_pc1_side == "left"
        text = "< pass" if left else "pass >"
        half_w = 0.5 * imgui.calc_text_size(text).x
        top = implot.get_plot_limits().y.max
        implot.push_style_color(implot.Col_.inlay_text, vec4(PC1_COLOR))
        implot.plot_text(text, value, top, imgui.ImVec2(-(half_w + 6) if left else half_w + 6, 8))
        implot.pop_style_color()

    # ------------------------------------------------------------------
    # right tab
    # ------------------------------------------------------------------

    def draw_tab(self) -> None:
        """The controls column, two tabs: Decision (with the event, navigation,
        view filter and files under it) and Recordings (mode, source, the
        table). It is the Curation tab on a figure's right bar, and the
        column beside the dashboard in the standalone hosts."""
        self.strip.report_right_tab("curation")
        self._mark_hovered()
        self._draw_tabs(source=True)

    def draw_embedded(self) -> None:
        """The tabs without the source picker, for a host that hands traces
        over itself (the line-scan viewer's ROI panel)."""
        self._mark_hovered()
        self._draw_tabs(source=False)

    def _draw_tabs(self, source: bool) -> None:
        if not imgui.begin_tab_bar("##curation_tabs"):
            return
        if imgui.begin_tab_item("Decision")[0]:
            self._draw_decision_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Recordings")[0]:
            self._draw_mode_row()
            if source:
                self._draw_source()
            self._draw_recordings()
            self._draw_controls()
            imgui.end_tab_item()
        imgui.end_tab_bar()

    def _draw_decision_tab(self) -> None:
        session = self._ready()
        if session is None:
            imgui.text_wrapped(self.status)
            return
        self._draw_decision(session)
        self._draw_event(session)
        self._draw_navigation(session)
        self._draw_filter(session)
        self._draw_files(session)

    def _draw_controls(self) -> None:
        imgui.text_wrapped(self.status)

    def _draw_mode_row(self) -> None:
        imgui.set_next_item_width(em(6))
        changed, idx = imgui.combo("mode", MODES.index(self.mode), list(MODES))
        set_tooltip(
            "fast: threshold the denoised trace, seed from the top 25% amplitudes.\n"
            "slow: threshold its low-pass view, seed the same way.\n"
            "manual: no seed template; Yes events build it.",
            show_mark=False,
        )
        if changed:
            self.set_mode(MODES[idx])
        if self.mode == "slow":
            imgui.same_line(0, em(0.6))
            imgui.set_next_item_width(em(4.5))
            _changed, value = imgui.input_float("Hz", self.slow_cutoff_hz, 0.0, 0.0, "%.0f")
            if imgui.is_item_deactivated_after_edit():
                self.set_slow_cutoff(value)
            set_tooltip("Low-pass cutoff of the slow analysis trace; changing it reloads.", show_mark=False)

    def _draw_source(self) -> None:
        section("Source")
        if imgui.button("data path..."):
            self.prompt.start(self.data_path or str(Path.home()))
        imgui.same_line(0, em(0.5))
        imgui.text_disabled(self.data_path or "none")
        if self.data_path and imgui.is_item_hovered():
            imgui.set_tooltip(self.data_path)

    def _draw_recordings(self) -> None:
        rows_shown = self.shown
        if not rows_shown:
            return
        section("Recordings")
        n_loaded = len(self.loaded())
        imgui.text_disabled(f"{n_loaded}/{len(rows_shown)} loaded")
        imgui.same_line(0, em(0.6))
        if imgui.small_button("load all"):
            self.load_all(None)
        set_tooltip("load every processed recording of every experiment", show_mark=False)
        if self._busy:
            imgui.text_wrapped(self.loading_line())
        if self._trace_sources and not self._busy:
            imgui.text_disabled("raw lines: the first load runs the wavelet denoiser, about 1 min per 100 s of recording; cached after")
        # no fixed height and no scroll region: the table takes as many
        # rows as it has and the tab itself scrolls when they overflow
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.sizing_fixed_fit
        )
        if not imgui.begin_table("##curation_recordings", 4, flags):
            return
        for name in ("experiment", "recording", "cand", "yes / no"):
            imgui.table_setup_column(name)
        imgui.table_headers_row()
        for rec in rows_shown:
            key = (self.mode, rec.rid)
            session = self.sessions.get(key)
            ready = session is not None and session.loaded and key not in self._busy
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(rec.experiment)
            imgui.table_next_column()
            short = rec.label.rsplit(" / ", 2)[-2:] if " / " in rec.label else [rec.label]
            clicked, _ = imgui.selectable(
                f"{' / '.join(short)}##rec{rec.rid}", rec.rid == self.current,
                imgui.SelectableFlags_.span_all_columns,
            )
            if clicked:
                self.load(rec.rid)
            if imgui.is_item_hovered():
                imgui.set_tooltip(rec.label if not rec.pre_denoised else f"{rec.label}\n{rec.source}")
            imgui.table_next_column()
            if ready:
                imgui.text(str(session.n))
            elif key in self._busy:
                imgui.text_disabled("...")
            elif rec.error:
                imgui.text_disabled("missing")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(rec.error)
            else:
                imgui.text_disabled("-" if rec.pre_denoised else "raw")
            imgui.table_next_column()
            if ready:
                yes, no, _un = session.counts()
                imgui.text(f"{yes} / {no}")
        imgui.end_table()

    def _draw_filter(self, session) -> None:
        section("View")
        self._draw_filter_body(session)

    def _draw_filter_body(self, session) -> None:
        current = session.view_filter
        # two per row: four in a row overrun the card's width
        for i, name in enumerate(FILTERS):
            if i % 2:
                imgui.same_line(em(5.5))
            if imgui.radio_button(name, current == name):
                session.set_view_filter(name)

    def _draw_decision(self, session) -> None:
        section("Decision")
        self._draw_decision_body(session)

    def _draw_decision_body(self, session) -> None:
        if not session.n:
            imgui.begin_disabled()
        width = imgui.ImVec2(em(4.2), em(1.8))
        with theme.button_colors((0.11, 0.60, 0.55, 1.0), (0.14, 0.72, 0.65, 1.0)):
            if imgui.button("Yes", width):
                session.set_label("yes")
        set_tooltip("y", show_mark=False)
        imgui.same_line(0, em(0.5))
        with theme.button_colors((0.84, 0.15, 0.24, 1.0), (0.95, 0.22, 0.30, 1.0)):
            if imgui.button("No", width):
                session.set_label("no")
        set_tooltip("n", show_mark=False)
        imgui.same_line(0, em(0.5))
        if imgui.button("Clear", width):
            session.set_label("unlabeled")
        set_tooltip("backspace", show_mark=False)
        if not session.n:
            imgui.end_disabled()
        # Box accept / reject: a mode. Right-drag a box on the trace or the
        # PCA, drag its edges to adjust, Apply (enter) labels what it holds.
        wide = imgui.ImVec2(em(5.6), em(1.8))
        for i, (mode, caption, base) in enumerate((
            ("yes", "Box accept", (0.11, 0.60, 0.55, 1.0)),
            ("no", "Box reject", (0.84, 0.15, 0.24, 1.0)),
        )):
            if i:
                imgui.same_line(0, em(0.5))
            hot = BOX_COLORS[mode]
            active = self.box_mode == mode
            with theme.button_colors(hot if active else base, hot):
                if imgui.button(caption, wide):
                    self.set_box_mode(mode)
            set_tooltip(
                f"box mode: right-drag a box on the trace or the PCA, adjust its edges, "
                f"then Apply (enter) labels every candidate inside {mode}; again or esc leaves",
                show_mark=False,
            )
        imgui.same_line(0, em(0.5))
        if not session.n:
            imgui.begin_disabled()
        if imgui.button("Clear all", width):
            session.clear_labels()
        set_tooltip(
            "drop every manual label of this recording (the file keeps its "
            "thresholds): the A2 / A3 / A4 rules decide every candidate again. "
            "Manual labels, including boxed ones, always win over the rules.",
            show_mark=False,
        )
        if not session.n:
            imgui.end_disabled()
        if self.box_mode is None:
            return
        n_box = int(len(self.boxed()))
        if not n_box:
            imgui.begin_disabled()
        if imgui.button("Apply", imgui.ImVec2(em(4.2), em(1.8))):
            self.apply_box()
        set_tooltip("enter", show_mark=False)
        if not n_box:
            imgui.end_disabled()
        imgui.same_line(0, em(0.5))
        if imgui.button("Cancel", imgui.ImVec2(em(4.2), em(1.8))):
            self.exit_box_mode()
        set_tooltip("esc", show_mark=False)
        if self._box_rect is None:
            imgui.text_disabled("right-drag a box on the trace or the PCA")
        else:
            imgui.text_colored(
                theme.to_vec4(BOX_COLORS[self.box_mode]),
                f"{n_box} candidate{'s' if n_box != 1 else ''} in the box -> {self.box_mode}",
            )

    def _draw_navigation(self, session) -> None:
        section("Navigation")
        self._draw_navigation_body(session)

    def _draw_navigation_body(self, session) -> None:
        if not session.n:
            imgui.begin_disabled()
        if imgui.arrow_button("##prev_event", imgui.Dir.left):
            session.step(-1)
        set_tooltip("previous candidate in view (left)", show_mark=False)
        imgui.same_line(0, em(0.3))
        if imgui.arrow_button("##next_event", imgui.Dir.right):
            session.step(1)
        set_tooltip("next candidate in view (right)", show_mark=False)
        imgui.same_line(0, em(0.6))
        imgui.text_disabled("prev / next")
        imgui.set_next_item_width(-em(3.5))
        changed, value = imgui.slider_int("event", session.current + 1, 1, max(1, session.n))
        if changed:
            session.select(value - 1)
        if not session.n:
            imgui.end_disabled()

    def _draw_event(self, session) -> None:
        section("Event")
        self._draw_event_body(session)

    def _draw_event_body(self, session) -> None:
        if not session.n:
            imgui.text_disabled("No candidate events found.")
            return
        info = session.event_info(session.current)
        imgui.text(f"event {info['index'] + 1}/{session.n}")
        imgui.text("label: ")
        imgui.same_line(0, 0)
        imgui.text_colored(theme.to_vec4(self._label_color(info["shown"])), info["shown"])
        imgui.text_disabled(f"time {info['time_s']:.3f} s · amplitude {info['amplitude']:.2f}")
        imgui.text_disabled(f"source: {info['source']}")
        score = info["template_cosine"]
        imgui.text_disabled(f"template cosine: {'n/a' if not np.isfinite(score) else f'{score:.2f}'}")
        if session.seeded:
            initial = info["initial_cosine"]
            initial_text = "n/a" if not np.isfinite(initial) else f"{initial:.2f}"
            auto_pass = "off" if session.auto_pass is None else f"{session.auto_pass:.2f}"
            pc1_line = (
                "off" if session.auto_pass_pc1 is None
                else f"{session.auto_pass_pc1:.2f} {session.auto_pass_pc1_side}"
            )
            # the checkbox is gone from the card: A4 at its floor rejects
            # nothing; an old JSON can still carry the flag off, so say so
            rejection = "" if session.waveform_rejection else ", waveform reject off"
            imgui.text_disabled(
                f"auto call: {info['auto_call'] or 'none'} (cosine {initial_text} at "
                f"A4 {session.auto_template_threshold:.2f}, A2 {auto_pass}, "
                f"PC1 {info['pc1']:.2f} at A3 {pc1_line}{rejection})"
            )

    def _draw_files(self, session) -> None:
        section("Files")
        self._draw_files_body(session)

    def _draw_files_body(self, session) -> None:
        yes, no, unlabeled = session.counts()
        imgui.text_disabled(f"yes {yes} · no {no} · unlabeled {unlabeled}")
        imgui.text_disabled(f"template source events: {len(session.template_source)}")
        # the path is long: wrap it to the column instead of running off it
        imgui.push_style_color(imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled))
        imgui.text_wrapped(f"curation file: {session.label_path}")
        imgui.pop_style_color()
        imgui.text_disabled(f"cache: {session.cache_status}")

    @staticmethod
    def _label_color(label: str):
        from mbo_utilities.vnoiser import LABEL_RGBA

        return LABEL_RGBA.get(label, LABEL_RGBA["unlabeled"])


def _build_catalog(root: Path, logger) -> list[Recording]:
    """Every recording under ``root``: each experiment's scan / domain traces
    for a spatial JEDI Data / animal / experiment / PF folder, else the raw
    ``.mat`` recordings of a folder (or the one file)."""
    from vnoiser.dataset import SpatialJediDataset, open_recording_dataset

    if SpatialJediDataset.can_open(root):
        dataset = SpatialJediDataset(root)
        if dataset.requires_experiment_selection:
            if dataset.scope == "data":
                animals = [path for _, path in dataset.animal_options()]
            else:
                animals = [str(root)]
            experiments = [
                path for animal in animals for _, path in dataset.experiment_options(animal)
            ]
        else:
            experiments = [str(dataset.fixed_experiment)]
        catalog = []
        for experiment in experiments:
            try:
                refs = dataset.select_experiment(experiment)
            except Exception as error:
                logger.warning("skipping %s: %s", experiment, error)
                continue
            catalog += [
                Recording(ref.recording_id, ref.label, ref.experiment, str(ref.pf_dir), True)
                for ref in refs
            ]
        return catalog
    try:
        dataset = open_recording_dataset(root)
    except FileNotFoundError:
        return []
    return [
        Recording(value, label, root.name, str(root), False)
        for label, value in dataset.recording_options()
    ]


def attach_curation_widget(parent: Any, focus: bool = False) -> EventCurationWidget | None:
    """Turn the curation widget on for a ``PreviewDataWidget``. A PF folder
    or a line scan on screen is opened in it (:meth:`EventCurationWidget.open_array`);
    anything else reopens the last data path. Returns None (logged) when it
    cannot be built."""
    widget = getattr(parent, "event_curation", None)
    if widget is not None:
        widget.focus_tab = widget.focus_tab or focus
        return widget
    data = getattr(getattr(parent, "image_widget", None), "data", None)
    arr = data[0] if data else None
    kind = curation_source(arr) if arr is not None else ""
    try:
        widget = EventCurationWidget(
            parent, strip=getattr(parent, "top_strip", None), data_path="" if kind else None,
        )
    except Exception:
        parent.logger.warning("event curation widget unavailable", exc_info=True)
        parent.event_curation = None
        return None
    if kind:
        widget.open_array(arr)
    widget.focus_tab = focus
    parent.event_curation = widget
    return widget


def detach_curation_widget(parent: Any) -> None:
    """Turn the curation widget off; its sessions are dropped, the JSON stays."""
    widget = getattr(parent, "event_curation", None)
    if widget is None:
        return
    widget.close()
    parent.event_curation = None


def install_hint() -> str:
    return f"vnoiser is not installed: {VNOISER_HINT}"
