"""Event curation of voltage traces with vnoiser.

The curation notebook's dashboard as one panel, over every recording a
``PF`` folder holds. Scanning a path finds the folder (the folder itself,
its traces file, the experiment folder holding it, or a line scan with one
beside it), lists every scan / domain trace the vnoiser pipeline wrote and
loads them in the background; a line scan without a folder lists its raw
ROIs for the denoiser instead. The ``Curation`` panel shows one recording
at a time in the notebook's grid: the
trace with its candidates (A) beside the threshold / slider card (A1 to A4);
the arrows flip through recordings. When the scan ran with real-time motion correction,
its motion correction (the RTMC shifts) sits over the trace on the same
time axis, behind the ``MC`` checkbox, and the trace's coherence with it by
frequency under the trace. The Decision and
Recordings tabs (labels, navigation, the table, the mode) are the column
beside the dashboard in the curation window (``gui/curation_viewer.py``,
``mbo curate``) and the card under the ROI table in the line-scan viewer.
Every plot reads the focused session, so a label, a threshold or a flip
updates all of them at once. Two methods: vnoiser's, every rule from
``vnoiser.curation`` through :class:`mbo_utilities.vnoiser.CurationSession`
(one per mode and recording, each with its own JSON file, as the notebook's
sections have), and ``slow_fast``, the fast peaks inside slow events of the
raw dF/F (:class:`mbo_utilities.annotation.slow_fast.SlowFastSession`, drawn
by :class:`~mbo_utilities.gui.slow_fast_panels.SlowFastPanelsMixin`), which
runs without vnoiser. ``compare`` draws the other method's events over the
trace.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from imgui_bundle import imgui, imgui_ctx, implot
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities import log
from mbo_utilities.analysis.slow_fast import SlowFastSettings
from mbo_utilities.analysis.trace_readouts import MotionCoupling, motion_coupling
from mbo_utilities.annotation.slow_fast import (
    LABEL_RGBA,
    SlowFastSession,
    curated_trace,
    kept_dprime,
    kept_times,
)
from mbo_utilities.annotation.slow_fast import MODE as SLOW_FAST
from mbo_utilities.arrays.mesc import rtmc_motion, unit_rtmc
from mbo_utilities.gui import _theme as theme
from mbo_utilities.gui._availability import HAS_VNOISER
from mbo_utilities.gui._files import PathPrompt, draw_path_prompt
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui._theme import card, em, section
from mbo_utilities.gui._top_strip import TopPanel, TopStrip
from mbo_utilities.gui.imgui.lines import (
    decimate_minmax,
    drag_hline,
    line,
    line_plot,
    subplots,
    vec4,
    vlines,
)
from mbo_utilities.gui.imgui.motion import MOTION_COLORS, MotionPlot
from mbo_utilities.gui.imgui.panels import draw_keybinds_popup
from mbo_utilities.gui.imgui.scatter import ScatterPlot
from mbo_utilities.gui.imgui.vnoiser_help import (
    KEYBINDS,
    TOOLTIP,
    draw_vnoiser_help,
)
from mbo_utilities.gui.slow_fast_panels import SlowFastPanelsMixin
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.install import VNOISER_HINT
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.results import (
    TRACES_PKL,
    ResultsArray,
    recording_id,
    results_dir_of,
    trace_label,
    vnoiser_run_for_mesc,
)
from mbo_utilities.vnoiser import MODES

if TYPE_CHECKING:
    from mbo_utilities.vnoiser.session import CurationSession

__all__ = [
    "KEYBINDS",
    "EventCurationWidget",
    "Recording",
]

# the trace (A, with its cards); the strip asks for MOTION_HEIGHT more while the motion plot sits over the trace in MOTION_SHARE of its cell, and COUPLING_HEIGHT more for the coherence plot under it
PANEL_HEIGHT = 300
MOTION_HEIGHT = 120
COUPLING_HEIGHT = 150
COUPLING_TICKS_HZ = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
MOTION_SHARE = 0.35
SLIDER_COL_EM = 4.2
SLIDER_GAP_EM = 0.4
SLIDER_CARD_PAD_EM = 1.4
FILTERS = ("all", "yes", "no", "unlabeled")

TRACE_COLOR = (0.85, 0.85, 0.85, 1.0)
LOWPASS_COLOR = (0.35, 0.60, 0.95, 1.0)
THRESHOLD_COLOR = (0.84, 0.15, 0.24, 1.0)
AUTO_PASS_COLOR = (0.16, 0.62, 0.56, 1.0)
PC1_COLOR = (0.62, 0.45, 0.90, 1.0)
COSINE_COLOR = (0.95, 0.68, 0.25, 1.0)
FOCUS_COLOR = THRESHOLD_COLOR
BOX_COLORS = {
    "yes": (0.14, 0.72, 0.65, 1.0),
    "no": (0.95, 0.22, 0.30, 1.0),
}  # the box and its rings take the label's colour
BOX_PLOT_FLAGS = (
    implot.Flags_.no_box_select | implot.Flags_.no_menus
)  # in box mode right-drag belongs to the box
BOX_TOOL_ID = 7

_MODE_TITLES = {
    "fast": "fast candidates",
    "slow": "slow (<{cutoff:g} Hz) candidates",
    "manual": "manual candidates",
    SLOW_FAST: "fast peaks in slow events",
}
WIDGET_MODES = (*MODES, SLOW_FAST)
METHODS = {"vnoiser": "vnoiser", SLOW_FAST: "slow + fast"}
COMPARE_COLOR = (0.85, 0.45, 0.95, 0.55)
COMPARE_LABELS = {"vnoiser": "vnoiser events", SLOW_FAST: "slow + fast peaks"}
CHANCE_COLOR = (0.62, 0.62, 0.65, 0.3)
COUPLING_TOOLTIP = (
    "the share of this trace's 1-50 Hz power the motion correction explains, past what "
    "the same trace gets moved along in time; p is how many of those moves reach it. "
    "The plot under the trace shows it by frequency: a line above the grey band follows "
    "the motion there. It says the two move together, never which moved which: a cell "
    "that fires with movement looks the same."
)
DPRIME_TOOLTIP = (
    "d': how far the peaks stand above the trace just before them, in noise SDs: the "
    "1 ms at each peak against the 1 ms ending 3 ms before it, over the n isolated "
    "candidates not called no. A threshold between the two gets Phi(d' / 2) right: "
    "1 is 69%, 2 is 84%, 3 is 93%. Candidates are picked at peaks, so noise alone "
    "scores above 0."
)


def _mode_title(mode: str, cutoff: float) -> str:
    return _MODE_TITLES[mode].format(cutoff=cutoff)


def _centred_text(x0: float, width: float, text: str) -> None:
    imgui.set_cursor_pos_x(x0 + max((width - imgui.calc_text_size(text).x) / 2, 0.0))
    imgui.text_disabled(text)


class UnitTraces:
    """The per-ROI traces of one AOD ROI unit, read the first time a ROI is
    asked for (on the curation worker, never at open): the pipeline's own
    PF traces or an ``F.npy`` when the unit has them, else the
    ``.curation/cache`` copy of an earlier reduction keyed by the file's
    size and mtime, else the reduction itself, cached there for next time.
    """

    def __init__(self, mesc_path: Path, unit: dict, channel: int):
        self.mesc_path = Path(mesc_path)
        self.unit = unit
        self.channel = int(channel)
        self.traces: np.ndarray | None = None

    def load(self) -> np.ndarray:
        """``(rois, samples)``, read or computed once."""
        from mbo_utilities.analysis.roi_traces import (
            Domain,
            TraceSettings,
            extract_traces,
            roi_geometry,
        )
        from mbo_utilities.arrays.mesc import MescArray
        from mbo_utilities.gui.linescan_viewer import saved_roi_traces

        if self.traces is not None:
            return self.traces
        arr = MescArray(self.mesc_path, unit=self.unit["key"])
        saved = saved_roi_traces(arr, self.channel)
        if saved is not None:
            self.traces = np.asarray(saved[0])
            return self.traces
        stat = self.mesc_path.stat()
        cache_dir = self.mesc_path.parent / ".curation" / "cache"
        cache = (
            cache_dir
            / f"{self.mesc_path.stem}_{self.unit['munit']}_ch{self.channel}_traces.npz"
        )
        if cache.exists():
            with np.load(cache) as saved:
                if (
                    int(saved["size"]) == stat.st_size
                    and int(saved["mtime_ns"]) == stat.st_mtime_ns
                ):
                    self.traces = np.asarray(saved["traces"])
                    return self.traces
        rois = sorted(roi_geometry(arr))
        found = extract_traces(
            arr,
            [Domain(str(r), [r]) for r in rois],
            TraceSettings(convert=True),
            channel=self.channel,
        )
        self.traces = np.stack([found.roi_mean[r] for r in rois]).astype(np.float32)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, traces=self.traces, size=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        return self.traces

    def roi(self, i: int) -> np.ndarray:
        return self.load()[int(i)]


def inverted_roi(traces, i: int) -> np.ndarray:
    """ROI ``i`` of a unit's traces turned over about its mean (``2 * mean - F``),
    pml_utilities' Invert Deflection: a dimming indicator's events point up.
    """
    from mbo_utilities.analysis.roi_traces import invert

    return invert(traces.roi(i))


def raw_linescan_traces(mesc_path, channel: int = 0) -> list[dict]:
    """Every recording traces come from in a ``.mesc`` (line scan,
    chessboard or ribbon patches, a single patch or a raster:
    ``is_trace_unit``): ``[{"key", "munit", "fs", "n_rois", "traces"}, ...]`` where
    ``traces`` is a :class:`UnitTraces`, read when first asked for. Nothing
    is read here beyond the units' metadata, so opening a file with many
    long scans costs nothing until a ROI is clicked.
    """
    from mbo_utilities.arrays.mesc import is_trace_unit, list_mesc_units

    mesc_path = Path(mesc_path)
    out = []
    for unit in list_mesc_units(mesc_path):
        if not is_trace_unit(unit):
            continue
        out.append(
            {
                "key": unit["key"],
                "munit": unit["munit"],
                "fs": float(unit["fs"]),
                "n_rois": int(unit["nrois"]),
                "traces": UnitTraces(mesc_path, unit, channel),
            }
        )
    return out


@dataclass
class Recording:
    """One curatable recording: one ROI trace of one unit of a run
    (``source`` is the run, ``unit`` and ``roi`` say which) or a raw trace
    handed over in memory (``source`` is the file it came from).
    """

    rid: str
    label: str
    source: str
    pre_denoised: bool
    errors: dict[str, str] = field(default_factory=dict)  # mode -> why it did not load
    unit: str = ""
    roi: str = ""


class EventCurationWidget(SlowFastPanelsMixin):
    """The curation panel and its controls on a host with a figure.

    Parameters
    ----------
    parent
        Anything with ``image_widget`` (a figure) and a ``logger``.
    strip : TopStrip, optional
        The host's top strip (or a ``PanelHost`` standing in for one); a
        private one is built without.
    data_path : str, optional
        Path to scan at once; None reopens the last one, "" none.
    """

    def __init__(
        self, parent: Any, strip: TopStrip | None = None, data_path: str | None = None
    ):
        self.parent = parent
        self.logger = (
            parent.logger if parent is not None else log.get("gui.event_curation")
        )
        self.figure = parent.image_widget.figure
        self._own_strip = strip is None
        self.strip = TopStrip(self.figure) if self._own_strip else strip
        self.strip.add_hook(self._frame)
        self.panel = TopPanel("curation", "Curation", self.draw_panel, PANEL_HEIGHT, 12)
        self.strip.register(self.panel)

        self.mode = "fast" if HAS_VNOISER else SLOW_FAST
        self._vnoiser_mode = "fast"
        self.slow_cutoff_hz = 40.0
        self.slow_fast_settings = SlowFastSettings()
        self._exporting = False  # slow / fast tables being written
        self.compare = False
        self.catalog: list[Recording] = []
        self.sessions: dict[tuple[str, str], CurationSession | SlowFastSession] = {}
        self.raw_inverted = True
        self._raw_scan: tuple[Path, int] | None = None
        self.current = ""
        self._run: ResultsArray | None = None
        self._trace_sources: dict[str, dict] = {}
        self.motion: dict[tuple[str, str], MotionPlot] = {}
        self.coupling: dict[tuple[str, str], MotionCoupling | None] = {}
        self._scan_keys: dict[str, tuple[str, str] | None] = {}
        self._motion_ratios = implot.SubplotsRowColRatios(
            row_ratios=[MOTION_SHARE, 1.0 - MOTION_SHARE]
        )
        self.show_motion = True
        self._motion_shown = False
        self.on_focus = None
        self._last_focus = None
        self.on_recording = None
        self._last_recording = None
        self.scope = None
        self.data_path = ""
        self.prompt = PathPrompt(
            "Curation data",
            path="",
            action="scan",
            hint="a PF folder (or the experiment folder holding it), or a line-scan .mesc",
        )
        self._folder_dialog = None
        self.status = "Set a data path: every scan / domain of the PF folder is listed and loaded."

        self._busy: set[tuple[str, str]] = set()
        self._jobs: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._active: tuple | None = None
        self.timeline_points = ScatterPlot("##curation_timeline_pts", marker_size=7.0)
        self._line_drags: dict[str, float] = {}
        self.autofit = True
        self.show_keybinds = False
        self.show_help = False
        self.box_mode: str | None = None
        self._box_rect: dict | None = None
        self._box: tuple[str, np.ndarray] | None = None
        self._fit_timeline = False
        self._timeline_key = None
        self._timeline_trace_cache: dict[str, tuple] = {}
        self._threshold_drag: float | None = None
        self._auto_pass_drag: float | None = None
        self._slider_pending: dict[str, float] = {}

        self._hovered = False
        self._closed = False

        if data_path is None:
            last = get_last_dir("vnoiser")
            data_path = str(last) if last else ""
        if data_path:
            self.scan(data_path)

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
        """True when the host shows this recording (see ``scope``)."""
        return self.scope is None or bool(self.scope(rec))

    @property
    def shown(self) -> list[Recording]:
        """The catalog narrowed to the host's scope."""
        return [r for r in self.catalog if self.in_scope(r)]

    def recording(self, rid: str) -> Recording | None:
        return next((r for r in self.catalog if r.rid == rid), None)

    def loaded(
        self, mode: str | None = None
    ) -> list[tuple[Recording, CurationSession]]:
        """Every loaded recording of ``mode`` (the current one), catalog order."""
        mode = mode or self.mode
        out = []
        for rec in self.shown:
            session = self.sessions.get((mode, rec.rid))
            if (
                session is not None
                and session.loaded
                and (mode, rec.rid) not in self._busy
            ):
                out.append((rec, session))
        return out

    def scan(self, path) -> None:
        """Catalog every ROI trace of a vnoiser run and start loading them.
        ``path`` is the run (a results file or a ``PF`` folder of pickles),
        its traces file, the folder holding it, or a line scan with one
        beside it.
        """
        self.sessions.clear()
        self._trace_sources.clear()
        self.catalog = []
        self.current = ""
        self._run = None
        path = Path(path).expanduser()
        note = ""
        if path.suffix.lower() == ".mesc":
            run = vnoiser_run_for_mesc(path)
            if run is None:
                self.data_path = ""
                self.status = (
                    f"{path.name} is a raw line scan with no vnoiser run beside it. Open it "
                    "with 'mbo curate <file>.mesc' and curate its lines there, or point at "
                    "a run."
                )
                return
            note = f" (vnoiser run of {path.name})"
            path = run
        run_dir = results_dir_of(path)
        if run_dir is None:
            self.data_path = ""
            self.status = (
                f"no vnoiser run at {path}: expected what the pipeline wrote (a results "
                f"file or a PF folder of {TRACES_PKL}), the folder holding it, or a line "
                "scan with one beside it."
            )
            return
        self.data_path = str(run_dir)
        try:
            self._run = ResultsArray(run_dir, source=False)
        except Exception as error:
            self.logger.warning("cannot open %s", run_dir, exc_info=True)
            self.status = f"cannot open {run_dir}: {error}"
            self.data_path = ""
            return
        self.catalog = _build_catalog(self._run)
        set_last_dir("vnoiser", self.data_path)
        self.prompt.path = self.data_path
        if not self.catalog:
            self.status = f"no ROI traces in {run_dir}"
            return
        self.status = f"{len(self.catalog)} recordings ({len(self._run.results.units)} units) in {run_dir}{note}"
        first = next((r for r in self.shown), None)
        if first is not None:
            self.current = first.rid
            self.load_all()

    @property
    def method(self) -> str:
        """``vnoiser`` or ``slow_fast``: which kind of session the focused one is."""
        return SLOW_FAST if self.mode == SLOW_FAST else "vnoiser"

    def set_method(self, method: str) -> None:
        """Switch method on the same recording; vnoiser comes back in its last mode."""
        self.set_mode(SLOW_FAST if method == SLOW_FAST else self._vnoiser_mode)

    def set_mode(self, mode: str) -> None:
        if mode not in WIDGET_MODES or mode == self.mode:
            return
        if mode in MODES and not HAS_VNOISER:
            self.status = f"vnoiser is not installed: {VNOISER_HINT}"
            return
        if mode in MODES:
            self._vnoiser_mode = mode
        self.mode = mode
        self._fit_timeline = True
        rec = self.recording(self.current)
        if rec is not None:
            self.load(self.current)
            if rec.pre_denoised:
                self.load_all()

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
        if rec.errors.get(self.mode):
            self.status = f"{rec.label}: {rec.errors[self.mode]}"
            return
        self.current = rid
        self._fit_timeline = True
        key = (self.mode, rid)
        if key in self.sessions or key in self._busy:
            return
        self._enqueue(rec, self.mode)

    def load_all(self) -> None:
        """Load every shown recording that can load without a click: the
        folder's traces and the raw traces handed over.
        """
        for rec in self.shown:
            if not rec.pre_denoised and rec.rid not in self._trace_sources:
                continue
            key = (self.mode, rec.rid)
            if key not in self.sessions and key not in self._busy:
                self._enqueue(rec, self.mode)
        if not self.current:
            first = next(
                (r for r in self.catalog if (self.mode, r.rid) in self._busy), None
            )
            if first is not None:
                self.current = first.rid

    def add_trace(
        self, trace, fs_hz, *, recording_id, label, source_path, curation_dir=None
    ) -> Recording:
        """List a trace held in memory (a line-scan ROI) as a raw recording;
        loading it runs vnoiser's denoiser on the worker unless a cache
        exists. See :meth:`CurationSession.load_trace`.
        """
        source_path = str(Path(source_path))
        rid = str(recording_id)
        rec = self.recording(rid)
        if rec is None:
            rec = Recording(rid, str(label), source_path, False)
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

    def load_trace(
        self, trace, fs_hz, *, recording_id, label, source_path, curation_dir=None
    ) -> None:
        """:meth:`add_trace`, then load it."""
        rec = self.add_trace(
            trace,
            fs_hz,
            recording_id=recording_id,
            label=label,
            source_path=source_path,
            curation_dir=curation_dir,
        )
        self.load(rec.rid)

    def scan_source(self, rec: Recording) -> tuple[str, str] | None:
        """The ``(mesc path, unit)`` a recording's scan was read from: the
        file a raw trace came from (its id is ``<stem>/<unit>/roi=<n>``), or
        the PF folder's source line scan (named in its ``pipeline.json``, or
        laid out beside it) and the scan's unit; None when no line scan is
        reachable.
        """
        source = self._trace_sources.get(rec.rid)
        if source is not None:
            path = Path(source["source_path"])
            parts = rec.rid.split("/")
            if path.suffix.lower() != ".mesc" or len(parts) < 3:
                return None
            return str(path), parts[1]
        run = self._run
        if not rec.pre_denoised or run is None or run.source_recording is None:
            return None
        unit = run.results.units.get(rec.unit)
        key = str(
            (unit.attrs.get("source_unit") if unit is not None else "")
            or f"MUnit_{rec.unit}"
        )
        return str(run.source_recording), key

    @property
    def motion_plot(self) -> MotionPlot | None:
        """The focused recording's motion correction, once its scan has been
        read and went through one; None otherwise.
        """
        plot = self.motion.get(self._scan_keys.get(self.current))
        return plot if plot else None

    def scan_raw_mesc(self, mesc_path, channel: int = 0, inverted=None) -> int:
        """Every ROI of every recording traces come from in a ``.mesc``
        (line scans, chessboard or ribbon patches, single patches) as a raw
        recording the denoiser runs on when clicked. ``inverted`` (default
        ``raw_inverted``) turns each trace over about its mean first, so a
        dimming indicator's events are the peaks curation looks for; the
        recording id says so, so labels and caches of one polarity never
        stand for the other. Returns how many.
        """
        mesc_path = Path(mesc_path)
        if inverted is not None:
            self.raw_inverted = bool(inverted)
        self._raw_scan = (mesc_path, int(channel))
        suffix = "/inverted" if self.raw_inverted else ""
        self.sessions.clear()
        self._trace_sources.clear()
        self.catalog = []
        self.current = ""
        self._run = None
        n = 0
        for unit in raw_linescan_traces(mesc_path, channel):
            for i in range(unit["n_rois"]):
                read = (
                    partial(inverted_roi, unit["traces"], i)
                    if self.raw_inverted
                    else partial(unit["traces"].roi, i)
                )
                self.add_trace(
                    read,
                    unit["fs"],
                    recording_id=f"{mesc_path.stem}/{unit['munit']}/roi={i}{suffix}",
                    label=f"{unit['munit']} ROI {i}",
                    source_path=mesc_path,
                )
                n += 1
        self.data_path = str(mesc_path)
        self.prompt.path = self.data_path
        if not n:
            self.status = f"no AOD ROI units in {mesc_path.name}"
        elif self.method == SLOW_FAST:
            self.status = f"{n} raw ROI traces in {mesc_path.name}; click a recording to detect its events"
        else:
            self.status = (
                f"{n} raw ROI traces in {mesc_path.name}; no PF folder beside it, so click a "
                "recording to run vnoiser's denoiser on it (minutes the first time, cached after)"
            )
        return n

    def set_raw_inverted(self, inverted: bool) -> None:
        """List the raw traces again the other way up, on the same ROI."""
        if self._raw_scan is None or bool(inverted) == self.raw_inverted:
            return
        current = self.current.removesuffix("/inverted")
        self.scan_raw_mesc(*self._raw_scan, inverted=inverted)
        rid = f"{current}/inverted" if inverted else current
        if self.recording(rid) is not None:
            self.current = rid

    def _enqueue(self, rec: Recording, mode: str) -> None:
        key = (mode, rec.rid)
        self._busy.add(key)
        source = self._trace_sources.get(rec.rid)
        run = self._run
        option = (
            replace(self.slow_fast_settings)
            if mode == SLOW_FAST
            else self.slow_cutoff_hz
        )
        self._jobs.put(
            (key, rec.label, partial(self._load_job, rec, mode, source, run, option))
        )
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._run_jobs, name="vnoiser-load", daemon=True
            )
            self._worker.start()

    def _load_job(self, rec: Recording, mode: str, source, run, option):
        """``option`` is the slow / fast settings or vnoiser's slow cutoff."""
        if mode == SLOW_FAST:
            session = SlowFastSession(option)
        else:
            from mbo_utilities.vnoiser import CurationSession

            session = CurationSession(rec.source, mode=mode, slow_cutoff_hz=option)
        if source is not None:
            if callable(
                source["trace"]
            ):  # a line-scan ROI's trace is a loader until now: the read runs on the worker
                source["trace"] = source["trace"]()
            message = session.load_trace(**source)
        else:
            message = session.load_run(run, rec.unit, rec.roi)
        scan_key = self.scan_source(rec)
        self._scan_keys[rec.rid] = scan_key
        if scan_key is not None and scan_key not in self.motion:  # read once per scan
            self.motion[scan_key] = MotionPlot(rtmc_motion(unit_rtmc(*scan_key)))
        plot = self.motion.get(scan_key)
        if plot:
            self.coupling[(mode, rec.rid)] = motion_coupling(
                curated_trace(session), plot.motion.at(session.t), session.fs
            )
        return session, message

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
        """The worker's job, its elapsed time and the queue, as one line."""
        active = self._active
        if active is None:
            return f"{len(self._busy)} queued" if self._busy else ""
        key, label, started = active
        elapsed = int(time.monotonic() - started)
        verb = "loading"
        if key[1] in self._trace_sources:
            verb = "detecting" if key[0] == SLOW_FAST else "denoising"
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
                    rec.errors[key[0]] = message.rsplit(": ", 1)[-1]
                if key == (self.mode, self.current) and self.loadable():
                    self.step_recording(1)
                continue
            self.sessions[key] = session
            if key == (self.mode, self.current):
                self.status = message
                self._fit_timeline = True
            elif not self._busy:
                self.status = f"{len(self.loaded())} recordings loaded"

    def _frame(self) -> None:
        self._drain()
        self._poll_folder_dialog()
        self._draw_prompt()
        self._handle_keys()
        self.show_help, self.show_keybinds = draw_vnoiser_help(
            self.show_help, self.show_keybinds
        )
        self.show_keybinds = draw_keybinds_popup(
            KEYBINDS, self.show_keybinds, "Curation keybinds"
        )
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
                    self.logger.debug(
                        "curation recording callback failed", exc_info=True
                    )
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
            start = (
                self.prompt.path
                if Path(self.prompt.path or "").exists()
                else str(Path.home())
            )
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
        if imgui.is_key_pressed(imgui.Key.h, False):
            self.show_help = not self.show_help
        if imgui.is_key_pressed(imgui.Key.escape, False):
            self.exit_box_mode()
        if self.box_mode is not None and imgui.is_key_pressed(imgui.Key.enter, False):
            self.apply_box()
        if imgui.is_key_pressed(imgui.Key.up_arrow, False):
            self.step_recording(-1)
        if imgui.is_key_pressed(imgui.Key.down_arrow, False):
            self.step_recording(1)
        if imgui.is_key_pressed(imgui.Key.m, False):
            self.set_method("vnoiser" if self.method == SLOW_FAST else SLOW_FAST)
        if imgui.is_key_pressed(imgui.Key.c, False):
            self.compare = not self.compare
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

    def loadable(self) -> list[Recording]:
        """The recordings a flip can land on: processed ones and traces handed over."""
        return [
            r
            for r in self.shown
            if not r.errors.get(self.mode)
            and (r.pre_denoised or r.rid in self._trace_sources)
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

    def draw_panel(self) -> None:
        """The Curation panel: the notebook's dashboard for one recording, the
        trace row (A with the threshold / auto-pass, Decision and Navigation
        cards), MOTION_HEIGHT taller while the motion plot shows, over the
        coherence plot of a recording long enough to have one.
        """
        self._mark_hovered()
        self._draw_flip_row()
        session = self._ready()
        if session is None:
            imgui.text_disabled(self.status)
            return
        plot = self.motion_plot
        motion = plot is not None and self.show_motion
        if motion != self._motion_shown:
            self._motion_shown = motion
            self._fit_timeline = True
            if plot is not None:
                plot.refit()
        timeline = PANEL_HEIGHT + (MOTION_HEIGHT if motion else 0)
        coupling = self.coupling.get((session.mode, self.current)) if motion else None
        self.panel.height = timeline + (COUPLING_HEIGHT if coupling else 0)
        if coupling is None:
            self._draw_timeline_row(session, plot)
            return
        top = imgui.get_content_region_avail().y * timeline / self.panel.height
        with imgui_ctx.begin_child("##curation_row_a", imgui.ImVec2(0, top)):
            self._draw_timeline_row(session, plot)
        self._draw_coupling(coupling)

    def _draw_coupling(self, coupling: MotionCoupling) -> None:
        """The trace's coherence with the motion by frequency, every axis in the
        motion plot's colour, over the band moved traces reach by chance.
        """
        f = coupling.frequency
        height = max(imgui.get_content_region_avail().y - 2, em(3))
        with line_plot(
            "##curation_coupling", "frequency (Hz)", "coherence", height=height
        ) as ok:
            if not ok:
                return
            top = 1.15 * max(coupling.coherence.max(), coupling.chance.max())
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
            implot.setup_axes_limits(f[0], f[-1], 0.0, top, implot.Cond_.always)
            implot.setup_axis_ticks(implot.ImAxis_.x1, COUPLING_TICKS_HZ)
            implot.setup_legend(
                implot.Location_.north_east, implot.LegendFlags_.horizontal
            )
            implot.plot_shaded(
                "chance",
                np.ascontiguousarray(f),
                np.zeros(f.size),
                np.ascontiguousarray(coupling.chance),
                implot.Spec(fill_color=vec4(CHANCE_COLOR)),
            )
            for label, y in coupling.axes.items():
                line(label, y, x=f, color=MOTION_COLORS.get(label[0], TRACE_COLOR))
            line("all axes", coupling.coherence, x=f, color=TRACE_COLOR, weight=2.0)

    def _draw_coupling_share(self, session, source: str) -> None:
        """The title line's share of the trace the motion explains."""
        coupling = self.coupling.get((session.mode, self.current))
        imgui.same_line(0, em(0.6))
        if coupling is None:
            imgui.text_disabled(f"{source}: recording too short to read")
            return
        p = "p < 0.001" if coupling.p < 0.001 else f"p {coupling.p:.3f}"
        imgui.text(f"{source} explains {coupling.share:.1%} ({p})")
        set_tooltip(COUPLING_TOOLTIP, show_mark=False)

    def _draw_timeline_row(
        self, session: CurationSession, plot: MotionPlot | None
    ) -> None:
        """A: the trace with its candidates, as wide as the row allows, and
        the slider card (A1 to A4) beside it. A scan that went through motion
        correction gets its motion plot over the trace, in linked subplots so
        the two share one time axis and line up; ``MC`` on the title line
        shows it.
        Decision and Navigation live in the controls column (``draw_tab``),
        not in this row.
        """
        kind = _mode_title(session.mode, self.slow_cutoff_hz)
        imgui.text_disabled(f"A. {kind}: {session.n} ({len(session.visible)} in view)")
        if not session.n:
            imgui.same_line(0, em(0.6))
            imgui.text_disabled(
                "trace excluded; untick exclude trace under Detection"
                if session.mode == SLOW_FAST and session.excluded
                else "lower the edge or peak SD"
                if session.mode == SLOW_FAST
                else "lower the threshold"
            )
        if session.mode != "slow":
            self._draw_dprime(session)
        imgui.same_line(0, 12)
        if imgui.small_button("keybinds"):
            self.show_keybinds = not self.show_keybinds
        set_tooltip("k", show_mark=False)
        imgui.same_line(0, em(0.6))
        if imgui.small_button("guide"):
            self.show_help = not self.show_help
        set_tooltip(f"h: {TOOLTIP}", show_mark=False)
        if plot is not None:
            imgui.same_line(0, em(1.2))
            _changed, self.show_motion = imgui.checkbox(
                "MC##curation_motion", self.show_motion
            )
            set_tooltip(
                f"{plot.y_label}: the motion correction applied while the scan ran, "
                "over the trace on the same time axis",
                show_mark=False,
            )
            self._draw_coupling_share(session, plot.motion.source)
        avail = imgui.get_content_region_avail()
        if session.mode == SLOW_FAST:
            n_cols = 3
        else:
            n_cols = 4 if session.seeded else 1
        card_w = (
            em(SLIDER_COL_EM) * n_cols
            + em(SLIDER_GAP_EM) * (n_cols - 1)
            + em(SLIDER_CARD_PAD_EM)
        )
        plot_w = max(avail.x - card_w - em(0.5), em(10))
        with imgui_ctx.begin_child("##curation_trace", imgui.ImVec2(plot_w, 0)):
            if plot is None or not self.show_motion:
                self._draw_timeline(session)
            else:
                height = max(imgui.get_content_region_avail().y - 2, 60.0)
                link = implot.SubplotFlags_.link_all_x | implot.SubplotFlags_.no_title
                with subplots(
                    "##curation_a", 2, 1, height, flags=link, ratios=self._motion_ratios
                ) as ok:
                    if ok:
                        focus = (
                            float(session.times_s[session.current])
                            if session.n
                            else None
                        )
                        plot.draw(
                            "##curation_motion",
                            cursor=focus,
                            duration_s=float(session.t[-1]),
                        )
                        self._draw_timeline(session)
        imgui.same_line(0, em(0.5))
        if session.mode == SLOW_FAST:
            self._draw_slow_fast_card(session, card_w, avail.y)
        else:
            self._draw_slider_card(session, card_w, avail.y)

    def _draw_dprime(self, session) -> None:
        """The title line's d'; slow candidates outlast its 1 ms windows."""
        d, n = kept_dprime(session)
        imgui.same_line(0, em(1.2))
        imgui.text(f"d' {d:.2f} (n {n})" if n > 1 else "d' n/a")
        set_tooltip(DPRIME_TOOLTIP, show_mark=False)

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
        self._draw_method_switch()
        if rec is not None and rec.rid in self._trace_sources and self._raw_scan:
            imgui.same_line(0, em(0.8))
            changed, inverted = imgui.checkbox(
                "inverted##raw_inverted", self.raw_inverted
            )
            set_tooltip(
                "Turn the raw traces over about their mean (2 x mean - F) before the "
                "denoiser, so a dimming indicator's (JEDI, ASAP) events are the peaks "
                "curation looks for. Off for an indicator that brightens. Each polarity "
                "keeps its own labels.",
                show_mark=False,
            )
            if changed:
                self.set_raw_inverted(inverted)
        if (self.mode, self.current) in self._busy:
            imgui.same_line(0, em(0.6))
            imgui.text_disabled(self.loading_line() or "loading...")

    def _draw_method_switch(self) -> None:
        """The two methods (m) and the other's events over the trace (c)."""
        imgui.same_line(0, em(1.2))
        for i, (method, caption) in enumerate(METHODS.items()):
            if i:
                imgui.same_line(0, em(0.4))
            disabled = method == "vnoiser" and not HAS_VNOISER
            imgui.begin_disabled(disabled)
            if imgui.radio_button(f"{caption}##method", self.method == method):
                self.set_method(method)
            imgui.end_disabled()
        set_tooltip(
            "m: vnoiser curates candidates on its denoised trace with a seed template; "
            "slow + fast curates the fast peaks inside slow events of the raw dF/F. "
            "Each keeps its own labels."
            + (
                f"\nvnoiser is not installed: {VNOISER_HINT}" if not HAS_VNOISER else ""
            ),
            show_mark=False,
        )
        imgui.same_line(0, em(0.8))
        imgui.begin_disabled(not HAS_VNOISER)
        _changed, self.compare = imgui.checkbox(
            "compare##curation_compare", self.compare
        )
        imgui.end_disabled()
        set_tooltip(
            "c: the other method's events on this trace (all but the ones called no) "
            "as faint lines over the trace",
            show_mark=False,
        )

    def _other_mode(self) -> str:
        return self._vnoiser_mode if self.mode == SLOW_FAST else SLOW_FAST

    def _draw_compare(self) -> None:
        """Inside panel A's open plot: the other method's events on the focused
        recording, loading them first when it can load without a click.
        """
        mode = self._other_mode()
        rec = self.recording(self.current)
        if rec is None or rec.errors.get(mode):
            return
        key = (mode, rec.rid)
        other = self.sessions.get(key)
        if other is None:
            if key not in self._busy and (rec.pre_denoised or mode == SLOW_FAST):
                self._enqueue(rec, mode)
            return
        if key in self._busy or not other.loaded:
            return
        method = SLOW_FAST if mode == SLOW_FAST else "vnoiser"
        vlines(COMPARE_LABELS[method], kept_times(other), COMPARE_COLOR, 1.0)

    def _cached_trace(
        self, name: str, t: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Min/max-decimated ``(t, y)`` for the timeline trace, cached per
        recording so zooming does not replot the full-resolution array every
        frame (mirrors the notebook's ``downsample_xy`` before plotly).
        """
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
            self._timeline_trace_cache = {}
        self._timeline_key = key
        self._fit_timeline = False
        height = max(imgui.get_content_region_avail().y - 2, 60.0)
        y_label = "dF/F" if session.mode == SLOW_FAST else "z"
        with line_plot(
            "##curation_timeline",
            "time (s)",
            y_label,
            height=height,
            fit=fit,
            flags=self._box_flags(),
        ) as ok:
            if not ok:
                return
            if self.compare and HAS_VNOISER:
                self._draw_compare()
            if session.mode == SLOW_FAST:
                self._slow_fast_items(session)
                return
            t = session.t
            t_plot, denoised_plot = self._cached_trace("denoised", t, session.denoised)
            line("denoised", denoised_plot, x=t_plot, color=TRACE_COLOR, weight=1.0)
            marker_source = session.denoised
            if session.mode == "slow":
                t_lp, lp_plot = self._cached_trace(
                    "analysis", t, session.analysis_trace
                )
                line(
                    f"<{self.slow_cutoff_hz:g} Hz low-pass",
                    lp_plot,
                    x=t_lp,
                    color=LOWPASS_COLOR,
                    weight=1.6,
                )
                marker_source = session.analysis_trace

            shown = (
                self._threshold_drag
                if self._threshold_drag is not None
                else session.threshold
            )
            value, held = drag_hline(1, shown, THRESHOLD_COLOR)
            if held:
                self._threshold_drag = value
            elif self._threshold_drag is not None:
                session.set_threshold(self._threshold_drag)
                self._threshold_drag = None

            if session.seeded and session.auto_pass is not None:
                shown = (
                    self._auto_pass_drag
                    if self._auto_pass_drag is not None
                    else session.auto_pass
                )
                value, held = drag_hline(2, shown, AUTO_PASS_COLOR)
                if held:
                    self._auto_pass_drag = value
                elif self._auto_pass_drag is not None:
                    session.set_auto_pass(self._auto_pass_drag)
                    self._auto_pass_drag = None

            ys = np.interp(session.times_s, t, marker_source) if session.n else []
            self._draw_candidate_points(session, session.times_s, ys)

    def _draw_candidate_points(self, session, times_s, ys) -> None:
        """Inside panel A's open plot: the candidates in view at ``(times_s,
        ys)`` in their label's colour, picking and box selection, the focus.
        """
        visible = session.visible
        if not session.n:
            return
        xs = np.asarray(times_s)[visible]
        ys = np.asarray(ys)[visible]
        focused = np.flatnonzero(visible == session.current)
        picked = self.timeline_points.items(
            xs,
            ys,
            session.colors(visible),
            focused=int(focused[0]) if focused.size else None,
            label="candidates",
            tooltip=lambda i: (
                f"candidate {visible[i] + 1}: {xs[i]:.3f} s, {session.label(visible[i])}"
            ),
        )
        if picked is not None:
            session.select(int(visible[picked]))
        self._box_select("timeline", xs, ys, visible, self.timeline_points)
        vlines(
            "##focus",
            [session.times_s[session.current]],
            FOCUS_COLOR,
            1.0,
            legend=False,
        )

    def _box_flags(self) -> int:
        """In box mode right-drag belongs to the box, on every plot that takes one."""
        return BOX_PLOT_FLAGS if self.box_mode is not None else 0

    def set_box_mode(self, mode: str | None) -> None:
        """Enter box mode for ``mode`` ("yes" / "no"); the same mode again
        leaves it. Any box drawn so far is dropped.
        """
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
        are ringed and remembered for :meth:`apply_box`.
        """
        if self.box_mode is None:
            return
        color = BOX_COLORS[self.box_mode]
        rect = (
            self._box_rect
            if self._box_rect is not None and self._box_rect["plot"] == name
            else None
        )
        mouse = implot.get_plot_mouse_pos()
        if rect is None:
            if not (implot.is_plot_hovered() and imgui.is_mouse_clicked(1)):
                return
            rect = self._box_rect = {
                "plot": name,
                "x0": float(mouse.x),
                "y0": float(mouse.y),
                "x1": float(mouse.x),
                "y1": float(mouse.y),
                "drawing": True,
            }
            self._box = None
        if rect["drawing"]:
            if imgui.is_mouse_down(1):
                rect["x1"], rect["y1"] = float(mouse.x), float(mouse.y)
            else:
                rect["drawing"] = False
            implot.drag_rect(
                BOX_TOOL_ID,
                rect["x0"],
                rect["y0"],
                rect["x1"],
                rect["y1"],
                vec4(color),
                implot.DragToolFlags_.no_inputs,
            )
        else:
            changed, x0, y0, x1, y1, _clicked, _hovered, _held = implot.drag_rect(
                BOX_TOOL_ID,
                rect["x0"],
                rect["y0"],
                rect["x1"],
                rect["y1"],
                vec4(color),
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
        drop the box; the mode stays on for the next one. Returns how many.
        """
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
        auto-pass peak (teal, the teal line), the PC1 auto-pass (purple; the
        arrow picks the passing side) and the seed-template cosine at or above which a candidate passes (amber).
        """
        title = "A1 - A4" if session.seeded else "A1"
        with card("##curation_sliders", title, height, width):
            set_tooltip(
                "Sliders apply on release. A1 threshold: candidates are local "
                "maxima of the trace above it. A2 peak: a candidate peaking at "
                "or above it passes whatever its shape. A3 PC1: every candidate on "
                "the passing side of this value along the snippets' first "
                "principal component passes (the arrow flips the side). A4 "
                "cosine: a candidate whose similarity to the seed template is "
                "at or above it passes, below it is rejected; at the bottom "
                "nothing is rejected. Manual labels always win. All saved per "
                "recording.",
            )
            lo, hi, _step = session.threshold_range
            self._slider_column(
                "a1",
                "thr",
                session.threshold,
                lo,
                hi,
                session.set_threshold,
                THRESHOLD_COLOR,
                f"{session.n} found",
                "candidate threshold: local maxima of the trace above it",
            )
            if not session.seeded:
                return
            alo, ahi, _step = session.auto_pass_range
            imgui.same_line(0, em(SLIDER_GAP_EM))
            self._slider_column(
                "a2",
                "peak",
                session.auto_pass if session.auto_pass is not None else ahi,
                alo,
                ahi,
                session.set_auto_pass,
                AUTO_PASS_COLOR,
                "off"
                if session.auto_pass is None
                else f"{session.auto_pass_count()}/{session.n}",
                "auto-pass peak: candidates whose marker sits at or above the teal "
                "line on the trace pass; at the bottom all pass",
            )
            plo, phi, _step = session.auto_pass_pc1_range
            side = session.auto_pass_pc1_side
            imgui.same_line(0, em(SLIDER_GAP_EM))
            self._slider_column(
                "a3",
                "PC1",
                session.auto_pass_pc1_shown,
                plo,
                phi,
                session.set_auto_pass_pc1,
                PC1_COLOR,
                "off"
                if session.auto_pass_pc1 is None
                else f"{session.auto_pass_pc1_count()}/{session.n}",
                f"PC1 auto-pass: candidates at or {'above' if side == 'right' else 'below'} this "
                "value along the snippets' first principal component pass; the arrow flips the side",
                side=side,
                flip=lambda: session.set_auto_pass_pc1_side(
                    "left" if side == "right" else "right"
                ),
            )
            clo, chi, _step = session.auto_template_threshold_range
            imgui.same_line(0, em(SLIDER_GAP_EM))
            self._slider_column(
                "a4",
                "cos",
                session.auto_template_threshold,
                clo,
                chi,
                session.set_auto_template_threshold,
                COSINE_COLOR,
                f"{session.auto_template_count()}/{session.n}",
                "cosine auto-pass: candidates whose seed-template cosine is at or above it pass, "
                "below it they are rejected",
            )

    def _slider_column(
        self,
        key,
        name,
        value,
        lo,
        hi,
        apply,
        color,
        count,
        tooltip,
        *,
        side=None,
        flip=None,
    ) -> None:
        """One column of the slider card: the range's top, a vertical slider
        that applies on release (a drag does not rebuild the candidates every
        frame), the range's bottom, the rule's short name (``side`` adds the
        PC1 arrow before it; ``flip`` is called when it is clicked) and a
        count. Everything is centred on a SLIDER_COL_EM column.
        """
        width = em(SLIDER_COL_EM)
        pending = self._slider_pending.get(key)
        shown = pending if pending is not None else float(value)
        line_h = imgui.get_text_line_height_with_spacing()
        track_h = max(imgui.get_content_region_avail().y - 4 * line_h - em(0.4), em(3))
        imgui.begin_group()
        x0 = imgui.get_cursor_pos_x()

        _centred_text(x0, width, f"{hi:.2f}")
        slider_w = em(2.2)
        imgui.set_cursor_pos_x(x0 + (width - slider_w) / 2)
        imgui.push_style_color(imgui.Col_.slider_grab, vec4(color, 0.85))
        imgui.push_style_color(imgui.Col_.slider_grab_active, vec4(color, 1.0))
        changed, shown = imgui.v_slider_float(
            f"##{key}",
            imgui.ImVec2(slider_w, track_h),
            shown,
            float(lo),
            float(hi),
            "%.2f",
        )
        imgui.pop_style_color(2)
        if changed:
            self._slider_pending[key] = shown
        if imgui.is_item_deactivated_after_edit():
            self._slider_pending.pop(key, None)
            apply(shown)
        _centred_text(x0, width, f"{lo:.2f}")
        if side is None:
            _centred_text(x0, width, name)
        else:
            arrow_w = imgui.get_frame_height()
            name_w = imgui.calc_text_size(name).x
            imgui.set_cursor_pos_x(
                x0 + max((width - arrow_w - em(0.3) - name_w) / 2, 0.0)
            )
            if imgui.arrow_button(
                f"##{key}_side", imgui.Dir.right if side == "right" else imgui.Dir.left
            ):
                flip()
            imgui.same_line(0, em(0.3))
            imgui.text_disabled(name)
        set_tooltip(tooltip, show_mark=False)
        _centred_text(x0, width, count)
        imgui.dummy(imgui.ImVec2(width, 1))
        imgui.end_group()

    def draw_tab(self) -> None:
        """The controls column, two tabs: Decision (with the event, navigation,
        view filter and files under it) and Recordings (mode, source, the
        table): the column beside the dashboard in the curation window.
        """
        self._mark_hovered()
        self._draw_tabs(source=True)

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
        if session.mode == SLOW_FAST:
            self._draw_detection(session)
        self._draw_navigation(session)
        self._draw_filter(session)
        self._draw_files(session)

    def _draw_controls(self) -> None:
        imgui.text_wrapped(self.status)

    def _draw_mode_row(self) -> None:
        if self.method == SLOW_FAST:
            imgui.text_disabled("method: slow + fast (m switches)")
            return
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
            _changed, value = imgui.input_float(
                "Hz", self.slow_cutoff_hz, 0.0, 0.0, "%.0f"
            )
            if imgui.is_item_deactivated_after_edit():
                self.set_slow_cutoff(value)
            set_tooltip(
                "Low-pass cutoff of the slow analysis trace; changing it reloads.",
                show_mark=False,
            )

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
        set_tooltip("load every scan / domain of the folder", show_mark=False)
        if self._busy:
            imgui.text_wrapped(self.loading_line())
        if self._trace_sources and not self._busy:
            imgui.text_disabled(
                "raw lines: the first load runs the wavelet denoiser, about 1 min per 100 s of recording; cached after"
            )
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.sizing_fixed_fit
        )
        if not imgui.begin_table("##curation_recordings", 3, flags):
            return
        for name in ("recording", "cand", "yes / no"):
            imgui.table_setup_column(name)
        imgui.table_headers_row()
        for rec in rows_shown:
            key = (self.mode, rec.rid)
            session = self.sessions.get(key)
            ready = session is not None and session.loaded and key not in self._busy
            imgui.table_next_row()
            imgui.table_next_column()
            clicked, _ = imgui.selectable(
                f"{rec.label}##rec{rec.rid}",
                rec.rid == self.current,
                imgui.SelectableFlags_.span_all_columns,
            )
            if clicked:
                self.load(rec.rid)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    rec.label if not rec.pre_denoised else f"{rec.label}\n{rec.source}"
                )
            imgui.table_next_column()
            if ready and session.mode == SLOW_FAST and session.excluded:
                imgui.text_disabled("excl")
            elif ready:
                imgui.text(str(session.n))
            elif key in self._busy:
                imgui.text_disabled("...")
            elif rec.errors.get(self.mode):
                imgui.text_disabled("missing")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(rec.errors[self.mode])
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
        wide = imgui.ImVec2(em(5.6), em(1.8))
        for i, (mode, caption, base) in enumerate(
            (
                ("yes", "Box accept", (0.11, 0.60, 0.55, 1.0)),
                ("no", "Box reject", (0.84, 0.15, 0.24, 1.0)),
            )
        ):
            if i:
                imgui.same_line(0, em(0.5))
            hot = BOX_COLORS[mode]
            active = self.box_mode == mode
            with theme.button_colors(hot if active else base, hot):
                if imgui.button(caption, wide):
                    self.set_box_mode(mode)
            set_tooltip(
                f"box mode: right-drag a box on the trace or panel D, adjust its edges, "
                f"then Apply (enter) labels every candidate inside {mode}; again or esc leaves",
                show_mark=False,
            )
        imgui.same_line(0, em(0.5))
        if not session.n:
            imgui.begin_disabled()
        if imgui.button("Clear all", width):
            session.clear_labels()
        set_tooltip(
            "drop every label of the peaks found now (the file keeps the settings)"
            if session.mode == SLOW_FAST
            else "drop every manual label of this recording (the file keeps its "
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
            imgui.text_disabled("right-drag a box on the trace or panel D")
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
        changed, value = imgui.slider_int(
            "event", session.current + 1, 1, max(1, session.n)
        )
        if changed:
            session.select(value - 1)
        if not session.n:
            imgui.end_disabled()

    def _draw_event(self, session) -> None:
        section("Event")
        self._draw_event_body(session)

    def _draw_event_body(self, session) -> None:
        if session.mode == SLOW_FAST:
            self._draw_slow_fast_event(session)
            return
        if not session.n:
            imgui.text_disabled("No candidate events found.")
            return
        info = session.event_info(session.current)
        imgui.text(f"event {info['index'] + 1}/{session.n}")
        imgui.text("label: ")
        imgui.same_line(0, 0)
        imgui.text_colored(
            theme.to_vec4(self._label_color(info["shown"])), info["shown"]
        )
        imgui.text_disabled(
            f"time {info['time_s']:.3f} s · peak {info['peak']:.2f} · amplitude {info['amplitude']:.2f}"
        )
        imgui.text_disabled(f"source: {info['source']}")
        score = info["template_cosine"]
        imgui.text_disabled(
            f"template cosine: {'n/a' if not np.isfinite(score) else f'{score:.2f}'}"
        )
        if session.seeded:
            initial = info["initial_cosine"]
            initial_text = "n/a" if not np.isfinite(initial) else f"{initial:.2f}"
            auto_pass = (
                "off" if session.auto_pass is None else f"{session.auto_pass:.2f}"
            )
            pc1_line = (
                "off"
                if session.auto_pass_pc1 is None
                else f"{session.auto_pass_pc1:.2f} {session.auto_pass_pc1_side}"
            )
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
        if session.mode == SLOW_FAST:
            self._draw_slow_fast_files(session)
            return
        yes, no, unlabeled = session.counts()
        imgui.text_disabled(f"yes {yes} · no {no} · unlabeled {unlabeled}")
        imgui.text_disabled(f"template source events: {len(session.template_source)}")
        imgui.push_style_color(
            imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        imgui.text_wrapped(f"curation file: {session.label_path}")  # the path is long
        imgui.pop_style_color()
        imgui.text_disabled(f"cache: {session.cache_status}")

    @staticmethod
    def _label_color(label: str):
        return LABEL_RGBA.get(label, LABEL_RGBA["unlabeled"])


def _build_catalog(run: ResultsArray) -> list[Recording]:
    """Every ROI trace a run holds that a method can curate (a denoised trace
    for vnoiser, dF/F or raw for slow / fast), units in the pipeline's order
    and ROIs in the run's; a unit without a sampling rate is left out.
    """
    return [
        Recording(
            recording_id(unit, roi),
            trace_label(unit, roi),
            str(run.path),
            True,
            unit=name,
            roi=roi,
        )
        for name, unit in run.results.units.items()
        if unit.fs and {"denoised", "dff", "raw"} & set(unit.traces)
        for roi in unit.roi_names
    ]
