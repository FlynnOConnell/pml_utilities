"""Event curation of voltage traces with vnoiser, inside the viewer.

The curation notebook's dashboard as viewer panels, over every recording a
data path holds at once. Scanning a path catalogs each animal, experiment
and scan / domain the pipeline processed, and loads them all in the
background; the ``All traces`` panel stacks them with their candidate
events, ``Curation`` is the focused one (the notebook's panel A with cards
A1 and A2) and ``Candidates`` its template, focused candidate, second-pass
preview and PCA (B to E). The Curation tab on the right bar lists the
recordings, switches mode, and carries the decision and navigation
controls. Every rule comes from ``vnoiser.curation`` through
:class:`mbo_utilities.vnoiser.CurationSession`: one per mode and
recording, each with its own JSON file, as the notebook's sections have.
"""

from __future__ import annotations

import logging
import queue
import threading
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
from mbo_utilities.gui.imgui.lines import decimate_minmax, drag_hline, line, line_plot, vlines
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

PANEL_HEIGHT = 280
CARD_WIDTH_EM = 8.5
FILTERS = ("all", "yes", "no", "unlabeled")
STACK_GAP = 1.25

TRACE_COLOR = (0.85, 0.85, 0.85, 1.0)
STACK_COLOR = (0.65, 0.65, 0.68, 1.0)
LOWPASS_COLOR = (0.35, 0.60, 0.95, 1.0)
THRESHOLD_COLOR = (0.84, 0.15, 0.24, 1.0)
AUTO_PASS_COLOR = (0.16, 0.62, 0.56, 1.0)
TEMPLATE_COLOR = (1.0, 1.0, 1.0, 1.0)
SNIPPET_COLOR = (0.30, 0.47, 0.66, 1.0)
ORIGINAL_COLOR = (0.62, 0.79, 0.91, 1.0)
FOCUS_COLOR = THRESHOLD_COLOR

KEYBINDS = (
    ("y", "label the focused candidate yes"),
    ("n", "label the focused candidate no"),
    ("backspace", "clear its label"),
    ("[ / ]", "previous / next candidate in view"),
    ("click", "focus a candidate on a trace or the PCA plot"),
    ("drag", "move the threshold / auto-pass line on the trace"),
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
        "- **All traces** (top): every loaded recording stacked, with its "
        "candidates coloured by label. Click a candidate to focus it.\n"
        "- **Curation** (top): the focused recording's trace. Drag the red "
        "line to change the candidate threshold; drag the teal line (or the "
        "A2 slider) to set the auto-pass amplitude.\n"
        "- **Candidates** (top): the current template, the focused candidate "
        "against it, the second-pass preview with rejected events removed, "
        "and the candidate PCA over 400 ms windows. Click a point to focus it.\n"
        "- **Curation** tab (right): the recordings table, mode, view filter, "
        "Yes / No / Clear, and navigation.\n\n"
        "Each mode saves to its own `PF/.curation/<mode>_template_curation.json`; "
        "reopening the same scan / domain restores it."
    )


def _mode_title(mode: str, cutoff: float) -> str:
    return _MODE_TITLES[mode].format(cutoff=cutoff)


@dataclass
class Recording:
    """One curatable recording: what a session opens and which id loads it."""

    rid: str
    label: str
    experiment: str
    source: str
    pre_denoised: bool


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
        # kept so the panel can ask for more height as recordings load
        self._all_panel = TopPanel(
            "all_traces", "All traces", self.draw_all_panel, PANEL_HEIGHT, "curation", 11
        )
        self.strip.register(self._all_panel)
        self.strip.register(
            TopPanel("curation", "Curation", self.draw_timeline_panel, PANEL_HEIGHT, "curation", 12)
        )
        self.strip.register(
            TopPanel("candidates", "Candidates", self.draw_candidate_panel, PANEL_HEIGHT, "curation", 13)
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

        self.timeline_points = ScatterPlot("##curation_timeline_pts", marker_size=7.0)
        self.stack_points = ScatterPlot("##curation_stack_pts", marker_size=3.5)
        self.pca = ScatterPlot("##curation_pca", marker_size=7.0)
        self.autofit = True
        self._fit_timeline = False
        self._fit_stack = False
        self._timeline_key = None
        self._stack_key = None
        self._panel_keys: dict[str, tuple] = {}
        self._decimated: dict[tuple, tuple] = {}
        self._threshold_drag: float | None = None
        self._auto_pass_drag: float | None = None
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
        for key in ("all_traces", "curation", "candidates"):
            self.strip.unregister(key)
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
        for rec in self.catalog:
            session = self.sessions.get((mode, rec.rid))
            if session is not None and session.loaded and (mode, rec.rid) not in self._busy:
                out.append((rec, session))
        return out

    def scan(self, path) -> None:
        """Catalog every recording under a path and start loading them."""
        self.sessions.clear()
        self._decimated.clear()
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
        first = next((r for r in self.catalog if r.pre_denoised), None)
        if first is not None:
            self.current = first.rid
            self.load_all(first.experiment)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES or mode == self.mode:
            return
        self.mode = mode
        self._fit_timeline = self._fit_stack = True
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
        for rec in self.catalog:
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

    def load_trace(self, trace, fs_hz, *, recording_id, label, source_path, curation_dir=None) -> None:
        """Curate a trace held in memory: vnoiser's denoiser runs on it on
        the worker unless a cache exists. See :meth:`CurationSession.load_trace`."""
        source_path = str(Path(source_path))
        rid = str(recording_id)
        if self.recording(rid) is None:
            self.catalog.append(Recording(rid, str(label), Path(source_path).stem, source_path, False))
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
        self.load(rid)

    def _enqueue(self, rec: Recording, mode: str) -> None:
        key = (mode, rec.rid)
        self._busy.add(key)
        source = self._trace_sources.get(rec.rid)
        cutoff = self.slow_cutoff_hz

        def work():
            session = CurationSession(rec.source, mode=mode, slow_cutoff_hz=cutoff)
            if source is not None:
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
            try:
                session, message = work()
            except Exception as error:
                job.fail(f"{type(error).__name__}: {error}")
                self._results.put((key, None, f"{label}: load failed: {error}"))
            else:
                job.done(message)
                self._results.put((key, session, message))
            finally:
                self._jobs.task_done()

    def wait(self, timeout: float | None = None) -> None:
        """Block until every queued load finishes and apply them (tests)."""
        import time

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
                continue
            self.sessions[key] = session
            self._fit_stack = True
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
        self._report_focus()
        self._hovered = False

    def _report_focus(self) -> None:
        session = self._ready()
        if session is None or not session.n:
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
        session = self._ready()
        if session is None or not session.n:
            return
        if imgui.is_key_pressed(imgui.Key.y, False):
            session.set_label("yes")
        if imgui.is_key_pressed(imgui.Key.n, False):
            session.set_label("no")
        if imgui.is_key_pressed(imgui.Key.backspace, False):
            session.set_label("unlabeled")
        if imgui.is_key_pressed(imgui.Key.left_bracket, True):
            session.step(-1)
        if imgui.is_key_pressed(imgui.Key.right_bracket, True):
            session.step(1)

    def _mark_hovered(self) -> None:
        if imgui.is_window_hovered(imgui.HoveredFlags_.root_and_child_windows):
            self._hovered = True

    # ------------------------------------------------------------------
    # top panel: every loaded trace
    # ------------------------------------------------------------------

    def _decimated_trace(self, rid: str, session: CurationSession):
        """``(t, y)`` of the analysis trace at a few thousand points, and its
        robust 0..1 scaling, cached per recording and mode."""
        y = session.analysis_trace
        key = (session.mode, rid, y.shape[0], float(session.fs))
        cached = self._decimated.get(key)
        if cached is None:
            idx, values = decimate_minmax(y, 4000)
            lo, hi = np.nanpercentile(y, (1, 99))
            span = hi - lo if hi > lo else 1.0
            cached = (idx / session.fs, values, float(lo), float(span))
            self._decimated[key] = cached
        return cached

    def draw_all_panel(self) -> None:
        """The All traces panel: every loaded recording stacked, with its
        candidates; click one to focus it."""
        self._mark_hovered()
        rows = self.loaded()
        if not rows:
            imgui.text_disabled(self.status)
            return
        n_busy = len(self._busy)
        imgui.text_disabled(
            f"{len(rows)} recordings · {_mode_title(self.mode, self.slow_cutoff_hz)}"
            + (f" · loading {n_busy} more" if n_busy else "")
        )
        imgui.same_line(0, 12)
        if imgui.button("fit##stack"):
            self._fit_stack = True
        key = (self.mode, len(rows))
        fit = self._fit_stack or key != self._stack_key
        if key != self._stack_key:
            # about 16 px per trace, within what a strip can sensibly take
            self._all_panel.height = int(min(max(PANEL_HEIGHT, 16 * len(rows) + 60), 640))
            self.strip.register(self._all_panel)
        self._stack_key = key
        self._fit_stack = False
        height = max(imgui.get_content_region_avail().y - 2, 60.0)
        with line_plot("##curation_stack", "time (s)", "", height=height, fit=fit, legend=False) as ok:
            if not ok:
                return
            xs_all, ys_all, colors_all, owners = [], [], [], []
            for row, (rec, session) in enumerate(rows):
                t, values, lo, span = self._decimated_trace(rec.rid, session)
                offset = (len(rows) - 1 - row) * STACK_GAP
                focused = rec.rid == self.current
                line(
                    f"##trace{row}", (values - lo) / span + offset, x=t,
                    color=TRACE_COLOR if focused else STACK_COLOR,
                    weight=1.4 if focused else 0.8, legend=False,
                )
                implot.plot_text(rec.label, float(t[0]), offset + 1.0, imgui.ImVec2(4, -6))
                if session.n:
                    marker_y = np.interp(session.times_s, session.t, session.analysis_trace)
                    xs_all.append(session.times_s)
                    ys_all.append((marker_y - lo) / span + offset)
                    colors_all.append(session.colors())
                    owners.append(np.full(session.n, row, dtype=int))
            if not xs_all:
                return
            xs = np.concatenate(xs_all)
            ys = np.concatenate(ys_all)
            owners = np.concatenate(owners)
            picked = self.stack_points.items(
                xs, ys, np.concatenate(colors_all),
                tooltip=lambda i: f"{rows[owners[i]][0].label}: {xs[i]:.3f} s",
            )
            if picked is not None:
                rec, session = rows[owners[picked]]
                first = int(np.flatnonzero(owners == owners[picked])[0])
                self.load(rec.rid)
                session.select(picked - first)

    # ------------------------------------------------------------------
    # top panel: the focused trace
    # ------------------------------------------------------------------

    def draw_timeline_panel(self) -> None:
        """The Curation panel: trace, candidate markers, threshold cards."""
        self._mark_hovered()
        session = self._ready()
        if session is None:
            imgui.text_disabled(self.status)
            return
        kind = _mode_title(session.mode, self.slow_cutoff_hz)
        imgui.text_disabled(
            f"A. {session.recording_label} · {kind}: {session.n} "
            f"({len(session.visible)} in view)"
        )
        imgui.same_line(0, 12)
        changed, self.autofit = imgui.checkbox("autofit", self.autofit)
        set_tooltip("Refit the axes whenever the recording or mode changes.", show_mark=False)
        if changed and self.autofit:
            self._fit_timeline = True
        imgui.same_line(0, 6)
        if imgui.button("fit"):
            self._fit_timeline = True
        set_tooltip(
            "drag pans, scroll zooms, double-click fits · shift+scroll zooms x "
            "only, alt+scroll zooms y only · drag the red line to move the "
            "threshold, the teal line the auto-pass amplitude",
            show_mark=False,
        )
        avail = imgui.get_content_region_avail()
        card_w = em(CARD_WIDTH_EM)
        n_cards = 2 if session.seeded else 1
        plot_w = max(avail.x - n_cards * (card_w + em(0.5)), em(10))
        with imgui_ctx.begin_child("##curation_trace", imgui.ImVec2(plot_w, 0)):
            self._draw_timeline(session)
        imgui.same_line(0, em(0.5))
        self._draw_threshold_card(session, card_w, avail.y)
        if session.seeded:
            imgui.same_line(0, em(0.5))
            self._draw_auto_pass_card(session, card_w, avail.y)

    def _draw_timeline(self, session: CurationSession) -> None:
        key = (session.mode, session.recording_id)
        fit = self._fit_timeline or (self.autofit and key != self._timeline_key)
        self._timeline_key = key
        self._fit_timeline = False
        height = max(imgui.get_content_region_avail().y - 2, 60.0)
        with line_plot("##curation_timeline", "time (s)", "z", height=height, fit=fit) as ok:
            if not ok:
                return
            t = session.t
            line("denoised", session.denoised, x=t, color=TRACE_COLOR, weight=1.0)
            marker_source = session.denoised
            if session.mode == "slow":
                line(
                    f"<{self.slow_cutoff_hz:g} Hz low-pass",
                    session.analysis_trace, x=t, color=LOWPASS_COLOR, weight=1.6,
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
            vlines("##focus", [session.times_s[session.current]], FOCUS_COLOR, 1.0, legend=False)

    def _draw_threshold_card(self, session, width, height) -> None:
        lo, hi, _step = session.threshold_range
        with card("##curation_a1", "A1. Threshold", height, width):
            set_tooltip(
                "Candidates are local maxima of the analysis trace above "
                "this value. Saved per recording.",
            )
            self._v_slider("threshold", session.threshold, lo, hi, session.set_threshold)

    def _draw_auto_pass_card(self, session, width, height) -> None:
        lo, hi, _step = session.auto_pass_range
        with card("##curation_a2", "A2. Auto-pass", height, width):
            set_tooltip(
                "Every candidate at or above this amplitude is auto-called "
                "pass regardless of template similarity. Saved per recording.",
            )
            shown = session.auto_pass if session.auto_pass is not None else hi
            self._v_slider("auto_pass", shown, lo, hi, session.set_auto_pass, extra_rows=1)
            changed, value = imgui.checkbox("waveform reject", session.waveform_rejection)
            set_tooltip(
                "Auto-reject candidates whose cosine similarity to the seed "
                f"template is at or below {session.auto_template_threshold:.2f}.",
                show_mark=False,
            )
            if changed:
                session.set_waveform_rejection(value)
            if session.auto_pass is None:
                imgui.text_disabled("off")

    def _v_slider(self, key, value, lo, hi, apply, extra_rows: int = 0) -> None:
        """A vertical slider that applies on release, so a drag does not
        rebuild the candidates every frame."""
        pending = self._slider_pending.get(key)
        shown = pending if pending is not None else float(value)
        height = max(imgui.get_content_region_avail().y - em(1.6) * (1 + extra_rows), em(3))
        changed, shown = imgui.v_slider_float(
            f"##{key}", imgui.ImVec2(em(2.2), height), shown, float(lo), float(hi), "%.2f"
        )
        if changed:
            self._slider_pending[key] = shown
        if imgui.is_item_deactivated_after_edit():
            self._slider_pending.pop(key, None)
            apply(shown)
        imgui.same_line(0, em(0.4))
        imgui.text_disabled(f"{hi:.2f}\n\n\n{lo:.2f}")

    # ------------------------------------------------------------------
    # top panel: the candidates
    # ------------------------------------------------------------------

    def draw_candidate_panel(self) -> None:
        """The Candidates panel: template, focused candidate, second pass, PCA."""
        self._mark_hovered()
        session = self._ready()
        if session is None:
            imgui.text_disabled(self.status)
            return
        if not session.n:
            imgui.text_disabled("No candidate events found; lower the threshold.")
            return
        avail = imgui.get_content_region_avail()
        gap = em(0.5)
        width = max((avail.x - 3 * gap) / 4, em(8))
        panels = (
            ("B. Current template", self._draw_template),
            (self._candidate_title(session), self._draw_candidate),
            ("D. Second pass preview", self._draw_second_pass),
            ("E. Candidate PCA (400 ms)", self._draw_pca),
        )
        for i, (title, draw) in enumerate(panels):
            if i:
                imgui.same_line(0, gap)
            with imgui_ctx.begin_child(f"##curation_panel{i}", imgui.ImVec2(width, 0)):
                imgui.text_colored(theme.to_vec4(theme.THEME.accent), title)
                draw(session)

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

    def _draw_template(self, session) -> None:
        source = session.template_source
        fit = self._fit_for("template", (session.mode, session.recording_id, len(source)))
        with line_plot("##curation_b", "aligned time (ms)", "baseline-subtracted z", fit=fit, legend=False) as ok:
            if not ok:
                return
            if session.template is None:
                implot.plot_text("No Yes events yet", 0.0, 0.0)
                return
            t_ms = session.template_time_ms
            for i in source:
                line(f"##snippet{i}", session.short_snippet(i), x=t_ms, color=SNIPPET_COLOR, alpha=0.18)
            line("template", session.template, x=t_ms, color=TEMPLATE_COLOR, weight=3.0)

    def _draw_candidate(self, session) -> None:
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
        with line_plot("##curation_c", "time from event (ms)", "z", legend=True) as ok:
            if not ok:
                return
            line("candidate", y, x=t_ms, color=SNIPPET_COLOR, weight=1.2)
            if session.template is not None:
                line("template", session.template, x=session.template_time_ms, color=TEMPLATE_COLOR, weight=2.5)
            vlines("##zero", [0.0], FOCUS_COLOR, 1.0, legend=False)

    def _draw_second_pass(self, session) -> None:
        i = session.current
        peak = session.aligned_index(i)
        fs = session.fs
        radius = max(1, int(round(0.5 * fs)))
        lo = max(0, peak - radius)
        hi = min(len(session.denoised), peak + radius + 1)
        x_ms = (np.arange(lo, hi) - peak) / fs * 1000.0
        fit = self._fit_for("second", (session.mode, session.recording_id, i, session.counts()))
        with line_plot("##curation_d", "time from event (ms)", "z", fit=fit, legend=True) as ok:
            if not ok:
                return
            line("original denoised", session.denoised[lo:hi], x=x_ms, color=ORIGINAL_COLOR, weight=1.1)
            line("rejected removed", session.second_pass[lo:hi], x=x_ms, color=THRESHOLD_COLOR, weight=1.4)
            vlines("##zero", [0.0], FOCUS_COLOR, 1.0, legend=False)

    def _draw_pca(self, session) -> None:
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
            focused=int(focused[0]) if focused.size else None,
            tooltip=lambda k: (
                f"event {visible[k] + 1}: {session.label(visible[k])}\n"
                f"PC1 {scores[visible[k], 0]:.2f}  PC2 {scores[visible[k], 1]:.2f}"
            ),
        )
        if picked is not None:
            session.select(int(visible[picked]))

    # ------------------------------------------------------------------
    # right tab
    # ------------------------------------------------------------------

    def draw_tab(self) -> None:
        """The Curation tab: source, recordings, mode, filter, decisions, navigation."""
        self.strip.report_right_tab("curation")
        self._mark_hovered()
        self._draw_mode_row()
        self._draw_source()
        self._draw_recordings()
        self._draw_controls()

    def draw_embedded(self) -> None:
        """The controls without the source picker, for a host that hands
        traces over itself (the line-scan viewer's ROI panel)."""
        self._mark_hovered()
        self._draw_mode_row()
        self._draw_recordings()
        self._draw_controls()

    def _draw_controls(self) -> None:
        imgui.text_wrapped(self.status)
        session = self._ready()
        if session is None:
            return
        self._draw_filter(session)
        self._draw_decision(session)
        self._draw_navigation(session)
        self._draw_event(session)
        self._draw_files(session)

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
        if not self.catalog:
            return
        section("Recordings")
        n_loaded = len(self.loaded())
        imgui.text_disabled(
            f"{n_loaded}/{len(self.catalog)} loaded"
            + (f", {len(self._busy)} loading" if self._busy else "")
        )
        imgui.same_line(0, em(0.6))
        if imgui.small_button("load all"):
            self.load_all(None)
        set_tooltip("load every processed recording of every experiment", show_mark=False)
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.scroll_y
            | imgui.TableFlags_.sizing_fixed_fit
        )
        rows = min(len(self.catalog), 12)
        height = imgui.get_text_line_height_with_spacing() * (rows + 1.5)
        if not imgui.begin_table("##curation_recordings", 4, flags, imgui.ImVec2(0, height)):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        for name in ("experiment", "recording", "cand", "yes / no"):
            imgui.table_setup_column(name)
        imgui.table_headers_row()
        for rec in self.catalog:
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
            else:
                imgui.text_disabled("-" if rec.pre_denoised else "raw")
            imgui.table_next_column()
            if ready:
                yes, no, _un = session.counts()
                imgui.text(f"{yes} / {no}")
        imgui.end_table()

    def _draw_filter(self, session) -> None:
        section("View")
        current = session.view_filter
        for i, name in enumerate(FILTERS):
            if i:
                imgui.same_line(0, em(0.5))
            if imgui.radio_button(name, current == name):
                session.set_view_filter(name)

    def _draw_decision(self, session) -> None:
        section("Decision")
        if not session.n:
            imgui.begin_disabled()
        width = imgui.ImVec2(em(4.5), em(2))
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

    def _draw_navigation(self, session) -> None:
        section("Navigation")
        if not session.n:
            imgui.begin_disabled()
        if imgui.button("< Event"):
            session.step(-1)
        imgui.same_line(0, em(0.5))
        if imgui.button("Event >"):
            session.step(1)
        imgui.set_next_item_width(-em(3.5))
        changed, value = imgui.slider_int("event", session.current + 1, 1, max(1, session.n))
        if changed:
            session.select(value - 1)
        if not session.n:
            imgui.end_disabled()

    def _draw_event(self, session) -> None:
        section("Event")
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
            imgui.text_disabled(
                f"auto call: {info['auto_call'] or 'none'} (cosine {initial_text} at "
                f"{session.auto_template_threshold:.2f}, auto-pass {auto_pass}, "
                f"waveform reject {'on' if session.waveform_rejection else 'off'})"
            )

    def _draw_files(self, session) -> None:
        section("Files")
        yes, no, unlabeled = session.counts()
        imgui.text_disabled(f"yes {yes} · no {no} · unlabeled {unlabeled}")
        imgui.text_disabled(f"template source events: {len(session.template_source)}")
        imgui.text_disabled(f"curation file: {session.label_path}")
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
    """Turn the curation widget on for a ``PreviewDataWidget``. Returns None
    (logged) when it cannot be built."""
    widget = getattr(parent, "event_curation", None)
    if widget is not None:
        widget.focus_tab = widget.focus_tab or focus
        return widget
    try:
        widget = EventCurationWidget(parent, strip=getattr(parent, "top_strip", None))
    except Exception:
        parent.logger.warning("event curation widget unavailable", exc_info=True)
        parent.event_curation = None
        return None
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
