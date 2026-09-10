"""Event curation of voltage traces with vnoiser, inside the viewer.

The curation notebook's dashboard as viewer panels. Two panels on the top
strip: ``Curation`` is the full trace with the candidate markers and the
threshold / auto-pass lines (the notebook's panel A with cards A1 and A2);
``Candidates`` holds the template, the focused candidate, the second-pass
preview and the candidate PCA (B to E). The Curation tab on the right bar
picks the data, the mode and the recording, and carries the decision and
navigation controls. Every rule comes from ``vnoiser.curation`` through
:class:`mbo_utilities.vnoiser.CurationSession`; each mode keeps its own
session and its own JSON file, as the notebook's sections do.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui, imgui_ctx, implot, portable_file_dialogs as pfd

from mbo_utilities.gui import _theme as theme
from mbo_utilities.gui._files import PathPrompt, draw_path_prompt
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui._theme import card, em, section
from mbo_utilities.gui._top_strip import TopPanel, TopStrip
from mbo_utilities.gui.imgui.lines import drag_hline, line, line_plot, vlines
from mbo_utilities.gui.imgui.scatter import ScatterPlot
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.install import VNOISER_HINT
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.vnoiser import MODES, CurationSession, pf_dir_for_mesc

__all__ = [
    "KEYBINDS",
    "EventCurationWidget",
    "attach_curation_widget",
    "detach_curation_widget",
    "help_markdown",
]

PANEL_HEIGHT = 280
CARD_WIDTH_EM = 8.5
FILTERS = ("all", "yes", "no", "unlabeled")

TRACE_COLOR = (0.85, 0.85, 0.85, 1.0)
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
    ("click", "focus a candidate on the trace or the PCA plot"),
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
        "vnoiser's curation notebook inside the viewer. Pick a data path "
        "(a `Data` folder, an animal or experiment folder, a `PF` folder, or "
        "a raw `.mat` recording), then the animal, experiment and scan / "
        "domain, and Load.\n\n"
        "### Modes\n\n"
        "- **fast**: thresholds the denoised trace; the template is seeded "
        "from the top 25% highest-amplitude candidates.\n"
        "- **slow**: thresholds a zero-phase low-pass view of the denoised "
        "trace (blue); the template is seeded the same way.\n"
        "- **manual**: no seed template; only Yes events shape it.\n\n"
        "### Panels\n\n"
        "- **Curation** (top): the trace with candidate markers. Drag the red "
        "line to change the candidate threshold; drag the teal line (or the "
        "A2 slider) to set the auto-pass amplitude.\n"
        "- **Candidates** (top): the current template, the focused candidate "
        "against it, the second-pass preview with rejected events removed, "
        "and the candidate PCA over 400 ms windows. Click a point to focus it.\n"
        "- **Curation** tab (right): mode, source, view filter, Yes / No / "
        "Clear, and navigation.\n\n"
        "Each mode saves to its own `PF/.curation/<mode>_template_curation.json`; "
        "reopening the same scan / domain restores it."
    )


def _mode_title(mode: str, cutoff: float) -> str:
    return _MODE_TITLES[mode].format(cutoff=cutoff)


class EventCurationWidget:
    """The curation panels and tab on a ``PreviewDataWidget``-like parent.

    Parameters
    ----------
    parent
        Anything with ``image_widget`` (a figure) and a ``logger``.
    strip : TopStrip, optional
        The figure's shared top strip; a private one is built without.
    """

    def __init__(self, parent: Any, strip: TopStrip | None = None, data_path: str | None = None):
        self.parent = parent
        self.logger = getattr(parent, "logger", None) or logging.getLogger(__name__)
        self.figure = parent.image_widget.figure
        self._own_strip = strip is None
        self.strip = TopStrip(self.figure) if self._own_strip else strip
        self.strip.add_hook(self._frame)
        self.strip.register(
            TopPanel("curation", "Curation", self.draw_timeline_panel, PANEL_HEIGHT, "curation", 12)
        )
        self.strip.register(
            TopPanel("candidates", "Candidates", self.draw_candidate_panel, PANEL_HEIGHT, "curation", 13)
        )

        self.mode = "fast"
        self.slow_cutoff_hz = 40.0
        self.sessions: dict[str, CurationSession] = {}
        if data_path is None:
            last = get_last_dir("vnoiser")
            data_path = str(last) if last else ""
        self.data_path = str(data_path)
        # a trace handed over in memory (a line-scan ROI), replayed on a
        # mode switch the way a picked recording is
        self._trace_source: dict | None = None
        # called with the focused candidate's time (s) whenever it changes
        self.on_focus = None
        self._last_focus = None
        self.prompt = PathPrompt(
            "Curation data",
            path=self.data_path,
            action="scan",
            hint="vnoiser Data folder, an experiment or PF folder, or a .mat recording",
        )
        self._folder_dialog = None
        self.animal = ""
        self.experiment = ""
        self.recording = ""
        self.status = "Set a data path, then choose a recording and Load."

        self._loading = False
        self._load_thread: threading.Thread | None = None
        self._load_results: queue.Queue = queue.Queue()

        self.timeline_points = ScatterPlot("##curation_timeline_pts", marker_size=7.0)
        self.pca = ScatterPlot("##curation_pca", marker_size=7.0)
        self.autofit = True
        self._fit_timeline = False
        self._timeline_key = None
        self._panel_keys: dict[str, tuple] = {}
        self._threshold_drag: float | None = None
        self._auto_pass_drag: float | None = None
        self._slider_pending: dict[str, float] = {}

        self._hovered = False
        self.focus_tab = False
        self._closed = False

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
        self.strip.unregister("candidates")
        if self._own_strip:
            self.strip.close()

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    @property
    def session(self) -> CurationSession | None:
        """The current mode's session, built on first use."""
        if not self.data_path:
            return None
        session = self.sessions.get(self.mode)
        if session is None:
            try:
                session = CurationSession(
                    self.data_path, mode=self.mode, slow_cutoff_hz=self.slow_cutoff_hz
                )
            except Exception as error:
                self.logger.warning("vnoiser cannot open %s", self.data_path, exc_info=True)
                self.status = f"cannot open {self.data_path}: {error}"
                self.data_path = ""
                return None
            self.sessions[self.mode] = session
            if self.experiment:
                session.select_experiment(self.experiment)
        return session

    def _ready(self) -> CurationSession | None:
        """The session when it can be read: not None, not mid-load, loaded."""
        if self._loading:
            return None
        session = self.session
        return session if session is not None and session.loaded else None

    def scan(self, path) -> None:
        """Point every mode at a new data path."""
        self.sessions.clear()
        self.animal = self.experiment = self.recording = ""
        self._trace_source = None
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
        session = self.session
        if session is None:
            return
        set_last_dir("vnoiser", self.data_path)
        self.prompt.path = self.data_path
        self.status = session.status + note
        if not session.has_dataset:
            self.status = (
                f"no vnoiser data at {self.data_path}: expected a Data folder "
                "(stan*/…_expt*/PF/denoised_trace_scans.pkl), an animal, experiment or "
                "PF folder, or a .mat recording."
            )
            return
        animals = session.animals
        if len(animals) == 1:
            self.animal = animals[0][1]

    def set_mode(self, mode: str) -> None:
        if mode not in MODES or mode == self.mode:
            return
        self.mode = mode
        self._fit_timeline = True
        self._reload()

    def set_slow_cutoff(self, cutoff_hz: float) -> None:
        cutoff_hz = float(cutoff_hz)
        if cutoff_hz <= 0 or cutoff_hz == self.slow_cutoff_hz:
            return
        self.slow_cutoff_hz = cutoff_hz
        self.sessions.pop("slow", None)
        if self.mode == "slow":
            self._reload()

    def _reload(self) -> None:
        """Bring the current mode's session onto what the others show."""
        session = self.session
        if session is None:
            return
        if self._trace_source is not None:
            if session.recording_id != self._trace_source["recording_id"]:
                self.load_trace(**self._trace_source)
        elif self.recording and session.recording_id != self.recording:
            self.load(self.recording)

    def load_trace(self, trace, fs_hz, *, recording_id, label, source_path, curation_dir=None) -> None:
        """Curate a trace held in memory, on a worker thread: vnoiser's
        denoiser runs on it unless a cache exists. See
        :meth:`CurationSession.load_trace`."""
        source_path = str(Path(source_path))
        if source_path != self.data_path:
            self.sessions.clear()
            self.animal = self.experiment = self.recording = ""
            self.data_path = source_path
        session = self.session
        if session is None or self._loading:
            return
        self._trace_source = {
            "trace": trace,
            "fs_hz": float(fs_hz),
            "recording_id": str(recording_id),
            "label": str(label),
            "source_path": source_path,
            "curation_dir": curation_dir,
        }
        source = self._trace_source
        job = get_process_manager().start_job("vnoiser", f"denoise + curate: {label}")
        self._loading = True
        self.status = f"denoising {label}... (cached after the first run)"

        def run():
            try:
                message = session.load_trace(**source)
            except Exception as error:
                job.fail(f"{type(error).__name__}: {error}")
                self._load_results.put((session, None, f"load failed: {error}"))
                return
            job.done(message)
            self._load_results.put((session, message, None))

        self._load_thread = threading.Thread(target=run, name="vnoiser-denoise", daemon=True)
        self._load_thread.start()

    def select_experiment(self, experiment: str) -> None:
        self.experiment = str(experiment)
        self.recording = ""
        for session in self.sessions.values():
            self.status = session.select_experiment(self.experiment)

    def load(self, recording_id: str) -> None:
        """Load one recording for the current mode on a worker thread."""
        session = self.session
        if session is None or self._loading or not recording_id:
            return
        self.recording = recording_id
        label = dict(session.recordings).get(recording_id, recording_id)
        job = get_process_manager().start_job("vnoiser", f"curation: {label}")
        self._loading = True
        self.status = f"loading {label}..."

        def run():
            try:
                message = session.load(recording_id)
            except Exception as error:
                job.fail(f"{type(error).__name__}: {error}")
                self._load_results.put((session, None, f"load failed: {error}"))
                return
            job.done(message)
            self._load_results.put((session, message, None))

        self._load_thread = threading.Thread(target=run, name="vnoiser-load", daemon=True)
        self._load_thread.start()

    def wait(self, timeout: float | None = None) -> None:
        """Block until a pending load finishes and apply it (tests)."""
        if self._load_thread is not None:
            self._load_thread.join(timeout)
        self._drain_loads()

    def _drain_loads(self) -> None:
        while True:
            try:
                _session, message, error = self._load_results.get_nowait()
            except queue.Empty:
                return
            self._loading = False
            self._load_thread = None
            self.status = error or message
            if error:
                self.logger.warning(self.status)
            self._fit_timeline = True
            self.pca.refit()
            self._panel_keys.clear()

    # ------------------------------------------------------------------
    # per frame
    # ------------------------------------------------------------------

    def _frame(self) -> None:
        self._drain_loads()
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
                session = self.session
                if session is not None and session.has_dataset:
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
    # top panel: the trace
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
        """The Curation tab: source, mode, filter, decisions, navigation."""
        self.strip.report_right_tab("curation")
        self._mark_hovered()
        self._draw_mode_row()
        self._draw_source()
        session = self.session
        if session is None:
            imgui.text_disabled(self.status)
            return
        self._draw_selection(session)
        self._draw_controls(session)

    def draw_embedded(self) -> None:
        """The controls without the source picker, for a host that hands
        traces over itself (the line-scan viewer's ROI panel)."""
        self._mark_hovered()
        self._draw_mode_row()
        session = self.session if self.data_path else None
        if session is None:
            imgui.text_wrapped(self.status)
            return
        self._draw_controls(session)

    def _draw_controls(self, session: CurationSession) -> None:
        imgui.text_wrapped(self.status)
        if self._loading or not session.loaded:
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

    def _draw_selection(self, session: CurationSession) -> None:
        if session.hierarchical:
            animals = session.animals
            self._combo("animal", animals, self.animal, self._pick_animal)
            experiments = session.experiments(self.animal) if self.animal else []
            self._combo("experiment", experiments, self.experiment, self.select_experiment)
        recordings = session.recordings
        self._combo("recording", recordings, self.recording, self._pick_recording)
        can_load = bool(self.recording) and not self._loading
        if not can_load:
            imgui.begin_disabled()
        if imgui.button("Load", imgui.ImVec2(em(6), 0)):
            self.load(self.recording)
        if not can_load:
            imgui.end_disabled()
        if self._loading:
            imgui.same_line(0, em(0.5))
            imgui.text_disabled("loading...")

    def _pick_animal(self, value: str) -> None:
        self.animal = value
        self.experiment = ""
        self.recording = ""

    def _pick_recording(self, value: str) -> None:
        self.recording = value

    def _combo(self, label, options, current, on_pick) -> None:
        """A combo over ``(label, value)`` options."""
        values = [value for _label, value in options]
        labels = [text for text, _value in options]
        idx = values.index(current) if current in values else -1
        if not options:
            labels, idx = [f"no {label}s found"], 0
        imgui.set_next_item_width(-em(5.5))
        changed, idx = imgui.combo(label, idx, labels)
        if changed and options and 0 <= idx < len(values) and values[idx] != current:
            on_pick(values[idx])

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
