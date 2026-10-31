"""The Traces panel: masknmf's TracePlot over the viewer's frames, the motion and behavior rows, the kind and dF/F settings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from masknmf.visualization.imgui import (
    GROUP_COLORS,
    TracePlot,
    tooltip,
)
from masknmf.visualization.rois import MARKED_COLOR

from mbo_utilities.analysis.dff import NOISE_ESTIMATES
from mbo_utilities.annotation import (
    DISPLAY_KINDS,
    DffSettings,
    RoiTrace,
    available_kinds,
    display_trace,
    displayed_kind,
    trace_profile,
    y_label,
)
from mbo_utilities.annotation.display import DFF_METHODS
from mbo_utilities.gui._theme import em
from mbo_utilities.gui.imgui.behavior import EPOCH_COLORS, BehaviorPlot
from mbo_utilities.gui.imgui.motion import MOTION_COLORS
from mbo_utilities.gui.roi._shared import (
    _TRACE_MODES,
    BEHAVIOR_PLOT_HEIGHT,
    DEFLECTION_MODES,
    MOTION_PANEL_HEIGHT,
    NOISE_LABELS,
    PANEL_HEIGHT,
)
from mbo_utilities.gui.roi_runs import (
    component_color,
)
from mbo_utilities.lazy_array import base_array
from mbo_utilities.results import (
    results_pipeline,
)


class TracesPanelMixin:
    """masknmf's TracePlot over the viewer's frames, with the motion and behavior rows."""

    def _trace_label(self, trace: RoiTrace) -> str:
        """Legend text for one row: the engine, then where it was read when
        that is not where the ROI was drawn, then a frame window.
        """
        if not trace.stands_for_roi:
            return trace.name
        parts = [trace.engine]
        index = self.store.uid_index(trace.uid)
        if index is not None:
            if trace.z != self.store.roi_z(index):
                parts.append(self._axis_tag("z", trace.z))
            if trace.c != self.store.roi_c(index):
                parts.append(self._axis_tag("c", trace.c))
        if trace.frames is not None:
            start, stop, step = trace.frames
            parts.append(f"t{start + 1}-{stop}" + (f"-{step}" if step > 1 else ""))
        elif trace.extra.get("tp_indices"):
            parts.append(f"{len(trace.extra['tp_indices'])} frames")
        return " ".join(parts)

    def _selection_trace_keys(self) -> list[tuple]:
        """Trace-table keys of the selection; empty when it has none."""
        if self.selected_derived is not None:
            if self.selected_derived[0] >= len(self.derived):
                return []
            return self._member_keys(*self.selected_derived)
        # mid store event the selection can still name an ROI the store just dropped
        if not 0 <= self.selected < self.n_rois:
            return []
        return [t.key for t in self.traces.for_roi(self.store.rois[self.selected].uid)]

    def _sync_trace_sel(self):
        """Point the trace table's picks at the selection, so it highlights the ROI the image shows."""
        self.trace_sel = set(self._selection_trace_keys())
        self.trace_picked = False
        self._plot_lines_key = None

    def _binning_tag(self, key) -> str:
        """`` x10`` when a trace was taken at another frame averaging than the data now shows (another time base)."""
        trace = self.traces.get(key)
        taken = int(trace.frame_average if trace is not None else 1) or 1
        now = int(getattr(self.host, "frame_average", 1) or 1)
        return f" x{taken}" if taken != now and taken > 1 else ""

    def fs(self) -> float | None:
        """The recording's raw sampling rate in Hz, or None; read once under any frame averaging, since every ``TimeAxis`` bins itself."""
        if not self._fs_read:
            self._fs_read = True
            rate = (
                getattr(base_array(self.iw.data[0]), "fs", None)
                if self.iw.data
                else None
            )
            self._fs_value = float(rate) if rate else None
        return self._fs_value

    @property
    def kind(self) -> str | None:
        """The kind the plot shows of each row; None lets every row's pipeline pick."""
        return self._kind

    @kind.setter
    def kind(self, kind: str | None) -> None:
        if kind != self._kind:
            self._kind = kind
            self._redisplay()

    @property
    def flip(self) -> bool:
        """The plot turns every row over (``display_trace``'s ``flip``)."""
        return self._flip

    @flip.setter
    def flip(self, on: bool) -> None:
        if bool(on) != self._flip:
            self._flip = bool(on)
            self._redisplay()

    def kind_options(self, rows) -> tuple[str, ...]:
        """Every kind any of ``rows`` can show, in ``DISPLAY_KINDS`` order."""
        offered = {kind for trace in rows for kind in available_kinds(trace)}
        return tuple(kind for kind in DISPLAY_KINDS if kind in offered)

    def plot_y_label(self, rows) -> str:
        """The y axis label of what the rows show: one when they agree, else joined."""
        subtract, invert = self.deflection()
        labels = list(
            dict.fromkeys(
                y_label(t, self.kind, subtract, invert, self.flip) for t in rows
            )
        )
        return " / ".join(label for label in labels if label) or DISPLAY_KINDS["dff"]

    def _redisplay(self) -> None:
        """The rows read differently now: drop the cached arrays, redraw the lines."""
        self._trace_display.clear()
        self._trace_stats.clear()
        self._trace_window_cache.clear()
        self._plot_lines_key = None
        self._vnoiser_lines_key = None

    def _window_spec(self) -> tuple[str, int]:
        """``(projection, size)`` of the viewer's window function.

        The preview trace gets the same window the image does, so what the
        plot shows is what the frame on screen shows.
        """
        return (
            str(getattr(self.host, "proj", "mean")),
            max(1, int(getattr(self.host, "window_size", 1))),
        )

    def _windowed(self, y):
        """``y`` under the viewer's rolling window; the raw array at size 1."""
        proj, size = self._window_spec()
        if y is None or size <= 1 or y.size < size:
            return y
        cached = self._trace_window_cache.get((id(y), proj, size))
        if cached is not None:
            return cached
        pad = (size - 1) // 2, size // 2
        padded = np.pad(y, pad, mode="edge")
        view = np.lib.stride_tricks.sliding_window_view(padded, size)
        func = {"max": np.max, "std": np.std}.get(proj, np.mean)
        out = np.ascontiguousarray(func(view, axis=-1), np.float32)
        if len(self._trace_window_cache) > 256:
            self._trace_window_cache.clear()
        self._trace_window_cache[(id(y), proj, size)] = out
        return out

    def viewer_deflection(self) -> tuple[bool, bool]:
        """The viewer's ``(Mean Subtraction, Invert Deflection)``."""
        return (
            bool(getattr(self.host, "mean_subtraction", False)),
            bool(getattr(self.host, "invert_deflection", False)),
        )

    def deflection(self) -> tuple[bool, bool]:
        """``(mean subtraction, invert)`` applied to measured rows: the panel's ``trace_subtract`` / ``trace_invert`` when set, else the viewer's."""
        return tuple(
            seen if mode == "viewer" else mode == "on"
            for mode, seen in zip(
                (self.trace_subtract, self.trace_invert), self.viewer_deflection()
            )
        )

    def _vnoiser_rows(self) -> tuple[list[RoiTrace], list[RoiTrace], Path | None]:
        """``(holding, every, run)``: the newest vnoiser run loaded here of the
        recording on screen (its results file), every domain row it has, and
        the ones holding the ROI on screen (the slice's z, a MESc unit's ROI
        axis); ``([], [], None)`` without a run.
        """
        from mbo_utilities.results import results_stamp

        src = base_array(self.iw.data[0]) if self.iw.data else None
        tail = str(getattr(src, "unit_key", "") or "").rsplit("/", 1)[-1]
        runs: dict = {}
        for t in self.traces:
            if t.engine != "vnoiser" or t.stands_for_roi or "unit" not in t.extra:
                continue
            made = str(t.extra.get("source_unit") or "").rsplit("/", 1)[-1]
            if tail and made and made != tail:
                continue
            runs.setdefault(t.path, []).append(t)
        if not runs:
            return [], [], None
        stamps = {p: results_stamp(p) if p is not None else None for p in runs}
        run = max(runs, key=lambda p: (stamps[p] is not None, stamps[p] or 0))
        line = int(self.slice.z or 0)
        holding = [
            t
            for t in runs[run]
            if line in (t.extra.get("lines") or [t.extra.get("line", t.z)])
        ]
        return holding, runs[run], run

    def _display(self, key) -> np.ndarray | None:
        """The cached display array of one trace key in the panel's kind, dF/F settings and deflection; None when the row is gone or empty."""
        deflection = self.deflection()
        if deflection != self._trace_deflection:
            self._trace_deflection = deflection
            self._redisplay()
        got = self._trace_display.get(key)
        if got is None:
            trace = self.traces.get(key)
            if trace is None:
                return None
            y = display_trace(trace, self.kind, self.dff, *deflection, self.flip)
            if y is None:
                return None
            got = np.ascontiguousarray(y, np.float32)
            self._trace_display[key] = got
        return got

    def _set_by_name(self, name: str):
        for si, s in enumerate(self.derived):
            if s.name == name:
                return si, s
        return None

    def _key_to_pair(self, key) -> tuple[int, int] | None:
        """``(si, k)`` behind one trace key, or None when the ROI is gone or the row stands for none."""
        trace = self.traces.get(key)
        if trace is None:
            return None
        if trace.stands_for_roi:
            index = self.store.uid_index(trace.uid)
            return (-1, index) if index is not None else None
        hit = self._set_by_name(trace.source)
        return (hit[0], int(trace.member)) if hit is not None else None

    def _trace_color(self, key) -> tuple[float, float, float] | None:
        """The mask color of the ROI behind one trace key, shared with the overlay."""
        pair = self._key_to_pair(key)
        if pair is None:
            return None
        si, k = pair
        if si < 0:
            return tuple(v / 255.0 for v in self.store.roi_rgb(k))
        return component_color(self.derived[si], k)

    def _trace_shown(self, key) -> tuple[float, str]:
        """``(sort value, display text)`` for a key's roi column."""
        trace = self.traces.get(key)
        if trace is None:
            return float(1 << 30), "?"
        if trace.stands_for_roi:
            index = self.store.uid_index(trace.uid)
            if index is not None:
                return float(index), f"{index}"
            return float((1 << 30) + trace.uid), f"uid {trace.uid}"
        if self._set_by_name(trace.source) is None:
            # rows that stand for no ROI sort after the drawn ones, in table order
            return float((1 << 30) + self.traces.keys.index(key)), trace.name
        index = self._promoted.get((trace.source, trace.member))
        if index is not None:
            return float(index), f"{index}"
        return float((1 << 30) + trace.member), f"{trace.member}"

    def _plot_lines(self):
        """``(header, [(label, key, rgb), ...])`` the plot shows, None for nothing.

        Rows picked in the trace table when they are not the selection's
        own; else, with a group of two or more or any pixel average, one
        line per member in its group color (masknmf's group); else every
        trace of the selected ROI, in its mask color when it has one.
        """
        if not self.show_traces:
            return None
        if self.trace_picked and self.trace_sel:
            keys = [key for key in self.traces.keys if key in self.trace_sel]
            lines = []
            for i, key in enumerate(keys):
                trace = self.traces.get(key)
                _v, shown = self._trace_shown(key)
                label = (
                    f"{shown} · {self._trace_label(trace)}"
                    if trace.stands_for_roi
                    else f"{trace.source} · {shown}"
                )
                rgb = (
                    GROUP_COLORS[i % len(GROUP_COLORS)]
                    if len(keys) > 1
                    else self._trace_color(key)
                )
                lines.append((label, key, rgb))
            return (f"{len(lines)} selected", lines) if lines else None
        members = [*self.buffer, *self.pixel_group]
        if len(members) > 1 or self.pixel_group:
            colors = self._group_colors()
            lines = []
            for member in members:
                key = self._member_line(member)
                if key is None or key not in self.traces:
                    continue
                rgb = colors.get(member, MARKED_COLOR)
                if len(member) == 3:
                    label = f"px {member[1]},{member[2]}"
                elif member[0] < 0:
                    label = f"ROI {member[1]}"
                else:
                    label = f"{self.derived[member[0]].name} {member[1]}"
                lines.append((label, key, rgb))
            return (f"{len(members)} grouped", lines) if lines else None
        keys = [key for key in self._selection_trace_keys() if key in self.traces]
        if not keys:
            return None
        pair = self._selection_pair()
        row = self._row_index.get(pair)
        lines = []
        for i, key in enumerate(keys):
            if len(keys) == 1:
                rgb = self._row_rgb(row) if row is not None else self._trace_color(key)
            else:
                rgb = GROUP_COLORS[i % len(GROUP_COLORS)]
            lines.append((self._trace_label(self.traces.get(key)), key, rgb))
        if self.selected >= 0:
            header = f"ROI {self.selected}"
        else:
            si, k = self.selected_derived
            header = f"{self.derived[si].name} row {k}"
        return header, lines

    def _spike_target(self, rows) -> tuple | None:
        """``(path, unit, roi, c)`` of the run behind the one ROI plotted,
        when that ROI came from a run's results; None otherwise.

        An algo row is ROI ``k`` of the unit its set was read from (a run
        folder or plane dir, or ``<file>.zarr/<unit>``); a results row of the
        trace table carries its file, unit and ROI.
        """
        if self.selected < 0 and self.selected_derived is not None and not self.buffer:
            si, k = self.selected_derived
            path = Path(self.derived[si].path)
            if results_pipeline(path.parent) is not None:
                return (path.parent, path.name, str(k), None)
            return (path, None, str(k), None) if path.is_dir() else None
        if len(rows) == 1 and rows[0] is not None and "roi" in (rows[0].extra or {}):
            row = rows[0]
            return (row.path, row.extra["unit"], row.extra["roi"], row.c)
        return None

    def _open_spike_average(self, path, unit, roi, c) -> None:
        from mbo_utilities.gui.launch import LaunchedWindow
        from mbo_utilities.gui.spike_average_viewer import launch_spike_average

        pid, log_name = launch_spike_average(path, unit, roi, c)
        self._windows.append(LaunchedWindow(pid, log_name, "Spike average"))
        self.status = f"spike average of {roi} in its own window (PID {pid})"

    def draw_traces(self):
        """The Traces panel: a row of controls, the behavior raster, then
        masknmf's stacked trace panels (the motion correction over the
        traces) on the viewer's frames, the playhead bound to the viewer's t.
        """
        self._note_hover()
        target = self._plot_lines()
        motion = self.motion if self.motion else None
        behavior = self.behavior if self.behavior else None
        imgui.begin_disabled(motion is None)
        changed, on = imgui.checkbox("MC", self.show_motion and motion is not None)
        if changed:
            self.show_motion = on
        imgui.end_disabled()
        tooltip(
            f"{motion.y_label}: the motion correction the whole recording went through, as a panel over the traces"
            if motion is not None
            else "This recording carries no motion correction."
        )
        imgui.same_line(0, em(0.8))
        imgui.begin_disabled(behavior is None)
        changed, on = imgui.checkbox(
            "Behavior", self.show_behavior and behavior is not None
        )
        if changed:
            self.show_behavior = on
        imgui.end_disabled()
        tooltip(
            f"{behavior.source}: what the animal did during the recording "
            f"({', '.join([*behavior.signals, *behavior.events, *behavior.epochs])}), over the traces, its epochs "
            "shaded behind them"
            if behavior is not None
            else "No behavior log was found for this recording: a file named after its subject and day, beside it or "
            "in a behavior folder."
        )
        rows = (
            []
            if target is None
            else [self.traces.get(key) for _label, key, _rgb in target[1]]
        )
        kinds = self.kind_options(rows)
        if kinds:
            imgui.same_line(0, em(0.8))
            shown = [displayed_kind(t, self.kind) for t in rows]
            current = self.kind if self.kind in kinds else shown[0]
            imgui.set_next_item_width(em(9))
            changed, sel = imgui.combo(
                "##trace_kind", kinds.index(current), list(kinds)
            )
            tooltip(
                "What each row shows, one line per row: dff (its pipeline's dF/F, or one computed here), raw (F "
                "alone), neuropil (Fneu alone), raw - neuropil (F minus the pipeline's share of Fneu), spikes, "
                "denoised, z-score. A row without that kind shows its pipeline's default."
            )
            if changed:
                self.kind = kinds[sel]
            imgui.same_line(0, em(0.4))
            changed, on = imgui.checkbox("flip##trace_flip", self.flip)
            if changed:
                self.flip = on
            tooltip(
                "Turn every row over: raw and neuropil about their mean (undoing the viewer's Invert Deflection "
                "when that is on), dF/F, z-score, denoised and spikes negated"
            )
            imgui.same_line(0, em(0.4))
            if imgui.small_button("Trace##trace_settings"):
                imgui.open_popup("##trace_settings")
            subtract, invert = self.deflection()
            tooltip(
                "How the rows' measured traces are shown: mean subtraction "
                f"({'on' if subtract else 'off'}) and inversion ({'on' if invert else 'off'}), each the viewer's or "
                "set here, and the baseline, noise and slow drift of a dF/F or z-score computed here (from raw - "
                "neuropil when the row has a neuropil, from the raw trace otherwise). A pipeline's own dF/F, "
                "z-score and denoised trace are shown as it wrote them."
            )
            self._draw_trace_settings(rows)
        vholding, vevery, vrun = self._vnoiser_rows()
        show_vnoiser = vrun is not None and self.show_vnoiser
        if vrun is not None or self._shows_trace_unit():
            imgui.same_line(0, em(0.8))
            imgui.begin_disabled(vrun is None)
            changed, on = imgui.checkbox("vnoiser", show_vnoiser)
            if changed:
                self.show_vnoiser = on
            imgui.end_disabled()
            unit = self.unit or "this recording"
            if vrun is None:
                tooltip(
                    f"No vnoiser run of {unit} is loaded. Run vnoiser opens its tab on this scan; when the run "
                    "finishes, its domains show here in a panel of their own, under the recording's trace."
                )
            elif vholding:
                tooltip(
                    f"vnoiser's {', '.join(t.label for t in vholding)} (the domain holding ROI {self.slice.z}) from "
                    f"{Path(vrun).name}, in a panel of its own under the recording's trace"
                )
            else:
                tooltip(
                    f"No domain of {Path(vrun).name} holds ROI {self.slice.z}; its panel stays empty here. "
                    f"Domains: {', '.join(t.label for t in vevery)}"
                )
            if show_vnoiser:
                vkinds = self.kind_options(vevery)
                current = (
                    self._vnoiser_kind
                    if self._vnoiser_kind in vkinds
                    else displayed_kind(vevery[0])
                )
                imgui.same_line(0, em(0.4))
                imgui.set_next_item_width(em(7))
                changed, sel = imgui.combo(
                    "##vnoiser_kind", vkinds.index(current), list(vkinds)
                )
                tooltip(
                    "What the vnoiser panel shows of the run's domain, as the run wrote it: denoised (the trace "
                    "its curation and peaks use), dff, zscore, raw (the domain's F)"
                )
                if changed:
                    self._vnoiser_kind = vkinds[sel]
            if self._shows_trace_unit():
                imgui.same_line(0, em(0.4))
                if imgui.small_button(
                    "Run vnoiser##traces" if vrun is None else "vnoiser tab##traces"
                ):
                    from mbo_utilities.gui.widgets.pipelines import open_pipeline

                    open_pipeline(self.host, "vnoiser", seed=True)
                tooltip(
                    f"The vnoiser tab (Process) on {unit}, the ROI on screen ticked to run: pick its domains, "
                    "pixels, polarity and dF/F there"
                )
        if callable(getattr(self.host, "reference_view", None)):
            imgui.same_line(0, em(0.8))
            if imgui.small_button("Reference image##traces"):
                self.host.reference_view()
            tooltip(
                "The picture this recording's lines or patches were drawn on, with them drawn and the slider's ROI "
                "thick. Click one there to select it."
            )
        spike_target = self._spike_target(rows)
        if spike_target is not None:
            imgui.same_line(0, em(0.8))
            if imgui.small_button("Spike average##traces"):
                self._open_spike_average(*spike_target)
            tooltip(
                "This ROI averaged around its spikes (its run's detected events, the events its curation accepts, "
                "or its trace over a threshold): the movie, the trace and every motion correction, in a window "
                "of its own"
            )
        for window in list(self._windows):
            state = window.poll()
            if state is not None:
                self._windows.remove(window)
                if state:
                    self.status = state
        imgui.same_line(0, em(0.8))
        if target is not None:
            proj, size = self._window_spec()
            window = f" · {proj} {size}" if size > 1 else ""
            imgui.text_disabled(
                (f"{self.unit} · " if self.unit else "")
                + f"{target[0]}, frame {self.current_frame()}{window}"
            )
            tooltip(
                "drag pans, scroll zooms, double-click fits, right-click for autofit and the x unit · shift+scroll "
                "zooms x only, alt+scroll zooms y only"
            )
        elif self.pending_traces is not None:
            self.pending_traces()
        else:
            imgui.text_disabled(
                "No traces shown. Click an ROI with traces, or draw one (a, r) with trace on draw."
            )
        show_motion = motion is not None and self.show_motion
        show_behavior = behavior is not None and self.show_behavior
        if (
            target is None
            and not show_motion
            and not show_behavior
            and not show_vnoiser
        ):
            # nothing to plot: the panel is its row of controls, the plot empty for when it comes back
            self._traces_panel.height = int(imgui.get_frame_height_with_spacing())
            traces_panel = self._traces_panel_name()
            if self.trace_plot is not None and self.trace_plot._lines.get(traces_panel):
                self._plot_lines_key = None
                self.trace_plot.set(traces_panel, [])
                self._plot_mode([])
            return
        self._traces_panel.height = (
            PANEL_HEIGHT
            + (MOTION_PANEL_HEIGHT - PANEL_HEIGHT)
            * (int(show_motion) + int(show_vnoiser))
            + BEHAVIOR_PLOT_HEIGHT * int(show_behavior)
        )
        if self.strip.collapsed or self.tdim is None:
            return
        vkind = None
        vpanel = None
        if show_vnoiser:
            vkind = displayed_kind(vevery[0], self._vnoiser_kind)
            vpanel = f"vnoiser {y_label(vevery[0], vkind)}"
        self._sync_plot(rows, vpanel)
        plot = self.trace_plot
        nt = len(plot.x)
        if vpanel is not None:
            vkey = (tuple(t.key for t in vholding), vkind, self._plot_frames)
            if vkey != self._vnoiser_lines_key:
                self._vnoiser_lines_key = vkey
                vlines = []
                for i, t in enumerate(vholding):
                    y = display_trace(t, vkind)
                    if y is None:
                        continue
                    rgb = (
                        GROUP_COLORS[i % len(GROUP_COLORS)]
                        if len(vholding) > 1
                        else None
                    )
                    vlines.append((t.label, self._on_frames(t.key, y, nt), rgb))
                plot.set(vpanel, vlines)
        lines_key = (
            None
            if target is None
            else tuple((label, key, rgb) for label, key, rgb in target[1]),
            self.kind,
            self.flip,
            repr(self.dff),
            self.deflection(),
            self._window_spec(),
            self._plot_frames,
        )
        if lines_key != self._plot_lines_key:
            self._plot_lines_key = lines_key
            lines = []
            for label, key, rgb in [] if target is None else target[1]:
                y = self._display(key)
                if y is None:
                    continue
                lines.append((label, self._on_frames(key, self._windowed(y), nt), rgb))
            plot.set(self._traces_panel_name(), lines)
            self._plot_mode(lines)
        plot.frame = self.current_frame()
        if show_behavior:
            self._draw_behavior_row(behavior)
        moved = plot.draw()
        if moved is not None:
            self.playhead.seek(self.viewer_axis().seconds(moved), source="trace_plot")

    def _shows_trace_unit(self) -> bool:
        """True when the recording on screen is a MESc scan vnoiser can run on and the widget sits in a viewer with a Process tab."""
        from mbo_utilities.arrays.mesc import MescArray, is_trace_unit, list_mesc_units

        src = base_array(self.iw.data[0]) if self.iw.data else None
        if self.host is None or not isinstance(src, MescArray):
            return False
        if (
            self._trace_unit[0] is not src
        ):  # the unit table is read once per recording shown
            rows = list_mesc_units(src.filenames[0])
            row = next((u for u in rows if u["key"] == src.unit_key), None)
            self._trace_unit = (src, row is not None and is_trace_unit(row))
        return self._trace_unit[1]

    def _draw_trace_settings(self, rows) -> None:
        """The popup setting how the rows' measured traces are shown: mean
        subtraction and inversion (the viewer's, or set here), the baseline
        of a dF/F computed here and the z-score's slow drift and noise. The
        dF/F starts from the first plotted row's pipeline settings and can go
        back to them.
        """
        if not imgui.begin_popup("##trace_settings"):
            return
        changed = False
        viewer = self.viewer_deflection()
        for label, attr, seen, about in (
            (
                "mean subtraction",
                "trace_subtract",
                viewer[0],
                "F - mean over time: the viewer's Mean Subtraction on the image, or set here",
            ),
            (
                "invert",
                "trace_invert",
                viewer[1],
                "2 mean - F, before any dF/F or z-score: a dimming indicator's events point up. The viewer's "
                "Invert Deflection on the image, or set here",
            ),
        ):
            names = [f"as the viewer ({'on' if seen else 'off'})", "off", "on"]
            imgui.set_next_item_width(em(10))
            moved, idx = imgui.combo(
                f"{label}##{attr}", DEFLECTION_MODES.index(getattr(self, attr)), names
            )
            tooltip(about)
            if moved:
                setattr(self, attr, DEFLECTION_MODES[idx])
                changed = True
        if self.dff is None:
            self.dff = replace(
                trace_profile(rows[0].engine).dff if rows else DffSettings()
            )
        d = self.dff
        imgui.separator_text("dF/F")
        methods = list(DFF_METHODS)
        imgui.set_next_item_width(em(12))
        moved, idx = imgui.combo(
            "baseline##dff_method", methods.index(d.method), list(DFF_METHODS.values())
        )
        tooltip(
            "F0 under dF/F = (F - F0) / F0. rolling max-min: suite2p's, the rolling max of the rolling min of "
            "a smoothed F. rolling percentile: that percentile of F in a moving window. Gaussian low-pass: the "
            "voltage archive's. bleaching fit: an exponential decay fitted to F. fixed window: the mean over "
            "a stretch of the recording, MESc's F0. static percentile: one value for the whole row."
        )
        if moved:
            d.method = methods[idx]
            changed = True
        fields = {
            "maxmin": (("window_s", "window (s)"), ("sigma_s", "smoothing (s)")),
            "rolling": (("window_s", "window (s)"), ("percentile", "percentile")),
            "gaussian": (("gaussian_s", "sigma (s)"),),
            "exponential": (),
            "window": (("f0_start_s", "from (s)"), ("f0_stop_s", "to (s)")),
            "percentile": (("percentile", "percentile"),),
        }[d.method]
        for name, label in fields:
            imgui.set_next_item_width(em(6))
            if name == "percentile":
                moved, value = imgui.slider_float(
                    f"{label}##dff_{name}", getattr(d, name), 1.0, 99.0, "%.0f"
                )
            else:
                moved, value = imgui.input_float(
                    f"{label}##dff_{name}", getattr(d, name), 0.1, 1.0, "%.2f"
                )
                value = max(0.0, value)
            if moved:
                setattr(d, name, value)
                changed = True
        if d.method == "window" and d.f0_stop_s <= d.f0_start_s:
            d.f0_stop_s = d.f0_start_s + 0.1
        if d.method != "percentile" and not all(t.fs for t in rows):
            imgui.text_disabled(
                "a row without a sampling rate uses the static percentile"
            )
        imgui.separator_text("z-score")
        noises = list(NOISE_ESTIMATES)
        imgui.set_next_item_width(em(10))
        moved, idx = imgui.combo(
            "noise##z_noise",
            noises.index(d.noise),
            [NOISE_LABELS[n] for n in noises],
        )
        tooltip(
            "What the z-score divides by: the SD of what is left after the slow drift, or 1.4826 x its median "
            "absolute deviation, which spikes do not inflate"
        )
        if moved:
            d.noise = noises[idx]
            changed = True
        imgui.set_next_item_width(em(6))
        moved, value = imgui.input_float(
            "slow drift (s)##z_slow", d.slow_s, 0.5, 5.0, "%.2f"
        )
        tooltip(
            "A Gaussian of this many seconds taken off the dF/F first; 0 takes off its mean"
        )
        if moved:
            d.slow_s = max(0.0, value)
            changed = True
        imgui.spacing()
        if imgui.small_button("pipeline defaults##trace_settings"):
            self.dff = None
            self.trace_subtract = self.trace_invert = "viewer"
            changed = True
        if changed:
            self._redisplay()
        imgui.end_popup()

    def _member_line(self, member) -> tuple | None:
        """The one trace key a group member plots: a pixel average's row,
        an algo row's own, a drawn ROI's newest.
        """
        if len(member) == 3:
            return self.pixels.get(member)
        si, k = member
        keys = (
            self._member_keys(si, k)
            if si >= 0
            else [t.key for t in self.traces.for_roi(self.store.rois[k].uid)]
        )
        return keys[-1] if keys else None

    def _plot_axis(self) -> tuple[int, np.ndarray | None]:
        """``(frames, seconds of each frame or None)``: the viewer's T axis, which the plot spans."""
        nt = max(int(self.iw.data[0].shape[0]), 1) if self.tdim is not None else 1
        if not self.fs() or nt < 2:
            return nt, None
        axis = self.viewer_axis()
        return nt, np.arange(nt, dtype=np.float64) / axis.per_second + axis.offset

    def _on_frames(self, key, y: np.ndarray, nt: int) -> np.ndarray:
        """One row's samples where the viewer's frames sit: as they are when
        the row is on the viewer's clock, else interpolated, NaN outside the
        stretch it covers (a frame window, another rate or binning).
        """
        src = self.trace_axis(self.traces.get(key))
        dst = self.viewer_axis()
        if len(y) == nt and src == dst:
            return y
        xs = np.arange(len(y), dtype=np.float64) / src.per_second + src.offset
        at = np.arange(nt, dtype=np.float64) / dst.per_second + dst.offset
        return np.interp(
            at, xs, y.astype(np.float64), left=np.nan, right=np.nan
        ).astype(np.float32)

    def _motion_lines(self, nt: int) -> list:
        """The recording's motion shifts on the plane on screen, on the viewer's frames."""
        motion = self.motion.motion if self.motion else None
        if motion is None:
            return []
        at = np.arange(nt, dtype=np.float64) / self.viewer_axis().per_second
        lines = []
        for label, (t, shift) in motion.traces.items():
            if motion.planes.get(label, self.slice.z) != self.slice.z:
                continue
            y = np.interp(
                at,
                np.asarray(t, np.float64),
                np.asarray(shift, np.float64),
                left=np.nan,
                right=np.nan,
            )
            lines.append(
                (
                    label,
                    y.astype(np.float32),
                    MOTION_COLORS.get(label[0], (0.8, 0.8, 0.8)),
                )
            )
        return lines

    def _traces_panel_name(self) -> str | None:
        """The trace plot's panel the selection's rows go in: after the motion
        correction, before the vnoiser run's.
        """
        if self.trace_plot is None:
            return None
        panels = self.trace_plot.panels
        return panels[1] if self._motion_panel and len(panels) > 1 else panels[0]

    def _sync_plot(self, rows, vnoiser_panel: str | None = None) -> None:
        """Build or reset the trace plot for the frames and panels on screen
        (the motion correction, the rows, the vnoiser run's own panel), then
        put the motion's lines in.
        """
        nt, timings = self._plot_axis()
        motion = bool(self.motion) and self.show_motion
        traces_panel = self.plot_y_label(rows) if rows else "traces"
        self._motion_panel = motion
        panels = (
            ((self.motion.y_label,) if motion else ())
            + (traces_panel,)
            + ((vnoiser_panel,) if vnoiser_panel else ())
        )
        key = (
            nt,
            None if timings is None else float(timings[-1]),
            panels,
            self.show_behavior,
        )
        if self.trace_plot is None:
            # no autofit, as in masknmf: the zoom set on one ROI's traces holds while selecting others
            self.trace_plot = TracePlot(panels, nt, timings, autofit=False)
            self.trace_plot._use_time = (
                self.trace_plot._time is not None
            )  # seconds whenever there is a rate
        elif key != self._plot_frames:
            same_frames = (
                self._plot_frames is not None and key[:2] == self._plot_frames[:2]
            )
            span = self.trace_plot._x_span
            self.trace_plot.reset(panels, nt, timings)
            if same_frames and span is not None:
                # only the panels changed: the x zoom holds and the new panel is not refit when its lines land
                self.trace_plot._x_span = span
                self.trace_plot._fitted = set(panels)
            if (  # a rate that stays keeps the unit picked
                timings is None
                or self._plot_frames is None
                or self._plot_frames[1] is None
            ):
                self.trace_plot._use_time = self.trace_plot._time is not None
        if key != self._plot_frames:
            self._plot_frames = key
            self._motion_key = None
            self._plot_lines_key = None
            self._vnoiser_lines_key = None
            if self.behavior and self.show_behavior:
                axis = self.viewer_axis()
                for i, (name, spans) in enumerate(
                    (self.behavior.behavior.epochs or {}).items()
                ):
                    spans = np.asarray(spans, np.float64)
                    if not len(spans):
                        continue
                    frames = np.rint(spans * axis.per_second).astype(np.int64)
                    self.trace_plot.span(
                        name,
                        frames[:, 0],
                        frames[:, 1],
                        EPOCH_COLORS[i % len(EPOCH_COLORS)],
                    )
        if motion and self._motion_key != self.slice.z:
            self._motion_key = self.slice.z
            self.trace_plot.set(panels[0], self._motion_lines(nt))

    def _plot_mode(self, lines) -> None:
        self.trace_plot.background = _TRACE_MODES["selection" if lines else "normal"]

    def _draw_behavior_row(self, behavior: BehaviorPlot) -> None:
        """The behavior raster over the trace panels, its x range the traces'."""
        plot = self.trace_plot
        axis = self.viewer_axis()
        x_per_second = (
            1.0 if plot._use_time and plot._time is not None else axis.per_second
        )
        span = plot._x_span
        moved, held = behavior.draw(
            "##roi_behavior_plot",
            BEHAVIOR_PLOT_HEIGHT - imgui.get_style().item_spacing.y,
            cursor=float(plot.x[plot.frame]),
            cursor_id=2,
            x_per_second=x_per_second,
            x_label="",
            x_axis=False,
            x_limits=span,
        )
        if held and moved is not None:
            self.playhead.seek(float(moved) / x_per_second, source="behavior_plot")
