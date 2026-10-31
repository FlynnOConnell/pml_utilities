"""Per-line traces of an AOD unit for the Traces tab: saved from the newest vnoiser run, else each line's mean computed on a background thread."""

from __future__ import annotations

import numpy as np

ON_THICKNESS = 2.5
SELECTED_THICKNESS = 4.5


def run_roi_traces(
    mesc_path, unit_key: str, n_rois: int, channel: int = 0, conversion=(1.0, 0.0)
) -> tuple[np.ndarray, str] | None:
    """``(K, T)`` per-ROI traces of a unit from the recording's last vnoiser
    run: each line's or patch's raw mean (the run's ``members/raw``), in
    photon counts (``conversion`` applied when the run kept raw counts).
    None when there is no run of this recording, the run read another
    channel or not every ROI, or it predates member traces; the caller
    then reads them itself. Never a denoised trace: those belong to domains.
    """
    from mbo_utilities.results import (
        ResultsArray,
        unit_for_source,
        vnoiser_run_for_mesc,
    )

    run_dir = vnoiser_run_for_mesc(mesc_path)
    if run_dir is None:
        return None
    run = ResultsArray(run_dir, source=False)
    if int(run.results.source.get("channel") or 0) != int(channel):
        return None
    name = unit_for_source(run.results, unit_key)
    unit = run.results.units.get(name) if name else None
    raw = None if unit is None else unit.member_traces.get("raw")
    if raw is None:
        return None
    ids = [int(i) for i in (unit.attrs.get("member_ids") or range(len(raw)))]
    if len(ids) != len(raw) or sorted(ids) != list(range(n_rois)):
        return None
    F = np.asarray(raw, dtype=np.float32)[np.argsort(ids)]
    if not run.results.source.get("convert"):
        F = F * np.float32(conversion[0]) + np.float32(conversion[1])
    return F, f"{run.pipeline} {unit.name} line means from {run_dir.name}"


def saved_roi_traces(ref_arr, channel: int = 0) -> tuple[np.ndarray, str] | None:
    """``(K, T)`` traces the recording's newest vnoiser run holds for a unit, in photon
    counts (each ROI's raw mean it read), with where they came from; None when there is
    no run. Nothing is written.
    """
    from mbo_utilities.analysis.roi_traces import roi_geometry

    n_rois = len(roi_geometry(ref_arr))
    munit = ref_arr.metadata["mesc_unit"].rsplit("/", 1)[-1]
    conv = ref_arr.metadata.get("mesc_channel_conversion") or []
    conversion = (
        (float(conv[channel]["scale"]), float(conv[channel]["offset"]))
        if channel < len(conv)
        else (1.0, 0.0)
    )
    return run_roi_traces(ref_arr.source_path, munit, n_rois, channel, conversion)


class TraceJob:
    """``(K, T)`` per-ROI traces for the Traces tab: the recording's vnoiser run
    when it has them, else the raw reduction (``roi_workflow.linescan_roi_means``)
    on a daemon thread so the window opens at once. The reads and means release the GIL, so the
    render loop stays responsive; ``result`` is set when done, ``error`` on
    failure, and ``progress`` is ``(rois done, rois total)`` meanwhile.

    With ``auto=False`` (the Options toggle, for imaging rigs) nothing is
    computed: the job idles and ``start()`` runs it when asked.
    """

    def __init__(self, ref_arr, channel: int, auto: bool = True):
        self.ref_arr = ref_arr
        self.channel = int(channel)
        self.result: np.ndarray | None = None
        self.error: BaseException | None = None
        self.source = ""
        from mbo_utilities.analysis.roi_traces import roi_geometry

        self.progress = (0, len(roi_geometry(ref_arr)))
        self.thread = None
        saved = saved_roi_traces(ref_arr, self.channel)
        if saved is not None:
            self.result, self.source = saved
            print(f"traces: {self.source}")
            return
        if auto:
            self.start()
        else:
            print(
                "traces: no vnoiser run holds traces for this unit and background computation "
                "is off (Options); use 'compute traces' in the line panel to run it now."
            )

    @property
    def idle(self) -> bool:
        """Nothing loaded and nothing running: waiting for ``start()``."""
        return self.thread is None and not self.done

    def start(self) -> None:
        import threading

        if not self.idle:
            return
        print(
            f"computing per-ROI traces ({self.ref_arr.shape[0]} timepoints, channel "
            f"{self.channel}) in the background..."
        )
        self.thread = threading.Thread(
            target=self._run, name="linescan-traces", daemon=True
        )
        self.thread.start()

    @property
    def done(self) -> bool:
        return self.result is not None or self.error is not None

    def _on_roi(self, i, k, seconds) -> None:
        self.progress = (i + 1, k)
        print(f"  traces: ROI {i + 1}/{k} done ({seconds:.1f}s)")

    def _run(self) -> None:
        from mbo_utilities.analysis.roi_traces import (
            TraceSettings,
            extract_traces,
            roi_geometry,
        )
        from mbo_utilities.roi_workflow import linescan_roi_means

        try:
            if self.ref_arr.metadata.get("mesc_roi_extents"):
                self.result = linescan_roi_means(
                    self.ref_arr, channel=self.channel, progress=self._on_roi
                )
            else:
                found = extract_traces(
                    self.ref_arr,
                    {f"roi{k}": [k] for k in roi_geometry(self.ref_arr)},
                    TraceSettings(convert=True),
                    channel=self.channel,
                    progress=self._on_roi,
                )
                self.result = found.F.astype(np.float32)
            self.source = f"computed from channel {self.channel}"
        except BaseException as e:
            self.error = e
            print(f"traces failed: {e!r}")

    def wait(self) -> np.ndarray | None:
        if self.thread is not None:
            self.thread.join()
        return self.result


class StandardTraces:
    """An AOD unit's per-ROI traces on the standard viewer (``mbo
    file.mesc``), a line scan's lines or a chessboard / ribbon scan's
    patches, one per Z index (a single-patch scan's one frame): a
    :class:`TraceJob` for the traces, its progress on the ROI widget's
    Traces tab until they are in, then one
    :class:`~mbo_utilities.annotation.RoiTrace` row per ROI in its trace
    table (``member`` rows under this unit's source name, in photon counts).
    The plotted row follows the ROI slider and the slider follows a row
    picked in the trace table, a vnoiser domain's to its first line; a
    vnoiser run's domains are the Traces tab's own panel, never this row.
    The unit's motion correction reaches the tab through the array
    (``motion_correction``). Kept on ``parent.linescan_traces``; ``close``
    takes the rows off.
    """

    def __init__(self, parent, arr):
        from mbo_utilities.preferences import get_linescan_auto_traces

        self.parent = parent
        self.arr = arr
        self.roi = parent.manual_roi
        self.strip = parent.top_strip
        self.iw = parent.image_widget
        names = tuple(getattr(self.iw, "_slider_dim_names", None) or ())
        self.roi_dim = next((d for d in names if d.lower() == "roi"), None)
        munit = arr.metadata["mesc_unit"].rsplit("/", 1)[-1]
        self.name = (
            f"{munit} lines"
            if arr.metadata.get("mesc_layout") == "packed"
            else f"{munit} patches"
        )
        view = getattr(parent, "slice", None)  # a bare host has no Slice
        channel = int(view.c or 0) if view is not None else 0
        self.job = TraceJob(arr, channel, auto=get_linescan_auto_traces())
        self.attached = False
        self._last_roi = None
        self._last_sel = None
        self.roi.pending_traces = self.draw_pending
        self.strip.add_hook(self)

    def __call__(self) -> None:
        if not self.attached and self.job.done and self.job.result is not None:
            self._attach()
        rows = self.roi.traces.from_source(self.name)
        if not rows:
            return
        n = len(rows)
        selected = (
            min(max(int(self.iw.indices[self.roi_dim]), 0), n - 1)
            if self.roi_dim is not None
            else 0
        )
        sel = self.roi.trace_sel
        if selected != self._last_roi:
            self._last_roi = selected
            self.roi.select_trace(("member", self.name, selected))
            self._last_sel = set(self.roi.trace_sel)
        elif sel != self._last_sel:
            self._last_sel = set(sel)
            if len(sel) == 1 and self.roi_dim is not None:
                trace = self.roi.traces.get(next(iter(sel)))
                if (
                    trace is not None
                    and trace.z is not None
                    and (
                        trace.source == self.name
                        or (trace.engine == "vnoiser" and "unit" in trace.extra)
                    )
                ):
                    self.iw.indices[self.roi_dim] = int(trace.z)
                    self._last_roi = int(trace.z)

    def draw_pending(self) -> None:
        """On the Traces tab while the job has nothing to show: progress
        while it computes, a ``compute traces`` button while it idles
        (background computation off in Options).
        """
        from imgui_bundle import imgui

        if self.job.error is not None:
            imgui.text_disabled(f"traces failed: {self.job.error!r}")
        elif self.job.idle:
            imgui.text_disabled("no vnoiser run holds traces for this unit")
            imgui.same_line(0, 8)
            if imgui.button("compute traces"):
                self.job.start()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "reduce every ROI to a trace on a background thread now. "
                    "Options turns this on automatically."
                )
        else:
            done, total = self.job.progress
            imgui.text_disabled(f"computing traces  ROI {done}/{total}")
            imgui.same_line(0, 8)
            imgui.progress_bar(done / total if total else 0.0, imgui.ImVec2(-1, 0), "")

    def _attach(self) -> None:
        from mbo_utilities.annotation import RoiTrace
        from mbo_utilities.lazy_array import base_array

        md = self.arr.metadata
        extents = md.get("mesc_roi_extents") or []
        n = len(extents) or int(self.job.result.shape[0])
        fs = float(self.arr.fs or 1.0)
        traces = np.asarray(self.job.result[:n], dtype=np.float32)
        positions = getattr(base_array(self.arr), "line_positions", None)
        self.roi.traces.drop_source(self.name)
        for k in range(traces.shape[0]):
            extra = {"line": k, "from": str(self.job.source), "unit": "photons"}
            label = f"ROI {k}"
            if positions is not None and k < len(positions):
                extra.update(positions[k])
            self.roi.traces.add(
                RoiTrace(
                    uid=0,
                    member=k,
                    source=self.name,
                    label=label,
                    fs=fs,
                    z=k,
                    c=int(self.job.channel),
                    F=traces[k],
                    extra=extra,
                )
            )
        self.roi.pending_traces = None
        self.attached = True

    def close(self) -> None:
        self.strip.remove_hook(self)
        self.roi.pending_traces = None
        self.roi.traces.drop_source(self.name)


def attach_standard_traces(parent) -> StandardTraces | None:
    """The per-ROI traces of a ``PreviewDataWidget`` showing an AOD ``.mesc``
    unit (a line scan, chessboard or ribbon scan: ``ROI_LAYOUTS``, one ROI
    per Z index; or a single-patch chessboard or ribbon scan, its frame the
    one ROI), on its ROI widget's Traces tab; None (nothing added) for
    anything else, and without the ROI widget (Widgets > Manual ROI
    Labeling), whose tab they live on.
    """
    from mbo_utilities.arrays.mesc import PATCH_MODALITIES, ROI_LAYOUTS, MescArray
    from mbo_utilities.lazy_array import base_array

    data = parent.image_widget.data if parent.image_widget is not None else None
    arr = base_array(data[0]) if data else None
    if not isinstance(arr, MescArray):
        return None
    md = arr.metadata
    single_patch = (
        md.get("mesc_layout") == "frames"
        and md.get("mesc_modality") in PATCH_MODALITIES
        and int(arr.shape[0]) > 1
    )
    if md.get("mesc_layout") not in ROI_LAYOUTS and not single_patch:
        return None
    if parent.manual_roi is None or parent.top_strip is None:
        return None
    traces = StandardTraces(parent, arr)
    parent.linescan_traces = traces
    return traces
