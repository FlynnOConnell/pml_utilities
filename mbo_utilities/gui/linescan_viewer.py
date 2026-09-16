"""Reference line-scan + Z-stack viewer for Femtonics AOD .mesc files.

The scientist draws each line-scan ROI on a reference Z-stack, one line at a
time, each at its own depth. This window puts those lines back where they
were drawn and keeps the three views of the same experiment in step:

- **Snapshot** (top middle): the raster image the lines were drawn on (the
  unit's ``BackgroundImagePath``, taken seconds before the scan), composite
  of the channels, every line on it; lines more than 1 um off its plane are
  faint. This is the ground truth for XY placement.
- **Z-stack** (top right): every line, drawn on the slice it was scanned at.
  Lines on the current slice are solid; lines from other depths stay as
  faint ghosts (toggle) so the stack is never "empty" while scrubbing, and a
  dot marks each line's start (kymograph column 0). The subplot title says
  which slice this is, its depth, which ROIs sit on it, and where the
  nearest line-carrying slices are above and below.
- **Reference** (top left): the line-scan unit itself. Its ROI slider selects
  a line: the line is highlighted on the stack and (by default) the Z slider
  jumps to its slice. Clicking a line on the stack selects it the other way.
- **Traces** (bottom): mean fluorescence along each line over time, one row
  per ROI, with a time cursor tied to the Reference's Timepoint slider
  (drag the cursor to scrub, scrub the slider to move the cursor).
- **ROI panel** (right): one row per line with slice, depth, offset from
  that slice, length and sampling; click a row to select. Prev/next-depth
  buttons (also ``[`` / ``]``) walk only through slices that carry lines;
  ``n`` / ``p`` step through ROIs.
- **Curation** (top edge + under the ROI panel, when vnoiser is installed):
  "curate ROI n" runs vnoiser's wavelet denoiser on the selected line's
  trace (cached beside the file) and opens the event curation panels on it;
  the focused candidate moves the Timepoint so the kymograph shows it.
  Labels go to ``.curation/<mode>_template_curation.json`` beside the file.

Panels are genuinely independent (own controller, own sliders, movable
separately): built on `fastplotlib.widgets.nd_widget` directly rather than
``MboNDViewer``, which syncs controllers and assumes every array shares the
same positional dim meaning (see ``gui/_ndviewer.py``), wrong for a
linescan (T/C/ROI) next to a Z-stack (C/Z).

Geometry comes from ``mbo_utilities.arrays.mesc_geometry`` (raw
``driftEndPoints`` / ``ReferenceViewportJSON`` / ``MinZ``/``MaxZ``/``ZDim``
attrs); the micron-to-pixel convention was verified on real data (see
``um_to_pixels``), ``--flip-y`` is an escape hatch only.

The Z-stack picker scores every stack against the chosen lines (fraction
of lines inside its field and depth range, pixel size) and defaults to the
one ``analysis.linescan.pair_reference_zstack`` picks. A stack holding none
of the lines is refused: drawing them on it is meaningless.

Usage:
    mbo linescan scan.mesc --view [--unit MUnit_35]
    mbo linescan <animal>/<expt> --view the notebook's experiment (or PF) folder:
                                        its <expt>/<expt>.mesc, Z-stack and PF traces
    (`mbo scan.mesc` opens the image viewer on the line scan with the curation
    widget; `mbo curate` is the dashboard alone, gui/curation_viewer.py)
    python -m mbo_utilities.gui.linescan_viewer [mesc_path] [--ref MUnit_x]
        [--zstack MUnit_y] [--zstack-file stack.mesc] [--channel 0] [--flip-y]
        [--no-traces] [--traces rois_linescan/MUnit_x] [--curate 0]
        [--no-curation] [--dry-run] [--screenshot out.png]

``--dry-run`` prints the unit choice, every stack's fit and the line
placement without opening a window; ``--screenshot`` renders the window
offscreen to a PNG and exits (no display needed).
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from mbo_utilities.arrays.mesc_geometry import (
    linescan_endpoints_um,
    neighbour_slices,
    roi_placements,
    slice_depths_um,
    slices_with_rois,
    um_to_pixels,
    viewport_geometry,
    zstack_depth_info,
)

ON_THICKNESS = 2.5
SELECTED_THICKNESS = 4.5
GHOST_THICKNESS = 1.0
GHOST_ALPHA = 0.28
START_DOT_SIZE = 9.0
TRACE_SEPARATION = 1.4
RTMC_COLORS = ((0.95, 0.35, 0.35, 0.9), (0.35, 0.85, 0.4, 0.9), (0.4, 0.55, 1.0, 0.9))
LINE_PANEL_WIDTH = 360
TRACES_PANEL_HEIGHT = 260
# with a motion-correction plot under the F plot
TRACES_RTMC_PANEL_HEIGHT = 400
# the F plot's share of the strip when the motion plot shows
TRACES_F_SHARE = 0.58


def _console_pick_unit(
    units: list[dict], label: str, *, extra: dict[str, str] | None = None, default: str | None = None,
) -> str | None:
    """Print a table of ``units`` and read a chosen index from stdin.

    ``extra`` adds one text column per unit key; ``default`` is the key a
    blank answer picks (with no default, blank cancels).
    """
    from mbo_utilities.gui.run_gui import _fmt_duration

    default_idx = next((i for i, u in enumerate(units) if u["key"] == default), None)
    print(f"\n{label} -- {len(units)} unit(s):")
    print(f"{'#':>3}  {'Unit':<14} {'Type':<16} {'T':>7} {'C':>2} "
          f"{'Z/ROI':>8} {'Y':>5} {'X':>5}  {'Duration':>9}  "
          f"{'Fit' if extra else ''}{'':<{29 if extra else 0}}Comment")
    for i, u in enumerate(units):
        t, c, z, y, x = u["shape"]
        if u["kind"] == "multicube" and u["nrois"] > 1:
            z_text = f"{u['nrois']}x{z // u['nrois']}"
        elif u["nrois"] > 1:
            z_text = f"{z} ROI"
        else:
            z_text = str(z)
        dur = _fmt_duration(u.get("duration_s")) or ""
        fit = f"{extra.get(u['key'], ''):<32}" if extra else ""
        mark = " <- default" if i == default_idx else ""
        print(f"{i:>3}  {u['munit']:<14} {u['modality_name']:<16} {t:>7} {c:>2} "
              f"{z_text:>8} {y:>5} {x:>5}  {dur:>9}  {fit}{u['comment']}{mark}")

    hint = f"blank = {default_idx}" if default_idx is not None else "blank to cancel"
    while True:
        try:
            raw = input(f"{label} index ({hint}): ").strip()
        except EOFError:  # no terminal: take the default, never hang
            raw = ""
            print()
        if not raw:
            return units[default_idx]["key"] if default_idx is not None else None
        try:
            idx = int(raw)
        except ValueError:
            print(f"enter a number 0-{len(units) - 1}, or blank to cancel.")
            continue
        if 0 <= idx < len(units):
            return units[idx]["key"]
        print(f"enter a number 0-{len(units) - 1}, or blank to cancel.")


def _print_metadata(label: str, arr) -> None:
    md = arr.metadata
    print(f"\n{label}: {md.get('mesc_unit')}")
    for key in ("fs", "dx", "dy", "dz", "num_timepoints", "num_zplanes",
                "nchannels", "Ly", "Lx", "comment"):
        if key in md:
            print(f"  {key}: {md[key]}")


def _panel_dims(arr, squeezed, role: str) -> tuple[tuple[str, ...], dict]:
    """This array's own slider-dim names, namespaced by ``role`` so two
    recordings never share a dim key, plus the 1-based ``RangeContinuous``
    for each (see module docstring)."""
    from fastplotlib.widgets.nd_widget._index import RangeContinuous

    labels = arr.slider_dim_labels
    dims = tuple(f"{role}: {name}" for name in labels)
    sizes = squeezed.shape[: len(dims)]
    ranges = {name: RangeContinuous(1, int(size) + 1, 1) for name, size in zip(dims, sizes)}
    return dims, ranges


def _limits(sample) -> tuple[float, float]:
    """(vmin, vmax) for a graphic from a sample of its data"""
    v = np.asarray(sample, dtype=np.float32).ravel()
    lo, hi = np.percentile(v, (0.5, 99.9))
    return float(lo), float(hi if hi > lo else lo + 1)


def pf_roi_traces(mesc_path, unit_key: str, n_rois: int) -> tuple[np.ndarray, str] | None:
    """``(K, T)`` traces for a line-scan unit from the experiment's PF folder:
    each ROI carries the denoised trace of the domain that averages it, ROIs
    outside every domain (the pipeline's background lines) are zero. Returns
    the array and a description, or None when there is no PF folder or the
    pipeline never processed this scan."""
    from mbo_utilities.arrays.pf import PfArray, pf_dir_of

    pf_dir = pf_dir_of(Path(mesc_path).parent.parent) or pf_dir_of(Path(mesc_path).parent)
    if pf_dir is None:
        return None
    scan = str(unit_key).rsplit("_", 1)[-1]
    pf = PfArray(pf_dir, source=False)
    if scan not in pf.traces:
        return None
    traces = pf.traces[scan]
    T = max(len(t) for t in traces.values())
    F = np.zeros((n_rois, T), dtype=np.float32)
    used = []
    for domain, rois in pf.domains.items():
        if domain not in traces:
            continue
        for roi in rois:
            if 0 <= roi < n_rois:
                F[roi, : len(traces[domain])] = traces[domain]
                used.append(domain)
    return F, f"PF scan {scan} ({len(set(used))} domains) from {pf_dir}"


def saved_roi_traces(ref_arr, traces_dir: Path | None = None) -> tuple[np.ndarray, str] | None:
    """``(K, T)`` traces a line-scan unit already has on disk, with where
    they came from: the experiment's PF folder first (the pipeline's own
    traces), else an ``F.npy`` a previous ``mbo linescan`` run left
    (``traces_dir``, or ``rois_linescan/<MUnit>`` beside the file). None
    when there is neither; nothing is written."""
    n_rois = len(ref_arr.metadata.get("mesc_roi_extents") or [])
    munit = ref_arr.metadata["mesc_unit"].rsplit("/", 1)[-1]
    pf = pf_roi_traces(ref_arr.source_path, munit, n_rois)
    if pf is not None:
        return pf
    if traces_dir is None:
        candidate = Path(ref_arr.source_path).parent / "rois_linescan" / munit
        if (candidate / "F.npy").exists():
            traces_dir = candidate
    if traces_dir is not None:
        F = np.load(Path(traces_dir) / "F.npy")
        return F.astype(np.float32, copy=False), f"F.npy from {traces_dir}"
    return None


class TraceJob:
    """``(K, T)`` per-ROI traces for the Traces tab, from the first source
    that has them: the experiment's PF folder (the pipeline's own domain
    traces, one per ROI), an ``F.npy`` a previous ``mbo linescan`` run left
    (given, or beside the file; never written here), else the raw
    reduction (``roi_workflow.linescan_roi_means``) on a daemon thread so
    the window opens at once. The reads and means release the GIL, so the
    render loop stays responsive; ``result`` is set when done, ``error`` on
    failure, and ``progress`` is ``(rois done, rois total)`` meanwhile.

    With ``auto=False`` (the Options toggle, for imaging rigs) nothing is
    computed: the job idles and ``start()`` runs it when asked."""

    def __init__(self, ref_arr, channel: int, traces_dir: Path | None, auto: bool = True):
        self.ref_arr = ref_arr
        self.channel = int(channel)
        self.result: np.ndarray | None = None
        self.error: BaseException | None = None
        self.source = ""
        n_rois = len(ref_arr.metadata.get("mesc_roi_extents") or [])
        self.progress = (0, n_rois)
        self.thread = None
        saved = saved_roi_traces(ref_arr, traces_dir)
        if saved is not None:
            self.result, self.source = saved
            print(f"traces: {self.source}")
            return
        if auto:
            self.start()
        else:
            print("traces: no F.npy or PF traces for this unit and background computation is "
                  "off (Options); use 'compute traces' in the line panel to run it now.")

    @property
    def idle(self) -> bool:
        """Nothing loaded and nothing running: waiting for ``start()``."""
        return self.thread is None and not self.done

    def start(self) -> None:
        import threading

        if not self.idle:
            return
        print(f"computing per-ROI traces ({self.ref_arr.shape[0]} timepoints, channel "
              f"{self.channel}) in the background; run `mbo linescan` once and pass --traces "
              "to skip this next time...")
        self.thread = threading.Thread(target=self._run, name="linescan-traces", daemon=True)
        self.thread.start()

    @property
    def done(self) -> bool:
        return self.result is not None or self.error is not None

    def _on_roi(self, i, k, seconds) -> None:
        self.progress = (i + 1, k)
        print(f"  traces: ROI {i + 1}/{k} done ({seconds:.1f}s)")

    def _run(self) -> None:
        from mbo_utilities.roi_workflow import linescan_roi_means

        try:
            self.result = linescan_roi_means(self.ref_arr, channel=self.channel, progress=self._on_roi)
            self.source = f"computed from channel {self.channel}"
        except BaseException as e:
            self.error = e
            print(f"traces failed: {e!r}")

    def wait(self) -> np.ndarray | None:
        if self.thread is not None:
            self.thread.join()
        return self.result


class TraceAttach:
    """Per-render poll (``figure.add_animations``) that builds the traces
    or curation panel once the ``TraceJob`` finishes, then removes itself."""

    def __init__(self, ndw, job: TraceJob, overlay, line_panel, ref_arr, mesc_path, ref_key,
                 curation: bool, curate: int | None):
        self.ndw = ndw
        self.job = job
        self.overlay = overlay
        self.line_panel = line_panel
        self.ref_arr = ref_arr
        self.mesc_path = mesc_path
        self.ref_key = ref_key
        self.curation = curation
        self.curate = curate
        ndw.figure.add_animations(self)

    def __call__(self) -> None:
        if not self.job.done:
            return
        self.ndw.figure.remove_animation(self)
        traces = self.job.result
        if traces is None:
            return
        traces = traces[: self.overlay.n]
        line_curation = None
        if self.curation:
            line_curation = LineCuration.build(
                self.ndw, self.overlay, self.ref_arr, traces, self.mesc_path, self.ref_key
            )
        from mbo_utilities.gui._top_strip import TopStrip

        # the raw trace and the RTMC curves get a tab beside Curation on the
        # same strip, or their own strip without curation
        strip = TopStrip(self.ndw.figure) if line_curation is None else line_curation.widget.strip
        self.ndw.linescan_traces = LineTracesPanel(
            self.ndw, self.overlay, traces, strip, line_curation is None, curves=self.ref_arr.curves,
        )
        if line_curation is not None:
            self.ndw.linescan_curation = line_curation
            self.line_panel.curation = line_curation
            if self.curate is not None:
                line_curation.curate(self.curate)


def _close_figure(figure) -> None:
    """Close a shown figure's window; an offscreen figure has no output, so
    fall back to its canvas."""
    try:
        figure.close()
    except AttributeError:
        try:
            figure.canvas.close()
        except Exception:
            pass


def default_linescan_unit(mesc_path, units: list[dict]) -> str:
    """Which line-scan unit to open when none was named: the first one the
    vnoiser pipeline processed (a PF scan with its number), else the first."""
    packed = [u for u in units if u.get("kind") == "packed"] or list(units)
    try:
        from mbo_utilities.vnoiser import pf_scan_for_mesc
    except ImportError:
        return packed[0]["key"]
    for u in packed:
        if pf_scan_for_mesc(mesc_path, u["key"]) is not None:
            return u["key"]
    return packed[0]["key"]


def _display_normalize(F: np.ndarray) -> np.ndarray:
    """Per-ROI robust 0..1 scaling so rows of very different brightness stack
    legibly; the panel is for seeing events in time, not absolute F."""
    lo = np.percentile(F, 5, axis=1, keepdims=True)
    hi = np.percentile(F, 95, axis=1, keepdims=True)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return np.clip((F - lo) / span, -0.5, 1.5)


class LineScanOverlay:
    """Lines, start dots, labels and titles on the Z-stack, kept in step with
    the Reference's ROI/Timepoint sliders and the trace cursor."""

    def __init__(
        self,
        ndw,
        *,
        ref_key: str,
        zstack_key: str,
        ref_dims: tuple[str, ...],
        zstack_dims: tuple[str, ...],
        placements: list[dict],
        pixel_lines: list[np.ndarray],
        colors: np.ndarray,
        slice_depths: np.ndarray,
        fs: float,
        traces: np.ndarray | None,
        z_index: int = 1,
        trace_index: int | None = 2,
        snapshot: dict | None = None,
    ):
        from mbo_utilities.gui._ndviewer import _ref_to_index

        self._ref_to_index = _ref_to_index
        self.ndw = ndw
        self.ref_key, self.zstack_key = ref_key, zstack_key
        self.placements = placements
        self.colors = colors
        self.slice_depths = slice_depths
        self.zdim = len(slice_depths)
        self.occupied = slices_with_rois(placements)
        self.n = len(placements)

        self.z_dim = next((d for d in zstack_dims if "Z-plane" in d), None)
        self.roi_dim = next((d for d in ref_dims if d.endswith(": ROI")), None)
        self.t_dim = next((d for d in ref_dims if "Timepoint" in d), None)

        self.show_ghosts = True
        self.follow_roi = True
        self.selected = 0
        self.slice = 0
        self.t_index = 0
        self._busy = False
        # called with the ROI index whenever the selection changes
        self.on_select: list = []

        ref_sp = ndw[0].subplot
        z_sp = ndw[z_index].subplot
        self.ref_subplot, self.z_subplot = ref_sp, z_sp

        # static copy of the lines on the snapshot they were drawn on; only
        # the selection highlight changes, and off-plane lines stay faint
        self.snap_lines = None
        self.snap_on_plane = None
        if snapshot is not None:
            sp = snapshot["subplot"]
            self.snap_on_plane = list(snapshot["on_plane"])
            self.snap_lines = sp.add_line_collection(
                snapshot["pixel_lines"], colors=colors, thickness=ON_THICKNESS, name="snapshot_lines",
            )
            for i, g in enumerate(self.snap_lines.graphics):
                g.add_event_handler(partial(self._on_line_click, i), "click")
                r, gg, b = colors[i][:3]
                g.colors = (r, gg, b, 1.0 if self.snap_on_plane[i] else GHOST_ALPHA)
                g.thickness = ON_THICKNESS if self.snap_on_plane[i] else GHOST_THICKNESS
            snap_starts = np.array([seg[0] for seg in snapshot["pixel_lines"]], dtype=np.float32)
            sp.add_scatter(
                snap_starts, colors=colors, sizes=START_DOT_SIZE, mode="simple",
                name="snapshot_starts",
            )
            for i, seg in enumerate(snapshot["pixel_lines"]):
                sp.add_text(
                    str(i), font_size=11, face_color=colors[i], outline_color="black",
                    outline_thickness=0.4, screen_space=True,
                    offset=(float(seg[0, 0]), float(seg[0, 1]), 1.0), anchor="bottom-left",
                )

        self.lines = z_sp.add_line_collection(
            pixel_lines, colors=colors, thickness=ON_THICKNESS, name="line_scan_rois",
        )
        for i, g in enumerate(self.lines.graphics):
            g.add_event_handler(partial(self._on_line_click, i), "click")
        starts = np.array([seg[0] for seg in pixel_lines], dtype=np.float32)
        self.starts = z_sp.add_scatter(
            starts, colors=colors, sizes=START_DOT_SIZE, mode="simple", name="line_starts",
        )
        self.labels = []
        for i, (seg, p) in enumerate(zip(pixel_lines, placements)):
            mid = seg.mean(axis=0)
            txt = z_sp.add_text(
                self._label_text(p),
                font_size=12,
                face_color=colors[i],
                outline_color="black",
                outline_thickness=0.4,
                screen_space=True,
                offset=(float(mid[0]), float(mid[1]), 1.0),
                anchor="bottom-left",
                name=f"line_label_{i}",
            )
            self.labels.append(txt)

        # traces panel, only when computed
        self.trace_stack = None
        self.selector = None
        self.fs = fs
        if traces is not None and trace_index is not None and len(ndw.figure) > trace_index:
            tr_sp = ndw[trace_index].subplot
            self.trace_subplot = tr_sp
            self._build_traces(tr_sp, traces)

        ndw.indices.add_event_handler(self._on_indices)
        ndw.figure.renderer.add_event_handler(self._on_key, "key_down")

        # NDWidget starts every slider at ref value 1; open on the slice that
        # carries ROI 0 rather than the bottom of the stack
        self._apply_slice(0)
        self._apply_selection(0)
        self.goto_slice(self.placements[0]["slice"])
        self._update_titles()

    # ----------------------------------------------------------------- traces
    def _build_traces(self, subplot, traces: np.ndarray) -> None:
        T = traces.shape[1]
        stride = max(1, T // 250_000)
        t_s = np.arange(0, T, stride, dtype=np.float32) / self.fs
        norm = _display_normalize(traces)[:, ::stride]
        data = [np.column_stack([t_s, norm[i]]).astype(np.float32) for i in range(self.n)]
        self.trace_stack = subplot.add_line_stack(
            data, colors=self.colors, thickness=1.0,
            separation=(0.0, TRACE_SEPARATION, 0.0), name="roi_traces",
        )
        # LineStack spaces rows by each row's data range plus ``separation``,
        # so read the real offsets back rather than assuming even spacing
        for i, p in enumerate(self.placements):
            y_row = float(self.trace_stack.graphics[i].offset[1])
            subplot.add_text(
                f"ROI {p['index']}",
                font_size=11,
                face_color=self.colors[i],
                outline_color="black",
                outline_thickness=0.4,
                screen_space=True,
                offset=(float(t_s[0]), y_row + 1.0, 1.0),
                anchor="bottom-left",
            )
        subplot.camera.maintain_aspect = False
        self.selector = self.trace_stack.add_linear_selector(selection=0.0, axis="x")
        self.selector.add_event_handler(self._on_selector, "selection")
        subplot.auto_scale()

    def _on_selector(self, ev) -> None:
        if self._busy or self.t_dim is None:
            return
        t_index = int(round(float(self.selector.selection) * self.fs))
        self._busy = True
        try:
            self.ndw.indices.set_dim_index(self.t_dim, t_index + 1)
        finally:
            self._busy = False

    # --------------------------------------------------------------- styling
    def _label_text(self, p: dict) -> str:
        # a ghost (another slice) gets its index only: seventeen full labels
        # on one panel are unreadable, and the side table has the numbers
        if p["slice"] != self.slice:
            return str(p["index"])
        text = f"ROI {p['index']}  z {p['z_um']:+.1f} um"
        if not p["in_range"]:
            text += f"  (outside stack by {abs(p['dz_um']):.1f} um)"
        return text

    def _apply_slice(self, k: int) -> None:
        self.slice = int(k)
        for i, p in enumerate(self.placements):
            on = p["slice"] == self.slice
            g = self.lines.graphics[i]
            r, gg, b = self.colors[i][:3]
            if on:
                g.visible = True
                g.colors = (r, gg, b, 1.0)
                g.thickness = SELECTED_THICKNESS if i == self.selected else ON_THICKNESS
                self.labels[i].visible = True
                self.labels[i].face_color = (r, gg, b, 1.0)
                self.starts.colors[i] = (r, gg, b, 1.0)
            else:
                g.visible = self.show_ghosts
                g.colors = (r, gg, b, GHOST_ALPHA)
                g.thickness = GHOST_THICKNESS
                self.labels[i].visible = self.show_ghosts
                self.labels[i].face_color = (r, gg, b, min(1.0, GHOST_ALPHA * 2))
                self.starts.colors[i] = (r, gg, b, GHOST_ALPHA if self.show_ghosts else 0.0)
            self.labels[i].text = self._label_text(p)

    def _apply_selection(self, i: int) -> None:
        changed = int(i) != self.selected
        self.selected = int(i)
        if changed:
            for fn in list(self.on_select):
                fn(self.selected)
        for j, p in enumerate(self.placements):
            g = self.lines.graphics[j]
            if p["slice"] == self.slice:
                g.thickness = SELECTED_THICKNESS if j == self.selected else ON_THICKNESS
        if self.trace_stack is not None:
            for j, g in enumerate(self.trace_stack.graphics):
                g.thickness = 2.5 if j == self.selected else 1.0
        if self.snap_lines is not None:
            for j, g in enumerate(self.snap_lines.graphics):
                base = ON_THICKNESS if self.snap_on_plane[j] else GHOST_THICKNESS
                g.thickness = SELECTED_THICKNESS if j == self.selected else base

    def _update_titles(self) -> None:
        p = self.placements[self.selected]
        ref_name = self.ref_key.rsplit("/", 1)[-1]
        z_name = self.zstack_key.rsplit("/", 1)[-1]
        self.ref_subplot.title = (
            f"{ref_name}  ROI {p['index']}  z {p['z_um']:+.1f} um  "
            f"slice {p['slice'] + 1}  {p['length_um']:.0f} um"
        )
        here = self.occupied.get(self.slice, [])
        z_here = self.slice_depths[self.slice]
        where = (
            "lines: " + ", ".join(f"ROI {r}" for r in here) if here else "no lines on this slice"
        )
        self.z_subplot.title = (
            f"{z_name}  slice {self.slice + 1}/{self.zdim}  z {z_here:+.1f} um  |  {where}"
        )

    # ---------------------------------------------------------------- events
    def _on_indices(self, indices: dict) -> None:
        if self.z_dim is not None:
            k = self._ref_to_index(indices[self.z_dim])
            if k != self.slice:
                self._apply_slice(k)
                self._apply_selection(self.selected)
        if self.roi_dim is not None:
            i = self._ref_to_index(indices[self.roi_dim])
            if i != self.selected:
                self._apply_selection(i)
                if self.follow_roi and not self._busy:
                    self.goto_slice(self.placements[i]["slice"])
        if self.t_dim is not None:
            self.t_index = self._ref_to_index(indices[self.t_dim])
        if self.t_dim is not None and self.selector is not None and not self._busy:
            self._busy = True
            try:
                self.selector.selection = self.t_index / self.fs
            finally:
                self._busy = False
        self._update_titles()

    def _on_line_click(self, i: int, ev) -> None:
        self.select_roi(i)

    def _on_key(self, ev) -> None:
        key = getattr(ev, "key", None)
        if key == "[":
            below, _ = neighbour_slices(self.slice, self.occupied)
            if below is not None:
                self.goto_slice(below)
        elif key == "]":
            _, above = neighbour_slices(self.slice, self.occupied)
            if above is not None:
                self.goto_slice(above)
        elif key == "n":
            self.select_roi((self.selected + 1) % self.n)
        elif key == "p":
            self.select_roi((self.selected - 1) % self.n)

    # ------------------------------------------------------------ public api
    def goto_slice(self, k: int) -> None:
        k = int(np.clip(k, 0, self.zdim - 1))
        if self.z_dim is None:
            return
        self._busy = True
        try:
            self.ndw.indices.set_dim_index(self.z_dim, k + 1)
        finally:
            self._busy = False
        if k != self.slice:  # handler may not fire when the value is unchanged
            self._apply_slice(k)
            self._apply_selection(self.selected)
            self._update_titles()

    def select_roi(self, i: int) -> None:
        i = int(i) % self.n
        if self.roi_dim is not None:
            # the indices handler applies the highlight and the Z jump
            self.ndw.indices.set_dim_index(self.roi_dim, i + 1)
        else:
            self._apply_selection(i)
            if self.follow_roi:
                self.goto_slice(self.placements[i]["slice"])
        self._update_titles()

    def set_show_ghosts(self, on: bool) -> None:
        if on != self.show_ghosts:
            self.show_ghosts = bool(on)
            self._apply_slice(self.slice)
            self._apply_selection(self.selected)

    def goto_time(self, t_s: float) -> None:
        """Move the Reference's Timepoint (and the trace cursor) to ``t_s``."""
        if self.t_dim is None:
            return
        index = max(0, int(round(float(t_s) * self.fs)))
        self.ndw.indices.set_dim_index(self.t_dim, index + 1)


class LinePanel:
    """Right-hand imgui panel: one row per line, depth navigation, toggles."""

    def __init__(self, ndw, overlay: LineScanOverlay, size: int = LINE_PANEL_WIDTH, curation=None,
                 units: list[dict] | None = None, switch=None, job: TraceJob | None = None):
        self.overlay = overlay
        self.curation = curation
        # the background trace computation, for a progress line until done
        self.job = job
        # the file's line-scan units; picking another reopens the window on it
        self.units = list(units or [])
        self.switch = switch
        ndw.figure.add_imgui_window(self.draw, location="right", size=size, title=None)

    def draw(self) -> None:
        from imgui_bundle import imgui

        ov = self.overlay
        if len(self.units) > 1 and self.switch is not None:
            labels = [
                f"{u['munit']}  {u['nrois']} lines  {u.get('comment') or ''}".rstrip()
                for u in self.units
            ]
            keys = [u["key"] for u in self.units]
            idx = keys.index(ov.ref_key) if ov.ref_key in keys else 0
            imgui.set_next_item_width(-1)
            changed, idx = imgui.combo("##linescan_unit", idx, labels)
            if imgui.is_item_hovered():
                imgui.set_tooltip("the file's line-scan units; picking one reopens on it")
            if changed and keys[idx] != ov.ref_key:
                self.switch(keys[idx])
                return
        imgui.text(f"{ov.n} line(s) on {len(ov.occupied)} of {ov.zdim} slices")
        z_here = ov.slice_depths[ov.slice]
        imgui.text(f"current slice {ov.slice + 1}/{ov.zdim}   z {z_here:+.1f} um")
        imgui.spacing()

        below, above = neighbour_slices(ov.slice, ov.occupied)
        imgui.begin_disabled(below is None)
        if imgui.button(f"< prev depth{'' if below is None else f'  (slice {below + 1})'}"):
            ov.goto_slice(below)
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(above is None)
        if imgui.button(f"next depth{'' if above is None else f'  (slice {above + 1})'} >"):
            ov.goto_slice(above)
        imgui.end_disabled()

        changed, val = imgui.checkbox("ROI slider jumps Z to the line's slice", ov.follow_roi)
        if changed:
            ov.follow_roi = val
        changed, val = imgui.checkbox("show lines from other depths (faint)", ov.show_ghosts)
        if changed:
            ov.set_show_ghosts(val)
        imgui.spacing()

        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.sizing_fixed_fit
        )
        if imgui.begin_table("line_rois", 7, flags):
            for name in ("", "ROI", "slice", "z um", "off um", "len um", "um/px"):
                imgui.table_setup_column(name)
            imgui.table_headers_row()
            for i, p in enumerate(ov.placements):
                r, g, b = ov.colors[i][:3]
                on_slice = p["slice"] == ov.slice
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.color_button(f"##sw{i}", imgui.ImVec4(r, g, b, 1.0), 0, imgui.ImVec2(12, 12))
                imgui.table_next_column()
                label = f"{p['index']}{'' if p['in_range'] else ' !'}##roi{i}"
                clicked, _ = imgui.selectable(
                    label, i == ov.selected, imgui.SelectableFlags_.span_all_columns
                )
                if clicked:
                    ov.select_roi(i)
                if not p["in_range"] and imgui.is_item_hovered():
                    imgui.set_tooltip(
                        f"scanned {abs(p['dz_um']):.1f} um outside the stack's depth range;\n"
                        "drawn on the nearest edge slice"
                    )
                col = imgui.ImVec4(1, 1, 1, 1) if on_slice else imgui.ImVec4(0.6, 0.6, 0.6, 1)
                for text in (
                    f"{p['slice'] + 1}",
                    f"{p['z_um']:+.1f}",
                    f"{p['dz_um']:+.2f}",
                    f"{p['length_um']:.1f}",
                    "" if p["sample_um"] is None else f"{p['sample_um']:.2f}",
                ):
                    imgui.table_next_column()
                    imgui.text_colored(col, text)
            imgui.end_table()

        imgui.spacing()
        imgui.text_wrapped(
            "dot = start of the line (kymograph column 0). "
            "Click a line or a row to select it. Keys: [ ] prev/next depth "
            "with lines, n / p next/prev ROI. 'off' is how far the line really "
            "sits from the slice it is drawn on."
        )
        if self.job is not None and self.job.idle:
            imgui.spacing()
            if imgui.button("compute traces"):
                self.job.start()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "no saved traces (F.npy or PF) for this unit; reduce every ROI to a "
                    "trace on a background thread now. Options turns this on automatically."
                )
        elif self.job is not None and not self.job.done:
            done, total = self.job.progress
            imgui.spacing()
            imgui.text_disabled(f"computing traces  ROI {done}/{total}")
            imgui.progress_bar(done / total if total else 0.0, imgui.ImVec2(-1, 0), "")
        if self.curation is not None:
            self.curation.draw()



class SliderSelection:
    """What :class:`LineTracesPanel` needs of a line-scan overlay, read off
    the standard viewer's sliders instead: the ROI slider is the selected
    line, the Timepoint slider is the cursor. ``mbo file.mesc`` opens a
    line-scan unit in that viewer, with no overlay."""

    def __init__(self, image_widget, n: int, fs: float):
        from mbo_utilities.annotation.store import CLASS_COLORS
        from mbo_utilities.arrays.features import find_slider_name

        self.iw = image_widget
        self.n = int(n)
        self.fs = float(fs)
        names = tuple(getattr(image_widget, "_slider_dim_names", None) or ())
        self.t_dim = find_slider_name(names, "t")
        self.roi_dim = next((d for d in names if d.lower() == "roi"), None)
        self.colors = np.array(
            [(*CLASS_COLORS[i % len(CLASS_COLORS)][:3], 1.0) for i in range(self.n)], dtype=np.float32
        )

    @property
    def selected(self) -> int:
        if self.roi_dim is None:
            return 0
        return min(max(int(self.iw.indices[self.roi_dim]), 0), self.n - 1)

    @property
    def t_index(self) -> int:
        return int(self.iw.indices[self.t_dim]) if self.t_dim is not None else 0

    def goto_time(self, t_s: float) -> None:
        if self.t_dim is not None:
            self.iw.indices[self.t_dim] = max(0, int(round(float(t_s) * self.fs)))


class PendingTracesPanel:
    """The ``Traces`` tab while its :class:`TraceJob` has nothing to show:
    progress while it computes, a ``compute traces`` button while it idles
    (background computation off in Options)."""

    def __init__(self, strip, job: TraceJob):
        from mbo_utilities.gui._top_strip import TopPanel

        self.strip = strip
        self.job = job
        strip.register(TopPanel("line_traces", "Line traces", self.draw, TRACES_PANEL_HEIGHT, None, 10))

    def close(self) -> None:
        self.strip.unregister("line_traces")

    def draw(self) -> None:
        from imgui_bundle import imgui

        if self.job.error is not None:
            imgui.text_disabled(f"traces failed: {self.job.error!r}")
        elif self.job.idle:
            imgui.text_disabled("no saved traces (F.npy or PF) for this unit")
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
            imgui.progress_bar(done / total if total else 0.0, imgui.ImVec2(-1, 0), "")


class StandardTraces:
    """The ``Traces`` tab on the standard viewer (``mbo file.mesc``) for a
    line-scan unit: a :class:`TraceJob` for the traces, a pending tab until
    they are in, then :class:`LineTracesPanel` on the viewer's own top
    strip, its cursor and line following the Timepoint and ROI sliders.
    Kept on ``parent.linescan_traces``; ``close`` takes it off the strip."""

    def __init__(self, parent, arr):
        from mbo_utilities.preferences import get_linescan_auto_traces

        self.parent = parent
        self.arr = arr
        self.strip = parent.top_strip
        self.job = TraceJob(arr, 0, None, auto=get_linescan_auto_traces())
        self.panel = None
        self.pending = None
        if self.job.done:
            self._attach()
        else:
            self.pending = PendingTracesPanel(self.strip, self.job)
            # polled at the top of every frame the strip draws
            self.strip.add_hook(self)

    def __call__(self) -> None:
        if not self.job.done:
            return
        self.strip.remove_hook(self)
        if self.pending is not None:
            self.pending.close()
            self.pending = None
        if self.job.result is not None:
            self._attach()
        else:
            self.pending = PendingTracesPanel(self.strip, self.job)

    def _attach(self) -> None:
        md = self.arr.metadata
        n = len(md.get("mesc_roi_extents") or []) or int(self.job.result.shape[0])
        # the curation widget lists this unit's ROIs with lazy loaders; give
        # it these traces so a click denoises at once instead of re-reading
        curation = getattr(self.parent, "event_curation", None)
        loaders = getattr(curation, "unit_traces", None) or {}
        loader = loaders.get(md["mesc_unit"].rsplit("/", 1)[-1])
        if loader is not None and loader.traces is None:
            loader.traces = np.asarray(self.job.result)
        selection = SliderSelection(self.parent.image_widget, n, float(md.get("fs") or 1.0))
        self.panel = LineTracesPanel(
            self.parent.image_widget, selection, self.job.result[:n], self.strip, False,
            curves=self.arr.curves,
        )

    def close(self) -> None:
        if self.pending is not None:
            self.pending.close()
            self.pending = None
        if self.panel is not None:
            self.panel.close()
            self.panel = None
        self.strip.remove_hook(self)


def attach_standard_traces(parent) -> StandardTraces | None:
    """The ``Traces`` tab for a ``PreviewDataWidget`` showing a line-scan
    ``.mesc`` unit; None (nothing registered) for anything else."""
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.lazy_array import base_array

    data = getattr(getattr(parent, "image_widget", None), "data", None)
    # the viewer wraps the array in proxies; the traces come from the file
    arr = base_array(data[0]) if data else None
    if not isinstance(arr, MescArray) or arr.metadata.get("mesc_layout") != "packed":
        return None
    if getattr(parent, "top_strip", None) is None:
        return None
    traces = StandardTraces(parent, arr)
    parent.linescan_traces = traces
    return traces


class LineTracesPanel:
    """The selected line's trace as an imgui plot on the top strip: F as a
    faint min/max band, a 25 ms smoothed line over it in the line's colour,
    and a time cursor tied to the Reference's Timepoint (drag it to scrub).

    A scan that ran with real-time motion correction gets a second plot
    under it: the RTMC X/Y/Z correction totals in um, on the same time
    axis (it follows the F plot's pan and zoom) with the same
    cursor, but its own y range and fit. The two are separate traces of
    separate things and are kept apart: the ``RTMC`` box hides the motion
    plot. Its own ``Traces`` tab, on the curation widget's strip when
    there is one.
    """

    def __init__(self, ndw, overlay: LineScanOverlay, traces: np.ndarray, strip, own_strip: bool,
                 tab: bool = True, curves: dict | None = None):
        from mbo_utilities.gui._top_strip import TopPanel

        from mbo_utilities.gui.imgui.lines import decimate_minmax

        # RTMC X/Y/Z totals (um) from the unit's timing curves, when the scan
        # ran with real-time motion correction; times are curve ms -> s.
        # decimated once like the F trace: tens of thousands of points per
        # curve every frame is what made the whole window lag
        self.rtmc = []
        for axis, name in (("X", "RTMC X correction (total)"),
                           ("Y", "RTMC Y correction (total)"),
                           ("Z", "RTMC Z correction (total)")):
            if curves is None or name not in curves:
                continue
            idx, values = decimate_minmax(curves[name]["values"], 4000)
            t = curves[name]["timestamps"][idx.astype(int)] / 1000.0
            self.rtmc.append((axis, np.ascontiguousarray(t), np.ascontiguousarray(values)))
        self.show_rtmc = bool(self.rtmc)
        self.ndw = ndw
        self.overlay = overlay
        self.strip = strip
        self._own_strip = own_strip
        self.fs = float(overlay.fs)
        self.traces = np.asarray(traces, dtype=np.float32)
        self.n = int(self.traces.shape[0])
        self.duration_s = float(max(self.traces.shape[1] - 1, 1)) / self.fs
        self._cache: dict[int, tuple] = {}
        self._fit = True
        self._last = None
        # x range of the F plot this frame, for the motion plot to follow
        self._xlim: tuple[float, float] | None = None
        if tab:
            height = TRACES_RTMC_PANEL_HEIGHT if self.rtmc else TRACES_PANEL_HEIGHT
            self.strip.register(TopPanel("line_traces", "Line traces", self.draw_tab, height, None, 10))

    def close(self) -> None:
        self.strip.unregister("line_traces")
        if self._own_strip:
            self.strip.close()

    def _prepared(self, i: int) -> tuple:
        """``(t_band, band, t_smooth, smooth)`` for line ``i``: raw F min/max
        per 4000 bins, and a 25 ms boxcar at a stride that keeps ~20k points."""
        got = self._cache.get(i)
        if got is None:
            from scipy.ndimage import uniform_filter1d

            from mbo_utilities.gui.imgui.lines import decimate_minmax

            y = self.traces[i].astype(np.float64)
            idx, band = decimate_minmax(y, 4000)
            k = max(1, int(round(0.025 * self.fs)))
            smooth = uniform_filter1d(y, size=k, mode="nearest") if k > 1 else y
            stride = max(1, int(np.ceil(y.size / 20000)))
            got = (idx / self.fs, band, np.arange(0, y.size, stride) / self.fs, smooth[::stride])
            self._cache[i] = got
        return got

    def draw_tab(self) -> None:
        from imgui_bundle import imgui

        ov = self.overlay
        imgui.text_disabled(
            f"ROI {ov.selected} · t {ov.t_index / self.fs:.3f} s · raw F (band) and 25 ms mean; "
            "drag the cursor to scrub, drag pans, scroll zooms"
        )
        imgui.same_line(0, 12)
        if imgui.button("fit##line_traces"):
            self._fit = True
        if self.rtmc:
            imgui.same_line(0, 12)
            _changed, self.show_rtmc = imgui.checkbox("RTMC##line_traces", self.show_rtmc)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "the scan's real-time motion correction: X/Y/Z totals (um) applied "
                    "while it ran, as a second plot on the same time axis"
                )
        avail = max(imgui.get_content_region_avail().y - 2, 60.0)
        if self.rtmc and self.show_rtmc:
            # the motion plot sits under F; it gives up its x label to F's
            top = max(avail * TRACES_F_SHARE, 40.0)
            self.draw(top)
            self.draw_rtmc(max(avail - top - 4, 40.0))
        else:
            self.draw(avail)

    def draw(self, height: float) -> None:
        from imgui_bundle import implot

        from mbo_utilities.gui.imgui.lines import drag_vline, line, line_plot

        ov = self.overlay
        i = int(ov.selected)
        if i != self._last:
            self._last = i
            self._fit = True
        fit, self._fit = self._fit, False
        with line_plot("##line_traces_plot", "time (s)", "F", height=height, fit=fit, legend=True) as ok:
            if not ok:
                return
            # the x axis never leaves the recording: no blank space past
            # either end, no panning beyond it
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, 0.0, self.duration_s)
            t_band, band, ts, smooth = self._prepared(i)
            r, g, b = (float(v) for v in ov.colors[i][:3])
            line(f"ROI {i} raw", band, x=t_band, color=(r, g, b, 0.28), weight=0.8)
            line(f"ROI {i}", smooth, x=ts, color=(r, g, b, 1.0), weight=1.8)
            cursor, held = drag_vline(99, ov.t_index / self.fs, (1.0, 0.85, 0.3, 0.9), 1.5)
            if held:
                ov.goto_time(cursor)
            lim = implot.get_plot_limits()
            self._xlim = (float(lim.x.min), float(lim.x.max))

    def draw_rtmc(self, height: float) -> None:
        """The motion-correction plot: X/Y/Z totals on the F plot's time
        range (set every frame from it) with the same cursor."""
        from imgui_bundle import implot

        from mbo_utilities.gui.imgui.lines import drag_vline, line, line_plot

        ov = self.overlay
        fit = self._xlim is None
        with line_plot("##line_rtmc_plot", "time (s)", "RTMC (um)", height=height, fit=fit,
                       legend=True) as ok:
            if not ok:
                return
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, 0.0, self.duration_s)
            if self._xlim is not None:
                implot.setup_axis_limits(implot.ImAxis_.x1, *self._xlim, implot.Cond_.always)
            for (axis, t, v), color in zip(self.rtmc, RTMC_COLORS):
                line(f"RTMC {axis}", v, x=t, color=color, weight=1.0)
            cursor, held = drag_vline(98, ov.t_index / self.fs, (1.0, 0.85, 0.3, 0.9), 1.5)
            if held:
                ov.goto_time(cursor)


class LineCuration:
    """vnoiser event curation of the selected line's trace.

    The curation panels (trace with candidates, template / candidate / PCA)
    claim the figure's top edge; this draws the controls under the ROI
    table. The selected ROI's raw trace goes through vnoiser's denoiser the
    first time (cached beside the file under ``.curation/cache``), and the
    focused candidate moves the Reference's Timepoint so the kymograph
    shows that event.
    """

    def __init__(self, widget, overlay: LineScanOverlay, traces: np.ndarray, mesc_path, ref_key: str):
        from mbo_utilities.vnoiser import pf_scan_for_mesc

        self.widget = widget
        self.overlay = overlay
        self.traces = traces
        self.mesc_path = Path(mesc_path)
        self.munit = ref_key.rsplit("/", 1)[-1]
        # the pipeline's processed traces for this scan, when the experiment
        # has a PF folder: what the curation notebook shows, per domain
        # (a group of lines), no denoising needed
        self.pf = pf_scan_for_mesc(self.mesc_path, ref_key)
        self.auto = self.pf is not None
        self.curated: int | None = None
        self.curated_domain: str | None = None
        if self.pf is not None:
            print(f"\nPF traces for scan {self.pf.scan_id}: {', '.join(self.pf.domains)} "
                  f"({self.pf.pf_dir})")
            # the curation shows this scan's domains: the recordings of the
            # unit on screen (another scan is the combo in the panel); the
            # experiment's other scans stay in the catalog but out of view
            scan_tag = f"scan={self.pf.scan_id}"
            widget.scope = lambda rec: scan_tag in rec.rid.split("/")
            # every domain of this scan loads now; a line click then just
            # focuses its domain
            widget.scan(str(self.pf.pf_dir))
            widget.status = "loading every domain; select a line to focus its trace"
        else:
            widget.status = "select a line, then denoise it"
        widget.on_focus = self._on_focus
        # a flip through recordings follows on the line overlay
        widget.on_recording = self._on_recording
        overlay.on_select.append(self._on_select)

    @classmethod
    def build(cls, ndw, overlay, ref_arr, traces, mesc_path, ref_key):
        """The curation widget on ``ndw``'s figure, or None (printed) when
        vnoiser is not installed."""
        import logging
        from types import SimpleNamespace

        from mbo_utilities.gui._availability import HAS_VNOISER
        from mbo_utilities.install import VNOISER_HINT

        if not HAS_VNOISER:
            print(f"\nvnoiser is not installed; no curation panels ({VNOISER_HINT}).")
            return None
        from mbo_utilities.gui.event_curation import EventCurationWidget

        parent = SimpleNamespace(
            image_widget=ndw, logger=logging.getLogger("linescan_viewer"),
            fpath=str(mesc_path),
        )
        widget = EventCurationWidget(parent, data_path="")
        return cls(widget, overlay, traces, mesc_path, ref_key)

    def recording_id(self, i: int) -> str:
        return f"{self.mesc_path.stem}/{self.munit}/roi={int(i)}"

    def _on_recording(self, rid: str) -> None:
        """Select a line of the domain (or the ROI) that was flipped to."""
        if "domain=" in rid and self.pf is not None:
            domain = rid.rsplit("domain=", 1)[-1]
            rois = self.pf.domains.get(domain, [])
            if rois and self.overlay.selected not in rois:
                self.curated_domain = domain
                self.overlay.select_roi(rois[0])
        elif "roi=" in rid:
            try:
                roi = int(rid.rsplit("roi=", 1)[-1])
            except ValueError:
                return
            self.curated = roi
            if roi != self.overlay.selected:
                self.overlay.select_roi(roi)

    def curate(self, i: int) -> None:
        """Curate line ``i``: its PF domain trace when the pipeline ran on
        this scan, else its raw trace through the denoiser."""
        domain = self.pf.domain_for_roi(i) if self.pf is not None else None
        if domain is not None:
            self.curate_domain(domain)
        else:
            self.denoise(i)

    def curate_domain(self, domain: str) -> None:
        """Load the pipeline's processed trace of ``domain`` (the notebook's
        data for this scan) into the curation."""
        if self.pf is None or domain not in self.pf.domains:
            return
        pf_dir = str(self.pf.pf_dir)
        if self.widget.data_path != pf_dir:
            # catalogs the experiment and loads this scan's domains
            self.widget.scan(pf_dir)
        self.curated_domain = domain
        self.curated = None
        self.widget.load(self.pf.recording_id(domain))

    def denoise(self, i: int) -> None:
        """Run the raw trace of line ``i`` through vnoiser's denoiser (or
        restore its cache) and curate it."""
        i = int(i)
        if not 0 <= i < len(self.traces):
            return
        self.curated = i
        self.curated_domain = None
        self.widget.load_trace(
            self.traces[i],
            self.overlay.fs,
            recording_id=self.recording_id(i),
            label=f"{self.munit} ROI {i}",
            source_path=self.mesc_path,
        )

    def _on_select(self, i: int) -> None:
        if not self.auto:
            return
        domain = self.pf.domain_for_roi(i) if self.pf is not None else None
        if domain is not None:
            if domain != self.curated_domain:
                self.curate_domain(domain)
        elif i != self.curated:
            self.denoise(i)

    def _on_focus(self, t_s: float) -> None:
        self.overlay.goto_time(t_s)

    def draw(self) -> None:
        from imgui_bundle import imgui

        from mbo_utilities.gui._theme import section

        section("Curation")
        i = self.overlay.selected
        loading = self.widget.loading
        domain = self.pf.domain_for_roi(i) if self.pf is not None else None
        imgui.begin_disabled(loading)
        if domain is not None:
            rois = ", ".join(str(r) for r in self.pf.domains[domain])
            if imgui.button(f"curate {domain}"):
                self.curate_domain(domain)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"the pipeline's processed trace for {domain} (lines {rois}), "
                    "as the curation notebook shows it"
                )
            imgui.same_line()
        if imgui.button(f"denoise ROI {i}"):
            self.denoise(i)
        imgui.end_disabled()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "run vnoiser's wavelet denoiser on this line's raw trace (minutes the "
                "first time, cached after) and detect candidate events"
            )
        imgui.same_line()
        _changed, self.auto = imgui.checkbox("on select", self.auto)
        if imgui.is_item_hovered():
            imgui.set_tooltip("curate every line as it is selected")
        if self.curated_domain is not None and self.curated_domain != domain:
            imgui.text_disabled(f"showing {self.curated_domain}")
        elif self.curated is not None and self.curated != i:
            imgui.text_disabled(f"showing ROI {self.curated}")
        self.widget.draw_embedded()


def build_overlay(ndw, mesc_path, ref_key, zstack_key, ref_arr, zstack_arr,
                  ref_dims, zstack_dims, *, flip_y: bool, traces, z_index: int = 1,
                  trace_index: int | None = 2, snapshot: dict | None = None,
                  zstack_path=None) -> LineScanOverlay | None:
    """Wire the overlay onto ``ndw``; prints why and returns ``None`` when the
    file lacks the geometry (older MESc, non-AOD unit). ``zstack_path`` is
    the stack's own file when it was saved apart from the line scan."""
    from mbo_utilities.annotation.store import CLASS_COLORS

    stack_path = mesc_path if zstack_path is None else zstack_path
    lines_um = linescan_endpoints_um(mesc_path, ref_key)
    if lines_um is None:
        print("\nno CoordinateMapJSON/driftEndPoints on the Reference unit "
              "-- skipping the line overlay.")
        return None
    vp = viewport_geometry(stack_path, zstack_key)
    if vp is None:
        print("\nno ReferenceViewportJSON on the Z-stack unit -- skipping the line overlay.")
        return None
    depth = zstack_depth_info(stack_path, zstack_key)
    if depth is None:
        print("\nno MinZ/MaxZ/ZDim on the Z-stack unit -- skipping the line overlay.")
        return None

    extents = ref_arr.metadata.get("mesc_roi_extents") or []
    if extents and len(extents) != len(lines_um):
        print(f"\nwarning: {len(lines_um)} driftEndPoints but {len(extents)} ROIs in the "
              "Reference unit; pairing them by order and truncating to the shorter.")
    n = min(len(lines_um), len(extents)) if extents else len(lines_um)
    lines_um = lines_um[:n]
    widths = [int(e["width"]) for e in extents[:n]] if extents else None
    placements = roi_placements(lines_um, depth, widths)
    if traces is not None:
        traces = traces[:n]

    ny, nx = int(zstack_arr.metadata["Ly"]), int(zstack_arr.metadata["Lx"])
    pixel_lines = [um_to_pixels(seg[:2].T, vp, ny, nx, flip_y=flip_y) for seg in lines_um]
    # (K, 4) RGBA array: fastplotlib reads a *list* of exactly four colour
    # tuples as one RGBA colour, so a 4-ROI unit would fail to draw
    colors = np.array(
        [(*CLASS_COLORS[i % len(CLASS_COLORS)][:3], 1.0) for i in range(n)], dtype=np.float32
    )

    snap = None
    if snapshot is not None:
        vp_snap = {"transl": snapshot["bg"]["transl"], "width": snapshot["bg"]["width"],
                   "height": snapshot["bg"]["height"]}
        sny, snx = snapshot["shape"]
        snap = {
            "subplot": snapshot["subplot"],
            "pixel_lines": [um_to_pixels(seg[:2].T, vp_snap, sny, snx, flip_y=flip_y) for seg in lines_um],
            "on_plane": [abs(float(seg[2].mean()) - snapshot["bg"]["z"]) <= 1.0 for seg in lines_um],
        }
    overlay = LineScanOverlay(
        ndw,
        ref_key=ref_key,
        zstack_key=zstack_key,
        ref_dims=ref_dims,
        zstack_dims=zstack_dims,
        placements=placements,
        pixel_lines=pixel_lines,
        colors=colors,
        slice_depths=slice_depths_um(depth),
        fs=float(ref_arr.metadata.get("fs") or 1.0),
        traces=traces,
        z_index=z_index,
        trace_index=trace_index,
        snapshot=snap,
    )

    occupied = slices_with_rois(placements)
    print(f"\n{n} line-scan ROI(s) on {len(occupied)} of {depth['zdim']} Z-stack slices "
          f"({ny}x{nx} px, FOV {vp['width']:.0f}x{vp['height']:.0f} um, "
          f"z {depth['min_z']:+.1f}..{depth['max_z']:+.1f} um):")
    print(f"  {'ROI':>3}  {'slice':>5}  {'z um':>8}  {'off um':>7}  {'len um':>7}  {'um/px':>6}  note")
    for p in placements:
        note = "" if p["in_range"] else "outside stack depth range"
        if p["tilted"]:
            note = (note + "; " if note else "") + "endpoints differ in z"
        spp = "" if p["sample_um"] is None else f"{p['sample_um']:.2f}"
        print(f"  {p['index']:>3}  {p['slice'] + 1:>5}  {p['z_um']:>+8.1f}  {p['dz_um']:>+7.2f}  "
              f"{p['length_um']:>7.1f}  {spp:>6}  {note}")
    if flip_y:
        print("Y mirrored (--flip-y): the verified convention is no mirror; use only if this rig differs.")
    return overlay


def open_linescan_viewer(
    mesc_path,
    *,
    ref_key: str | None = None,
    zstack_key: str | None = None,
    zstack_path=None,
    channel: int = 0,
    flip_y: bool = False,
    traces_dir=None,
    no_traces: bool = False,
    curation: bool = True,
    curate: int | None = None,
    ask: bool = False,
    dry_run: bool = False,
    screenshot=None,
    run_loop: bool = True,
):
    """Open the line-scan + Z-stack viewer on a ``.mesc``.

    ``mbo scan.mesc`` lands here when the unit picked in the dialog is an
    AOD line scan. With ``ask`` the missing keys are asked for on the
    console (the script's behaviour); without it the first line-scan unit
    and the paired Z-stack are taken and printed. The stack may live in a
    sibling ``<name>_zstack.mesc`` or the file given as ``zstack_path``.

    Returns the ``NDWidget`` once it is shown (None when nothing opened);
    ``run_loop`` runs the event loop until the window closes.
    """
    from mbo_utilities.arrays.mesc import MescArray, list_mesc_units
    from mbo_utilities.analysis.linescan import (
        _background_planes, _composite, background_image, pair_reference_zstack, zstack_candidates,
    )

    mesc_path = Path(mesc_path)
    units = list_mesc_units(mesc_path)
    linescan_units = [u for u in units if u["kind"] == "packed"]
    if not linescan_units:
        raise SystemExit(f"no linescan units found in {mesc_path}")
    # the Z-stack may live in its own file (Asako's rig saves
    # <expt>_zstack.mesc beside the line scan): look there when told, else
    # beside the line scan for a sibling named that way, else in the file
    zstack_path = None if zstack_path is None else Path(zstack_path)
    if zstack_path is None:
        sibling = mesc_path.parent.parent / f"{mesc_path.stem}_zstack.mesc"
        if not any(u["modality_name"] == "zstack" for u in units) and sibling.exists():
            zstack_path = sibling
    if zstack_path is None:
        zstack_path = mesc_path
        stack_units = units
    else:
        stack_units = list_mesc_units(zstack_path)
    zstack_units = [u for u in stack_units if u["modality_name"] == "zstack"]
    if not zstack_units:
        raise SystemExit(f"no zstack units found in {zstack_path}")

    print(f"{mesc_path.name}: {len(units)} unit(s), {len(linescan_units)} linescan; "
          f"{len(zstack_units)} zstack in {zstack_path.name}.")

    def _resolve(key, pool, label, extra=None, default=None):
        if key is None:
            if not ask:
                key = default if default is not None else default_linescan_unit(mesc_path, pool)
                print(f"{label}: {key}")
            else:
                return _console_pick_unit(pool, label, extra=extra, default=default)
        keys = {u["key"] for u in pool} | {u["munit"] for u in pool}
        if key not in keys:
            raise SystemExit(f"{label}: {key!r} is not one of {sorted(keys)}")
        return next(u["key"] for u in pool if key in (u["key"], u["munit"]))

    ref_key = _resolve(ref_key, linescan_units, "Reference (linescan)")
    if ref_key is None:
        print("cancelled.")
        return None

    # score every stack against these lines; the picker shows the fit and
    # defaults to the paired one, and a stack holding none of the lines is
    # refused - drawing them on it would be meaningless
    cands = {
        c["key"]: c
        for c in zstack_candidates(mesc_path, ref_key, stack_units, zstack_path=zstack_path)
    }
    paired = pair_reference_zstack(mesc_path, ref_key, stack_units, zstack_path=zstack_path)
    fit = {}
    for u in zstack_units:
        c = cands.get(u["key"])
        if c is None:
            fit[u["key"]] = "no geometry"
        else:
            fit[u["key"]] = (f"xy {c['xy_fraction']:3.0%} z {c['z_fraction']:3.0%} "
                             f"{c['um_per_px']:.2f}um/px{' COARSE' if c['coarse'] else ''}")
    zstack_key = _resolve(zstack_key, zstack_units, "Z-stack", extra=fit,
                          default=paired["key"] if paired else None)
    if zstack_key is None:
        print("cancelled.")
        return None
    chosen = cands.get(zstack_key)
    if chosen is None or chosen["xy_fraction"] == 0:
        raise SystemExit(
            f"{zstack_key.rsplit('/', 1)[-1]} holds none of {ref_key.rsplit('/', 1)[-1]}'s lines in its "
            f"field; nothing to overlay. Paired stack: "
            f"{paired['munit'] if paired else 'none in this file'}."
        )
    if chosen["coarse"]:
        print(f"warning: {chosen['munit']} is {chosen['um_per_px']:.2f} um/px, under 10 px per line; "
              "the overlay will be blobs.")
    if chosen["xy_fraction"] < 1:
        print(f"warning: only {chosen['xy_fraction']:.0%} of the lines fall inside {chosen['munit']}'s field.")

    ref_arr = MescArray(mesc_path, unit=ref_key)
    zstack_arr = MescArray(zstack_path, unit=zstack_key)
    _print_metadata("Reference", ref_arr)
    _print_metadata("Z-stack", zstack_arr)

    bg = background_image(mesc_path, ref_key)
    if bg is not None:
        print(f"\nSnapshot (drawn on): {bg['munit']}  {bg['width']:.0f}x{bg['height']:.0f} um, "
              f"{bg['shape'][-1]} px, z {bg['z']:.2f}")
    else:
        print("\nno BackgroundImagePath on this unit: no snapshot panel.")

    if dry_run:
        from mbo_utilities.arrays.mesc_geometry import roi_placements, zstack_depth_info

        placements = roi_placements(
            linescan_endpoints_um(mesc_path, ref_key), zstack_depth_info(zstack_path, zstack_key),
            [int(e["width"]) for e in ref_arr.metadata["mesc_roi_extents"]],
        )
        print(f"\n{len(placements)} line(s) on {chosen['munit']}:")
        print(f"  {'ROI':>3}  {'slice':>5}  {'z um':>8}  {'off um':>7}  {'snap dz':>8}  note")
        for p_ in placements:
            seg = linescan_endpoints_um(mesc_path, ref_key)[p_["index"]]
            dz_snap = (float(seg[2].mean()) - bg["z"]) if bg else float("nan")
            note = "" if p_["in_range"] else "outside stack depth range"
            print(f"  {p_['index']:>3}  {p_['slice'] + 1:>5}  {p_['z_um']:>+8.1f}  {p_['dz_um']:>+7.2f}  "
                  f"{dz_snap:>+8.1f}  {note}")
        return None

    from mbo_utilities.gui.run_gui import _after_show, _figure_kwargs_for_here, _squeeze_for_viewer
    from mbo_utilities.gui._ndviewer import _ROW, _COL, _ref_to_index, sliders_height
    from mbo_utilities.gui._top_strip import strip_height
    from mbo_utilities.gui.event_curation import PANEL_HEIGHT as CURATION_PANEL_HEIGHT
    from fastplotlib.widgets.nd_widget import NDWidget

    job = None
    if not no_traces:
        from mbo_utilities.preferences import get_linescan_auto_traces

        print()
        job = TraceJob(ref_arr, channel, traces_dir, auto=get_linescan_auto_traces())

    ref_view = _squeeze_for_viewer(ref_arr)
    zstack_view = _squeeze_for_viewer(zstack_arr)
    ref_dims, ref_ranges = _panel_dims(ref_arr, ref_view, "Reference")
    zstack_dims, zstack_ranges = _panel_dims(zstack_arr, zstack_view, "Z-stack")

    # the traces are an imgui panel on the top strip, so the image panels
    # take the whole canvas
    names = [f"Reference [{ref_key.rsplit('/', 1)[-1]}]"]
    if bg is not None:
        names.append(f"Snapshot [{bg['munit']}]")
    names.append(f"Z-stack [{zstack_key.rsplit('/', 1)[-1]}]")
    n_top = len(names)
    # --screenshot renders offscreen and reads the frame back, the same route
    # scripts/capture_docs.py uses: no window, no event loop, works headless
    if screenshot is not None:
        figure_kwargs = {"canvas": "offscreen", "size": (1500, 950)}
    else:
        # the strip holds the curation panel, else the raw traces tab
        # sized for the panel the traces will bring, even while they compute
        rtmc = bool(ref_arr.metadata.get("mesc_rtmc"))
        panel = (0 if job is None else CURATION_PANEL_HEIGHT if curation
                 else TRACES_RTMC_PANEL_HEIGHT if rtmc else TRACES_PANEL_HEIGHT)
        figure_kwargs = _figure_kwargs_for_here(
            fit=dict(
                image_hw=ref_view.shape[-2:],
                grid=(1, n_top),
                top=strip_height(panel) if panel else 0,
                bottom=sliders_height(len(ref_ranges) + len(zstack_ranges)),
                right=LINE_PANEL_WIDTH,
            )
        )
    extents = [(i / n_top, (i + 1) / n_top, 0.0, 1.0) for i in range(n_top)]
    snap_index = 1 if bg is not None else None
    z_index = n_top - 1
    trace_index = None

    ndw = NDWidget(
        ranges={**ref_ranges, **zstack_ranges},
        extents=extents,
        names=names,
        controller_ids=None,  # independent controller per subplot
        **figure_kwargs,
    )

    spatial = (_ROW, _COL)
    vmin, vmax = _limits(ref_arr[: min(2000, ref_arr.shape[0]), channel])
    ndw[0].add_nd_image(
        data=ref_view, dims=ref_dims + spatial, display_dims=spatial,
        compute_histogram=True, name="Reference",
        slider_maps={d: _ref_to_index for d in ref_dims},
        graphic_kwargs={"vmin": vmin, "vmax": vmax},
    )
    # limits from the channel shown first (channel 0): the red channel's soma
    # would set a ceiling that leaves a green dendrite nearly black
    vmin, vmax = _limits(zstack_arr[0, 0, int(zstack_arr.shape[2]) // 2])
    zstack_ndg = ndw[z_index].add_nd_image(
        data=zstack_view, dims=zstack_dims + spatial, display_dims=spatial,
        compute_histogram=True, name="Z-stack",
        slider_maps={d: _ref_to_index for d in zstack_dims},
        graphic_kwargs={"vmin": vmin, "vmax": vmax},
    )

    snapshot = None
    if bg is not None:
        planes = _background_planes(mesc_path, bg)
        rgb = _composite(planes, ref_arr.metadata.get("channel_names") or []).astype(np.float32)
        snap_sp = ndw[snap_index].subplot
        snap_sp.add_image(rgb, name="snapshot")
        snap_sp.camera.maintain_aspect = True
        snapshot = {"subplot": snap_sp, "bg": bg, "shape": planes[0].shape}

    overlay = build_overlay(
        ndw, mesc_path, ref_key, zstack_key, ref_arr, zstack_arr,
        ref_dims, zstack_dims, flip_y=flip_y, traces=None,
        z_index=z_index, trace_index=trace_index, snapshot=snapshot,
        zstack_path=zstack_path,
    )
    line_curation = None
    traces_panel = None
    if overlay is not None:
        if job is not None and screenshot is not None:
            # a screenshot needs the panels in the frame, so wait here;
            # otherwise the panels attach when the thread finishes
            job.start()
            job.wait()
        if job is not None and job.done and job.result is not None:
            traces = job.result[: overlay.n]
            from mbo_utilities.gui._top_strip import TopStrip

            if curation:
                line_curation = LineCuration.build(ndw, overlay, ref_arr, traces, mesc_path, ref_key)
            # the raw trace and the RTMC curves get a tab beside Curation on
            # the same strip, or their own strip without curation
            strip = TopStrip(ndw.figure) if line_curation is None else line_curation.widget.strip
            traces_panel = LineTracesPanel(ndw, overlay, traces, strip, line_curation is None,
                                           curves=ref_arr.curves)

        def switch(key: str) -> None:
            # a new window for the other scan on the running loop, then this
            # one goes away; the loop keeps running for the new window
            open_linescan_viewer(
                mesc_path, ref_key=key, zstack_key=zstack_key, zstack_path=zstack_path,
                channel=channel, flip_y=flip_y, traces_dir=None, no_traces=no_traces,
                curation=curation, run_loop=False,
            )
            _close_figure(ndw.figure)

        pending = job is not None and not job.done
        line_panel = LinePanel(ndw, overlay, curation=line_curation, units=linescan_units,
                               switch=switch, job=job if pending else None)
        if pending:
            ndw.linescan_trace_attach = TraceAttach(
                ndw, job, overlay, line_panel, ref_arr, mesc_path, ref_key, curation, curate
            )
    # keep the overlay and panels alive with the widget
    ndw.linescan_overlay = overlay
    ndw.linescan_curation = line_curation
    ndw.linescan_traces = traces_panel
    ndw.linescan_trace_job = job

    ndw.show()
    _after_show(ndw)
    if line_curation is not None and curate is not None:
        line_curation.curate(curate)

    if screenshot is not None:
        import imageio.v3 as iio

        if line_curation is not None and curate is not None:
            print("waiting for the denoiser...", flush=True)
            line_curation.widget.wait()
        for _ in range(10):
            ndw.figure.canvas.draw()
        # NDWidget fetches slices through the event loop, which never runs
        # here; put the current slice on the Z-stack graphic by hand so the
        # frame shows the state an interactive session reaches
        if overlay is not None:
            zstack_ndg.graphic.data[:] = np.asarray(zstack_view[0, overlay.slice], dtype=np.float32)
        frame = None
        for _ in range(20):
            frame = ndw.figure.canvas.draw()
        iio.imwrite(str(screenshot), np.asarray(frame)[..., :3])
        print(f"screenshot -> {screenshot}", flush=True)
        return ndw

    if run_loop:
        import fastplotlib as fpl

        fpl.loop.run()
    return ndw


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesc_path", nargs="?", type=Path)
    ap.add_argument("--ref", help="linescan unit key, e.g. MSession_0/MUnit_3 (skips the prompt)")
    ap.add_argument("--zstack", help="zstack unit key (skips the prompt)")
    ap.add_argument("--zstack-file", type=Path, default=None,
                    help="the .mesc holding the Z-stack when it was saved separately from "
                         "the line scan (default: look in mesc_path)")
    ap.add_argument("--no-curation", action="store_true",
                    help="skip the vnoiser event-curation panels even when vnoiser is installed")
    ap.add_argument("--curate", type=int, default=None,
                    help="denoise and curate this ROI as soon as the window opens")
    ap.add_argument("--channel", type=int, default=0, help="channel for the traces panel")
    ap.add_argument("--flip-y", action="store_true", help="mirror the lines vertically")
    ap.add_argument("--no-traces", action="store_true", help="skip the per-ROI trace panel")
    ap.add_argument("--traces", type=Path, default=None,
                    help="an `mbo linescan` output dir (holds F.npy) to plot instead of "
                         "recomputing; default looks for rois_linescan/<MUnit_n>/ beside the file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the unit choice, stack fit and line placement, then exit without a window")
    ap.add_argument("--screenshot", type=Path, default=None,
                    help="export the window to this PNG a few seconds after it opens, then exit")
    args = ap.parse_args(argv)

    mesc_path = args.mesc_path
    if mesc_path is None:
        from mbo_utilities.gui.run_gui import _select_file

        selected, *_ = _select_file()
        if isinstance(selected, list):
            selected = selected[0] if selected else None
        if not selected:
            print("no file selected.")
            return
        mesc_path = Path(selected)

    open_linescan_viewer(
        mesc_path, ref_key=args.ref, zstack_key=args.zstack, zstack_path=args.zstack_file,
        channel=args.channel, flip_y=args.flip_y, traces_dir=args.traces,
        no_traces=args.no_traces, curation=not args.no_curation, curate=args.curate,
        ask=True, dry_run=args.dry_run, screenshot=args.screenshot,
    )


if __name__ == "__main__":
    main()
