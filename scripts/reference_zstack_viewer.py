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
    python scripts/reference_zstack_viewer.py [mesc_path] [--ref MUnit_x]
        [--zstack MUnit_y] [--channel 0] [--flip-y] [--no-traces]
        [--traces rois_linescan/MUnit_x] [--dry-run] [--screenshot out.png]

``--dry-run`` prints the unit choice, every stack's fit and the line
placement without opening a window; ``--screenshot`` renders the window
offscreen to a PNG and exits (no display needed).
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pygfx

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


def _console_pick_unit(
    units: list[dict], label: str, *, extra: dict[str, str] | None = None, default: str | None = None,
) -> str | None:
    """Print a table of ``units`` and read a chosen index from stdin.

    Mirrors the columns of run_gui.py's Qt unit picker (``_prompt_for_mesc_unit``)
    rendered to the terminal: that dialog needs a Qt binding this install may
    not have (`pyqt6` is a base dependency only on Linux, see `pyproject.toml`).
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


def _load_or_compute_traces(ref_arr, channel: int, traces_dir: Path | None) -> np.ndarray:
    """``(K, T)`` per-ROI traces: ``F.npy`` from a ``mbo linescan`` output dir
    when one is given or found beside the file, else computed now with the
    same reduction (``roi_workflow.linescan_roi_means``)."""
    from mbo_utilities.roi_workflow import linescan_roi_means

    if traces_dir is None:
        munit = ref_arr.metadata["mesc_unit"].rsplit("/", 1)[-1]
        candidate = Path(ref_arr.source_path).parent / "rois_linescan" / munit
        if (candidate / "F.npy").exists():
            traces_dir = candidate
    if traces_dir is not None:
        F = np.load(Path(traces_dir) / "F.npy")
        print(f"traces: loaded {F.shape} from {traces_dir}")
        return F.astype(np.float32, copy=False)

    def _progress(i, k, seconds):
        print(f"  traces: ROI {i + 1}/{k} done ({seconds:.1f}s)")

    print(f"computing per-ROI traces ({ref_arr.shape[0]} timepoints, channel {channel}); "
          "run `mbo linescan` once and pass --traces to skip this next time...")
    return linescan_roi_means(ref_arr, channel=channel, progress=_progress)


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
        self._busy = False

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
                snap_starts, colors=[pygfx.Color(*c).hex for c in colors], sizes=START_DOT_SIZE,
                uniform_size=True, name="snapshot_starts",
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
        # hex strings: fastplotlib reads a colour *array* with exactly 3 or 4
        # rows as one RGB(A) colour, so a 3- or 4-ROI unit would fail to draw
        self.starts = z_sp.add_scatter(
            starts, colors=[pygfx.Color(*c).hex for c in colors], sizes=START_DOT_SIZE,
            uniform_size=True, name="line_starts",
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
            separation=TRACE_SEPARATION, name="roi_traces",
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
        self.selected = int(i)
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
        if self.t_dim is not None and self.selector is not None and not self._busy:
            t = self._ref_to_index(indices[self.t_dim])
            self._busy = True
            try:
                self.selector.selection = t / self.fs
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


class LinePanel:
    """Right-hand imgui panel: one row per line, depth navigation, toggles."""

    def __init__(self, ndw, overlay: LineScanOverlay, size: int = 360):
        from mbo_utilities.gui._edge_window import EdgeWindow

        self.overlay = overlay

        class _Window(EdgeWindow):
            def update(win_self):  # noqa: N805
                self.draw()

        self.window = _Window(ndw.figure, size=size, location="right", title=None)

    def draw(self) -> None:
        from imgui_bundle import imgui

        ov = self.overlay
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


def build_overlay(ndw, mesc_path, ref_key, zstack_key, ref_arr, zstack_arr,
                  ref_dims, zstack_dims, *, flip_y: bool, traces, z_index: int = 1,
                  trace_index: int | None = 2, snapshot: dict | None = None) -> LineScanOverlay | None:
    """Wire the overlay onto ``ndw``; prints why and returns ``None`` when the
    file lacks the geometry (older MESc, non-AOD unit)."""
    from mbo_utilities.annotation.store import CLASS_COLORS

    lines_um = linescan_endpoints_um(mesc_path, ref_key)
    if lines_um is None:
        print("\nno CoordinateMapJSON/driftEndPoints on the Reference unit "
              "-- skipping the line overlay.")
        return None
    vp = viewport_geometry(mesc_path, zstack_key)
    if vp is None:
        print("\nno ReferenceViewportJSON on the Z-stack unit -- skipping the line overlay.")
        return None
    depth = zstack_depth_info(mesc_path, zstack_key)
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesc_path", nargs="?", type=Path)
    ap.add_argument("--ref", help="linescan unit key, e.g. MSession_0/MUnit_3 (skips the prompt)")
    ap.add_argument("--zstack", help="zstack unit key (skips the prompt)")
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

    from mbo_utilities.arrays.mesc import MescArray, list_mesc_units
    from mbo_utilities.analysis.linescan import (
        _background_planes, _composite, background_image, pair_reference_zstack, zstack_candidates,
    )

    units = list_mesc_units(mesc_path)
    linescan_units = [u for u in units if u["kind"] == "packed"]
    zstack_units = [u for u in units if u["modality_name"] == "zstack"]
    if not linescan_units:
        raise SystemExit(f"no linescan units found in {mesc_path}")
    if not zstack_units:
        raise SystemExit(f"no zstack units found in {mesc_path}")

    print(f"{mesc_path.name}: {len(units)} unit(s), "
          f"{len(linescan_units)} linescan, {len(zstack_units)} zstack.")

    def _resolve(key, pool, label, extra=None, default=None):
        if key is None:
            return _console_pick_unit(pool, label, extra=extra, default=default)
        keys = {u["key"] for u in pool} | {u["munit"] for u in pool}
        if key not in keys:
            raise SystemExit(f"{label}: {key!r} is not one of {sorted(keys)}")
        return next(u["key"] for u in pool if key in (u["key"], u["munit"]))

    ref_key = _resolve(args.ref, linescan_units, "Reference (linescan)")
    if ref_key is None:
        print("cancelled.")
        return

    # score every stack against these lines; the picker shows the fit and
    # defaults to the paired one, and a stack holding none of the lines is
    # refused - drawing them on it would be meaningless
    cands = {c["key"]: c for c in zstack_candidates(mesc_path, ref_key, units)}
    paired = pair_reference_zstack(mesc_path, ref_key, units)
    fit = {}
    for u in zstack_units:
        c = cands.get(u["key"])
        if c is None:
            fit[u["key"]] = "no geometry"
        else:
            fit[u["key"]] = (f"xy {c['xy_fraction']:3.0%} z {c['z_fraction']:3.0%} "
                             f"{c['um_per_px']:.2f}um/px{' COARSE' if c['coarse'] else ''}")
    zstack_key = _resolve(args.zstack, zstack_units, "Z-stack", extra=fit,
                          default=paired["key"] if paired else None)
    if zstack_key is None:
        print("cancelled.")
        return
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
    zstack_arr = MescArray(mesc_path, unit=zstack_key)
    _print_metadata("Reference", ref_arr)
    _print_metadata("Z-stack", zstack_arr)

    bg = background_image(mesc_path, ref_key)
    if bg is not None:
        print(f"\nSnapshot (drawn on): {bg['munit']}  {bg['width']:.0f}x{bg['height']:.0f} um, "
              f"{bg['shape'][-1]} px, z {bg['z']:.2f}")
    else:
        print("\nno BackgroundImagePath on this unit: no snapshot panel.")

    if args.dry_run:
        from mbo_utilities.arrays.mesc_geometry import roi_placements, zstack_depth_info

        placements = roi_placements(
            linescan_endpoints_um(mesc_path, ref_key), zstack_depth_info(mesc_path, zstack_key),
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
        return

    from mbo_utilities.gui.run_gui import _after_show, _figure_kwargs_for_here, _squeeze_for_viewer
    from mbo_utilities.gui._ndviewer import _ROW, _COL, _ref_to_index
    from fastplotlib.widgets.nd_widget import NDWidget

    traces = None
    if not args.no_traces:
        print()
        traces = _load_or_compute_traces(ref_arr, args.channel, args.traces)

    ref_view = _squeeze_for_viewer(ref_arr)
    zstack_view = _squeeze_for_viewer(zstack_arr)
    ref_dims, ref_ranges = _panel_dims(ref_arr, ref_view, "Reference")
    zstack_dims, zstack_ranges = _panel_dims(zstack_arr, zstack_view, "Z-stack")

    # --screenshot renders offscreen and reads the frame back, the same route
    # scripts/capture_docs.py uses: no window, no event loop, works headless
    figure_kwargs = ({"canvas": "offscreen", "size": (1500, 950)} if args.screenshot is not None
                     else _figure_kwargs_for_here())
    top = 0.68 if traces is not None else 1.0
    names = [f"Reference [{ref_key.rsplit('/', 1)[-1]}]"]
    if bg is not None:
        names.append(f"Snapshot [{bg['munit']}]")
    names.append(f"Z-stack [{zstack_key.rsplit('/', 1)[-1]}]")
    n_top = len(names)
    extents = [(i / n_top, (i + 1) / n_top, 0.0, top) for i in range(n_top)]
    snap_index = 1 if bg is not None else None
    z_index = n_top - 1
    trace_index = None
    if traces is not None:
        extents.append((0.0, 1.0, top, 1.0))
        names.append("Traces")
        trace_index = n_top

    ndw = NDWidget(
        ref_ranges={**ref_ranges, **zstack_ranges},
        extents=extents,
        names=names,
        controller_ids=None,  # independent controller per subplot
        **figure_kwargs,
    )

    def _set_contrast(ndg, vmin: float, vmax: float) -> None:
        # NDImage doesn't take vmin/vmax at construction - set on its
        # histogram widget when present, else the graphic itself, as
        # MboNDViewer._style_graphic does.
        cb = ndg.histogram_widget
        target = cb if cb is not None else ndg.graphic
        target.vmax = float(vmax)
        target.vmin = float(vmin)

    spatial = (_ROW, _COL)
    ref_ndg = ndw[0].add_nd_image(
        data=ref_view, dims=ref_dims + spatial, spatial_dims=spatial,
        compute_histogram=True, name="Reference",
        slider_dim_transforms={d: _ref_to_index for d in ref_dims},
    )
    def _limits(sample) -> tuple[float, float]:
        v = np.asarray(sample, dtype=np.float32).ravel()
        lo, hi = np.percentile(v, (0.5, 99.9))
        return float(lo), float(hi if hi > lo else lo + 1)

    _set_contrast(ref_ndg, *_limits(ref_arr[: min(2000, ref_arr.shape[0]), args.channel]))
    zstack_ndg = ndw[z_index].add_nd_image(
        data=zstack_view, dims=zstack_dims + spatial, spatial_dims=spatial,
        compute_histogram=True, name="Z-stack",
        slider_dim_transforms={d: _ref_to_index for d in zstack_dims},
    )
    # limits from the channel shown first (channel 0): the red channel's soma
    # would set a ceiling that leaves a green dendrite nearly black
    _set_contrast(zstack_ndg, *_limits(zstack_arr[0, 0, int(zstack_arr.shape[2]) // 2]))

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
        ref_dims, zstack_dims, flip_y=args.flip_y, traces=traces,
        z_index=z_index, trace_index=trace_index, snapshot=snapshot,
    )
    if overlay is not None:
        LinePanel(ndw, overlay)

    ndw.show()
    _after_show(ndw)

    import fastplotlib as fpl

    if args.screenshot is not None:
        import imageio.v3 as iio

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
        iio.imwrite(str(args.screenshot), np.asarray(frame)[..., :3])
        print(f"screenshot -> {args.screenshot}", flush=True)
        return

    fpl.loop.run()


if __name__ == "__main__":
    main()
