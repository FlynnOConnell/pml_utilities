"""Run orchestration and derived-set state for the manual ROI widget.

``RoiRunManager`` tracks the widget's background work: in-process runs
(extract / demix / discover on a thread, shown as process-manager jobs)
and full-plane suite2p / masknmf spawns (detached subprocesses polled
through their progress files). GPU-heavy runs serialize on one lock so two
demixes never fight over the device.

``DerivedSet`` wraps one pixel unit of a results file
(:class:`~mbo_utilities.results.ResultUnit`: a suite2p plane, a masknmf run,
a manual run's slice) for display - its footprints in the frame, per-set
color, visibility, discarded rows, accept / reject state and a pick map.
Traces live in the session model's :class:`~mbo_utilities.annotation.RoiTraceTable`,
keyed by store uid so deleting an ROI never remaps anyone else's rows;
:func:`unit_trace` turns one unit row into such a row.
Everything here is figure-free and imports without masknmf, so it is
unit-testable with a stub process manager.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.annotation import ROI_COLORS, RoiTrace, class_color
from mbo_utilities.arrays.features._dim_tags import filename_tags
from mbo_utilities.gui.widgets.process_manager import LocalJob, get_process_manager
from mbo_utilities.results import ResultUnit

__all__ = [
    "DerivedSet",
    "MASK_MODES",
    "RING_SCALE",
    "RoiRun",
    "RoiRunManager",
    "SELECTED_ALPHA",
    "SET_COLORS",
    "build_pick_map",
    "component_color",
    "derived_comps",
    "derived_outline",
    "derived_rgba",
    "feathered_rgba",
    "footprint_center",
    "footprint_edges",
    "footprint_radius",
    "outline_data",
    "outline_paths",
    "ring",
    "set_color",
    "unit_trace",
]

SELECTED_ALPHA = 0.9  # matches the manual overlay's selection opacity

# per-set overlay hues, bright against gnuplot2 and the class colors
SET_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.85, 0.10),
    (0.10, 0.90, 0.90),
    (1.00, 0.45, 0.85),
    (0.45, 1.00, 0.35),
    (1.00, 0.55, 0.15),
    (0.55, 0.55, 1.00),
)


def set_color(index: int) -> tuple[float, float, float]:
    """Rgb in 0-1 for a derived set (wraps past the palette end)."""
    return SET_COLORS[index % len(SET_COLORS)]


@dataclass
class RoiRun:
    """One background run: an in-process job (``job``) or a spawn (``pid``)."""

    kind: str  # "extract" | "demix" | "suite2p" | "masknmf"
    tag: str
    description: str
    job: LocalJob | None = None
    pid: int | None = None
    out_root: Path | None = None
    out_dirs: list[Path] = field(default_factory=list)
    planes: list[int] = field(default_factory=list)
    box: tuple[int, int, int, int] | None = None
    error: str | None = None
    finished: bool = False
    loaded: bool = False


def _raised_at(error: BaseException) -> str:
    """`` (file.py:123)`` for the innermost frame of ``error``, or ""."""
    import traceback

    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        return ""
    last = frames[-1]
    return f" ({Path(last.filename).name}:{last.lineno})"


class RoiRunManager:
    """Submit, poll and stop the ROI widget's background runs.

    In-process runs go through :meth:`submit` - a ``LocalJob`` in the
    process console plus a daemon thread; ``heavy=True`` runs queue on one
    lock so the GPU is never shared. Full-plane pipelines go through
    :meth:`spawn`. :meth:`poll` (once per frame) yields each finished run
    exactly once.

    Parameters
    ----------
    pm : optional
        Process manager for ``submit`` / ``spawn``; the global one when
        None. Tests inject a stub here and into ``poll``.
    """

    def __init__(self, pm=None):
        self._pm = pm
        self.runs: list[RoiRun] = []
        self._results: queue.Queue = queue.Queue()
        self._heavy = threading.Lock()
        self.logger = log.get("gui.roi_runs")

    def submit(self, run: RoiRun, fn, *, heavy: bool = False) -> RoiRun:
        """Run ``fn(job)`` on a daemon thread; its return value reaches
        :meth:`poll` as the run's payload (None and ``run.error`` set when
        it raised).
        """
        pm = self._pm or get_process_manager()
        run.job = pm.start_job(run.kind, run.description)

        def work():
            job = run.job
            try:
                if heavy:
                    job.set_progress(0.0, "waiting for gpu")
                    with self._heavy:
                        payload = fn(job)
                else:
                    payload = fn(job)
            except Exception as error:  # noqa: BLE001 - reported on the job
                self.logger.exception(f"{run.description} failed")
                message = f"{type(error).__name__}: {error}{_raised_at(error)}"
                job.fail(message)
                self._results.put((run, None, message))
                return
            job.done()
            self._results.put((run, payload, None))

        thread = threading.Thread(target=work, name=f"roi-run-{run.tag}", daemon=True)
        thread.start()
        self.runs.append(run)
        return run

    def spawn(self, run: RoiRun, task_type: str, args: dict) -> RoiRun:
        """Start a detached worker for ``run``; progress arrives through the
        process manager's sidecar polling.
        """
        pm = self._pm or get_process_manager()
        if run.out_root is None and args.get("output_dir"):
            run.out_root = Path(args["output_dir"])
        pid = pm.spawn(
            task_type,
            args,
            run.description,
            output_path=str(run.out_root) if run.out_root is not None else None,
        )
        if pid is None:
            self._results.put((run, None, "failed to start worker"))
        else:
            run.pid = pid
        self.runs.append(run)
        return run

    def poll(self, pm) -> list[tuple[RoiRun, object]]:
        """``(run, payload)`` per newly finished run; errors land on
        ``run.error`` (spawned runs always carry a None payload).
        """
        done: list[tuple[RoiRun, object]] = []
        while True:
            try:
                run, payload, error = self._results.get_nowait()
            except queue.Empty:
                break
            run.finished = True
            run.error = error
            done.append((run, payload))
        spawned = [r for r in self.runs if r.pid is not None and not r.finished]
        if spawned:
            by_pid = {p.pid: p for p in pm.get_running()}
            for run in spawned:
                info = by_pid.get(run.pid)
                if info is None:
                    # a completed entry is pruned from the console after a
                    # few minutes; outputs on disk mean the run succeeded
                    if run.out_root is None or not finished_dirs(
                        run.out_root, run.planes
                    ):
                        run.error = "process stopped"
                elif info.status == "completed":
                    pass
                elif info.status == "error":
                    run.error = info.status_message or "process failed"
                else:
                    continue
                run.finished = True
                done.append((run, None))
        return done

    def stop(self, run: RoiRun, pm) -> None:
        """Kill a spawned run; in-process runs cannot be cancelled."""
        if run.pid is not None and not run.finished:
            pm.kill(run.pid)

    @property
    def active(self) -> list[RoiRun]:
        return [r for r in self.runs if not r.finished]

    @property
    def busy(self) -> bool:
        return bool(self.active)


def build_pick_map(footprints, shape: tuple[int, int]) -> np.ndarray:
    """``(Y, X)`` int32 row index per pixel, -1 background.

    ``footprints`` are ``(ypix, xpix, lam)`` per row. Rows are painted in
    ascending peak-lam order, so the strongest component wins contested
    pixels - picking matches what the overlay shows on top.
    """
    pick = np.full(shape, -1, np.int32)
    peaks = [float(np.max(lam)) if len(lam) else 0.0 for _y, _x, lam in footprints]
    for k in np.argsort(peaks, kind="stable"):
        ypix, xpix, _lam = footprints[k]
        pick[ypix, xpix] = k
    return pick


@dataclass(eq=False)
class DerivedSet:
    """One loaded unit's components, as shown: overlay color, visibility,
    discarded rows, accepted flags, per-row class labels and the pick map.
    Promoted rows are never stored here - the widget recomputes them from
    the store's ``source`` strings.

    ``path`` is the set's identity: the plane dir the unit was read from,
    or ``<file>.zarr/<unit>``. ``z`` is the plane of the array shown the
    footprints sit on. ``accepted`` starts from the unit's ``iscell`` and
    is the mutable curation state (kept in the rois zarr, never written
    into the run); ``classes`` maps a row to a class index in the shared
    label set; ``colors`` maps a row to an explicit group color (float rgb
    in 0-1) that wins over the class / hue color.
    """

    unit: ResultUnit
    path: Path
    name: str
    color: tuple[float, float, float]
    z: int = 0
    kind: str = "suite2p"
    algo: str = ""
    visible: bool = True
    discarded: set[int] = field(default_factory=set)
    classes: dict[int, int] = field(default_factory=dict)
    colors: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    accepted: np.ndarray | None = None
    footprints: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=list
    )
    pick_map: np.ndarray | None = None

    def __post_init__(self):
        unit = self.unit
        if unit.member_kind != "pixel" or unit.image_shape is None:
            raise ValueError(
                f"{unit.name}: only pixel units with an image shape load as ROIs"
            )
        self.path = Path(self.path)
        ly, lx = self.shape
        if not self.footprints:
            weights = (
                unit.weights if unit.weights is not None else [None] * len(unit.members)
            )
            for members, lam in zip(unit.members, weights, strict=True):
                ypix, xpix = np.divmod(np.asarray(members, np.int64).ravel(), lx)
                self.footprints.append(
                    (
                        ypix.astype(np.int32),
                        xpix.astype(np.int32),
                        np.ones(ypix.size, np.float32)
                        if lam is None
                        else np.asarray(lam, np.float32),
                    )
                )
        if self.pick_map is None:
            self.pick_map = build_pick_map(self.footprints, (ly, lx))
        if self.accepted is None:
            iscell = unit.iscell
            if iscell is not None and len(iscell) == len(self.footprints):
                self.accepted = np.asarray(iscell)[:, 0] > 0
            else:
                self.accepted = np.ones(len(self.footprints), bool)

    def __len__(self) -> int:
        return len(self.footprints)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.unit.image_shape[0]), int(self.unit.image_shape[1]))

    @property
    def engine(self) -> str:
        """The engine a manual run used, else the pipeline's name."""
        return str(self.unit.attrs.get("engine") or self.kind)

    @property
    def read_z(self) -> int | None:
        z = self.unit.attrs.get("z") if "engine" in self.unit.attrs else None
        return None if z is None else int(z)

    @property
    def read_c(self) -> int | None:
        c = self.unit.attrs.get("c") if "engine" in self.unit.attrs else None
        return None if c is None else int(c)

    @property
    def frames(self) -> tuple[int, int, int] | None:
        frames = self.unit.attrs.get("frames")
        return None if not frames else tuple(int(v) for v in frames)

    @property
    def tp_indices(self) -> list[int] | None:
        found = self.unit.attrs.get("tp_indices")
        return None if not found else [int(t) for t in found]

    @property
    def uids(self) -> list[int] | None:
        """The store uids a manual run's rows stand for, when every row has one."""
        uids = self.unit.attrs.get("uids")
        if not uids or len(uids) != len(self.footprints):
            return None
        return [int(u) for u in uids]

    @property
    def has_prob(self) -> bool:
        """Whether the unit carries a classifier probability (suite2p's ``iscell[:, 1]``)."""
        iscell = self.unit.iscell
        return iscell is not None and np.ndim(iscell) == 2 and iscell.shape[1] > 1

    def prob(self, k: int) -> float:
        return float(self.unit.iscell[k, 1]) if self.has_prob else np.nan

    def trace(self, k: int, *, z: int, c: int) -> RoiTrace | None:
        """Row ``k`` as a member row of the trace table, read at ``z`` / ``c``
        unless the run recorded where it read; None without a raw trace.
        """
        return unit_trace(
            self.unit,
            k,
            uid=0,
            member=k,
            source=self.name,
            path=self.path,
            z=z if self.read_z is None else self.read_z,
            c=c if self.read_c is None else self.read_c,
            engine=self.engine,
        )


def unit_trace(
    unit: ResultUnit,
    k: int,
    *,
    uid: int,
    member,
    source: str,
    path,
    z: int,
    c: int,
    engine: str,
) -> RoiTrace | None:
    """Row ``k`` of a unit's traces as one :class:`RoiTrace`: ``raw`` is
    ``F``, ``neuropil`` ``Fneu``, ``dff`` ``norm`` and every other kind
    rides in ``kinds``; None when the unit has no ``raw`` trace.
    """
    raw = unit.traces.get("raw")
    if raw is None or not 0 <= k < len(raw):
        return None
    fneu = unit.traces.get("neuropil")
    norm = unit.traces.get("dff")
    frames = unit.attrs.get("frames")
    tp = unit.attrs.get("tp_indices")
    return RoiTrace(
        uid=int(uid),
        member=member,
        source=source,
        z=int(z),
        c=int(c),
        engine=str(engine),
        F=np.asarray(raw[k], np.float32),
        Fneu=None if fneu is None else np.asarray(fneu[k], np.float32),
        norm=None if norm is None else np.asarray(norm[k], np.float32),
        kinds={
            kind: np.asarray(v[k], np.float32)
            for kind, v in unit.traces.items()
            if kind not in ("raw", "neuropil", "dff")
        },
        frames=None if not frames else tuple(int(v) for v in frames),
        fs=unit.fs,
        path=None if path is None else Path(path),
        extra={} if not tp or frames else {"tp_indices": [int(t) for t in tp]},
    )


def _rim(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a boolean mask (4-connected)."""
    core = mask.copy()
    core[1:, :] &= mask[:-1, :]
    core[:-1, :] &= mask[1:, :]
    core[:, 1:] &= mask[:, :-1]
    core[:, :-1] &= mask[:, 1:]
    return mask & ~core


MASK_MODES = ("circle", "outline", "fill")
RING_SEGMENTS = 36  # reads as round at any sane zoom
RING_SCALE = 1.15  # over the equal-area radius, so the ring clears the mask
MIN_RING_RADIUS = 2.0  # px: a few-pixel cell still gets a ring worth seeing
HALO_SCALE = 1.7  # the selection ring sits outside the ROI's own


def footprint_center(ypix, xpix) -> tuple[float, float]:
    """``(x, y)`` centre of a footprint in world coordinates.

    Image pixel ``(row, col)`` covers world ``[col, col + 1) x [row, row + 1)``,
    so a pixel centre sits half a pixel past its index.
    """
    return float(np.mean(xpix)) + 0.5, float(np.mean(ypix)) + 0.5


def footprint_radius(ypix, xpix, scale: float = RING_SCALE) -> float:
    """Radius of the circle standing in for a footprint: its equal-area
    circle (``sqrt(npix / pi)``, which for a compact mask lands just outside
    the edge), scaled, and floored at ``MIN_RING_RADIUS`` so the few-pixel
    masks this was written for still get something to look at.
    """
    n = max(len(ypix), 1)
    return max(float(np.sqrt(n / np.pi)) * float(scale), MIN_RING_RADIUS)


def ring(cx: float, cy: float, r: float, segments: int = RING_SEGMENTS) -> np.ndarray:
    """Closed circle as ``(segments + 1, 2)`` ``(x, y)`` points."""
    t = np.linspace(0.0, 2.0 * np.pi, segments + 1, dtype=np.float32)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


# The corners bounding each edge, as (dx, dy) off the top-left corner of
# pixel (row, col) - which in world coordinates is (col, row). One row per
# side, in the order the sides are stacked below.
_EDGE_FROM = np.array([(0, 0), (0, 1), (0, 0), (1, 0)], np.float32)
_EDGE_TO = np.array([(1, 0), (1, 1), (0, 1), (1, 1)], np.float32)


def footprint_edges(ypix, xpix) -> np.ndarray:
    """A footprint's mask/background border as ``(m, 2)`` line points: the
    unit edges of its boundary pixels, separated by NaN rows.

    cellpose traces its outlines into a polygon (``cv2.findContours``) and
    suite2p paints a rim of boundary pixels; walking the pixel edges instead
    needs no ordering pass and puts the stroke on the true border rather than
    half a pixel inside it, which matters when a cell is four pixels across.
    The mask is worked in its own bounding box, padded so an edge pixel still
    has a neighbour to be missing.
    """
    if not len(ypix):
        return np.zeros((0, 2), np.float32)
    ypix = np.asarray(ypix, np.int64)
    xpix = np.asarray(xpix, np.int64)
    y0, x0 = int(ypix.min()), int(xpix.min())
    h = int(ypix.max()) - y0 + 1
    w = int(xpix.max()) - x0 + 1
    mask = np.zeros((h + 2, w + 2), bool)
    mask[ypix - y0 + 1, xpix - x0 + 1] = True
    inner = mask[1:-1, 1:-1]
    sides = np.stack(
        [
            inner & ~mask[:-2, 1:-1],  # nothing above: its top edge shows
            inner & ~mask[2:, 1:-1],  # nothing below
            inner & ~mask[1:-1, :-2],  # nothing to the left
            inner & ~mask[1:-1, 2:],  # nothing to the right
        ]
    )
    side, rows, cols = np.nonzero(sides)
    if not len(side):
        return np.zeros((0, 2), np.float32)
    corner = np.column_stack([cols + x0, rows + y0]).astype(np.float32)
    out = np.full((3 * len(side) - 1, 2), np.nan, np.float32)
    out[0::3] = corner + _EDGE_FROM[side]
    out[1::3] = corner + _EDGE_TO[side]
    return out


def outline_paths(
    ypix,
    xpix,
    mode: str = "circle",
    scale: float = RING_SCALE,
    segments: int = RING_SEGMENTS,
) -> list[np.ndarray]:
    """One footprint's paths in ``mode``: its traced border, or the circle
    standing in for it. An empty footprint gives nothing to draw.
    """
    if mode == "outline":
        edges = footprint_edges(ypix, xpix)
        if len(edges):
            return [edges]
    if not len(ypix):
        return []
    cx, cy = footprint_center(ypix, xpix)
    return [ring(cx, cy, footprint_radius(ypix, xpix, scale), segments)]


def outline_data(
    comps,
    mode: str = "circle",
    halo=(),
    scale: float = RING_SCALE,
    segments: int = RING_SEGMENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """``(positions (m, 3), colors (m, 4))`` for a line graphic drawing every
    component as a thin closed path.

    ``comps`` are the same ``(ypix, xpix, lam, rgb, fill)`` tuples
    ``feathered_rgba`` takes, so one list feeds either overlay - ``lam`` goes
    unused here and ``fill`` becomes the stroke alpha. The paths are stitched
    into one buffer separated by NaN rows, which pygfx breaks the line on.
    ``halo`` footprints (``(ypix, xpix)`` pairs) get a white circle outside
    them as well: the vector answer to the filled overlay's white rim.
    """
    paths: list[np.ndarray] = []
    colors: list[tuple] = []
    for ypix, xpix, _lam, rgb, fill in comps:
        rgba = (*np.asarray(rgb, np.float32).tolist(), float(fill))
        for path in outline_paths(ypix, xpix, mode, scale, segments):
            paths.append(path)
            colors.append(rgba)
    for ypix, xpix in halo:
        if not len(ypix):
            continue
        cx, cy = footprint_center(ypix, xpix)
        paths.append(
            ring(cx, cy, footprint_radius(ypix, xpix, scale) * HALO_SCALE, segments)
        )
        colors.append((1.0, 1.0, 1.0, 1.0))
    return stitch_paths(paths, colors)


def stitch_paths(paths, colors) -> tuple[np.ndarray, np.ndarray]:
    """Join ``(n, 2)`` paths into one ``(m, 3)`` position array with a NaN
    row between pieces, plus the matching ``(m, 4)`` per-vertex colors.
    """
    if not paths:
        return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
    total = sum(len(p) for p in paths) + len(paths) - 1
    pos = np.full((total, 3), np.nan, np.float32)
    col = np.zeros((total, 4), np.float32)
    at = 0
    for path, rgba in zip(paths, colors):
        n = len(path)
        pos[at : at + n, :2] = path
        pos[at : at + n, 2] = 0.0
        col[at : at + n] = rgba
        at += n + 1
    return pos, col


def derived_comps(
    sets_on_z: list[DerivedSet],
    alpha: float,
    selected: tuple[DerivedSet, int] | None = None,
    grouped: frozenset | set = frozenset(),
) -> tuple[list, tuple | None, list]:
    """What one plane's derived sets draw: ``(comps, selected, halo)``.

    ``comps`` are the ``(ypix, xpix, lam, rgb, fill)`` tuples both renderers
    take, one per visible, undiscarded row - rejected rows dimmed, group
    members at ``SELECTED_ALPHA``. ``selected`` is the ``(ypix, xpix, rgb)``
    of the shown component, None when it is not on this plane. ``halo`` are
    the footprints of the selection and the group, which the vector overlay
    rings in white.
    """
    comps: list = []
    halo: list = []
    sel = None
    for s in sets_on_z:
        if not s.visible:
            continue
        for k, (ypix, xpix, lam) in enumerate(s.footprints):
            if k in s.discarded:
                continue
            fill = alpha if s.accepted[k] else alpha * 0.35
            if (id(s), k) in grouped:
                fill = SELECTED_ALPHA
                halo.append((ypix, xpix))
            comps.append((ypix, xpix, lam, component_color(s, k), fill))
    if selected is not None:
        s, k = selected
        if s in sets_on_z and s.visible and k not in s.discarded:
            ypix, xpix, _lam = s.footprints[k]
            sel = (ypix, xpix, component_color(s, k))
            halo.append((ypix, xpix))
    return comps, sel, halo


def derived_rgba(
    shape: tuple[int, int],
    sets_on_z: list[DerivedSet],
    alpha: float,
    selected: tuple[DerivedSet, int] | None = None,
    grouped: frozenset | set = frozenset(),
) -> np.ndarray:
    """``(ny, nx, 4)`` uint8 overlay for the derived sets of one plane.

    Each component's pixels get its own color (``component_color``) at
    ``lam / lam.max() * alpha``; where components overlap the higher alpha
    wins color and coverage. ``selected`` (a ``(set, row)`` pair) is filled
    at ``SELECTED_ALPHA`` with a white rim; ``grouped`` (``(id(set), row)``
    pairs of a multi-selection) fills at ``SELECTED_ALPHA`` too. Discarded
    rows and invisible sets are skipped; rejected rows draw dimmed.
    """
    comps, sel, _halo = derived_comps(sets_on_z, alpha, selected, grouped)
    return feathered_rgba(shape, comps, sel)


def derived_outline(
    sets_on_z: list[DerivedSet],
    mode: str = "circle",
    selected: tuple[DerivedSet, int] | None = None,
    grouped: frozenset | set = frozenset(),
    scale: float = RING_SCALE,
    segments: int = RING_SEGMENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """``(positions, colors)`` drawing one plane's derived sets as thin
    paths - the same rows ``derived_rgba`` fills, outlined instead.

    Strokes are opaque: a hairline at the fill overlay's opacity is
    invisible. Only rejected rows draw dimmed, and the selection and the
    group keep their white ring.
    """
    comps, _sel, halo = derived_comps(sets_on_z, 1.0, selected, grouped)
    return outline_data(comps, mode, halo, scale, segments)


def component_color(s: DerivedSet, k: int) -> tuple[float, float, float]:
    """One component's rgb in 0-1: its explicit group color when set, its
    class color when labeled, else a hue of its own - the same treatment
    drawn ROIs get.
    """
    rgb = s.colors.get(k)
    if rgb is not None:
        return tuple(rgb)
    ci = s.classes.get(k)
    if ci is not None:
        return class_color(ci)
    offset = sum(map(ord, s.name)) * 61
    r, g, b = ROI_COLORS[(k + offset) % len(ROI_COLORS)]
    return (r / 255.0, g / 255.0, b / 255.0)


def feathered_rgba(shape: tuple[int, int], comps, selected=None) -> np.ndarray:
    """``(ny, nx, 4)`` uint8 overlay from ``(ypix, xpix, lam, rgb, fill)``
    components: each pixel takes ``lam / lam.max() * fill`` alpha, and
    where components overlap the higher alpha wins color and coverage.
    ``selected`` (``(ypix, xpix, rgb)``) fills at ``SELECTED_ALPHA`` with
    a white rim.
    """
    ny, nx = shape
    rgba = np.zeros((ny, nx, 4), np.uint8)
    best = np.zeros((ny, nx), np.float32)
    for ypix, xpix, lam, rgb, fill in comps:
        color = np.rint(np.asarray(rgb, np.float32) * 255).astype(np.uint8)
        lam = np.asarray(lam, np.float32)
        peak = float(lam.max()) if lam.size else 0.0
        a = lam / peak * fill if peak > 0 else np.full(lam.shape, fill, np.float32)
        win = a > best[ypix, xpix]
        yy, xx = ypix[win], xpix[win]
        best[yy, xx] = a[win]
        rgba[yy, xx, :3] = color
        rgba[yy, xx, 3] = np.rint(a[win] * 255).astype(np.uint8)
    if selected is not None:
        ypix, xpix, rgb = selected
        mask = np.zeros((ny, nx), bool)
        mask[ypix, xpix] = True
        fill = np.uint8(round(SELECTED_ALPHA * 255))
        rgba[mask, :3] = np.rint(np.asarray(rgb, np.float32) * 255).astype(np.uint8)
        rgba[mask, 3] = fill
        rgba[_rim(mask)] = (255, 255, 255, fill)
    return rgba


def run_dir_complete(d) -> bool:
    """True when ``d`` holds a loadable run: ``stat.npy`` + ``ops.npy``, a
    results file (``mbo_utilities.results``), a masknmf run folder holding
    one, or one unit inside one (``<file>.zarr/zplane01``).
    """
    from mbo_utilities.arrays.masknmf_run import is_masknmf_run
    from mbo_utilities.results import newest_results, results_pipeline

    d = Path(d)
    if (d / "stat.npy").exists() and (d / "ops.npy").exists():
        return True
    if results_pipeline(d) is not None:
        return True
    if is_masknmf_run(d) and newest_results(d) is not None:
        return True
    return (
        d.parent.suffix == ".zarr"
        and results_pipeline(d.parent) is not None
        and (d / "zarr.json").is_file()
    )


def finished_dirs(out_root, planes=None) -> list[Path]:
    """Completed output dirs under ``out_root`` for 1-based ``planes`` (any
    plane when None): the root itself when it holds outputs, else its
    direct children whose name carries that plane's ``zplaneNN`` tag
    (suite2p's ``zplane01_tp00001-00400``, masknmf's
    ``<stamp>_masknmf_zplane01``), read through ``filename_tags`` so a
    plane never matches another's by prefix.
    """
    root = Path(out_root)
    if run_dir_complete(root):
        return [root]
    wanted = None if planes is None else {int(p) for p in planes}
    out: list[Path] = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if not d.is_dir():
            continue
        z = [t.start for t in filename_tags(d.name) if t.definition.label == "zplane"]
        if not z or (wanted is not None and int(z[0]) not in wanted):
            continue
        if run_dir_complete(d):
            out.append(d)
    return out


def _default_masknmf_settings() -> dict:
    from mbo_utilities.masknmf.params import MasknmfSettings

    return MasknmfSettings().to_dict()


def masknmf_settings(host) -> dict | None:
    """The masknmf settings the Run tab is holding, or None when it has not
    built that pipeline yet.
    """
    instances = getattr(host, "_pipeline_instances", None) or {}
    settings = getattr(instances.get("MaskNMF"), "settings", None)
    if settings is None:
        return None
    try:
        return settings.to_dict()
    except Exception:
        return None
