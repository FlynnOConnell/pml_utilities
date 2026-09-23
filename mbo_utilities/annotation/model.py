"""The manual-ROI session model: store, trace table, the slice on screen.

``RoiModel`` is what a view binds to. It composes the label store
(:class:`RoiLabelStore`), the trace table (:class:`RoiTraceTable`) and the
viewer's slider position, re-emits the two tables' events under its own
name and adds ``"view"`` when the position moves onto another plane. The
ROI widget, the Process tab's ROI pipeline and a script all read the same
questions off it: which plane is on screen, where was an ROI drawn, where
would a run read its pixels (:meth:`targets`), what has it been measured
with (:meth:`traced`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from cmap import Colormap

from mbo_utilities.annotation.events import ModelEvent, Observable
from mbo_utilities.annotation.store import RoiLabelStore
from mbo_utilities.annotation.traces import RoiTrace, RoiTraceTable

__all__ = ["COLUMNS", "RoiModel", "RunTarget"]

# the per-ROI numbers the store can answer for ``column`` / ``colorize``
COLUMNS: tuple[str, ...] = ("plane", "z", "c", "area", "class")


@dataclass(frozen=True)
class RunTarget:
    """Where one run reads one ROI: the mask from store ``plane``, the
    pixels from z-plane ``z`` and channel ``c`` of the movie."""

    index: int
    uid: int
    plane: int
    z: int
    c: int


class RoiModel(Observable):
    events = ("rois", "traces", "view")

    def __init__(self, store: RoiLabelStore, traces: RoiTraceTable | None = None):
        super().__init__()
        self._store: RoiLabelStore | None = None
        self._traces: RoiTraceTable | None = None
        self.view: dict[str, int] = {}
        self.plane = 0
        self.traces = traces if traces is not None else RoiTraceTable()
        self.store = store

    def _forward(self, event: ModelEvent) -> None:
        self._emit(event.type, **event.info)

    def close(self) -> None:
        """Stop forwarding: the store and the table may outlive this model
        (parked for the next widget) and must not call into it."""
        if self._store is not None:
            self._store.remove_event_handler(self._forward, "rois")
        if self._traces is not None:
            self._traces.remove_event_handler(self._forward, "traces")
        self.clear_event_handlers()

    @property
    def traces(self) -> RoiTraceTable:
        return self._traces

    @traces.setter
    def traces(self, traces: RoiTraceTable) -> None:
        """Swap the trace table (a parked one from the previous widget),
        keeping the subscribers."""
        if self._traces is not None:
            self._traces.remove_event_handler(self._forward, "traces")
        self._traces = traces
        traces.add_event_handler(self._forward, "traces")

    @property
    def store(self) -> RoiLabelStore:
        return self._store

    @store.setter
    def store(self, store: RoiLabelStore) -> None:
        """Swap the label store (a restore, a unit switch), keeping the
        subscribers; the plane is re-derived from the current view."""
        if self._store is not None:
            self._store.remove_event_handler(self._forward, "rois")
        self._store = store
        store.add_event_handler(self._forward, "rois")
        self.plane = store.plane_of(self.view)

    @property
    def z(self) -> int:
        """0-based z-plane on screen (0 when z does not key the volume)."""
        name = self._store.axis_name("z")
        return int(self.view.get(name, 0)) if name is not None else 0

    @property
    def c(self) -> int:
        """0-based channel on screen (0 when c does not key the volume)."""
        name = self._store.axis_name("c")
        return int(self.view.get(name, 0)) if name is not None else 0

    def set_view(self, pos: Mapping[str, int]) -> bool:
        """Record the slider position; emits ``"view"`` and returns True
        when it lands on another plane."""
        self.view = {str(k): int(v) for k, v in pos.items()}
        plane = self._store.plane_of(self.view)
        if plane == self.plane:
            return False
        previous, self.plane = self.plane, plane
        self._emit("view", plane=plane, previous=previous)
        return True

    def plane_pos(self, plane: int) -> dict[str, int]:
        return self._store.plane_pos(plane)

    def plane_label(self, plane: int) -> str:
        return self._store.plane_label(plane)

    def rois_on_screen(self) -> list[int]:
        return self._store.rois_on_plane(self.plane)

    def targets(self, indices, z: int | None = None, c: int | None = None) -> list[RunTarget]:
        """One :class:`RunTarget` per drawn ROI of ``indices``: the mask
        stays on the plane it was drawn on; the pixels come from ``z`` /
        ``c`` when given, else from where the ROI was drawn."""
        out = []
        for i in indices:
            i = int(i)
            if not 0 <= i < len(self._store.rois):
                continue
            record = self._store.rois[i]
            out.append(
                RunTarget(
                    index=i,
                    uid=record.uid,
                    plane=record.plane,
                    z=self._store.roi_z(i) if z is None else int(z),
                    c=self._store.roi_c(i) if c is None else int(c),
                )
            )
        return out

    def traced(self, index: int, z: int | None = None, c: int | None = None, engine: str | None = None) -> list[RoiTrace]:
        """The traces drawn ROI ``index`` already has at the coordinates
        given (any, when none are)."""
        if not 0 <= index < len(self._store.rois):
            return []
        return self.traces.at(self._store.rois[index].uid, z=z, c=c, engine=engine)

    def uid_index(self, uid: int) -> int | None:
        return self._store.uid_index(uid)

    def column(self, name: str) -> dict[int, float]:
        """One number per drawn ROI, keyed by uid: its ``plane``, ``z``, ``c``,
        ``area`` or ``class`` (unlabeled ROIs are left out of that one)."""
        if name not in COLUMNS:
            raise KeyError(f"unknown column {name!r}; one of {COLUMNS}")
        store = self._store
        out: dict[int, float] = {}
        for i, record in enumerate(store.rois):
            if name == "class":
                if record.class_index < 0:
                    continue
                value = record.class_index
            elif name == "z":
                value = store.roi_z(i)
            elif name == "c":
                value = store.roi_c(i)
            else:
                value = getattr(record, name)
            out[record.uid] = float(value)
        return out

    def colorize(self, values: Mapping[int, float] | None, cmap: str = "viridis", categorical: bool = False, vmin: float | None = None, vmax: float | None = None) -> dict[int, tuple[int, int, int]]:
        """Color every ROI in ``values`` (uid -> number) through a colormap,
        like fastplotlib's ``cmap_transform``: continuous values are scaled
        between ``vmin`` / ``vmax`` (their extremes when None), categorical
        ones get one color per distinct value in sorted order. ROIs not in
        ``values`` keep their own color. None (or nothing) clears the tint.
        Returns the uint8 rgb per uid that was applied."""
        if not values:
            self._store.set_tint(None)
            return {}
        colormap = Colormap(cmap)
        uids = list(values)
        v = np.asarray([float(values[u]) for u in uids], float)
        # one color comes back as a bare rgba row; keep every case (n, 4)
        if categorical:
            levels = sorted(set(v.tolist()))
            lut = np.atleast_2d(np.asarray(colormap.lut(max(len(levels), 1))))[:, :3]
            rgb = lut[[levels.index(x) for x in v.tolist()]]
        else:
            lo = float(np.nanmin(v)) if vmin is None else float(vmin)
            hi = float(np.nanmax(v)) if vmax is None else float(vmax)
            norm = np.zeros_like(v) if hi <= lo else np.clip((v - lo) / (hi - lo), 0.0, 1.0)
            rgb = np.atleast_2d(np.asarray(colormap(norm)))[:, :3]
        tint = {
            u: tuple(int(round(float(c) * 255)) for c in row) for u, row in zip(uids, rgb)
        }
        self._store.set_tint(tint)
        return tint
