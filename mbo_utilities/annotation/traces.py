"""The trace table: which ROI was measured where, with what, and the result.

One :class:`RoiTrace` per measurement. A drawn ROI is identified by its
store ``uid`` and the trace by ``(uid, z, c, engine)``: the same mask read
on another channel or z-plane, or with another engine, is another row, and
re-running the same one replaces its row. Rows that stand for no drawn ROI
(an algorithm's component, a results file's line) carry ``source`` and
``member`` instead and are keyed ``(source, member)``.

The table is :class:`Observable`: it emits ``"traces"`` with
``info["action"]`` ``add``, ``remove`` or ``clear`` and the ``key`` of the
row, so the plot and the tables redraw from the event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mbo_utilities.annotation.events import Observable

__all__ = ["ENGINES", "FULL_IMAGE", "RoiTrace", "RoiTraceTable", "trace_key"]

# selector order; a run dir's ops["roi_workflow"]["engine"] uses the same names
ENGINES: tuple[str, ...] = ("mean", "suite2p", "masknmf")
# the source of rows that measure the whole frame rather than a drawn mask
FULL_IMAGE = "full image"


def trace_key(uid: int, z: int, c: int, engine: str) -> tuple:
    """Identity of a drawn ROI's trace: the mask, where it was read, how."""
    return ("roi", int(uid), int(z), int(c), str(engine))


@dataclass(eq=False)
class RoiTrace:
    """One measured trace.

    ``uid`` names the drawn ROI (0 for a row that stands for none), ``z``
    and ``c`` the 0-based plane and channel the pixels were read from,
    ``engine`` how (``ENGINES``), ``source`` the run or origin that produced
    it (``"quick"``, ``"rois_manual"``, ``"find01"``, ``"MUnit_3 lines"``,
    ``FULL_IMAGE``) and ``member`` the row inside that source for rows
    without a uid (an int, or a string such as ``"z2c1"`` for a whole-frame
    read).
    ``frames`` is the ``(start, stop, step)`` window of source frames the
    trace covers (``arrays.features._slicing.index_window`` of the frames
    read), None for the whole recording or a gapped selection, which then
    lists its frames in ``extra["tp_indices"]``; ``frame_average`` is the
    temporal binning.
    """

    uid: int
    z: int = 0
    c: int = 0
    engine: str = "mean"
    source: str = "quick"
    member: int | str | None = None
    F: np.ndarray | None = None
    Fneu: np.ndarray | None = None
    norm: np.ndarray | None = None
    frames: tuple[int, int, int] | None = None
    frame_average: int = 1
    fs: float | None = None
    label: str = ""
    path: Path | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        if self.member is not None:
            member = self.member if isinstance(self.member, str) else int(self.member)
            return ("member", self.source, member)
        return trace_key(self.uid, self.z, self.c, self.engine)

    @property
    def n_frames(self) -> int:
        return 0 if self.F is None else int(np.shape(self.F)[-1])

    @property
    def stands_for_roi(self) -> bool:
        """Whether the row belongs to a drawn ROI (uid-keyed)."""
        return self.member is None

    def record(self) -> dict:
        """The row as plain values (no arrays), for a table or a sidecar."""
        return {
            "uid": int(self.uid),
            "z": int(self.z),
            "c": int(self.c),
            "engine": self.engine,
            "source": self.source,
            "member": self.member,
            "frames": None if self.frames is None else [int(v) for v in self.frames],
            "frame_average": int(self.frame_average),
            "fs": self.fs,
            "label": self.label,
            "n_frames": self.n_frames,
            "path": None if self.path is None else str(self.path),
        }


class RoiTraceTable(Observable):
    """Rows keyed by :attr:`RoiTrace.key`, in insertion order."""

    events = ("traces",)

    def __init__(self):
        super().__init__()
        self._rows: dict[tuple, RoiTrace] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(list(self._rows.values()))

    def __contains__(self, key) -> bool:
        return key in self._rows

    def __bool__(self) -> bool:
        return bool(self._rows)

    @property
    def rows(self) -> list[RoiTrace]:
        return list(self._rows.values())

    @property
    def keys(self) -> list[tuple]:
        return list(self._rows)

    def get(self, key) -> RoiTrace | None:
        return self._rows.get(key)

    def add(self, trace: RoiTrace) -> RoiTrace:
        """Insert a row, replacing the one with the same key."""
        key = trace.key
        replaced = key in self._rows
        if replaced:
            del self._rows[key]
        self._rows[key] = trace
        self._emit("traces", action="add", key=key, replaced=replaced)
        return trace

    def remove(self, key) -> RoiTrace | None:
        trace = self._rows.pop(key, None)
        if trace is not None:
            self._emit("traces", action="remove", key=key)
        return trace

    def clear(self) -> None:
        if not self._rows:
            return
        self._rows.clear()
        self._emit("traces", action="clear")

    def for_roi(self, uid: int) -> list[RoiTrace]:
        """Every trace of one drawn ROI, in insertion order."""
        return [t for t in self._rows.values() if t.member is None and t.uid == int(uid)]

    def at(self, uid: int, z: int | None = None, c: int | None = None, engine: str | None = None) -> list[RoiTrace]:
        """The traces of ``uid`` matching every given coordinate."""
        return [
            t for t in self.for_roi(uid)
            if (z is None or t.z == int(z))
            and (c is None or t.c == int(c))
            and (engine is None or t.engine == engine)
        ]

    def from_source(self, source: str) -> list[RoiTrace]:
        return [t for t in self._rows.values() if t.source == source]

    def sources(self) -> list[str]:
        """Origins present, first-seen order."""
        out: list[str] = []
        for t in self._rows.values():
            if t.source not in out:
                out.append(t.source)
        return out

    def prune(self, uids) -> list[tuple]:
        """Drop the uid-keyed rows whose ROI is not in ``uids``; rows that
        stand for no drawn ROI stay. Returns the keys removed."""
        keep = {int(u) for u in uids}
        gone = [k for k, t in self._rows.items() if t.member is None and t.uid not in keep]
        for key in gone:
            del self._rows[key]
        if gone:
            self._emit("traces", action="remove", key=None, keys=gone)
        return gone

    def drop_source(self, source: str) -> list[tuple]:
        """Remove every row of one origin. Returns the keys removed."""
        gone = [k for k, t in self._rows.items() if t.source == source]
        for key in gone:
            del self._rows[key]
        if gone:
            self._emit("traces", action="remove", key=None, keys=gone)
        return gone

    def records(self) -> list[dict]:
        return [t.record() for t in self._rows.values()]
