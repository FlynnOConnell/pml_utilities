"""What the animal did during a recording, on the recording's clock.

One GUI-free object, :class:`Behavior`, whatever logged it. Its three parts
are the ones NWB and Neo use, under plain names: ``signals`` are continuous
measurements sampled in time (NWB ``TimeSeries``, Neo ``AnalogSignal``: the
position on a treadmill, the speed), ``events`` are instants (NWB events,
Neo ``Event``: licks, rewards, laps) and ``epochs`` are intervals (NWB
``TimeIntervals``, Neo ``Epoch``: a reward zone, a trial). Every time is in
seconds on the recording's T axis, as ``MotionCorrection`` reports its
shifts: the logger's clock is shifted so the imaging sync pulse it recorded
lands at 0 (``offset_s``).

A reader turns one logger's file into a ``Behavior``; ``READERS`` maps a
suffix to it (``.tdml``: BehaviorMate, ``tdml.py``). ``read_behavior``
dispatches, ``find_behavior`` finds the file recorded with a recording
(the same subject and day in its name, beside the recording, one folder up
or in a sibling ``behavior`` folder) and ``behavior_for`` puts it on the
array's ``behavior`` facet the first time it is asked for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.behavior.tdml import read_tdml
from mbo_utilities.lazy_array import LazyArray, base_array

__all__ = [
    "READERS",
    "Behavior",
    "BehaviorSignal",
    "behavior_for",
    "find_behavior",
    "read_behavior",
]

logger = log.get("behavior")


@dataclass
class BehaviorSignal:
    """One continuous measurement: ``values`` sampled at ``t`` seconds on the
    recording's clock, in ``unit``.
    """

    t: np.ndarray
    values: np.ndarray
    unit: str = ""

    @property
    def duration_s(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0


@dataclass
class Behavior:
    """A session's behavior: ``signals`` by name, ``events`` by name (times),
    ``epochs`` by name (``(n, 2)`` start and stop), all in seconds on the
    recording's clock. ``sync`` keeps the imaging sync pulses on the logger's
    own clock and ``offset_s`` the logger time the recording started at.
    ``info`` is the logger's own settings, verbatim.
    """

    source: str
    signals: dict[str, BehaviorSignal] = field(default_factory=dict)
    events: dict[str, np.ndarray] = field(default_factory=dict)
    epochs: dict[str, np.ndarray] = field(default_factory=dict)
    sync: np.ndarray | None = None
    offset_s: float = 0.0
    start: datetime | None = None
    subject: str = ""
    path: Path | None = None
    info: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.signals or self.events or self.epochs)

    @property
    def duration_s(self) -> float:
        """The last time over everything, 0 without one."""
        last = [s.duration_s for s in self.signals.values()]
        last += [float(t[-1]) for t in self.events.values() if len(t)]
        last += [float(e[-1, 1]) for e in self.epochs.values() if len(e)]
        return max(last, default=0.0)


READERS = {".tdml": read_tdml}


def read_behavior(path) -> Behavior:
    """The ``Behavior`` in ``path``, by the reader its suffix names."""
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"no behavior reader for {path.suffix!r}; known: {sorted(READERS)}"
        )
    return reader(path)


def find_behavior(recording) -> Path | None:
    """The behavior file recorded with ``recording``: one a reader can read,
    beside it, one folder up or in a sibling folder whose name starts with
    ``behavior``, named after the same subject and day as the recording (the
    first two ``_`` words of its name, ``u005a04_20260915``). Of several, the
    one sharing the longest prefix with the recording's name wins, then the
    first by name, and the rest are logged.
    """
    recording = Path(recording)
    words = recording.stem.split("_")
    if len(words) < 2:
        return None
    key = f"{words[0]}_{words[1]}"
    folders = [recording.parent, recording.parent.parent]
    if recording.parent.parent.is_dir():
        folders += [
            d
            for d in recording.parent.parent.iterdir()
            if d.is_dir() and d.name.lower().startswith("behavio")
        ]
    found = []
    for folder in dict.fromkeys(folders):
        if not folder.is_dir():
            continue
        for candidate in sorted(folder.iterdir()):
            if (
                candidate.is_file()
                and candidate.suffix.lower() in READERS
                and candidate.stem.startswith(key)
            ):
                found.append(candidate)
    if not found:
        return None
    found.sort(
        key=lambda p: (-len(os.path.commonprefix([p.stem, recording.stem])), p.name)
    )
    if len(found) > 1:
        logger.warning(
            f"{len(found)} behavior files match {recording.name}; using {found[0].name}, "
            f"not {', '.join(p.name for p in found[1:])}"
        )
    return found[0]


def behavior_for(arr) -> Behavior | None:
    """``arr.behavior``, found and read on first use: the file recorded with
    ``arr.source_path`` (``find_behavior``). None when there is none or the
    file cannot be read; the answer is kept on the array either way.
    """
    arr = base_array(arr)
    if not isinstance(arr, LazyArray):
        return None
    if hasattr(arr, "_behavior"):
        return arr._behavior
    source = getattr(arr, "source_path", None)
    path = find_behavior(source) if isinstance(source, (str, Path)) else None
    behavior = None
    if path is not None:
        try:
            behavior = read_behavior(path)
            logger.info(
                f"behavior for {Path(source).name}: {path.name} "
                f"({', '.join(behavior.signals)}; "
                f"{', '.join(f'{len(v)} {k}' for k, v in behavior.events.items())})"
            )
        except (OSError, ValueError, KeyError) as e:
            logger.warning(f"behavior file {path.name} unreadable: {e}")
    arr.behavior = behavior
    return behavior
