"""One time shared by every view of a recording.

A :class:`Playhead` holds the time in seconds on the recording's clock and
emits ``"time"`` when it moves. Each view keeps its own axis as a
:class:`TimeAxis` (the viewer's frames at its binning, a trace's samples
over a frame window, a motion curve in seconds) and converts through it, so
a scrub on any plot lands on the same instant everywhere. Without a
sampling rate the clock is raw frames (``fs = 1``) and every axis is in
frames. After fastplotlib's TimeStore, with an offset per axis so a trace
read from a frame window sits where it was recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

from mbo_utilities.annotation.events import Observable

__all__ = ["Playhead", "TimeAxis"]


@dataclass(frozen=True)
class TimeAxis:
    """Units of one view against the clock: ``per_second`` units per
    second, with the axis's zero at ``offset`` seconds.
    """

    per_second: float = 1.0
    offset: float = 0.0

    def units(self, seconds: float) -> float:
        return (float(seconds) - self.offset) * self.per_second

    def seconds(self, units: float) -> float:
        return float(units) / self.per_second + self.offset

    def on(self, other: TimeAxis) -> tuple[float, float]:
        """``(xscale, xstart)`` placing this axis's samples on ``other``:
        sample ``i`` sits at ``xstart + i * xscale`` there.
        """
        return (
            other.per_second / self.per_second,
            (self.offset - other.offset) * other.per_second,
        )

    @classmethod
    def sampled(cls, fs, frame_average: int = 1, first_frame: int = 0) -> TimeAxis:
        """The axis of samples taken every ``frame_average`` raw frames from
        raw frame ``first_frame``, at ``fs`` raw frames per second (raw
        frames are the clock when ``fs`` is unknown).
        """
        rate = float(fs) if fs else 1.0
        binning = max(int(frame_average or 1), 1)
        return cls(per_second=rate / binning, offset=float(first_frame) / rate)


class Playhead(Observable):
    """The time on screen, in seconds; ``seek`` moves it and tells every
    subscriber who moved it, so a view can ignore its own seeks.
    """

    events = ("time",)

    def __init__(self, time: float = 0.0):
        super().__init__()
        self.time = float(time)

    def seek(self, seconds: float, source=None) -> bool:
        """Move to ``seconds`` (never before zero); True when it moved."""
        seconds = max(float(seconds), 0.0)
        if abs(seconds - self.time) < 1e-9:
            return False
        previous, self.time = self.time, seconds
        self._emit("time", seconds=seconds, previous=previous, source=source)
        return True
