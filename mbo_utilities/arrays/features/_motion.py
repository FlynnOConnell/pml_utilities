"""Motion correction a recording went through, as shift traces on its time axis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MOTION_AXES = ("X", "Y", "Z")


@dataclass
class MotionCorrection:
    """The shifts one motion-correction stage applied over a recording.

    One trace per axis on the recording's own time axis: ``traces`` maps a
    label to ``(t, shift)``, ``t`` in seconds from the start of the recording
    and ``shift`` in ``unit``. A label starts with its axis letter (``"X"``,
    ``"Z layer 3"``) so a consumer can colour by axis. ``source`` names the
    stage the way a plot shows it: ``"RTMC"`` for the AOD's real-time
    correction (``MescArray.motion_correction``, in µm); a registration's
    per-frame offsets (suite2p's ``xoff`` / ``yoff``, masknmf's shifts, in px)
    take the same shape.
    """

    source: str
    unit: str
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.traces)

    @property
    def duration_s(self) -> float:
        """The last sample's time over every trace, 0 without one."""
        return max((float(t[-1]) for t, _ in self.traces.values() if len(t)), default=0.0)
