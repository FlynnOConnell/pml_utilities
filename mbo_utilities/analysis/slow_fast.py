"""Slow events and the fast peaks inside them, on one dF/F trace.

A port of ``detect_trace`` from the CA1 voltage project's
``detect_slow_fast_events.py`` (the detection saved as
``detected_slow_fast_raw_slow3sd.pkl``), with dicts of arrays in place of its
DataFrames. The trace is events-positive dF/F (``-dfof_raw`` for a dimming
indicator):

1. baseline: subtract the ``base_window_s`` moving median of the trace's
   ``base_sigma_ms`` Gaussian;
2. slow trace: a ``slow_sigma_ms`` Gaussian (12.7 ms is about a 10 Hz
   low-pass). A slow event is a stretch above ``edge_sd`` robust SD, stretches
   closer than ``merge_ms`` merged, kept when it reaches ``slow_sd``; events
   touching the first or last ``skip_s`` are dropped;
3. fast trace: a ``fast_sigma_ms`` Gaussian. Noise is the robust SD of fast
   minus slow; a fast peak has prominence >= ``k_prom`` x noise and lies inside
   a slow event.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_widths

__all__ = [
    "FAST_COLUMNS",
    "SLOW_COLUMNS",
    "SlowFastEvents",
    "SlowFastSettings",
    "detect_slow_fast",
    "moving_median",
    "robust_sd",
    "slow_periods",
]

SLOW_COLUMNS = (
    "start_s",
    "end_s",
    "duration_ms",
    "peak_s",
    "amplitude",
    "amplitude_sd",
    "n_fast",
)
FAST_COLUMNS = (
    "sample_index",
    "time_s",
    "amplitude",
    "prominence",
    "prominence_noise",
    "width_ms",
    "slow_id",
    "amplitude_fine",
    "amplitude_raw",
)
MEDIAN_RATE_HZ = 50.0


@dataclass
class SlowFastSettings:
    """The detection's parameters; the defaults are the published run's."""

    slow_sigma_ms: float = 12.7
    edge_sd: float = 1.5
    slow_sd: float = 3.0
    merge_ms: float = 100.0
    k_prom: float = 5.0
    fast_sigma_ms: float = 2.0
    base_sigma_ms: float = 4.0
    base_window_s: float = 10.0
    skip_s: float = 2.0
    fine_sigma_ms: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> SlowFastSettings:
        """Known keys only, so a file written by a newer version still reads."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in values.items() if k in names})


@dataclass
class SlowFastEvents:
    """One trace's detection: the tables and the traces they were read off.

    ``slow`` and ``fast`` hold one array per column of :data:`SLOW_COLUMNS`
    and :data:`FAST_COLUMNS`, the original script's DataFrame columns. A fast
    peak's ``slow_id`` is the row of the slow event it sits in.
    """

    fs: float
    x: np.ndarray
    slow_trace: np.ndarray
    fast_trace: np.ndarray
    fine_trace: np.ndarray
    sd: float
    noise: float
    slow: dict[str, np.ndarray]
    fast: dict[str, np.ndarray]

    @property
    def n_slow(self) -> int:
        return len(self.slow["start_s"])

    @property
    def n_fast(self) -> int:
        return len(self.fast["sample_index"])


def robust_sd(x: np.ndarray) -> float:
    """1.4826 x the median absolute deviation of the finite values."""
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def moving_median(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """``window_s`` moving median computed at about 50 Hz and interpolated back."""
    step = max(1, int(fs / MEDIAN_RATE_HZ))
    m = median_filter(x[::step], size=int(window_s * fs / step) | 1, mode="reflect")
    return np.interp(np.arange(len(x)), np.arange(len(m)) * step, m)


def slow_periods(
    slow: np.ndarray,
    sd: float,
    fs: float,
    edge_sd: float,
    slow_sd: float,
    merge_ms: float,
) -> np.ndarray:
    """``(n, 2)`` sample ``[start, stop)`` of the stretches above ``edge_sd`` x
    ``sd``, merged across gaps shorter than ``merge_ms`` (0 never merges), kept
    when they reach ``slow_sd`` x ``sd``.
    """
    edges = np.diff(np.r_[0, (slow > edge_sd * sd).astype(int), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    gap = int(merge_ms * fs / 1000)
    merged: list[list[int]] = []
    for a, b in zip(starts, stops, strict=True):
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    kept = [(a, b) for a, b in merged if slow[a:b].max() >= slow_sd * sd]
    return np.asarray(kept, dtype=np.int64).reshape(-1, 2)


def detect_slow_fast(
    trace: np.ndarray, fs: float, settings: SlowFastSettings | None = None
) -> SlowFastEvents:
    """Slow events and the fast peaks inside them for one events-positive trace."""
    s = settings or SlowFastSettings()
    raw = np.asarray(trace, dtype=np.float64)
    ## NOTE: sigma * fs / 1000 as the reference writes it; sigma * (fs / 1000) rounds differently
    x = raw - moving_median(
        gaussian_filter1d(raw, s.base_sigma_ms * fs / 1000), fs, s.base_window_s
    )
    slow = gaussian_filter1d(x, s.slow_sigma_ms * fs / 1000)
    fast = gaussian_filter1d(x, s.fast_sigma_ms * fs / 1000)
    fine = (
        gaussian_filter1d(x, s.fine_sigma_ms * fs / 1000) if s.fine_sigma_ms > 0 else x
    )

    sd = robust_sd(slow)
    skip = int(s.skip_s * fs)
    periods = slow_periods(slow, sd, fs, s.edge_sd, s.slow_sd, s.merge_ms)
    periods = periods[(periods[:, 0] >= skip) & (periods[:, 1] <= len(x) - skip)]
    tops = np.array([a + int(np.argmax(slow[a:b])) for a, b in periods], dtype=np.int64)

    noise = robust_sd(fast - slow)
    slow_id = np.full(len(x), -1, dtype=np.int64)
    for i, (a, b) in enumerate(periods):
        slow_id[a:b] = i
    peaks, props = find_peaks(fast, prominence=s.k_prom * noise)
    inside = slow_id[peaks] >= 0
    peaks, prominence = peaks[inside], props["prominences"][inside]
    width_ms = (
        peak_widths(fast, peaks, rel_height=0.5)[0] / fs * 1000
        if len(peaks)
        else np.zeros(0)
    )
    fast_table = {
        "sample_index": peaks.astype(np.int64),
        "time_s": peaks / fs,
        "amplitude": fast[peaks],
        "prominence": prominence,
        "prominence_noise": prominence / noise if noise else prominence,
        "width_ms": width_ms,
        "slow_id": slow_id[peaks],
        "amplitude_fine": fine[peaks],
        "amplitude_raw": x[peaks],
    }
    slow_table = {
        "start_s": periods[:, 0] / fs,
        "end_s": periods[:, 1] / fs,
        "duration_ms": (periods[:, 1] - periods[:, 0]) / fs * 1000,
        "peak_s": tops / fs,
        "amplitude": slow[tops],
        "amplitude_sd": slow[tops] / sd if sd else slow[tops],
        "n_fast": np.bincount(fast_table["slow_id"], minlength=len(periods)).astype(
            np.int64
        ),
    }
    return SlowFastEvents(
        fs=float(fs),
        x=x,
        slow_trace=slow,
        fast_trace=fast,
        fine_trace=fine,
        sd=sd,
        noise=noise,
        slow=slow_table,
        fast=fast_table,
    )
