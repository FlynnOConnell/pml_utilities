"""d' and the motion coupling of a trace, on synthetic traces with a known answer."""

from __future__ import annotations

import math

import numpy as np
import scipy.signal
from mbo_utilities.analysis.trace_readouts import (
    dprime,
    event_dprime,
    motion_coupling,
)

FS_HZ = 1000.0
BAND_HZ = (1.0, 40.0)


def _spikes_in_noise(fs: float, height: float, seed: int = 0) -> tuple:
    """400 s of white noise whose 1 ms means have unit SD, whatever ``fs``, with
    1 ms spikes of ``height`` every 0.5 s.
    """
    per_ms = int(fs / 1000)
    trace = np.random.default_rng(seed).normal(
        scale=np.sqrt(per_ms), size=400000 * per_ms
    )
    peaks = np.arange(1, 799) * 500 * per_ms
    trace[peaks[:, None] + np.arange(per_ms) - per_ms // 2] += height
    return trace, peaks


def _band_noise(n: int, seed: int) -> np.ndarray:
    """Unit-SD noise in ``BAND_HZ``: as autocorrelated as a real shift or trace."""
    sos = scipy.signal.butter(2, BAND_HZ, "bandpass", fs=FS_HZ, output="sos")
    y = scipy.signal.sosfiltfilt(sos, np.random.default_rng(seed).normal(size=n))
    return y / y.std()


def _axes(n: int, seed: int) -> dict[str, np.ndarray]:
    """X, Y and Z, where Y follows X a few samples late: axes share motion, out of phase."""
    x, y, z = (_band_noise(n, seed + k) for k in range(3))
    return {"X": x, "Y": 0.8 * np.roll(x, 6) + 0.6 * y, "Z": z}


def test_dprime_of_two_gaussians_is_their_separation_in_sds():
    rng = np.random.default_rng(0)
    bottom, top = rng.normal(0.0, 1.0, 20000), rng.normal(3.0, 1.0, 20000)
    assert abs(dprime(top, bottom) - 3.0) < 0.05
    assert math.isnan(dprime(np.zeros(0), bottom))
    assert math.isnan(dprime(np.ones(5), np.ones(5)))


def test_event_windows_sit_at_the_peak_and_before_its_rise():
    trace = np.arange(100, dtype=float)
    trace[50] = 1000.0
    d, n = event_dprime(trace, np.array([50]), FS_HZ)
    assert n == 1 and math.isnan(d)  # one event has no variance to pool
    rng = np.random.default_rng(1)
    base = rng.normal(size=3000)
    peaks = np.arange(100, 2900, 100)
    spiked = base.copy()
    spiked[peaks] += 4.0
    spiked_before = base.copy()
    spiked_before[peaks - 4] += 4.0
    assert event_dprime(spiked, peaks, FS_HZ)[0] > 2.5
    assert event_dprime(spiked_before, peaks, FS_HZ)[0] < -2.5


def test_only_isolated_events_count():
    trace = np.random.default_rng(2).normal(size=2000)
    # 2 has no room before it, 130 follows 100 inside the quiet time, 901 follows 900
    events = np.array([2, 100, 130, 500, 900, 901])
    assert event_dprime(trace, events, FS_HZ)[1] == 3


def test_dprime_does_not_depend_on_the_sampling_rate():
    for fs in (1000.0, 3000.0):
        trace, peaks = _spikes_in_noise(fs, 3.0)
        assert abs(event_dprime(trace, peaks, fs)[0] - 3.0) < 0.3


def test_coupling_reads_the_share_of_power_it_explains():
    n = int(120 * FS_HZ)
    motion = _axes(n, 0)
    driven = 0.7 * np.roll(motion["X"], 3) - 0.5 * motion["Y"] + 0.3 * motion["Z"]
    noise = _band_noise(n, 9) * driven.std()
    for gain, truth in ((0.5, 0.8), (1.0, 0.5), (2.0, 0.2)):
        coupling = motion_coupling(driven + gain * noise, motion, FS_HZ)
        assert abs(coupling.share - truth) < 0.08
        assert coupling.p < 0.01
        # the joint coherence is an R^2: it cannot pass 1, and no axis passes it
        assert coupling.coherence.max() <= 1.0
        assert all(
            np.all(a <= coupling.coherence + 1e-9) for a in coupling.axes.values()
        )


def test_a_trace_with_its_own_rhythm_is_not_coupled():
    n = int(120 * FS_HZ)
    motion = _axes(n, 0)
    coupling = motion_coupling(_band_noise(n, 20), motion, FS_HZ)
    assert coupling.p > 0.05 and coupling.share < 0.02
    assert np.all(coupling.chance > 0)


def test_the_opening_step_every_scan_shares_is_left_out():
    n = int(60 * FS_HZ)
    step = np.where(np.arange(n) < 0.5 * FS_HZ, 40.0, 0.0)
    motion = {k: v + step for k, v in _axes(n, 0).items()}
    assert motion_coupling(_band_noise(n, 20) + step, motion, FS_HZ).p > 0.05


def test_a_lagged_or_inverted_coupling_counts_the_same():
    n = int(60 * FS_HZ)
    motion = _axes(n, 0)
    coupling = motion_coupling(-np.roll(motion["Z"], 25), motion, FS_HZ)
    assert coupling.share > 0.8


def test_a_recording_too_short_to_read_has_no_coupling():
    n = int(11 * FS_HZ)
    assert motion_coupling(_band_noise(n, 1), _axes(n, 0), FS_HZ) is None


def test_the_band_starts_at_1_hz_at_a_rate_that_does_not_divide_evenly():
    fs = 2777.7777777777774
    n = int(30 * fs)
    y = np.random.default_rng(3).normal(size=n)
    coupling = motion_coupling(y, {"X": np.random.default_rng(4).normal(size=n)}, fs)
    assert abs(coupling.frequency[0] - 1.0) < 0.01
    assert abs(coupling.frequency[-1] - 50.0) < 0.01


def test_an_axis_that_never_moved_is_left_out():
    n = int(30 * FS_HZ)
    motion = {**_axes(n, 0), "Z": np.zeros(n)}
    assert sorted(motion_coupling(_band_noise(n, 1), motion, FS_HZ).axes) == ["X", "Y"]
