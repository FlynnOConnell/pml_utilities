"""d' and the motion coupling of a trace, on synthetic traces with a known answer."""

from __future__ import annotations

import math

import numpy as np

from mbo_utilities.analysis.trace_readouts import dprime, event_dprime

FS_HZ = 1000.0


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
