"""What can be read off traces sampled in time: their power spectrum, how they
correlate, and what they do around events.

Every function takes plain arrays on one clock, so the same read-out runs on a
raw trace or on anything already filtered: a band-limited correlation is
:func:`correlation_matrix` of band-passed traces. Events come from
:func:`detect_events`, thresholded on a robust noise estimate so the events
themselves do not raise it.
"""

from __future__ import annotations

import numpy as np
import scipy.signal

__all__ = [
    "MAD_TO_SD",
    "correlation_matrix",
    "detect_events",
    "noise_floor_hz",
    "power_spectrum",
    "triggered_average",
]

# the median absolute deviation of a Gaussian times this is its SD
MAD_TO_SD = 1.4826
# the top of the spectrum, as a share of its bins, that is taken as noise
FLOOR_SHARE = 0.25
# 1 Hz bins over minutes scatter by a few percent, so this is well clear of them
FLOOR_FACTOR = 1.25
SMOOTH_BINS = 5


def power_spectrum(
    y: np.ndarray, fs: float, resolution_hz: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """``(frequency, power)`` of ``y`` along its last axis by Welch's method,
    in segments ``1 / resolution_hz`` seconds long; power is per Hz, in the
    square of ``y``'s unit.
    """
    segment = min(int(round(fs / resolution_hz)), y.shape[-1])
    return scipy.signal.welch(y, fs=fs, nperseg=segment, detrend="constant", axis=-1)


def noise_floor_hz(frequency: np.ndarray, power: np.ndarray) -> float:
    """The frequency a spectrum first falls within ``FLOOR_FACTOR`` of the
    median of its top ``FLOOR_SHARE``, the band a white noise floor fills.
    """
    smooth = np.convolve(power[1:], np.ones(SMOOTH_BINS) / SMOOTH_BINS, mode="valid")
    floor = np.median(power[int(len(power) * (1 - FLOOR_SHARE)) :])
    return float(frequency[1 + np.argmax(smooth <= FLOOR_FACTOR * floor)])


def correlation_matrix(traces: np.ndarray) -> np.ndarray:
    """Pearson r of ``(n_traces, n_samples)``, ``(n_traces, n_traces)``; a
    flat trace correlates with nothing, so its row and column are 0.
    """
    centred = traces - traces.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=1)
    norm = np.where(norm > 0, norm, np.inf)
    unit = centred / norm[:, None]
    return unit @ unit.T


def detect_events(
    y: np.ndarray,
    fs: float,
    threshold_sd: float = 4.0,
    gap_ms: float = 5.0,
    polarity: int = 1,
) -> np.ndarray:
    """Sample indices of ``y``'s peaks past ``threshold_sd`` robust SDs of its
    median, at least ``gap_ms`` apart; ``polarity`` -1 finds troughs.
    """
    if polarity not in (1, -1):
        raise ValueError(f"polarity must be 1 or -1, got {polarity!r}")
    centred = polarity * (y - np.median(y))
    sd = MAD_TO_SD * np.median(np.abs(centred))
    if sd == 0:
        return np.zeros(0, np.int64)
    distance = max(int(round(gap_ms * fs / 1000)), 1)
    peaks, _ = scipy.signal.find_peaks(
        centred, height=threshold_sd * sd, distance=distance
    )
    return peaks


def triggered_average(
    traces: np.ndarray, events: np.ndarray, before: int, after: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(mean, sem, n)`` of ``(n_traces, n_samples)`` around ``events``, each
    ``(n_traces, before + after + 1)``; an event too near an end is left out.
    Each snippet keeps its own level: the mean over the first ``before // 2``
    samples is taken off, so a slow drift does not add to the average.
    """
    n_samples = traces.shape[1]
    kept = events[(events >= before) & (events + after < n_samples)]
    if len(kept) == 0:
        empty = np.zeros((traces.shape[0], before + after + 1))
        return empty, empty, 0
    window = kept[:, None] + np.arange(-before, after + 1)  ## (n_events, n_lags)
    snippets = traces[:, window].astype(np.float64)  ## (n_traces, n_events, n_lags)
    snippets -= snippets[:, :, : max(before // 2, 1)].mean(axis=2, keepdims=True)
    sem = snippets.std(axis=1) / np.sqrt(len(kept))
    return snippets.mean(axis=1), sem, len(kept)
