"""What can be read off traces sampled in time: their power spectrum, how they
correlate, what they do around events, how far those events stand out, and how
much of them their motion correction explains.

Every function takes plain arrays on one clock, so the same read-out runs on a
raw trace or on anything already filtered: a band-limited correlation is
:func:`correlation_matrix` of band-passed traces. Events come from
:func:`detect_events`, thresholded on a robust noise estimate so the events
themselves do not raise it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal

__all__ = [
    "MAD_TO_SD",
    "MotionCoupling",
    "correlation_matrix",
    "detect_events",
    "dprime",
    "event_dprime",
    "motion_coupling",
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
# a voltage spike fills about 1 ms and peaks about 3 ms after its onset (ASAP7y's
# rise); 50 ms is the quiet imgui_ophys's d' lesson asks of a voltage event
DPRIME_WINDOW_MS = 1.0
DPRIME_RISE_MS = 3.0
DPRIME_QUIET_MS = 50.0
# 1 s windows beat 2 and 4 s on real scans: fewer false calls, more hits, and more
# windows in a short recording. The first seconds hold RTMC's settling step, which
# every scan shares, and below 10 s a real coupling is mostly missed
COUPLING_WINDOW_S = 1.0
COUPLING_LOW_HZ = 1.0
COUPLING_HIGH_HZ = 50.0
COUPLING_SKIP_S = 2.0
COUPLING_MIN_S = 10.0
COUPLING_RATE_HZ = 400.0
# windows overlap this many times over, so a short recording still has many shifts
COUPLING_HOPS = 8


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


def dprime(top: np.ndarray, bottom: np.ndarray) -> float:
    """``(mean top - mean bottom) / sqrt(0.5 * (var top + var bottom))``, the
    sensitivity index of Zhang et al. 2023 (jGCaMP8), after McNicol 2004; NaN
    when a pool is empty or both are constant.
    """
    if top.size == 0 or bottom.size == 0:
        return float("nan")
    pooled = np.sqrt(0.5 * (top.var() + bottom.var()))
    return float((top.mean() - bottom.mean()) / pooled) if pooled > 0 else float("nan")


def event_dprime(trace: np.ndarray, events: np.ndarray, fs: float) -> tuple[float, int]:
    """``(d', n)`` of ``trace`` at the peaks ``events`` (sample indices): each
    isolated peak's mean over ``DPRIME_WINDOW_MS`` around it against the same
    length ending ``DPRIME_RISE_MS`` before it. A window's mean is one
    observation, so d' does not depend on the sampling rate. A peak needs
    ``DPRIME_QUIET_MS`` to itself beforehand and a whole window afterwards;
    one too near an end is left out.
    """
    events = np.sort(np.asarray(events, dtype=np.int64))
    width = max(int(round(DPRIME_WINDOW_MS * fs / 1000)), 1)
    rise = int(round(DPRIME_RISE_MS * fs / 1000))
    gaps = np.diff(events)
    alone = np.ones(events.size, dtype=bool)
    alone[1:] &= gaps >= DPRIME_QUIET_MS * fs / 1000
    alone[:-1] &= gaps >= width
    peaks = events[alone]
    peaks = peaks[(peaks >= rise + width) & (peaks + width <= trace.size)]
    window = peaks[:, None] + np.arange(width)  ## (n_events, width)
    top = trace[window - width // 2].mean(axis=1)
    bottom = trace[window - rise - width].mean(axis=1)
    return dprime(top, bottom), len(peaks)


@dataclass
class MotionCoupling:
    """How much of one trace its recording's motion correction explains.

    ``share`` is the part of the trace's power from ``COUPLING_LOW_HZ`` to
    ``COUPLING_HIGH_HZ`` the motion explains, past what the same trace gets moved
    along in time, so it reads as a fraction of that power. ``p`` ranks the real
    alignment among those moved traces. A coupling says the two move together,
    never which moved which.
    """

    frequency: np.ndarray  ## (n_bins,) Hz
    coherence: np.ndarray  ## (n_bins,) all axes together
    axes: dict[str, np.ndarray]  ## label: (n_bins,) that axis alone
    chance: np.ndarray  ## (n_bins,) 95th percentile of the moved traces' coherence
    share: float
    p: float


def motion_coupling(
    trace: np.ndarray, motion: dict[str, np.ndarray], fs: float
) -> MotionCoupling | None:
    """The coherence of ``trace`` with ``motion`` (label: shift, on the trace's
    clock), None with under ``COUPLING_MIN_S`` to read.

    Coherence compares each frequency with itself, so a slow drift in both adds
    nothing to their spikes' band, and any fixed lag or sign is the same to it.
    Every window's spectrum is taken once; the trace's windows moved along by whole
    hops, wrapping round, are the null, which keeps both signals' own rhythm and
    only breaks their alignment. A plain correlation's p-value assumes independent
    samples and called 78% of trace and motion pairs from different scans coupled.
    """
    start = int(COUPLING_SKIP_S * fs)
    if (trace.size - start) / fs < COUPLING_MIN_S:
        return None
    moving = {label: y for label, y in motion.items() if np.ptp(y) > 0}
    factor = max(int(fs // COUPLING_RATE_HZ), 1)
    rows = np.stack([trace, *moving.values()]).astype(np.float64)[:, start:]
    rows = rows[:, : rows.shape[1] // factor * factor]
    rows = rows.reshape(len(rows), -1, factor).mean(axis=2)
    size = int(round(COUPLING_WINDOW_S * fs / factor))
    hop = max(size // COUPLING_HOPS, 1)
    windows = np.lib.stride_tricks.sliding_window_view(rows, size, axis=1)[:, ::hop]
    spectra = np.fft.rfft(
        scipy.signal.detrend(windows, axis=2) * np.hanning(size), axis=2
    )  ## (n_rows, n_windows, n_frequencies)
    frequency = np.fft.rfftfreq(size, factor / fs)
    ## NOTE: whole-sample windows put bins a hair off whole Hz, 0.9999 at 2777.8 Hz
    half_bin = frequency[1] / 2
    band = (frequency > COUPLING_LOW_HZ - half_bin) & (
        frequency < COUPLING_HIGH_HZ + half_bin
    )
    x, m = spectra[0][:, band], spectra[1:][:, :, band]
    n_windows = x.shape[0]
    ## NOTE: cross[a, s] is the trace moved s windows along against axis a; x before conj(m) is what keeps R^2 <= 1
    cross = (
        np.fft.ifft(np.fft.fft(x, axis=0) * np.conj(np.fft.fft(m, axis=1)), axis=1)
        / n_windows
    )  ## (n_axes, n_windows, n_bins)
    power = (np.abs(x) ** 2).mean(axis=0)
    inner = np.einsum("akf,bkf->fab", m, np.conj(m)) / n_windows
    joint = (
        np.einsum("asf,fab,bsf->sf", cross, np.linalg.pinv(inner), np.conj(cross)).real
        / power
    )  ## (n_windows, n_bins)
    clear = -(-size // hop)  ## nearer shifts share samples with the real one
    null = joint[clear : n_windows - clear + 1]
    shares = joint @ power / power.sum()
    null_shares = shares[clear : n_windows - clear + 1]
    return MotionCoupling(
        frequency=frequency[band],
        coherence=joint[0],
        axes={
            label: np.abs(cross[a, 0]) ** 2 / (power * inner[:, a, a].real)
            for a, label in enumerate(moving)
        },
        chance=np.percentile(null, 95, axis=0),
        share=max(float(shares[0] - np.median(null_shares)), 0.0),
        p=float((1 + np.sum(null_shares >= shares[0])) / (1 + null_shares.size)),
    )
