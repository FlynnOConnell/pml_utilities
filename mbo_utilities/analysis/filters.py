"""Filters for traces sampled in time, sized in Hz and milliseconds so one
setting means the same thing on a 30 Hz movie and a 3 kHz line scan.

:class:`FilterSettings` names one filter and :func:`apply_filter` runs it
along the last axis. Every filter keeps the trace's length and phase: the
Butterworth ones run forwards and backwards (``sosfiltfilt``), the notch
through ``filtfilt``, the smoothers are symmetric. :func:`gaussian` is
``scipy.ndimage.gaussian_filter1d`` through an FFT once the kernel is long,
where the direct sum is slow: a 5 s Gaussian at 2.8 kHz is a 111k-tap kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import scipy.signal

__all__ = ["FILTERS", "FilterSettings", "apply_filter", "describe", "gaussian"]

FILTERS = ("none", "lowpass", "highpass", "bandpass", "notch", "gaussian", "mean")
# above this sigma (samples) the FFT is faster than scipy.ndimage's direct sum
FFT_SIGMA = 64.0
TRUNCATE = 4.0


@dataclass
class FilterSettings:
    """One filter: ``kind`` (:data:`FILTERS`), the band edges ``low_hz`` and
    ``high_hz`` (``lowpass`` keeps below ``high_hz``, ``highpass`` above
    ``low_hz``, ``bandpass`` between), the Butterworth ``order``, the
    ``notch_hz`` line and its quality ``notch_q``, and ``width_ms``: the
    Gaussian's sigma or the moving mean's length.
    """

    kind: str = "none"
    low_hz: float = 1.0
    high_hz: float = 100.0
    order: int = 4
    notch_hz: float = 60.0
    notch_q: float = 30.0
    width_ms: float = 2.0

    def __post_init__(self):
        if self.kind not in FILTERS:
            raise ValueError(f"filter must be one of {FILTERS}, got {self.kind!r}")


def gaussian(x: np.ndarray, sigma: float, axis: int = -1) -> np.ndarray:
    """``scipy.ndimage.gaussian_filter1d(x, sigma, axis)`` (reflect edges,
    truncated at 4 sigma), through an FFT when ``sigma`` is over ``FFT_SIGMA``.
    """
    x = np.asarray(x, np.float64)
    if sigma <= FFT_SIGMA:
        return scipy.ndimage.gaussian_filter1d(x, sigma, axis=axis, truncate=TRUNCATE)
    radius = int(TRUNCATE * float(sigma) + 0.5)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / float(sigma)) ** 2)
    kernel /= kernel.sum()
    x = np.moveaxis(x, axis, -1)
    pad = [(0, 0)] * (x.ndim - 1) + [(radius, radius)]
    ## NOTE: ndimage's "reflect" repeats the edge sample, which is numpy's "symmetric"
    padded = np.pad(x, pad, mode="symmetric")
    kernel = kernel.reshape((1,) * (x.ndim - 1) + (-1,))
    out = scipy.signal.oaconvolve(padded, kernel, mode="valid", axes=-1)
    return np.moveaxis(out, -1, axis)


def apply_filter(y: np.ndarray, fs: float, settings: FilterSettings) -> np.ndarray:
    """``y`` filtered along its last axis, sampled at ``fs`` Hz.

    Raises
    ------
    ValueError
        If a band edge is not between 0 and the Nyquist rate ``fs / 2``.
    """
    y = np.asarray(y, np.float64)
    kind = settings.kind
    if kind == "none":
        return y
    fs = float(fs)
    nyquist = fs / 2
    if kind in ("lowpass", "highpass", "bandpass", "notch"):
        edges = {
            "lowpass": [settings.high_hz],
            "highpass": [settings.low_hz],
            "bandpass": [settings.low_hz, settings.high_hz],
            "notch": [settings.notch_hz],
        }[kind]
        if not all(0 < e < nyquist for e in edges):
            raise ValueError(
                f"{kind} edges {edges} Hz must be between 0 and {nyquist:g} Hz "
                f"(half the {fs:g} Hz rate)"
            )
        if kind == "bandpass" and not edges[0] < edges[1]:
            raise ValueError(f"bandpass needs low < high, got {edges} Hz")
    if kind == "notch":
        b, a = scipy.signal.iirnotch(settings.notch_hz, settings.notch_q, fs=fs)
        return scipy.signal.filtfilt(b, a, y, axis=-1)
    if kind in ("lowpass", "highpass", "bandpass"):
        cutoff = {
            "lowpass": settings.high_hz,
            "highpass": settings.low_hz,
            "bandpass": [settings.low_hz, settings.high_hz],
        }[kind]
        sos = scipy.signal.butter(
            int(settings.order), cutoff, btype=kind, fs=fs, output="sos"
        )
        return scipy.signal.sosfiltfilt(sos, y, axis=-1)
    samples = settings.width_ms * 1e-3 * fs
    if kind == "gaussian":
        return gaussian(y, max(samples, 0.5), axis=-1)
    return scipy.ndimage.uniform_filter1d(
        y, max(int(round(samples)), 1), axis=-1, mode="nearest"
    )


def describe(settings: FilterSettings) -> str:
    """The filter in a few words: ``low-pass 100 Hz``, ``2 ms Gaussian``."""
    s = settings
    return {
        "none": "unfiltered",
        "lowpass": f"low-pass {s.high_hz:g} Hz",
        "highpass": f"high-pass {s.low_hz:g} Hz",
        "bandpass": f"band-pass {s.low_hz:g}-{s.high_hz:g} Hz",
        "notch": f"{s.notch_hz:g} Hz notch",
        "gaussian": f"{s.width_ms:g} ms Gaussian",
        "mean": f"{s.width_ms:g} ms mean",
    }[s.kind]
