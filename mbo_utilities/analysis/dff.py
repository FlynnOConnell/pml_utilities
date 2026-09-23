"""dF/F over a baseline: the rolling max-min baseline sized in seconds, and
the static percentile one, both on ``(K, T)`` float arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

__all__ = ["dfof_maxmin", "dfof_percentile", "maxmin_baseline"]


def maxmin_baseline(
    F: np.ndarray, fs: float, window_s: float = 5.0, sigma_s: float = 0.05
) -> np.ndarray:
    """Rolling max-min baseline, sized in seconds via ``fs``.

    Same two-pass smooth -> rolling-max -> rolling-min baseline as a
    suite2p-style dF/F, but sized in seconds rather than a fixed frame
    count: a line-scan's frame rate (~1-2.5 kHz) is one to two orders of
    magnitude higher than a raster-scanned movie's (~10-30 Hz), so a fixed
    frame-count window would be the wrong number of seconds here. Uses
    ``scipy.ndimage``'s O(T) sliding max/min filters rather than a per-frame
    python loop, since a line-scan run has far more timepoints.
    """
    fs = float(fs)
    window = max(3, int(round(window_s * fs)))
    sigma = max(0.5, sigma_s * fs)
    smoothed = gaussian_filter1d(F, sigma=sigma, axis=1)
    rolled_max = maximum_filter1d(smoothed, size=window, axis=1, mode="nearest")
    return minimum_filter1d(rolled_max, size=window, axis=1, mode="nearest")


def dfof_maxmin(
    F: np.ndarray, fs: float, window_s: float = 5.0, sigma_s: float = 0.05
) -> np.ndarray:
    """Rolling max-min baseline dF/F as a fraction, no neuropil term (see
    :func:`maxmin_baseline`); a baseline at or below zero gives zero there.
    """
    F = np.asarray(F, np.float32)
    baseline = maxmin_baseline(F, fs, window_s, sigma_s)
    out = np.zeros_like(F)
    np.divide(F - baseline, baseline, out=out, where=baseline > 0)
    return out


def dfof_percentile(F: np.ndarray, percentile: float = 20.0) -> np.ndarray:
    """dF/F as a fraction over a static per-row percentile baseline, the way
    ``lbm_suite2p_python`` plots a suite2p trace; an all-zero row stays zero.
    """
    F = np.asarray(F, np.float32)
    out = np.zeros_like(F)
    if not F.size:
        return out
    f0 = np.percentile(F, percentile, axis=1, keepdims=True)
    f0 = np.where(f0 > 1e-6, f0, 1e-6).astype(np.float32)
    live = np.any(F != 0, axis=1, keepdims=True)
    return np.where(live, (F - f0) / f0, 0.0).astype(np.float32)
