"""dF/F over a baseline, on ``(K, T)`` float arrays.

The rolling max-min baseline sized in seconds and the static percentile one
(the Traces panel's), and :func:`dfof_zscore`, the voltage pipeline's stages
0-1 with a choice of baseline (:data:`BASELINES`): the archive's Gaussian, a
rolling percentile, the rolling max-min, a bleaching exponential, the mean of
a fixed window (MESc's F0) or none.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import (
    gaussian_filter1d,
    maximum_filter1d,
    minimum_filter1d,
    percentile_filter,
)
from scipy.optimize import OptimizeWarning, curve_fit

__all__ = [
    "BASELINES",
    "NOISE_ESTIMATES",
    "baseline",
    "dfof_maxmin",
    "dfof_percentile",
    "dfof_zscore",
    "exponential_baseline",
    "maxmin_baseline",
    "robust_sd",
    "rolling_percentile",
]

# how F0 is found under dF/F = (F - F0) / F0: a Gaussian low-pass of F (the
# archive's, sigma in samples), a rolling percentile or max-min of F (window in
# seconds), an exponential bleaching fit, the mean over a fixed time window, or
# none (the z-score is taken of F itself)
BASELINES = ("gaussian", "percentile", "maxmin", "exponential", "window", "none")
# what the z-score divides by: the SD of dF/F (the archive's) or 1.4826 x the
# median absolute deviation, which spikes do not inflate
NOISE_ESTIMATES = ("sd", "mad")


def maxmin_baseline(
    F: np.ndarray, fs: float, window_s: float = 5.0, sigma_s: float = 0.05
) -> np.ndarray:
    """Rolling max-min baseline, sized in seconds via ``fs``.

    suite2p's baseline: smooth, then the rolling maximum of the rolling
    minimum, so F0 follows the bottom of F and sits under it. Sized in
    seconds rather than a fixed frame count: a line-scan's frame rate (~1-2.5
    kHz) is one to two orders of magnitude higher than a raster-scanned
    movie's (~10-30 Hz), so a fixed frame-count window would be the wrong
    number of seconds here. Uses ``scipy.ndimage``'s O(T) sliding max/min
    filters rather than a per-frame python loop, since a line-scan run has
    far more timepoints.
    """
    fs = float(fs)
    window = max(3, int(round(window_s * fs)))
    sigma = max(0.5, sigma_s * fs)
    smoothed = gaussian_filter1d(F, sigma=sigma, axis=1)
    rolled_min = minimum_filter1d(smoothed, size=window, axis=1, mode="nearest")
    return maximum_filter1d(rolled_min, size=window, axis=1, mode="nearest")


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


def rolling_percentile(
    F: np.ndarray, fs: float, window_s: float = 5.0, percentile: float = 20.0
) -> np.ndarray:
    """Rolling ``percentile`` of each row over ``window_s`` seconds.

    Taken on 10 ms bin means and interpolated back, so a kilohertz trace
    costs what a 100 Hz one does. The ends are mirrored, not repeated: an
    AOD scan's first few milliseconds are dark, and a repeated first bin
    would fill half a window and pull the baseline to zero.
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    T = F.shape[1]
    b = max(1, int(round(0.01 * float(fs))))
    nb = int(np.ceil(T / b))
    padded = np.pad(F, ((0, 0), (0, nb * b - T)), mode="edge")
    bins = padded.reshape(F.shape[0], nb, b).mean(axis=2)
    size = max(1, int(round(window_s * float(fs) / b)))
    low = percentile_filter(bins, percentile, size=(1, size), mode="reflect")
    centers = (np.arange(nb) + 0.5) * b - 0.5
    t = np.arange(T)
    return np.stack([np.interp(t, centers, row) for row in low])


def _decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def exponential_baseline(F: np.ndarray, fs: float) -> np.ndarray:
    """``a * exp(-t / tau) + c`` fitted to each row: photobleaching.

    Fitted on at most 2000 bin means; a row the fit cannot follow (no decay,
    a rise) gets its least-squares line instead.
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    T = F.shape[1]
    t = np.arange(T) / float(fs)
    b = max(1, T // 2000)
    nb = T // b
    tb = t[: nb * b].reshape(nb, b).mean(axis=1)
    out = np.empty_like(F)
    for i, row in enumerate(F):
        yb = row[: nb * b].reshape(nb, b).mean(axis=1)
        span = float(t[-1]) if T > 1 else 1.0
        p0 = (float(yb[0] - yb[-1]), span / 3.0, float(yb[-1]))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                p, _ = curve_fit(
                    _decay,
                    tb,
                    yb,
                    p0=p0,
                    bounds=(
                        [-np.inf, 1.0 / float(fs), -np.inf],
                        [np.inf, 100 * span, np.inf],
                    ),
                    maxfev=4000,
                )
            out[i] = _decay(t, *p)
        except (RuntimeError, ValueError):
            out[i] = np.polyval(np.polyfit(t, row, 1), t)
    return out


def baseline(
    F: np.ndarray,
    fs: float,
    method: str = "gaussian",
    *,
    sigma: float = 1500.0,
    window_s: float = 5.0,
    percentile: float = 20.0,
    f0_window_s=(0.0, 1.0),
) -> np.ndarray | None:
    """F0 of each row by ``method`` (:data:`BASELINES`); None for ``none``.

    ``sigma`` is the Gaussian's in samples; ``window_s`` the rolling
    windows' in seconds (``percentile`` for the percentile one);
    ``f0_window_s`` the ``(start, stop)`` seconds the ``window`` method averages.
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    if method == "gaussian":
        return gaussian_filter1d(F, sigma=float(sigma), axis=1)
    if method == "percentile":
        return rolling_percentile(F, fs, window_s, percentile)
    if method == "maxmin":
        return maxmin_baseline(F, fs, window_s)
    if method == "exponential":
        return exponential_baseline(F, fs)
    if method == "window":
        T = F.shape[1]
        i0 = min(T - 1, max(0, int(round(float(f0_window_s[0]) * float(fs)))))
        i1 = min(T, max(i0 + 1, int(round(float(f0_window_s[1]) * float(fs)))))
        return np.repeat(F[:, i0:i1].mean(axis=1, keepdims=True), T, axis=1)
    if method == "none":
        return None
    raise ValueError(f"baseline method must be one of {BASELINES}, got {method!r}")


def robust_sd(x: np.ndarray) -> np.ndarray:
    """1.4826 x the median absolute deviation of each row, ``(K, 1)``."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    med = np.median(x, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=1, keepdims=True)


def dfof_zscore(
    F: np.ndarray,
    fs: float,
    *,
    method: str = "gaussian",
    sigma: float = 1500.0,
    window_s: float = 5.0,
    percentile: float = 20.0,
    f0_window_s=(0.0, 1.0),
    n_startup: int = 1000,
    sigma_baseline: float = 5000.0,
    noise: str = "sd",
    negative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """``(dfof, z)`` of each row: the voltage pipeline's stages 0-1.

    dF/F over the ``method`` baseline (see :func:`baseline`; ``none`` keeps
    F), the first ``n_startup`` samples replaced by the mean of the rest
    (filter start-up), then ``z = (dfof - slow) / noise`` with ``slow`` a
    Gaussian of ``sigma_baseline`` samples (0 skips it) and ``noise`` the SD
    of dF/F or the robust SD of what is left (:data:`NOISE_ESTIMATES`),
    negated with ``negative`` (a dimming indicator's events point up). With
    the defaults this is ``vnoiser.preprocess.domain_zscore`` exactly. A
    baseline at or below zero gives dF/F zero there.
    """
    if noise not in NOISE_ESTIMATES:
        raise ValueError(f"noise must be one of {NOISE_ESTIMATES}, got {noise!r}")
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    f0 = baseline(
        F,
        fs,
        method,
        sigma=sigma,
        window_s=window_s,
        percentile=percentile,
        f0_window_s=f0_window_s,
    )
    if f0 is None:
        dfof = F.copy()
    else:
        dfof = np.zeros_like(F)
        np.divide(F - f0, f0, out=dfof, where=f0 > 0)
    n = int(n_startup)
    if n > 0 and dfof.shape[1] > n:
        dfof[:, :n] = dfof[:, n:].mean(axis=1, keepdims=True)
    slow = (
        gaussian_filter1d(dfof, sigma=float(sigma_baseline), axis=1)
        if sigma_baseline and float(sigma_baseline) > 0
        else 0.0
    )
    residual = dfof - slow
    scale = dfof.std(axis=1, keepdims=True) if noise == "sd" else robust_sd(residual)
    z = np.zeros_like(residual)
    np.divide(residual, scale, out=z, where=scale > 0)
    if negative:
        z = -z
    return dfof, z
