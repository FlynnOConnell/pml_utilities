"""Resolution pyramids for OME-NGFF 0.5 multiscale zarr images (readable by napari-ome-zarr)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DownsampleMethod = Literal[
    "mean", "nearest", "gaussian", "local_mean", "median", "mode"
]


@dataclass
class PyramidLevel:
    """single resolution level in a pyramid."""

    level: int  # 0 = full resolution
    shape: tuple[int, ...]
    scale: tuple[float, ...]  # per axis, relative to level 0
    path: str  # zarr array path, str(level)


@dataclass
class PyramidConfig:
    """
    Configuration for pyramid generation.

    Parameters
    ----------
    max_layers : int
        maximum number of additional resolution levels beyond level 0.
        default 4 means levels 0, 1, 2, 3, 4 (5 total).
    scale_factors : tuple[int, ...]
        per-axis downsampling factors. default (1, 1, 2, 2) for TZYX
        means T and Z unchanged, Y and X downsampled by 2 per level.
    method : DownsampleMethod
        downsampling method (``downsample_block``); "mean" by default.
    min_size : int
        stop adding levels when any spatial dimension falls below this.
        default 64 pixels.
    """

    max_layers: int = 4
    scale_factors: tuple[int, ...] = (1, 1, 2, 2)
    method: DownsampleMethod = "mean"
    min_size: int = 64

    def get_scale_factors_for_ndim(self, ndim: int) -> tuple[int, ...]:
        """Scale factors for an array of ``ndim``: the configured ones when they fit, else Y and X halved."""
        if ndim == len(self.scale_factors):
            return self.scale_factors
        if ndim == 3:
            return (1, 2, 2)
        if ndim == 4:
            return self.scale_factors
        if ndim == 5:
            return (1, 1, 1, 2, 2)
        return (1,) * (ndim - 2) + (2, 2)


def compute_pyramid_shapes(
    base_shape: tuple[int, ...],
    config: PyramidConfig | None = None,
) -> list[PyramidLevel]:
    """Every level's shape and scale for ``base_shape``, level 0 first, without generating data.

    Levels stop before Y or X (the last two axes) falls below ``config.min_size``.
    """
    if config is None:
        config = PyramidConfig()

    ndim = len(base_shape)
    scale_factors = config.get_scale_factors_for_ndim(ndim)

    levels = []
    current_shape = base_shape
    base_scale = (1.0,) * ndim

    for level in range(config.max_layers + 1):
        cumulative_scale = tuple(
            base_scale[i] * (scale_factors[i] ** level) for i in range(ndim)
        )

        levels.append(
            PyramidLevel(
                level=level,
                shape=current_shape,
                scale=cumulative_scale,
                path=str(level),
            )
        )

        next_shape = tuple(
            max(1, s // scale_factors[i]) for i, s in enumerate(current_shape)
        )
        if any(next_shape[i] < config.min_size for i in range(-2, 0)):
            break

        current_shape = next_shape

    return levels


def downsample_block(
    data: np.ndarray,
    factors: tuple[int, ...],
    method: DownsampleMethod = "mean",
) -> np.ndarray:
    """Downsample ``data`` by one integer factor per axis.

    ``method``: ``mean`` average pooling (intensity data), ``median`` windowed
    median (webknossos' default for intensity), ``mode`` windowed mode
    (webknossos' default for labels and masks), ``nearest`` nearest neighbor,
    ``gaussian`` blur then subsample, ``local_mean`` skimage's local mean.
    """
    if len(factors) != data.ndim:
        raise ValueError(f"factors {factors} must match data.ndim {data.ndim}")

    if all(f == 1 for f in factors):
        return data

    if method == "nearest":
        slices = tuple(slice(None, None, f) for f in factors)
        return data[slices].copy()

    if method == "mean":
        return _downsample_mean(data, factors)

    if method == "median":
        return _downsample_median(data, factors)

    if method == "mode":
        return _downsample_mode(data, factors)

    if method == "gaussian":
        return _downsample_gaussian(data, factors)

    if method == "local_mean":
        return _downsample_local_mean(data, factors)

    raise ValueError(f"unknown downsampling method: {method}")


def _downsample_mean(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Downsample using mean pooling: trim to whole blocks, split each axis into (blocks, factor), average the factor axes."""
    out_shape = tuple(s // f for s, f in zip(data.shape, factors, strict=True))
    new_shape = []
    for s, f in zip(out_shape, factors, strict=True):
        new_shape.extend([s, f])
    axes_to_mean = tuple(range(1, 2 * len(factors), 2))
    slices = tuple(slice(None, s * f) for s, f in zip(out_shape, factors, strict=True))
    reshaped = data[slices].reshape(new_shape)
    return reshaped.mean(axis=axes_to_mean).astype(data.dtype)


def _downsample_median(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Downsample using the windowed median (matches webknossos MEDIAN).

    Same block grouping as ``_downsample_mean``; ``np.median`` reduces over
    the per-axis factor axes. Median is an order statistic, so the grouping
    order within a block is irrelevant. Cast back to the source dtype, which
    truncates toward zero on even-count blocks exactly as webknossos does.
    """
    out_shape = tuple(s // f for s, f in zip(data.shape, factors, strict=True))
    new_shape = []
    for s, f in zip(out_shape, factors, strict=True):
        new_shape.extend([s, f])
    axes_to_reduce = tuple(range(1, 2 * len(factors), 2))
    slices = tuple(slice(None, s * f) for s, f in zip(out_shape, factors, strict=True))
    reshaped = data[slices].reshape(new_shape)
    return np.median(reshaped, axis=axes_to_reduce).astype(data.dtype)


def _downsample_mode(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Downsample using the windowed mode (matches webknossos MODE).

    Best for label/segmentation data: each output voxel is the most frequent
    value in its block, so no value that is not present in the source is ever
    produced. Ties go to the value seen first in raster order, identical to
    webknossos' numba kernel.
    """
    out_shape = tuple(s // f for s, f in zip(data.shape, factors, strict=True))
    new_shape = []
    for s, f in zip(out_shape, factors, strict=True):
        new_shape.extend([s, f])
    slices = tuple(slice(None, s * f) for s, f in zip(out_shape, factors, strict=True))
    reshaped = data[slices].reshape(new_shape)

    nd = len(factors)
    data_axes = tuple(range(0, 2 * nd, 2))
    factor_axes = tuple(range(1, 2 * nd, 2))
    window = int(np.prod(factors))
    moved = np.ascontiguousarray(np.transpose(reshaped, data_axes + factor_axes))
    flat = moved.reshape(-1, window)
    result = _mode_rows(flat)
    return result.reshape(out_shape).astype(data.dtype)


def _mode_rows_py(flat: np.ndarray) -> np.ndarray:
    """Row-wise mode over a 2D ``(n_blocks, window)`` array.

    Counts values in first-seen order and returns the value with the highest
    count, ties kept at the first-seen value (numba-compiled on first use).
    """
    m, w = flat.shape
    out = np.empty(m, dtype=flat.dtype)
    values = np.empty(w, dtype=flat.dtype)
    counts = np.empty(w, dtype=np.int64)
    for r in range(m):
        n = 0
        for c in range(w):
            v = flat[r, c]
            hit = False
            for k in range(n):
                if values[k] == v:
                    counts[k] += 1
                    hit = True
                    break
            if not hit:
                values[n] = v
                counts[n] = 1
                n += 1
        best = 0
        for k in range(1, n):
            if counts[k] > counts[best]:
                best = k
        out[r] = values[best]
    return out


_mode_rows_compiled = None


def _mode_rows(flat: np.ndarray) -> np.ndarray:
    """numba-compiled :func:`_mode_rows_py`, imported lazily on first use."""
    global _mode_rows_compiled
    if _mode_rows_compiled is None:
        import numba

        _mode_rows_compiled = numba.njit(cache=True, nogil=True)(_mode_rows_py)
    return _mode_rows_compiled(flat)


def _downsample_gaussian(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Downsample with gaussian blur for antialiasing."""
    from scipy.ndimage import gaussian_filter

    sigma = tuple(0.5 * (f - 1) for f in factors)
    blurred = gaussian_filter(data.astype(np.float32), sigma=sigma)
    slices = tuple(slice(None, None, f) for f in factors)
    return blurred[slices].astype(data.dtype)


def _downsample_local_mean(data: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Downsample using skimage's ``downscale_local_mean``."""
    from skimage.transform import downscale_local_mean

    return downscale_local_mean(data, factors).astype(data.dtype)
