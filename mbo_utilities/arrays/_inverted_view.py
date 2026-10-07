"""Read-time deflection inversion over any 5D TCZYX lazy array.

Frame ``t`` is ``2 * mean - source[t]``, ``mean`` the source's per-pixel
temporal mean, so an indicator that dims on activity (ASAP, Voltron) reads
positive-going while keeping its baseline. It is the viewer's Invert
Deflection as data: a pipeline or writer handed this view processes the
flipped movie. The mean comes from the whole source, computed on the first
read, so a frame window processes the same values the full recording would.
"""

from __future__ import annotations

import numpy as np

from mbo_utilities.arrays._base import _normalize_key, temporal_mean
from mbo_utilities.arrays._registration import _TCZYX, _validated_tczyx_shape
from mbo_utilities.arrays.features._frame_average import INVERT_DEFLECTION_KEY
from mbo_utilities.lazy_array import LazyArray

__all__ = ["InvertedDeflectionView", "invert_deflection"]


class InvertedDeflectionView(LazyArray):
    """5D TCZYX float32 view of ``source`` flipped about its temporal mean.

    ``mean`` is ``(C, Z, Y, X)``; None computes it from ``source`` on first
    read (``arrays._base.temporal_mean``).
    """

    def __init__(self, source, mean: np.ndarray | None = None):
        _validated_tczyx_shape(source)
        self._source = source
        self._mean = None if mean is None else np.asarray(mean, dtype=np.float32)

    @property
    def source(self):
        return self._source

    @property
    def _arr(self):
        return self._source

    @property
    def invert_deflection(self) -> bool:
        return True

    @property
    def mean(self) -> np.ndarray:
        """The ``(C, Z, Y, X)`` image every frame is flipped about."""
        if self._mean is None:
            self._mean = temporal_mean(self._source)
        return self._mean

    @property
    def reader_kwargs(self) -> dict:
        kwargs = dict(getattr(self._source, "reader_kwargs", None) or {})
        kwargs[INVERT_DEFLECTION_KEY] = True
        return kwargs

    @property
    def dtype(self):
        return np.dtype(np.float32)

    @property
    def dims(self) -> tuple[str, ...]:
        return _TCZYX

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        return _validated_tczyx_shape(self._source)

    @property
    def metadata(self) -> dict:
        meta = dict(getattr(self._source, "metadata", None) or {})
        meta[INVERT_DEFLECTION_KEY] = True
        meta["processing_history"] = [
            *(meta.get("processing_history") or []),
            {"step": INVERT_DEFLECTION_KEY},
        ]
        return meta

    @metadata.setter
    def metadata(self, value):
        meta = dict(value or {})
        meta.pop(INVERT_DEFLECTION_KEY, None)
        history = meta.get("processing_history")
        if history:
            meta["processing_history"] = [
                h
                for h in history
                if not (isinstance(h, dict) and h.get("step") == INVERT_DEFLECTION_KEY)
            ]
        self._source.metadata = meta

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        key = key + (slice(None),) * (5 - len(key))
        data = np.asarray(self._source[key], dtype=np.float32)
        return 2 * self.mean[key[1:]] - data

    def __array__(self, dtype=None, copy=None):
        frame = np.asarray(self[0, 0, 0])
        return frame if dtype is None else frame.astype(dtype)

    def __getattr__(self, name):
        # filenames, source_path, roi, ... come from the source; underscore
        # names are not forwarded so __init__ stays recursion-safe
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_source"), name)

    def __setattr__(self, name, value):
        # reader settings the writers set (roi, fix_phase, ...) belong to the source
        if name.startswith("_") or isinstance(
            getattr(type(self), name, None), property
        ):
            object.__setattr__(self, name, value)
            return
        setattr(self._source, name, value)

    def _imwrite(self, outpath, **kwargs):
        from mbo_utilities.arrays._base import _imwrite_base

        return _imwrite_base(self, outpath, **kwargs)

    def __repr__(self) -> str:
        return (
            f"InvertedDeflectionView(shape={self.shape}, "
            f"source={type(self._source).__name__})"
        )


def invert_deflection(source, enabled: bool = True):
    """``source`` flipped about its temporal mean, or unflipped when not ``enabled``.

    Never stacks: an existing view is re-based on its source.
    """
    if isinstance(source, InvertedDeflectionView):
        source = source.source
    return InvertedDeflectionView(source) if enabled else source
