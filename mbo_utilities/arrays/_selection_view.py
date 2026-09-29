"""Read-time selection of T, C and Z indices over a 5D lazy array.

``arr.isel(C=[1])`` or ``imread(path, channel=1)`` is a ``SelectionView``: the
source with only the chosen timepoints, channels or z-planes, still 5D (a
single channel is ``C == 1``, never dropped). Reads go to the source; nothing
is copied. Like ``FrameAveragedView`` it sits under the writers, so reader
settings the writers change (``roi``, ``fix_phase``) forward to the source,
and ``reader_kwargs`` carries ``channel`` so a worker that re-opens the path
gets the same selection.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from mbo_utilities.arrays._base import _normalize_key
from mbo_utilities.lazy_array import DIMS, LazyArray

__all__ = ["SelectionView"]

_SELECTABLE = ("T", "C", "Z")


def _as_key(indices: list[int]):
    """A list of source indices as the cheapest key that reads them: a slice
    when they are evenly spaced and increasing, else the list.
    """
    if not indices:
        return slice(0, 0)
    if len(indices) == 1:
        return slice(indices[0], indices[0] + 1)
    step = indices[1] - indices[0]
    if step > 0 and all(b - a == step for a, b in zip(indices, indices[1:])):
        return slice(indices[0], indices[-1] + 1, step)
    return list(indices)


def _offset_through(source_fn, t_idx, c_idx, z_idx, t, c, z):
    return source_fn(t_idx[int(t)], c_idx[int(c)], z_idx[int(z)])


class SelectionView(LazyArray):
    """``source`` restricted to chosen T, C and Z indices, as
    ``(len(T), len(C), len(Z), Y, X)``.

    Parameters
    ----------
    source : LazyArray
        The 5D array to select from; never modified.
    indexers : dict
        Axis letter (``"T"``, ``"C"``, ``"Z"``) to 0-based indices: an int, a
        slice or a list. An axis left out keeps all of its indices.
    """

    def __init__(self, source, indexers: dict):
        unknown = set(indexers) - set(_SELECTABLE)
        if unknown:
            raise ValueError(f"isel selects {_SELECTABLE}, got {sorted(unknown)}")
        self._source = source
        self._metadata = None
        self._indexers = {}
        for axis, index in indexers.items():
            n = source.shape[DIMS.index(axis)]
            indices = np.arange(n)[index]
            indices = [int(i) for i in np.atleast_1d(indices)]
            if not indices:
                raise IndexError(f"isel({axis}={index!r}) selects nothing of {n}")
            self._indexers[axis] = indices

    @property
    def source(self):
        """The wrapped source array (never modified)."""
        return self._source

    @property
    def _arr(self):
        """The wrapped source, for one-level `_arr` unwrapping by callers."""
        return self._source

    @property
    def indexers(self) -> dict[str, list[int]]:
        """0-based source indices per selected axis."""
        return {axis: list(indices) for axis, indices in self._indexers.items()}

    def _indices(self, axis: str) -> list[int]:
        own = self._indexers.get(axis)
        if own is not None:
            return own
        return list(range(self._source.shape[DIMS.index(axis)]))

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        # read through every call: the writers change the source's ROI
        # selection (and so its Y/X) while holding this view
        t, c, z, y, x = self._source.shape
        sizes = {"T": t, "C": c, "Z": z}
        for axis, indices in self._indexers.items():
            sizes[axis] = len(indices)
        return (sizes["T"], sizes["C"], sizes["Z"], y, x)

    @property
    def dtype(self):
        return self._source.dtype

    @property
    def dims(self) -> tuple[str, ...]:
        return DIMS

    @property
    def reader_kwargs(self) -> dict:
        """``imread`` kwargs that re-create this view: the source's own plus
        ``channel`` for a single selected channel.
        """
        kwargs = dict(getattr(self._source, "reader_kwargs", None) or {})
        channels = self._indexers.get("C")
        if channels is not None and len(channels) == 1:
            kwargs["channel"] = channels[0]
        return kwargs

    @property
    def metadata(self) -> dict:
        """The source's metadata with counts, ``fs`` and ``dz`` following the
        selection; what the writers set back is kept on the view.
        """
        if self._metadata is not None:
            return self._metadata
        from mbo_utilities.metadata import OutputMetadata

        return OutputMetadata(
            source=dict(getattr(self._source, "metadata", None) or {}),
            source_shape=tuple(self._source.shape),
            source_dims=DIMS,
            selections=self.indexers,
        ).to_dict()

    @metadata.setter
    def metadata(self, value):
        self._metadata = dict(value)

    def __len__(self) -> int:
        return self.nt

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        if len(key) > 5:
            raise IndexError(f"too many indices for 5D array: {len(key)}")
        key = list(key) + [slice(None)] * (5 - len(key))
        for axis, indices in self._indexers.items():
            i = DIMS.index(axis)
            selected = np.asarray(indices)[key[i]]
            key[i] = int(selected) if selected.ndim == 0 else _as_key(selected.tolist())
        return self._source[tuple(key)]

    def __array__(self, dtype=None, copy=None):
        # one (Y, X) frame, never the whole array
        data = np.asarray(self[0, 0, 0])
        return data.astype(dtype) if dtype is not None else data

    def __getattr__(self, name):
        # forward reader attributes (filenames, source_path, roi, ...) to the
        # source; underscore names are not forwarded so __init__ stays
        # recursion-safe
        if name.startswith("_"):
            raise AttributeError(name)
        source = object.__getattribute__(self, "_source")
        if name == "get_offset_at":
            # the writers ask with this view's indices
            return partial(
                _offset_through,
                getattr(source, name),
                self._indices("T"),
                self._indices("C"),
                self._indices("Z"),
            )
        return getattr(source, name)

    def __setattr__(self, name, value):
        # own state and own properties stay here; anything else is a reader
        # setting (roi, fix_phase, use_fft, ...) and belongs to the source
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
        selected = ", ".join(f"{a}={v}" for a, v in self._indexers.items())
        return (
            f"SelectionView(shape={self.shape}, dtype={self.dtype}, "
            f"{selected}, source={type(self._source).__name__})"
        )
