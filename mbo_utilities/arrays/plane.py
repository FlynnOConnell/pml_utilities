"""``(T, Y, X)`` views over one channel and z-plane of any indexable array.

The one thing everything here needs from a movie is

    arr[t, c, z, y, x]  ->  numpy

for integer / slice / list keys on every axis (rank 5, 4 ``TZYX``, 3 ``TYX``
or 2 ``YX`` arrays are all accepted). ``PlaneMovie`` forwards
``movie[t, y0:y1, x0:x1]`` straight to the array, so a spatially lazy array
is read only inside the box asked for. It is the shape masknmf's pipelines,
the mean traces and the viewers' traces read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mbo_utilities.arrays.features._slicing import index_window

__all__ = ["PlaneMovie", "as_movie", "movie_dims"]


def _index_len(key, n: int) -> int | None:
    """Length of axis ``n`` after indexing with ``key``; None when dropped."""
    if isinstance(key, (int, np.integer)):
        return None
    if isinstance(key, slice):
        return len(range(*key.indices(n)))
    if isinstance(key, range):
        return len(key)
    return len(np.asarray(key).reshape(-1))


def _shift(key, off: int, n: int):
    """``key`` on a length-``n`` cropped axis, moved by ``off`` into the source."""
    if isinstance(key, (int, np.integer)):
        k = int(key) + (n if key < 0 else 0)
        if not 0 <= k < n:
            raise IndexError(f"index {key} out of range for a length-{n} crop")
        return k + off
    if isinstance(key, slice):
        lo, hi, step = key.indices(n)
        if len(range(lo, hi, step)) == 0:
            return slice(0, 0)
        lo, hi = lo + off, hi + off
        return slice(lo, hi if hi >= 0 else None, step)
    a = np.asarray(key).reshape(-1)
    return np.where(a < 0, a + n, a) + off


_RANK_DIMS = {2: "YX", 3: "TYX", 4: "TZYX", 5: "TCZYX"}


def movie_dims(arr) -> tuple[str, ...]:
    """Axis names of ``arr``: its own ``dims`` when they match its rank and
    name Y and X, else the rank's default letters.
    """
    nd = getattr(arr, "ndim", None)
    nd = int(np.ndim(arr) if nd is None else nd)
    dims = tuple(str(d).upper() for d in (getattr(arr, "dims", None) or ()))
    if len(dims) == nd and {"Y", "X"} <= set(dims):
        return dims
    if nd not in _RANK_DIMS:
        raise ValueError(f"movie source must be 2-5D, got ndim={nd}")
    return tuple(_RANK_DIMS[nd])


class PlaneMovie:
    """``(T, Y, X)`` view over one channel / z-plane of any indexable array.

    ``movie[t, y, x]`` is forwarded to the array by axis name (``arr.dims``,
    or T/C/Z/Y/X by rank), other axes pinned at ``z`` / ``c`` / 0, and the
    result is coerced to numpy and reshaped to what numpy indexing would
    give, so consumers never see a backend's squeeze quirks. This is all
    the ROI pipeline, masknmf compression and the viewer's traces rely on:
    if the array slices in y and x, everything here works. ``crop`` returns
    a rectangular view whose keys shift into the source frame on the way
    through; ``box`` records where that view sits. ``select`` does the same
    along T for any frame selection (the ``tp_indices`` every worker task
    takes); ``t_indices`` records it and ``t_range`` the contiguous case.
    """

    y0 = 0
    x0 = 0
    t_map: np.ndarray | None = (
        None  # the source frames this view covers, in order; None is every frame
    )

    def __init__(self, arr, z: int = 0, c: int = 0):
        self.arr = arr
        self.dims = movie_dims(arr)
        self.z = int(z)
        self.c = int(c)
        size = dict(zip(self.dims, (int(s) for s in arr.shape)))
        nt, nc, nz = size.get("T", 1), size.get("C", 1), size.get("Z", 1)
        if not 0 <= self.z < nz:
            raise IndexError(f"z={z} out of range for {nz} plane(s)")
        if not 0 <= self.c < nc:
            raise IndexError(f"c={c} out of range for {nc} channel(s)")
        self.shape = (nt, size["Y"], size["X"])
        self.nz, self.nc = nz, nc
        self.ndim = 3

    @property
    def dtype(self):
        return np.dtype(getattr(self.arr, "dtype", np.float32))

    def __len__(self):
        return self.shape[0]

    @property
    def box(self) -> tuple[int, int, int, int] | None:
        """``(y0, y1, x0, x1)`` of this view in the source frame; None when
        the view covers the whole frame.
        """
        _, ny, nx = self.shape
        size = dict(zip(self.dims, (int(s) for s in self.arr.shape)))
        if (self.y0, self.x0) == (0, 0) and (ny, nx) == (size["Y"], size["X"]):
            return None
        return (self.y0, self.y0 + ny, self.x0, self.x0 + nx)

    def _view(self) -> PlaneMovie:
        out = PlaneMovie(self.arr, z=self.z, c=self.c)
        out.y0, out.x0, out.t_map = self.y0, self.x0, self.t_map
        out.shape = self.shape
        return out

    def crop(self, y0: int, y1: int, x0: int, x1: int) -> PlaneMovie:
        """View of rows ``y0:y1`` and columns ``x0:x1`` sharing this movie's
        array; crops of crops compose.
        """
        _, ny, nx = self.shape
        if not (0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx):
            raise IndexError(f"crop ({y0}:{y1}, {x0}:{x1}) outside {ny}x{nx}")
        out = self._view()
        out.y0, out.x0 = self.y0 + int(y0), self.x0 + int(x0)
        out.shape = (self.shape[0], int(y1 - y0), int(x1 - x0))
        return out

    @property
    def t_indices(self) -> list[int] | None:
        """The 0-based source frames this view reads, in order; None when
        it reads every frame. The same list a worker task's ``tp_indices``
        carries.
        """
        return None if self.t_map is None else [int(t) for t in self.t_map]

    @property
    def t_range(self) -> tuple[int, int] | None:
        """``(t0, t1)`` when the view is one contiguous window of the
        source's T; None for every frame or a strided / gapped selection.
        """
        window = index_window(self.t_indices)
        if window is None or window[2] != 1:
            return None
        return (window[0], window[1])

    def select(self, indices) -> PlaneMovie:
        """View over the source frames ``indices`` (0-based, in the order
        given: a ``parse_timepoint_selection`` result, a range, a list),
        sharing this movie's array and crop; selections of selections
        compose. Every frame (None) is this movie.
        """
        if indices is None:
            return self
        idx = np.asarray(list(indices), dtype=np.int64).reshape(-1)
        nt = self.shape[0]
        if idx.size == 0 or idx.min() < 0 or idx.max() >= nt:
            raise IndexError(f"frame selection outside {nt} frames")
        out = self._view()
        out.t_map = idx if self.t_map is None else self.t_map[idx]
        out.shape = (int(idx.size), self.shape[1], self.shape[2])
        return out

    def window(self, t0: int, t1: int) -> PlaneMovie:
        """View of frames ``t0:t1`` sharing this movie's array and crop."""
        nt = self.shape[0]
        if not 0 <= t0 < t1 <= nt:
            raise IndexError(f"window ({t0}:{t1}) outside {nt} frames")
        return self.select(range(int(t0), int(t1)))

    def _full_key(self, t, y, x):
        nt, ny, nx = self.shape
        if self.box is not None:
            y = _shift(y, self.y0, ny)
            x = _shift(x, self.x0, nx)
        if self.t_map is not None:
            if isinstance(t, (int, np.integer)):
                t = int(self.t_map[int(t)])
            elif isinstance(t, slice):
                t = self.t_map[t]
            else:
                t = self.t_map[np.asarray(t).reshape(-1)]
        axes = {"T": t, "C": self.c, "Z": self.z, "Y": y, "X": x}
        return tuple(axes.get(d, 0) for d in self.dims)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > 3:
            raise IndexError(f"PlaneMovie takes at most 3 keys, got {len(key)}")
        t, y, x = (key + (slice(None),) * 3)[:3]
        if isinstance(t, range):
            t = slice(t.start, t.stop, t.step)
        if isinstance(t, (list, np.ndarray)):
            t = np.asarray(t).reshape(-1)
        nt, ny, nx = self.shape
        want = tuple(
            n
            for n in (_index_len(t, nt), _index_len(y, ny), _index_len(x, nx))
            if n is not None
        )
        if "T" not in self.dims and not (
            isinstance(t, (int, np.integer)) or _index_len(t, 1) == 1
        ):
            raise IndexError("a source without a time axis has a single frame")
        out = np.asarray(self.arr[self._full_key(t, y, x)])
        return out.reshape(want)

    def frames(self, t0: int, t1: int, y=slice(None), x=slice(None)) -> np.ndarray:
        """``(t1 - t0, ...)`` block, always with a leading time axis."""
        out = self[t0:t1, y, x]
        return (
            out.reshape((-1,) + out.shape[-2:])
            if out.ndim >= 2
            else out.reshape(-1, 1, 1)
        )


def as_movie(source, z: int = 0, c: int = 0) -> PlaneMovie:
    """``PlaneMovie`` for a lazy array, a numpy array or an ``imread``-able path."""
    if isinstance(source, PlaneMovie):
        return source
    if isinstance(source, (str, Path)):
        from mbo_utilities.reader import imread

        source = imread(source)
    return PlaneMovie(source, z=z, c=c)
