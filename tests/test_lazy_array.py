"""LazyArray base: a subclass implements only `shape`, `__getitem__`,
`dtype` and `can_open` and sets `self._metadata`; dims, metadata and
dimension_specs come from the base.
"""

from __future__ import annotations

import numpy as np
from mbo_utilities.lazy_array import LazyArray


class _Minimal(LazyArray):
    """Smallest possible 5D reader — everything else is inherited."""

    def __init__(self, shape, metadata=None):
        self._s = shape
        self._metadata = metadata or {}

    @property
    def shape(self):
        return self._s

    @property
    def dtype(self):
        return np.dtype("uint16")

    def __getitem__(self, key):
        return np.zeros(self._s, dtype="uint16")[key]


class TestLazyArrayBase:
    def test_shape_ndim_from_base(self):
        arr = _Minimal((4, 2, 3, 8, 8))
        assert arr.shape == (4, 2, 3, 8, 8)
        assert arr.ndim == 5
        assert (arr.nt, arr.nc, arr.nz, arr.ny, arr.nx) == arr.shape

    def test_dims_default_tczyx(self):
        assert _Minimal((4, 2, 3, 8, 8)).dims == ("T", "C", "Z", "Y", "X")

    def test_metadata_default_dict(self):
        arr = _Minimal((1, 1, 1, 8, 8))
        assert arr.metadata == {}
        arr.metadata = {"foo": 1}
        assert arr.metadata["foo"] == 1

    def test_metadata_stores_dims_plainly(self):
        # base keeps `dims` as plain metadata; reported dims are always TCZYX
        arr = _Minimal((4, 2, 3, 8, 8))
        arr.metadata = {"dims": ("T", "C", "Z", "Y", "X"), "foo": 1}
        assert arr.metadata["foo"] == 1
        assert arr.metadata["dims"] == ("T", "C", "Z", "Y", "X")
        assert arr.dims == ("T", "C", "Z", "Y", "X")

    def test_metadata_rank_mismatched_dims_does_not_raise(self):
        # a 4-element dims stored on a 5D array (e.g. round-tripped from a
        # single-channel write) must store cleanly, not raise.
        arr = _Minimal((4, 1, 3, 8, 8))
        arr.metadata = {"dims": ("T", "Z", "Y", "X")}
        assert arr.metadata["dims"] == ("T", "Z", "Y", "X")
        assert arr.dims == ("T", "C", "Z", "Y", "X")

    def test_explicit_dims_setter(self):
        arr = _Minimal((4, 2, 3, 8, 8))
        arr.dims = ("T", "C", "Z", "Y", "X")
        assert arr.dims == ("T", "C", "Z", "Y", "X")

    def test_dimension_specs_reactive(self):
        arr = _Minimal((4, 2, 3, 8, 8))
        assert arr.dimension_specs.num_channels == 2
        assert arr.dimension_specs.num_zplanes == 3

    def test_slider_dims_drop_singletons(self):
        # T=4 has a slider; C=1/Z=1 singletons are dropped; Y/X never sliders
        arr = _Minimal((4, 1, 1, 8, 8))
        assert arr.slider_dims == ("t",)

    def test_dim_index_and_has_dim(self):
        arr = _Minimal((4, 2, 3, 8, 8))
        assert arr.dim_index("Z") == 2
        assert arr.has_dim("C")
        assert not arr.has_dim("R")
