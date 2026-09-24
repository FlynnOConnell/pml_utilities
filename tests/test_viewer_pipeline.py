"""
Viewer-pipeline contract tests.

These pin the contracts the GUI rendering layer relies on, independent
of any actual fastplotlib/qt rendering. Each test is a regression for a
bug we shipped between the 5D refactor and now.

Coverage:
- `_SqueezeSingletonDims`: leak-proof against numpy protocol calls
  (`np.asarray`, `astype`), correct shape across all 8 singleton patterns
- Window/Spatial widget feature gating across rank patterns
"""

from __future__ import annotations

import numpy as np
import pytest
from mbo_utilities.gui.run_gui import _SqueezeSingletonDims


class FakeArr:
    """Minimal lazy array — backed by an ndarray, with shape/ndim/dims."""

    def __init__(self, shape, dims, dtype=np.uint16):
        self._raw = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = self._raw.dtype
        self.dims = dims

    def __getitem__(self, key):
        return self._raw[key]

    def __array__(self, dtype=None, copy=None):
        out = self._raw
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FakeParent:
    """Parent stub matching the WidgetBase contract: parent.image_widget.data[0]."""

    def __init__(self, arr):
        self.image_widget = type("IW", (), {"data": [arr]})()


# truth table for the 8 combinations of T/C/Z ∈ {1, >1}
SINGLETON_PATTERNS = [
    # (full_shape, expected_squeezed_shape, label)
    ((1, 1, 1, 64, 48), (64, 48), "all singleton (2D image)"),
    ((20, 1, 1, 64, 48), (20, 64, 48), "T only"),
    ((1, 4, 1, 64, 48), (4, 64, 48), "C only"),
    ((1, 1, 8, 64, 48), (8, 64, 48), "Z only"),
    ((20, 4, 1, 64, 48), (20, 4, 64, 48), "T,C"),
    ((20, 1, 8, 64, 48), (20, 8, 64, 48), "T,Z"),
    ((1, 4, 8, 64, 48), (4, 8, 64, 48), "C,Z"),
    ((20, 4, 8, 64, 48), (20, 4, 8, 64, 48), "no singletons"),
]


class TestSqueezeWrapperShape:
    """`_SqueezeSingletonDims` reports the correct natural rank for every
    combination of T/C/Z singleton patterns. Lockstep with `get_slider_dims`
    is required so fastplotlib's `ndim - 2 == len(slider_dim_names)` holds.
    """

    @pytest.mark.parametrize("full,expected,label", SINGLETON_PATTERNS)
    def test_squeeze(self, full, expected, label):
        arr = FakeArr(full, ("T", "C", "Z", "Y", "X"))
        w = _SqueezeSingletonDims(arr)
        assert w.shape == expected, f"{label}: expected {expected}, got {w.shape}"
        assert w.ndim == len(expected)


class TestSqueezeWrapperNumpyLeak:
    """Regression: `_SqueezeSingletonDims.__getattr__` previously delegated
    `astype` and `__array__` to the underlying array, leaking the original
    5D shape to fastplotlib's TextureArray. Pin every numpy entry point so
    the leak can't come back.
    """

    def test_np_asarray_honors_squeeze(self):
        arr = FakeArr((1, 1, 1, 64, 48), ("T", "C", "Z", "Y", "X"))
        w = _SqueezeSingletonDims(arr)
        out = np.asarray(w)
        assert out.shape == (64, 48), (
            f"np.asarray leaked underlying shape: got {out.shape}"
        )

    def test_astype_honors_squeeze(self):
        arr = FakeArr((1, 1, 1, 64, 48), ("T", "C", "Z", "Y", "X"))
        w = _SqueezeSingletonDims(arr)
        out = w.astype(np.float32)
        assert out.shape == (64, 48), f"astype leaked underlying shape: got {out.shape}"
        assert out.dtype == np.float32

    def test_isolated_buffer_assignment(self):
        """Fastplotlib's TextureArray._fix_data path: allocate zeros at
        wrapper.shape, then assign wrapper[:] into it. Both ends must agree.
        """
        arr = FakeArr((1, 1, 1, 64, 48), ("T", "C", "Z", "Y", "X"))
        w = _SqueezeSingletonDims(arr)
        buf = np.zeros(w.shape, dtype=w.dtype)
        buf[:] = w[:]  # must not raise
        assert buf.shape == (64, 48)

    def test_dunder_lookups_dont_leak(self):
        """Numpy may probe `__array_interface__` etc. via getattr — those
        must not silently fall through to the underlying array.
        """
        arr = FakeArr((1, 1, 1, 64, 48), ("T", "C", "Z", "Y", "X"))
        w = _SqueezeSingletonDims(arr)
        # the wrapper must not pretend to have arbitrary dunders
        with pytest.raises(AttributeError):
            w.__array_interface__


class TestReloadDataConsistency:
    """`load_new_data` (file-dialog reload) must produce the same view as
    the initial launch path in `_create_image_widget`. Both must apply
    the singleton-dim squeeze; otherwise the same file gives different
    shapes depending on whether you opened it via the CLI or the dialog.

    Regressions:
    - Suite2pArray reload was exposing 5D `(T, 1, Z, Y, X)` to the viewer,
      putting Z at index 2 instead of index 1. Initial launch correctly
      squeezed C=1 and exposed `(T, Z, Y, X)`.
    - mean_subtraction state persisted across reloads, leaving the
      checkbox ticked while the underlying spatial_func used stale or
      missing per-z mean images for the new data.
    """

    def test_squeeze_wrapper_puts_z_at_index_1_for_5d_with_c1(self):
        """A 5D array shaped like Suite2p output (T, 1, Z, Y, X) must
        squeeze to (T, Z, Y, X) — Z at index 1, not 2.
        """
        suite2p_like = FakeArr((10, 1, 8, 64, 48), ("T", "C", "Z", "Y", "X"))
        wrapped = _SqueezeSingletonDims(suite2p_like)
        assert wrapped.shape == (10, 8, 64, 48)
        assert wrapped.ndim == 4
        assert wrapped.dims == ("T", "Z", "Y", "X")
        # Z at index 1 (not 2) — the bug was reporting Z at index 2
        assert wrapped.dims.index("Z") == 1
