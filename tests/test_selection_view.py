"""
Tests for `arr.isel(...)` / `imread(channel=)`: a lazy selection that keeps
the array 5D.
"""

import mbo_utilities
import numpy as np
import pytest
import tifffile
from mbo_utilities.arrays import NumpyArray
from mbo_utilities.arrays._selection_view import SelectionView
from mbo_utilities.lazy_array import base_array


@pytest.fixture
def tczyx():
    return np.arange(6 * 3 * 4 * 8 * 5, dtype="uint16").reshape(6, 3, 4, 8, 5)


def test_isel_keeps_every_axis(tczyx):
    a = NumpyArray(tczyx)
    v = a.isel(C=1)
    assert isinstance(v, SelectionView)
    assert v.shape == (6, 1, 4, 8, 5)
    assert v.ndim == 5
    assert np.array_equal(v[:], tczyx[:, 1:2])


def test_isel_several_axes(tczyx):
    v = NumpyArray(tczyx).isel(T=slice(0, 6, 2), C=[0, 2], Z=[3, 1])
    assert v.shape == (3, 2, 2, 8, 5)
    assert np.array_equal(v[:], tczyx[::2][:, [0, 2]][:, :, [3, 1]])


def test_isel_indexing_follows_numpy(tczyx):
    v = NumpyArray(tczyx).isel(Z=[1, 3])
    expected = tczyx[:, :, [1, 3]]
    assert np.array_equal(v[2, 1, 0], expected[2, 1, 0])
    assert np.array_equal(v[-1], expected[-1])
    assert np.array_equal(v[1:4, :, 1, 2:6], expected[1:4, :, 1, 2:6])
    assert np.array_equal(v[..., 0], expected[..., 0])


def test_isel_rejects_y_and_x(tczyx):
    with pytest.raises(ValueError, match="isel selects"):
        NumpyArray(tczyx).isel(Y=0)


def test_isel_metadata_follows_the_selection(tczyx):
    a = NumpyArray(tczyx, metadata={"fs": 10.0, "dz": 5.0})
    v = a.isel(T=slice(0, 6, 2), C=0, Z=[0, 2])
    md = v.metadata
    assert md["num_color_channels"] == 1
    assert md["num_zplanes"] == 2
    assert md["num_timepoints"] == 3
    assert md["fs"] == pytest.approx(5.0)
    assert md["dz"] == pytest.approx(10.0)


def test_isel_reader_settings_reach_the_source(tczyx):
    a = NumpyArray(tczyx)
    v = a.isel(C=0)
    v.some_setting = 3
    assert a.some_setting == 3
    assert base_array(v) is a


def test_imread_channel_is_a_selection(tmp_path):
    p = tmp_path / "tc.tif"
    x = np.arange(10 * 16 * 12, dtype="uint16").reshape(10, 16, 12)
    tifffile.imwrite(p, np.stack([x, x + 1], 1), imagej=True, metadata={"axes": "TCYX"})

    full = mbo_utilities.imread(p)
    assert full.shape == (10, 2, 1, 16, 12)
    one = mbo_utilities.imread(p, channel=1)
    assert one.shape == (10, 1, 1, 16, 12)
    assert np.array_equal(one[2, 0, 0], x[2] + 1)
    assert one.reader_kwargs["channel"] == 1
    again = mbo_utilities.imread(p, **one.reader_kwargs)
    assert np.array_equal(again[:], one[:])


def test_imread_squeeze_is_gone(tmp_path):
    p = tmp_path / "ts.tif"
    tifffile.imwrite(p, np.zeros((10, 16, 12), dtype="uint16"))
    with pytest.raises(TypeError, match="always 5D"):
        mbo_utilities.imread(p, squeeze=True)


def test_selection_writes(tmp_path, tczyx):
    v = NumpyArray(tczyx).isel(C=2)
    out = tmp_path / "out"
    mbo_utilities.imwrite(v, out, ext=".zarr")
    back = mbo_utilities.imread(next(out.rglob("*.zarr")))
    assert back.shape == (6, 1, 4, 8, 5)
    assert np.array_equal(back[:], tczyx[:, 2:3])
