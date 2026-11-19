"""PlaneMovie, the (T, Y, X) view over one plane and channel, and the mean traces read through it."""

from __future__ import annotations


import numpy as np
import pytest
from mbo_utilities.annotation.measure import _factorized, feather_mask, masks_label_image, measure_mean, plane_masks
from mbo_utilities.arrays.plane import PlaneMovie, _shift, as_movie, movie_dims
from mbo_utilities.annotation.store import RoiLabelStore


def _importable(name: str) -> bool:
    # importorskip only catches ImportError; a half-merged editable install
    # raises SyntaxError and would fail the test instead of skipping it
    try:
        __import__(name)
    except Exception:
        return False
    return True


T, LY, LX = 60, 48, 40


def _disc(cy, cx, r, shape=(LY, LX)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


@pytest.fixture
def store():
    s = RoiLabelStore(nz=2, ny=LY, nx=LX, label_names=("soma", "dendrite"))
    s.add_roi(0, _disc(10, 10, 4))  # 0: z0 soma
    s.add_roi(0, _disc(30, 25, 5))  # 1: z0 dendrite
    s.add_roi(1, _disc(20, 20, 4))  # 2: z1 soma
    s.add_roi(1, _disc(40, 30, 3))  # 3: z1 unlabeled
    s.set_class(0, 0)
    s.set_class(1, 1)
    s.set_class(2, 0)
    return s


def test_plane_masks_keeps_the_order_given(store):
    masks, kept = plane_masks(store, 1, [3, 2, 0])
    assert kept == [3, 2]
    assert [int(m.sum()) for m in masks] == [store.rois[3].area, store.rois[2].area]
    img = masks_label_image(masks)
    assert set(np.unique(img)) == {0, 1, 2}
    assert (img == 1).sum() == store.rois[3].area
    masks0, kept0 = plane_masks(store, 0, [2, 3])
    assert kept0 == [] and masks0 == []


class _SpatialOnly:
    """Minimal 5-D array that honours y/x keys - the whole contract."""

    def __init__(self, data):
        self.data = data  # (T, C, Z, Y, X)
        self.shape = data.shape
        self.ndim = 5
        self.dtype = data.dtype
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.data[key]


def test_plane_movie_forwards_spatial_keys_and_normalizes_shape():
    data = np.arange(3 * 2 * 2 * 5 * 6, dtype=np.int16).reshape(3, 2, 2, 5, 6)
    arr = _SpatialOnly(data)
    m = as_movie(arr, z=1, c=1)
    assert m.shape == (3, 5, 6) and m.ndim == 3 and len(m) == 3
    np.testing.assert_array_equal(m[1:3, 2:4, 1:5], data[1:3, 1, 1, 2:4, 1:5])
    assert arr.reads[-1] == (slice(1, 3), 1, 1, slice(2, 4), slice(1, 5))
    assert m[2].shape == (5, 6)
    assert m[:, 3, 4].shape == (3,)
    assert m[[0, 2], 1, 1:3].shape == (2, 2)
    assert m.frames(0, 2).shape == (2, 5, 6)
    with pytest.raises(IndexError):
        as_movie(arr, z=5)


def test_plane_movie_crop_shifts_keys():
    data = np.arange(4 * 1 * 1 * 10 * 12, dtype=np.int16).reshape(4, 1, 1, 10, 12)
    arr = _SpatialOnly(data)
    m = as_movie(arr)
    assert m.box is None
    c = m.crop(2, 8, 3, 11)
    assert c.shape == (4, 6, 8) and c.box == (2, 8, 3, 11)
    assert c[0, 0, 0] == data[0, 0, 0, 2, 3]
    assert c[0, -1, -1] == data[0, 0, 0, 7, 10]
    np.testing.assert_array_equal(c[1], data[1, 0, 0, 2:8, 3:11])
    np.testing.assert_array_equal(c[:, 1:3, 2:5], data[:, 0, 0, 3:5, 5:8])
    assert arr.reads[-1][3] == slice(3, 5, 1) and arr.reads[-1][4] == slice(5, 8, 1)
    np.testing.assert_array_equal(c[0, [0, 2], 1], data[0, 0, 0, [2, 4], 4])
    # crops of crops compose
    cc = c.crop(1, 4, 2, 6)
    assert cc.box == (3, 6, 5, 9)
    np.testing.assert_array_equal(cc[2], data[2, 0, 0, 3:6, 5:9])
    assert cc.frames(0, 2).shape == (2, 3, 4)
    # a crop covering the whole frame is not a crop
    assert m.crop(0, 10, 0, 12).box is None
    with pytest.raises(IndexError):
        c.crop(0, 7, 0, 1)
    with pytest.raises(IndexError):
        c[0, 6, 0]


def test_feather_mask():
    mask = np.zeros((9, 9), bool)
    mask[2:7, 2:7] = True
    w = feather_mask(mask, edge_width=2)
    assert w[4, 4] == 1.0 and 0 < w[2, 2] < 1.0
    assert w[~mask].max() == 0.0


def test_shift_matches_numpy_on_a_crop():
    # incl. empty negative-step slices, whose indices() clamp start to -1;
    # the shifted slice must stay empty, not become a full reverse read
    src = np.arange(10)
    keys = (
        slice(2, None, -1),
        slice(None, None, -1),
        slice(-10, -8, -1),
        slice(4, 1, -1),
        slice(1, 4),
        slice(0, 0),
        slice(3, 3, -1),
    )
    for off in (0, 3):
        crop = src[off : off + 5]
        for key in keys:
            np.testing.assert_array_equal(
                src[_shift(key, off, 5)], crop[key], err_msg=f"{off=} {key=}"
            )


def test_plane_movie_lower_ranks():
    d4 = np.zeros((4, 2, 3, 3), np.float32)
    assert as_movie(d4, z=1).shape == (4, 3, 3)
    d3 = np.zeros((4, 3, 3), np.float32)
    assert as_movie(d3).shape == (4, 3, 3)
    d2 = np.ones((3, 3), np.float32)
    m = as_movie(d2)
    assert m.shape == (1, 3, 3) and m[0, 1, 1] == 1.0
    assert as_movie(m) is m


def test_measure_mean_reads_only_the_bbox():
    T = 40
    data = np.zeros((T, 1, 1, 20, 30), np.float32)
    data[:, 0, 0, 5:8, 10:14] = np.arange(T)[:, None, None]
    data[:, 0, 0, 6, 11] += 100
    arr = _SpatialOnly(data)
    mask = np.zeros((20, 30), bool)
    mask[5:8, 10:14] = True
    F, Fneu = measure_mean(as_movie(arr), [mask], neuropil=False, batch=7)
    assert Fneu is None and F.shape == (1, T)
    np.testing.assert_allclose(F[0], data[:, 0, 0][:, mask].mean(axis=1), rtol=1e-5)
    assert all(k[3] == slice(5, 8) and k[4] == slice(10, 14) for k in arr.reads)
    # the ring reads around the mask and never inside another roi
    other = np.zeros((20, 30), bool)
    other[8:10, 10:14] = True
    data[:, 0, 0, 8:10, 10:14] = 1000
    F, Fneu = measure_mean(
        as_movie(_SpatialOnly(data)),
        [mask],
        mask | other,
        inner=1,
        outer=3,
        min_npix=4,
    )
    assert Fneu.shape == (1, T) and Fneu.max() < 1000
    np.testing.assert_allclose(F[0], data[:, 0, 0][:, mask].mean(axis=1), rtol=1e-5)


# cpu, and no denoiser: its patch sampler finds nothing on a 48x40 field
def _toy_pmd(t=24, h=14, w=12, rank=5, seed=0):
    import torch

    from masknmf import CompressionArray

    rng = np.random.default_rng(seed)
    u = rng.random((h * w, rank)).astype(np.float32)
    u[rng.random((h * w, rank)) > 0.4] = 0.0
    return CompressionArray.from_tensors(
        (t, h, w),
        torch.from_numpy(u).to_sparse_coo(),
        torch.from_numpy(rng.standard_normal((rank, t)).astype(np.float32)),
        torch.from_numpy(rng.random((h, w)).astype(np.float32)),
        torch.from_numpy((rng.random((h, w)) + 0.5).astype(np.float32)),
    )


def test_measure_mean_factorized_single_read():
    if not (_importable("masknmf") and _importable("torch")):
        pytest.skip("masknmf/torch not importable")
    pmd = _toy_pmd()
    assert _factorized(pmd) and not _factorized(pmd[:])
    mask = _disc(6, 5, 3, shape=pmd.shape[1:])
    np.testing.assert_allclose(
        measure_mean(as_movie(pmd), [mask], neuropil=False)[0],
        measure_mean(as_movie(pmd[:]), [mask], neuropil=False)[0],
        rtol=1e-5,
        atol=1e-5,
    )


class _NamedArray:
    """A numpy array with ``dims``, the way a labelled lazy array presents itself."""

    def __init__(self, data, dims):
        self.data, self.dims, self.shape, self.ndim = data, dims, data.shape, data.ndim

    def __getitem__(self, key):
        return self.data[key]


def test_plane_movie_resolves_axes_by_name():
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(3, 2, 4, 5)
    arr = _NamedArray(data, ("Z", "T", "Y", "X"))
    assert movie_dims(arr) == ("Z", "T", "Y", "X")
    movie = PlaneMovie(arr, z=2)
    assert movie.shape == (2, 4, 5) and movie.nz == 3
    np.testing.assert_array_equal(movie[1], data[2, 1])
    np.testing.assert_array_equal(movie[:, 1:3, 0], data[2, :, 1:3, 0])
    assert movie_dims(data) == ("T", "Z", "Y", "X")


def test_plane_movie_window_shifts_t_keys():
    data = np.arange(10 * 4 * 5, dtype=np.float32).reshape(10, 4, 5)
    movie = PlaneMovie(data)
    assert movie.t_range is None
    win = movie.window(3, 7)
    assert win.shape == (4, 4, 5) and win.t_range == (3, 7)
    np.testing.assert_array_equal(win[:], data[3:7])
    np.testing.assert_array_equal(win[1], data[4])
    np.testing.assert_array_equal(win[-1, 1:3, 2], data[6, 1:3, 2])
    # crops keep the window and windows of windows compose
    crop = win.crop(1, 3, 0, 2)
    np.testing.assert_array_equal(crop[:], data[3:7, 1:3, 0:2])
    again = win.window(1, 3)
    assert again.t_range == (4, 6)
    np.testing.assert_array_equal(again.frames(0, 2), data[4:6])
    with pytest.raises(IndexError):
        movie.window(8, 12)


def test_plane_movie_select_reads_any_frame_selection():
    from mbo_utilities.arrays.features._slicing import (
        index_window,
        parse_timepoint_selection,
    )

    data = np.arange(10 * 4 * 5, dtype=np.float32).reshape(10, 4, 5)
    movie = PlaneMovie(data)
    assert movie.t_indices is None and movie.select(None) is movie
    # the selection string every Save As / pipeline row takes: "1:9:2,5:6"
    sel = parse_timepoint_selection("1:9:2,5:6", 10)
    picked = movie.select(sel.final_indices)
    assert picked.t_indices == [0, 2, 6, 8] and picked.shape == (4, 4, 5)
    np.testing.assert_array_equal(picked[:], data[[0, 2, 6, 8]])
    np.testing.assert_array_equal(picked[1:3], data[[2, 6]])
    np.testing.assert_array_equal(picked[[0, 3]], data[[0, 8]])
    assert picked.t_range is None and index_window(picked.t_indices) is None
    # a stride is a window with a step; a contiguous one has a range too
    strided = movie.select(range(0, 10, 2))
    assert index_window(strided.t_indices) == (0, 9, 2) and strided.t_range is None
    assert movie.window(3, 7).t_range == (3, 7)
    # selections of selections compose, and crops keep them
    inner = strided.select([1, 2])
    assert inner.t_indices == [2, 4]
    np.testing.assert_array_equal(inner.crop(1, 3, 0, 2)[:], data[[2, 4], 1:3, 0:2])
    with pytest.raises(IndexError):
        movie.select([0, 10])


