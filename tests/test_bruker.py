"""BrukerArray: Bruker HDF5 exports read by their dimension labels."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from mbo_utilities import imread, imwrite
from mbo_utilities.arrays import BrukerArray, H5Array
from mbo_utilities.reader import source_reader_kwargs

# the layout of the files seen so far: (t, z, y, x, c)
SHAPE = (12, 1, 16, 20, 1)


@pytest.fixture
def bruker_h5(tmp_path):
    data = np.arange(np.prod(SHAPE), dtype=np.int16).reshape(SHAPE)
    path = tmp_path / "u005a04_20260915_FamfDay1_FOV1-002.h5"
    with h5py.File(path, "w") as f:
        d = f.create_dataset("imaging", data=data, chunks=(1, 1, 16, 20, 1))
        for i, label in enumerate("tzyxc"):
            d.dims[i].label = label
        d.attrs["channel_names"] = np.array([b"Green"])
        d.attrs["element_size_um"] = np.array([1.0, 1.10195842, 1.10195842])
        d.attrs["frame_period"] = 0.0333093252463988
        d.attrs["imaging_system"] = "bruker"
    return path, data


@pytest.fixture
def bruker_volume_h5(tmp_path):
    data = np.random.default_rng(0).integers(0, 1000, (4, 3, 2, 8, 10), dtype=np.int16)
    path = tmp_path / "volume.h5"
    with h5py.File(path, "w") as f:
        d = f.create_dataset("imaging", data=data)
        for i, label in enumerate("tzcyx"):
            d.dims[i].label = label
        d.attrs["element_size_um"] = np.array([5.0, 0.8, 0.6])
        d.attrs["frame_period"] = 0.1
        d.attrs["imaging_system"] = b"Bruker"
    return path, data


class TestDispatch:
    def test_bruker_file_opens_as_bruker(self, bruker_h5):
        path, _ = bruker_h5
        assert BrukerArray.can_open(path)
        assert isinstance(imread(path), BrukerArray)

    def test_plain_h5_stays_h5(self, tmp_path):
        path = tmp_path / "plain.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("mov", data=np.zeros((3, 8, 8), dtype=np.int16))
        assert not BrukerArray.can_open(path)
        assert isinstance(imread(path), H5Array)

    def test_non_h5_is_declined(self, tmp_path):
        path = tmp_path / "broken.h5"
        path.write_bytes(b"not hdf5")
        assert not BrukerArray.can_open(path)

    def test_a_folder_of_recordings_opens_the_first_as_bruker(self, bruker_h5, caplog):
        """A folder of Bruker files is one recording per file: the first by
        name opens through the Bruker reader, not the plain H5 one that would
        guess the axes of a (t, z, y, x, c) dataset as T, C, Z, Y, X.
        """
        path, data = bruker_h5
        second = path.with_name("u005a04_20260916_FamDay2_FOV1-002.h5")
        with h5py.File(path, "r") as src, h5py.File(second, "w") as dst:
            src.copy("imaging", dst)
        arr = imread(path.parent)
        assert isinstance(arr, BrukerArray)
        assert arr.filenames == [path]
        assert arr.shape == (12, 1, 1, 16, 20)
        assert "holds 2 recordings" in caplog.text
        assert second.name in caplog.text
        # a list opens the first listed, whatever the names say
        caplog.clear()
        arr = imread([second, path])
        assert isinstance(arr, BrukerArray) and arr.filenames == [second]
        assert path.name in caplog.text
        assert isinstance(imread([path]), BrukerArray)


class TestShape:
    def test_labels_decide_axes(self, bruker_h5):
        path, _ = bruker_h5
        arr = imread(path)
        assert arr.shape == (12, 1, 1, 16, 20)
        assert arr.dims == ("T", "C", "Z", "Y", "X")
        assert arr.metadata["h5_raw_dims"] == "TZYXC"

    def test_indexing_matches_source(self, bruker_h5):
        path, data = bruker_h5
        arr = imread(path)
        expected = np.moveaxis(data, 4, 1)
        np.testing.assert_array_equal(arr[:], expected)
        np.testing.assert_array_equal(arr[3, 0, 0, 2:5, 1:7], expected[3, 0, 0, 2:5, 1:7])
        assert arr[2:5].shape == (3, 1, 1, 16, 20)
        assert np.asarray(arr).shape == (16, 20)

    def test_permuted_volume(self, bruker_volume_h5):
        path, data = bruker_volume_h5
        arr = imread(path)
        assert arr.shape == (4, 2, 3, 8, 10)
        np.testing.assert_array_equal(arr[:], np.moveaxis(data, 2, 1))
        np.testing.assert_array_equal(arr[1, 1, 2], data[1, 2, 1])


class TestMetadata:
    def test_single_plane(self, bruker_h5):
        path, _ = bruker_h5
        arr = imread(path)
        assert arr.dx == pytest.approx(1.10195842)
        assert arr.dy == pytest.approx(1.10195842)
        assert "dz" not in arr.metadata
        assert arr.fs == pytest.approx(1 / 0.0333093252463988)
        assert arr.metadata["channel_names"] == ["Green"]
        assert arr.metadata["imaging_system"] == "bruker"
        assert "DIMENSION_LABELS" not in arr.metadata

    def test_element_size_follows_spatial_order(self, bruker_volume_h5):
        path, _ = bruker_volume_h5
        arr = imread(path)
        assert (arr.dz, arr.dy, arr.dx) == (5.0, 0.8, 0.6)
        assert arr.fs == pytest.approx(10.0)

    def test_counts_agree_with_shape(self, bruker_volume_h5):
        path, _ = bruker_volume_h5
        md = imread(path).metadata
        assert (md["num_timepoints"], md["num_color_channels"], md["num_zplanes"]) == (4, 2, 3)
        assert (md["Ly"], md["Lx"]) == (8, 10)


class TestRoundTrip:
    def test_reader_kwargs(self, bruker_h5):
        path, _ = bruker_h5
        arr = imread(path)
        assert source_reader_kwargs(arr) == {"dataset": "imaging"}
        assert imread(path, **source_reader_kwargs(arr)).shape == arr.shape

    def test_write_zarr(self, bruker_h5, tmp_path):
        path, data = bruker_h5
        arr = imread(path)
        out = tmp_path / "out"
        imwrite(arr, out, ext=".zarr", overwrite=True)
        back = imread(next(out.rglob("*.zarr")))
        assert back.shape == arr.shape
        np.testing.assert_array_equal(back[:], arr[:])
        assert back.dx == pytest.approx(arr.dx)
        assert back.fs == pytest.approx(arr.fs)
