"""Lab4H5Array: the standardized imaging H5 every lab4 converter writes."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from mbo_utilities import imread, imwrite
from mbo_utilities.arrays import H5Array, Lab4H5Array
from mbo_utilities.reader import source_reader_kwargs

# (T, Z, Y, X, C), as lab4/convert/mini2p and bruker write it
SHAPE = (12, 3, 16, 20, 2)


@pytest.fixture
def mini2p_h5(tmp_path):
    data = np.random.default_rng(0).integers(0, 1000, SHAPE, dtype=np.int16)
    path = tmp_path / "mini2p.h5"
    with h5py.File(path, "w") as out:
        out.create_dataset("imaging/data", data=data, chunks=True)
        out["imaging"].attrs["frame_rate_hz"] = 9.5
        out["imaging"].attrs["pixel_size_um"] = 0.8
        out["imaging"].attrs["channel_names"] = ["Green", "Red"]
        out["imaging"].attrs["z_step_um"] = 0.0
        out["imaging"].attrs["start_time"] = "2026-09-15T10:00:00"
        out["imaging"].attrs["device_specifics"] = json.dumps(
            {"objective": "mini", "n_planes": 3, "start_frame_cropped": 0}
        )
    return path, data


@pytest.fixture
def aod_h5(tmp_path):
    """A chessboard scan: four ROIs on the Z slot, one channel."""
    data = np.random.default_rng(1).integers(0, 1000, (8, 4, 6, 6, 1), dtype=np.uint16)
    path = tmp_path / "MSession_0_MUnit_3.h5"
    with h5py.File(path, "w") as out:
        out.create_dataset("imaging/data", data=data, chunks=True)
        out["imaging"].attrs["frame_rate_hz"] = 120.0
        out["imaging"].attrs["pixel_size_um"] = 1.5
        out["imaging"].attrs["channel_names"] = ["Green"]
        out["imaging"].attrs["start_time"] = "2026-09-14T09:00:00-05:00"
        out["imaging"].attrs["device_specifics"] = json.dumps(
            {"modality": 8, "modality_name": "chessboard", "input_entry": "MSession_0/MUnit_3"}
        )
    return path, data


@pytest.fixture
def slm_h5(tmp_path):
    """The SLM route: dimension labels, frame_period, an empty start_time."""
    data = np.random.default_rng(2).integers(0, 1000, (5, 1, 8, 10, 1), dtype=np.uint16)
    path = tmp_path / "imaging.h5"
    with h5py.File(path, "w") as out:
        d = out.require_group("imaging").create_dataset("data", data=data)
        for i, label in enumerate("tzyxc"):
            d.dims[i].label = label
        out["imaging"].attrs["frame_rate_hz"] = 1 / 0.033
        out["imaging"].attrs["frame_period"] = 0.033
        out["imaging"].attrs["pixel_size_um"] = 1.1
        out["imaging"].attrs["channel_names"] = ["Ch2"]
        out["imaging"].attrs["start_time"] = ""
        out["imaging"].attrs["imaging_system"] = "bruker"
    return path, data


class TestDispatch:
    def test_lab4_file_opens_as_lab4(self, mini2p_h5):
        path, _ = mini2p_h5
        assert Lab4H5Array.can_open(path)
        assert isinstance(imread(path), Lab4H5Array)

    def test_imaging_data_without_lab4_attrs_stays_h5(self, tmp_path):
        path = tmp_path / "bare.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("imaging/data", data=np.zeros((3, 1, 4, 4, 1), np.int16))
        assert not Lab4H5Array.can_open(path)
        assert isinstance(imread(path), H5Array)

    def test_non_h5_is_declined(self, tmp_path):
        path = tmp_path / "broken.h5"
        path.write_bytes(b"not hdf5")
        assert not Lab4H5Array.can_open(path)


class TestShape:
    def test_tzyxc_maps_to_tczyx(self, mini2p_h5):
        path, data = mini2p_h5
        arr = imread(path)
        assert arr.shape == (12, 2, 3, 16, 20)
        assert arr.metadata["h5_raw_dims"] == "TZYXC"
        expected = np.moveaxis(data, 4, 1)
        np.testing.assert_array_equal(arr[:], expected)
        np.testing.assert_array_equal(arr[3, 1, 2, 2:5], expected[3, 1, 2, 2:5])
        assert np.asarray(arr).shape == (16, 20)

    def test_dimension_labels_are_read(self, slm_h5):
        path, data = slm_h5
        arr = imread(path)
        assert arr.shape == (5, 1, 1, 8, 10)
        np.testing.assert_array_equal(arr[:, 0, 0], data[:, 0, :, :, 0])

    def test_aod_rois_sit_on_z(self, aod_h5):
        path, data = aod_h5
        arr = imread(path)
        assert arr.shape == (8, 1, 4, 6, 6)
        assert arr.z_is_roi
        assert arr.slider_dim_labels == ("Timepoint", "ROI")
        np.testing.assert_array_equal(arr[:, 0, 2], data[:, 2, :, :, 0])

    def test_a_volume_slides_over_planes(self, mini2p_h5):
        path, _ = mini2p_h5
        arr = imread(path)
        assert not arr.z_is_roi
        assert arr.slider_dim_labels == ("Timepoint", "Channel", "Z-plane")


class TestMetadata:
    def test_canonical_values_resolve(self, mini2p_h5):
        path, _ = mini2p_h5
        arr = imread(path)
        assert arr.fs == pytest.approx(9.5)
        assert arr.dx == arr.dy == pytest.approx(0.8)
        assert arr.metadata["channel_names"] == ["Green", "Red"]
        assert arr.metadata["device_specifics"]["objective"] == "mini"
        assert arr.metadata["start_time"] == "2026-09-15T10:00:00"

    def test_placeholders_stay_absent(self, mini2p_h5, slm_h5):
        assert "z_step_um" not in imread(mini2p_h5[0]).metadata
        assert "start_time" not in imread(slm_h5[0]).metadata

    def test_counts_agree_with_shape(self, mini2p_h5):
        md = imread(mini2p_h5[0]).metadata
        assert (md["num_timepoints"], md["num_color_channels"], md["num_zplanes"]) == (
            12,
            2,
            3,
        )
        assert (md["Ly"], md["Lx"]) == (16, 20)


class TestRoundTrip:
    def test_reader_kwargs(self, mini2p_h5):
        path, _ = mini2p_h5
        arr = imread(path)
        assert source_reader_kwargs(arr) == {}
        assert imread(path, **source_reader_kwargs(arr)).shape == arr.shape

    def test_write_zarr(self, mini2p_h5, tmp_path):
        path, _ = mini2p_h5
        arr = imread(path)
        out = tmp_path / "out"
        imwrite(arr, out, ext=".zarr", overwrite=True)
        back = imread(next(out.rglob("*.zarr")))
        assert back.shape == arr.shape
        np.testing.assert_array_equal(back[:], arr[:])
        assert back.dx == pytest.approx(arr.dx)
        assert back.fs == pytest.approx(arr.fs)
