"""Lab4H5Array: the standardized imaging H5 every lab4 converter writes."""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import pytest
from mbo_utilities import imread, imwrite
from mbo_utilities.arrays import H5Array, Lab4H5Array, Suite2pArray
from mbo_utilities.file_io import load_plane_ops
from mbo_utilities.reader import source_reader_kwargs
from mbo_utilities.results import Results

# (T, Z, Y, X, C), as lab4/convert/mini2p and bruker write it
SHAPE = (12, 3, 16, 20, 2)


def write_suite2p_v1_plane(plane_dir, nt, ly, lx, shift):
    """A suite2p 1.0 plane as lab4 runs it: no ops.npy, its parts split three ways."""
    plane_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    rng.integers(0, 100, (nt, ly, lx), dtype=np.int16).tofile(plane_dir / "data.bin")
    rng.integers(0, 100, (nt, ly, lx), dtype=np.int16).tofile(
        plane_dir / "raw_data.bin"
    )
    settings = {
        "fs": 9.5,
        "tau": 1.0,
        "nplanes": 1,
        "registration": {"nonrigid": True, "block_size": (8, 8), "batch_size": 100},
        "extraction": {"neuropil_coefficient": 0.7, "batch_size": 500},
    }
    reg = {
        "refImg": np.ones((ly, lx), np.float32),
        "meanImg": np.full((ly, lx), 2.0, np.float32),
        "xoff": np.full(nt, shift, np.int32),
        "yoff": -np.full(nt, shift, np.int32),
        "yrange": [1, ly - 1],
        "xrange": [1, lx - 1],
    }
    detect = {"Vcorr": np.zeros((ly - 2, lx - 2), np.float32)}
    for name, value in (
        ("settings", settings),
        ("reg_outputs", reg),
        ("detect_outputs", detect),
    ):
        np.save(plane_dir / f"{name}.npy", value, allow_pickle=True)
    stat = np.array(
        [{"ypix": np.array([1, 2]), "xpix": np.array([3, 3]), "lam": np.ones(2)}],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat, allow_pickle=True)
    for name in ("F", "Fneu", "spks"):
        np.save(plane_dir / f"{name}.npy", rng.random((1, nt)).astype(np.float32))
    np.save(plane_dir / "iscell.npy", np.array([[1.0, 0.9]], np.float32))


@pytest.fixture
def lab4_session(mini2p_h5):
    """A recording with two suite2p runs beside it, the second one newer."""
    path, data = mini2p_h5
    nt, _, ly, lx, _ = data.shape
    old = path.parent / "aaaa" / "suite2p" / "plane0"
    new = path.parent / "bbbb" / "suite2p" / "plane0"
    write_suite2p_v1_plane(old, nt, ly, lx, shift=1)
    write_suite2p_v1_plane(new, nt, ly, lx, shift=3)
    os.utime(old, (1_000_000, 1_000_000))
    return path, new


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
            {
                "modality": 8,
                "modality_name": "chessboard",
                "input_entry": "MSession_0/MUnit_3",
            }
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


class TestSuite2pV1:
    def test_ops_are_rebuilt_from_the_split_files(self, lab4_session):
        _, plane = lab4_session
        ops = load_plane_ops(plane)
        assert ops["fs"] == 9.5
        assert ops["nonrigid"] is True and tuple(ops["block_size"]) == (8, 8)
        assert (ops["Ly"], ops["Lx"], ops["nframes"]) == (16, 20, 12)
        assert ops["Vcorr"].shape == (14, 18)

    def test_ops_npy_wins_over_settings(self, lab4_session):
        _, plane = lab4_session
        np.save(plane / "ops.npy", {"fs": 30.0, "Ly": 16, "Lx": 20}, allow_pickle=True)
        assert load_plane_ops(plane)["fs"] == 30.0

    def test_plane_opens_as_suite2p(self, lab4_session):
        _, plane = lab4_session
        arr = imread(plane)
        assert isinstance(arr, Suite2pArray)
        assert arr.shape == (12, 1, 1, 16, 20)
        assert arr.fs == pytest.approx(9.5)
        assert np.all(arr.motion_correction.traces["X"][1] == 3)
        assert arr.results.units["zplane01"].n_rois == 1

    def test_raw_binary_is_found(self, lab4_session):
        _, plane = lab4_session
        arr = Suite2pArray(plane, use_raw=True)
        assert arr.filenames == [plane / "raw_data.bin"]

    def test_results_place_rois_and_carry_fs_and_images(self, lab4_session):
        _, plane = lab4_session
        unit = Results.from_suite2p(plane).units["zplane01"]
        assert unit.fs == pytest.approx(9.5)
        assert unit.image_shape == (16, 20)
        np.testing.assert_array_equal(unit.members[0], [1 * 20 + 3, 2 * 20 + 3])
        assert set(unit.images) == {"mean", "ref", "corr"}
        assert unit.images["corr"].shape == (16, 20)
        assert set(unit.traces) == {"raw", "neuropil", "spikes"}

    def test_cropped_images_sit_in_the_full_frame(self, lab4_session):
        _, plane = lab4_session
        detect = {"Vcorr": np.ones((14, 18), np.float32)}
        np.save(plane / "detect_outputs.npy", detect, allow_pickle=True)
        corr = Results.from_suite2p(plane).units["zplane01"].images["corr"]
        assert corr[1:15, 1:19].all()
        assert not corr[0].any() and not corr[:, 0].any()

    def test_rois_without_a_frame_size_are_refused(self, lab4_session):
        _, plane = lab4_session
        for name in ("settings", "reg_outputs", "detect_outputs"):
            (plane / f"{name}.npy").unlink()
        with pytest.raises(ValueError, match="cannot be placed"):
            Results.from_suite2p(plane)


class TestLab4Runs:
    def test_newest_run_gives_results_and_shifts(self, lab4_session):
        path, plane = lab4_session
        arr = imread(path)
        assert arr.suite2p_planes == [plane]
        mc = arr.motion_correction
        assert (mc.source, mc.unit) == ("suite2p", "px")
        np.testing.assert_array_equal(mc.traces["X"][1], 3)
        np.testing.assert_array_equal(mc.traces["Y"][1], -3)
        assert mc.traces["X"][0][-1] == pytest.approx(11 / 9.5)
        unit = arr.results.units["zplane01"]
        assert unit.attrs["plane_dir"] == str(plane)

    def test_no_run_gives_nothing(self, aod_h5):
        arr = imread(aod_h5[0])
        assert arr.suite2p_planes == []
        assert arr.motion_correction is None
        assert arr.results is None
