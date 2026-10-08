"""masknmf run folders: the registered movie rebuilt from the recording the run read."""

import json

import h5py
import numpy as np
import pytest
import tifffile

from mbo_utilities import imread
from mbo_utilities.arrays.masknmf_run import MasknmfRunArray, is_masknmf_run, run_raw_movie
from mbo_utilities.results import Results, ResultUnit, results_name

T, Y, X = 20, 6, 8


@pytest.fixture
def recording(tmp_path):
    rng = np.random.default_rng(0)
    data = (100 + rng.random((T, Y, X)) * 50).astype(np.int16)
    path = tmp_path / "rec.tif"
    tifffile.imwrite(path, data)
    return path, data


@pytest.fixture
def run(tmp_path, recording):
    """A run folder without a registration: frames 4..15 of the recording, inverted."""
    path, _ = recording
    folder = tmp_path / "20261007T120000_masknmf_zplane01_tp00005-00016"
    folder.mkdir()
    with h5py.File(folder / "results.hdf5", "w") as f:
        f.create_group("CompressionArray")
    movie = {
        "path": str(path),
        "reader_kwargs": {},
        "read_features": {"invert_deflection": True},
        "plane": 1,
        "z": 0,
        "c": 0,
        "frames": [4, 16, 1],
        "tp_indices": None,
        "fs": 30.0,
    }
    config = {"pipeline": "mbo_utilities.masknmf", "inputs": {"movie": movie}, "configs": {}}
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def flipped(data):
    data = data.astype(np.float32)
    return 2 * data.mean(axis=0) - data


def test_a_run_folder_opens_as_its_movie(run):
    assert is_masknmf_run(run) and not is_masknmf_run(run.parent)
    arr = imread(run)
    assert isinstance(arr, MasknmfRunArray)
    assert arr.shape == (12, 1, 1, Y, X) and arr.dtype == np.float32
    assert arr.metadata["fs"] == 30.0


def test_raw_is_the_recording_as_the_run_read_it(run, recording):
    _, data = recording
    raw = run_raw_movie(run)
    assert raw.shape == (12, Y, X)
    np.testing.assert_allclose(raw[:], flipped(data)[4:16], rtol=1e-5)


def test_without_a_registration_the_movie_is_the_raw_one(run, recording):
    _, data = recording
    arr = MasknmfRunArray(run)
    assert arr.shifts is None and arr.motion_correction is None
    np.testing.assert_allclose(arr[3, 0, 0], flipped(data)[7], rtol=1e-5)
    np.testing.assert_allclose(arr[2:5, 0, 0, 1:3], flipped(data)[6:9, 1:3], rtol=1e-5)


def test_shifts_become_motion_traces(run):
    shifts = np.stack([np.arange(12), -np.arange(12)], axis=1).astype(np.float32)
    with h5py.File(run / "results.hdf5", "a") as f:
        f.create_group("RigidRegistrationArray").create_dataset("shifts", data=shifts)
    mc = MasknmfRunArray(run).motion_correction
    assert mc.source == "masknmf" and mc.unit == "px"
    t, y = mc.traces["Y"]
    np.testing.assert_allclose(t, np.arange(12) / 30.0)
    np.testing.assert_array_equal(y, shifts[:, 0])
    np.testing.assert_array_equal(mc.traces["X"][1], shifts[:, 1])


def test_the_results_file_in_the_folder_is_the_arrays_results(run):
    from mbo_utilities.gui.roi_runs import run_dir_complete

    assert MasknmfRunArray(run).results is None
    assert not run_dir_complete(run)
    unit = ResultUnit(
        name="zplane01",
        kind="plane",
        index=1,
        fs=30.0,
        roi_names=["0"],
        traces={"raw": np.ones((1, 12), np.float32)},
        members=[np.array([0, 1])],
        weights=[np.ones(2, np.float32)],
        image_shape=(Y, X),
        attrs={"plane_dir": str(run), "z": 3},
    )
    Results(pipeline="masknmf", units={"zplane01": unit}).write(
        run / results_name("rec.tif", pipeline="masknmf")
    )
    results = MasknmfRunArray(run).results
    assert results.units["zplane01"].attrs["z"] == 0
    assert run_dir_complete(run)
    assert Results.open(run).units["zplane01"].n_rois == 1


def test_a_run_on_an_in_memory_array_has_no_movie(run):
    config = json.loads((run / "config.json").read_text())
    config["inputs"]["movie"]["path"] = None
    (run / "config.json").write_text(json.dumps(config))
    assert run_raw_movie(run) is None
    with pytest.raises(ValueError):
        MasknmfRunArray(run)
