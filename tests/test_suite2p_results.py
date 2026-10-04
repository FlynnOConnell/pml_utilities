"""A suite2p output opened as a ``Suite2pArray`` carries the detection it wrote
on ``arr.results`` (AGENTS.md §7.5): one unit per plane in the array's z order.
"""

import numpy as np
from mbo_utilities import imread
from mbo_utilities.arrays.features import MotionCorrection
from mbo_utilities.arrays.numpy import NumpyArray
from mbo_utilities.arrays.suite2p import Suite2pArray
from mbo_utilities.results import Results

T, Y, X = 6, 8, 10


def _plane(plane_dir, plane_1based, detected=True, offsets=None):
    plane_dir.mkdir(parents=True)
    mov = np.full((T, Y, X), 100, np.int16)
    mov.tofile(plane_dir / "data.bin")
    ops = {"Ly": Y, "Lx": X, "fs": 5.0, "nframes": T, "plane": plane_1based}
    if offsets is not None:
        ops["xoff"], ops["yoff"] = offsets
    np.save(plane_dir / "ops.npy", ops, allow_pickle=True)
    if not detected:
        return
    stat = np.array(
        [
            {
                "ypix": np.array([1, 1]),
                "xpix": np.array([2, 3]),
                "lam": np.array([0.5, 1.0], np.float32),
            }
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "F.npy", np.full((1, T), float(plane_1based), np.float32))
    np.save(plane_dir / "Fneu.npy", np.zeros((1, T), np.float32))
    np.save(plane_dir / "spks.npy", np.ones((1, T), np.float32))
    np.save(plane_dir / "iscell.npy", np.array([[1, 0.9]], np.float32))


def test_a_volume_answers_one_unit_per_plane_in_z_order(tmp_path):
    # plane numbers with a gap: z is the position in the stack, not the number
    _plane(tmp_path / "zplane03_tp00001-00006", 3)
    _plane(tmp_path / "zplane01_tp00001-00006", 1)
    arr = imread(tmp_path)
    assert isinstance(arr, Suite2pArray) and arr.shape == (T, 1, 2, Y, X)
    results = arr.results
    assert isinstance(results, Results) and results.pipeline == "suite2p"
    assert list(results.units) == ["zplane01", "zplane03"]
    units = list(results.units.values())
    assert [u.index for u in units] == [1, 3] and [u.attrs["z"] for u in units] == [
        0,
        1,
    ]
    assert [u.attrs["plane_dir"] for u in units] == [
        str(tmp_path / "zplane01_tp00001-00006"),
        str(tmp_path / "zplane03_tp00001-00006"),
    ]
    # each unit holds its own plane's traces, with the spikes along
    np.testing.assert_allclose(units[1].traces["raw"], 3.0)
    assert set(units[0].traces) == {"raw", "neuropil", "spikes"}
    assert units[0].fs == 5.0 and units[0].image_shape == (Y, X)
    # read once
    assert arr.results is results


def test_a_plane_dir_answers_its_one_unit_at_z_zero(tmp_path):
    _plane(tmp_path / "zplane01_tp00001-00006", 1)
    _plane(tmp_path / "zplane02_tp00001-00006", 2)
    plane = imread(tmp_path / "zplane02_tp00001-00006")
    assert plane.shape == (T, 1, 1, Y, X)
    assert list(plane.results.units) == ["zplane02"]
    assert plane.results["zplane02"].attrs["z"] == 0
    # the plane's data.bin opens the same plane
    by_file = imread(tmp_path / "zplane02_tp00001-00006" / "data.bin")
    assert list(by_file.results.units) == ["zplane02"]


def test_a_registration_only_run_has_no_results(tmp_path):
    _plane(tmp_path / "zplane01_tp00001-00006", 1, detected=False)
    arr = imread(tmp_path)
    assert isinstance(arr, Suite2pArray) and arr.results is None
    # a volume with one detected plane keeps that plane's z
    _plane(tmp_path / "zplane02_tp00001-00006", 2)
    arr = imread(tmp_path)
    assert arr.shape[2] == 2
    assert [(u.name, u.attrs["z"]) for u in arr.results.units.values()] == [
        ("zplane02", 1)
    ]


def test_a_registered_plane_reports_its_offsets_as_motion_correction(tmp_path):
    xoff, yoff = np.arange(T, dtype=np.int32) - 2, np.full(T, 3, np.int32)
    _plane(tmp_path / "zplane01_tp00001-00006", 1, offsets=(xoff, yoff))
    motion = imread(tmp_path / "zplane01_tp00001-00006").motion_correction
    assert isinstance(motion, MotionCorrection) and motion
    assert (motion.source, motion.unit, motion.planes) == ("suite2p", "px", {})
    assert list(motion.traces) == ["X", "Y"]
    t, shift = motion.traces["X"]
    # seconds on the recording's time axis, fs 5
    np.testing.assert_allclose(t, np.arange(T) / 5.0)
    np.testing.assert_array_equal(shift, xoff)
    np.testing.assert_array_equal(motion.traces["Y"][1], yoff)


def test_a_plane_that_never_moved_reports_no_motion_correction(tmp_path):
    _plane(tmp_path / "a" / "zplane01_tp00001-00006", 1)
    assert imread(tmp_path / "a" / "zplane01_tp00001-00006").motion_correction is None
    still = np.zeros(T, np.int32)
    _plane(tmp_path / "b" / "zplane01_tp00001-00006", 1, offsets=(still, still))
    assert imread(tmp_path / "b" / "zplane01_tp00001-00006").motion_correction is None


def test_a_volumes_offsets_are_listed_by_the_plane_they_belong_to(tmp_path):
    ramp = np.arange(T, dtype=np.int32)
    _plane(tmp_path / "zplane01_tp00001-00006", 1, offsets=(ramp, -ramp))
    # registered apart from its neighbours, and this one never ran
    _plane(tmp_path / "zplane02_tp00001-00006", 2)
    _plane(tmp_path / "zplane03_tp00001-00006", 3, offsets=(2 * ramp, ramp))
    motion = imread(tmp_path).motion_correction
    assert motion.planes == {
        "X zplane01_tp00001-00006": 0,
        "Y zplane01_tp00001-00006": 0,
        "X zplane03_tp00001-00006": 2,
        "Y zplane03_tp00001-00006": 2,
    }
    assert list(motion.traces) == list(motion.planes)
    np.testing.assert_array_equal(motion.traces["X zplane03_tp00001-00006"][1], 2 * ramp)


def test_results_can_be_handed_to_any_array():
    arr = NumpyArray(np.zeros((2, 4, 4), np.float32), dims="TYX")
    assert arr.results is None
    given = Results(pipeline="test")
    arr.results = given
    assert arr.results is given
