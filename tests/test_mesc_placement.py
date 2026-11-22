"""``analysis.mesc_placement``: measuring where a scan sits in a Z-stack.

Fixture: one textured volume of tissue. The Z-stack rasters it; the scan's
picture rasters a plane of it, but both units store positions in frames the
file does not relate, so the true shift is only in the pixels. The lines' time
means are the tissue under them.
"""

from __future__ import annotations

import json
import shutil

import h5py
import numpy as np
import pytest
from mbo_utilities.analysis.mesc_placement import (
    NCC_PASS,
    place_scan,
    zstack_files,
)
from mbo_utilities.arrays.mesc import MescArray
from mbo_utilities.arrays.mesc_geometry import image_overlays, zstack_depth_info
from scipy import ndimage

UM_PER_PX = 0.5
STACK_PX = 200
PICTURE_PX = 120
N_SLICES = 16
Z_STEP = 2.0
STACK_CORNER = (100.0, 200.0, -50.0)
# where the tissue the picture shows really sits, in stack pixels and slices
PICTURE_AT = (40, 30, 7)
# the picture's stored position disagrees with that by this much
STAGE_ERROR = (31.0, -17.0, 5.0)
RTMC = (-2.0, 1.0, 3.0)
N_FRAMES = 6
LINE_PX = 40
N_LINES = 6
MIN_Z = -14.0


def tissue(seed: int = 0) -> np.ndarray:
    """``(Z, Y, X)`` of blobs a few pixels wide, uncorrelated a few slices apart."""
    rng = np.random.default_rng(seed)
    volume = ndimage.gaussian_filter(
        rng.normal(size=(N_SLICES, STACK_PX, STACK_PX)), (1.2, 3, 3)
    )
    return 3000 + 40000 * np.clip(volume - volume.min(), 0, None)


def line_ends() -> list[list]:
    """Stored ends of ``N_LINES`` lines, in the picture's frame, on its plane."""
    px, py, k = PICTURE_AT
    picture_x = STACK_CORNER[0] + px * UM_PER_PX + STAGE_ERROR[0]
    picture_y = STACK_CORNER[1] + py * UM_PER_PX + STAGE_ERROR[1]
    z = stored_picture_z()
    return [
        [
            [picture_x + 4 + 7 * i, picture_x + 4 + 7 * i + 12],
            [picture_y + 5 + 8 * i, picture_y + 5 + 8 * i + 6],
            [z, z],
        ]
        for i in range(N_LINES)
    ]


def stored_picture_z() -> float:
    return STACK_CORNER[2] + MIN_Z + PICTURE_AT[2] * Z_STEP + STAGE_ERROR[2]


def write_stack(group, volume, corner, time=0):
    stack = group.create_group("MUnit_0")
    stack.attrs.update(
        {
            "MethodType": 2,
            "VecChannelsSize": 1,
            "TStepInMs": 1.0,
            "MeasurementDatePosix": time,
            "MinZ": MIN_Z,
            "MaxZ": MIN_Z + (N_SLICES - 1) * Z_STEP,
            "ZDim": N_SLICES,
        }
    )
    stack.attrs["ReferenceViewportJSON"] = json.dumps(
        {
            "viewports": [
                {
                    "geomTransTransl": list(corner),
                    "width": STACK_PX * UM_PER_PX,
                    "height": STACK_PX * UM_PER_PX,
                }
            ]
        }
    )
    stack.create_dataset("Channel_0", data=volume.astype(np.uint16))


def write_scan(path, volume, picture=None, time=600, beam_um=(0.0, 0.0)):
    """A line scan beside its picture; ``volume`` is the tissue under both and
    ``beam_um`` how far from the stored lines the beam really was.
    """
    px, py, k = PICTURE_AT
    if picture is None:
        picture = volume[k, py : py + PICTURE_PX, px : px + PICTURE_PX]
    lines = line_ends()
    means = []
    for x, y, _z in lines:
        col = (
            np.linspace(x[0], x[1], LINE_PX, endpoint=False)
            + (x[1] - x[0]) / LINE_PX / 2
            - STAGE_ERROR[0]
            - STACK_CORNER[0]
            + beam_um[0]
        ) / UM_PER_PX
        row = (
            np.linspace(y[0], y[1], LINE_PX, endpoint=False)
            + (y[1] - y[0]) / LINE_PX / 2
            - STAGE_ERROR[1]
            - STACK_CORNER[1]
            + beam_um[1]
        ) / UM_PER_PX
        means.append(
            ndimage.map_coordinates(volume[k], [row - 0.5, col - 0.5], order=1)
        )
    page = np.tile(np.concatenate(means)[None, :], (N_FRAMES, 1))
    with h5py.File(path, "a") as f:
        session = f.require_group("MSession_0")
        scan = session.create_group("MUnit_1")
        scan.attrs.update(
            {
                "MethodType": 6,
                "VecChannelsSize": 1,
                "TStepInMs": 2.0,
                "MeasurementDatePosix": time,
                "BackgroundImagePath": "/MSession_1/MUnit_0",
            }
        )
        boxes = [
            {
                "lowerLeftFramePix": [LINE_PX * i + 1, 1],
                "upperRightFramePix": [LINE_PX * (i + 1), 1],
            }
            for i in range(N_LINES)
        ]
        scan.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": boxes, "driftEndPoints": lines}]}
        )
        scan.create_dataset("Channel_0", data=page[None].astype(np.uint16))
        for axis, value in zip("XYZ", RTMC):
            scan.create_group(f"Curve_{axis}")
        snapshot = f.require_group("MSession_1").create_group("MUnit_0")
        snapshot.attrs.update(
            {
                "MethodType": 1,
                "VecChannelsSize": 1,
                "TStepInMs": 1.0,
                "MeasurementDatePosix": time,
                "ImageRoleDebugString": "background",
            }
        )
        snapshot.attrs["ReferenceViewportJSON"] = json.dumps(
            {
                "viewports": [
                    {
                        "geomTransTransl": [
                            STACK_CORNER[0] + px * UM_PER_PX + STAGE_ERROR[0],
                            STACK_CORNER[1] + py * UM_PER_PX + STAGE_ERROR[1],
                            stored_picture_z(),
                        ],
                        "width": PICTURE_PX * UM_PER_PX,
                        "height": PICTURE_PX * UM_PER_PX,
                    }
                ]
            }
        )
        snapshot.create_dataset("Channel_0", data=picture[None].astype(np.uint16))


@pytest.fixture(scope="module")
def scan_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("placement") / "scan.mesc"
    volume = tissue()
    with h5py.File(path, "w") as f:
        write_stack(f.create_group("MSession_0"), volume, STACK_CORNER)
    write_scan(path, volume)
    return path


def open_scan(path, rtmc=RTMC):
    arr = MescArray(path, unit="MSession_0/MUnit_1")
    arr._rtmc = {
        f"{axis} total": {"t": np.arange(3.0), "um": np.full(3, value)}
        for axis, value in zip("XYZ", rtmc)
    }
    return arr


def test_the_picture_gives_the_shift_the_file_does_not(scan_path):
    """The stored positions are off by ``STAGE_ERROR``; the lines' means lie
    on the tissue as stored, so RTMC stays out of the shift.
    """
    arr = open_scan(scan_path)
    try:
        placement = place_scan(arr, scan_path, "MSession_0/MUnit_0")
    finally:
        arr.close()
    assert placement.placed and placement.score > 0.9
    expected = -np.array(STAGE_ERROR)
    np.testing.assert_allclose(placement.shift_um, expected, atol=1.0)
    assert not placement.with_rtmc
    assert placement.gap_s == 600


def test_rtmc_joins_the_shift_when_the_beam_left_the_tissue(tmp_path):
    """The means lie under stored + RTMC, which only the recording can show."""
    rtmc = (-4.0, 3.0, 0.0)
    path = tmp_path / "scan.mesc"
    volume = tissue()
    with h5py.File(path, "w") as f:
        write_stack(f.create_group("MSession_0"), volume, STACK_CORNER)
    write_scan(path, volume, beam_um=rtmc[:2])
    arr = open_scan(path, rtmc)
    try:
        placement = place_scan(arr, path, "MSession_0/MUnit_0")
    finally:
        arr.close()
    assert placement.with_rtmc and placement.confirmed
    expected = np.array(rtmc) - STAGE_ERROR
    np.testing.assert_allclose(placement.shift_um, expected, atol=1.0)


def test_the_recording_confirms_a_right_placement(scan_path):
    arr = open_scan(scan_path)
    arr._rtmc = {}
    try:
        placement = place_scan(arr, scan_path, "MSession_0/MUnit_0")
    finally:
        arr.close()
    assert placement.profile_r > 0.8 and placement.confirmed


def test_lines_land_on_the_slice_the_tissue_is_on(scan_path):
    arr = open_scan(scan_path)
    arr._rtmc = {}
    try:
        placement = place_scan(arr, scan_path, "MSession_0/MUnit_0")
    finally:
        arr.close()
    records = image_overlays(
        scan_path,
        "MSession_0/MUnit_1",
        "MSession_0/MUnit_0",
        shift_um=placement.shift_um,
    )
    assert [r["slice"] for r in records] == [PICTURE_AT[2]] * N_LINES


@pytest.mark.parametrize("tissue_on", [0, 1])
def test_the_channel_the_recording_fits_wins(tmp_path, tissue_on):
    """Scan, picture and stack hold the tissue on one channel and noise on the
    other: that channel answers wherever the viewer is.
    """
    path = tmp_path / "scan.mesc"
    volume = tissue()
    with h5py.File(path, "w") as f:
        write_stack(f.create_group("MSession_0"), volume, STACK_CORNER)
    write_scan(path, volume)
    rng = np.random.default_rng(1)
    with h5py.File(path, "r+") as f:
        for key in ("MSession_0/MUnit_0", "MSession_0/MUnit_1", "MSession_1/MUnit_0"):
            unit = f[key]
            data = unit["Channel_0"][:]
            noise = rng.integers(0, 60000, data.shape, dtype=np.uint16)
            del unit["Channel_0"]
            unit["Channel_0"] = data if tissue_on == 0 else noise
            unit["Channel_1"] = noise if tissue_on == 0 else data
            unit.attrs["VecChannelsSize"] = 2
    arr = open_scan(path, (0.0, 0.0, 0.0))
    try:
        placement = place_scan(arr, path, "MSession_0/MUnit_0")
    finally:
        arr.close()
    assert placement.channel == tissue_on and placement.confirmed
    np.testing.assert_allclose(placement.shift_um, -np.array(STAGE_ERROR), atol=1.0)


def test_a_top_to_bottom_stack_is_not_guessed_at(tmp_path):
    path = tmp_path / "stack.mesc"
    with h5py.File(path, "w") as f:
        write_stack(f.create_group("MSession_0"), tissue(), STACK_CORNER)
        f["MSession_0/MUnit_0"].attrs["ScanningDirection"] = 1
    with pytest.raises(ValueError, match="topToBottom"):
        zstack_depth_info(path, "MSession_0/MUnit_0")


def test_a_picture_of_other_tissue_is_refused(tmp_path):
    """No match, no position: the score says so and the ROIs get no check."""
    path = tmp_path / "other.mesc"
    with h5py.File(path, "w") as f:
        write_stack(f.create_group("MSession_0"), tissue(), STACK_CORNER)
    write_scan(path, tissue(seed=1))
    arr = open_scan(path)
    try:
        placement = place_scan(arr, path, "MSession_0/MUnit_0")
    finally:
        arr.close()
    assert placement.score < NCC_PASS and not placement.placed
    assert placement.profile_r is None and not placement.confirmed


def test_a_picture_larger_than_the_stack_still_matches(tmp_path):
    """The stack's field is inside the picture: only the overlap is scored."""
    volume = tissue()
    px, py, k = PICTURE_AT
    small = volume[:, 60:140, 60:140]
    path = tmp_path / "wide.mesc"
    with h5py.File(path, "w") as f:
        stack_px = 80
        write_stack(f.create_group("MSession_0"), small, STACK_CORNER)
        viewports = json.loads(f["MSession_0/MUnit_0"].attrs["ReferenceViewportJSON"])
        viewports["viewports"][0]["width"] = stack_px * UM_PER_PX
        viewports["viewports"][0]["height"] = stack_px * UM_PER_PX
        f["MSession_0/MUnit_0"].attrs["ReferenceViewportJSON"] = json.dumps(viewports)
    write_scan(
        path, volume, picture=volume[k, py : py + PICTURE_PX, px : px + PICTURE_PX]
    )
    arr = open_scan(path)
    arr._rtmc = {}
    try:
        placement = place_scan(arr, path, "MSession_0/MUnit_0")
    finally:
        arr.close()
    assert placement.placed
    # the small stack's corner is 60 px into the tissue, the picture's 40, 30
    expected_x = -STAGE_ERROR[0] + (60 - 60) * UM_PER_PX
    np.testing.assert_allclose(
        placement.shift_um[0], expected_x - 60 * UM_PER_PX, atol=1.0
    )


def test_a_stack_in_another_file_is_measured_the_same_way(scan_path, tmp_path):
    path = tmp_path / "scan.mesc"
    shutil.copy(scan_path, path)
    (tmp_path / "scan_z.mesc").write_bytes(scan_path.read_bytes())
    arr = open_scan(path)
    arr._rtmc = {}
    try:
        placement = place_scan(arr, tmp_path / "scan_z.mesc", "MSession_0/MUnit_0")
    finally:
        arr.close()
    assert placement.placed
    assert placement.stack_path == str(tmp_path / "scan_z.mesc")


def test_a_scan_without_a_picture_has_no_placement(scan_path, tmp_path):
    path = tmp_path / "bare.mesc"
    shutil.copy(scan_path, path)
    with h5py.File(path, "a") as f:
        del f["MSession_0/MUnit_1"].attrs["BackgroundImagePath"]
    arr = MescArray(path, unit="MSession_0/MUnit_1")
    try:
        assert place_scan(arr, path, "MSession_0/MUnit_0") is None
    finally:
        arr.close()


def test_stack_files_are_found_beside_the_scan_or_one_folder_up(tmp_path):
    inner = tmp_path / "expt" / "expt"
    inner.mkdir(parents=True)
    scan = inner / "expt.mesc"
    scan.touch()
    assert zstack_files(scan) == []
    (inner.parent / "expt_z.mesc").touch()
    (inner / "expt_zstack.mesc").touch()
    assert zstack_files(scan) == [
        inner / "expt_zstack.mesc",
        inner.parent / "expt_z.mesc",
    ]


def test_a_fast_stacks_rtmc_is_a_file_coordinate_not_drift(scan_path, tmp_path):
    """The same pixels under a fast stack that stores RTMC: the shift grows by
    it, because the stack's own coordinates now carry that much, and the lines
    still land on the slice the tissue is on.
    """
    stack_rtmc = (3.0, -2.0, 4.0)
    path = tmp_path / "scan.mesc"
    shutil.copy(scan_path, path)
    with h5py.File(path, "a") as f:
        stack = f["MSession_0/MUnit_0"]
        stack.attrs["ZModeDebugString"] = "fast"
        for i, (axis, value) in enumerate(zip("XYZ", stack_rtmc)):
            curve = stack.create_group(f"Curve_{i}")
            curve.attrs["Name"] = f"RTMC {axis} correction (total)"
            curve.attrs["CurveDataXRawDelta"] = 1.0
            curve.create_dataset("CurveDataYIdxNextSample", data=np.arange(1, 5))
            curve.create_dataset("CurveDataYRawData", data=np.full(4, value))
    arr = open_scan(path)
    arr._rtmc = {}
    try:
        placement = place_scan(arr, path, "MSession_0/MUnit_0")
    finally:
        arr.close()
    assert placement.placed and placement.confirmed
    expected = np.array(stack_rtmc) - STAGE_ERROR
    np.testing.assert_allclose(placement.shift_um, expected, atol=1.0)
    records = image_overlays(
        path, "MSession_0/MUnit_1", "MSession_0/MUnit_0", shift_um=placement.shift_um
    )
    assert [r["slice"] for r in records] == [PICTURE_AT[2]] * N_LINES


@pytest.mark.parametrize(
    "quaternion", [[0.0, 0.259, 0.0, 0.966], [0.0, 0.0, 0.3827, 0.9239]]
)
def test_a_picture_rotated_against_its_stack_is_not_placed(
    scan_path, tmp_path, quaternion
):
    """Tilted 30 degrees about y, or turned 45 degrees in its own plane: the
    match is a translation, so neither is placed.
    """
    path = tmp_path / "scan.mesc"
    shutil.copy(scan_path, path)
    with h5py.File(path, "a") as f:
        picture = f["MSession_1/MUnit_0"]
        viewports = json.loads(picture.attrs["ReferenceViewportJSON"])
        viewports["viewports"][0]["geomTransRot"] = quaternion
        picture.attrs["ReferenceViewportJSON"] = json.dumps(viewports)
    arr = open_scan(path)
    try:
        assert place_scan(arr, path, "MSession_0/MUnit_0") is None
    finally:
        arr.close()
