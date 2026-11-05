"""``arrays.mesc_geometry``: AOD lines and patches in microns and on images.

Fixture: a Z-stack whose FOV corner sits at (100, 200, -50) um, 40 x 32 um,
11 slices from z = -10 to +10 um relative to that origin (2 um steps), and a
line-scan unit with four lines at different depths: two on one slice, one
alone, one scanned below the stack.
"""

from __future__ import annotations

import json
import shutil

import h5py
import numpy as np
import pytest
from mbo_utilities.arrays.mesc_geometry import (
    image_overlays,
    line_positions,
    roi_outlines_um,
    um_to_pixels,
    viewport_geometry,
    zstack_depth_info,
)

TRANSL = (100.0, 200.0, -50.0)
# the snapshot the lines were drawn on: same field, focused 4 um below the stack origin
SNAP_TRANSL = (100.0, 200.0, -54.0)
LINE_COLORS = ["#FFFF0000", "#FF00FF00", "#FF0000FF", "#FFFFFFFF"]
LINES = [
    # x0,x1     y0,y1       z0,z1   (absolute microns)
    [[110, 130], [210, 210], [-54, -54]],  # z -4 -> slice 3
    [[120, 120], [204, 228], [-46, -46]],  # z +4 -> slice 7
    [[105, 135], [220, 230], [-53.9, -53.9]],  # z -3.9 -> slice 3 too
    [[112, 118], [212, 212], [-70, -70]],  # z -20 -> below the stack
]
PATCHES = [
    # x0..x3        y0..y3                z0..z3   (a chessboard patch's corners, absolute microns)
    [
        [110, 130, 130, 110],
        [210, 210, 230, 230],
        [-46, -46, -46, -46],
    ],  # z +4 -> slice 7
    [
        [100, 120, 120, 100],
        [200, 200, 220, 220],
        [-70, -70, -70, -70],
    ],  # below the stack
]


@pytest.fixture(scope="module")
def mesc_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("geom") / "geom.mesc"
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")
        z = s.create_group("MUnit_0")
        z.attrs.update(
            {
                "MethodType": 2,
                "VecChannelsSize": 1,
                "TStepInMs": 1.0,
                "MeasurementDatePosix": 0,
                "Comment": "zstack",
                "MinZ": -10.0,
                "MaxZ": 10.0,
                "ZDim": 11,
            }
        )
        z.attrs["ReferenceViewportJSON"] = json.dumps(
            {
                "viewports": [
                    {"geomTransTransl": list(TRANSL), "width": 40.0, "height": 32.0}
                ]
            }
        )
        z.create_dataset("Channel_0", data=np.zeros((11, 64, 80), np.uint16))

        ls = s.create_group("MUnit_1")
        ls.attrs.update(
            {
                "MethodType": 6,
                "VecChannelsSize": 1,
                "TStepInMs": 2.0,
                "MeasurementDatePosix": 1,
                "Comment": "linescan",
            }
        )
        # four one-line ROIs packed side by side on the 8 x 8 page, 1-based corners
        boxes = [
            {"lowerLeftFramePix": [2 * i + 1, 1], "upperRightFramePix": [2 * i + 2, 1]}
            for i in range(4)
        ]
        ls.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": boxes, "driftEndPoints": LINES}]}
        )
        ls.attrs["BackgroundImagePath"] = "/MSession_0/MUnit_4"
        ls.attrs["ROIJSON"] = json.dumps({"rois": [{"color": c} for c in LINE_COLORS]})
        ls.create_dataset("Channel_0", data=np.zeros((1, 8, 8), np.uint16))

        bare = s.create_group("MUnit_2")
        bare.attrs.update(
            {
                "MethodType": 1,
                "VecChannelsSize": 1,
                "TStepInMs": 1.0,
                "MeasurementDatePosix": 2,
            }
        )
        bare.create_dataset("Channel_0", data=np.zeros((2, 4, 4), np.uint16))

        chess = s.create_group("MUnit_3")
        chess.attrs.update(
            {
                "MethodType": 8,
                "VecChannelsSize": 1,
                "TStepInMs": 5.0,
                "MeasurementDatePosix": 3,
                "Comment": "chessboard",
            }
        )
        chess.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": [], "contours": PATCHES}]}
        )
        # MESc 4.6.2 layout: a scanner list and a 0-based index beside it
        chess.attrs["MultiROIProtocolJSON"] = json.dumps(
            {
                "protocol": {"mainPatternIndex": 1, "scanners": [{"name": "AO1"}]},
                "scanPatterns": {
                    "patterns": [
                        {
                            "centerPoints": [[0.0], [0.0], [0.0]],
                            "pixelSizeX": 1.0,
                            "rotation": [0, 0, 0, 1],
                        },
                        {
                            "centerPoints": [
                                [120.0, 110.0],
                                [220.0, 210.0],
                                [-46.0, -70.0],
                            ],
                            "pixelSizeX": 1.0,
                            "rotation": [0, 0, 0, 1],
                        },
                    ]
                },
            }
        )
        chess.create_dataset("Channel_0", data=np.zeros((1, 20, 40), np.uint16))

        snap = s.create_group("MUnit_4")
        snap.attrs.update(
            {
                "MethodType": 1,
                "VecChannelsSize": 1,
                "TStepInMs": 1.0,
                "MeasurementDatePosix": 0,
                "ImageRoleDebugString": "background",
            }
        )
        snap.attrs["ReferenceViewportJSON"] = json.dumps(
            {
                "viewports": [
                    {
                        "geomTransTransl": list(SNAP_TRANSL),
                        "width": 40.0,
                        "height": 32.0,
                    }
                ]
            }
        )
        snap.create_dataset("Channel_0", data=np.zeros((1, 64, 80), np.uint16))
    return path


@pytest.fixture(scope="module")
def depth(mesc_path):
    return zstack_depth_info(mesc_path, "MSession_0/MUnit_0")


@pytest.fixture(scope="module")
def lines(mesc_path):
    return roi_outlines_um(mesc_path, "MSession_0/MUnit_1")


def test_readers(mesc_path, depth, lines):
    assert len(lines) == 4 and lines[0].shape == (3, 2)
    vp = viewport_geometry(mesc_path, "MSession_0/MUnit_0")
    assert vp == {"transl": TRANSL, "width": 40.0, "height": 32.0, "angle": 0.0}
    assert depth == {"transl_z": -50.0, "min_z": -10.0, "max_z": 10.0, "zdim": 11}


def test_readers_return_none_without_attrs(mesc_path):
    key = "MSession_0/MUnit_2"
    assert roi_outlines_um(mesc_path, key) is None
    assert viewport_geometry(mesc_path, key) is None
    assert zstack_depth_info(mesc_path, key) is None
    # a unit with a viewport but no MinZ/MaxZ/ZDim is not a usable stack
    assert zstack_depth_info(mesc_path, "MSession_0/MUnit_1") is None


def test_line_positions_measure_each_roi(mesc_path):
    rows = line_positions(mesc_path, "MSession_0/MUnit_1", [20, 24, 10, 6])
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["start_um"] == [110.0, 210.0, -54.0]
    assert rows[0]["end_um"] == [130.0, 210.0, -54.0]
    assert np.allclose([r["length_um"] for r in rows], [20, 24, np.hypot(30, 10), 6])
    assert np.allclose(
        [r["sample_um"] for r in rows], [1.0, 1.0, np.hypot(30, 10) / 10, 1.0]
    )
    assert all(
        r["sample_um"] is None
        for r in line_positions(mesc_path, "MSession_0/MUnit_1")
    )


def test_a_patch_measures_its_first_side(mesc_path):
    patches = roi_outlines_um(mesc_path, "MSession_0/MUnit_3")
    assert len(patches) == 2 and patches[0].shape == (3, 4)
    rows = line_positions(mesc_path, "MSession_0/MUnit_3", [20, 20])
    assert np.allclose([r["length_um"] for r in rows], [20.0, 20.0])
    assert np.allclose([r["sample_um"] for r in rows], [1.0, 1.0])
    vp = viewport_geometry(mesc_path, "MSession_0/MUnit_0")
    corners = um_to_pixels(patches[0][:2].T, vp, 64, 80)
    assert np.allclose(corners, [[20, 20], [60, 20], [60, 60], [20, 60]])


def test_um_to_pixels(mesc_path, lines):
    vp = viewport_geometry(mesc_path, "MSession_0/MUnit_0")
    ny, nx = 64, 80  # 0.5 um/px in both axes
    px = um_to_pixels(lines[0][:2].T, vp, ny, nx)
    assert np.allclose(px, [[20, 20], [60, 20]])
    # non-square pixels: 40 um over 20 px wide, 32 um over 64 px tall
    px = um_to_pixels(lines[0][:2].T, vp, 64, 20)
    assert np.allclose(px, [[5, 20], [15, 20]])


def test_a_rotated_viewport_turns_points_back_onto_its_pixels():
    """A 45 degree rig: ``geomTransRot`` is ``[x, y, z, w]``, and a point one
    pixel along the image's x axis sits at ``transl + R(+45) [1, 0]``.
    """
    half = np.pi / 8
    with_rotation = {
        "transl": (10.0, 20.0, 0.0),
        "width": 8.0,
        "height": 8.0,
        "angle": 2 * half,
    }
    along_x = np.array([[10.0 + np.cos(2 * half), 20.0 + np.sin(2 * half)]])
    assert np.allclose(um_to_pixels(along_x, with_rotation, 8, 8), [[1.0, 0.0]])
    along_y = np.array([[10.0 - np.sin(2 * half), 20.0 + np.cos(2 * half)]])
    assert np.allclose(um_to_pixels(along_y, with_rotation, 8, 8), [[0.0, 1.0]])


def test_the_viewport_reads_its_rotation(mesc_path, tmp_path):
    path = tmp_path / "turned.mesc"
    shutil.copy(mesc_path, path)
    with h5py.File(path, "a") as f:
        unit = f["MSession_0/MUnit_0"]
        viewports = json.loads(unit.attrs["ReferenceViewportJSON"])
        viewports["viewports"][0]["geomTransRot"] = [0.0, 0.0, 0.3827, 0.9239]
        unit.attrs["ReferenceViewportJSON"] = json.dumps(viewports)
    assert viewport_geometry(path, "MSession_0/MUnit_0")["angle"] == pytest.approx(
        np.pi / 4, abs=1e-3
    )


def test_three_point_lines_measure_start_to_end(mesc_path, tmp_path):
    """MESc 4.6.2 writes start, midpoint, end; the length is the whole segment."""
    path = tmp_path / "midpoint.mesc"
    shutil.copy(mesc_path, path)
    with h5py.File(path, "a") as f:
        unit = f["MSession_0/MUnit_1"]
        maps = json.loads(unit.attrs["CoordinateMapJSON"])
        maps["maps"][0]["driftEndPoints"][0] = [
            [110, 120, 130],
            [210, 210, 210],
            [-54, -54, -54],
        ]
        unit.attrs["CoordinateMapJSON"] = json.dumps(maps)
    (row, *_) = line_positions(path, "MSession_0/MUnit_1", [20, 24, 10, 6])
    assert row["length_um"] == 20.0 and row["sample_um"] == 1.0


def test_snapshot_overlay_is_every_line_drawn_on_it(mesc_path):
    recs = image_overlays(mesc_path, "MSession_0/MUnit_1", "MSession_0/MUnit_4")
    assert [r["roi"] for r in recs] == [0, 1, 2, 3]
    assert all(r["kind"] == "line" and r["slice"] is None for r in recs)
    assert np.allclose(recs[0]["pixels"], [[20, 20], [60, 20]])
    assert recs[0]["color"] == (1.0, 0.0, 0.0) and recs[3]["color"] == (1.0, 1.0, 1.0)
    # an image without a position, or a unit without ROIs, carries nothing
    assert image_overlays(mesc_path, "MSession_0/MUnit_1", "MSession_0/MUnit_2") == []
    assert image_overlays(mesc_path, "MSession_0/MUnit_2", "MSession_0/MUnit_4") == []


def test_zstack_overlay_places_lines_and_patches_on_slices(mesc_path):
    """Each ROI on the slice nearest its depth. The line and the patch scanned
    20 um below the stack are left out: the stack holds no picture of them, and
    drawing them on slice 0 would put an outline on tissue they never touched.
    """
    recs = image_overlays(mesc_path, "MSession_0/MUnit_1", "MSession_0/MUnit_0")
    assert [r["roi"] for r in recs] == [0, 1, 2]
    assert [r["slice"] for r in recs] == [3, 7, 3]
    (patch,) = image_overlays(mesc_path, "MSession_0/MUnit_3", "MSession_0/MUnit_0")
    assert patch["kind"] == "patch" and patch["color"] is None
    assert np.allclose(
        patch["pixels"], [[20, 20], [60, 20], [60, 60], [20, 60], [20, 20]]
    )


def test_a_shift_moves_rois_in_and_out_of_a_stack(mesc_path):
    """A measured shift decides the slice and the pixels; a line moved past
    the stack's depth or field is left out, one moved into it is kept.
    """
    recs = image_overlays(
        mesc_path,
        "MSession_0/MUnit_1",
        "MSession_0/MUnit_0",
        shift_um=(2.0, 1.0, 4.0),
    )
    assert [r["roi"] for r in recs] == [0, 1, 2]
    assert [r["slice"] for r in recs] == [5, 9, 5]
    assert np.allclose(recs[0]["pixels"], [[24, 22], [64, 22]])
    recs = image_overlays(
        mesc_path, "MSession_0/MUnit_1", "MSession_0/MUnit_0", shift_um=(0, 0, 16.0)
    )
    assert [r["roi"] for r in recs] == [3]
    assert (
        image_overlays(
            mesc_path, "MSession_0/MUnit_1", "MSession_0/MUnit_0", shift_um=(500, 0, 0)
        )
        == []
    )


def test_list_mesc_units_reports_outlines_and_links(mesc_path):
    from mbo_utilities.arrays.mesc import list_mesc_units

    units = {u["munit"]: u for u in list_mesc_units(mesc_path)}
    assert (units["MUnit_1"]["outline_kind"], units["MUnit_1"]["n_outlines"]) == (
        "line",
        4,
    )
    assert (units["MUnit_3"]["outline_kind"], units["MUnit_3"]["n_outlines"]) == (
        "patch",
        2,
    )
    assert (units["MUnit_0"]["outline_kind"], units["MUnit_0"]["n_outlines"]) == (
        None,
        0,
    )
    assert units["MUnit_1"]["background_unit"] == "MSession_0/MUnit_4"
    assert units["MUnit_4"]["scans"] == ["MSession_0/MUnit_1"]
    assert units["MUnit_0"]["scans"] == [] and units["MUnit_1"]["scans"] == []
    assert all(u["rtmc_of"] == [] for u in units.values())


def test_a_ribbon_outline_is_a_closed_patch(mesc_path, tmp_path):
    """A ribbon's contour is one long side then the other walked back, many
    points each; it closes like a four-corner patch, so both short edges draw,
    and its length runs along the first side.
    """
    path = tmp_path / "ribbon.mesc"
    shutil.copy(mesc_path, path)
    xs = np.linspace(110, 130, 5)
    ribbon = [
        [*xs, *xs[::-1]],
        [*[210.0] * 5, *[220.0] * 5],
        [-46.0] * 10,
    ]
    with h5py.File(path, "a") as f:
        unit = f["MSession_0/MUnit_3"]
        maps = json.loads(unit.attrs["CoordinateMapJSON"])
        maps["maps"][0]["contours"][0] = ribbon
        unit.attrs["CoordinateMapJSON"] = json.dumps(maps)
    assert line_positions(path, "MSession_0/MUnit_3")[0]["length_um"] == 20.0
    (rec, _) = image_overlays(path, "MSession_0/MUnit_3", "MSession_0/MUnit_4")
    assert rec["kind"] == "patch"
    assert rec["pixels"].shape == (11, 2)
    assert np.allclose(rec["pixels"][0], rec["pixels"][-1])
    assert np.allclose(rec["pixels"][-2:], [[20, 40], [20, 20]])


def test_a_reference_units_placeholder_viewport_is_no_position(tmp_path):
    """MEScan stamps an RTMC reference unit with a 1 um square at the origin;
    that is not where anything was scanned, so it has no viewport.
    """
    path = tmp_path / "ref.mesc"
    with h5py.File(path, "w") as f:
        u = f.create_group("MSession_1").create_group("MUnit_0")
        u.attrs.update(
            {
                "MethodType": 1,
                "VecChannelsSize": 1,
                "TStepInMs": 1.0,
                "MeasurementDatePosix": 0,
                "ImageRoleDebugString": "motionCorrection",
            }
        )
        u.attrs["ReferenceViewportJSON"] = json.dumps(
            {"viewports": [{"geomTransTransl": [0, 0, 0], "width": 1, "height": 1}]}
        )
        u.create_dataset("Channel_0", data=np.zeros((3, 20, 20), np.uint16))
    assert viewport_geometry(path, "MSession_1/MUnit_0") is None
    assert image_overlays(path, "MSession_1/MUnit_0", "MSession_1/MUnit_0") == []
