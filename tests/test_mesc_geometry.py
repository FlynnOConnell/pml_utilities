"""``arrays.mesc_geometry``: placing AOD line-scan lines inside a Z-stack.

Fixture: a Z-stack whose FOV corner sits at (100, 200, -50) um, 40 x 32 um,
11 slices from z = -10 to +10 um relative to that origin (2 um steps), and a
line-scan unit with four lines at different depths: two on one slice, one
alone, one scanned below the stack.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from mbo_utilities.arrays.mesc_geometry import (
    image_overlays,
    linescan_endpoints_um,
    neighbour_slices,
    roi_outlines_um,
    roi_placements,
    roi_slice_indices,
    slice_depths_um,
    slices_with_rois,
    um_to_pixels,
    viewport_geometry,
    zstack_contents,
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
    [[110, 130, 130, 110], [210, 210, 230, 230], [-46, -46, -46, -46]],  # z +4 -> slice 7
    [[100, 120, 120, 100], [200, 200, 220, 220], [-70, -70, -70, -70]],  # below the stack
]


@pytest.fixture(scope="module")
def mesc_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("geom") / "geom.mesc"
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")
        z = s.create_group("MUnit_0")
        z.attrs.update(
            {"MethodType": 2, "VecChannelsSize": 1, "TStepInMs": 1.0,
             "MeasurementDatePosix": 0, "Comment": "zstack",
             "MinZ": -10.0, "MaxZ": 10.0, "ZDim": 11}
        )
        z.attrs["ReferenceViewportJSON"] = json.dumps(
            {"viewports": [{"geomTransTransl": list(TRANSL), "width": 40.0, "height": 32.0}]}
        )
        z.create_dataset("Channel_0", data=np.zeros((11, 64, 80), np.uint16))

        ls = s.create_group("MUnit_1")
        ls.attrs.update(
            {"MethodType": 6, "VecChannelsSize": 1, "TStepInMs": 2.0,
             "MeasurementDatePosix": 1, "Comment": "linescan"}
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
        bare.attrs.update({"MethodType": 1, "VecChannelsSize": 1, "TStepInMs": 1.0,
                           "MeasurementDatePosix": 2})
        bare.create_dataset("Channel_0", data=np.zeros((2, 4, 4), np.uint16))

        chess = s.create_group("MUnit_3")
        chess.attrs.update({"MethodType": 8, "VecChannelsSize": 1, "TStepInMs": 5.0,
                            "MeasurementDatePosix": 3, "Comment": "chessboard"})
        chess.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": [], "contours": PATCHES}]}
        )
        # MESc 4.6.2 layout: a scanner list and a 0-based index beside it
        chess.attrs["MultiROIProtocolJSON"] = json.dumps(
            {
                "protocol": {"mainPatternIndex": 1, "scanners": [{"name": "AO1"}]},
                "scanPatterns": {
                    "patterns": [
                        {"centerPoints": [[0.0], [0.0], [0.0]], "pixelSizeX": 1.0, "rotation": [0, 0, 0, 1]},
                        {
                            "centerPoints": [[120.0, 110.0], [220.0, 210.0], [-46.0, -70.0]],
                            "pixelSizeX": 1.0,
                            "rotation": [0, 0, 0, 1],
                        },
                    ]
                },
            }
        )
        chess.create_dataset("Channel_0", data=np.zeros((1, 20, 40), np.uint16))

        snap = s.create_group("MUnit_4")
        snap.attrs.update({"MethodType": 1, "VecChannelsSize": 1, "TStepInMs": 1.0,
                           "MeasurementDatePosix": 0, "ImageRoleDebugString": "background"})
        snap.attrs["ReferenceViewportJSON"] = json.dumps(
            {"viewports": [{"geomTransTransl": list(SNAP_TRANSL), "width": 40.0, "height": 32.0}]}
        )
        snap.create_dataset("Channel_0", data=np.zeros((1, 64, 80), np.uint16))
    return path


@pytest.fixture(scope="module")
def depth(mesc_path):
    return zstack_depth_info(mesc_path, "MSession_0/MUnit_0")


@pytest.fixture(scope="module")
def lines(mesc_path):
    return linescan_endpoints_um(mesc_path, "MSession_0/MUnit_1")


def test_readers(mesc_path, depth, lines):
    assert len(lines) == 4 and lines[0].shape == (3, 2)
    vp = viewport_geometry(mesc_path, "MSession_0/MUnit_0")
    assert vp == {"transl": TRANSL, "width": 40.0, "height": 32.0}
    assert depth == {"transl_z": -50.0, "min_z": -10.0, "max_z": 10.0, "zdim": 11}


def test_readers_return_none_without_attrs(mesc_path):
    key = "MSession_0/MUnit_2"
    assert linescan_endpoints_um(mesc_path, key) is None
    assert viewport_geometry(mesc_path, key) is None
    assert zstack_depth_info(mesc_path, key) is None
    # a unit with a viewport but no MinZ/MaxZ/ZDim is not a usable stack
    assert zstack_depth_info(mesc_path, "MSession_0/MUnit_1") is None


def test_slice_depths(depth):
    assert np.allclose(slice_depths_um(depth), np.arange(-10, 11, 2))


def test_placements(depth, lines):
    p = roi_placements(lines, depth, sample_counts=[20, 24, 10, 6])
    assert [q["slice"] for q in p] == [3, 7, 3, 0]
    assert [q["in_range"] for q in p] == [True, True, True, False]
    assert np.allclose([q["z_um"] for q in p], [-4.0, 4.0, -3.9, -20.0])
    assert np.allclose([q["slice_z_um"] for q in p], [-4.0, 4.0, -4.0, -10.0])
    # 'off' is the true distance from the slice the line is drawn on
    assert np.allclose([q["dz_um"] for q in p], [0.0, 0.0, 0.1, -10.0])
    assert np.allclose([q["length_um"] for q in p], [20.0, 24.0, np.hypot(30, 10), 6.0])
    assert np.allclose([q["sample_um"] for q in p], [1.0, 1.0, np.hypot(30, 10) / 10, 1.0])
    assert not any(q["tilted"] for q in p)
    assert np.array_equal(roi_slice_indices(lines, depth), [3, 7, 3, 0])


def test_placements_without_sample_counts(depth, lines):
    assert all(q["sample_um"] is None for q in roi_placements(lines, depth))


def test_tilted_line_uses_mean_depth(depth):
    seg = np.array([[0, 1], [0, 0], [-56, -52]], dtype=float)  # z -6..-2 -> mean -4
    (p,) = roi_placements([seg], depth)
    assert p["tilted"] and p["slice"] == 3 and p["z_um"] == -4.0


def test_slices_with_rois_and_neighbours(depth, lines):
    occ = slices_with_rois(roi_placements(lines, depth))
    assert occ == {0: [3], 3: [0, 2], 7: [1]}
    assert neighbour_slices(3, occ) == (0, 7)
    assert neighbour_slices(0, occ) == (None, 3)
    assert neighbour_slices(7, occ) == (3, None)
    assert neighbour_slices(5, occ) == (3, 7)  # a slice with no lines


def test_chessboard_patches_place_like_lines(mesc_path, depth):
    patches = roi_outlines_um(mesc_path, "MSession_0/MUnit_3")
    assert len(patches) == 2 and patches[0].shape == (3, 4)
    # the line-scan name reads the same attribute and gives the same outlines
    assert all(np.array_equal(a, b) for a, b in zip(linescan_endpoints_um(mesc_path, "MSession_0/MUnit_3"), patches))
    p = roi_placements(patches, depth, sample_counts=[20, 20])
    assert [q["slice"] for q in p] == [7, 0]
    assert [q["in_range"] for q in p] == [True, False]
    assert np.allclose([q["z_um"] for q in p], [4.0, -20.0])
    assert np.allclose([q["length_um"] for q in p], [20.0, 20.0])
    assert np.allclose([q["sample_um"] for q in p], [1.0, 1.0])
    assert not any(q["tilted"] for q in p)
    vp = viewport_geometry(mesc_path, "MSession_0/MUnit_0")
    corners = um_to_pixels(patches[0][:2].T, vp, 64, 80)
    assert np.allclose(corners, [[20, 20], [60, 20], [60, 60], [20, 60]])


def test_um_to_pixels(mesc_path, lines):
    vp = viewport_geometry(mesc_path, "MSession_0/MUnit_0")
    ny, nx = 64, 80  # 0.5 um/px in both axes
    px = um_to_pixels(lines[0][:2].T, vp, ny, nx)
    assert np.allclose(px, [[20, 20], [60, 20]])
    flipped = um_to_pixels(lines[0][:2].T, vp, ny, nx, flip_y=True)
    assert np.allclose(flipped, [[20, 44], [60, 44]])
    # non-square pixels: 40 um over 20 px wide, 32 um over 64 px tall
    px = um_to_pixels(lines[0][:2].T, vp, 64, 20)
    assert np.allclose(px, [[5, 20], [15, 20]])


def test_three_point_lines_measure_start_to_end(depth):
    # MESc 4.6.2 writes start, midpoint, end; the length is the whole segment
    seg = np.array([[110, 120, 130], [210, 210, 210], [-54, -54, -54]], dtype=float)
    (p,) = roi_placements([seg], depth, sample_counts=[20])
    assert p["length_um"] == 20.0 and p["sample_um"] == 1.0 and p["slice"] == 3


def test_snapshot_overlay_is_every_line_drawn_on_it(mesc_path):
    recs = image_overlays(mesc_path, "MSession_0/MUnit_4")
    assert [r["unit"] for r in recs] == ["MSession_0/MUnit_1"] * 4
    assert [r["roi"] for r in recs] == [0, 1, 2, 3]
    assert all(r["kind"] == "line" and r["slice"] is None for r in recs)
    # the snapshot plane is z -54: lines 0 and 2 are on it, 1 and 3 are not
    assert [r["on_plane"] for r in recs] == [True, False, True, False]
    assert np.allclose([r["dz_um"] for r in recs], [0.0, 8.0, 0.1, -16.0])
    assert np.allclose(recs[0]["pixels"], [[20, 20], [60, 20]])
    assert recs[0]["color"] == (1.0, 0.0, 0.0) and recs[3]["color"] == (1.0, 1.0, 1.0)
    # the snapshot's own field and other images stay out
    assert image_overlays(mesc_path, "MSession_0/MUnit_2") == []


def test_zstack_overlay_places_lines_and_patches_on_slices(mesc_path):
    recs = image_overlays(mesc_path, "MSession_0/MUnit_0")
    assert [(r["munit"], r["roi"]) for r in recs] == [
        ("MUnit_1", 0), ("MUnit_1", 1), ("MUnit_1", 2), ("MUnit_1", 3), ("MUnit_3", 0), ("MUnit_3", 1),
    ]
    assert [r["slice"] for r in recs] == [3, 7, 3, 0, 7, 0]
    assert [r["on_plane"] for r in recs] == [True, True, True, False, True, False]
    assert np.allclose([r["dz_um"] for r in recs], [0.0, 0.0, 0.1, -10.0, 0.0, -10.0])
    patch = recs[4]
    assert patch["kind"] == "patch" and patch["color"] is None
    assert np.allclose(patch["pixels"], [[20, 20], [60, 20], [60, 60], [20, 60], [20, 20]])


def test_list_mesc_units_reports_outlines_and_links(mesc_path):
    from mbo_utilities.arrays.mesc import list_mesc_units

    units = {u["munit"]: u for u in list_mesc_units(mesc_path)}
    assert (units["MUnit_1"]["outline_kind"], units["MUnit_1"]["n_outlines"]) == ("line", 4)
    assert (units["MUnit_3"]["outline_kind"], units["MUnit_3"]["n_outlines"]) == ("patch", 2)
    assert (units["MUnit_0"]["outline_kind"], units["MUnit_0"]["n_outlines"]) == (None, 0)
    assert units["MUnit_1"]["background_unit"] == "MSession_0/MUnit_4"
    assert units["MUnit_4"]["scans"] == ["MSession_0/MUnit_1"]
    assert units["MUnit_0"]["scans"] == [] and units["MUnit_1"]["scans"] == []
    assert all(u["rtmc_of"] == [] for u in units.values())


def test_zstack_contents_lists_the_scans_inside_each_stack(mesc_path):
    assert zstack_contents(mesc_path) == {
        "MSession_0/MUnit_0": ["MSession_0/MUnit_1", "MSession_0/MUnit_3"]
    }


def test_overlay_skips_a_unit_whose_outlines_do_not_pair_with_its_rois(mesc_path):
    from mbo_utilities.arrays.mesc import list_mesc_units

    units = list_mesc_units(mesc_path)
    for u in units:
        if u["munit"] == "MUnit_1":
            u["nrois"] = 3
    recs = image_overlays(mesc_path, "MSession_0/MUnit_0", units)
    assert {r["munit"] for r in recs} == {"MUnit_3"}
