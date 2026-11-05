"""The reference image of a MESc scan: the picture its lines were drawn on
and the Z-stack around it, max projected, with the lines drawn.

Fixture: a textured Z-stack (11 slices, 2 um apart) whose field holds a
four-line scan (6 frames; lines on slices 3, 7 and 3, and a fourth scanned
10 um below the stack) drawn on a picture of slice 3, so the picture registers
at no shift.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback

import h5py
import numpy as np
import pytest
from mbo_utilities.analysis.mesc_placement import StackPlacement
from scipy import ndimage

TRANSL = (100.0, 200.0, -50.0)
SNAP_TRANSL = (100.0, 200.0, -54.0)
LINES = [
    [[110, 130], [210, 210], [-54, -54]],
    [[120, 120], [204, 228], [-46, -46]],
    [[105, 135], [220, 230], [-53.9, -53.9]],
    [[112, 118], [212, 212], [-70, -70]],
]


def tissue(seed: int) -> np.ndarray:
    """``(11, 64, 80)`` uint16 blobs a few pixels wide, different on every slice."""
    rng = np.random.default_rng(seed)
    blobs = ndimage.gaussian_filter(rng.normal(size=(11, 64, 80)), (1, 2, 2))
    return ((blobs - blobs.min()) / np.ptp(blobs) * 1000).astype(np.uint16)


STACK = tissue(seed=1)


@pytest.fixture(scope="module")
def mesc_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("reference") / "reference.mesc"
    rng = np.random.default_rng(0)
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
        z.create_dataset("Channel_0", data=STACK)

        ls = s.create_group("MUnit_1")
        ls.attrs.update(
            {
                "MethodType": 6,
                "VecChannelsSize": 1,
                "TStepInMs": 2.0,
                "MeasurementDatePosix": 1,
                "Comment": "linescan",
                "BackgroundImagePath": "/MSession_0/MUnit_4",
            }
        )
        boxes = [
            {"lowerLeftFramePix": [2 * i + 1, 1], "upperRightFramePix": [2 * i + 2, 1]}
            for i in range(4)
        ]
        ls.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": boxes, "driftEndPoints": LINES}]}
        )
        # packed: one raw frame holding the 6 timepoints of every line as rows
        ls.create_dataset(
            "Channel_0", data=rng.integers(0, 1000, (1, 6, 8)).astype(np.uint16)
        )

        # the picture: three frames, the brightest pixel value in the last one
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
        picture = np.repeat(STACK[3][None], 3, axis=0)
        picture[2, 10, 10] = 2000
        snap.create_dataset("Channel_0", data=picture)
    return path


def open_unit(path, key):
    from mbo_utilities.arrays.mesc import MescArray

    return MescArray(path, unit=key)


def test_reference_images_of_a_line_scan(mesc_path):
    """The picture the lines were drawn on carries every line whatever its
    depth; the Z-stack carries only the lines scanned inside it, and is
    projected over the slices they sit on, not over the whole stack.
    """
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.mesc_reference import picture_image, stack_images

    mesc = MescArray(mesc_path, unit=1)
    try:
        picture = picture_image(mesc, open_unit, 0)
        (stack,), refused = stack_images(mesc, open_unit, 0)
    finally:
        mesc.close()
    assert picture.key == "MUnit_4 picture" and picture.placement is None
    assert stack.key == "MUnit_0 slices 4-8" and refused == []
    assert stack.placement.placed
    np.testing.assert_allclose(stack.placement.shift_um, 0.0, atol=0.5)
    # the picture is the max over its frames
    assert picture.image.shape == (64, 80) and picture.image.dtype == np.float32
    assert float(picture.image[0, 0]) == STACK[3, 0, 0]
    assert float(picture.image[10, 10]) == 2000.0
    assert [r["roi"] for r in picture.records] == [0, 1, 2, 3]
    assert all(r["pixels"].shape == (2, 2) for r in picture.records)
    np.testing.assert_allclose(stack.image, STACK[3:8].max(axis=0))
    assert [r["roi"] for r in stack.records] == [0, 1, 2]
    assert [r["slice"] for r in stack.records] == [3, 7, 3]


def test_a_stack_file_beside_the_scan_is_measured_too(mesc_path, tmp_path):
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.mesc_reference import stack_images

    path = tmp_path / "reference.mesc"
    shutil.copy(mesc_path, path)
    shutil.copy(mesc_path, tmp_path / "reference_z.mesc")
    mesc = MescArray(path, unit=1)
    try:
        images, _ = stack_images(mesc, open_unit, 0)
    finally:
        mesc.close()
    assert [im.key for im in images] == [
        "MUnit_0 slices 4-8",
        "reference_z MUnit_0 slices 4-8",
    ]
    assert images[1].placement.stack_path == str(tmp_path / "reference_z.mesc")


def test_a_stack_of_other_tissue_is_refused_and_carries_nothing(mesc_path, tmp_path):
    """Its stored position is right on top of the lines, and that is no evidence."""
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.mesc_reference import stack_images

    path = tmp_path / "other.mesc"
    shutil.copy(mesc_path, path)
    with h5py.File(path, "a") as f:
        f["MSession_0/MUnit_0/Channel_0"][...] = tissue(seed=2)
    mesc = MescArray(path, unit=1)
    try:
        images, (refused,) = stack_images(mesc, open_unit, 0)
    finally:
        mesc.close()
    assert images == [] and not refused.placed


def test_a_sampled_projection_stays_a_max(mesc_path, monkeypatch):
    """A long picture is read from evenly spaced frames: the projection is
    still the max of what was read, at the picture's shape.
    """
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui import mesc_reference
    from mbo_utilities.gui.mesc_reference import picture_image, stack_images

    # room for one frame of the picture at a time: frames 0 and 2 are read
    monkeypatch.setattr(mesc_reference, "MAX_ELEMENTS", 64 * 80 * 2)
    mesc = MescArray(mesc_path, unit=1)
    try:
        picture = picture_image(mesc, open_unit, 0)
        (stack,), _ = stack_images(mesc, open_unit, 0)
    finally:
        mesc.close()
    assert picture.image.shape == (64, 80)
    assert float(picture.image[10, 10]) == 2000.0
    np.testing.assert_allclose(stack.image, STACK[3:8].max(axis=0))


def test_the_reader_places_its_lines(mesc_path):
    """``MescArray.line_positions``: each line's stored ends, read once; None
    for a unit without ROIs.
    """
    from mbo_utilities.arrays.mesc import MescArray

    scan = MescArray(mesc_path, unit=1)
    stack = MescArray(mesc_path, unit=0)
    try:
        positions = scan.line_positions
        assert [p["index"] for p in positions] == [0, 1, 2, 3]
        assert positions[1]["start_um"] == [120.0, 204.0, -46.0]
        assert scan.line_positions is positions
        assert stack.line_positions is None
    finally:
        scan.close()
        stack.close()


def test_trace_rows_on_a_line_scan_know_their_line(mesc_path):
    """On a line-scan viewer every row finds its line through the reader: a
    results row by its line, a drawn ROI's quick trace by the ROI slider it
    was read on; the table shows no depth for it.
    """
    pytest.importorskip("fastplotlib.widgets.nd_widget")
    from mbo_utilities.annotation import RoiTrace
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui.manual_roi import ManualRoiWidget
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    arr = MescArray(mesc_path, unit=1)
    iw = MboNDViewer(
        data=display_wrap(arr),
        slider_dim_names=arr.slider_dim_labels,
        figure_kwargs={"size": (640, 480)},
    )
    iw.show()
    widget = ManualRoiWidget(iw, fpath=None, auto_trace=False)
    try:
        assert [p["index"] for p in widget.line_positions] == [0, 1, 2, 3]
        results_row = widget.traces.add(
            RoiTrace(
                uid=0,
                member=1,
                source="results",
                engine="vnoiser",
                label="roi1",
                F=np.zeros(6, np.float32),
                z=1,
                c=0,
                extra={"line": 1},
            )
        )
        drawn_row = widget.traces.add(
            RoiTrace(uid=5, F=np.zeros(6, np.float32), z=2, c=0)
        )
        assert widget._line_position(results_row)["start_um"] == [120.0, 204.0, -46.0]
        assert widget._line_position(drawn_row)["start_um"] == [105.0, 220.0, -53.9]
        assert widget._trace_cells(results_row.key)[1:4] == ("2", "0", "vnoiser")
        assert len(widget._trace_cells(drawn_row.key)) == 5
        # a row's own record wins over the recording's
        results_row.extra["length_um"] = 2.5
        assert widget._line_position(results_row)["length_um"] == 2.5
    finally:
        widget.close()
        iw.close()
        arr.close()


def test_a_stack_or_picture_has_no_reference_image(mesc_path):
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.mesc_reference import picture_image, stack_images

    stack = MescArray(mesc_path, unit=0)
    snap = MescArray(mesc_path, unit="MUnit_4")
    try:
        for unit in (stack, snap):
            assert picture_image(unit, open_unit, 0) is None
            assert stack_images(unit, open_unit, 0) == ([], [])
    finally:
        stack.close()
        snap.close()


def test_a_placement_says_what_the_recording_found():
    from mbo_utilities.gui.mesc_reference import placement_summary

    placed = StackPlacement("z.mesc", "MUnit_3", (1.0, -2.0, 3.0), 0.8, 600.0)
    assert "+1.0, -2.0, +3.0 um" in placement_summary(placed)
    assert "10 min before" in placement_summary(placed)
    assert "cannot check" in placement_summary(placed)
    checked = StackPlacement(
        "z.mesc", "MUnit_3", (0.0, 0.0, 0.0), 0.8, -60.0, 0.7, 0.01
    )
    assert "1 min after" in placement_summary(checked)
    assert "recording confirms it" in placement_summary(checked)
    failed = StackPlacement("z.mesc", "MUnit_3", (0.0, 0.0, 0.0), 0.8, 0.0, 0.1, 0.4)
    assert "does not confirm" in placement_summary(failed)


class ViewerHost:
    """A viewer-hosting parent: what the reference view reads."""

    def __init__(self, iw):
        self.image_widget = iw


def test_reference_popup_draws_and_highlights_the_sliders_roi(mesc_path):
    """On a real figure: the popup builds the images for the shown line scan,
    draws without raising, and draws every line solid with the slider's ROI
    thick, on the picture and on the stack alike.
    """
    pytest.importorskip("fastplotlib.widgets.nd_widget")
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui._top_strip import TopStrip
    from mbo_utilities.gui.mesc_reference import (
        HALO_THICKNESS,
        ON_ALPHA,
        ON_THICKNESS,
        SELECTED_THICKNESS,
        ReferenceView,
        roi_slider,
    )
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    arr = MescArray(mesc_path, unit=1)
    iw = MboNDViewer(
        data=display_wrap(arr),
        slider_dim_names=arr.slider_dim_labels,
        figure_kwargs={"size": (640, 480)},
    )
    iw.show()
    strip = TopStrip(iw.figure)
    host = ViewerHost(iw)
    view = ReferenceView(host)
    errors = []

    def guarded(*_args):
        try:
            strip.update()
            view.draw()
        except Exception:
            errors.append(traceback.format_exc())

    try:
        assert view.open(arr, open_unit)
        assert view.is_open and view.unit == "MUnit_1"
        assert view.viewer.current_key == "MUnit_4 picture"
        assert list(view.images) == ["MUnit_4 picture"]
        strip._update_calls[:] = [guarded]
        deadline = time.monotonic() + 60
        while view.placing and time.monotonic() < deadline:
            iw.figure.canvas.draw()
        for _ in range(3):
            iw.figure.canvas.draw()
        assert not errors, errors[0]
        # the stack landed behind the picture without moving the popup off it
        assert list(view.images) == ["MUnit_4 picture", "MUnit_0 slices 4-8"]
        assert view.viewer.current_key == "MUnit_4 picture"
        caption = view.caption(view.images["MUnit_0 slices 4-8"])
        assert "ROI 1 slice 4" in caption and "3 of 4 lines" in caption
        lines = view.contours("MUnit_4 picture")
        # the others dimmed, the selected one last over a white halo
        assert [t for _p, _c, t in lines] == [
            ON_THICKNESS,
            ON_THICKNESS,
            ON_THICKNESS,
            HALO_THICKNESS,
            SELECTED_THICKNESS,
        ]
        assert [c[3] for _p, c, _t in lines] == [ON_ALPHA, ON_ALPHA, ON_ALPHA, 1.0, 1.0]
        assert lines[3][1] == (1.0, 1.0, 1.0, 1.0)
        assert all(p.shape == (2, 2) for p, _c, _t in lines)
        # the stack draws the three lines scanned inside it, not the fourth
        assert [t for _p, _c, t in view.contours("MUnit_0 slices 4-8")] == [
            ON_THICKNESS,
            ON_THICKNESS,
            HALO_THICKNESS,
            SELECTED_THICKNESS,
        ]
        # clicking a line in the popup selects its ROI: the slider (found by
        # position on a line scan) moves and the highlight follows
        pts = np.asarray(view.images["MUnit_4 picture"].records[2]["pixels"], float)
        # a quarter of the way along: the midpoint sits on the vertical line 2 as well
        near = pts[0] + 0.25 * (pts[1] - pts[0])
        assert view.pick("MUnit_4 picture", float(near[1]), float(near[0])) == 2
        assert int(iw.indices[roi_slider(iw.dim_names)]) == 2
        shown = view.contours("MUnit_4 picture")
        assert [t for _p, _c, t in shown] == [
            ON_THICKNESS,
            ON_THICKNESS,
            ON_THICKNESS,
            HALO_THICKNESS,
            SELECTED_THICKNESS,
        ]
        assert np.array_equal(shown[-1][0], pts[:, ::-1])
        assert view.pick("MUnit_4 picture", -40.0, -40.0) is None
        assert view.pick("nowhere", 0.0, 0.0) is None
        assert view.contours("nowhere") == []
        # the same unit again is not rebuilt, and opens on the picture
        built = view._built
        assert view.open(arr, open_unit)
        assert view._built is built and view.viewer.current_key == "MUnit_4 picture"
        # an answer for a unit no longer shown is dropped
        view._place(("old.mesc", "MUnit_9", 0, ()), arr, open_unit, 0)
        assert view._placed is None
        view.close()
        assert not view.is_open
    finally:
        view.cleanup()
        iw.close()
        arr.close()


def test_a_replaced_image_keeps_its_texture_until_the_frame_is_drawn(mesc_path):
    """``set_images`` runs inside a frame whose draw list may already hold the
    old texture: freeing it there is a ``KeyError`` in the imgui backend.
    """
    pytest.importorskip("fastplotlib.widgets.nd_widget")
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui._top_strip import TopStrip
    from mbo_utilities.gui.imgui.summary import SummaryImageViewer
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    arr = MescArray(mesc_path, unit=1)
    iw = MboNDViewer(
        data=display_wrap(arr),
        slider_dim_names=arr.slider_dim_labels,
        figure_kwargs={"size": (640, 480)},
    )
    iw.show()
    strip = TopStrip(iw.figure)
    viewer = SummaryImageViewer(iw.figure, title="test", window_id="test_retired")
    image = np.zeros((8, 8), np.float32)
    try:
        viewer.set_images({"a": image}, selected="a")
        viewer.open()
        strip._update_calls[:] = [viewer.draw]
        iw.figure.canvas.draw()
        backend = iw.figure.imgui_renderer.backend
        drawn = set(backend._texture_views)
        viewer.set_images({"b": image}, selected="b")
        assert drawn <= set(backend._texture_views)
        iw.figure.canvas.draw()
        assert not drawn <= set(backend._texture_views)
    finally:
        viewer.cleanup()
        iw.close()
        arr.close()
