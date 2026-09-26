"""The reference image of a MESc scan: the picture its lines were drawn on
and the Z-stack around it, max projected, with the lines drawn.

Fixture: a Z-stack (11 slices, 2 um apart, slice k filled with k) whose
field holds a four-line scan (6 frames; lines on slices 3, 7 and 3, and a
fourth scanned 10 um below the stack) drawn on a picture focused 4 um under
the stack's origin.
"""

from __future__ import annotations

import json
import traceback
from functools import partial

import h5py
import numpy as np
import pytest

TRANSL = (100.0, 200.0, -50.0)
SNAP_TRANSL = (100.0, 200.0, -54.0)
LINES = [
    [[110, 130], [210, 210], [-54, -54]],
    [[120, 120], [204, 228], [-46, -46]],
    [[105, 135], [220, 230], [-53.9, -53.9]],
    [[112, 118], [212, 212], [-70, -70]],
]


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
        stack = np.broadcast_to(
            np.arange(11, dtype=np.uint16)[:, None, None], (11, 64, 80)
        )
        z.create_dataset("Channel_0", data=np.ascontiguousarray(stack))

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
        picture = np.full((3, 64, 80), 7, np.uint16)
        picture[2, 10, 10] = 9
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
    from mbo_utilities.gui.mesc_reference import reference_images

    mesc = MescArray(mesc_path, unit=1)
    try:
        images = reference_images(mesc, partial(open_unit, mesc_path), 0)
    finally:
        mesc.close()
    assert [im.key for im in images] == ["MUnit_4 picture", "MUnit_0 slices 4-8"]
    picture, stack = images
    # each image says which unit it is and, for a stack, the slice to open it at
    assert (picture.unit, picture.slice) == ("MSession_0/MUnit_4", None)
    assert (stack.unit, stack.slice) == ("MSession_0/MUnit_0", 3)
    # the picture is the max over its frames
    assert picture.image.shape == (64, 80) and picture.image.dtype == np.float32
    assert float(picture.image[0, 0]) == 7.0 and float(picture.image[10, 10]) == 9.0
    assert [r["roi"] for r in picture.records] == [0, 1, 2, 3]
    assert all(r["pixels"].shape == (2, 2) for r in picture.records)
    # slice k holds the value k: slices 3..7 project to 7, not the stack's 10
    np.testing.assert_allclose(stack.image, 7.0)
    assert [r["roi"] for r in stack.records] == [0, 1, 2]
    assert [r["slice"] for r in stack.records] == [3, 7, 3]


def test_a_sampled_projection_stays_a_max(mesc_path, monkeypatch):
    """A long picture is read from evenly spaced frames: the projection is
    still the max of what was read, at the picture's shape.
    """
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui import mesc_reference
    from mbo_utilities.gui.mesc_reference import reference_images

    # room for one frame of the picture at a time: frames 0 and 2 are read
    monkeypatch.setattr(mesc_reference, "MAX_ELEMENTS", 64 * 80 * 2)
    mesc = MescArray(mesc_path, unit=1)
    try:
        picture, stack = reference_images(mesc, partial(open_unit, mesc_path), 0)
    finally:
        mesc.close()
    assert picture.image.shape == (64, 80) and float(picture.image[10, 10]) == 9.0
    np.testing.assert_allclose(stack.image, 7.0)


def test_the_reader_places_its_lines(mesc_path):
    """``MescArray.line_positions``: each line's depth against the picture
    it was drawn on, read once; None for a unit without ROIs.
    """
    from mbo_utilities.arrays.mesc import MescArray

    scan = MescArray(mesc_path, unit=1)
    stack = MescArray(mesc_path, unit=0)
    try:
        positions = scan.line_positions
        assert [p["index"] for p in positions] == [0, 1, 2, 3]
        np.testing.assert_allclose(
            [p["dz_um"] for p in positions], [0.0, 8.0, 0.1, -16.0]
        )
        # and on the Z-stack holding them: the slice each sits on. The fourth
        # was scanned below the stack, so no stack holds it
        assert [p["stack"] for p in positions] == ["MSession_0/MUnit_0"] * 3 + [None]
        assert [p["slice"] for p in positions] == [3, 7, 3, None]
        assert [p["in_stack"] for p in positions] == [True, True, True, None]
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
    from mbo_utilities.gui.app.apps.mesc import display_wrap

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
                engine="voltage",
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
        assert widget._line_position(results_row)["slice"] == 7
        assert widget._trace_cells(results_row.key)[1:4] == ("2", "0", "voltage")
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
    from mbo_utilities.gui.mesc_reference import reference_images

    stack = MescArray(mesc_path, unit=0)
    snap = MescArray(mesc_path, unit="MUnit_4")
    try:
        assert reference_images(stack, partial(open_unit, mesc_path), 0) == []
        assert reference_images(snap, partial(open_unit, mesc_path), 0) == []
    finally:
        stack.close()
        snap.close()


SHOWN = []


def record_show(unit, at):
    SHOWN.append((unit, at))


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
    from mbo_utilities.gui.app.apps.mesc import display_wrap

    arr = MescArray(mesc_path, unit=1)
    iw = MboNDViewer(
        data=display_wrap(arr),
        slider_dim_names=arr.slider_dim_labels,
        figure_kwargs={"size": (640, 480)},
    )
    iw.show()
    strip = TopStrip(iw.figure)
    host = ViewerHost(iw)
    SHOWN.clear()
    view = ReferenceView(iw, on_show=record_show)
    errors = []

    def guarded(*_args):
        try:
            strip.update()
            view.draw()
        except Exception:
            errors.append(traceback.format_exc())

    try:
        assert view.open(arr, partial(open_unit, mesc_path))
        assert view.is_open and view.unit == "MUnit_1"
        assert view.viewer.current_key == "MUnit_4 picture"
        strip._update_calls[:] = [guarded]
        for _ in range(3):
            iw.figure.canvas.draw()
        assert not errors, errors[0]
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
        assert view.open(arr, partial(open_unit, mesc_path))
        assert view._built is built and view.viewer.current_key == "MUnit_4 picture"
        view.close()
        assert not view.is_open
    finally:
        view.cleanup()
        iw.close()
        arr.close()
