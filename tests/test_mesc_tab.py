"""The MESc tab: a table of every measurement unit in the open file."""

from __future__ import annotations

import json
import time

import h5py
import numpy as np
import pytest
from imgui_bundle import imgui

ROWS = []
TEXTS = []
DISABLED = []
REAL_SELECTABLE = imgui.selectable
REAL_TEXT = imgui.text
REAL_TEXT_DISABLED = imgui.text_disabled


def spy_selectable(label, selected, *a, **k):
    ROWS.append((label.split("##")[0], selected))
    return REAL_SELECTABLE(label, selected, *a, **k)


def spy_text(text):
    TEXTS.append(text)
    return REAL_TEXT(text)


def spy_text_disabled(text):
    DISABLED.append(text)
    return REAL_TEXT_DISABLED(text)


@pytest.fixture(scope="module")
def mesc_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesc_tab") / "two_units.mesc"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")
        for i, (step, comment) in enumerate(
            ((50.0, "baseline\nsecond line"), (25.0, "drug on"))
        ):
            u = s.create_group(f"MUnit_{i}")
            u.attrs.update(
                {"MethodType": 1, "VecChannelsSize": 1, "TStepInMs": step,
                 "MeasurementDatePosix": 1_700_000_000 + 100 * i, "Comment": comment}
            )
            u.create_dataset(
                "Channel_0",
                data=rng.integers(0, 4000, (8 * (i + 1), 16, 24)).astype(np.uint16),
            )
    return path


class FakeImageWidget:
    def __init__(self, data, dim_names=(), z=0):
        self.data = data
        self.dim_names = tuple(dim_names)
        self.indices = dict.fromkeys(self.dim_names, z)


class FakeParent:
    def __init__(self, data, dim_names=(), z=0):
        self.image_widget = FakeImageWidget(data, dim_names, z)


def draw_frames(widget, n=2):
    """Draw ``widget`` for ``n`` frames on a bare imgui context."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    # wide enough for every column: a clipped column draws nothing
    io.display_size = imgui.ImVec2(2400, 700)
    # imgui 1.92 builds fonts lazily once a renderer claims texture support
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        for _ in range(n):
            imgui.new_frame()
            imgui.set_next_window_size(imgui.ImVec2(2300, 600))
            imgui.begin("host")
            widget.draw()
            imgui.end()
            imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)


def test_unit_row_reports_shape_rate_and_comment(mesc_path):
    from mbo_utilities.arrays.mesc import list_mesc_units
    from mbo_utilities.gui.widgets.mesc_units import UNIT_COLUMNS, unit_row

    first, second = list_mesc_units(mesc_path)
    cells, keys = unit_row(first)
    assert len(cells) == len(keys) == len(UNIT_COLUMNS)
    assert cells[:2] == ("MSession_0", "MUnit_0")
    assert cells[2] == "timeseries"
    # no outlines, no picture, no Z-stack: those columns stay empty; RTMC is off
    assert cells[4:8] == ("-", "-", "no", "-")
    assert cells[8:13] == ("8", "1", "1", "16", "24")
    assert cells[13] == "20.0 Hz"
    assert cells[14] == "0 s"
    assert cells[15].startswith("2023-11-14 ") and "T" not in cells[15]
    assert cells[16] == "baseline / second line"
    # numeric columns sort as numbers, not strings
    assert keys[4] == 0 and keys[6] == 0 and keys[7] == 0
    assert keys[8] == 8 and keys[13] == 20.0
    assert unit_row(second)[1][8] == 16


@pytest.fixture(scope="module")
def linked_mesc_path(tmp_path_factory):
    """A line scan in MSession_0 whose snapshot and RTMC stream sit in MSession_1."""
    path = tmp_path_factory.mktemp("mesc_tab") / "linked.mesc"
    with h5py.File(path, "w") as f:
        scan = f.create_group("MSession_0").create_group("MUnit_0")
        scan.attrs.update(
            {"MethodType": 6, "VecChannelsSize": 1, "TStepInMs": 2.0, "MeasurementDatePosix": 1,
             "BackgroundImagePath": "/MSession_1/MUnit_0",
             "MotionCorrectionImagePath": "/MSession_1/MUnit_1"}
        )
        boxes = [
            {"lowerLeftFramePix": [2 * i + 1, 1], "upperRightFramePix": [2 * i + 2, 1]}
            for i in range(4)
        ]
        lines = [[[110, 130], [210, 210], [-54, -54]]] * 4
        scan.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": boxes, "driftEndPoints": lines}]}
        )
        scan.create_dataset("Channel_0", data=np.zeros((1, 8, 8), np.uint16))
        refs = f.create_group("MSession_1")
        for i, role in enumerate(("background", "motionCorrection")):
            u = refs.create_group(f"MUnit_{i}")
            u.attrs.update(
                {"MethodType": 1, "VecChannelsSize": 1, "TStepInMs": 10.0,
                 "MeasurementDatePosix": 1, "ImageRoleDebugString": role}
            )
            u.create_dataset("Channel_0", data=np.zeros((1, 8, 8), np.uint16))
    return path


def spy_small_button(label):
    BUTTONS.append(label.split("##")[0] + "##" + label.split("##")[1].split("_")[0])
    return False


def spy_begin_popup(str_id, flags=0):
    # pretend every cell popup is open so its options draw in the host window
    return str_id.startswith(("##pic_", "##rtmc_", "##stack_"))


def spy_end_popup():
    return None


BUTTONS = []
REAL_SMALL_BUTTON = imgui.small_button
REAL_BEGIN_POPUP = imgui.begin_popup
REAL_END_POPUP = imgui.end_popup


def test_companions_fold_into_the_scans_row(linked_mesc_path):
    """The scan's picture and RTMC reference unit are cells of its row, not
    rows of their own: the picture cell is a button naming the picture, the
    RTMC cell says no (a reference unit, no curves). With the popups forced
    open their options go through the real ``selectable``; the checkbox
    gives the companions rows again."""
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.widgets.mesc_units import MescTabWidget, display_wrap

    arr = MescArray(linked_mesc_path, unit=0)
    widget = MescTabWidget(FakeParent([display_wrap(arr)]))
    BUTTONS.clear()
    ROWS.clear()
    imgui.small_button = spy_small_button
    imgui.begin_popup, imgui.end_popup = spy_begin_popup, spy_end_popup
    imgui.selectable = spy_selectable
    try:
        draw_frames(widget)
        folded = [r for r in ROWS]
        widget._show_companions = True
        ROWS.clear()
        draw_frames(widget)
        unfolded = [r for r in ROWS]
    finally:
        imgui.small_button = REAL_SMALL_BUTTON
        imgui.begin_popup, imgui.end_popup = REAL_BEGIN_POPUP, REAL_END_POPUP
        imgui.selectable = REAL_SELECTABLE
        arr.close()
    assert "MUnit_0##pic" in BUTTONS and "?##mesc" in BUTTONS and "Reference image##reference" in BUTTONS
    # RTMC is plain text, never a button
    assert not any(b.endswith("##rtmc") for b in BUTTONS)
    assert folded[-3:] == [
        ("MSession_0", True),
        ("Display MSession_1/MUnit_0", False),
        ("Reference image: this scan's lines on MUnit_0", False),
    ]
    assert [r[0] for r in unfolded[-5:]] == [
        "MSession_0", "Display MSession_1/MUnit_0", "Reference image: this scan's lines on MUnit_0",
        "MSession_1", "MSession_1",
    ]


def test_unit_row_names_the_companions(linked_mesc_path):
    from mbo_utilities.arrays.mesc import list_mesc_units
    from mbo_utilities.gui.widgets.mesc_units import (
        PICTURE_COLUMN,
        RTMC_COLUMN,
        STACK_COLUMN,
        companions,
        describe_unit,
        rtmc_on,
        unit_row,
    )

    scan, snap, stream = list_mesc_units(linked_mesc_path)
    assert companions([scan, snap, stream]) == {snap["key"]: scan["key"], stream["key"]: scan["key"]}
    # a reference unit alone is not RTMC on; curves are, with or without samples
    assert rtmc_on(scan) is False and rtmc_on(snap) is False
    assert rtmc_on({"rtmc": ["X total", "Z total"], "rtmc_armed": True}) is True
    assert rtmc_on({"rtmc": [], "rtmc_armed": True}) is True
    cells, keys = unit_row(scan)
    assert cells[:2] == ("MSession_0", "MUnit_0")
    assert cells[4] == "4 lines" and cells[PICTURE_COLUMN] == "MUnit_0" and cells[RTMC_COLUMN] == "no"
    assert keys[4] == 4 and keys[RTMC_COLUMN] == 0 and cells[STACK_COLUMN] == "-"
    moved = unit_row({**scan, "rtmc": ["X total"], "rtmc_armed": True})
    assert moved[0][RTMC_COLUMN] == "yes" and moved[1][RTMC_COLUMN] == 1
    # a Z-stack's scans come from the geometry, handed in by the tab
    cells, keys = unit_row(scan, stacks=["MSession_0/MUnit_9"])
    assert cells[STACK_COLUMN] == "MUnit_9" and keys[STACK_COLUMN] == 1
    stack = {**snap, "kind": "zstack"}
    assert unit_row(stack, stacks=[scan["key"], "x"])[0][STACK_COLUMN] == "2 scans"
    assert unit_row(snap)[0][:2] == ("MSession_1", "MUnit_0")
    assert unit_row(snap)[0][PICTURE_COLUMN] == "-"
    by_key = {u["key"]: u for u in (scan, snap, stream)}
    text = describe_unit(scan, by_key)
    assert text.startswith("MSession_0/MUnit_0\nLine scan: 4 lines of 2 samples, 1 px wide, 500 times a second")
    assert "Picture: MSession_1/MUnit_0" in text
    assert "RTMC off; reference pixels MSession_1/MUnit_1" in text
    assert "RTMC on; reference pixels MSession_1/MUnit_1" in describe_unit(
        {**scan, "rtmc": [], "rtmc_armed": True}, by_key
    )
    assert describe_unit(snap, by_key).splitlines()[1].startswith("The picture the lines or patches were drawn on")
    assert "Picture of MSession_0/MUnit_0" in describe_unit(snap, by_key)
    assert "RTMC reference pixels of MSession_0/MUnit_0" in describe_unit(stream, by_key)


def test_tab_appears_first_and_only_for_mesc_data(mesc_path):
    """The MESc tab is a .mesc file's first tab, ahead of Image, and is
    supported only for a viewer showing a MescArray."""
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.widgets import get_tab_widgets
    from mbo_utilities.gui.widgets.mesc_units import MescTabWidget, display_wrap

    arr = MescArray(mesc_path, unit=0)
    try:
        names = [type(w).__name__ for w in get_tab_widgets(FakeParent([display_wrap(arr)]))]
        assert "MescTabWidget" in names
        assert names.index("MescTabWidget") < names.index("PreviewTabWidget")
        assert MescTabWidget.is_supported(FakeParent([display_wrap(arr)]))
        assert not MescTabWidget.is_supported(FakeParent([np.zeros((2, 4, 4))]))
        assert not MescTabWidget.is_supported(FakeParent([]))
        assert not MescTabWidget.is_supported(FakeParent(None))
    finally:
        arr.close()


def test_table_draws_every_unit_and_highlights_the_open_one(mesc_path):
    """One frame on a bare imgui context: a row per unit, the open one
    selected, the comment in the last column."""
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.widgets.mesc_units import MescTabWidget, display_wrap

    arr = MescArray(mesc_path, unit=1)
    widget = MescTabWidget(FakeParent([display_wrap(arr)]))
    ROWS.clear()
    TEXTS.clear()
    DISABLED.clear()
    imgui.selectable, imgui.text = spy_selectable, spy_text
    imgui.text_disabled = spy_text_disabled
    try:
        draw_frames(widget)
    finally:
        imgui.selectable, imgui.text = REAL_SELECTABLE, REAL_TEXT
        imgui.text_disabled = REAL_TEXT_DISABLED
        arr.close()
    # the session cell carries the row's selectable; the unit is the next cell
    assert ROWS[-2:] == [("MSession_0", False), ("MSession_0", True)]
    assert "MUnit_0" in TEXTS and "MUnit_1" in TEXTS
    assert "baseline / second line" in TEXTS
    assert "drug on" in TEXTS
    assert "Click a row to display that unit." in DISABLED


def test_split_roi_views_stand_down(mesc_path):
    """`--roi 0` fans one unit's ROIs across subplots; swapping would strand
    all but one, so the tab says so instead of offering to switch."""
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.widgets.mesc_units import MescTabWidget, display_wrap

    arr = MescArray(mesc_path, unit=0)
    widget = MescTabWidget(FakeParent([display_wrap(arr), display_wrap(arr)]))
    DISABLED.clear()
    imgui.text_disabled = spy_text_disabled
    try:
        draw_frames(widget)
    finally:
        imgui.text_disabled = REAL_TEXT_DISABLED
        arr.close()
    assert "Split ROIs: reopen without --roi to switch units." in DISABLED
    assert "Click a row to display that unit." not in DISABLED


@pytest.fixture(scope="module")
def stack_mesc_path(tmp_path_factory):
    """A Z-stack (11 slices, 2 um apart, FOV corner at z = -50 um) and a line
    scan with four lines: two on slice 3, one on slice 7, one below the stack."""
    path = tmp_path_factory.mktemp("mesc_tab") / "stack.mesc"
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")
        z = s.create_group("MUnit_0")
        z.attrs.update(
            {"MethodType": 2, "VecChannelsSize": 1, "TStepInMs": 1.0,
             "MeasurementDatePosix": 0, "Comment": "zstack",
             "MinZ": -10.0, "MaxZ": 10.0, "ZDim": 11}
        )
        z.attrs["ReferenceViewportJSON"] = json.dumps(
            {"viewports": [{"geomTransTransl": [100.0, 200.0, -50.0], "width": 40.0, "height": 32.0}]}
        )
        z.create_dataset("Channel_0", data=np.zeros((11, 64, 80), np.uint16))
        ls = s.create_group("MUnit_1")
        ls.attrs.update(
            {"MethodType": 6, "VecChannelsSize": 1, "TStepInMs": 2.0,
             "MeasurementDatePosix": 1, "Comment": "linescan"}
        )
        boxes = [
            {"lowerLeftFramePix": [2 * i + 1, 1], "upperRightFramePix": [2 * i + 2, 1]}
            for i in range(4)
        ]
        lines = [
            [[110, 130], [210, 210], [-54, -54]],
            [[120, 120], [204, 228], [-46, -46]],
            [[105, 135], [220, 230], [-53.9, -53.9]],
            [[112, 118], [212, 212], [-70, -70]],
        ]
        ls.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": boxes, "driftEndPoints": lines}]}
        )
        ls.create_dataset("Channel_0", data=np.zeros((1, 8, 8), np.uint16))
    return path


def test_zstack_column_names_the_stack_around_a_scan(stack_mesc_path):
    """A scan's row names the Z-stack whose field holds its lines, worked
    out from the micron positions; the stack's row counts its scans. Both
    are buttons whose popup displays the other unit; a scan's also opens
    the reference image on the stack."""
    from mbo_utilities.arrays.mesc import MescArray, list_mesc_units
    from mbo_utilities.gui.widgets.mesc_units import (
        STACK_COLUMN,
        MescTabWidget,
        display_wrap,
        unit_row,
    )

    stack, scan = list_mesc_units(stack_mesc_path)
    arr = MescArray(stack_mesc_path, unit=1)
    parent = FakeParent([display_wrap(arr)], arr.slider_dim_labels)
    widget = MescTabWidget(parent)
    stacks = widget._stacks(arr)
    assert stacks == {stack["key"]: [scan["key"]], scan["key"]: [stack["key"]]}
    assert unit_row(scan, stacks[scan["key"]])[0][STACK_COLUMN] == "MUnit_0"
    assert unit_row(stack, stacks[stack["key"]])[0][STACK_COLUMN] == "1 scan"
    BUTTONS.clear()
    ROWS.clear()
    imgui.small_button = spy_small_button
    imgui.begin_popup, imgui.end_popup = spy_begin_popup, spy_end_popup
    imgui.selectable = spy_selectable
    try:
        draw_frames(widget)
    finally:
        imgui.small_button = REAL_SMALL_BUTTON
        imgui.begin_popup, imgui.end_popup = REAL_BEGIN_POPUP, REAL_END_POPUP
        imgui.selectable = REAL_SELECTABLE
        arr.close()
    assert "MUnit_0##stack" in BUTTONS and "1 scan##stack" in BUTTONS
    labels = [r[0] for r in ROWS]
    assert "Display MUnit_0" in labels and "Reference image: this scan's lines on MUnit_0" in labels
    assert "Display MUnit_1" in labels and "Reference image: this scan's lines on MUnit_1" not in labels


def pump(widget, seconds: float = 60.0):
    """Poll the ROI widget's background work until it finishes."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        widget._poll_jobs()
        if not widget.busy:
            widget._poll_jobs()
            return
        time.sleep(0.02)
    raise TimeoutError("background work did not finish")


def test_sidecars_are_named_after_the_unit(tmp_path):
    from mbo_utilities.gui.roi_runs import registry_path
    from mbo_utilities.roi_workflow import labels_path

    mesc = tmp_path / "scan.mesc"
    assert labels_path(mesc).name == "manual_labels.zarr"
    assert labels_path(mesc, "MSession_0_MUnit_3").name == "manual_labels_MSession_0_MUnit_3.zarr"
    assert labels_path(mesc, "MSession_0_MUnit_3").parent == tmp_path
    assert registry_path(mesc, "MSession_0_MUnit_3").name == "roi_runs_MSession_0_MUnit_3.json"


def test_switching_units_keeps_rois_and_traces_per_unit(mesc_path):
    """Each unit is its own recording: its ROIs and traces leave the screen
    with it and come back when it is shown again, and each autosaves under
    its own name beside the file."""
    pytest.importorskip("fastplotlib.widgets.nd_widget")
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.run_gui import _create_image_widget
    from mbo_utilities.gui.widgets.mesc_units import MescTabWidget
    from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget
    from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled, widget_enabled

    was = widget_enabled("manual_roi")
    set_widget_enabled("manual_roi", True, persist=False)
    arr = MescArray(mesc_path, unit=0)
    iw = _create_image_widget(arr, widget="preview", figure_kwargs_override={"size": (640, 480)})
    try:
        gui = next(w for w in iw.figure.imgui_windows.values() if isinstance(w, PreviewDataWidget))
        gui.sync_manual_roi(True)
        first = gui.manual_roi
        assert first is not None and first.unit == "MUnit_0"
        assert first.tag == "MSession_0_MUnit_0"
        assert first._save_target().name == "manual_labels_MSession_0_MUnit_0.zarr"
        assert first.run_prefix == "rois_MSession_0_MUnit_0_"
        first.auto_trace = False
        first.add_roi([(2.0, 2.0), (9.0, 2.0), (9.0, 9.0), (2.0, 9.0)])
        first.quick_trace(0)
        pump(first)
        assert first.n_rois == 1 and first.has_traces()

        tab = MescTabWidget(gui)
        tab._switch(arr.units[1])
        second = gui.manual_roi
        assert second is not None and second is not first and second.unit == "MUnit_1"
        assert second.n_rois == 0 and not second.has_traces()
        assert second._save_target().name == "manual_labels_MSession_0_MUnit_1.zarr"
        second.add_roi([(3.0, 3.0), (8.0, 3.0), (8.0, 8.0), (3.0, 8.0)])
        second.add_roi([(12.0, 3.0), (18.0, 3.0), (18.0, 9.0), (12.0, 9.0)])
        assert second.n_rois == 2

        tab._switch(arr.units[0])
        back = gui.manual_roi
        assert back.unit == "MUnit_0" and back.n_rois == 1 and back.has_traces()
        assert back.store is first.store
        tab._switch(arr.units[1])
        assert gui.manual_roi.unit == "MUnit_1" and gui.manual_roi.n_rois == 2
        assert (mesc_path.parent / "manual_labels_MSession_0_MUnit_0.zarr").exists()
        assert (mesc_path.parent / "manual_labels_MSession_0_MUnit_1.zarr").exists()
        assert not (mesc_path.parent / "manual_labels.zarr").exists()
    finally:
        iw.close()
        set_widget_enabled("manual_roi", was, persist=False)
