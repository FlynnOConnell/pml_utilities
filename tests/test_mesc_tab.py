"""The MESc tab: a table of every measurement unit in the open file."""

from __future__ import annotations

import json

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
                {
                    "MethodType": 1,
                    "VecChannelsSize": 1,
                    "TStepInMs": step,
                    "MeasurementDatePosix": 1_700_000_000 + 100 * i,
                    "Comment": comment,
                }
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
    # no outlines and no picture: those columns stay empty; RTMC is off
    assert cells[4:7] == ("-", "-", "no")
    assert cells[7:12] == ("8", "1", "1", "16", "24")
    assert cells[12] == "20.0 Hz"
    assert cells[13] == "0 s"
    assert cells[14].startswith("2023-11-14 ") and "T" not in cells[14]
    assert cells[15] == "baseline / second line"
    # numeric columns sort as numbers, not strings
    assert keys[4] == 0 and keys[6] == 0
    assert keys[7] == 8 and keys[12] == 20.0
    assert unit_row(second)[1][7] == 16


@pytest.fixture(scope="module")
def linked_mesc_path(tmp_path_factory):
    """A line scan in MSession_0 whose snapshot and RTMC stream sit in MSession_1."""
    path = tmp_path_factory.mktemp("mesc_tab") / "linked.mesc"
    with h5py.File(path, "w") as f:
        scan = f.create_group("MSession_0").create_group("MUnit_0")
        scan.attrs.update(
            {
                "MethodType": 6,
                "VecChannelsSize": 1,
                "TStepInMs": 2.0,
                "MeasurementDatePosix": 1,
                "BackgroundImagePath": "/MSession_1/MUnit_0",
                "MotionCorrectionImagePath": "/MSession_1/MUnit_1",
            }
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
                {
                    "MethodType": 1,
                    "VecChannelsSize": 1,
                    "TStepInMs": 10.0,
                    "MeasurementDatePosix": 1,
                    "ImageRoleDebugString": role,
                }
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
    gives the companions rows again.
    """
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.widgets.mesc_units import (
        IMAGE_ICON,
        MescTabWidget,
        display_wrap,
    )

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
    # the picture cell is one button, which opens the reference image. No
    # popup, and RTMC is plain text, never a button
    assert f"{IMAGE_ICON} MUnit_0##pic" in BUTTONS and "?##mesc" in BUTTONS
    assert not any(
        b.endswith(("##rtmc", "##picimg", "##stack", "##stackimg")) for b in BUTTONS
    )
    assert folded[-1:] == [("MSession_0", True)]
    assert [r[0] for r in unfolded[-3:]] == ["MSession_0", "MSession_1", "MSession_1"]


def test_unit_row_names_the_companions(linked_mesc_path):
    from mbo_utilities.arrays.mesc import list_mesc_units
    from mbo_utilities.gui.widgets.mesc_units import (
        PICTURE_COLUMN,
        RTMC_COLUMN,
        companions,
        describe_unit,
        rtmc_on,
        unit_row,
    )

    scan, snap, stream = list_mesc_units(linked_mesc_path)
    assert companions([scan, snap, stream]) == {
        snap["key"]: scan["key"],
        stream["key"]: scan["key"],
    }
    # a reference unit alone is not RTMC on; curves are, with or without samples
    assert rtmc_on(scan) is False and rtmc_on(snap) is False
    assert rtmc_on({"rtmc": ["X total", "Z total"], "rtmc_armed": True}) is True
    assert rtmc_on({"rtmc": [], "rtmc_armed": True}) is True
    cells, keys = unit_row(scan)
    assert cells[:2] == ("MSession_0", "MUnit_0")
    assert (
        cells[4] == "4 lines"
        and cells[PICTURE_COLUMN] == "MUnit_0"
        and cells[RTMC_COLUMN] == "no"
    )
    assert keys[4] == 4 and keys[RTMC_COLUMN] == 0
    moved = unit_row({**scan, "rtmc": ["X total"], "rtmc_armed": True})
    assert moved[0][RTMC_COLUMN] == "yes" and moved[1][RTMC_COLUMN] == 1
    assert unit_row(snap)[0][:2] == ("MSession_1", "MUnit_0")
    assert unit_row(snap)[0][PICTURE_COLUMN] == "-"
    by_key = {u["key"]: u for u in (scan, snap, stream)}
    text = describe_unit(scan, by_key)
    assert text.startswith(
        "MSession_0/MUnit_0\nLine scan: 4 lines of 2 samples, 1 px wide, 500 times a second"
    )
    assert "Picture: MSession_1/MUnit_0" in text
    assert "RTMC off; reference pixels MSession_1/MUnit_1" in text
    assert "RTMC on; reference pixels MSession_1/MUnit_1" in describe_unit(
        {**scan, "rtmc": [], "rtmc_armed": True}, by_key
    )
    assert (
        describe_unit(snap, by_key)
        .splitlines()[1]
        .startswith("The picture the lines or patches were drawn on")
    )
    assert "Picture of MSession_0/MUnit_0" in describe_unit(snap, by_key)
    assert "RTMC reference pixels of MSession_0/MUnit_0" in describe_unit(
        stream, by_key
    )


def test_tab_is_supported_only_for_mesc_data(mesc_path):
    """The MESc tab is supported only for a viewer showing a MescArray."""
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.widgets.mesc_units import MescTabWidget, display_wrap

    arr = MescArray(mesc_path, unit=0)
    try:
        assert MescTabWidget.is_supported(FakeParent([display_wrap(arr)]))
        assert not MescTabWidget.is_supported(FakeParent([np.zeros((2, 4, 4))]))
        assert not MescTabWidget.is_supported(FakeParent([]))
        assert not MescTabWidget.is_supported(FakeParent(None))
    finally:
        arr.close()


def test_table_draws_every_unit_and_highlights_the_open_one(mesc_path):
    """One frame on a bare imgui context: a row per unit, the open one
    selected, the comment in the last column.
    """
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
    all but one, so the tab says so instead of offering to switch.
    """
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


def test_the_table_says_nothing_about_z_stacks(stack_mesc_path):
    """A scan's row names only the picture its lines were drawn on. Where the
    stack around them is, and which slice they sit on, belongs to the
    reference image, next to the lines themselves.
    """
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.widgets.mesc_units import (
        UNIT_COLUMNS,
        MescTabWidget,
        display_wrap,
    )

    assert "Z-stack" not in [name for name, _hidden in UNIT_COLUMNS]
    arr = MescArray(stack_mesc_path, unit=1)
    parent = FakeParent([display_wrap(arr)], arr.slider_dim_labels)
    widget = MescTabWidget(parent)
    BUTTONS.clear()
    imgui.small_button = spy_small_button
    try:
        draw_frames(widget)
    finally:
        imgui.small_button = REAL_SMALL_BUTTON
        arr.close()
    # this file's scan has no picture, so it has no button either
    assert not any("##stack" in b or "##pic" in b for b in BUTTONS)


def test_sidecars_are_named_after_the_unit(tmp_path):
    from mbo_utilities.gui.roi_runs import registry_path
    from mbo_utilities.roi_workflow import labels_path

    mesc = tmp_path / "scan.mesc"
    assert labels_path(mesc).name == "manual_labels.zarr"
    assert (
        labels_path(mesc, "MSession_0_MUnit_3").name
        == "manual_labels_MSession_0_MUnit_3.zarr"
    )
    assert labels_path(mesc, "MSession_0_MUnit_3").parent == tmp_path
    assert (
        registry_path(mesc, "MSession_0_MUnit_3").name
        == "roi_runs_MSession_0_MUnit_3.json"
    )
