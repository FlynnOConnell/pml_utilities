"""The MESc tab: a table of every measurement unit in the open file."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from imgui_bundle import imgui

ROWS = []
TEXTS = []
REAL_SELECTABLE = imgui.selectable
REAL_TEXT = imgui.text


def spy_selectable(label, selected, *a, **k):
    ROWS.append((label.split("##")[0], selected))
    return REAL_SELECTABLE(label, selected, *a, **k)


def spy_text(text):
    TEXTS.append(text)
    return REAL_TEXT(text)

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
    def __init__(self, data):
        self.data = data

class FakeParent:
    def __init__(self, data):
        self.image_widget = FakeImageWidget(data)

def test_unit_row_reports_shape_rate_and_comment(mesc_path):
    from mbo_utilities.arrays.mesc import list_mesc_units
    from mbo_utilities.gui.widgets.mesc_units import UNIT_COLUMNS, unit_row

    first, second = list_mesc_units(mesc_path)
    cells, keys = unit_row(first)
    assert len(cells) == len(keys) == len(UNIT_COLUMNS)
    assert cells[:2] == ("MSession_0", "MUnit_0")
    assert cells[2] == "timeseries"
    # no outlines, no links: those columns stay empty
    assert cells[4:6] == ("-", "-")
    assert cells[6:11] == ("8", "1", "1", "16", "24")
    assert cells[11] == "20.0 Hz"
    assert cells[12] == "0 s"
    assert cells[13].startswith("2023-11-14 ") and "T" not in cells[13]
    assert cells[14] == "baseline / second line"
    # numeric columns sort as numbers, not strings
    assert keys[4] == 0 and keys[5] == 0 and keys[6] == 8 and keys[11] == 20.0
    assert unit_row(second)[1][6] == 16


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
    # pretend every links popup is open so its rows draw in the host window
    return str_id.startswith("##links_")


def spy_end_popup():
    return None


BUTTONS = []
REAL_SMALL_BUTTON = imgui.small_button
REAL_BEGIN_POPUP = imgui.begin_popup
REAL_END_POPUP = imgui.end_popup


def test_table_draws_a_links_button_only_for_paired_units(linked_mesc_path):
    """The links cell is a button carrying the count, or a dash: the scan
    has two (snapshot, RTMC stream), its snapshot and stream one each. With
    the popups forced open their rows go through the real ``selectable``."""
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.widgets.mesc_units import MescTabWidget, display_wrap

    arr = MescArray(linked_mesc_path, unit=0)
    widget = MescTabWidget(FakeParent([display_wrap(arr)]))
    BUTTONS.clear()
    ROWS.clear()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1400, 700)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    imgui.small_button = spy_small_button
    imgui.begin_popup, imgui.end_popup = spy_begin_popup, spy_end_popup
    imgui.selectable = spy_selectable
    try:
        for _ in range(2):
            imgui.new_frame()
            imgui.set_next_window_size(imgui.ImVec2(1300, 600))
            imgui.begin("host")
            widget.draw()
            imgui.end()
            imgui.end_frame()
    finally:
        imgui.small_button = REAL_SMALL_BUTTON
        imgui.begin_popup, imgui.end_popup = REAL_BEGIN_POPUP, REAL_END_POPUP
        imgui.selectable = REAL_SELECTABLE
        imgui.destroy_context(ctx)
        arr.close()
    assert BUTTONS[-3:] == ["2##links", "1##links", "1##links"]
    assert [r[0] for r in ROWS[-7:]] == [
        "MSession_0", "drawn on  MSession_1/MUnit_0  ·  timeseries",
        "RTMC stream  MSession_1/MUnit_1  ·  timeseries",
        "MSession_1", "background of  MSession_0/MUnit_0  ·  linescan",
        "MSession_1", "RTMC of  MSession_0/MUnit_0  ·  linescan",
    ]


def test_unit_links_and_row_counts(linked_mesc_path):
    from mbo_utilities.arrays.mesc import list_mesc_units
    from mbo_utilities.gui.widgets.mesc_units import LINKS_COLUMN, unit_links, unit_row

    scan, snap, stream = list_mesc_units(linked_mesc_path)
    assert unit_links(scan) == [
        ("drawn on", "MSession_1/MUnit_0"), ("RTMC stream", "MSession_1/MUnit_1"),
    ]
    assert unit_links(snap) == [("background of", "MSession_0/MUnit_0")]
    assert unit_links(stream) == [("RTMC of", "MSession_0/MUnit_0")]
    # a Z-stack's scans come from the geometry, handed in by the tab
    assert unit_links(snap, ["MSession_0/MUnit_2"])[-1] == ("holds", "MSession_0/MUnit_2")
    cells, keys = unit_row(scan)
    assert cells[:2] == ("MSession_0", "MUnit_0")
    assert cells[4] == "4 lines" and cells[LINKS_COLUMN] == "2"
    assert keys[4] == 4 and keys[LINKS_COLUMN] == 2
    assert unit_row(snap)[0][:2] == ("MSession_1", "MUnit_0")
    assert unit_row(snap)[0][LINKS_COLUMN] == "1"

def test_tab_appears_only_for_mesc_data(mesc_path):
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.widgets import get_tab_widgets
    from mbo_utilities.gui.widgets.mesc_units import MescTabWidget, display_wrap

    arr = MescArray(mesc_path, unit=0)
    try:
        names = [type(w).__name__ for w in get_tab_widgets(FakeParent([display_wrap(arr)]))]
        assert "MescTabWidget" in names
        assert names.index("PreviewTabWidget") < names.index("MescTabWidget")
        assert not MescTabWidget.is_supported(FakeParent([np.zeros((2, 4, 4))]))
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
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(900, 700)
    # imgui 1.92 builds fonts lazily once a renderer claims texture support
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    imgui.selectable, imgui.text = spy_selectable, spy_text
    try:
        for _ in range(2):
            imgui.new_frame()
            imgui.set_next_window_size(imgui.ImVec2(800, 600))
            imgui.begin("host")
            widget.draw()
            imgui.end()
            imgui.end_frame()
    finally:
        imgui.selectable, imgui.text = REAL_SELECTABLE, REAL_TEXT
        imgui.destroy_context(ctx)
        arr.close()
    # the session cell carries the row's selectable; the unit is the next cell
    assert ROWS[-2:] == [("MSession_0", False), ("MSession_0", True)]
    assert "MUnit_0" in TEXTS and "MUnit_1" in TEXTS
    assert "baseline / second line" in TEXTS
    assert "drug on" in TEXTS
