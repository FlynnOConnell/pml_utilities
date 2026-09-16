"""The MESc tab: a table of every measurement unit in the open file."""

from __future__ import annotations

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
    assert cells[0] == "MUnit_0"
    assert cells[1] == "timeseries"
    assert cells[3:8] == ("8", "1", "1", "16", "24")
    assert cells[8] == "20.0 Hz"
    assert cells[9] == "0 s"
    assert cells[10].startswith("2023-11-14 ") and "T" not in cells[10]
    assert cells[11] == "baseline / second line"
    # numeric columns sort as numbers, not strings
    assert keys[3] == 8 and keys[8] == 20.0
    assert unit_row(second)[1][3] == 16

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
    assert ROWS[-2:] == [("MUnit_0", False), ("MUnit_1", True)]
    assert "baseline / second line" in TEXTS
    assert "drug on" in TEXTS
