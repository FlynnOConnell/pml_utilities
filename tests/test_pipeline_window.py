"""A pipeline's configuration opened anywhere: the Process tab's widget in a
floating window, seeded from the recording and slice on screen.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from imgui_bundle import imgui

pytest.importorskip("vnoiser")

BUTTONS = []
REAL_SMALL_BUTTON = imgui.small_button


def spy_small_button(label, *a, **k):
    BUTTONS.append(label.split("##")[0])
    return REAL_SMALL_BUTTON(label, *a, **k)


@pytest.fixture(scope="module")
def roi_mesc(tmp_path_factory):
    """Two AOD line-scan units, four lines each."""
    path = tmp_path_factory.mktemp("pipeline_window") / "session1.mesc"
    with h5py.File(path, "w") as f:
        session = f.create_group("MSession_0")
        for n in (3, 5):
            unit = session.create_group(f"MUnit_{n}")
            unit.attrs.update(
                {
                    "MethodType": 6,
                    "VecChannelsSize": 1,
                    "TStepInMs": 1.0,
                    "MeasurementDatePosix": 0,
                    "CoordinateMapJSON": json.dumps(
                        {
                            "maps": [
                                {
                                    "measurementROIs": [
                                        {
                                            "lowerLeftFramePix": [2 * i + 1, 1],
                                            "upperRightFramePix": [2 * i + 2, 1],
                                        }
                                        for i in range(4)
                                    ]
                                }
                            ]
                        }
                    ),
                }
            )
            unit.create_dataset("Channel_0", data=np.zeros((1, 20, 8), np.uint16))
    return path


class FakeImageWidget:
    def __init__(self, data, dim_names=(), indices=None):
        self.data = data
        self.dim_names = tuple(dim_names)
        self.indices = dict(indices or {})


def fake_host(mesc, unit_key, roi=0):
    shown = SimpleNamespace(unit_key=unit_key, filenames=[mesc], metadata={})
    return SimpleNamespace(
        fpath=mesc,
        image_widget=FakeImageWidget(
            [shown], ("Timepoint", "ROI"), {"Timepoint": 0, "ROI": roi}
        ),
    )


def frames(fn, n=2):
    """Run ``fn`` for ``n`` frames on a bare imgui context."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1600, 900)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        for _ in range(n):
            imgui.new_frame()
            fn()
            imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)


def test_open_in_a_window_seeds_the_unit_and_slice_on_screen(roi_mesc):
    from mbo_utilities.gui.widgets.pipelines import (
        draw_pipeline_windows,
        open_pipeline,
    )

    host = fake_host(roi_mesc, "MSession_0/MUnit_5", roi=2)
    widget = open_pipeline(host, "Voltage", "window", seed=True)
    assert widget is not None
    assert host._pipeline_instances["Voltage"] is widget
    assert host._pipeline_windows == ["Voltage"]
    assert widget._scans == {"MSession_0/MUnit_3": False, "MSession_0/MUnit_5": True}
    assert widget._voltage_z_selection == "3"
    assert widget._voltage_c_selection == "1"
    assert widget._voltage_tp_selection == "1:20"
    # the same widget again, nothing rebuilt, still one window
    assert open_pipeline(host, "Voltage", "window") is widget
    assert host._pipeline_windows == ["Voltage"]
    frames(lambda: draw_pipeline_windows(host))
    assert host._pipeline_windows == ["Voltage"]


def test_open_in_the_tab_selects_it_there(roi_mesc):
    from mbo_utilities.gui.widgets.pipelines import open_pipeline

    host = fake_host(roi_mesc, "MSession_0/MUnit_3")
    widget = open_pipeline(host, "Voltage", "tab")
    assert widget is host._pipeline_instances["Voltage"]
    assert host._selected_pipeline_name == "Voltage"
    assert host._force_run_tab is True
    assert not hasattr(host, "_pipeline_windows")
    assert open_pipeline(host, "No such pipeline") is None


def test_a_picture_on_screen_leaves_the_scans_as_seeded(roi_mesc):
    from mbo_utilities.gui.widgets.pipelines import open_pipeline

    host = fake_host(roi_mesc, "MSession_0/MUnit_9", roi=1)
    widget = open_pipeline(host, "Voltage", "window", seed=True)
    assert widget._scans == {"MSession_0/MUnit_3": True, "MSession_0/MUnit_5": True}
    assert widget._voltage_z_selection == "1:4"


def test_shown_name_is_the_unit_else_the_file(roi_mesc, tmp_path):
    from mbo_utilities.gui.widgets.pipelines import shown_name

    assert shown_name(fake_host(roi_mesc, "MSession_0/MUnit_5")) == "MUnit_5"
    plain = SimpleNamespace(
        fpath=tmp_path / "x.tif", image_widget=FakeImageWidget([SimpleNamespace()])
    )
    assert shown_name(plain) == "x.tif"
    assert shown_name(SimpleNamespace(image_widget=None)) == ""


def test_quick_pipelines_are_the_view_seeded_ones_that_apply(roi_mesc, tmp_path):
    from mbo_utilities.gui.widgets.pipelines import (
        _register_pipelines,
        quick_pipelines,
    )

    _register_pipelines()
    host = fake_host(roi_mesc, "MSession_0/MUnit_3")
    assert [cls.name for cls in quick_pipelines(host)] == ["Voltage"]
    other = SimpleNamespace(
        fpath=tmp_path / "x.tif",
        image_widget=FakeImageWidget(
            [SimpleNamespace(filenames=[tmp_path / "x.tif"], metadata={})]
        ),
    )
    assert quick_pipelines(other) == []


def test_the_mesc_tab_offers_the_button_and_it_opens_the_window(roi_mesc):
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.widgets.mesc_units import MescTabWidget, display_wrap
    from mbo_utilities.gui.widgets.pipelines import _register_pipelines

    _register_pipelines()
    arr = MescArray(roi_mesc, unit="MSession_0/MUnit_5")
    host = SimpleNamespace(
        fpath=roi_mesc,
        image_widget=FakeImageWidget(
            [display_wrap(arr)], ("Timepoint", "ROI"), {"Timepoint": 0, "ROI": 1}
        ),
    )
    widget = MescTabWidget(host)
    BUTTONS.clear()
    imgui.small_button = spy_small_button
    try:
        frames(lambda: (imgui.begin("host"), widget.draw(), imgui.end()))
    finally:
        imgui.small_button = REAL_SMALL_BUTTON
        arr.close()
    assert "Voltage on MUnit_5" in BUTTONS
