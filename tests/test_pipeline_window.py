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


SPAWNED = []
REAL_BUTTON = imgui.button


def press_run(label, *a, **k):
    return REAL_BUTTON(label, *a, **k) or label == "Run Voltage"


class FakeProcessManager:
    def spawn(self, **kwargs):
        SPAWNED.append(kwargs)
        return 4242


def fake_process_manager():
    return FakeProcessManager()


GUIDE_DRAWS = []


def count_guide(is_open, keys_open):
    GUIDE_DRAWS.append(is_open)
    return is_open, keys_open


def press_guide(label, *a, **k):
    return REAL_SMALL_BUTTON(label, *a, **k) or "vnoiser guide" in label


def line_scan_unit(session, n, lines=4, channels=1):
    """An AOD line-scan unit ``MUnit_<n>`` of 20 frames."""
    unit = session.create_group(f"MUnit_{n}")
    unit.attrs.update(
        {
            "MethodType": 6,
            "VecChannelsSize": channels,
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
                                for i in range(lines)
                            ]
                        }
                    ]
                }
            ),
        }
    )
    for c in range(channels):
        unit.create_dataset(
            f"Channel_{c}", data=np.zeros((1, 20, 2 * lines), np.uint16)
        )


@pytest.fixture(scope="module")
def roi_mesc(tmp_path_factory):
    """Two AOD line-scan units, four lines each."""
    path = tmp_path_factory.mktemp("pipeline_window") / "session1.mesc"
    with h5py.File(path, "w") as f:
        session = f.create_group("MSession_0")
        for n in (3, 5):
            line_scan_unit(session, n)
    return path


@pytest.fixture(scope="module")
def uneven_mesc(tmp_path_factory):
    """Two AOD line-scan units: four lines in one channel, six lines in two."""
    path = tmp_path_factory.mktemp("pipeline_window_uneven") / "session2.mesc"
    with h5py.File(path, "w") as f:
        session = f.create_group("MSession_0")
        line_scan_unit(session, 3)
        line_scan_unit(session, 5, lines=6, channels=2)
    return path


class FakeImageWidget:
    def __init__(self, data, dim_names=(), indices=None):
        self.data = data
        self.dim_names = tuple(dim_names)
        self.indices = dict(indices or {})


def fake_host(mesc, unit_key, roi=0, channel=None):
    """A host showing ``unit_key`` with its sliders on ``roi`` and, for a
    two-channel unit, ``channel``.
    """
    shown = SimpleNamespace(unit_key=unit_key, filenames=[mesc], metadata={})
    names, indices = ("Timepoint", "ROI"), {"Timepoint": 0, "ROI": roi}
    if channel is not None:
        names = ("Timepoint", "Channel", "ROI")
        indices["Channel"] = channel
    return SimpleNamespace(
        fpath=mesc, image_widget=FakeImageWidget([shown], names, indices)
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


def test_the_domain_table_follows_the_scan_on_screen(uneven_mesc):
    """One domain per ROI of the scan on screen, refitted when a scan with
    other ROIs is shown; a table that was loaded or edited stays while it
    names only ROIs the scan has, and gets the ROI on screen as a domain of
    its own when it leaves it out.
    """
    from mbo_utilities.gui.widgets.pipelines import open_pipeline

    host = fake_host(uneven_mesc, "MSession_0/MUnit_3", roi=1)
    widget = open_pipeline(host, "Voltage", "window", seed=True)
    assert widget._domain_rows == [[f"roi{i}", str(i)] for i in range(4)]
    host.image_widget = fake_host(
        uneven_mesc, "MSession_0/MUnit_5", roi=5, channel=1
    ).image_widget
    open_pipeline(host, "Voltage", "window", seed=True)
    assert widget._scans == {"MSession_0/MUnit_3": False, "MSession_0/MUnit_5": True}
    assert widget._domain_rows == [[f"roi{i}", str(i)] for i in range(6)]
    assert widget._voltage_z_selection == "6" and widget._voltage_c_selection == "2"
    widget._domain_rows = [["soma", "0,1"], ["dend", "2:3"]]
    open_pipeline(host, "Voltage", "window", seed=True)
    assert widget._domain_rows == [["soma", "0,1"], ["dend", "2:3"], ["roi5", "5"]]
    assert widget._domains() == {"soma": [0, 1], "dend": [2, 3], "roi5": [5]}
    # four lines again: the table names a line this scan lacks
    host.image_widget = fake_host(uneven_mesc, "MSession_0/MUnit_3", roi=2).image_widget
    open_pipeline(host, "Voltage", "window", seed=True)
    assert widget._domain_rows == [[f"roi{i}", str(i)] for i in range(4)]
    assert widget._voltage_z_selection == "3" and widget._voltage_c_selection == "1"


def test_a_run_of_a_scan_with_other_rois_does_not_seed_the_domains(tmp_path):
    """A previous run's domain table seeds a scan with as many ROIs as the
    scans that run processed, never one with another number of them.
    """
    from mbo_utilities.gui.widgets.pipelines import open_pipeline
    from mbo_utilities.results import (
        Results,
        ResultUnit,
        pipeline_files,
        results_name,
    )

    mesc = tmp_path / "session3.mesc"
    with h5py.File(mesc, "w") as f:
        session = f.create_group("MSession_0")
        line_scan_unit(session, 3)
        line_scan_unit(session, 4)
        line_scan_unit(session, 5, lines=6)
    source = {"mesc": str(mesc), "units": {"3": "MSession_0/MUnit_3"}}
    scan = ResultUnit(
        name="scan3",
        kind="scan",
        index=3,
        fs=1000.0,
        roi_names=["soma"],
        traces={"denoised": np.zeros((1, 20))},
        member_kind="line",
        members=[np.array([0, 1])],
        attrs={"scan_id": "3"},
    )
    run = Results(pipeline="voltage", units={scan.name: scan}, source=source).write(
        tmp_path / results_name(mesc, pipeline="voltage")
    )
    pipeline_files(run).mkdir()
    (pipeline_files(run) / "pipeline.json").write_text(
        json.dumps(
            {
                "domains": {"soma": [0, 1]},
                "scan_ids": ["3"],
                "source": source,
                "settings": {"runtime": {"output_format": "zarr"}},
            }
        )
    )
    same = fake_host(mesc, "MSession_0/MUnit_4")
    widget = open_pipeline(same, "Voltage", "window", seed=True)
    assert widget._domain_rows == [["soma", "0,1"]]
    assert widget._last_status.startswith("Loaded the previous run's scans and domains")
    other = fake_host(mesc, "MSession_0/MUnit_5")
    widget = open_pipeline(other, "Voltage", "window", seed=True)
    assert widget._domain_rows == [[f"roi{i}", str(i)] for i in range(6)]
    assert widget._last_status.startswith("Loaded the previous run's settings")


def test_run_submits_the_scan_roi_and_channel_on_screen(uneven_mesc, monkeypatch):
    """Opened from the button and run untouched, the worker gets the scan on
    screen, the ROI its slider is on as ``planes`` and the channel its slider
    is on, whatever scan the widget was last set to.
    """
    from mbo_utilities.gui.widgets.pipelines import (
        draw_pipeline_windows,
        open_pipeline,
    )

    monkeypatch.setattr(
        "mbo_utilities.gui.widgets.process_manager.get_process_manager",
        fake_process_manager,
    )
    host = fake_host(uneven_mesc, "MSession_0/MUnit_3", roi=1)
    open_pipeline(host, "Voltage", "window", seed=True)
    host.image_widget = fake_host(
        uneven_mesc, "MSession_0/MUnit_5", roi=5, channel=1
    ).image_widget
    widget = open_pipeline(host, "Voltage", "window", seed=True)
    SPAWNED.clear()
    monkeypatch.setattr(imgui, "button", press_run)
    frames(lambda: draw_pipeline_windows(host), n=1)
    assert len(SPAWNED) == 1 and SPAWNED[0]["task_type"] == "voltage"
    args = SPAWNED[0]["args"]
    assert args["input_path"] == str(uneven_mesc)
    assert args["units"] == ["MSession_0/MUnit_5"]
    assert args["planes"] == [6] and args["channel"] == 1 and args["frames"] is None
    assert args["domains"] == {f"roi{i}": [i] for i in range(6)}
    assert args["output_dir"] == str(uneven_mesc.parent)
    assert widget._last_status.startswith("Started (PID 4242)")


def test_the_voltage_window_opens_the_vnoiser_guide(roi_mesc, monkeypatch):
    """Its button toggles the guide, drawn once a frame even when the tab and
    the popped-out window both draw the widget.
    """
    from mbo_utilities.gui.widgets.pipelines import open_pipeline

    monkeypatch.setattr(
        "mbo_utilities.gui.widgets.pipelines.voltage.draw_vnoiser_help", count_guide
    )
    widget = open_pipeline(fake_host(roi_mesc, "MSession_0/MUnit_5"), "Voltage")
    GUIDE_DRAWS.clear()
    frames(
        lambda: (
            imgui.begin("host"),
            widget.draw_config(),
            imgui.push_id("window"),
            widget.draw_config(),
            imgui.pop_id(),
            imgui.end(),
        )
    )
    assert GUIDE_DRAWS == [False, False]
    monkeypatch.setattr(imgui, "small_button", press_guide)
    frames(lambda: (imgui.begin("host"), widget.draw_config(), imgui.end()), n=1)
    assert widget._help_open and GUIDE_DRAWS[-1] is True


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
