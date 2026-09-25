"""A recording's behavior: BehaviorMate's log as signals, events and epochs
on the recording's clock, found beside the recording, drawn under the traces.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from mbo_utilities.lazy_array import LazyArray


def write_tdml(path: Path, sync: bool = True) -> Path:
    """A short BehaviorMate session: one sync pulse at 0.5 s, 400 position
    samples 10 ms apart at 20 mm each (the 3000 mm track wraps twice) then a
    reset to 500 mm, one lick, one reward valve, one reward context, one lap.
    """
    rows = [
        {"git_revision": "abc", "version": "0.1.5"},
        {"settings": {"track_length": 3000, "sync_pin": 23, "position_scale": 1.7}},
        {"mouse": "u005a04", "experiment_group": "g", "start": "2026-09-15 11:16:28"},
    ]
    if sync:
        rows += [
            {
                "behavior_controller": {
                    "valve": {"pin": 23, "action": "open"},
                    "millis": 1,
                },
                "time": 0.5,
            },
            {
                "behavior_controller": {
                    "valve": {"pin": 23, "action": "close"},
                    "millis": 2,
                },
                "time": 0.6,
            },
        ]
    for k in range(400):
        rows.append(
            {
                "position_controller": {"position": {"dt": 10, "dy": 2}, "millis": k},
                "y": float((k * 20) % 3000),
                "time": 0.5 + k * 0.01,
            }
        )
    rows.append(
        {
            "position_controller": {"position": {"dt": 10, "dy": 2}, "millis": 400},
            "y": 500.0,
            "time": 4.5,
        }
    )
    rows += [
        {
            "behavior_controller": {
                "lick": {"pin": 3, "action": "start", "sensor": 0},
                "millis": 3,
            },
            "time": 1.5,
        },
        {
            "behavior_controller": {
                "lick": {"pin": 3, "action": "stop", "sensor": 0},
                "millis": 4,
            },
            "time": 1.6,
        },
        {
            "behavior_controller": {
                "context": {"action": "start", "id": "reward"},
                "millis": 5,
            },
            "time": 1.9,
        },
        {
            "behavior_controller": {"valve": {"pin": 52, "action": "open"}, "millis": 6},
            "time": 2.0,
        },
        {
            "behavior_controller": {
                "valve": {"pin": 52, "action": "close"},
                "millis": 7,
            },
            "time": 2.1,
        },
        {
            "behavior_controller": {
                "context": {"action": "stop", "id": "reward"},
                "millis": 8,
            },
            "time": 2.4,
        },
        {"lap": 0, "time": 2.5, "message": "no tag"},
        {"stop": "2026-09-15 11:33:00", "time": 4.5},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


class FakeRecording(LazyArray):
    """A recording at ``path`` with nothing behind it: the facet memo needs
    only ``source_path``, which ``LazyArray`` derives from ``filenames``.
    """

    def __init__(self, path):
        self.filenames = [Path(path)]


def test_tdml_reads_signals_events_and_epochs_on_the_recording_clock(tmp_path):
    from mbo_utilities.behavior import read_behavior

    b = read_behavior(write_tdml(tmp_path / "u005a04_20260915111628.tdml"))
    assert b and b.source == "BehaviorMate 0.1.5" and b.subject == "u005a04"
    assert b.start == datetime(2026, 9, 15, 11, 16, 28)
    assert b.offset_s == 0.5 and list(b.sync) == [0.5]
    assert b.info["track_length"] == 3000
    pos = b.signals["position"]
    assert pos.unit == "mm" and len(pos.t) == 401
    assert pos.t[0] == pytest.approx(0.0) and pos.t[-1] == pytest.approx(4.0)
    assert pos.values[149] == 2980.0 and pos.values[150] == 0.0
    speed = b.signals["speed"]
    assert speed.unit == "mm/s" and len(speed.t) == 400
    # the wraps at the track's end are unwrapped, so the speed never dips
    # there; the reset to 500 mm is a gap, not a run backwards
    assert np.allclose(speed.values[:-1], 2000.0, rtol=1e-3)
    assert np.isnan(speed.values[-1])
    assert list(b.events) == ["lick", "reward", "lap"]
    np.testing.assert_allclose(b.events["lick"], [1.0])
    np.testing.assert_allclose(b.events["reward"], [1.5])
    np.testing.assert_allclose(b.events["lap"], [2.0])
    np.testing.assert_allclose(b.epochs["reward"], [[1.4, 1.9]])
    assert b.duration_s == pytest.approx(4.0)


def test_without_a_sync_pulse_the_logger_clock_stands(tmp_path):
    from mbo_utilities.behavior import read_behavior

    b = read_behavior(write_tdml(tmp_path / "u005a04_20260915111628.tdml", sync=False))
    assert b.offset_s == 0.0 and len(b.sync) == 0
    assert b.signals["position"].t[0] == pytest.approx(0.5)
    np.testing.assert_allclose(b.events["reward"], [2.0])


def test_unknown_suffix_is_refused(tmp_path):
    from mbo_utilities.behavior import read_behavior

    (tmp_path / "x.csv").write_text("t,y\n")
    with pytest.raises(ValueError, match="no behavior reader"):
        read_behavior(tmp_path / "x.csv")


def test_find_behavior_matches_subject_and_day(tmp_path):
    from mbo_utilities.behavior import find_behavior

    imagings = tmp_path / "imagings"
    imagings.mkdir()
    rec = imagings / "u005a04_20260915_FamfDay1_FOV1-002.h5"
    rec.touch()
    beh = tmp_path / "behavior"
    beh.mkdir()
    write_tdml(beh / "u005a04_20260916123111.tdml")
    mine = write_tdml(beh / "u005a04_20260915111628.tdml")
    assert find_behavior(rec) == mine
    assert find_behavior(imagings / "u009a01_20260915_x.h5") is None
    assert find_behavior(tmp_path / "noday.h5") is None
    # a log beside the recording named after it wins over the sibling folder's
    beside = write_tdml(imagings / "u005a04_20260915_FamfDay1_FOV1-002.tdml")
    assert find_behavior(rec) == beside


def test_behavior_for_reads_once_and_keeps_the_answer(tmp_path):
    from mbo_utilities import imread
    from mbo_utilities.behavior import behavior_for

    imagings = tmp_path / "imagings"
    imagings.mkdir()
    rec = imagings / "u005a04_20260915_FamfDay1_FOV1-002.h5"
    rec.touch()
    write_tdml(imagings / "u005a04_20260915111628.tdml")
    arr = FakeRecording(rec)
    b = behavior_for(arr)
    assert b is not None and arr.behavior is b and behavior_for(arr) is b
    none = FakeRecording(imagings / "u009a01_20260101_x.h5")
    assert behavior_for(none) is None and hasattr(none, "_behavior")
    lazy = imread(np.zeros((3, 4, 4), np.float32))
    assert lazy.behavior is None
    lazy.behavior = b
    assert lazy.behavior is b


def test_nice_ticks_cover_the_data_and_only_the_data():
    from mbo_utilities.gui.imgui.behavior import nice_ticks

    ticks = nice_ticks(0.001, 3000.0)
    np.testing.assert_array_equal(ticks, [0.0, 1000.0, 2000.0, 3000.0])
    assert not np.signbit(ticks[0])
    np.testing.assert_array_equal(nice_ticks(-120.0, 400.0), [0.0, 200.0, 400.0])
    np.testing.assert_array_equal(nice_ticks(5.0, 5.0), [5.0])
    assert nice_ticks(0.0, 0.037).max() <= 0.037


def test_behavior_plot_decimates_and_draws(tmp_path):
    from mbo_utilities.behavior import read_behavior
    from mbo_utilities.gui.imgui.behavior import BehaviorPlot

    b = read_behavior(write_tdml(tmp_path / "u005a04_20260915111628.tdml"))
    plot = BehaviorPlot(b, points=50)
    assert plot and list(plot.signals) == ["position", "speed"]
    # min/max decimated: far fewer points than samples, t and values paired
    t, values, unit = plot.signals["position"]
    assert 50 <= len(t) <= 100 and t.shape == values.shape and unit == "mm"
    assert plot.lanes == ["lick", "reward", "lap"]
    assert plot.epochs == ["reward"] and plot.lane_total() == pytest.approx(9.0)
    assert plot.duration_s == pytest.approx(4.0)
    assert not BehaviorPlot(None)
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1200, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    result = None
    try:
        for _ in range(2):
            imgui.new_frame()
            imgui.set_next_window_size(imgui.ImVec2(1000, 600))
            imgui.begin("host")
            result = plot.draw(
                "##behavior", 300.0, cursor=1.0, cursor_id=1, x_per_second=1.0
            )
            imgui.end()
            imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)
    assert result == (1.0, False)
