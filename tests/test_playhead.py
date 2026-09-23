"""gui.playhead: one time in seconds, every view's own axis against it."""

from __future__ import annotations

import pytest
from mbo_utilities.gui.playhead import Playhead, TimeAxis


def test_axis_converts_both_ways():
    axis = TimeAxis(per_second=10.0, offset=2.0)
    assert axis.units(2.0) == 0.0
    assert axis.units(2.5) == pytest.approx(5.0)
    assert axis.seconds(5.0) == pytest.approx(2.5)
    assert axis.seconds(axis.units(7.25)) == pytest.approx(7.25)


def test_sampled_axis_from_rate_binning_and_window():
    # 10 Hz movie, one sample per 4 raw frames, starting at raw frame 8
    axis = TimeAxis.sampled(10.0, frame_average=4, first_frame=8)
    assert axis.per_second == pytest.approx(2.5)
    assert axis.offset == pytest.approx(0.8)
    assert axis.seconds(0) == pytest.approx(0.8)  # the first sample is frame 8
    assert axis.seconds(1) == pytest.approx(1.2)  # the next is frame 12
    # no rate: raw frames are the clock
    assert TimeAxis.sampled(None) == TimeAxis(1.0, 0.0)
    assert TimeAxis.sampled(0, frame_average=2, first_frame=3) == TimeAxis(0.5, 3.0)


def test_placing_one_axis_on_another():
    seconds = TimeAxis(1.0)
    frames = TimeAxis.sampled(10.0)
    binned = TimeAxis.sampled(10.0, frame_average=4, first_frame=8)
    # a binned window on the seconds axis: 0.4 s per sample, starting at 0.8 s
    assert binned.on(seconds) == (pytest.approx(0.4), pytest.approx(0.8))
    # the same on the viewer's frame axis: 4 frames per sample, from frame 8
    assert binned.on(frames) == (pytest.approx(4.0), pytest.approx(8.0))
    # an axis on itself is the identity
    assert frames.on(frames) == (1.0, 0.0)
    # ms axis
    assert frames.on(TimeAxis(1000.0)) == (pytest.approx(100.0), 0.0)


def test_seek_emits_once_with_its_source():
    head = Playhead()
    seen = []
    head.add_event_handler(seen.append, "time")
    assert head.seek(1.5, source="plot") is True
    assert head.seek(1.5) is False, "no event for the same time"
    assert head.seek(-3.0) is True and head.time == 0.0, "never before zero"
    assert [
        (e.info["seconds"], e.info["previous"], e.info["source"]) for e in seen
    ] == [
        (1.5, 0.0, "plot"),
        (0.0, 1.5, None),
    ]
