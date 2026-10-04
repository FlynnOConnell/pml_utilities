"""The slice on screen (AGENTS.md §7.6): one observable per host, roles from
the viewer, an event only when something but T moves.
"""

import numpy as np
from mbo_utilities.gui.slice import Slice, viewer_positions, viewer_roles


def test_roles_name_the_axis_whatever_the_slider_is_called():
    s = Slice()
    assert s.move({"t": 3, "z": 1}, {"t": "t", "z": "z"})
    assert (s.t, s.c, s.z) == (3, 0, 1) and s.of("c") is None
    s = Slice()
    s.move(
        {"Timepoint": 2, "Channel": 1, "ROI": 4},
        {"Timepoint": "t", "Channel": "c", "ROI": "z"},
    )
    assert (s.t, s.c, s.z) == (2, 1, 4)
    assert s.of("ROI") == 4 and s.of("z") == 4 and s.of("nope") is None
    assert s.positions == {"Timepoint": 2, "Channel": 1, "ROI": 4}


def test_only_a_non_time_move_emits():
    s = Slice()
    seen = []
    s.add_event_handler(seen.append, "slice")
    roles = {"t": "t", "c": "c", "z": "z"}
    assert s.move({"t": 0, "c": 0, "z": 0}, roles)
    assert not s.move({"t": 5, "c": 0, "z": 0}, roles)
    assert s.t == 5 and len(seen) == 1
    assert s.move({"t": 5, "c": 0, "z": 2}, roles, source="me")
    assert s.move({"t": 5, "c": 1, "z": 2}, roles)
    assert not s.move({"t": 6, "c": 1, "z": 2}, roles)
    assert len(seen) == 3
    event = seen[1]
    assert event.type == "slice" and event.source is s
    assert event.info["source"] == "me"
    assert event.info["previous"] == {"c": 0, "z": 0}
    assert event.info["positions"] == {"t": 5, "c": 0, "z": 2}
    # the handler sees the new position already
    assert seen[2].info["positions"]["c"] == 1


def test_a_data_swap_can_change_the_sliders():
    s = Slice()
    s.move({"t": 1, "z": 2}, {"t": "t", "z": "z"})
    assert s.move({"t": 0, "c": 0, "z": 0}, {"t": "t", "c": "c", "z": "z"})
    assert (s.c, s.z) == (0, 0) and set(s.positions) == {"t", "c", "z"}
    assert s.move({"t": 0}, {"t": "t"})
    assert s.z == 0 and s.of("z") is None


class _Viewer:
    def __init__(self, names, indices, data):
        self.dim_names = names
        self.indices = indices
        self.data = [data]


def test_viewer_roles_come_from_the_kept_axes_else_from_position():
    from mbo_utilities.gui.run_gui import _ScrubTimingProxy, _SqueezeSingletonDims

    # a z-stack: T is squeezed away, the two sliders are C and Z
    stack = _ScrubTimingProxy(_SqueezeSingletonDims(np.zeros((1, 2, 3, 4, 4))))
    iw = _Viewer(("Channel", "Zplane"), {"Channel": 1, "Zplane": 2}, stack)
    assert viewer_roles(iw) == {"Channel": "c", "Zplane": "z"}
    assert viewer_positions(iw) == (
        {"Channel": 1, "Zplane": 2},
        {"Channel": "c", "Zplane": "z"},
    )
    # a plain array: the sliders are T, C, Z by position
    plain = _Viewer(("a", "b"), {"a": 0, "b": 1}, np.zeros((5, 3, 4, 4)))
    assert viewer_roles(plain) == {"a": "t", "b": "z"}
