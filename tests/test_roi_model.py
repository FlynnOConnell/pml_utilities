"""mbo_utilities.annotation's session model: the store's plane geometry,
the observable events, the trace table and RoiModel's run targets. Pure
numpy; no canvas.
"""

from __future__ import annotations

import numpy as np
import pytest
from mbo_utilities.annotation import (
    COLUMNS,
    ENGINES,
    FULL_IMAGE,
    ModelEvent,
    Observable,
    RoiLabelStore,
    RoiModel,
    RoiTrace,
    RoiTraceTable,
    RunTarget,
    trace_key,
)


def disk(ny, nx, cy, cx, r):
    yy, xx = np.mgrid[:ny, :nx]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


@pytest.fixture
def cz_store():
    """(c, z) keyed planes: 2 channels x 3 z-planes, z fastest."""
    store = RoiLabelStore(6, 16, 16, min_pixels=1)
    store.plane_axes = (("c", 2), ("z", 3))
    return store


class _Recorder:
    def __init__(self):
        self.events: list[ModelEvent] = []

    def __call__(self, event):
        self.events.append(event)

    @property
    def actions(self):
        return [e.info.get("action") for e in self.events]


class TestObservable:
    def test_handlers_get_typed_events_once(self):
        class Thing(Observable):
            events = ("a", "b")

        thing = Thing()
        seen = _Recorder()
        thing.add_event_handler(seen, "a")
        thing.add_event_handler(seen, "a")  # a repeat registration is a no-op
        thing._emit("a", action="x")
        thing._emit("b", action="y")
        assert [e.type for e in seen.events] == ["a"]
        assert seen.events[0].info == {"action": "x"} and seen.events[0].source is thing
        thing.remove_event_handler(seen)
        thing._emit("a", action="z")
        assert len(seen.events) == 1

    def test_unknown_types_are_refused_and_blocking_silences(self):
        class Thing(Observable):
            events = ("a",)

        thing = Thing()
        with pytest.raises(ValueError):
            thing.add_event_handler(lambda e: None, "nope")
        seen = _Recorder()
        thing.add_event_handler(seen)  # no types = every type
        thing.block_events(True)
        assert thing._emit("a") is None
        thing.block_events(False)
        thing._emit("a")
        assert len(seen.events) == 1


class TestPlaneGeometry:
    def test_plain_z_store_keeps_plane_equal_to_z(self):
        store = RoiLabelStore(3, 8, 8)
        store.plane_axes = (("z", 3),)
        assert store.plane_of({"z": 2, "t": 5}) == 2
        assert store.plane_pos(2) == {"z": 2}
        assert store.plane_label(2) == "3"
        assert store.axis_name("z") == "z" and store.axis_name("c") is None
        assert store.axis_size("z") == 3 and store.axis_size("c") == 1

    def test_depthless_store_is_one_plane(self):
        store = RoiLabelStore(1, 8, 8)
        assert store.plane_of({"z": 4}) == 0
        assert store.plane_pos(0) == {} and store.plane_label(0) == "1"
        store.add_roi(0, disk(8, 8, 4, 4, 2))
        assert store.roi_z(0) == 0 and store.roi_c(0) == 0

    def test_channel_and_z_fold_with_z_fastest(self, cz_store):
        assert cz_store.plane_of({"c": 0, "z": 2}) == 2
        assert cz_store.plane_of({"c": 1, "z": 2}) == 5
        assert cz_store.plane_of({"c": 9, "z": -1}) == 3  # clipped, not raised
        assert cz_store.plane_pos(5) == {"c": 1, "z": 2}
        assert cz_store.plane_label(5) == "c2·z3"

    def test_roi_z_and_c_decode_the_plane(self, cz_store):
        cz_store.add_roi(5, disk(16, 16, 8, 8, 3))
        cz_store.add_roi(1, disk(16, 16, 8, 8, 3))
        assert (cz_store.roi_z(0), cz_store.roi_c(0)) == (2, 1)
        assert (cz_store.roi_z(1), cz_store.roi_c(1)) == (1, 0)
        assert cz_store.roi_pos(0) == {"c": 1, "z": 2}

    def test_isoview_names_resolve_by_alias(self):
        store = RoiLabelStore(4, 8, 8)
        store.plane_axes = (("Cam", 2), ("Zplane", 2))
        assert store.axis_name("c") == "Cam" and store.axis_name("z") == "Zplane"
        store.add_roi(3, disk(8, 8, 4, 4, 2))
        assert (store.roi_z(0), store.roi_c(0)) == (1, 1)


class TestStoreEvents:
    def test_every_mutation_emits_one_rois_event(self):
        store = RoiLabelStore(1, 8, 8, min_pixels=1)
        seen = _Recorder()
        store.add_event_handler(seen, "rois")
        store.add_label_name("soma")
        store.add_roi(0, disk(8, 8, 4, 4, 2))
        store.set_class(0, 0)
        store.set_note(0, "n")
        store.set_color(0, (1, 2, 3))
        store.delete_roi(0)
        store.clear()
        assert seen.actions == [
            "labels",
            "add",
            "class",
            "note",
            "color",
            "delete",
            "clear",
        ]
        add = seen.events[1]
        assert add.info["index"] == 0 and add.info["uid"] == 1
        # a refused add (nothing free) emits nothing
        store.add_roi(0, np.zeros((8, 8), bool))
        assert len(seen.events) == 7

    def test_snapshot_carries_no_handlers(self):
        store = RoiLabelStore(1, 8, 8, min_pixels=1)
        seen = _Recorder()
        store.add_event_handler(seen, "rois")
        snap = store.snapshot()
        snap.add_roi(0, disk(8, 8, 4, 4, 2))
        assert seen.events == []


class TestTraceTable:
    def test_rows_are_keyed_by_where_and_how(self):
        table = RoiTraceTable()
        seen = _Recorder()
        table.add_event_handler(seen)
        table.add(RoiTrace(uid=7, z=0, c=0, engine="mean", F=np.ones(4)))
        table.add(RoiTrace(uid=7, z=0, c=1, engine="mean", F=np.zeros(4)))
        table.add(
            RoiTrace(
                uid=7, z=0, c=0, engine="suite2p", F=np.ones(4), source="rois_manual"
            )
        )
        assert len(table) == 3
        assert [t.c for t in table.for_roi(7)] == [0, 1, 0]
        assert [t.engine for t in table.at(7, c=0)] == ["mean", "suite2p"]
        assert table.at(7, z=0, c=1, engine="mean")[0].F.sum() == 0
        assert trace_key(7, 0, 0, "mean") in table
        assert seen.actions == ["add", "add", "add"]
        assert not seen.events[0].info["replaced"]

    def test_rerunning_the_same_measurement_replaces_the_row(self):
        table = RoiTraceTable()
        seen = _Recorder()
        table.add_event_handler(seen)
        first = table.add(RoiTrace(uid=1, F=np.ones(3), source="quick"))
        second = table.add(RoiTrace(uid=1, F=np.full(3, 2.0), source="rois_manual"))
        assert len(table) == 1 and table.get(first.key) is second
        assert seen.events[-1].info["replaced"] is True
        assert table.sources() == ["rois_manual"]

    def test_member_rows_stand_for_no_roi_and_survive_pruning(self):
        table = RoiTraceTable()
        table.add(RoiTrace(uid=1, F=np.ones(3)))
        table.add(RoiTrace(uid=2, F=np.ones(3)))
        line = table.add(
            RoiTrace(
                uid=0, source="MUnit_3 lines", member=4, F=np.ones(3), label="ROI 4"
            )
        )
        assert line.key == ("member", "MUnit_3 lines", 4)
        assert not line.stands_for_roi
        gone = table.prune([2])
        assert gone == [trace_key(1, 0, 0, "mean")]
        assert [t.key for t in table] == [trace_key(2, 0, 0, "mean"), line.key]
        assert table.drop_source("MUnit_3 lines") == [line.key]
        assert len(table) == 1

    def test_records_are_plain_values(self):
        table = RoiTraceTable()
        table.add(
            RoiTrace(
                uid=3,
                z=1,
                c=2,
                engine="masknmf",
                F=np.ones(5),
                frames=(0, 5, 1),
                fs=10.0,
            )
        )
        (row,) = table.records()
        assert row == {
            "uid": 3,
            "z": 1,
            "c": 2,
            "engine": "masknmf",
            "source": "quick",
            "member": None,
            "frames": [0, 5, 1],
            "frame_average": 1,
            "fs": 10.0,
            "label": "",
            "n_frames": 5,
            "path": None,
        }
        assert ENGINES == ("mean", "suite2p", "masknmf")

    def test_remove_and_clear_emit(self):
        table = RoiTraceTable()
        seen = _Recorder()
        table.add_event_handler(seen)
        key = table.add(RoiTrace(uid=1, F=np.ones(2))).key
        assert table.remove(key) is not None and table.remove(key) is None
        table.add(RoiTrace(uid=1, F=np.ones(2)))
        table.clear()
        table.clear()  # nothing to clear, no event
        assert seen.actions == ["add", "remove", "add", "clear"]


class TestRoiModel:
    def test_view_folds_to_a_plane_and_emits_on_change(self, cz_store):
        model = RoiModel(cz_store)
        seen = _Recorder()
        model.add_event_handler(seen, "view")
        assert model.plane == 0 and (model.z, model.c) == (0, 0)
        assert model.set_view({"t": 3, "c": 1, "z": 2}) is True
        assert model.plane == 5 and (model.z, model.c) == (2, 1)
        assert (
            model.set_view({"t": 4, "c": 1, "z": 2}) is False
        )  # t is not a plane axis
        assert [e.info for e in seen.events] == [{"plane": 5, "previous": 0}]
        assert model.plane_label(5) == "c2·z3"

    def test_store_and_trace_events_are_forwarded(self, cz_store):
        model = RoiModel(cz_store)
        seen = _Recorder()
        model.add_event_handler(seen)
        cz_store.add_roi(0, disk(16, 16, 8, 8, 3))
        model.traces.add(RoiTrace(uid=1, F=np.ones(2)))
        assert [(e.type, e.info["action"]) for e in seen.events] == [
            ("rois", "add"),
            ("traces", "add"),
        ]

    def test_swapping_the_store_rewires_the_forwarding(self, cz_store):
        model = RoiModel(cz_store)
        seen = _Recorder()
        model.add_event_handler(seen, "rois")
        model.set_view({"c": 1, "z": 1})
        other = RoiLabelStore(3, 16, 16, min_pixels=1)
        other.plane_axes = (("z", 3),)
        model.store = other
        assert model.plane == 1  # re-derived from the same view
        cz_store.add_roi(0, disk(16, 16, 8, 8, 3))
        other.add_roi(0, disk(16, 16, 8, 8, 3))
        assert len(seen.events) == 1 and seen.events[0].source is model

    def test_targets_read_where_drawn_unless_told_otherwise(self, cz_store):
        model = RoiModel(cz_store)
        cz_store.add_roi(5, disk(16, 16, 8, 8, 3))  # c1, z2
        cz_store.add_roi(0, disk(16, 16, 8, 8, 3))  # c0, z0
        assert model.targets([0, 1, 9]) == [
            RunTarget(index=0, uid=1, plane=5, z=2, c=1),
            RunTarget(index=1, uid=2, plane=0, z=0, c=0),
        ]
        # the same masks read on the other channel keep their planes
        assert model.targets([0], c=0) == [RunTarget(index=0, uid=1, plane=5, z=2, c=0)]
        assert model.targets([1], z=1, c=1) == [
            RunTarget(index=1, uid=2, plane=0, z=1, c=1)
        ]

    def test_traced_asks_the_table_by_roi(self, cz_store):
        model = RoiModel(cz_store)
        cz_store.add_roi(5, disk(16, 16, 8, 8, 3))
        uid = cz_store.rois[0].uid
        model.traces.add(RoiTrace(uid=uid, z=2, c=1, F=np.ones(2)))
        model.traces.add(RoiTrace(uid=uid, z=2, c=0, F=np.ones(2)))
        assert len(model.traced(0)) == 2
        assert [t.c for t in model.traced(0, c=0)] == [0]
        assert model.traced(0, engine="masknmf") == []
        assert model.traced(4) == []
        assert model.rois_on_screen() == []
        model.set_view({"c": 1, "z": 2})
        assert model.rois_on_screen() == [0]


class TestAxisRoles:
    def test_roles_name_the_axes_when_the_labels_do_not(self):
        store = RoiLabelStore(6, 8, 8, min_pixels=1)
        store.plane_axes = (("Channel", 2), ("ROI", 3))
        assert store.axis_name("z") is None  # "ROI" is nobody's alias
        store.axis_roles = {"z": "ROI", "c": "Channel"}
        assert store.axis_name("z") == "ROI" and store.axis_name("c") == "Channel"
        store.add_roi(5, disk(8, 8, 4, 4, 2))
        assert (store.roi_z(0), store.roi_c(0)) == (2, 1)
        # a role naming an axis the volume is not keyed by falls back to aliases
        store.axis_roles = {"z": "depth"}
        assert store.axis_name("z") is None
        assert store.snapshot().axis_roles == {"z": "depth"}


class TestColorize:
    def _three(self):
        store = RoiLabelStore(6, 16, 16, min_pixels=1)
        store.plane_axes = (("c", 2), ("z", 3))
        store.add_label_name("soma")
        store.add_roi(0, disk(16, 16, 4, 4, 2))  # c0 z0
        store.add_roi(4, disk(16, 16, 8, 8, 3))  # c1 z1
        store.add_roi(5, disk(16, 16, 12, 12, 1))  # c1 z2
        store.set_class(1, 0)
        return store

    def test_columns_are_per_uid_numbers(self):
        store = self._three()
        model = RoiModel(store)
        assert model.column("z") == {1: 0.0, 2: 1.0, 3: 2.0}
        assert model.column("c") == {1: 0.0, 2: 1.0, 3: 1.0}
        assert model.column("plane") == {1: 0.0, 2: 4.0, 3: 5.0}
        assert model.column("area") == {r.uid: float(r.area) for r in store.rois}
        assert model.column("class") == {2: 0.0}, "unlabeled ROIs have no class value"
        with pytest.raises(KeyError):
            model.column("nope")
        assert "z" in COLUMNS

    def test_continuous_values_map_through_the_colormap(self):
        store = self._three()
        model = RoiModel(store)
        seen = _Recorder()
        store.add_event_handler(seen, "rois")
        before = [store.roi_rgb(i) for i in range(3)]
        tint = model.colorize(model.column("area"), cmap="viridis")
        assert set(tint) == {1, 2, 3}
        smallest = min(store.rois, key=lambda r: r.area).uid
        largest = max(store.rois, key=lambda r: r.area).uid
        assert tint[smallest] == (68, 1, 84), "viridis starts dark purple"
        assert tint[largest] == (253, 231, 37), "and ends yellow"
        assert [store.roi_rgb(i) for i in range(3)] == [tint[r.uid] for r in store.rois]
        assert seen.actions == ["tint"]
        # a tint beats the class color and the explicit group color
        store.set_color(0, (9, 9, 9))
        assert store.roi_rgb(0) == tint[1]
        # clearing brings every other color back
        assert model.colorize(None) == {}
        store.set_color(0, None)
        assert [store.roi_rgb(i) for i in range(3)] == before

    def test_categorical_values_get_one_color_per_level(self):
        store = self._three()
        model = RoiModel(store)
        tint = model.colorize(model.column("c"), cmap="tab10", categorical=True)
        assert tint[2] == tint[3] != tint[1]
        assert tint[1] == (31, 119, 180) and tint[2] == (255, 127, 14)
        # rois left out of the values keep their own color
        only = model.colorize({2: 1.0}, cmap="viridis")
        assert set(only) == {2}
        assert store.roi_rgb(0) != only[2]
        # equal values do not divide by zero
        flat = model.colorize({1: 3.0, 2: 3.0})
        assert flat[1] == flat[2]
        # one level, one ROI: the colormap's single row is still a row
        assert model.colorize({1: 0.0}, cmap="tab10", categorical=True) == {
            1: (31, 119, 180)
        }
        assert model.colorize({2: 5.0}, cmap="viridis") == {2: (68, 1, 84)}

    def test_full_image_rows_take_string_members(self):
        table = RoiTraceTable()
        row = table.add(
            RoiTrace(uid=0, source=FULL_IMAGE, member="z2c1", z=2, c=1, F=np.ones(3))
        )
        assert row.key == ("member", FULL_IMAGE, "z2c1") and not row.stands_for_roi
        table.add(
            RoiTrace(uid=0, source=FULL_IMAGE, member="z2c1", z=2, c=1, F=np.zeros(3))
        )
        assert len(table) == 1 and table.get(row.key).F.sum() == 0
        assert table.prune([]) == [] and len(table) == 1
