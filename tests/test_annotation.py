"""Tests for the GUI-free annotation model and the rois zarr.

``mbo_utilities.annotation`` is the half of the manual ROI tool that has no
GUI imports (see its README); these tests need no canvas.
"""

from __future__ import annotations

import numpy as np
import pytest
from mbo_utilities.annotation import (
    UNLABELED,
    RoiFile,
    RoiLabelStore,
    RoiRecord,
    RoiTrace,
    class_color,
)


def disk(ny, nx, cy, cx, r):
    yy, xx = np.mgrid[:ny, :nx]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


@pytest.fixture
def store():
    return RoiLabelStore(3, 32, 32, min_pixels=4)


class TestStore:
    def test_add_roi_keeps_its_pixels_on_its_plane(self, store):
        index = store.add_roi(1, disk(32, 32, 10, 10, 3))
        assert index == 0
        assert store.rois[0].plane == 1
        assert store.label_image(1)[10, 10] == 1
        assert store.label_image(0).max() == 0 and store.label_image(2).max() == 0
        assert store.counts == [store.rois[0].area]
        assert np.array_equal(store.mask(0), disk(32, 32, 10, 10, 3))
        ypix, xpix, weights = store.footprint(0)
        assert ypix.size == store.rois[0].area and np.all(weights == 1.0)

    def test_overlapping_rois_both_keep_every_pixel(self, store):
        store.add_roi(0, disk(32, 32, 10, 10, 4))
        store.add_roi(0, disk(32, 32, 13, 10, 4))
        assert (
            store.rois[0].area
            == store.rois[1].area
            == int(disk(32, 32, 10, 10, 4).sum())
        )
        shared = disk(32, 32, 10, 10, 4) & disk(32, 32, 13, 10, 4)
        assert shared.any()
        # the label image gives a shared pixel to the smaller roi (a tie: the later one)
        image = store.label_image(0)
        assert set(np.unique(image[shared])) <= {1, 2}
        assert image[10, 7] == 1 and image[13, 13] == 2

    def test_smaller_roi_wins_the_label_image(self, store):
        store.add_roi(0, disk(32, 32, 10, 10, 6))
        store.add_roi(0, disk(32, 32, 10, 10, 2))
        image = store.label_image(0)
        assert image[10, 10] == 2 and image[10, 15] == 1
        assert store.roi_at(0, 10, 10) == 1 and store.roi_at(0, 10, 15) == 0

    def test_same_pixels_on_other_plane(self, store):
        store.add_roi(0, disk(32, 32, 10, 10, 4))
        index = store.add_roi(1, disk(32, 32, 10, 10, 4))
        assert store.rois[index].area == store.rois[0].area

    def test_too_small_returns_none(self, store):
        assert store.add_roi(0, np.zeros((32, 32), bool)) is None
        one_px = np.zeros((32, 32), bool)
        one_px[5, 5] = True
        assert store.add_roi(0, one_px) is None
        assert store.rois == []

    def test_pixels_and_weights(self, store):
        pixels = np.array([5 * 32 + 7, 3 * 32 + 1, 5 * 32 + 7, 9 * 32 + 9, 0])
        weights = np.array([0.5, 0.25, 0.5, 1.0, 2.0])
        index = store.add_roi(2, pixels=pixels, weights=weights, source="run:3")
        record = store.rois[index]
        assert record.pixels.tolist() == [0, 3 * 32 + 1, 5 * 32 + 7, 9 * 32 + 9]
        assert record.weights.tolist() == [2.0, 0.25, 0.5, 1.0]
        assert record.source == "run:3" and record.area == 4
        ypix, xpix, w = store.footprint(index)
        assert ypix.tolist() == [0, 3, 5, 9] and xpix.tolist() == [0, 1, 7, 9]
        assert w.tolist() == [2.0, 0.25, 0.5, 1.0]

    def test_polygon_rides_along(self, store):
        poly = [(8.5, 9.0), (12.0, 9.0), (12.0, 13.0), (8.5, 13.0)]
        index = store.add_roi(0, disk(32, 32, 10, 10, 3), polygon=poly)
        assert store.rois[index].polygon.shape == (4, 2)
        assert store.rois[index].polygon.dtype == np.float32
        assert store.rois[store.add_roi(0, disk(32, 32, 20, 20, 3))].polygon is None

    def test_delete_keeps_the_rest(self, store):
        for i in range(3):
            store.add_roi(i, disk(32, 32, 8 + 5 * i, 8, 3))
        assert store.delete_roi(0)
        assert len(store.rois) == 2
        assert store.label_image(0).max() == 0
        # the old ROI 1 (on plane 1) is now index 0, label 1
        assert store.label_image(1).max() == 1 and store.rois[0].plane == 1
        assert store.roi_at(1, 13, 8) == 0

    def test_delete_out_of_range(self, store):
        assert not store.delete_roi(0)
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        assert not store.delete_roi(-1)
        assert not store.delete_roi(5)
        assert len(store.rois) == 1

    def test_clear(self, store):
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        store.clear()
        assert store.rois == [] and store.label_image(0).max() == 0

    def test_roi_at(self, store):
        store.add_roi(1, disk(32, 32, 10, 10, 3))
        assert store.roi_at(1, 10, 10) == 0
        assert store.roi_at(0, 10, 10) == -1
        assert store.roi_at(1, 0, 0) == -1
        assert store.roi_at(9, 10, 10) == -1

    def test_label_names_and_classes(self, store):
        assert store.add_label_name("soma") == 0
        assert store.add_label_name("dendrite") == 1
        assert store.add_label_name("soma") == 0  # no duplicate
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        store.add_roi(0, disk(32, 32, 20, 20, 3))
        store.set_class(0, 1)
        assert store.class_counts() == [0, 1]
        store.set_class(0, UNLABELED)
        assert store.class_counts() == [0, 0]
        with pytest.raises(IndexError):
            store.set_class(0, 2)
        with pytest.raises(ValueError):
            store.add_label_name("  ")

    def test_add_roi_assigns_uids_and_source(self, store):
        assert store.next_uid == 1
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        store.add_roi(1, disk(32, 32, 20, 20, 3), source="rois_a:2")
        assert [r.uid for r in store.rois] == [1, 2]
        assert [r.source for r in store.rois] == ["", "rois_a:2"]
        assert store.next_uid == 3
        assert store.uid_index(2) == 1
        assert store.uid_index(99) is None

    def test_delete_never_reuses_uids(self, store):
        for i in range(3):
            store.add_roi(0, disk(32, 32, 8 + 5 * i, 8, 3))
        store.delete_roi(2)  # holds uid 3, the max so far
        assert store.next_uid == 4
        index = store.add_roi(0, disk(32, 32, 8, 24, 3))
        assert store.rois[index].uid == 4
        assert store.uid_index(3) is None

    def test_adopted_records_get_fresh_uids(self):
        rois = [
            RoiRecord(plane=0, pixels=[0], weights=None, uid=5),
            RoiRecord(plane=0, pixels=[1], weights=None),
            RoiRecord(plane=0, pixels=[2], weights=None, uid=5),
        ]
        store = RoiLabelStore(1, 8, 8, rois=rois)
        assert [r.uid for r in store.rois] == [5, 6, 7]
        assert store.next_uid == 8
        assert store.label_image(0)[0, :3].tolist() == [1, 2, 3]

    def test_snapshot_isolation(self, store):
        store.add_label_name("soma")
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        snap = store.snapshot()
        store.add_roi(1, disk(32, 32, 20, 20, 3))
        store.set_class(0, 0)
        store.set_note(0, "changed")
        assert len(snap.rois) == 1
        assert snap.rois[0].class_index == UNLABELED
        assert snap.rois[0].note == ""
        assert snap.label_image(1).max() == 0
        assert snap.next_uid == 2 and store.next_uid == 3
        assert snap.label_names == ("soma",) and snap.min_pixels == 4
        snap.add_roi(2, disk(32, 32, 5, 5, 2))
        assert store.label_image(2).max() == 0

    def test_restore_takes_a_snapshot_back_in_place(self, store):
        seen = []
        store.add_event_handler(lambda ev: seen.append(ev.info["action"]), "rois")
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        snap = store.snapshot()
        store.add_roi(1, disk(32, 32, 20, 20, 3))
        store.delete_roi(0)
        store.restore(snap)
        assert [r.uid for r in store.rois] == [1]
        assert store.label_image(1).max() == 0 and store.label_image(0)[10, 10] == 1
        assert store.next_uid == 3  # never goes back down
        assert seen[-1] == "restore"
        with pytest.raises(ValueError):
            store.restore(RoiLabelStore(1, 4, 4))

    def test_colors_follow_class(self, store):
        store.add_label_name("soma")
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        hue = store.roi_rgb(0)
        store.set_class(0, 0)
        expected = tuple(int(round(c * 255)) for c in class_color(0))
        assert store.roi_rgb(0) == expected
        assert store.roi_rgb(0) != hue
        lut = store.color_lut()
        assert tuple(lut[1]) == expected and tuple(lut[0]) == (0, 0, 0)


class TestRoiFile:
    def _populated(self):
        store = RoiLabelStore(3, 32, 32, label_names=("soma", "dendrite"))
        store.add_roi(0, disk(32, 32, 10, 10, 4), polygon=[(6, 10), (14, 10), (10, 14)])
        store.add_roi(2, disk(32, 32, 20, 20, 3))
        store.set_class(0, 1)
        store.set_note(1, "faint, recheck")
        return store

    def test_round_trip(self, tmp_path):
        store = self._populated()
        path = tmp_path / "movie.rois.zarr"
        RoiFile(path).save(store, source={"path": "/data/movie.tif"})
        got = RoiFile.load(path)
        restored = got.store
        assert restored.shape == store.shape
        assert restored.label_names == ("soma", "dendrite")
        assert [r.plane for r in restored.rois] == [0, 2]
        assert [r.area for r in restored.rois] == [r.area for r in store.rois]
        for i in range(2):
            assert np.array_equal(restored.rois[i].pixels, store.rois[i].pixels)
            assert np.array_equal(restored.rois[i].weights, store.rois[i].weights)
        assert restored.rois[0].class_index == 1
        assert restored.rois[1].class_index == UNLABELED
        assert restored.rois[1].note == "faint, recheck"
        assert restored.rois[0].polygon.tolist() == [[6, 10], [14, 10], [10, 14]]
        assert restored.rois[1].polygon is None
        assert got.source == {"path": "/data/movie.tif"}
        assert got.traces == [] and got.runs == []

    def test_overlaps_and_weights_survive(self, tmp_path):
        store = RoiLabelStore(1, 16, 16)
        store.add_roi(0, disk(16, 16, 8, 8, 5))
        store.add_roi(
            0, pixels=[8 * 16 + 8, 8 * 16 + 9], weights=[0.5, 2.0], source="run:1"
        )
        path = tmp_path / "movie.rois.zarr"
        RoiFile(path).save(store)
        restored = RoiFile.load(path).store
        assert restored.rois[0].area == store.rois[0].area
        assert restored.rois[1].pixels.tolist() == [8 * 16 + 8, 8 * 16 + 9]
        assert restored.rois[1].weights.tolist() == [0.5, 2.0]
        assert restored.rois[1].source == "run:1"
        assert restored.label_image(0)[8, 8] == 2 and restored.label_image(0)[8, 4] == 1

    def test_save_overwrites_whole_file(self, tmp_path):
        store = self._populated()
        path = tmp_path / "movie.rois.zarr"
        writer = RoiFile(path)
        writer.save(store)
        store.add_roi(1, disk(32, 32, 16, 16, 3))
        store.set_note(2, "new one")
        store.delete_roi(0)
        writer.save(store)
        restored = RoiFile.load(path).store
        assert [r.plane for r in restored.rois] == [2, 1]
        assert restored.rois[1].note == "new one"
        assert not path.with_name(path.name + ".tmp").exists()
        assert not path.with_name(path.name + ".old").exists()

    def test_color_and_plane_axes_round_trip(self, tmp_path):
        import zarr

        store = RoiLabelStore(6, 32, 32, label_names=("soma",))
        store.plane_axes = (("c", 2), ("z", 3))
        store.add_roi(store.plane_of({"c": 1, "z": 2}), disk(32, 32, 10, 10, 4))
        store.add_roi(0, disk(32, 32, 20, 20, 3))
        store.set_color(0, (255, 64, 0))
        store.set_class(0, 0)
        assert store.roi_rgb(0) == (255, 64, 0)  # wins over the class color
        path = tmp_path / "movie.rois.zarr"
        RoiFile(path).save(store)
        restored = RoiFile.load(path).store
        assert restored.rois[0].color == (255, 64, 0)
        assert restored.rois[1].color is None
        assert restored.roi_rgb(0) == (255, 64, 0)
        assert restored.plane_axes == (("c", 2), ("z", 3))
        assert restored.roi_z(0) == 2 and restored.roi_c(0) == 1
        store.set_color(0, None)
        assert store.roi_rgb(0) != (255, 64, 0)
        drawn = zarr.open_group(str(path), mode="r")["labels/drawn"]
        (scales,) = drawn.attrs["ome"]["multiscales"]
        assert [a["name"] for a in scales["axes"]] == ["c", "z", "y", "x"]
        assert drawn["0"].shape == (2, 3, 32, 32) and drawn["0"][1, 2][10, 10] == 1

    def test_uid_source_round_trip(self, tmp_path):
        store = self._populated()
        store.add_roi(1, disk(32, 32, 16, 16, 3), source="rois_a:4")
        path = tmp_path / "movie.rois.zarr"
        RoiFile(path).save(store)
        restored = RoiFile.load(path).store
        assert [r.uid for r in restored.rois] == [1, 2, 3]
        assert [r.source for r in restored.rois] == ["", "", "rois_a:4"]
        assert restored.next_uid == store.next_uid == 4

    def test_next_uid_survives_deleting_the_max_uid_roi(self, tmp_path):
        store = self._populated()  # uids 1, 2; next_uid 3
        store.delete_roi(1)
        path = tmp_path / "movie.rois.zarr"
        RoiFile(path).save(store)
        restored = RoiFile.load(path).store
        assert restored.next_uid == 3
        index = restored.add_roi(1, disk(32, 32, 16, 16, 3))
        assert restored.rois[index].uid == 3  # uid 2 is never reused

    def test_traces_and_runs_round_trip(self, tmp_path):
        store = self._populated()
        traces = [
            RoiTrace(
                uid=1,
                z=0,
                c=0,
                engine="mean",
                source="quick",
                F=np.arange(5, dtype=np.float32),
                Fneu=np.ones(5, np.float32),
                frames=(0, 5, 1),
                fs=10.0,
            ),
            RoiTrace(
                uid=0,
                z=2,
                c=1,
                engine="mean",
                source="pixel",
                member="pixel 3,4",
                F=np.zeros(3, np.float32),
                extra={"pixel": (3, 4)},
            ),
            RoiTrace(uid=2, z=2, c=0, engine="suite2p", source="run_a", F=np.ones(5)),
        ]
        runs = [
            {
                "path": tmp_path / "movie.a.2026-10-28-10-00-00.manual.zarr",
                "discarded": {3, 1},
                "classes": {2: 0},
                "colors": {1: (1.0, 0.5, 0.0)},
                "iscell": {4: False},
            }
        ]
        path = tmp_path / "movie.rois.zarr"
        RoiFile(path).save(store, traces, runs=runs)
        got = RoiFile.load(path)
        # the run's row is not the widget's to keep
        assert [t.source for t in got.traces] == ["quick", "pixel"]
        quick, pixel = got.traces
        assert quick.key == traces[0].key
        assert quick.F.tolist() == [0, 1, 2, 3, 4] and quick.Fneu.tolist() == [1] * 5
        assert quick.frames == (0, 5, 1) and quick.fs == 10.0
        assert pixel.key == traces[1].key and pixel.Fneu is None
        assert pixel.extra["pixel"] == (3, 4)
        assert got.runs == [
            {
                "path": str(runs[0]["path"]),
                "discarded": [1, 3],
                "classes": {"2": 0},
                "colors": {"1": [1.0, 0.5, 0.0]},
                "iscell": {"4": False},
            }
        ]

    def test_ngff_layout(self, tmp_path):
        import zarr

        store = self._populated()
        store.plane_axes = (("z", 3),)
        path = tmp_path / "movie.rois.zarr"
        RoiFile(path).save(
            store, source={"path": str(tmp_path / "movie.zarr")}, scale=(5.0, 1.5, 1.5)
        )
        root = zarr.open_group(str(path), mode="r")
        assert root.attrs["mbo_rois"] == 1
        assert root.attrs["shape"] == [3, 32, 32]
        assert root.attrs["label_names"] == ["soma", "dendrite"]
        assert root.attrs["source"]["path"].endswith("movie.zarr")
        rois = root["rois"]
        assert rois["offsets"][:].tolist() == [
            0,
            store.rois[0].area,
            store.rois[0].area + store.rois[1].area,
        ]
        assert rois["uid"][:].tolist() == [1, 2]
        assert rois["plane"][:].tolist() == [0, 2]
        assert rois.attrs["note"] == ["", "faint, recheck"]
        labels = root["labels"]
        assert labels.attrs["ome"] == {"version": "0.5", "labels": ["drawn"]}
        drawn = labels["drawn"]
        ome = drawn.attrs["ome"]
        assert ome["version"] == "0.5"
        (scales,) = ome["multiscales"]
        assert [a["name"] for a in scales["axes"]] == ["z", "y", "x"]
        assert scales["datasets"][0]["coordinateTransformations"] == [
            {"type": "scale", "scale": [5.0, 1.5, 1.5]}
        ]
        label = ome["image-label"]
        assert label["source"]["image"].endswith("movie.zarr")
        assert [p["label-value"] for p in label["properties"]] == [1, 2]
        assert label["properties"][0]["class"] == "dendrite"
        assert len(label["colors"]) == 2 and label["colors"][0]["label-value"] == 1
        arr = drawn["0"]
        assert arr.dtype == np.uint32 and arr.shape == (3, 32, 32)
        assert arr[0][10, 10] == 1 and arr[2][20, 20] == 2

    def test_load_errors(self, tmp_path):
        import zarr

        with pytest.raises(FileNotFoundError):
            RoiFile.load(tmp_path / "missing.zarr")
        path = tmp_path / "flat.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.attrs["version"] = "0.5"
        with pytest.raises(ValueError):
            RoiFile.load(path)
