"""The ROI store's exchange with Fiji and plain files (``annotation/imagej``)."""

from __future__ import annotations

import csv

import numpy as np
import roifile
import tifffile
from mbo_utilities.annotation import RoiLabelStore, RoiTrace
from mbo_utilities.annotation.imagej import (
    export_footprints_npz,
    export_label_tiff,
    export_roiset,
    export_traces,
    import_roiset,
    mask_outline,
)


def disk(ny, nx, cy, cx, r):
    yy, xx = np.mgrid[:ny, :nx]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def test_roiset_round_trip_on_a_2d_image(tmp_path):
    store = RoiLabelStore(1, 32, 32, label_names=("soma",))
    square = [(5.0, 5.0), (12.0, 5.0), (12.0, 12.0), (5.0, 12.0)]
    store.add_roi(0, disk(32, 32, 20, 20, 4), polygon=square)
    store.add_roi(0, disk(32, 32, 20, 20, 4))  # a promoted mask: no polygon
    store.set_class(0, 0)
    path = export_roiset(store, tmp_path / "RoiSet.zip")
    rois = roifile.roiread(path)
    assert [r.name for r in rois] == ["0001-soma", "0002"]
    assert rois[0].position == 1 and rois[0].z_position == 1
    np.testing.assert_array_equal(rois[0].coordinates(), square)
    back = RoiLabelStore(1, 32, 32)
    assert import_roiset(back, path) == [0, 1]
    assert back.rois[0].note == "0001-soma" and back.rois[0].source == "imagej"
    assert np.array_equal(back.rois[0].polygon, np.asarray(square, np.float32))
    # a filled polygon on integer corners covers the square inclusive of its edge
    assert back.mask(0)[5:13, 5:13].all() and back.rois[0].area == 64
    # the outline round trip keeps the disc
    agree = (back.mask(1) & store.mask(1)).sum() / store.mask(1).sum()
    assert agree > 0.9


def test_roiset_round_trip_on_a_stack(tmp_path):
    store = RoiLabelStore(6, 32, 32)
    store.plane_axes = (("c", 2), ("z", 3))
    plane = store.plane_of({"c": 1, "z": 2})
    store.add_roi(plane, disk(32, 32, 10, 10, 4), polygon=[(6, 10), (14, 10), (10, 14)])
    path = export_roiset(store, tmp_path / "RoiSet.zip")
    (roi,) = (
        [roifile.roiread(path)]
        if not isinstance(roifile.roiread(path), list)
        else roifile.roiread(path)
    )
    assert roi.z_position == 3 and roi.c_position == 2 and roi.t_position == 0
    back = RoiLabelStore(6, 32, 32)
    back.plane_axes = (("c", 2), ("z", 3))
    import_roiset(back, path)
    assert back.rois[0].plane == plane
    assert back.roi_z(0) == 2 and back.roi_c(0) == 1


def test_mask_outline_traces_the_mask():
    mask = np.zeros((16, 16), bool)
    mask[3:9, 4:12] = True
    points = mask_outline(mask)
    assert points.shape[1] == 2 and points.dtype == np.float32
    assert points[:, 0].min() == 4 and points[:, 0].max() == 11
    assert points[:, 1].min() == 3 and points[:, 1].max() == 8


def test_label_tiff_is_an_imagej_hyperstack(tmp_path):
    store = RoiLabelStore(6, 16, 16)
    store.plane_axes = (("c", 2), ("z", 3))
    store.add_roi(store.plane_of({"c": 1, "z": 2}), disk(16, 16, 8, 8, 3))
    store.add_roi(0, disk(16, 16, 4, 4, 2))
    path = export_label_tiff(store, tmp_path / "labels.tif", scale=(5.0, 1.5, 1.5))
    with tifffile.TiffFile(path) as tf:
        meta = tf.imagej_metadata
        series = tf.series[0]
        data = series.asarray()
    assert meta["hyperstack"] and meta["slices"] == 3 and meta["channels"] == 2
    assert meta["spacing"] == 5.0 and meta["unit"] == "um"
    assert series.axes == "ZCYX" and data.shape == (3, 2, 16, 16)
    assert data.dtype == np.uint16
    assert data[2, 1][8, 8] == 1 and data[0, 0][4, 4] == 2 and data[1, 0].max() == 0


def test_traces_csv_has_one_column_per_row(tmp_path):
    rows = [
        RoiTrace(
            uid=1, z=0, c=0, engine="mean", source="quick", F=np.arange(4.0), fs=2.0
        ),
        RoiTrace(
            uid=2,
            z=0,
            c=0,
            engine="mean",
            source="quick",
            F=np.array([7.0, 8.0]),
            frames=(2, 4, 1),
            fs=2.0,
            label="cell b",
        ),
    ]
    path = export_traces(rows, tmp_path / "traces.csv")
    with open(path) as f:
        got = list(csv.reader(f))
    assert got[0] == ["time_s", "quick None", "cell b"]
    assert got[1] == ["0", "0", ""] and got[3] == ["1", "2", "7"]
    assert len(got) == 5
    rows[1].fs = 3.0
    path = export_traces(rows, tmp_path / "frames.csv")
    with open(path) as f:
        assert next(csv.reader(f))[0] == "frame"


def test_footprints_npz_is_masknmfs_seed_file(tmp_path):
    store = RoiLabelStore(2, 8, 8)
    store.add_roi(1, pixels=[9, 10], weights=[0.5, 2.0])
    store.add_roi(0, pixels=[0])
    got = np.load(export_footprints_npz(store, tmp_path / "footprints.npz"))
    fp = got["spatial_footprints"]
    assert fp.shape == (8, 8, 2) and fp.dtype == np.float32
    assert fp[1, 1, 0] == 0.5 and fp[1, 2, 0] == 2.0 and fp[0, 0, 1] == 1.0
    assert got["plane"].tolist() == [1, 0] and got["uid"].tolist() == [1, 2]
