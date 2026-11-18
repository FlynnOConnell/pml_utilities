"""The trace zarr: round trip, the layout another package reads with plain zarr, and the derived label image."""

import numpy as np
import pytest
import zarr

from mbo_utilities.annotation.neurite_trace import TRACE_VERSION, Trace, read_trace, write_trace

SHAPE = (6, 40, 50)
VOXEL_UM = (3.0, 0.5, 0.5)
SOURCE = {"path": "/data/stack.mesc", "reader_kwargs": {"unit": "MSession_0/MUnit_1"}, "unit": "MSession_0/MUnit_1"}


def two_paths_and_a_link() -> Trace:
    """A skeleton-like path on plane 1, an accepted link (two far ends) on plane 4, and one proposal."""
    along_row = np.array([[1, 10, x] for x in range(5, 25)], np.float32)
    accepted_link = np.array([[4, 5, 5], [4, 35, 45]], np.float32)
    links = np.array([[[1, 10, 24], [1, 10, 30]]], np.float32)
    return Trace(paths={3: along_row, 7: accepted_link}, links=links, voxel_um=VOXEL_UM, channel=1)


def test_a_trace_reads_back_as_written_with_ids_in_file_order(tmp_path):
    trace = two_paths_and_a_link()
    write_trace(trace, tmp_path / "stack.trace.zarr", shape=SHAPE, source=SOURCE)
    loaded = read_trace(tmp_path / "stack.trace.zarr")
    assert list(loaded.paths) == [0, 1]
    for points, points_loaded in zip(trace.paths.values(), loaded.paths.values()):
        np.testing.assert_array_equal(points_loaded, points)
    np.testing.assert_array_equal(loaded.links, trace.links)
    assert loaded.voxel_um == VOXEL_UM and loaded.channel == 1


def test_plain_zarr_reads_the_paths_links_and_attrs(tmp_path):
    trace = two_paths_and_a_link()
    write_trace(trace, tmp_path / "stack.trace.zarr", shape=SHAPE, source=SOURCE)
    root = zarr.open_group(str(tmp_path / "stack.trace.zarr"), mode="r")
    offsets, points = root["paths/offsets"][:], root["paths/points"][:]
    assert offsets.tolist() == [0, 20, 22] and points.shape == (22, 3) and points.dtype == np.float32
    np.testing.assert_array_equal(points[offsets[1] : offsets[2]], trace.paths[7])
    assert root["links"].shape == (1, 2, 3)
    attrs = dict(root.attrs)
    assert attrs["mbo_trace"] == TRACE_VERSION and attrs["shape"] == list(SHAPE)
    assert attrs["voxel_um"] == list(VOXEL_UM) and attrs["channel"] == 1 and attrs["source"] == SOURCE


def test_the_label_image_is_ngff_with_each_path_drawn_as_its_index_plus_one(tmp_path):
    write_trace(two_paths_and_a_link(), tmp_path / "stack.trace.zarr", shape=SHAPE)
    root = zarr.open_group(str(tmp_path / "stack.trace.zarr"), mode="r")
    assert root["labels"].attrs["ome"]["labels"] == ["paths"]
    ome = root["labels/paths"].attrs["ome"]
    multiscale = ome["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == ["z", "y", "x"]
    assert multiscale["datasets"][0]["coordinateTransformations"][0]["scale"] == list(VOXEL_UM)
    assert "image-label" in ome
    volume = root["labels/paths/0"][:]
    assert volume.shape == SHAPE and volume.dtype == np.uint32
    assert np.all(volume[1, 10, 5:25] == 1)
    assert volume[4, 5, 5] == 2 and volume[4, 35, 45] == 2 and volume[4, 20, 25] == 2
    assert np.count_nonzero(volume == 2) == 41  ## the line between the two ends, one voxel per step along x


def test_an_empty_trace_writes_and_reads_empty(tmp_path):
    write_trace(Trace(), tmp_path / "empty.trace.zarr", shape=SHAPE)
    loaded = read_trace(tmp_path / "empty.trace.zarr")
    assert loaded.paths == {} and loaded.links.shape == (0, 2, 3)
    assert not zarr.open_group(str(tmp_path / "empty.trace.zarr"), mode="r")["labels/paths/0"][:].any()


def test_a_second_write_replaces_the_first(tmp_path):
    write_trace(two_paths_and_a_link(), tmp_path / "stack.trace.zarr", shape=SHAPE)
    write_trace(Trace(voxel_um=VOXEL_UM), tmp_path / "stack.trace.zarr", shape=SHAPE)
    assert read_trace(tmp_path / "stack.trace.zarr").paths == {}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stack.trace.zarr"]


def test_a_zarr_that_is_not_a_trace_is_refused(tmp_path):
    zarr.open_group(str(tmp_path / "other.zarr"), mode="w")
    with pytest.raises(ValueError, match="not a trace zarr"):
        read_trace(tmp_path / "other.zarr")
