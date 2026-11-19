"""The stack annotator offscreen, on a small synthetic stack: build, paint, fill, trace, undo."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import skimage.draw
from click.testing import CliRunner

pytest.importorskip("skan")
pytest.importorskip("fastplotlib")

from fastplotlib.widgets.nd_widget._async import run_sync

from mbo_utilities.cli import main
from mbo_utilities.gui import DataVis, stack_annotate
from mbo_utilities.gui.manual_roi import attach_roi_widget

N_Z, N_Y, N_X = 8, 64, 64
VOXEL_UM = (2.0, 0.5, 0.5)


def render(state) -> np.ndarray:
    """Draw the figure, imgui panels included; the NDWidget fetches on the event loop, which nothing runs here."""
    for nd_image in state.nd_images.values():
        run_sync(nd_image._set_indices_())
    for _ in range(2):
        frame = np.asarray(state.viewer.figure.canvas.draw())
    return frame


@pytest.fixture
def two_stacks_mesc(tmp_path):
    """A .mesc with two Z-stacks (MUnit_0, MUnit_2) and a time series between them, two channels each."""
    path = tmp_path / "two_stacks.mesc"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        session = f.create_group("MSession_0")
        for k, (method_type, shape) in enumerate([(2, (12, 64, 48)), (1, (30, 64, 48)), (2, (8, 40, 40))]):
            unit = session.create_group(f"MUnit_{k}")
            unit.attrs.update(
                {"MethodType": method_type, "VecChannelsSize": 2, "TStepInMs": 33.0, "MeasurementDatePosix": 1, "Comment": f"unit {k}"}
            )
            for c in range(2):
                unit.create_dataset(f"Channel_{c}", data=rng.integers(0, 4000, shape).astype(np.uint16))
    return path


@pytest.fixture
def state(tmp_path):
    rng = np.random.default_rng(0)
    volume = rng.normal(10.0, 1.0, (1, N_Z, N_Y, N_X)).astype(np.float32)  ## TZYX
    for z in range(N_Z):
        rr, cc = skimage.draw.line(10, 5 + 4 * z, 50, 20 + 4 * z)
        volume[0, z, rr, cc] = 100.0
    filepath_stack = tmp_path / "stack.npy"
    np.save(filepath_stack, volume)
    state = stack_annotate.build(filepath_stack=str(filepath_stack), voxel_um=VOXEL_UM)
    state.widget.store.add_label_name("dendrite")
    render(state)
    return state


def test_a_stack_opens_in_four_views_and_draws(state):
    assert state.arr.shape[-3:] == (N_Z, N_Y, N_X)
    assert set(state.nd_images) >= {"xz", "yz", "volume"}
    assert render(state).std() > 0


def test_a_brush_stroke_on_xy_is_one_roi_on_that_plane_and_undo_takes_it_back(state):
    widget, store = state.widget, state.widget.store
    state.viewer.indices[widget.zdim] = 3
    render(state)
    state.brush.active = True
    for point in ((20, 20), (22, 24), (24, 28)):
        stack_annotate.stamp(state=state, point=point)
    stack_annotate.commit_stroke(state=state)
    assert [store.roi_z(i) for i in range(len(store.rois))] == [3]
    widget.undo()
    assert len(store.rois) == 0


def test_fill_between_adds_one_roi_per_plane_between_two_in_the_first_ones_class(state):
    widget, store = state.widget, state.widget.store
    for z in (1, 5):
        mask = np.zeros((N_Y, N_X), bool)
        mask[20:30, 10 + 4 * z : 20 + 4 * z] = True
        store.set_class(store.add_roi(z, mask), 0)
    widget.buffer = [(-1, 0), (-1, 1)]
    stack_annotate.fill_between(state=state)
    assert sorted(store.roi_z(i) for i in range(len(store.rois))) == [1, 2, 3, 4, 5]
    assert {roi.class_index for roi in store.rois} == {0}
    render(state)


def test_one_undo_history_covers_rois_and_trace_edits(state):
    widget, store = state.widget, state.widget.store
    stack_annotate.start_trace(state=state)
    state.trace_job.result()
    render(state)
    layer = state.trace
    assert layer is not None and len(layer.trace.paths) > 0
    widget._push_undo()
    store.add_roi(2, np.pad(np.ones((4, 4), bool), ((10, N_Y - 14), (10, N_X - 14))))
    n_paths = len(layer.trace.paths)
    layer.picked_paths = {next(iter(layer.trace.paths))}
    stack_annotate.edit_trace(state=state, action="delete")
    assert len(layer.trace.paths) == n_paths - 1
    widget.undo()
    assert len(layer.trace.paths) == n_paths and len(store.rois) == 1
    widget.undo()
    assert len(store.rois) == 0


def test_a_saved_trace_sits_beside_the_rois_zarr_and_opens_with_the_stack(state):
    stack_annotate.start_trace(state=state)
    state.trace_job.result()
    render(state)
    stack_annotate.save_trace(state=state)
    filepath = Path(stack_annotate.filepath_trace(state=state))
    assert filepath.name == "stack.trace.zarr" and filepath.parent == Path(state.widget._save_target()).parent
    reopened = stack_annotate.build(filepath_stack=state.filepath_stack, voxel_um=VOXEL_UM)
    assert len(reopened.trace.trace.paths) == len(state.trace.trace.paths)
    assert reopened.trace.trace.channel == state.trace.trace.channel


def test_a_short_unit_name_opens_that_z_stack(two_stacks_mesc):
    state = stack_annotate.build(filepath_stack=str(two_stacks_mesc), unit="MUnit_2", voxel_um=VOXEL_UM)
    assert state.arr.unit_key == "MSession_0/MUnit_2" and state.arr.shape[-3:] == (8, 40, 40)


def test_a_unit_that_is_not_a_z_stack_is_refused_with_the_z_stacks_named(two_stacks_mesc):
    with pytest.raises(ValueError, match="MUnit_1 is not a Z-stack.*MSession_0/MUnit_0, MSession_0/MUnit_2"):
        stack_annotate.open_stack(filepath_stack=str(two_stacks_mesc), unit="MUnit_1", voxel_um=VOXEL_UM)


def test_the_cli_lists_the_z_stacks_of_a_file_with_several_and_opens_none(two_stacks_mesc):
    result = CliRunner().invoke(main, ["stack-annotate", str(two_stacks_mesc)])
    assert result.exit_code != 0
    assert "MUnit_0   2 x 12 x 64 x 48" in result.output and "MUnit_2   2 x 8 x 40 x 40" in result.output
    assert "MUnit_1" not in result.output and "--unit MUnit_2" in result.output


def test_rois_drawn_here_are_the_studios_rois_for_that_unit_and_back(two_stacks_mesc):
    state = stack_annotate.build(filepath_stack=str(two_stacks_mesc), unit="MUnit_2", voxel_um=VOXEL_UM)
    store = state.widget.store
    store.add_label_name("spine")
    mask = np.zeros((40, 40), bool)
    mask[10:20, 10:20] = True
    store.set_class(store.add_roi(store.plane_of({"c": 0, "z": 3}), mask), 0)
    state.widget.save()
    vis = DataVis(str(two_stacks_mesc), unit="MSession_0/MUnit_2", size=(640, 480))
    try:
        studio = attach_roi_widget(vis.widget)
        assert studio._save_target() == state.widget._save_target()
        assert [(studio.store.roi_z(0), studio.store.roi_c(0))] == [(3, 0)] and studio.store.label_names == ("spine",)
        studio.store.add_roi(studio.store.plane_of({"Channel": 1, "Z-plane": 5}), mask)
        studio.save()
    finally:
        vis.close()
    reopened = stack_annotate.build(filepath_stack=str(two_stacks_mesc), unit="MUnit_2", voxel_um=VOXEL_UM).widget.store
    assert sorted((reopened.roi_z(i), reopened.roi_c(i)) for i in range(len(reopened.rois))) == [(3, 0), (5, 1)]
