"""The stack annotator offscreen, on a small synthetic stack: build, paint, fill, trace, undo."""

from pathlib import Path

import numpy as np
import pytest
import skimage.draw

pytest.importorskip("skan")
pytest.importorskip("fastplotlib")

from fastplotlib.widgets.nd_widget._async import run_sync

from mbo_utilities.gui import stack_annotate

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
