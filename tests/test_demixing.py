"""masknmf demixing results: the reader, the sibling listing and the Demixing tab."""

from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import scipy.sparse
import pytest
from imgui_bundle import imgui

T, Y, X, R, K = 12, 6, 8, 3, 4
ROWS = []
TEXTS = []
REAL_SELECTABLE = imgui.selectable
REAL_TEXT = imgui.text


def spy_selectable(label, selected, *a, **k):
    ROWS.append((label.split("##")[0], selected))
    return REAL_SELECTABLE(label, selected, *a, **k)


def spy_text(text):
    TEXTS.append(text)
    return REAL_TEXT(text)


def write_demixing(path, fs=None, labels=True, seed=0):
    """A DemixingResults hdf5 in masknmf's layout: sparse_coo factors, a
    trace matrix c, and a footprint per ROI on a distinct pixel block."""
    rng = np.random.default_rng(seed)
    pixels = Y * X
    with h5py.File(path, "w") as f:
        g = f.create_group("DemixingResults")
        g.create_dataset("shape", data=np.array([T, Y, X]))
        u = g.create_group("u")
        u.attrs["layout"] = "sparse_coo"
        u.create_dataset("indices", data=np.array([np.arange(pixels), np.arange(pixels) % R]))
        u.create_dataset("values", data=rng.random(pixels).astype(np.float32))
        u.create_dataset("size", data=np.array([pixels, R]))
        g.create_dataset("v", data=rng.random((R, T)).astype(np.float32))
        a = g.create_group("a")
        a.attrs["layout"] = "sparse_coo"
        rows = np.concatenate([np.arange(k * 4, k * 4 + 4) for k in range(K)])
        cols = np.repeat(np.arange(K), 4)
        a.create_dataset("indices", data=np.array([rows, cols]))
        a.create_dataset("values", data=np.tile([0.25, 0.5, 0.75, 1.0], K).astype(np.float32))
        a.create_dataset("size", data=np.array([pixels, K]))
        c = rng.random((T, K)).astype(np.float32)
        c[:, 1] *= 10  # roi 1 has the largest peak
        g.create_dataset("c", data=c)
        g.create_dataset("b", data=np.zeros(pixels, np.float32))
        g.create_dataset("mean_img", data=rng.random((Y, X)).astype(np.float32))
        g.create_dataset("var_img", data=np.ones((Y, X), np.float32))
        g.create_dataset("iscell", data=np.array([True, True, False, True]))
        if labels:
            g.create_dataset("class_labels", data=np.array([0, 1, 0, 2]))
            g.create_dataset("label_names", data=np.array([b"soma", b"edge", b"dendrite"]))
        if fs:
            f.attrs["mbo_provenance"] = json.dumps({"fs": fs, "input": "abc"})
    return path


class FakeImageWidget:
    def __init__(self, data):
        self.data = data


class FakeParent:
    def __init__(self, data):
        self.image_widget = FakeImageWidget(data)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A glutamate/calcium run folder with a result per channel and pass."""
    root = tmp_path_factory.mktemp("glu_ca_run")
    write_demixing(root / "calcium_spine_demixing.hdf5", fs=19.66, seed=1)
    write_demixing(root / "glutamate_spine_demixing.hdf5", seed=2)
    write_demixing(root / "glutamate_global_activity_demixing.hdf5", labels=False, seed=3)
    # an unrelated hdf5 in the same folder is not a result
    with h5py.File(root / "pmd_calcium.hdf5", "w") as f:
        f.create_dataset("PMDArray/u", data=np.zeros(3))
    return root


@pytest.fixture(scope="module")
def planes_dir(tmp_path_factory):
    """The mbo masknmf layout: one demixing_results.hdf5 per zplane folder."""
    root = tmp_path_factory.mktemp("planes")
    for z in (1, 2):
        (root / f"zplane0{z}").mkdir()
        write_demixing(root / f"zplane0{z}" / "demixing_results.hdf5", fs=9.6, seed=z)
    return root


def test_reader_shape_metadata_and_rois(run_dir):
    from mbo_utilities.arrays.demixing import VIEWS, DemixingArray

    arr = DemixingArray(run_dir / "calcium_spine_demixing.hdf5")
    assert arr.shape == (T, len(VIEWS), 1, Y, X)
    assert arr.dims == ("T", "C", "Z", "Y", "X")
    assert arr.dtype == np.float32
    assert arr.metadata["fs"] == 19.66
    assert arr.metadata["num_timepoints"] == T and arr.metadata["Lx"] == X
    assert arr.metadata["demixing_num_rois"] == K
    assert arr.slider_dim_labels == ("Timepoint", "View")
    assert arr.num_rois == K
    assert arr.roi_labels == ["soma", "edge", "soma", "dendrite"]
    assert arr.iscell.tolist() == [True, True, False, True]
    assert arr.traces.shape == (T, K)
    assert arr.footprints.shape == (Y * X, K)
    fp = arr.footprint(1)
    assert fp.shape == (Y, X)
    assert np.count_nonzero(fp) == 4 and fp.flat[7] == 1.0
    # the representative frame is the stored mean image, no factor rebuild
    assert np.asarray(arr).shape == (Y, X)
    unlabeled = DemixingArray(run_dir / "glutamate_global_activity_demixing.hdf5")
    assert unlabeled.roi_labels == ["-"] * K
    assert "fs" not in unlabeled.metadata


def test_imread_dispatches_demixing_files_over_h5(run_dir):
    from mbo_utilities.arrays.demixing import DemixingArray
    from mbo_utilities.arrays.h5 import H5Array
    from mbo_utilities.lazy_array import _dispatch, register_array_class

    register_array_class(DemixingArray)
    demix = run_dir / "calcium_spine_demixing.hdf5"
    assert DemixingArray.can_open(demix)
    assert not DemixingArray.can_open(run_dir / "pmd_calcium.hdf5")
    assert not DemixingArray.can_open(run_dir)
    assert _dispatch(demix) is DemixingArray
    assert _dispatch(run_dir / "pmd_calcium.hdf5") is H5Array


def test_list_demixing_results_names_channels(run_dir):
    from mbo_utilities.arrays.demixing import list_demixing_results

    entries = list_demixing_results(run_dir / "glutamate_spine_demixing.hdf5")
    assert [e["label"] for e in entries] == [
        "calcium_spine", "glutamate_global_activity", "glutamate_spine",
    ]
    assert [e["channel"] for e in entries] == ["calcium", "glutamate", "glutamate"]
    assert all(e["shape"] == (T, Y, X) and e["num_rois"] == K for e in entries)


def test_list_demixing_results_walks_plane_folders(planes_dir):
    from mbo_utilities.arrays.demixing import list_demixing_results

    entries = list_demixing_results(planes_dir / "zplane02" / "demixing_results.hdf5")
    assert [e["label"] for e in entries] == ["zplane02", "zplane01"]
    assert [e["channel"] for e in entries] == [None, None]


def test_roi_rows_and_footprint_overlay(run_dir):
    from mbo_utilities.arrays.demixing import DemixingArray
    from mbo_utilities.gui.widgets.demixing import ROI_COLUMNS, footprint_rgba, roi_rows

    arr = DemixingArray(run_dir / "calcium_spine_demixing.hdf5")
    rows = roi_rows(arr)
    assert len(rows) == K
    cells, keys = rows[2]
    assert len(cells) == len(keys) == len(ROI_COLUMNS)
    assert cells[:3] == ("2", "soma", "no")
    assert keys[0] == 2 and keys[2] is False
    assert max(rows, key=lambda r: r[1][3])[1][0] == 1

    rgba = footprint_rgba(arr, range(K), selected=1)
    assert rgba.shape == (Y, X, 4)
    flat = rgba.reshape(-1, 4)
    assert flat[4:8, :3].tolist() == [[255, 255, 90]] * 4  # roi 1 highlighted
    assert flat[7, 3] == 220 and flat[4, 3] == 100  # alpha follows the weight
    assert flat[0, 3] > 0 and flat[16, 3] == 0  # roi 0 painted; pixel 16 belongs to no roi
    only_selected = footprint_rgba(arr, (), selected=3)
    assert only_selected.reshape(-1, 4)[:12, 3].tolist() == [0] * 12


def test_tab_appears_only_for_demixing_data(run_dir):
    from mbo_utilities.arrays.demixing import DemixingArray
    from mbo_utilities.gui.widgets import get_tab_widgets
    from mbo_utilities.gui.widgets.demixing import DemixingTabWidget
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    arr = DemixingArray(run_dir / "calcium_spine_demixing.hdf5")
    names = [type(w).__name__ for w in get_tab_widgets(FakeParent([display_wrap(arr)]))]
    assert "DemixingTabWidget" in names
    assert names.index("PreviewTabWidget") < names.index("DemixingTabWidget")
    assert not DemixingTabWidget.is_supported(FakeParent([np.zeros((2, 4, 4))]))


def test_tab_draws_channels_and_every_roi(run_dir):
    """One frame on a bare imgui context: the channel combo, a row per ROI
    with its label, and no selection to start."""
    from mbo_utilities.arrays.demixing import DemixingArray
    from mbo_utilities.gui.widgets.demixing import DemixingTabWidget
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    arr = DemixingArray(run_dir / "glutamate_spine_demixing.hdf5")
    widget = DemixingTabWidget(FakeParent([display_wrap(arr)]))
    ROWS.clear()
    TEXTS.clear()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(900, 700)
    # imgui 1.92 builds fonts lazily once a renderer claims texture support
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    imgui.selectable, imgui.text = spy_selectable, spy_text
    try:
        for _ in range(2):
            imgui.new_frame()
            imgui.set_next_window_size(imgui.ImVec2(800, 600))
            imgui.begin("host")
            widget.draw()
            imgui.end()
            imgui.end_frame()
    finally:
        imgui.selectable, imgui.text = REAL_SELECTABLE, REAL_TEXT
        imgui.destroy_context(ctx)
    assert [e["label"] for e in widget._siblings] == [
        "calcium_spine", "glutamate_global_activity", "glutamate_spine",
    ]
    assert ROWS[-K:] == [("0", False), ("1", False), ("2", False), ("3", False)]
    assert "dendrite" in TEXTS and "edge" in TEXTS
    assert widget._selected is None


def test_frames_are_rebuilt_with_numpy_without_torch(run_dir, monkeypatch):
    # MBO_GPU=0 forces the cpu path, which must never import torch or masknmf
    monkeypatch.setenv("MBO_GPU", "0")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "masknmf", None)
    from mbo_utilities.arrays.demixing import DemixingArray

    path = run_dir / "calcium_spine_demixing.hdf5"
    with h5py.File(path, "r") as f:
        g = f["DemixingResults"]
        ui = g["u/indices"][()]
        u = scipy.sparse.csr_matrix((g["u/values"][()], (ui[0], ui[1])), shape=tuple(g["u/size"][()]))
        ai = g["a/indices"][()]
        a = scipy.sparse.csr_matrix((g["a/values"][()], (ai[0], ai[1])), shape=tuple(g["a/size"][()]))
        v = g["v"][()]
        c = g["c"][()]
    pmd = np.asarray((u @ v).T).reshape(T, Y, X)
    ac = np.asarray((a @ c.T).T).reshape(T, Y, X)

    arr = DemixingArray(path)
    assert arr[3, 1, 0].shape == (Y, X)
    np.testing.assert_allclose(arr[3, 1, 0], ac[3], rtol=1e-5)
    np.testing.assert_allclose(arr[5, 0, 0], pmd[5], rtol=1e-5)
    # the fixture stores a zero baseline and no background terms
    np.testing.assert_allclose(arr[2, 2, 0], pmd[2] - ac[2], rtol=1e-5, atol=1e-6)
    assert arr[0:2].shape == (2, 3, 1, Y, X)
    assert arr[0, :, 0, 1:3, :].shape == (3, 2, X)
    np.testing.assert_allclose(arr[0:2][:, 0, 0], pmd[0:2], rtol=1e-5)
    assert arr._results is None
    arr.close()
    assert arr._factors is None


def test_numpy_frames_match_masknmf(run_dir):
    # masknmf's import fails with AttributeError on a mismatched fastplotlib
    # pin, which importorskip would report as a failure
    try:
        import masknmf
    except Exception as e:
        pytest.skip(f"masknmf not importable: {e}")
    from mbo_utilities.arrays.demixing import DemixingArray

    arr = DemixingArray(run_dir / "calcium_spine_demixing.hdf5", device="cpu")
    res = masknmf.DemixingResults.from_hdf5(run_dir / "calcium_spine_demixing.hdf5", device="cpu")
    np.testing.assert_allclose(arr[3, 1, 0], np.asarray(res.ac_array[3]), rtol=1e-5)
    np.testing.assert_allclose(arr[5, 0, 0], np.asarray(res.pmd_array[5]), rtol=1e-5)
    np.testing.assert_allclose(arr[2, 2, 0], np.asarray(res.residual_array[2]), rtol=1e-5, atol=1e-6)
    assert arr._results is None
    arr.close()
