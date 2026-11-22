"""Process > MaskNMF: masknmf's Parameters popup drawn in the Run tab, and the worker args it spawns."""

from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

pytest.importorskip("masknmf.visualization.imgui.parameters")

from mbo_utilities.arrays.numpy import NumpyArray  # noqa: E402
from mbo_utilities.gui.widgets.pipelines import masknmf as masknmf_widget  # noqa: E402


def _parent(tmp_path):
    arr = NumpyArray(np.zeros((20, 2, 16, 16), np.float32), dims="TZYX")
    arr.metadata["fs"] = 30.0
    return SimpleNamespace(
        image_widget=SimpleNamespace(data=[arr]),
        fpath=str(tmp_path / "movie.tif"),
        nz=2,
        nc=1,
        invert_deflection=False,
        frame_average=1,
        _custom_metadata={},
        _masknmf_frame_average=1,
    )


def _draw(widget, n=3, open_parameters=False):
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1400, 900)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        for i in range(n):
            imgui.new_frame()
            imgui.begin("run tab")
            if open_parameters and i == 0:
                widget.parameters.open()
            widget.draw_config()
            imgui.end()
            imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)


def test_the_tab_and_the_parameters_popup_draw(tmp_path):
    widget = masknmf_widget.MaskNMFPipelineWidget(_parent(tmp_path))
    _draw(widget, open_parameters=True)
    assert widget.parameters.slug == masknmf_widget.PIPELINE_INITIAL
    assert widget._selected_planes() == [1, 2]


def test_run_spawns_the_parameters_config(tmp_path, monkeypatch):
    spawned = []
    pm = SimpleNamespace(spawn=lambda **kw: spawned.append(kw) or 123)
    monkeypatch.setattr("mbo_utilities.gui.widgets.process_manager.get_process_manager", lambda: pm)
    widget = masknmf_widget.MaskNMFPipelineWidget(_parent(tmp_path))
    _draw(widget, n=1)
    widget._outdir = str(tmp_path / "out")
    widget._submit(widget._selected_planes())
    (job,) = spawned
    args = job["args"]
    assert job["task_type"] == "masknmf"
    assert args["config"] == widget.parameters.config()
    assert args["config"]["pipeline"] == "TwoPhotonCalciumPipeline"
    assert args["runtime"] == widget.parameters.runtime()
    assert args["run_args"] == widget.parameters.run_args()
    assert args["planes"] == [1, 2]
