"""The Voltage Preprocessing widget: its preset, its run arguments and its movie viewer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

imgui = pytest.importorskip("imgui_bundle").imgui

from mbo_utilities.arrays.numpy import NumpyArray  # noqa: E402
from mbo_utilities.masknmf.params import STAGE_SKIP, MasknmfSettings  # noqa: E402

SPAWNED = []
VIEWED = []
PRESSED = set()
REAL_BUTTON = imgui.button


def press(label, *a, **k):
    return REAL_BUTTON(label, *a, **k) or label.split("##")[0] in PRESSED


class FakeProcessManager:
    def spawn(self, **kwargs):
        SPAWNED.append(kwargs)
        return 4242


def fake_process_manager():
    return FakeProcessManager()


def fake_launch(module, args, log_name):
    VIEWED.append((module, args))
    return 4243


def frames(fn, n=2):
    """Run ``fn`` for ``n`` frames on a bare imgui context."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1600, 900)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        for _ in range(n):
            imgui.new_frame()
            fn()
            imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)


@pytest.fixture
def host(tmp_path):
    movie = NumpyArray(
        np.zeros((40, 1, 1, 8, 8), np.int16), dims="TCZYX", metadata={"fs": 500.0}
    )
    return SimpleNamespace(
        fpath=str(tmp_path / "movie.tif"),
        image_widget=SimpleNamespace(data=[movie]),
        invert_deflection=True,
        nz=1,
        nc=1,
    )


@pytest.fixture
def widget(host):
    from mbo_utilities.gui.widgets.pipelines.voltage_preprocessing import (
        VoltagePreprocessingWidget,
    )

    return VoltagePreprocessingWidget(host)


def test_the_preset_registers_on_a_denoised_copy_and_stops_before_demixing(widget):
    s = widget.settings
    assert s.registration.denoised_reference is True
    assert s.demixing.do_demixing == STAGE_SKIP
    assert widget.default_settings().to_dict() == s.to_dict()
    assert MasknmfSettings().registration.denoised_reference is False


def test_it_is_listed_after_masknmf():
    from mbo_utilities.gui.widgets.pipelines import _register_pipelines_sync, _PIPELINE_CLASSES

    _register_pipelines_sync()
    names = [cls.name for cls in _PIPELINE_CLASSES]
    assert names.index("Voltage Preprocessing") == names.index("MaskNMF") + 1


def test_it_draws_with_the_settings_popup_open(widget):
    widget._show_settings_popup = True
    widget._show_slice_popup = True
    frames(widget.draw_config, n=3)
    assert widget._invert_deflection is True


def test_run_sends_the_inverted_movie_and_the_preset(widget, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mbo_utilities.gui.widgets.process_manager.get_process_manager",
        fake_process_manager,
    )
    monkeypatch.setattr(imgui, "button", press)
    PRESSED.clear()
    PRESSED.add("Run Preprocessing")
    widget._outdir = str(tmp_path / "out")
    SPAWNED.clear()
    frames(widget.draw_config, n=1)
    assert len(SPAWNED) == 1
    args = SPAWNED[0]["args"]
    assert SPAWNED[0]["task_type"] == "masknmf"
    assert args["invert_deflection"] is True
    assert args["settings"]["registration"]["denoised_reference"] is True
    assert SPAWNED[0]["description"] == "Voltage Preprocessing plane01"


def test_view_movies_opens_the_newest_run_folder(widget, tmp_path, monkeypatch):
    import os

    monkeypatch.setattr("mbo_utilities.gui.launch.launch_window", fake_launch)
    monkeypatch.setattr(imgui, "button", press)
    PRESSED.clear()
    PRESSED.add("View movies")
    out = tmp_path / "out"
    names = ("20261007T120000_masknmf_zplane01", "20261007T110000_masknmf_zplane01")
    for i, name in enumerate(names):
        (out / name).mkdir(parents=True)
        (out / name / "config.json").write_text("{}")
        os.utime(out / name / "config.json", (i, i))
    widget._outdir = str(out)
    VIEWED.clear()
    frames(widget.draw_config, n=1)
    run = out / "20261007T110000_masknmf_zplane01"
    assert VIEWED == [("mbo_utilities.gui.registration_viewer", [str(run)])]
    assert "4243" in widget._last_status
