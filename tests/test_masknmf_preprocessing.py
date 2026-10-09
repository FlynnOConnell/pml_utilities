"""The MaskNMF widget's Denoise before registration checkbox and its movie viewer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import psutil
import pytest

imgui = pytest.importorskip("imgui_bundle").imgui

from mbo_utilities.arrays.numpy import NumpyArray  # noqa: E402

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


def gone_process(pid):
    raise psutil.NoSuchProcess(pid)


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
    from mbo_utilities.gui.widgets.pipelines.masknmf import MaskNMFPipelineWidget

    return MaskNMFPipelineWidget(host)


def test_denoising_before_registration_starts_off(widget):
    assert widget.settings.registration.denoised_reference is False


def test_the_process_tab_lists_masknmf_and_not_rois_or_voltage_preprocessing():
    from mbo_utilities.gui.widgets.pipelines import (
        _PIPELINE_CLASSES,
        _register_pipelines_sync,
    )

    _register_pipelines_sync()
    names = [cls.name for cls in _PIPELINE_CLASSES]
    assert "MaskNMF" in names
    assert "ROIs" not in names and "Voltage Preprocessing" not in names


def test_it_draws_with_the_settings_popup_open(widget):
    widget.settings.registration.denoised_reference = True
    widget._show_settings_popup = True
    widget._show_slice_popup = True
    frames(widget.draw_config, n=3)
    assert widget._invert_deflection is True


def test_run_sends_the_inverted_movie_and_the_denoised_copy(
    widget, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "mbo_utilities.gui.widgets.process_manager.get_process_manager",
        fake_process_manager,
    )
    monkeypatch.setattr(imgui, "button", press)
    PRESSED.clear()
    PRESSED.add("Run MaskNMF")
    widget.settings.registration.denoised_reference = True
    widget._outdir = str(tmp_path / "out")
    SPAWNED.clear()
    frames(widget.draw_config, n=1)
    assert len(SPAWNED) == 1
    args = SPAWNED[0]["args"]
    assert SPAWNED[0]["task_type"] == "masknmf"
    assert args["invert_deflection"] is True
    assert args["settings"]["registration"]["denoised_reference"] is True
    assert SPAWNED[0]["description"] == "MaskNMF plane01"


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
    assert VIEWED == [("mbo_utilities.gui.reg_denoise_viewer", [str(run)])]
    assert "4243" in widget._last_status


def test_a_viewer_that_crashes_puts_its_error_in_the_status_line(
    widget, tmp_path, monkeypatch
):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "20261008_150000_movies_run1.log").write_text(
        "Traceback (most recent call last):\n  ...\nValueError: no raw movie\n"
    )
    monkeypatch.setattr(
        "mbo_utilities.gui.widgets.pipelines.masknmf.get_mbo_dirs",
        lambda: {"logs": logs},
    )
    monkeypatch.setattr(
        "mbo_utilities.gui.widgets.pipelines.masknmf.psutil.Process", gone_process
    )
    widget._viewer_launch = (999999, "movies_run1")
    frames(widget.draw_config, n=1)
    assert widget._viewer_launch is None
    assert widget._last_status.startswith("QC viewer failed: ValueError: no raw movie")


def test_a_viewer_closed_normally_leaves_the_status_alone(
    widget, tmp_path, monkeypatch
):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "20261008_150000_movies_run1.log").write_text("opening viewer\n")
    monkeypatch.setattr(
        "mbo_utilities.gui.widgets.pipelines.masknmf.get_mbo_dirs",
        lambda: {"logs": logs},
    )
    monkeypatch.setattr(
        "mbo_utilities.gui.widgets.pipelines.masknmf.psutil.Process", gone_process
    )
    widget._viewer_launch = (999999, "movies_run1")
    widget._last_status = "Opening run1"
    frames(widget.draw_config, n=1)
    assert widget._viewer_launch is None and widget._last_status == "Opening run1"


def test_a_run_folder_that_demixed_names_its_results_file(tmp_path):
    import h5py
    from mbo_utilities.arrays.masknmf_run import run_demixing

    run = tmp_path / "20261008T143623_masknmf_zplane01"
    run.mkdir()
    (run / "config.json").write_text("{}")
    with h5py.File(run / "results.hdf5", "w") as f:
        f.create_group("RigidRegistrationArray")
    assert run_demixing(run) is None
    with h5py.File(run / "results.hdf5", "a") as f:
        f.create_group("DemixingResults")
    assert run_demixing(run) == run / "results.hdf5"
    assert run_demixing(tmp_path) is None
    assert run_demixing(run / "results.hdf5") is None


def test_registration_qc_refuses_a_path_that_is_not_a_run(tmp_path):
    import click
    from mbo_utilities.gui.run_gui import run_gui

    movie = tmp_path / "movie.tif"
    movie.write_bytes(b"")
    with pytest.raises(click.ClickException, match="not a MaskNMF run folder"):
        run_gui(data_in=str(movie), qc=True)


def test_the_launcher_offers_registration_qc():
    from mbo_utilities.gui.widgets.file_dialog import FileDialog

    assert "Registration QC (MaskNMF run)" in FileDialog().gui_modes
