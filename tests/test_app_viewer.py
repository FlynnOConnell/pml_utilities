"""The app host on the viewer: the playhead, the sliders and opening other data."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from imgui_bundle import imgui
from imgui_debugger import ConfigStore

ui = pytest.importorskip("fastplotlib.ui")
if not hasattr(ui, "ImguiWindow"):
    pytest.skip("needs a fastplotlib with ImguiWindow", allow_module_level=True)
pytest.importorskip("fastplotlib.widgets.nd_widget")

from mbo_utilities import imread  # noqa: E402
from mbo_utilities.arrays import NumpyArray, average_frames  # noqa: E402
from mbo_utilities.arrays.features import find_slider_name  # noqa: E402
from mbo_utilities.gui.app import _app, build_host  # noqa: E402
from mbo_utilities.gui.app.apps.viewer import blur  # noqa: E402
from mbo_utilities.gui.app.demo import movie_data  # noqa: E402


def confirm_prompt(prompt):
    """Stands in for draw_path_prompt, submitting the path an open prompt holds."""
    return (prompt.path if prompt.open else None), False


def tap(host, key: str) -> None:
    """Press and release one key, a frame each, the way a user would."""
    io = imgui.get_io()
    io.add_key_event(getattr(imgui.Key, key), True)
    host.figure.canvas.force_draw()
    io.add_key_event(getattr(imgui.Key, key), False)
    host.figure.canvas.force_draw()


@pytest.fixture
def host():
    host = build_host(movie_data(nt=24, ny=32, nx=32), size=(900, 600))
    host.figure.show()
    host.figure.canvas.force_draw()
    yield host
    host.close()
    host.apps["viewer"].viewer.close()


def t_slider(host) -> str:
    return find_slider_name(host.apps["viewer"].viewer.dim_names, "t")


def test_the_viewer_draws_on_the_figure_and_owns_the_bottom_edge(host):
    viewer = host.apps["viewer"].viewer
    assert host.figure is viewer.figure
    assert host.slots == []
    assert host.data.shape == (24, 1, 1, 32, 32)
    assert host.figure.imgui_windows["bottom"] is viewer.ndwidget.ui_sliders
    assert host.figure.imgui_windows["right"] is host.docks["right"]


def test_moving_the_t_slider_moves_the_playhead(host):
    viewer = host.apps["viewer"].viewer
    viewer.indices[t_slider(host)] = 5
    host.figure.canvas.force_draw()
    assert host.frame == 5
    assert host.playhead.time == pytest.approx(5 / host.data.fs)


def test_seeking_the_playhead_moves_the_t_slider(host):
    host.seek_frame(9)
    host.figure.canvas.force_draw()
    assert host.apps["viewer"].viewer.current_index[t_slider(host)] == 9


def test_opening_other_data_swaps_the_viewers_array(host, tmp_path):
    np.save(tmp_path / "short.npy", movie_data(nt=7, ny=16, nx=16))
    host.seek_frame(9)
    host.figure.canvas.force_draw()

    host.set_data(imread(tmp_path / "short.npy"))
    host.figure.canvas.force_draw()
    viewer = host.apps["viewer"].viewer
    assert viewer.data[0].shape == (7, 16, 16)
    assert host.frame == 0
    assert viewer.current_index[t_slider(host)] == 0
    assert host.title().endswith("short.npy")


def test_a_panel_is_rebuilt_for_the_data_it_shows(host):
    image = np.ones((32, 32), dtype=np.float32)
    summaries = host.apps["summary_images"]
    assert summaries.available(host) is False

    host.set_data(
        NumpyArray(movie_data(nt=5, ny=32, nx=32), metadata={"meanImg": image})
    )
    summaries.open = True
    host.figure.canvas.force_draw()
    assert summaries.available(host) is True
    assert summaries.widget is not None

    host.set_data(imread(movie_data(nt=5, ny=32, nx=32)))
    assert summaries.widget is None
    assert summaries.available(host) is False


def test_rebinning_keeps_the_blur_and_another_file_drops_it(host):
    app = host.apps["viewer"]
    app.sigma = 1.5
    app.viewer.spatial_func = partial(blur, sigma=app.sigma)

    host.set_data(average_frames(host.data, 4))
    host.figure.canvas.force_draw()
    assert host.data.shape[0] == 6
    assert app.sigma == 1.5
    assert app.viewer.spatial_func is not None

    host.set_data(imread(movie_data(nt=8, ny=32, nx=32)))
    host.figure.canvas.force_draw()
    assert app.sigma == 0.0
    assert app.viewer.spatial_func is None


def test_every_section_of_the_panel_draws(host, monkeypatch):
    monkeypatch.setattr(imgui, "collapsing_header", lambda *args, **kwargs: True)
    _app._reported.discard("viewer")
    host.figure.canvas.force_draw()
    host.set_data(average_frames(host.data, 3))
    host.figure.canvas.force_draw()
    assert "viewer" not in _app._reported


def test_the_blur_smooths_the_frame():
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4, 4] = 1.0
    smoothed = blur(frame, 1.0)
    assert smoothed[4, 4] < 1.0
    assert smoothed.sum() == pytest.approx(1.0, abs=1e-4)


def test_the_debug_panels_are_registered_as_apps(host):
    ids = {app.id for app in host.apps.values()}
    assert {"panel_Debugger", "panel_MetricsPanel", "panel_DemoPanel"} <= ids
    assert host.apps["panel_Debugger"].panel.config.target is host


def test_a_panel_app_opens_the_window_itself(host):
    metrics = host.apps["panel_MetricsPanel"]
    metrics.open = True
    assert metrics.panel.visible is True
    host.figure.canvas.force_draw()
    host.figure.canvas.force_draw()
    assert metrics.open is True
    metrics.open = False
    host.figure.canvas.force_draw()
    assert metrics.panel.visible is False


def test_a_path_that_does_not_open_keeps_the_prompt_up(host, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mbo_utilities.gui.app.apps.open.draw_path_prompt", confirm_prompt
    )
    before = host.data
    opener = host.apps["open_file"]
    opener.open = True
    opener.prompt.path = str(tmp_path / "missing.tif")
    host.figure.canvas.force_draw()

    assert opener.open is True
    assert opener.prompt.status.startswith("not found")
    assert host.data is before
    opener.open = False


def test_a_file_that_does_not_read_says_why(host, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mbo_utilities.gui.app.apps.open.draw_path_prompt", confirm_prompt
    )
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a tiff")
    opener = host.apps["open_file"]
    opener.open = True
    opener.prompt.path = str(broken)
    host.figure.canvas.force_draw()

    assert opener.open is True
    assert opener.prompt.status
    assert not opener.prompt.status.startswith("not found")
    opener.open = False


def test_an_apps_shortcut_shows_and_hides_it(host):
    keybinds = host.apps["keybinds"]
    assert keybinds.open is False
    tap(host, "k")
    assert keybinds.open is True
    tap(host, "k")
    assert keybinds.open is False


def test_the_arrow_keys_step_the_playhead(host):
    host.figure.canvas.force_draw()
    tap(host, "right_arrow")
    tap(host, "right_arrow")
    assert host.frame == 2
    tap(host, "left_arrow")
    assert host.frame == 1


def test_the_help_keybinds_and_options_windows_draw(host):
    _app._reported.clear()
    for app_id in ("help", "keybinds", "options"):
        host.apps[app_id].open = True
    host.figure.canvas.force_draw()
    host.figure.canvas.force_draw()
    assert not {"help", "keybinds", "options"} & _app._reported


def test_the_host_remembers_which_apps_were_showing(tmp_path):
    store = ConfigStore(tmp_path)
    first = build_host(movie_data(nt=4, ny=16, nx=16), size=(600, 400), store=store)
    first.figure.show()
    first.apps["log"].open = True
    first.apps["viewer"].open = False
    first.figure.canvas.force_draw()
    first.close()
    first.apps["viewer"].viewer.close()

    second = build_host(movie_data(nt=4, ny=16, nx=16), size=(600, 400), store=store)
    assert second.apps["log"].open is True
    assert second.apps["viewer"].open is False
    second.close()
    second.apps["viewer"].viewer.close()
