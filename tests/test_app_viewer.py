"""The app host on the viewer: the playhead, the sliders and opening other data."""

from __future__ import annotations

import numpy as np
import pytest

ui = pytest.importorskip("fastplotlib.ui")
if not hasattr(ui, "ImguiWindow"):
    pytest.skip("needs a fastplotlib with ImguiWindow", allow_module_level=True)
pytest.importorskip("fastplotlib.widgets.nd_widget")

from mbo_utilities import imread  # noqa: E402
from mbo_utilities.arrays import NumpyArray  # noqa: E402
from mbo_utilities.arrays.features import find_slider_name  # noqa: E402
from mbo_utilities.gui.app.apps.open import NOTE  # noqa: E402
from mbo_utilities.gui.app.demo import movie_data  # noqa: E402


def confirm_path(title, is_open, path, hint, action, browse=None, note="", theme=None):
    """Stands in for draw_path_popup, confirming the path it is given."""
    return True, path, True


@pytest.fixture
def host():
    from mbo_utilities.gui.app import build_host

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
    monkeypatch.setattr("mbo_utilities.gui.app.apps.open.draw_path_popup", confirm_path)
    before = host.data
    opener = host.apps["open"]
    opener.path = str(tmp_path / "missing.tif")
    opener.open = True
    host.figure.canvas.force_draw()

    assert opener.open is True
    assert opener.note != NOTE
    assert "missing.tif" not in host.title()
    assert host.data is before
    opener.open = False
