"""The app host: docks that claim an edge, and apps that swap on the subplots."""

from __future__ import annotations

import numpy as np
import pytest

ui = pytest.importorskip("fastplotlib.ui")
if not hasattr(ui, "ImguiWindow"):
    pytest.skip("needs a fastplotlib with ImguiWindow", allow_module_level=True)

from mbo_utilities.gui.app import App  # noqa: E402


@pytest.fixture(scope="module")
def host():
    from mbo_utilities.gui.app import build_host
    from mbo_utilities.gui.app.demo import movie_data

    host = build_host(movie_data(nt=24, ny=32, nx=32), size=(900, 600))
    host.figure.show()
    host.figure.canvas.force_draw()
    yield host
    host.close()


def test_the_host_starts_with_the_demo_apps_on_the_slots(host):
    assert host.stage == {0: "movie", 1: "traces"}
    assert [type(g).__name__ for g in host.slots[0].graphics] == ["ImageGraphic"]
    assert all(type(g).__name__ == "LineGraphic" for g in host.slots[1].graphics)


def test_a_dock_claims_its_edge_and_gives_it_back(host):
    for app in host.apps.values():
        app.open = app.dock == "right"
    host.figure.canvas.force_draw()
    assert host.figure.imgui_windows["right"] is host.docks["right"]
    narrow = host.figure.get_pygfx_render_area()[2]

    for app in host.apps.values():
        app.open = False
    host.figure.canvas.force_draw()
    assert host.figure.imgui_windows["right"] is None
    assert host.figure.get_pygfx_render_area()[2] > narrow


def test_mounting_an_app_takes_it_off_the_slot_it_was_on(host):
    host.mount("movie", 0)
    host.mount("traces", 1)
    host.mount("traces", 0)
    host.figure.canvas.force_draw()
    assert host.stage == {0: "traces", 1: None}
    assert host.slots[1].graphics == ()
    assert host.apps["traces"].mounted is host.slots[0]
    assert host.apps["movie"].graphic is None


def test_a_frame_hook_runs_while_the_app_is_closed(host):
    host.mount("movie", 0)
    movie = host.apps["movie"]
    movie.open = False
    host.index = 3
    host.figure.canvas.force_draw()
    assert movie.player.t == 3
    assert movie._shown == 3


def test_a_window_app_owns_its_texture(host):
    viewer = host.apps["image_viewer"]
    viewer.open = True
    host.figure.canvas.force_draw()
    host.figure.canvas.force_draw()
    assert viewer.texture.ref is not None
    assert (viewer.texture.height, viewer.texture.width) == host.data.shape[1:]
    viewer.close()
    assert viewer.texture is None


def test_opening_other_data_rebuilds_what_is_mounted(host):
    host.mount("traces", 1)
    host.figure.canvas.force_draw()
    before = np.asarray(host.apps["traces"].traces[0]).copy()

    host.set_data(host.data[:12] * 2.0)
    host.figure.canvas.force_draw()
    after = np.asarray(host.apps["traces"].traces[0])
    assert host.index == 0
    assert after.shape == (12,)
    assert not np.array_equal(before[:12], after)


def test_the_trace_offset_restacks_the_lines(host):
    host.mount("traces", 1)
    host.figure.canvas.force_draw()
    traces = host.apps["traces"]
    traces.offset = 3.0
    for i, line in enumerate(traces.lines):
        line.data[:, 1] = traces.traces[i] + i * traces.offset
    host.figure.canvas.force_draw()
    tops = [float(np.asarray(line.data[:, 1]).mean()) for line in traces.lines]
    assert tops == pytest.approx([0.0, 3.0, 6.0, 9.0], abs=1e-4)


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


class Exploding(App):
    """An app whose body raises, to prove the draw boundary holds."""

    id = "exploding"
    title = "Exploding"
    window = True
    start_open = True

    def draw_canvas(self, host, size):
        raise RuntimeError("boom")


def test_an_app_that_raises_does_not_take_the_frame_with_it(host):
    host.register(Exploding())
    host.figure.canvas.force_draw()
    host.figure.canvas.force_draw()
    assert host.apps["exploding"].open is True
    assert host.apps["movie"].mounted is host.slots[0]
    host.apps["exploding"].open = False
