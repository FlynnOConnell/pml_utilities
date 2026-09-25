"""The preview window's panels, hosted as apps."""

from __future__ import annotations

import numpy as np
import pytest

ui = pytest.importorskip("fastplotlib.ui")
if not hasattr(ui, "ImguiWindow"):
    pytest.skip("needs a fastplotlib with ImguiWindow", allow_module_level=True)
apps_module = pytest.importorskip("mbo_utilities.gui.app.apps")


def confirm_prompt(prompt):
    """Stands in for draw_path_prompt, submitting the path an open prompt holds."""
    return (prompt.path if prompt.open else None), False


def forget(*args, **kwargs):
    """Stands in for the preference writers, so a test leaves no recent files."""


@pytest.fixture(scope="module")
def host():
    from mbo_utilities.arrays import NumpyArray
    from mbo_utilities.gui.app import build_host

    image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    host = build_host(
        NumpyArray(image, metadata={"meanImg": image}),
        apps=apps_module.ported_apps(),
        size=(900, 600),
    )
    host.figure.show()
    host.figure.canvas.force_draw()
    yield host
    host.close()
    host.viewer.close()


def test_every_ported_app_declares_where_it_is_drawn():
    ported = apps_module.ported_apps()
    assert len({app.id for app in ported}) == len(ported)
    # an action (curate) and a background job are the only apps with no place
    placeless = {app.id for app in ported if not (app.dock or app.window)}
    assert placeless == {"curate", "isoview_projections"}


def test_a_panel_is_available_for_the_data_it_reads(host):
    assert host.apps["summary_images"].available(host) is True
    assert host.apps["projections"].available(host) is False
    assert host.apps["tile_grid"].available(host) is False


def test_an_available_panel_claims_the_edge_it_asked_for(host):
    host.apps["summary_images"].open = True
    host.figure.canvas.force_draw()
    assert host.figure.imgui_windows["left"] is host.docks["left"]
    assert host.apps["summary_images"]._cmap_synced_with_fpl is True

    host.apps["summary_images"].open = False
    host.figure.canvas.force_draw()
    assert host.figure.imgui_windows["left"] is None


def test_the_metadata_app_reads_the_open_array(host):
    viewer = host.apps["metadata"]
    assert viewer.available(host) is True
    viewer.open = True
    host.figure.canvas.force_draw()
    viewer.open = False


def test_the_log_app_receives_records(host):
    from mbo_utilities import log

    panel = host.apps["log"].panel
    before = len(panel.messages)
    log.get("gui.app").info("a line the log app should show")
    assert len(panel.messages) == before + 1
    assert panel.messages[-1][3] == "a line the log app should show"


def test_debug_panels_become_apps_that_own_their_window():
    panels = apps_module.debug_apps(object())
    assert len({app.id for app in panels}) == len(panels)
    assert all(app.window and app.owns_window for app in panels)
    first = panels[0]
    first.open = True
    assert first.panel.visible is True
    first.open = False
    assert first.panel.visible is False


def test_the_open_app_loads_a_file_into_the_host(host, tmp_path, monkeypatch):
    import tifffile

    movie = np.linspace(0, 1, 6 * 8 * 8, dtype=np.float32).reshape(6, 8, 8)
    path = tmp_path / "opened.tif"
    tifffile.imwrite(path, movie)

    monkeypatch.setattr(
        "mbo_utilities.gui.app.apps.open.draw_path_prompt", confirm_prompt
    )
    monkeypatch.setattr("mbo_utilities.gui.app.apps.open.add_recent_file", forget)
    monkeypatch.setattr("mbo_utilities.gui.app.apps.open.set_last_dir", forget)
    opener = host.apps["open_file"]
    opener.open = True
    opener.prompt.path = str(path)
    host.figure.canvas.force_draw()

    assert opener.open is False
    assert host.data.shape == (6, 1, 1, 8, 8)
    assert host.frame == 0
