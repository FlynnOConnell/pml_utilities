"""The app host on the viewer: the playhead, the sliders and opening other data."""

from __future__ import annotations

import time

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
from mbo_utilities.gui._save_as import draw_saveas_popup  # noqa: E402
from mbo_utilities.gui.app import _app, build_host  # noqa: E402
from mbo_utilities.gui.app.apps.viewer import filter_frame  # noqa: E402
from mbo_utilities.gui.app.demo import movie_data  # noqa: E402


def confirm_prompt(prompt):
    """Stands in for draw_path_prompt, submitting the path an open prompt holds."""
    return (prompt.path if prompt.open else None), False


# what the spies below saw drawn this test
TAB_LABELS = []
COLORED_TEXT = []
REAL_BEGIN_TAB_ITEM = imgui.begin_tab_item
REAL_TEXT_COLORED = imgui.text_colored


def spy_begin_tab_item(label, *args, **kwargs):
    """Stands in for imgui.begin_tab_item, keeping the label."""
    TAB_LABELS.append(label)
    return REAL_BEGIN_TAB_ITEM(label, *args, **kwargs)


def spy_text_colored(color, text, *args, **kwargs):
    """Stands in for imgui.text_colored, keeping the text."""
    COLORED_TEXT.append(text)
    return REAL_TEXT_COLORED(color, text, *args, **kwargs)


class ThreeRois:
    """An array that asks to be split into its three ROIs, as ``roi=0`` does."""

    def __init__(self):
        self.roi = 0
        self.fix_phase = True

    def iter_rois(self):
        yield from (1, 2, 3) if self.roi == 0 else (self.roi,)


def settle(host) -> None:
    """Draw until the host's summary stats are in, at most ten seconds."""
    deadline = time.time() + 10
    while not all(host.zstats.done) and time.time() < deadline:
        host.figure.canvas.force_draw()
        time.sleep(0.05)
    host.figure.canvas.force_draw()


# every source record_save_source was handed, newest last
SAVE_SOURCES = []


def record_save_source(save, source):
    """Stands in for draw_saveas_popup, keeping each source it is handed."""
    SAVE_SOURCES.append(source)
    draw_saveas_popup(save, source)


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
    host.viewer.close()


def t_slider(host) -> str:
    return find_slider_name(host.viewer.dim_names, "t")


def test_the_viewer_draws_on_the_figure_and_owns_the_bottom_edge(host):
    viewer = host.viewer
    assert host.figure is viewer.figure
    assert host.slots == []
    assert host.data.shape == (24, 1, 1, 32, 32)
    assert host.figure.imgui_windows["bottom"] is viewer.ndwidget.ui_sliders
    assert host.figure.imgui_windows["right"] is host.docks["right"]


def test_moving_the_t_slider_moves_the_playhead(host):
    viewer = host.viewer
    viewer.indices[t_slider(host)] = 5
    host.figure.canvas.force_draw()
    assert host.frame == 5
    assert host.playhead.time == pytest.approx(5 / host.data.fs)


def test_seeking_the_playhead_moves_the_t_slider(host):
    host.seek_frame(9)
    host.figure.canvas.force_draw()
    assert host.viewer.current_index[t_slider(host)] == 9


def test_opening_other_data_swaps_the_viewers_array(host, tmp_path):
    np.save(tmp_path / "short.npy", movie_data(nt=7, ny=16, nx=16))
    host.seek_frame(9)
    host.figure.canvas.force_draw()

    host.set_data(imread(tmp_path / "short.npy"))
    host.figure.canvas.force_draw()
    viewer = host.viewer
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
    assert summaries._cmap_synced_with_fpl is True

    host.set_data(imread(movie_data(nt=5, ny=32, nx=32)))
    assert summaries._cmap_synced_with_fpl is False
    assert summaries.available(host) is False


def test_rebinning_keeps_the_blur_and_another_file_drops_it(host):
    app = host.apps["viewer"]
    app.sigma = 1.5
    app.apply_filters(host)

    host.set_data(average_frames(host.data, 4))
    host.figure.canvas.force_draw()
    assert host.data.shape[0] == 6
    assert app.sigma == 1.5
    assert host.viewer.spatial_func is not None

    host.set_data(imread(movie_data(nt=8, ny=32, nx=32)))
    host.figure.canvas.force_draw()
    assert app.sigma == 0.0
    assert host.viewer.spatial_func is None


def test_every_section_of_the_panel_draws(host, monkeypatch):
    monkeypatch.setattr(imgui, "collapsing_header", lambda *args, **kwargs: True)
    _app._reported.discard("viewer")
    host.figure.canvas.force_draw()
    host.set_data(average_frames(host.data, 3))
    host.figure.canvas.force_draw()
    assert "viewer" not in _app._reported


def test_the_host_computes_summary_stats_for_every_plane(host):
    settle(host)
    assert host.zstats.done == [True]
    assert host.zstats.means[0][()].shape[0] >= 1


def test_mean_subtraction_waits_for_the_stats_then_applies(host):
    settle(host)
    app = host.apps["viewer"]
    app.mean_subtraction = True
    app.apply_filters(host)
    assert app._subtracting is True
    assert host.viewer.spatial_func.keywords["mean"].shape == (32, 32)

    host.set_data(imread(movie_data(nt=6, ny=32, nx=32)))
    assert app.mean_subtraction is False
    assert host.viewer.spatial_func is None


def test_the_signal_quality_window_draws(host):
    settle(host)
    _app._reported.discard("signal_quality")
    host.apps["signal_quality"].open = True
    host.figure.canvas.force_draw()
    host.figure.canvas.force_draw()
    assert "signal_quality" not in _app._reported


def test_the_console_lists_running_stats_and_draws(host):
    _app._reported.discard("console")
    host.zstats.running = [True]
    host.zstats.progress = [0.5]
    work = host.apps["console"].work(host)
    assert [item["key"] for item in work] == ["zstats_0"]
    host.apps["console"].open = True
    host.figure.canvas.force_draw()
    host.figure.canvas.force_draw()
    assert "console" not in _app._reported


def test_set_metadata_edits_reach_the_viewer_and_go_with_the_data(host):
    from mbo_utilities.gui._metadata_editor import _apply_set

    _app._reported.difference_update({"set_metadata", "metadata"})
    edits = host.metadata_edits
    edits.inputs["dz"] = "2.5"
    _apply_set(edits, host.data, "dz", float)
    assert edits.values["dz"] == 2.5
    for app_id in ("set_metadata", "metadata"):
        host.apps[app_id].open = True
    host.figure.canvas.force_draw()
    host.figure.canvas.force_draw()
    assert not {"set_metadata", "metadata"} & _app._reported

    host.set_data(imread(movie_data(nt=4, ny=16, nx=16)))
    assert host.metadata_edits.values == {}


def test_save_as_opens_on_its_key_and_reads_the_display(host, monkeypatch):
    monkeypatch.setattr(
        "mbo_utilities.gui.app.apps.save_as.draw_saveas_popup", record_save_source
    )
    SAVE_SOURCES.clear()
    host.viewer.window_funcs = {t_slider(host): (np.max, 5)}
    app = host.apps["viewer"]
    app.sigma = 2.0
    app.apply_filters(host)

    tap(host, "s")
    host.figure.canvas.force_draw()
    save = host.apps["save_as"]
    assert save.open is True
    assert save.save.modal_open is True
    source = SAVE_SOURCES[-1]
    assert (source.projection, source.window, source.sigma) == ("max", 5, 2.0)
    assert source.source_frames == 24
    save.open = False


def test_the_process_tab_draws_its_pipelines(host):
    _app._reported.discard("run")
    run = host.apps["run"]
    run.open = True
    # a dock draws only its selected tab, so Process is left the only one
    host.apps["viewer"].open = False
    host.figure.canvas.force_draw()
    host.figure.canvas.force_draw()
    assert "run" not in _app._reported
    context = host.context
    assert context.image_widget is host.viewer
    assert (context.nz, context.nc, context.frame_average) == (1, 1, 1)
    host.set_data(average_frames(host.data, 2))
    assert context.frame_average == 2
    assert context._s2p_frame_average == 2


def test_showing_rois_turns_labeling_on_and_hiding_parks_them(host):
    rois = host.apps["manual_roi"]
    rois.open = True
    host.figure.canvas.force_draw()
    widget = host.context.manual_roi
    assert widget is not None
    assert host.strip.panels

    rois.open = False
    host.figure.canvas.force_draw()
    assert host.context.manual_roi is None
    assert host.context._manual_roi_store is widget.store

    rois.open = True
    host.figure.canvas.force_draw()
    assert host.context.manual_roi.store is widget.store

    host.set_data(average_frames(host.data, 2))
    host.figure.canvas.force_draw()
    assert host.context.manual_roi.store is widget.store

    host.set_data(imread(movie_data(nt=8, ny=32, nx=32)))
    host.figure.canvas.force_draw()
    assert host.context.manual_roi.store is not widget.store
    rois.open = False
    host.figure.canvas.force_draw()


def test_the_isoview_editors_apply_only_to_isoview_data(host):
    for tool in ("crop", "segment", "deadpixel"):
        assert host.apps[f"isoview_{tool}"].available(host) is False
    assert host.apps["align_views"].available(host) is False
    assert host.apps["pollen"].available(host) is False


def test_the_biohpc_and_cloud_windows_draw(host):
    _app._reported.difference_update({"biohpc", "cloud"})
    for app_id in ("biohpc", "cloud"):
        host.apps[app_id].open = True
    host.figure.canvas.force_draw()
    host.figure.canvas.force_draw()
    assert not {"biohpc", "cloud"} & _app._reported


def test_an_array_split_by_roi_gives_one_view_each():
    from mbo_utilities.gui.app.apps.viewer import split_rois

    views, names = split_rois(ThreeRois())
    assert [view.roi for view in views] == [1, 2, 3]
    assert names == ["ROI 1", "ROI 2", "ROI 3"]
    assert not any(view.fix_phase for view in views)
    assert split_rois(imread(movie_data(nt=2, ny=8, nx=8)))[1] is None


def test_saved_rois_beside_the_data_turn_labeling_on(tmp_path):
    from mbo_utilities.annotation import LabelsZarr, RoiLabelStore

    np.save(tmp_path / "movie.npy", movie_data(nt=4, ny=32, nx=32))
    store = RoiLabelStore(1, 32, 32)
    store.add_roi(0, np.pad(np.ones((8, 8), bool), 12))
    LabelsZarr(tmp_path / "manual_labels.zarr").save(store)

    host = build_host(tmp_path / "movie.npy", size=(600, 400))
    try:
        host.figure.show()
        host.figure.canvas.force_draw()
        assert host.apps["manual_roi"].open is True
        assert host.context.manual_roi.counts == [64]
    finally:
        host.close()
        host.viewer.close()


def test_help_and_keybinds_gain_the_roi_pages_with_labeling_on(host, monkeypatch):
    monkeypatch.setattr(imgui, "begin_tab_item", spy_begin_tab_item)
    monkeypatch.setattr(imgui, "text_colored", spy_text_colored)
    for app_id in ("help", "keybinds"):
        host.apps[app_id].open = True
    TAB_LABELS.clear()
    host.figure.canvas.force_draw()
    assert "ROI Labeling" not in TAB_LABELS

    host.apps["manual_roi"].open = True
    host.figure.canvas.force_draw()
    TAB_LABELS.clear()
    COLORED_TEXT.clear()
    host.figure.canvas.force_draw()
    assert "ROI Labeling" in TAB_LABELS
    assert "ROIs" in COLORED_TEXT
    host.apps["manual_roi"].open = False
    host.figure.canvas.force_draw()


def test_a_pipeline_turning_rois_on_gets_the_widget_at_once(host):
    host.context.sync_manual_roi(True)
    assert host.context.manual_roi is not None
    assert host.apps["manual_roi"].open is True
    host.context.sync_manual_roi(False)
    assert host.context.manual_roi is None


def test_the_filter_subtracts_the_mean_before_the_blur():
    frame = np.full((4, 4), 3.0, dtype=np.float32)
    mean = np.full((4, 4), 1.0, dtype=np.float32)
    assert np.allclose(filter_frame(frame, mean=mean), 2.0)


def test_the_blur_smooths_the_frame():
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4, 4] = 1.0
    smoothed = filter_frame(frame, sigma=1.0)
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


def test_space_plays_and_pauses_the_t_slider(host):
    playing = host.viewer._sliders_ui._playing
    host.figure.canvas.force_draw()
    assert not playing[t_slider(host)]
    tap(host, "space")
    assert playing[t_slider(host)]
    tap(host, "space")
    assert not playing[t_slider(host)]


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
    first.viewer.close()

    second = build_host(movie_data(nt=4, ny=16, nx=16), size=(600, 400), store=store)
    assert second.apps["log"].open is True
    assert second.apps["viewer"].open is False
    second.close()
    second.viewer.close()
