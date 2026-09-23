"""The figure's shared top strip: the menu row, the panels features register
on it, and the Signal Quality split (plot on top, table in the right tab)."""

import time

import numpy as np
import pytest

FIGURE_SIZE = (900, 700)


@pytest.fixture
def figure():
    from mbo_utilities.gui._ndviewer import MboNDViewer

    data = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)
    iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
    iw.show()
    yield iw.figure
    iw.close()


def panel(key, label="P", height=100, priority=100, min_width=0.0):
    from mbo_utilities.gui._top_strip import TopPanel

    return TopPanel(key, label, lambda: None, height, priority, min_width)


class TestTopStrip:
    def test_claims_the_top_edge_and_sizes_to_the_menu_row(self, figure):
        from mbo_utilities.gui._top_strip import MENU_HEIGHT, TopStrip

        strip = TopStrip(figure)
        assert figure.imgui_windows["top"] is strip
        assert strip.size == MENU_HEIGHT
        strip.close()
        assert figure.imgui_windows.get("top") is None

    def test_sizes_to_the_selected_panel_and_back(self, figure):
        from mbo_utilities.gui._top_strip import MENU_HEIGHT, TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", height=100, priority=10))
        short = strip.size
        assert short > MENU_HEIGHT
        # a taller panel that is not selected does not stretch the strip
        strip.register(panel("b", height=240, priority=20))
        assert strip.size == short
        strip.active = "b"
        strip._resize()
        assert strip.size > short
        strip.unregister("b")
        assert strip.size == short
        strip.unregister("a")
        assert strip.size == MENU_HEIGHT

    def test_a_panel_taller_than_the_window_leaves_the_images_room(self, figure):
        from mbo_utilities.gui._top_strip import MIN_RENDER_AREA, TopStrip

        strip = TopStrip(figure)
        canvas_height = figure.canvas.get_logical_size()[1]
        strip.register(panel("a", height=int(canvas_height * 3)))
        bottom = figure._edge_size("bottom")
        assert canvas_height - strip.size - bottom >= MIN_RENDER_AREA
        # the user may still drag it taller than the automatic cap
        auto = strip.size
        strip.resize_to(auto + 40)
        assert strip.size > auto

    def test_registering_the_same_key_replaces(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", label="one"))
        strip.register(panel("a", label="two"))
        assert [p.label for p in strip.panels] == ["two"]

    def test_panels_order_by_priority(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("late", priority=50))
        strip.register(panel("early", priority=10))
        assert [p.key for p in strip.panels] == ["early", "late"]

    def test_unregistering_the_active_panel_moves_on(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", priority=10))
        strip.register(panel("b", priority=20))
        assert strip.active == "a"
        strip.unregister("a")
        assert strip.active == "b"
        strip.unregister("b")
        assert strip.active is None

    def test_a_taller_window_does_not_stretch_the_strip(self, figure):
        """Every extra pixel goes to the images; the panel keeps the height
        it asked for."""
        from mbo_utilities.gui._top_strip import TopStrip, strip_height

        strip = TopStrip(figure)
        strip.register(panel("a", height=100))
        assert strip.size == strip_height(100)
        figure.canvas.set_logical_size(FIGURE_SIZE[0], FIGURE_SIZE[1] * 2)
        strip._resize()
        assert strip.size == strip_height(100)

    def test_hooks_run_once_per_frame(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        calls = []
        hook = lambda: calls.append(1)  # noqa: E731
        strip.add_hook(hook)
        strip.add_hook(hook)  # adding twice must not double it
        for _ in range(2):
            figure.canvas.draw()
        assert calls == [1, 1]
        strip.remove_hook(hook)
        figure.canvas.draw()
        assert calls == [1, 1]


class TestSignalQualitySplit:
    """The plot goes on the top strip, the table stays in the right tab."""

    @pytest.fixture(autouse=True)
    def _signal_quality_on(self):
        """The widget ships off; these tests are about it being on."""
        from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled

        set_widget_enabled("signal_quality", True, persist=False)
        yield
        set_widget_enabled("signal_quality", False, persist=False)

    @staticmethod
    def _gui(shape=(4, 1, 5, 32, 32)):
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui.run_gui import _create_image_widget
        from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget

        data = np.random.default_rng(0).random(shape).astype(np.float32)
        iw = _create_image_widget(
            NumpyArray(data, dims="TCZYX"),
            widget="preview",
            figure_kwargs_override={"size": FIGURE_SIZE},
        )
        gui = next(
            w for w in iw.figure.imgui_windows.values()
            if isinstance(w, PreviewDataWidget)
        )
        return iw, gui

    @staticmethod
    def _fake_zstats(gui, n=5):
        rng = np.random.default_rng(1)
        stats = {
            "mean": rng.random(n) * 100,
            "std": rng.random(n) * 10,
            "snr": rng.random(n) * 5,
        }
        gui._zstats = [{(): stats}]
        gui._zstats_done = [True]
        gui.nz = n

    def test_menu_draws_in_the_strip_not_the_right_panel(self, monkeypatch):
        import mbo_utilities.gui.widgets.preview_data as pd

        iw, gui = self._gui()
        try:
            assert gui.top_strip is iw.figure.imgui_windows["top"]
            calls = []
            monkeypatch.setattr(pd, "draw_menu_bar", lambda parent: calls.append(1))
            iw.figure.canvas.draw()
            assert calls == [1], "the menu row draws exactly once a frame"
            # ... and the strip is the only thing drawing it
            gui.top_strip.draw_menu = None
            calls.clear()
            iw.figure.canvas.draw()
            assert calls == []
        finally:
            iw.close()

    def test_panel_appears_once_zstats_are_done(self):
        iw, gui = self._gui()
        try:
            assert not gui.top_strip.has("zstats")
            self._fake_zstats(gui)
            gui._sync_top_panels()
            assert gui.top_strip.has("zstats")
            gui._zstats_done = [False]
            gui._sync_top_panels()
            assert not gui.top_strip.has("zstats")
        finally:
            iw.close()

    def test_panel_follows_the_widgets_menu_toggle(self):
        from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled

        iw, gui = self._gui()
        try:
            self._fake_zstats(gui)
            gui._sync_top_panels()
            assert gui.top_strip.has("zstats")
            set_widget_enabled("signal_quality", False, persist=False)
            gui._sync_top_panels()
            assert not gui.top_strip.has("zstats")
        finally:
            iw.close()

    def test_the_two_halves_draw(self):
        import traceback

        iw, gui = self._gui()
        try:
            self._fake_zstats(gui)
            gui._sync_top_panels()
            errors = []

            def body():
                for draw in (gui.draw_stats_plot, gui.draw_stats_section):
                    try:
                        draw()
                    except Exception:
                        errors.append(traceback.format_exc())

            gui.top_strip._update_calls[:] = [body]
            for _ in range(2):
                iw.figure.canvas.draw()
            assert not errors, errors[0]
        finally:
            iw.close()


class TestTopStripResize:
    """The grab bar makes the strip adjustable and collapsible, the way
    fastplotlib's right and bottom edge windows are."""

    def test_collapse_shuts_to_the_menu_row_and_back(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", height=180))
        tall = strip.size
        assert not strip.collapsed

        strip.toggle_collapsed()
        assert strip.collapsed
        assert strip.size == strip.shut_size < tall

        strip.toggle_collapsed()
        assert not strip.collapsed
        assert strip.size == tall

    def test_a_drag_pins_the_height_over_the_panel_request(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        roi = panel("a", height=180)
        strip.register(roi)
        strip.resize_to(320)
        assert strip.size == 320

        # a panel asking for more no longer moves it
        roi.height = 600
        strip._resize()
        assert strip.size == 320

        strip.reset_size()
        assert strip.size != 320

    def test_resize_never_goes_under_the_shut_height(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", height=180))
        strip.resize_to(10)
        assert strip.size == strip.shut_size

    def test_collapse_remembers_a_pinned_height(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", height=180))
        strip.resize_to(300)
        strip.toggle_collapsed()
        assert strip.size == strip.shut_size
        strip.toggle_collapsed()
        assert strip.size == 300

    def test_reset_size_clears_a_collapse_too(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", height=180))
        auto = strip.size
        strip.toggle_collapsed()
        strip.reset_size()
        assert not strip.collapsed
        assert strip.size == auto

    def test_the_bare_strip_has_no_handle(self, figure):
        from mbo_utilities.gui._top_strip import MENU_HEIGHT, TopStrip

        # nothing registered: there is no panel to shut, so no bar is drawn
        strip = TopStrip(figure)
        assert strip.size == MENU_HEIGHT

    def test_shut_keeps_the_tab_row(self, figure):
        """The tabs stay visible so the user knows the panels are there."""
        from imgui_bundle import imgui

        from mbo_utilities.gui._top_strip import TopStrip, strip_height

        strip = TopStrip(figure)
        drawn = []
        strip.register(panel("a", height=180))
        strip.panels[0].draw = lambda: drawn.append(1)
        strip.toggle_collapsed()
        assert strip.size == strip_height(0)
        bars = []
        real = imgui.begin_tab_bar
        imgui.begin_tab_bar = lambda *a, **k: bars.append(1) or real(*a, **k)
        try:
            figure.canvas.draw()
        finally:
            imgui.begin_tab_bar = real
        assert bars == [1]
        assert drawn == [], "a shut strip draws the tab headers, not the body"

    def test_clicking_a_tab_opens_a_shut_strip(self, figure, monkeypatch):
        from imgui_bundle import imgui

        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", height=180))
        tall = strip.size
        strip.toggle_collapsed()
        monkeypatch.setattr(imgui, "is_item_clicked", lambda *a, **k: True)
        figure.canvas.draw()
        assert not strip.collapsed
        strip._resize()
        assert strip.size == tall

    def test_a_panel_that_needs_more_width_widens_the_window_once(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        width, height = figure.canvas.get_logical_size()
        strip.register(panel("a", height=100, min_width=width + 300))
        figure.canvas.draw()
        assert figure.canvas.get_logical_size()[0] == width + 300
        # the user narrows it again: the strip does not fight back
        figure.canvas.set_logical_size(width, height)
        figure.canvas.draw()
        assert figure.canvas.get_logical_size()[0] == width


class TestMenuRowCluster:
    """The status / metadata / help / keybinds buttons sit at the right end
    of the menu row, and the status button says what is idle."""

    @staticmethod
    def _gui(shape=(4, 1, 5, 32, 32)):
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui.run_gui import _create_image_widget
        from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget

        data = np.random.default_rng(0).random(shape).astype(np.float32)
        iw = _create_image_widget(
            NumpyArray(data, dims="TCZYX"),
            widget="preview",
            figure_kwargs_override={"size": FIGURE_SIZE},
        )
        gui = next(
            w for w in iw.figure.imgui_windows.values()
            if isinstance(w, PreviewDataWidget)
        )
        return iw, gui

    def _buttons(self):
        """``[(label, cursor_x, window_width)]`` for one drawn frame."""
        from imgui_bundle import imgui

        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        iw, gui = self._gui()
        # the status button reports whatever the shared process manager is
        # holding, and other tests leave finished work in it; empty it just
        # before the frame that is measured. The widget's own z-stats thread
        # reports there too, so a first frame starts it and the measured frame
        # waits for it to finish
        iw.figure.canvas.draw()
        deadline = time.time() + 30
        while any(gui._zstats_running) and time.time() < deadline:
            time.sleep(0.05)
        pm = get_process_manager()
        for job in pm.get_jobs():
            pm.clear_job(job.job_id)
        pm._processes.clear()
        seen = []
        real = imgui.button

        def spy(label, *args, **kwargs):
            seen.append((label, imgui.get_cursor_pos_x(), imgui.get_window_width()))
            return real(label, *args, **kwargs)

        imgui.button = spy
        try:
            iw.figure.canvas.draw()
        finally:
            imgui.button = real
            iw.close()
        return seen

    def test_the_cluster_is_named_and_right_aligned(self):
        seen = self._buttons()
        labels = [label for label, _x, _w in seen]
        status = [row for row in seen if "Console: Idle" in row[0]]
        assert status, f"the status button names the console: {labels}"
        # one Help and one Keybinds button, not a second ROI pair
        assert any(label == "Help" for label, _x, _w in seen), labels
        assert any(label == "Keybinds" for label, _x, _w in seen), labels
        assert not [label for label in labels if label.startswith("ROI ")], labels
        # the whole cluster starts past the middle of the row
        _label, x, width = status[0]
        assert x > width * 0.5, f"status button at {x} of {width}"

    def test_the_panel_has_no_title_bar(self):
        """fastplotlib draws a custom full-width title box for an edge window
        with a title; a static "Data Preview" label only cost the panel a row
        of height."""
        iw, gui = self._gui()
        try:
            assert gui._title is None
        finally:
            iw.close()
