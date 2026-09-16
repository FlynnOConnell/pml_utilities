"""The fastplotlib config every mbo figure gets, and the window fit that
goes with it."""

import numpy as np
import pytest

FIGURE_SIZE = (600, 500)


class TestConfig:
    def test_subplots_come_thin_and_without_a_toolbar(self):
        from mbo_utilities.gui._fpl_config import FRAME_SPACING, TITLE_FONT_SIZE
        from mbo_utilities.gui._ndviewer import MboNDViewer

        data = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        try:
            iw.show()
            subplot = iw.figure[0, 0]
            assert subplot.toolbar is False
            assert subplot.frame._spacing["bottom"] == FRAME_SPACING["bottom"]
            assert subplot.title.font_size == TITLE_FONT_SIZE
            assert subplot.axes.visible is False
        finally:
            iw.close()

    def test_the_image_fills_its_viewport(self):
        """auto_scale sets the camera to the image's bounding box; zoom 1 shows
        exactly that, so the camera's world window matches the image."""
        from mbo_utilities.gui._ndviewer import MboNDViewer

        data = np.random.default_rng(0).random((4, 30, 60)).astype(np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        try:
            iw.show()
            for _ in range(3):
                iw.figure.canvas.draw()
            camera = iw.figure[0, 0].camera
            assert camera.zoom == pytest.approx(1.0)
            assert (camera.width, camera.height) == pytest.approx((60, 30))
        finally:
            iw.close()


class TestFitFigureSize:
    def test_the_image_fills_the_height_and_the_width_follows(self):
        from mbo_utilities.gui._fpl_config import HISTOGRAM_WIDTH, SUBPLOT_PAD_H, SUBPLOT_PAD_W
        from mbo_utilities.gui.run_gui import fit_figure_size

        w, h = fit_figure_size((1800, 1000), (512, 512), top=36, bottom=107, right=300)
        img = 1000 - 36 - 107 - SUBPLOT_PAD_H
        assert h == 1000
        assert w == img + HISTOGRAM_WIDTH + SUBPLOT_PAD_W + 300

    def test_min_width_only_widens(self):
        from mbo_utilities.gui.run_gui import fit_figure_size

        narrow = fit_figure_size((1800, 600), (512, 512))
        wide = fit_figure_size((1800, 600), (512, 512), min_width=1400)
        assert wide == (1400, narrow[1])

    def test_a_wide_image_is_capped_by_the_box_width(self):
        from mbo_utilities.gui.run_gui import fit_figure_size

        w, h = fit_figure_size((1200, 1000), (100, 1000), top=36)
        assert w == 1200
        assert h < 1000

    def test_the_grid_multiplies_columns(self):
        from mbo_utilities.gui._fpl_config import HISTOGRAM_WIDTH, SUBPLOT_PAD_H, SUBPLOT_PAD_W
        from mbo_utilities.gui.run_gui import fit_figure_size

        w, h = fit_figure_size((4000, 1000), (256, 256), grid=(1, 2), right=300)
        img = 1000 - SUBPLOT_PAD_H
        assert h == 1000
        assert w == 2 * (img + HISTOGRAM_WIDTH + SUBPLOT_PAD_W) + 300

    @pytest.mark.parametrize("roi", [False, True])
    def test_create_image_widget_sizes_the_canvas_to_the_data(self, monkeypatch, roi):
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui._ndviewer import sliders_height
        from mbo_utilities.gui._top_strip import MENU_HEIGHT, MENU_MIN_WIDTH, strip_height
        from mbo_utilities.gui.manual_roi import PANEL_HEIGHT, roi_panel_min_width
        from mbo_utilities.gui.run_gui import _create_image_widget, fit_figure_size
        from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled, widget_enabled

        monkeypatch.setenv("RENDERCANVAS_FORCE_OFFSCREEN", "1")
        was = {k: widget_enabled(k) for k in ("manual_roi", "signal_quality")}
        set_widget_enabled("manual_roi", roi, persist=False)
        set_widget_enabled("signal_quality", False, persist=False)
        data = np.random.default_rng(0).random((4, 1, 3, 40, 80)).astype(np.float32)
        try:
            # the ROI panel on the strip is planned for: its height and a width
            # that keeps its cards on one row (the card count reads the same
            # toggles the widget does, so it is taken before the toggle goes back)
            want = fit_figure_size(
                (1000, 1000), (40, 80),
                top=strip_height(PANEL_HEIGHT) if roi else MENU_HEIGHT,
                bottom=sliders_height(2), right=300,
                min_width=max(MENU_MIN_WIDTH, roi_panel_min_width()) if roi else MENU_MIN_WIDTH,
            )
            iw = _create_image_widget(NumpyArray(data, dims="TCZYX"), widget="preview")
            try:
                got = tuple(int(v) for v in iw.figure.canvas.get_logical_size())
                assert got == want
                strip = iw.figure.imgui_windows["top"]
                for _ in range(3):
                    iw.figure.canvas.draw()
                panels = [(p.key, p.height) for p in strip.panels]
                assert strip.size == (strip_height(PANEL_HEIGHT) if roi else MENU_HEIGHT), panels
            finally:
                iw.close()
        finally:
            for key, value in was.items():
                set_widget_enabled(key, value, persist=False)
