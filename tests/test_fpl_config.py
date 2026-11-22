"""The fastplotlib config every mbo figure gets, and the window fit that
goes with it.
"""

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

    def test_the_image_starts_under_the_title(self):
        """fastplotlib draws the title 4 px below the frame's top; the render
        area must start under its text or the image covers it.
        """
        from mbo_utilities.gui._ndviewer import MboNDViewer

        data = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        try:
            iw.show()
            frame = iw.figure[0, 0].frame
            title_bottom = frame.rect[1] + iw.figure[0, 0].title.font_size + 4
            assert frame.get_render_rect()[1] >= title_bottom
        finally:
            iw.close()

    def test_the_image_fills_its_viewport(self):
        """auto_scale sets the camera to the image's bounding box; zoom 1 shows
        exactly that, so the camera's world window matches the image.
        """
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
        from mbo_utilities.gui._fpl_config import (
            HISTOGRAM_WIDTH,
            SUBPLOT_PAD_H,
            SUBPLOT_PAD_W,
        )
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
        from mbo_utilities.gui._fpl_config import (
            HISTOGRAM_WIDTH,
            SUBPLOT_PAD_H,
            SUBPLOT_PAD_W,
        )
        from mbo_utilities.gui.run_gui import fit_figure_size

        w, h = fit_figure_size((4000, 1000), (256, 256), grid=(1, 2), right=300)
        img = 1000 - SUBPLOT_PAD_H
        assert h == 1000
        assert w == 2 * (img + HISTOGRAM_WIDTH + SUBPLOT_PAD_W) + 300

    @pytest.mark.parametrize("roi", [False, True])
    def test_create_image_widget_sizes_the_canvas_to_the_data(self, monkeypatch, roi):
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui._ndviewer import sliders_height
        from mbo_utilities.gui._top_strip import (
            MENU_HEIGHT,
            MENU_MIN_WIDTH,
        )
        from mbo_utilities.gui.run_gui import (
            _PREVIEW_WIDTH,
            _create_image_widget,
            fit_figure_size,
        )
        from mbo_utilities.gui.widgets.widget_toggles import (
            set_widget_enabled,
            widget_enabled,
        )

        monkeypatch.setenv("RENDERCANVAS_FORCE_OFFSCREEN", "1")
        was = {k: widget_enabled(k) for k in ("manual_roi", "signal_quality")}
        set_widget_enabled("manual_roi", roi, persist=False)
        set_widget_enabled("signal_quality", False, persist=False)
        data = np.random.default_rng(0).random((4, 1, 3, 40, 80)).astype(np.float32)
        try:
            # with no traces to load the Traces tab takes no room, so only the
            # menu row is planned for; the ROI controls are a right-bar tab
            want = fit_figure_size(
                (1000, 1000),
                (40, 80),
                top=MENU_HEIGHT,
                bottom=sliders_height(2),
                right=_PREVIEW_WIDTH,
                min_width=MENU_MIN_WIDTH,
            )
            iw = _create_image_widget(NumpyArray(data, dims="TCZYX"), widget="preview")
            try:
                got = tuple(int(v) for v in iw.figure.canvas.get_logical_size())
                assert got == want
                strip = iw.figure.imgui_windows["top"]
                for _ in range(3):
                    iw.figure.canvas.draw()
                assert [p.key for p in strip.panels] == (["traces"] if roi else [])
                assert strip.shown == []
                assert strip.size == MENU_HEIGHT
            finally:
                iw.close()
        finally:
            for key, value in was.items():
                set_widget_enabled(key, value, persist=False)

    def test_a_recording_with_traces_and_motion_opens_on_both(
        self, monkeypatch, tmp_path
    ):
        """A suite2p plane with ROIs and rigid offsets: the subplot carries the
        plane's name, the strip is sized for the motion panel over the traces,
        and the first ROI's trace is plotted.
        """
        from mbo_utilities import imread
        from mbo_utilities.gui._top_strip import strip_height
        from mbo_utilities.gui.manual_roi import MOTION_PANEL_HEIGHT
        from mbo_utilities.gui.run_gui import _create_image_widget
        from mbo_utilities.gui.widgets.widget_toggles import (
            set_widget_enabled,
            widget_enabled,
        )

        monkeypatch.setenv("RENDERCANVAS_FORCE_OFFSCREEN", "1")
        plane = tmp_path / "zplane01_tp00001-00020"
        plane.mkdir()
        nt, ny, nx = 20, 64, 64
        rng = np.random.default_rng(0)
        rng.integers(0, 100, (nt, ny, nx)).astype(np.int16).tofile(plane / "data.bin")
        ops = {"Ly": ny, "Lx": nx, "fs": 5.0, "nframes": nt, "plane": 1}
        ops["xoff"], ops["yoff"] = rng.standard_normal((2, nt)).astype(np.float32)
        np.save(plane / "ops.npy", ops, allow_pickle=True)
        ypix, xpix = np.mgrid[10:14, 10:14].reshape(2, -1)
        stat = np.array(
            [{"ypix": ypix, "xpix": xpix, "lam": np.ones(16, np.float32)}],
            dtype=object,
        )
        np.save(plane / "stat.npy", stat)
        np.save(plane / "F.npy", rng.random((1, nt)).astype(np.float32) + 10)
        np.save(plane / "Fneu.npy", np.ones((1, nt), np.float32))
        np.save(plane / "spks.npy", np.ones((1, nt), np.float32))
        np.save(plane / "iscell.npy", np.array([[1, 0.9]], np.float32))

        was = widget_enabled("signal_quality")
        set_widget_enabled("signal_quality", False, persist=False)
        try:
            iw = _create_image_widget(imread(plane), widget="preview")
            try:
                assert iw.figure[0, 0].title.text == plane.name
                strip = iw.figure.imgui_windows["top"]
                for _ in range(3):
                    iw.figure.canvas.draw()
                (panel,) = strip.panels
                assert panel.height == MOTION_PANEL_HEIGHT
                assert strip.size == strip_height(MOTION_PANEL_HEIGHT)
                roi = panel.draw.__self__
                assert roi.selected_derived == (0, 0)
                assert len(roi._plot_lines()[1]) == 1
            finally:
                iw.close()
        finally:
            set_widget_enabled("signal_quality", was, persist=False)
            set_widget_enabled("manual_roi", False, persist=False)
