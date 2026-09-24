"""The figure's shared top strip: the menu row, the panels features register
on it, and the Signal Quality split (plot on top, table in the right tab).
"""

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
        it asked for.
        """
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
