"""The figure's top edge, shared by everything that wants the full canvas width.

fastplotlib keeps one imgui window per edge, so the menu row, Manual ROI's
trace plot, the Signal Quality plot, the line-scan viewer's traces and the
curation dashboard cannot each own the top. They register a :class:`TopPanel`
here instead: the strip
draws the menu row, then a tab bar over whatever is registered, and shrinks
back to the menu row alone when nothing is. Selecting a panel never selects
anything in the right bar, and nothing there selects a panel.

Work a feature needs every frame regardless of which tab is on top — polling
jobs, keyboard handling, its own floating windows — goes in a frame hook, not
in a panel body.

A grab bar along the strip's bottom edge drags it taller or shorter and
double-clicks shut, the way fastplotlib's right and bottom edge windows work
(it does not draw one for the top edge, so the strip draws its own). Shut,
the strip keeps its tab row so the panels stay discoverable; clicking a tab
opens it again. Dragging pins the height until :meth:`TopStrip.reset_size`;
until then the strip is exactly as tall as the panel showing asked for.

A panel that knows the canvas width it needs to draw on one row says so with
``min_width``; the strip widens the window to it once, as far as the screen
allows.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui, imgui_ctx

__all__ = ["MENU_HEIGHT", "MENU_MIN_WIDTH", "TopPanel", "TopStrip", "strip_height"]

# the menu row on its own: the 20 px menu bar inside the window padding
MENU_HEIGHT = 36
# canvas width that keeps the menu row on one line: the File / Widgets / Docs
# menus and the status cluster at the 14 px font, plus the window padding
MENU_MIN_WIDTH = 560
# the tab row: a 20 px tab bar and the spacing under it
TAB_BAR_HEIGHT = 24
# around a panel body beyond the menu row and the grab bar: the 8 px the menu
# child keeps under its bar, the spacing, the tab row, and the window padding
# under the body
PANEL_PAD = 8 + 4 + TAB_BAR_HEIGHT + 8
# the grab bar along the bottom edge
HANDLE_HEIGHT = 14
# never size or drag the strip so far down that the canvas has less than
# this left to render into, after the other edge windows (the NDWidget's
# slider block along the bottom) have taken theirs
MIN_RENDER_AREA = 150
# and never squeeze a panel under this on its own account
MIN_PANEL = 60


def strip_height(panel_height: int) -> int:
    """Height of the strip showing a panel of ``panel_height``."""
    return MENU_HEIGHT + PANEL_PAD + int(panel_height) + HANDLE_HEIGHT


@dataclass
class TopPanel:
    """One tab in the strip.

    Parameters
    ----------
    key : str
        Identity, for :meth:`TopStrip.focus` and :meth:`TopStrip.unregister`.
    label : str
        Tab caption.
    draw : callable
        Body, called only while the tab is selected.
    height : int
        Pixels the body wants; the strip sizes itself to the tallest panel.
    priority : int
        Tab order, lower first.
    min_width : float
        Canvas width the body needs to draw on one row, when it knows; the
        strip widens the window to the widest panel's once.
    """

    key: str
    label: str
    draw: Callable[[], None]
    height: int = 200
    priority: int = 100
    min_width: float = 0.0


class TopStrip(ImguiWindow):
    """The top edge window: a menu row, then the registered panels as tabs."""

    def __init__(self, figure, draw_menu: Callable[[], None] | None = None):
        super().__init__()
        self.figure = figure
        self.draw_menu = draw_menu
        self.panels: list[TopPanel] = []
        self.hooks: list[Callable[[], None]] = []
        self.active: str | None = None
        self._focus: str | None = None
        self._closed = False
        # our own resize state: ImguiWindow._collapsed drives its edge-window
        # drawing, so shutting the strip must not touch it
        self._shut = False
        self._manual: int | None = None
        self._before_collapse: int | None = None
        self._cursor_set = False
        # the last width the strip widened the window to, so a window the
        # user made narrower afterwards is left alone
        self._fit_tried = 0
        figure.add_imgui_window(
            self, location="top", size=self._want_size(), title=None
        )

    def register(self, panel: TopPanel) -> None:
        """Add or replace the panel with this key."""
        self.panels = sorted(
            [p for p in self.panels if p.key != panel.key] + [panel],
            key=lambda p: (p.priority, p.label),
        )
        if self.active is None:
            self.active = panel.key
        self._resize()

    def unregister(self, key: str) -> None:
        if not any(p.key == key for p in self.panels):
            return
        self.panels = [p for p in self.panels if p.key != key]
        if self.active == key:
            self.active = self.panels[0].key if self.panels else None
        self._resize()

    def has(self, key: str) -> bool:
        return any(p.key == key for p in self.panels)

    def add_hook(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` at the top of every frame, whatever tab is selected."""
        if fn not in self.hooks:
            self.hooks.append(fn)

    def remove_hook(self, fn: Callable[[], None]) -> None:
        if fn in self.hooks:
            self.hooks.remove(fn)

    def close(self) -> None:
        """Give the top edge back to the figure."""
        if self._closed:
            return
        self._closed = True
        if self.figure.imgui_windows.get("top") is self:
            self.figure.remove_imgui_window("top")

    def focus(self, key: str) -> None:
        """Select ``key`` on the next frame."""
        if self.has(key):
            self._focus = key

    def _panel(self, key: str | None) -> TopPanel | None:
        return next((p for p in self.panels if p.key == key), None)

    @property
    def handle_height(self) -> int:
        """Thickness of the grab bar along the bottom edge."""
        return HANDLE_HEIGHT

    @property
    def shut_size(self) -> int:
        """Height of the strip with the panels shut: the menu row, the tab row
        and the bar, with no body between them.
        """
        return strip_height(0)

    @property
    def collapsed(self) -> bool:
        """True while the panels are shut and only the menu row shows."""
        return self._shut

    def toggle_collapsed(self) -> None:
        """Shut the panels away, or bring them back at the previous height."""
        if self._shut:
            self._shut = False
            self._manual, self._before_collapse = self._before_collapse, None
        else:
            self._before_collapse = self._manual
            self._shut = True
        self._resize()

    def resize_to(self, size: int) -> None:
        """Hold the strip at ``size`` px instead of sizing it to its panel."""
        self._shut = False
        self._manual = max(int(size), self.shut_size)
        self._resize()

    def reset_size(self) -> None:
        """Size the strip to whatever panel is showing again."""
        self._shut = False
        self._manual = self._before_collapse = None
        self._resize()

    def _want_size(self) -> int:
        """Menu row plus the selected panel, unless the user set a height.

        The strip is as tall as what it is *showing*, not as tall as the
        tallest thing registered. It also leaves the images at least
        ``MIN_RENDER_AREA`` after the other edge windows, so a tall panel
        (the curation dashboard over the line traces) on a short window
        shrinks instead of pushing the viewport negative. A drag on the
        grab bar pins the height until :meth:`reset_size`.
        """
        if not self.panels:
            return MENU_HEIGHT
        if self._shut:
            return self.shut_size
        if self._manual is not None:
            return self._manual
        panel = self._panel(self.active)
        height = (
            panel.height if panel is not None else max(p.height for p in self.panels)
        )
        try:
            canvas_height = float(self.figure.canvas.get_logical_size()[1])
        except Exception:
            canvas_height = 0.0
        if canvas_height > 0:
            room = (
                canvas_height - self._other_edges() - MIN_RENDER_AREA - strip_height(0)
            )
            height = min(height, max(room, MIN_PANEL))
        return strip_height(height)

    def _other_edges(self) -> float:
        """Canvas height the other edge windows take (the bottom one; the
        strip is the only top window).
        """
        try:
            return float(self.figure._edge_size("bottom"))
        except Exception:
            return 0.0

    def _resize(self) -> None:
        want = self._want_size()
        if self.size != want:
            self.size = want

    def update(self) -> None:
        # the selected panel sets the height; resize before drawing so the
        # rect the figure laid out this frame is the one we fill
        self._resize()
        self._fit_width()
        for hook in list(self.hooks):
            hook()
        if self.draw_menu is not None:
            self.draw_menu()
        if not self.panels:
            return
        # the tabs live above the grab bar, never under it; the body clips
        # rather than scrolls, panels size themselves to the room they get
        with imgui_ctx.begin_child(
            "##strip_body",
            imgui.ImVec2(0, -float(self.handle_height)),
            window_flags=imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse,
        ):
            self._draw_tabs()
        self._draw_handle()

    def _fit_width(self) -> None:
        want = math.ceil(max((p.min_width for p in self.panels), default=0.0))
        width, height = self.figure.canvas.get_logical_size()
        if want <= width or want == self._fit_tried:
            return
        self._fit_tried = want
        from mbo_utilities.gui.run_gui import screen_box

        box = screen_box()
        if box is not None:
            want = min(want, box[0])
        if want > width:
            self.figure.canvas.set_logical_size(want, height)

    def _draw_handle(self) -> None:
        """The grab bar along the bottom edge: drag to resize, double click
        to shut or reopen the panels.
        """
        thickness = float(self.handle_height)
        imgui.set_cursor_pos(imgui.ImVec2(0.0, imgui.get_window_height() - thickness))
        imgui.invisible_button(
            "##top_resize", imgui.ImVec2(imgui.get_window_width(), thickness)
        )
        hovered, active = imgui.is_item_hovered(), imgui.is_item_active()
        rect_min, rect_max = imgui.get_item_rect_min(), imgui.get_item_rect_max()

        if hovered and imgui.is_mouse_double_clicked(0):
            self.toggle_collapsed()

        if hovered or active:
            if not self._cursor_set:
                self._set_cursor("ns_resize")
                self._cursor_set = True
        elif self._cursor_set:
            self._set_cursor("default")
            self._cursor_set = False

        if active and imgui.is_mouse_dragging(0):
            # the bar is the strip's bottom edge, so dragging down grows it -
            # but never past leaving the canvas too little to render into
            delta = imgui.get_mouse_drag_delta(0).y
            imgui.reset_mouse_drag_delta(0)
            if delta > 0 and self._render_height() - delta < MIN_RENDER_AREA:
                delta = 0.0
            if delta:
                self.resize_to(round(self.size + delta))

        draw = imgui.get_window_draw_list()
        strong = hovered or active
        line = imgui.get_color_u32(
            imgui.ImVec4(0.9, 0.9, 0.9, 1.0)
            if strong
            else imgui.ImVec4(0.5, 0.5, 0.5, 0.8)
        )
        draw.add_rect_filled(
            rect_min,
            rect_max,
            imgui.get_color_u32(
                imgui.ImVec4(0.2, 0.2, 0.2, 0.8)
                if strong
                else imgui.ImVec4(0.15, 0.15, 0.15, 0.6)
            ),
        )
        mid_y = (rect_min.y + rect_max.y) * 0.5
        center_x = (rect_min.x + rect_max.x) * 0.5
        for i in (-1, 0, 1):
            draw.add_circle_filled(imgui.ImVec2(center_x + i * 7.0, mid_y), 2, line)

    def _set_cursor(self, name: str) -> None:
        try:
            self.figure.canvas.set_cursor(name)
        except Exception:
            pass

    def _render_height(self) -> float:
        """Canvas height left for pygfx, or a big number when unknown."""
        try:
            return float(self.figure.get_pygfx_render_area()[3])
        except Exception:
            return float("inf")

    def _draw_tabs(self) -> None:
        focus, self._focus = self._focus, None
        reopen = False
        if imgui.begin_tab_bar("##top_strip_tabs"):
            active = None
            for panel in self.panels:
                flags = (
                    imgui.TabItemFlags_.set_selected
                    if focus == panel.key
                    else imgui.TabItemFlags_.none
                )
                selected = imgui.begin_tab_item(panel.label, None, flags)[0]
                # shut, the tabs are only headers: a click on any of them opens the strip
                if self._shut and imgui.is_item_clicked():
                    reopen = True
                if selected:
                    active = panel.key
                    if not self._shut:
                        panel.draw()
                    imgui.end_tab_item()
            imgui.end_tab_bar()
            if active is not None:
                self.active = active
        if reopen:
            self.toggle_collapsed()
