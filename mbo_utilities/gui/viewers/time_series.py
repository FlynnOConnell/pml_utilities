"""Time series viewer for standard calcium imaging data.

Renders the main tab bar. Every tab is a ``placement = "tab"`` widget (see
``gui/widgets/tabs.py``), so this viewer only orders them, honours the
Widgets menu, and keeps one tab's failure from taking the GUI down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui, imgui_ctx  # noqa: F401  (imgui_ctx: tab bodies)

from . import BaseViewer

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["TimeSeriesViewer"]

# tabs that leave the tab bar for a window of their own while the figure docks
DOCKED_TABS = ("Image", "Signal Quality", "Process", "ROIs")


class TimeSeriesViewer(BaseViewer):
    """Viewer for time-series calcium imaging data (TZYX)."""

    name = "Time Series Viewer"

    def __init__(
        self,
        image_widget: ImageWidget,
        fpath: str | list[str],
        parent=None,
        **kwargs,
    ):
        super().__init__(image_widget, fpath, parent=parent, **kwargs)
        self._tab_widgets: list | None = None

    def draw(self) -> None:
        """Draw the main tab bar."""
        docking = self.image_widget.figure.imgui_docking
        if imgui.begin_tab_bar("MainPreviewTabs"):
            for widget in self._widgets():
                if docking and self._label(widget) in DOCKED_TABS:
                    continue
                self._draw_tab(widget)
            imgui.end_tab_bar()

    def draw_docked(self, label: str) -> None:
        from mbo_utilities.gui.widgets import widget_is_visible

        widget = self._widget(label)
        if widget is None:
            return
        if not widget_is_visible(widget):
            imgui.text_disabled("Switched off in the Widgets menu.")
            return
        reason = widget.tab_disabled()
        if reason is not None:
            imgui.push_text_wrap_pos(0.0)
            imgui.text_disabled(reason or f"{label} is not available yet.")
            imgui.pop_text_wrap_pos()
            return
        self._draw_body(widget)

    def docked_wants_focus(self, label: str) -> bool:
        from mbo_utilities.gui.widgets import widget_is_visible

        widget = self._widget(label)
        return (
            widget is not None and widget_is_visible(widget) and widget.wants_focus()
        )

    def _widget(self, label: str):
        """The tab widget captioned ``label``, None when this data has none."""
        for widget in self._widgets():
            if self._label(widget) == label:
                return widget
        return None

    @staticmethod
    def _label(widget) -> str:
        return widget.tab_label or widget.name

    def _widgets(self) -> list:
        """Supported tab widgets, in priority order (built once)."""
        if self._tab_widgets is None:
            from mbo_utilities.gui.widgets import get_tab_widgets

            self._tab_widgets = get_tab_widgets(self.parent)
        return self._tab_widgets

    def cleanup(self) -> None:
        """Release what the tabs put on the figure before the viewer is rebuilt."""
        from mbo_utilities.gui.widgets import cleanup_all_widgets

        cleanup_all_widgets(self._tab_widgets or [])
        self._tab_widgets = None

    def _draw_body(self, widget) -> None:
        """Draw one tab's contents; a failure shows as an error line in its place."""
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
        try:
            widget.draw()
        except Exception as e:
            self.logger.exception(f"{self._label(widget)} tab failed")
            imgui.push_text_wrap_pos(
                imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
            )
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {e}")
            imgui.pop_text_wrap_pos()
        finally:
            imgui.pop_style_var(2)

    def _draw_tab(self, widget) -> None:
        """Draw one tab, unless the Widgets menu has it switched off."""
        from mbo_utilities.gui.widgets import widget_is_visible

        if not widget_is_visible(widget):
            return

        label = self._label(widget)
        reason = widget.tab_disabled()
        if reason is not None:
            imgui.begin_disabled()

        flags = (
            imgui.TabItemFlags_.set_selected
            if widget.wants_focus()
            else imgui.TabItemFlags_.none
        )
        if imgui.begin_tab_item(label, None, flags)[0]:
            self._draw_body(widget)
            imgui.end_tab_item()

        if reason is not None:
            imgui.end_disabled()
            if reason and imgui.is_item_hovered(
                imgui.HoveredFlags_.allow_when_disabled
            ):
                imgui.set_tooltip(reason)
