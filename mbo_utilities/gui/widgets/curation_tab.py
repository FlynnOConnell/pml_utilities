"""The Curation tab: vnoiser event curation controls.

The plots live on the top strip (``gui/event_curation.py``); this tab is the
seam to its source, decision and navigation controls, greyed out with the
install hint when vnoiser is missing.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities.gui._availability import HAS_VNOISER
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.install import VNOISER_HINT

__all__ = ["CurationTabWidget"]


class CurationTabWidget(Widget):
    name = "Curation"
    tab_label = "Curation"
    placement = "tab"
    toggle_key = "vnoiser"
    priority = 47

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def tab_disabled(self) -> str | None:
        if not HAS_VNOISER:
            return f"vnoiser is not installed: {VNOISER_HINT}"
        if getattr(self.parent, "event_curation", None) is None:
            return "Enable Widgets > Event Curation."
        return None

    def wants_focus(self) -> bool:
        widget = getattr(self.parent, "event_curation", None)
        if widget is None:
            return False
        if widget.focus_tab:
            widget.focus_tab = False
            return True
        strip = getattr(self.parent, "top_strip", None)
        return strip is not None and strip.take_right_focus("curation")

    def draw(self) -> None:
        widget = getattr(self.parent, "event_curation", None)
        if widget is None:
            imgui.text_disabled("Event Curation is off.")
            return
        with imgui_ctx.begin_child("##CurationContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none):
            widget.draw_tab()
