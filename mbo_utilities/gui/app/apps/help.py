"""The help pages shipped with the package, one tab each."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui._help_viewer import (
    DOCS,
    MESC_DOC,
    ROI_DOC,
    load_doc,
    render_markdown,
)
from mbo_utilities.gui.app._app import App


class HelpApp(App):
    """The markdown help pages, with the MESc page while a ``.mesc`` unit is open
    and the ROI guide while manual ROI labeling is on.

    ``show(filename)`` opens the window on one page, for a panel's (?).
    """

    id = "help"
    title = "Help"
    menu = "Docs"
    shortcut = "h"
    window = True
    order = 10
    window_size = (650, 550)

    def __init__(self):
        super().__init__()
        # the page to select on the next frame, then None
        self.wanted: str | None = None

    def show(self, filename: str) -> None:
        self.open = True
        self.wanted = filename

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        docs = list(DOCS)
        if "mesc_unit" in (getattr(host.data, "metadata", None) or {}):
            docs.append(("MESc files", MESC_DOC))
        if host.context is not None and host.context.manual_roi is not None:
            docs.append(("ROI Labeling", ROI_DOC))
        shown = docs[0][1]
        if imgui.begin_tab_bar("##help_pages"):
            for name, filename in docs:
                flags = (
                    imgui.TabItemFlags_.set_selected
                    if filename == self.wanted
                    else imgui.TabItemFlags_.none
                )
                if imgui.begin_tab_item(name, None, flags)[0]:
                    shown = filename
                    imgui.end_tab_item()
            imgui.end_tab_bar()
        self.wanted = None
        if imgui.begin_child(
            "##help_page", imgui.ImVec2(0, 0), imgui.ChildFlags_.borders
        ):
            render_markdown(load_doc(shown))
        imgui.end_child()
