"""
Menu bar and process status indicator.

This module contains the main menu bar and process status indicator.
"""

from __future__ import annotations

import webbrowser
from typing import Any

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities.gui._availability import HAS_VNOISER
from mbo_utilities.gui._dialogs import start_open_prompt
from mbo_utilities.gui._imgui_helpers import PopupAutoSize
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.gui.widgets.style_editor import draw_style_menu_item
from mbo_utilities.gui.widgets.widget_toggles import draw_widgets_menu
from mbo_utilities.install import VNOISER_HINT


def draw_menu_bar(parent: Any):
    """Draw the menu row: the File / Widgets / Docs menus, then the process
    status and Metadata Viewer buttons. Drawn in the figure's top strip
    (``gui/_top_strip.py``), which spans the canvas's full width.
    """
    with imgui_ctx.begin_child(
        "menu",
        window_flags=imgui.WindowFlags_.menu_bar,
        child_flags=imgui.ChildFlags_.auto_resize_y
        | imgui.ChildFlags_.always_auto_resize,
    ):
        if imgui.begin_menu_bar():
            if imgui.begin_menu("File", True):
                # Open File / Open Folder: a typed path with the native
                # dialog as a browse shortcut, so a remote kernel still works
                if imgui.menu_item("Open File", "o", p_selected=False, enabled=True)[0]:
                    start_open_prompt(parent, "file")
                if imgui.menu_item(
                    "Open Folder", "Shift+O", p_selected=False, enabled=True
                )[0]:
                    start_open_prompt(parent, "folder")
                imgui.separator()
                if imgui.menu_item(
                    "Set Metadata", "Shift+M", p_selected=False, enabled=True
                )[0]:
                    parent._show_metadata_popup = True
                imgui.separator()
                # Check if current data supports imwrite
                can_save = parent.is_mbo_scan
                if parent.image_widget and parent.image_widget.data:
                    arr = parent.image_widget.data[0]
                    can_save = hasattr(arr, "_imwrite")
                if imgui.menu_item("Save as", "s", p_selected=False, enabled=can_save)[
                    0
                ]:
                    parent.save_as.open_requested = True
                # the curation window (`mbo curate`) on the open PF folder or
                # .mesc, in its own process; its module brings hello_imgui
                from mbo_utilities.gui.curation_viewer import (
                    curation_target,
                    launch_curation_window,
                )

                target = curation_target(getattr(parent, "fpath", None))
                if imgui.menu_item(
                    "Curate",
                    "",
                    p_selected=False,
                    enabled=HAS_VNOISER and target is not None,
                )[0]:
                    launch_curation_window(target)
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                    imgui.set_tooltip(
                        f"vnoiser is not installed: {VNOISER_HINT}"
                        if not HAS_VNOISER
                        else "Open a PF folder or a .mesc line scan first."
                        if target is None
                        else f"Open {target.name} in the curation window (its own window, what `mbo curate` opens)."
                    )
                imgui.separator()
                if imgui.menu_item("Options", "", p_selected=False, enabled=True)[0]:
                    parent._show_options_popup = True
                draw_style_menu_item()
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Sizes, spacing and colours of the running imgui style. "
                        "Saved under ~/.mbo/imgui and applied at the next launch."
                    )
                imgui.end_menu()
            draw_widgets_menu(parent)
            if imgui.begin_menu("Docs", True):
                if imgui.menu_item("Help", "h", p_selected=False, enabled=True)[0]:
                    parent._show_help_popup = True
                if imgui.menu_item("Keybinds", "k", p_selected=False, enabled=True)[0]:
                    parent._show_keybinds_popup = True
                imgui.separator()
                if imgui.menu_item("Online Docs", "", p_selected=False, enabled=True)[
                    0
                ]:
                    webbrowser.open(
                        "https://millerbrainobservatory.github.io/mbo_utilities/"
                    )
                imgui.end_menu()
            # the strip spans the canvas, so the status and metadata buttons
            # fit on the menu row instead of a second line under it
            parent.save_as.clear_stale_progress()
            draw_process_status_indicator(parent, in_menu_bar=True)
            imgui.end_menu_bar()


# label, hotkey hint pairs of the buttons that follow the status button; the
# widths are measured together so the cluster can be right-aligned in one go
_TRAILING_BUTTONS = (("Metadata Viewer", "(m)"), ("Help", "(h)"), ("Keybinds", "(k)"))


def _toggle_metadata_viewer(parent: Any) -> None:
    parent.show_metadata_viewer = not parent.show_metadata_viewer


def _open_help(parent: Any) -> None:
    parent._show_help_popup = True


def _open_keybinds(parent: Any) -> None:
    parent._show_keybinds_popup = not getattr(parent, "_show_keybinds_popup", False)


def _status_cluster_width(status_text: str) -> float:
    """Width of the status button plus every button after it."""
    style = imgui.get_style()
    pad, gap = style.frame_padding.x * 2.0, style.item_spacing.x
    width = imgui.calc_text_size(status_text).x + pad
    for label, hint in _TRAILING_BUTTONS:
        width += gap + imgui.calc_text_size(label).x + pad
        width += gap + imgui.calc_text_size(hint).x
    return width


def _right_align(width: float) -> None:
    """Put the cursor so ``width`` of content ends at the row's right edge."""
    style = imgui.get_style()
    x = imgui.get_window_width() - width - style.item_spacing.x
    if x > imgui.get_cursor_pos_x():
        imgui.set_cursor_pos_x(x)


def process_status(progress_items: list) -> tuple[str, str, imgui.ImVec4]:
    """The status button's label, its widest label and its colour.

    Counts spawned processes, in-process jobs and ``progress_items`` (the
    caller's own running work). The widest label fixes the button's width
    while a running percentage changes, so the click target stays put.
    """
    try:
        from imgui_bundle import icons_fontawesome as fa

        icon_idle = fa.ICON_FA_CIRCLE
        icon_running = fa.ICON_FA_SPINNER
        icon_error = fa.ICON_FA_EXCLAMATION_TRIANGLE
        icon_check = fa.ICON_FA_CHECK_CIRCLE
    except (ImportError, AttributeError):
        icon_idle, icon_running, icon_error, icon_check = (
            "\uf111",
            "\uf110",
            "\uf071",
            "\uf058",
        )

    pm = get_process_manager()
    pm.cleanup_finished()
    # in-process jobs count the same as spawned ones, so a click shows up somewhere
    procs = [*pm.get_running(), *pm.get_jobs()]
    running = [p for p in procs if p.is_alive()]
    completed = [p for p in procs if not p.is_alive() and p.status == "completed"]
    errors = [p for p in procs if not p.is_alive() and p.status == "error"]
    n_running = len(running) + sum(1 for i in progress_items if not i.get("done"))

    if errors:
        text = f"{icon_error} Error ({len(errors)})"
        return text, text, imgui.ImVec4(0.8, 0.2, 0.2, 1.0)
    if n_running:
        color = imgui.ImVec4(0.85, 0.45, 0.0, 1.0)
        if progress_items:
            done = sum(i["progress"] for i in progress_items) / len(progress_items)
            return (
                f"{icon_running} Running ({n_running}) {int(done * 100)}%",
                f"{icon_running} Running ({n_running}) 100%",
                color,
            )
        text = f"{icon_running} Running ({n_running})"
        return text, text, color
    green = imgui.ImVec4(0.15, 0.55, 0.15, 1.0)
    if completed:
        word = "task" if len(completed) == 1 else "tasks"
        text = f"{icon_check} Completed {len(completed)} {word}"
        return text, text, green
    text = f"{icon_idle} Console: Idle"
    return text, text, green


def draw_status_button(text: str, widest: str, color: imgui.ImVec4) -> bool:
    """Draw the rounded status button in ``color``; True when it was clicked."""
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 5.0)
    imgui.push_style_color(imgui.Col_.button, color)
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1, 1, 1, 1))
    imgui.push_style_color(
        imgui.Col_.button_hovered,
        imgui.ImVec4(
            min(color.x + 0.1, 1.0),
            min(color.y + 0.1, 1.0),
            min(color.z + 0.1, 1.0),
            1.0,
        ),
    )
    imgui.push_style_color(imgui.Col_.button_active, color)
    width = imgui.calc_text_size(widest).x + imgui.get_style().frame_padding.x * 2.0
    clicked = imgui.button(f"{text}##process_status", imgui.ImVec2(width, 0.0))
    imgui.pop_style_color(4)
    imgui.pop_style_var()
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
    return clicked


def draw_process_status_indicator(parent: Any, in_menu_bar: bool = False):
    """Draw the process status button and the Metadata / Help / Keybinds
    buttons beside it, pinned to the row's right edge.

    A menu bar already lays its items out in a row, and a ``same_line`` there
    puts the second button back on top of the first, so the caller says which
    it is.
    """
    from mbo_utilities.gui.widgets.progress_bar import _get_active_progress_items

    text, widest, color = process_status(_get_active_progress_items(parent))
    _right_align(_status_cluster_width(widest))
    if draw_status_button(text, widest, color):
        # the console is a plain window, so the status button toggles it
        if getattr(parent, "_process_console_open", False):
            parent._process_console_open = False
        else:
            parent._show_process_console = True

    # metadata, help and keybinds in one dark grey, each with its hotkey greyed beside it
    if not in_menu_bar:
        imgui.same_line()
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 5.0)
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.2, 0.2, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.3, 0.3, 1.0))
    imgui.push_style_color(
        imgui.Col_.button_active, imgui.ImVec4(0.15, 0.15, 0.15, 1.0)
    )
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.9, 0.9, 0.9, 1.0))

    actions = (_toggle_metadata_viewer, _open_help, _open_keybinds)
    for i, ((label, hint), action) in enumerate(zip(_TRAILING_BUTTONS, actions)):
        if i and not in_menu_bar:
            imgui.same_line()
        if imgui.button(label):
            action(parent)
        if not in_menu_bar:
            imgui.same_line(0, 2)
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(hint)

    imgui.pop_style_color(4)
    imgui.pop_style_var()


def _roi_keybinds(parent: Any) -> list[tuple[str, str | None]]:
    """The ROI tool's keys as one more section, when that widget is on.

    The tool used to carry its own Help and Keys buttons; one app has one
    of each, so its keys live here and its guide is a tab in the help
    viewer.
    """
    roi = getattr(parent, "manual_roi", None)
    if roi is None:
        return []
    from mbo_utilities.gui.manual_roi import KEYBINDS

    return [("", ""), ("ROI Labeling", None), *KEYBINDS]


def draw_keybinds_popup(parent: Any):
    """Draw the keybinds cheatsheet popup.

    `_show_keybinds_popup` is the desired state: True = should be open,
    False = should be closed. The 'k' shortcut flips it.
    """
    if not hasattr(parent, "_show_keybinds_popup"):
        parent._show_keybinds_popup = False
    if not hasattr(parent, "_keybinds_popup_actually_open"):
        parent._keybinds_popup_actually_open = False
    if not hasattr(parent, "_keybinds_sizer"):
        parent._keybinds_sizer = PopupAutoSize("Keybinds", auto_resize=False)

    # sync imgui's popup state with desired state
    if parent._show_keybinds_popup and not parent._keybinds_popup_actually_open:
        parent._keybinds_sizer.before_open()
        imgui.open_popup("Keybinds")
        parent._keybinds_popup_actually_open = True

    imgui.set_next_window_size(imgui.ImVec2(320, 380), imgui.Cond_.first_use_ever)
    popup_open = imgui.begin_popup_modal(
        "Keybinds", flags=imgui.WindowFlags_.no_saved_settings
    )[0]
    if popup_open:
        # if user flipped the flag off (e.g. pressed k again), close the popup
        if not parent._show_keybinds_popup:
            imgui.close_current_popup()
            parent._keybinds_popup_actually_open = False
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Keyboard Shortcuts")
        imgui.separator()
        imgui.dummy(imgui.ImVec2(0, 5))

        # keybinds data: (key, description)
        keybinds = [
            ("Navigation", None),
            ("\u2190 / \u2192", "Previous / Next frame (T)"),
            ("\u2191 / \u2193", "Previous / Next z-plane (Z)"),
            ("Shift + \u2190/\u2192", "Jump 10 frames (T)"),
            ("Shift + \u2191/\u2193", "Jump 10 z-planes (Z)"),
            ("", ""),
            ("File", None),
            ("o", "Open file"),
            ("Shift + O", "Open folder"),
            ("Shift + M", "Set metadata"),
            ("s", "Save as"),
            ("", ""),
            ("View", None),
            ("m", "Toggle metadata viewer"),
            ("Shift + L", "Focus metadata search (when viewer open)"),
            ("p / Enter", "Toggle side panel"),
            ("Space", "Play / pause"),
            ("v", "Reset vmin/vmax"),
            ("Shift + V", "Toggle auto-contrast on Z"),
            ("c", "Toggle fix scan-phase"),
            ("Shift + C", "Toggle sub-pixel scan-phase"),
            ("", ""),
            ("Help", None),
            ("h", "Open help"),
            ("k", "Open/close this popup"),
        ]
        keybinds += _roi_keybinds(parent)

        table_flags = (
            imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
        )
        if imgui.begin_table("keybinds_table", 2, table_flags):
            # wide enough for the ROI section's chords ("ctrl+click")
            imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed, 110)
            imgui.table_setup_column("desc", imgui.TableColumnFlags_.width_stretch)

            for key, desc in keybinds:
                if desc is None:
                    # section header
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.dummy(imgui.ImVec2(0, 3))
                    imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), key)
                    imgui.table_next_column()
                elif key == "":
                    # spacer
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.table_next_column()
                else:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text_colored(imgui.ImVec4(0.9, 0.9, 0.5, 1.0), key)
                    imgui.table_next_column()
                    imgui.text(desc)

            imgui.end_table()

        imgui.dummy(imgui.ImVec2(0, 10))
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()
            parent._show_keybinds_popup = False
            parent._keybinds_popup_actually_open = False

        imgui.end_popup()
    else:
        # imgui reports popup is not being drawn (e.g. user pressed escape)
        parent._keybinds_popup_actually_open = False
        parent._show_keybinds_popup = False
