"""Keyboard handling: the ROI tool's binds and the per-frame key poll."""

from __future__ import annotations

from imgui_bundle import imgui
from masknmf.visualization.imgui import (
    draw_keybinds_popup,
)
from masknmf.visualization.imgui.keybinds import pressed

from mbo_utilities.gui._keyboard import claim_keys
from mbo_utilities.gui.roi._shared import (
    _CLAIMED_KEYS,
    ROI_KEYS,
)


class KeysMixin:
    """Keyboard handling: the ROI tool's binds and the per-frame key poll."""

    def _keys_live(self) -> bool:
        """The pointer is over the image (no imgui window under it), over one
        of this tool's panels drawn last frame, or the keybinds window is up.
        """
        io = imgui.get_io()
        return (
            not io.want_capture_mouse
            or imgui.get_frame_count() - self._panel_hovered <= 1
            or self.keybinds_open
        )

    def _note_hover(self) -> None:
        """Stamp this frame when the pointer is over the panel being drawn."""
        if imgui.is_window_hovered(imgui.HoveredFlags_.child_windows):
            self._panel_hovered = imgui.get_frame_count()

    def handle_keys(self):
        """Masknmf's DEMIXING keys with its labeling keys and this tool's own
        (``ROI_KEYS``), only while the pointer is where this tool draws.
        """
        io = imgui.get_io()
        if io.want_text_input or not self._keys_live():
            return
        claim_keys(_CLAIMED_KEYS)
        keys = ROI_KEYS
        if pressed(keys["delete"]):
            self.delete_selected()
        if pressed(keys["escape"]):
            if self.keybinds_open:
                self.keybinds_open = False
            elif self.region_selector is not None:
                self._drop_region()
            else:
                self.deselect()
        if pressed(keys["select_all"]):
            self.select_all()
        if pressed(keys["undo"]):
            self.undo()
        stride = 10 if io.key_shift else 1
        if pressed(keys["down"]):
            self.step(stride)
        if pressed(keys["up"]):
            self.step(-stride)
        if pressed(keys["right"]):
            self.set_frame(self.current_frame() + stride)
        if pressed(keys["left"]):
            self.set_frame(self.current_frame() - stride)
        if pressed(keys["masks"]):
            self.set_masks(not self.show_masks)
        if pressed(keys["contours"]):
            self.set_contours(not self.show_contours)
        if pressed(keys["rings"]):
            self.cycle_contour_shape()
        if pressed(keys["follow"]):
            self.toggle_follow()
        if pressed(keys["trace_follow"]) and self.trace_plot is not None:
            self.trace_plot.follow = not self.trace_plot.follow
        if pressed(keys["pixel_trace"]):
            self.set_pixel_traces(not self.pixel_traces)
        if pressed(keys["roi"]):
            if self.region_selector is not None and not self._drawing():
                self._commit_region()
            elif self.region_selector is None:
                self._start_region()
        if pressed(keys["poly"]):
            self._start_region()
        if pressed(keys["help"]):
            self.open_guide()
        if pressed(keys["keybinds"]):
            self.keybinds_open = not self.keybinds_open
        if pressed(keys["unlabeled"]):
            self.next_unlabeled()
        if pressed(keys["run"]):
            self.run_selection()
        if self.selected_derived is not None:
            if pressed(keys["promote"]):
                self.promote_derived(*self.selected_derived)
            if pressed(keys["discard"]):
                self.discard_derived(*self.selected_derived, advance=True)
            if pressed(keys["accept"]):
                self.set_accepted(*self.selected_derived)
        if self._selection_pair() is not None or self.buffer:
            picked = self.classes.hotkey_pressed()
            if picked is not None:
                self.assign_class(picked)

    def _frame(self):
        """Per-frame work the strip runs whatever tab is on top: background
        jobs, the region, the keys, and the tool's own windows.
        """
        self._poll_jobs()
        self._poll_region()
        self._poll_folder_dialog()
        self.handle_keys()
        self.keybinds_open = draw_keybinds_popup(
            ROI_KEYS, self.keybinds_open, "ROI keybinds"
        )
        self.summary.draw()
