"""Key chords written as text, the way menus and the keybinds sheet show them."""

from __future__ import annotations

from imgui_bundle import imgui

MODIFIERS = {
    "Shift": imgui.Key.mod_shift,
    "Ctrl": imgui.Key.mod_ctrl,
    "Alt": imgui.Key.mod_alt,
}
# names that are not the key's own letter or digit
NAMED = {
    "Space": imgui.Key.space,
    "Enter": imgui.Key.enter,
    "Left": imgui.Key.left_arrow,
    "Right": imgui.Key.right_arrow,
    "Up": imgui.Key.up_arrow,
    "Down": imgui.Key.down_arrow,
}


def chord(text: str) -> int:
    """``"Shift+O"`` as the imgui key chord it names."""
    *modifiers, key = text.split("+")
    value = int(NAMED[key] if key in NAMED else getattr(imgui.Key, key.lower()))
    for modifier in modifiers:
        value |= int(MODIFIERS[modifier])
    return value


def pressed(text: str, repeat: bool = False) -> bool:
    """Whether the chord ``text`` went down this frame; never while typing in a field."""
    if imgui.get_io().want_text_input:
        return False
    flags = imgui.InputFlags_.route_global
    if repeat:
        flags |= imgui.InputFlags_.repeat
    return imgui.shortcut(chord(text), flags)
