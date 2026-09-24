"""Arrow keys a panel uses for itself, kept from the viewer's stepping."""

from __future__ import annotations

from imgui_bundle import imgui

# frame on which a widget used each arrow key for itself (see claim_arrow_keys);
# the viewer leaves that direction alone on that frame and the next, whatever
# the draw order
_ARROW_KEYS = ("left_arrow", "right_arrow", "up_arrow", "down_arrow")
_arrow_claims: dict[str, int] = dict.fromkeys(_ARROW_KEYS, -2)


def claim_arrow_keys(keys: tuple[str, ...] = _ARROW_KEYS):
    """Keep the viewer from also stepping T / Z on this frame's presses of ``keys``."""
    frame = imgui.get_frame_count()
    for key in keys:
        _arrow_claims[key] = frame


def arrow_claimed(key: str) -> bool:
    return imgui.get_frame_count() - _arrow_claims[key] <= 1
