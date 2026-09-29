"""Registry and menu for the "Widgets" menu-bar entry.

Every toggleable piece of UI is one :class:`WidgetEntry`, drawn as one
checkbox; the draw code for each widget asks :func:`widget_enabled` before
rendering. Below the checkboxes the menu opens the floating tool windows:
the style editor, the imgui debugger, BioHPC and the cloud runner.

State is keyed by entry (``"preview"``) and persisted in preferences, so
toggles survive a restart. Anything absent from the stored mapping falls
back to the registry default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from imgui_bundle import imgui

from mbo_utilities.gui.widgets.imgui_debug import draw_imgui_debug_menu_item
from mbo_utilities.gui.widgets.style_editor import draw_style_menu_item
from mbo_utilities.preferences import get_widget_toggles, set_widget_toggles

__all__ = [
    "WidgetEntry",
    "WIDGET_REGISTRY",
    "draw_widgets_menu",
    "get_entry",
    "reset_widget_toggles",
    "set_widget_enabled",
    "widget_enabled",
]


@dataclass(frozen=True)
class WidgetEntry:
    """A widget in the Widgets menu: one checkbox."""

    key: str
    label: str
    default: bool = True
    tooltip: str = ""
    # called as on_toggle(parent, enabled) right after the user flips the
    # checkbox; lets a widget build/tear down live state (the ROI overlay,
    # say) instead of only gating its draw.
    on_toggle: Callable[[Any, bool], None] | None = field(default=None, compare=False)


def _toggle_manual_roi(parent: Any, enabled: bool) -> None:
    """Create or drop the ROI overlay and its edge windows when toggled."""
    sync = getattr(parent, "sync_manual_roi", None)
    if sync is not None:
        sync(enabled)


WIDGET_REGISTRY: tuple[WidgetEntry, ...] = (
    WidgetEntry(
        key="preview",
        label="Image",
        tooltip="The Image tab and the control panels stacked inside it.",
    ),
    WidgetEntry(
        key="mesc",
        label="MESc",
        tooltip="Every recording in the open .mesc file: shape, rate, the lines "
        "or patches it scanned, the picture they were drawn on, whether "
        "RTMC was on and the Z-stack around it; click a row to display it.",
    ),
    WidgetEntry(
        key="signal_quality",
        label="Signal Quality",
        tooltip="Per-plane z-stats and signal-quality plots.",
        default=False,
    ),
    WidgetEntry(
        key="run",
        label="Process",
        tooltip="Registration / segmentation pipelines.",
    ),
    WidgetEntry(
        key="manual_roi",
        label="Manual ROI Labeling",
        tooltip="Freehand ROI drawing and labelling: the ROIs tab holds the "
        "controls over the ROI table, the trace plot is a panel over the "
        "image and the Traces tab lists every trace. Running ROIs is the "
        "Process tab's ROIs pipeline.",
        default=False,
        on_toggle=_toggle_manual_roi,
    ),
)

_BY_KEY = {e.key: e for e in WIDGET_REGISTRY}

# lazily loaded from preferences, then kept in sync on every write
_state: dict[str, bool] | None = None


def get_entry(key: str) -> WidgetEntry | None:
    """Return the registry entry for ``key``, or None."""
    return _BY_KEY.get(key)


def _load() -> dict[str, bool]:
    global _state
    if _state is None:
        try:
            _state = get_widget_toggles()
        except Exception:
            _state = {}
    return _state


def _persist() -> None:
    try:
        set_widget_toggles(_load())
    except Exception:
        pass


def _default(key: str) -> bool:
    entry = _BY_KEY.get(key)
    return True if entry is None else entry.default


def widget_enabled(key: str) -> bool:
    """Whether widget ``key`` should be drawn."""
    return bool(_load().get(key, _default(key)))


def set_widget_enabled(key: str, value: bool, persist: bool = True) -> None:
    """Turn widget ``key`` on or off."""
    _load()[key] = bool(value)
    if persist:
        _persist()


def _fire(parent: Any, on_toggle, key: str, value: bool) -> None:
    """Run a widget toggle callback, logging rather than raising."""
    if parent is None or on_toggle is None:
        return
    try:
        on_toggle(parent, value)
    except Exception:
        logger = getattr(parent, "logger", None)
        if logger is not None:
            logger.exception(f"widget toggle failed for {key}")


def _apply_all(parent: Any, chosen, persist: bool = True) -> None:
    """Set every widget to ``chosen(entry)`` and fire callbacks."""
    state = _load()
    for entry in WIDGET_REGISTRY:
        was_on = widget_enabled(entry.key)
        value = chosen(entry)
        state[entry.key] = value
        if was_on != value:
            _fire(parent, entry.on_toggle, entry.key, value)
    if persist:
        _persist()


def reset_widget_toggles(parent: Any = None, persist: bool = True) -> None:
    """Restore every widget to its registry default."""
    _apply_all(parent, lambda entry: entry.default, persist)


def _set_all(parent: Any, value: bool) -> None:
    _apply_all(parent, lambda entry: value)


def draw_widgets_menu(parent: Any) -> None:
    """Draw the "Widgets" menu. Call inside an active menu bar."""
    if not imgui.begin_menu("Widgets", True):
        return

    for entry in WIDGET_REGISTRY:
        enabled = widget_enabled(entry.key)
        clicked, new_value = imgui.menu_item(entry.label, "", enabled, True)
        if entry.tooltip and imgui.is_item_hovered():
            imgui.set_tooltip(entry.tooltip)
        if clicked and new_value != enabled:
            set_widget_enabled(entry.key, new_value)
            _fire(parent, entry.on_toggle, entry.key, new_value)

    imgui.separator()
    if imgui.menu_item("Enable All", "", False, True)[0]:
        _set_all(parent, True)
    if imgui.menu_item("Disable All", "", False, True)[0]:
        _set_all(parent, False)
    if imgui.menu_item("Reset to Defaults", "", False, True)[0]:
        reset_widget_toggles(parent)

    imgui.separator()
    # windows, not tabs: these open as floating windows over the viewer
    draw_style_menu_item()
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Sizes, spacing and colours of the running imgui style. "
            "Saved under ~/.mbo/imgui and applied at the next launch."
        )
    draw_imgui_debug_menu_item()
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "The variable inspector over this window's state, and Dear "
            "ImGui's own metrics, debug log, ID stack tool and demo."
        )
    if imgui.menu_item("BioHPC...", "", False, True)[0]:
        parent._show_biohpc = True
    if imgui.menu_item("Cloud (GPU)...", "", False, True)[0]:
        parent._show_cloud = True
    imgui.end_menu()
