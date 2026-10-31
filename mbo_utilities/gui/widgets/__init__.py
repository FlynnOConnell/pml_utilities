"""Widget registry: every ``Widget`` subclass in this package is found on first use.

A new widget is a module here with a ``Widget`` subclass that implements
``is_supported()`` and ``draw()``; it shows whenever ``is_supported()`` is True.
"""

import importlib
import pkgutil
import traceback
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.menu_bar import draw_menu_bar

_WIDGET_CLASSES: list[type[Widget]] = []


def _discover_widgets() -> None:
    """Imports every public module here once and collects its ``Widget`` subclasses by priority."""
    if _WIDGET_CLASSES:
        return

    package_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith("_"):
            continue

        try:
            module = importlib.import_module(f".{module_info.name}", package=__name__)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Widget)
                    and attr is not Widget
                ):
                    _WIDGET_CLASSES.append(attr)
        except Exception:
            pass

    _WIDGET_CLASSES.sort(key=lambda w: w.priority)


def _instantiate(parent: Any, placement: str) -> list[Widget]:
    """Instantiate every supported widget with the given placement."""
    _discover_widgets()

    supported = []
    for widget_cls in _WIDGET_CLASSES:
        if widget_cls.placement != placement:
            continue
        try:
            if widget_cls.is_supported(parent):
                supported.append(widget_cls(parent))
        except Exception:
            traceback.print_exc()

    supported.sort(key=lambda w: w.priority)
    return supported


def get_supported_widgets(parent: Any) -> list[Widget]:
    """Returns the supported panel widgets for ``parent``, instantiated, by priority."""
    return _instantiate(parent, "panel")


def get_tab_widgets(parent: Any) -> list[Widget]:
    """Get the supported widgets that render as their own viewer tab."""
    return _instantiate(parent, "tab")


def widget_is_visible(widget: Widget) -> bool:
    """Returns True when the Widgets menu has this widget switched on."""
    if widget.toggle_key is None:
        return True
    from mbo_utilities.gui.widgets.widget_toggles import widget_enabled

    return widget_enabled(widget.toggle_key)


def draw_all_widgets(parent: Any, widgets: list[Widget]) -> None:
    """Draw all supported widgets that the Widgets menu leaves switched on."""
    for widget in widgets:
        if not widget_is_visible(widget):
            continue
        try:
            widget.draw()
        except Exception as e:
            error_msg = f"Error in {widget.name}: {e}"
            parent.logger.exception(error_msg)
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), error_msg)


def cleanup_all_widgets(widgets: list[Widget]) -> None:
    """Calls ``cleanup()`` on each widget as the GUI closes; one that fails does not stop the rest."""
    for widget in widgets:
        try:
            widget.cleanup()
        except Exception:
            pass


__all__ = [
    "Widget",
    "cleanup_all_widgets",
    "draw_all_widgets",
    "draw_menu_bar",
    "get_supported_widgets",
    "get_tab_widgets",
    "widget_is_visible",
]
