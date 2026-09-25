"""
GUI module with lazy imports to avoid loading heavy dependencies
(torch, cupy, suite2p, wgpu, imgui_bundle) until actually needed.

The CLI entry point (mbo command) imports this module, so we must keep
top-level imports minimal for fast startup of light operations like
--check-install.

Architecture
------------
The GUI is organized into these components:

- **App**: the viewer and the apps around it (app/), what ``mbo`` opens
- **Widgets**: the panels and pipeline settings the apps draw (widgets/)
- **Viewers**: panels that replace the standard ones for one kind of data
"""

__all__ = [
    "GridSearchViewer",
    # Entry points
    "DataVis",
    "get_default_ini_path",
    "run_gui",
    "set_qt_icon",
    "setup_imgui",
]


def __getattr__(name):
    """Lazy import heavy GUI modules only when accessed."""
    if name == "run_gui":
        from .run_gui import run_gui

        return run_gui
    if name == "DataVis":
        from .data_vis import DataVis

        return DataVis
    if name == "GridSearchViewer":
        from .widgets.grid_search import GridSearchViewer

        return GridSearchViewer
    if name == "setup_imgui":
        from ._setup import setup_imgui

        return setup_imgui
    if name == "set_qt_icon":
        from ._setup import set_qt_icon

        return set_qt_icon
    if name == "get_default_ini_path":
        from ._setup import get_default_ini_path

        return get_default_ini_path

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
