"""Apps ported from the preview window, drawn by the host instead of it."""

from __future__ import annotations

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.app.apps.debug import debug_apps
from mbo_utilities.gui.app.apps.diagnostics import DiagnosticsApp
from mbo_utilities.gui.app.apps.log import LogApp
from mbo_utilities.gui.app.apps.metadata import MetadataApp
from mbo_utilities.gui.app.apps.open import OpenApp
from mbo_utilities.gui.app.apps.panels import (
    ProjectionsApp,
    SummaryImagesApp,
    TileGridApp,
)

__all__ = [
    "DiagnosticsApp",
    "LogApp",
    "MetadataApp",
    "OpenApp",
    "ProjectionsApp",
    "SummaryImagesApp",
    "TileGridApp",
    "debug_apps",
    "ported_apps",
]


def ported_apps() -> list[App]:
    return [
        OpenApp(),
        ProjectionsApp(),
        SummaryImagesApp(),
        TileGridApp(),
        MetadataApp(),
        DiagnosticsApp(),
        LogApp(),
    ]
