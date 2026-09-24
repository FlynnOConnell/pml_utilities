"""Apps ported from the preview window, drawn by the host instead of it."""

from __future__ import annotations

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.app.apps.console import ConsoleApp
from mbo_utilities.gui.app.apps.debug import debug_apps
from mbo_utilities.gui.app.apps.diagnostics import DiagnosticsApp
from mbo_utilities.gui.app.apps.help import HelpApp
from mbo_utilities.gui.app.apps.keybinds import KeybindsApp
from mbo_utilities.gui.app.apps.log import LogApp
from mbo_utilities.gui.app.apps.metadata import MetadataApp
from mbo_utilities.gui.app.apps.open import OpenApp
from mbo_utilities.gui.app.apps.options import OptionsApp
from mbo_utilities.gui.app.apps.panels import (
    ProjectionsApp,
    SummaryImagesApp,
    TileGridApp,
)
from mbo_utilities.gui.app.apps.run import RunApp
from mbo_utilities.gui.app.apps.save_as import SaveAsApp
from mbo_utilities.gui.app.apps.set_metadata import SetMetadataApp
from mbo_utilities.gui.app.apps.signal_quality import SignalQualityApp
from mbo_utilities.gui.app.apps.viewer import ViewerApp

__all__ = [
    "ConsoleApp",
    "DiagnosticsApp",
    "HelpApp",
    "KeybindsApp",
    "LogApp",
    "MetadataApp",
    "OpenApp",
    "OptionsApp",
    "ProjectionsApp",
    "RunApp",
    "SaveAsApp",
    "SetMetadataApp",
    "SignalQualityApp",
    "SummaryImagesApp",
    "TileGridApp",
    "ViewerApp",
    "debug_apps",
    "ported_apps",
]


def ported_apps() -> list[App]:
    return [
        OpenApp("file"),
        OpenApp("folder"),
        SaveAsApp(),
        OptionsApp(),
        HelpApp(),
        KeybindsApp(),
        ProjectionsApp(),
        SummaryImagesApp(),
        TileGridApp(),
        MetadataApp(),
        SetMetadataApp(),
        SignalQualityApp(),
        RunApp(),
        ConsoleApp(),
        DiagnosticsApp(),
        LogApp(),
    ]
