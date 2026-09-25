"""Apps ported from the preview window, drawn by the host instead of it."""

from __future__ import annotations

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.app.apps.console import ConsoleApp
from mbo_utilities.gui.app.apps.curate import CurateApp
from mbo_utilities.gui.app.apps.debug import debug_apps
from mbo_utilities.gui.app.apps.diagnostics import DiagnosticsApp
from mbo_utilities.gui.app.apps.help import HelpApp
from mbo_utilities.gui.app.apps.isoview import IsoviewProjectionsJob, IsoviewToolApp
from mbo_utilities.gui.app.apps.keybinds import KeybindsApp
from mbo_utilities.gui.app.apps.log import LogApp
from mbo_utilities.gui.app.apps.manual_roi import ManualRoiApp
from mbo_utilities.gui.app.apps.mesc import MescApp
from mbo_utilities.gui.app.apps.metadata import MetadataApp
from mbo_utilities.gui.app.apps.open import OpenApp
from mbo_utilities.gui.app.apps.options import OptionsApp
from mbo_utilities.gui.app.apps.align_views import AlignViewsApp
from mbo_utilities.gui.app.apps.projections import ProjectionsApp
from mbo_utilities.gui.app.apps.summary_images import SummaryImagesApp
from mbo_utilities.gui.app.apps.tile_grid import TileGridApp
from mbo_utilities.gui.app.apps.pollen import PollenApp
from mbo_utilities.gui.app.apps.remote import BiohpcApp, CloudApp
from mbo_utilities.gui.app.apps.run import RunApp
from mbo_utilities.gui.app.apps.save_as import SaveAsApp
from mbo_utilities.gui.app.apps.set_metadata import SetMetadataApp
from mbo_utilities.gui.app.apps.signal_quality import SignalQualityApp
from mbo_utilities.gui.app.apps.viewer import ViewerApp

__all__ = [
    "AlignViewsApp",
    "BiohpcApp",
    "CloudApp",
    "ConsoleApp",
    "CurateApp",
    "DiagnosticsApp",
    "HelpApp",
    "IsoviewProjectionsJob",
    "IsoviewToolApp",
    "KeybindsApp",
    "LogApp",
    "ManualRoiApp",
    "MescApp",
    "MetadataApp",
    "OpenApp",
    "OptionsApp",
    "PollenApp",
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
        CurateApp(),
        OptionsApp(),
        HelpApp(),
        KeybindsApp(),
        ProjectionsApp(),
        SummaryImagesApp(),
        TileGridApp(),
        AlignViewsApp(),
        IsoviewToolApp("crop"),
        IsoviewToolApp("segment"),
        IsoviewToolApp("deadpixel"),
        IsoviewProjectionsJob(),
        PollenApp(),
        BiohpcApp(),
        CloudApp(),
        MetadataApp(),
        SetMetadataApp(),
        SignalQualityApp(),
        RunApp(),
        ManualRoiApp(),
        MescApp(),
        ConsoleApp(),
        DiagnosticsApp(),
        LogApp(),
    ]
