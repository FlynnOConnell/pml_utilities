"""The apps the host registers: the built-in set, and any a package adds.

A built-in is listed in ``ported_apps``. A package adds one by declaring an
entry point under ``mbo_utilities.apps`` that names an ``App`` subclass, or
a function returning a list of them; ``plugin_apps`` loads the group.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from mbo_utilities import log
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

logger = log.get("gui.app")

ENTRY_POINT_GROUP = "mbo_utilities.apps"

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
    "plugin_apps",
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


def plugin_apps() -> list[App]:
    """The apps installed packages register under ``mbo_utilities.apps``.

    An entry point that fails to load or build is logged and skipped: a
    broken plugin never takes the host down.
    """
    apps = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            made = ep.load()()
        except Exception:
            logger.exception(f"app entry point {ep.name!r} could not be loaded")
            continue
        apps.extend(made if isinstance(made, list) else [made])
    return apps
