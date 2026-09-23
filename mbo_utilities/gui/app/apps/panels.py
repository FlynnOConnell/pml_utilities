"""The preview window's image panels, as apps on an edge of their own."""

from __future__ import annotations

from mbo_utilities.gui.app._widget_app import WidgetApp
from mbo_utilities.gui.widgets.projections import ProjectionsViewer
from mbo_utilities.gui.widgets.summary_image import SummaryImageViewer
from mbo_utilities.gui.widgets.tile_grid import TileGridViewer


class ProjectionsApp(WidgetApp):
    """XY, XZ and YZ projections written beside an isoview stack."""

    id = "projections"
    title = "Projections"
    dock = "left"
    order = 40
    size = 330
    widget_class = ProjectionsViewer


class SummaryImagesApp(WidgetApp):
    """The 2D summary images an array carries in its metadata."""

    id = "summary_images"
    title = "Summary Images"
    dock = "left"
    order = 41
    size = 330
    widget_class = SummaryImageViewer


class TileGridApp(WidgetApp):
    """The tiles of a tiled acquisition, laid out as they were scanned."""

    id = "tile_grid"
    title = "Tile Grid"
    dock = "left"
    order = 42
    size = 330
    widget_class = TileGridViewer
