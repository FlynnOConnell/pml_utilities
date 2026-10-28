"""The Page Axis panel: a plain TIFF's pages reopened on T, Z or C in the viewer."""

import numpy as np
import pytest
import tifffile
from mbo_utilities.gui.run_gui import _create_image_widget
from mbo_utilities.gui.widgets.page_axis import (
    PageAxisWidget,
    page_axis_reading,
    page_axis_tiff,
    reread_pages,
)
from mbo_utilities.lazy_array import base_array
from mbo_utilities.reader import imread, source_reader_kwargs

SHAPE = (6, 16, 18)


@pytest.fixture
def viewer(tmp_path):
    data = (np.random.default_rng(0).random(SHAPE) * 1000).astype(np.uint16)
    tifffile.imwrite(tmp_path / "plain.tif", data)
    iw = _create_image_widget(
        imread(tmp_path / "plain.tif"),
        widget=True,
        figure_kwargs_override={"size": (900, 600)},
    )
    panel = iw.figure.imgui_windows["right"]
    yield iw, panel
    iw.close()


def test_a_plain_tiff_gets_the_panel_and_opens_on_time(viewer):
    _, panel = viewer
    arr = page_axis_tiff(panel)
    assert arr is not None
    assert PageAxisWidget.is_supported(panel)
    assert page_axis_reading(arr) == "time"
    assert arr.shape == (6, 1, 1, 16, 18)


def test_rereading_puts_the_pages_on_the_named_axis(viewer):
    _, panel = viewer
    reread_pages(panel, "z-planes")
    arr = base_array(panel.image_widget.data[0])
    assert arr.shape == (1, 1, 6, 16, 18)
    assert page_axis_reading(arr) == "z-planes"
    assert source_reader_kwargs(arr) == {"dims": "ZYX"}
    reread_pages(panel, "channels")
    arr = base_array(panel.image_widget.data[0])
    assert arr.shape == (1, 6, 1, 16, 18)
    reread_pages(panel, "time")
    arr = base_array(panel.image_widget.data[0])
    assert arr.shape == (6, 1, 1, 16, 18)
    assert source_reader_kwargs(arr) == {"dims": "TYX"}
