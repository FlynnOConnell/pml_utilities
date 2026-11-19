"""The volume crosshair: dragging a line or clicking a panel moves the point, and
every panel and 3D plane follows it, drawn offscreen.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

fpl = pytest.importorskip("fastplotlib")

from mbo_utilities.gui.volume_crosshair import VolumeCrosshair  # noqa: E402

SHAPE = (6, 20, 30)
VOXEL_SIZE = (4.0, 1.0, 0.5)


@pytest.fixture
def crosshair():
    volume = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    viewer = VolumeCrosshair(volume, voxel_size_zyx=VOXEL_SIZE, size=(400, 400))
    viewer.figure.canvas.draw()
    return viewer, volume


def test_starts_at_the_center(crosshair):
    viewer, _ = crosshair
    assert viewer.position == (3, 10, 15)


def test_position_sets_every_slice_line_and_plane(crosshair):
    viewer, volume = crosshair
    viewer.position = (2, 7, 11)

    np.testing.assert_array_equal(viewer._images["xy"].data.value, volume[2])
    np.testing.assert_array_equal(viewer._images["xz"].data.value, volume[:, 7, :])
    np.testing.assert_array_equal(viewer._images["yz"].data.value, volume[:, :, 11].T)
    assert viewer._selectors[("xy", "x")].selection == pytest.approx(11 * 0.5)
    assert viewer._selectors[("yz", "z")].selection == pytest.approx(2 * 4.0)
    assert viewer._planes["z"].plane[3] == pytest.approx(-8.0)
    assert viewer._outlines["x"].data.value[:, 0] == pytest.approx(5.5)


def test_position_clips_to_the_volume(crosshair):
    viewer, _ = crosshair
    viewer.position = (-3, 100, 29)
    assert viewer.position == (0, 19, 29)


def test_dragging_a_line_moves_the_point(crosshair):
    viewer, _ = crosshair
    viewer._selectors[("xz", "z")].selection = 1 * 4.0
    assert viewer.position == (1, 10, 15)
    assert viewer._selectors[("yz", "z")].selection == pytest.approx(4.0)


def test_clicking_a_panel_moves_the_point(crosshair):
    viewer, _ = crosshair
    viewer._on_click("yz", SimpleNamespace(pick_info={"index": (5, 12)}))
    assert viewer.position == (5, 12, 15)
