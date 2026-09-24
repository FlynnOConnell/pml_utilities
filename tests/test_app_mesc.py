"""The app host on a .mesc file: the MESc tab opens units, ROIs follow each unit."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

ui = pytest.importorskip("fastplotlib.ui")
if not hasattr(ui, "ImguiWindow"):
    pytest.skip("needs a fastplotlib with ImguiWindow", allow_module_level=True)
pytest.importorskip("fastplotlib.widgets.nd_widget")

from mbo_utilities import imread  # noqa: E402
from mbo_utilities.gui.app import _app, build_host  # noqa: E402


@pytest.fixture
def mesc_path(tmp_path):
    path = tmp_path / "two_units.mesc"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        session = f.create_group("MSession_0")
        for i, step in enumerate((50.0, 25.0)):
            unit = session.create_group(f"MUnit_{i}")
            unit.attrs.update(
                {
                    "MethodType": 1,
                    "VecChannelsSize": 1,
                    "TStepInMs": step,
                    "MeasurementDatePosix": 1_700_000_000 + 100 * i,
                    "Comment": f"unit {i}",
                }
            )
            unit.create_dataset(
                "Channel_0",
                data=rng.integers(0, 4000, (8 * (i + 1), 16, 24)).astype(np.uint16),
            )
    return path


@pytest.fixture
def host(mesc_path):
    host = build_host(imread(mesc_path), size=(900, 600))
    host.figure.show()
    host.figure.canvas.force_draw()
    yield host
    host.close()
    host.viewer.close()


def unit(host, key: str) -> dict:
    tab = host.apps["mesc"].tab
    return next(u for u in tab._mesc.units if u["key"].endswith(key))


def test_the_mesc_tab_draws_first_for_a_mesc_file(host):
    _app._reported.discard("mesc")
    host.figure.canvas.force_draw()
    assert host.apps["mesc"].available(host)
    assert host.apps["mesc"].tab is not None
    assert "mesc" not in _app._reported


def test_a_row_opens_its_unit_into_the_host(host):
    host.figure.canvas.force_draw()
    before = host.data.unit_key
    target = "MUnit_1" if before.endswith("MUnit_0") else "MUnit_0"
    host.apps["mesc"].tab._switch(unit(host, target))
    host.figure.canvas.force_draw()
    assert host.data.unit_key.endswith(target)
    assert host.viewer.data[0].shape[0] == host.data.shape[0]


def test_each_unit_keeps_its_own_rois(host):
    host.figure.canvas.force_draw()
    rois = host.apps["manual_roi"]
    rois.open = True
    host.figure.canvas.force_draw()
    first = host.data.unit_key
    store = host.context.manual_roi.store

    other = "MUnit_1" if first.endswith("MUnit_0") else "MUnit_0"
    host.apps["mesc"].tab._switch(unit(host, other))
    host.figure.canvas.force_draw()
    assert host.context.manual_roi.store is not store

    host.apps["mesc"].tab._switch(unit(host, first.rsplit("/", 1)[-1]))
    host.figure.canvas.force_draw()
    assert host.context.manual_roi.store is store
    rois.open = False
    host.figure.canvas.force_draw()
