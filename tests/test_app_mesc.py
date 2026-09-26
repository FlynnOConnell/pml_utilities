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
from mbo_utilities.lazy_array import base_array  # noqa: E402


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
    tab = host.apps["mesc"]
    return next(u for u in tab._mesc.units if u["key"].endswith(key))


def test_the_mesc_tab_draws_first_for_a_mesc_file(host):
    _app._reported.discard("mesc")
    host.figure.canvas.force_draw()
    assert host.apps["mesc"].available(host)
    assert host.apps["mesc"]._hooked
    assert "mesc" not in _app._reported
    right = [app.id for app in host.ordered() if app.dock == "right"]
    assert right.index("mesc") < right.index("viewer")


def test_a_row_opens_its_unit_into_the_host(host):
    host.figure.canvas.force_draw()
    before = host.data.unit_key
    target = "MUnit_1" if before.endswith("MUnit_0") else "MUnit_0"
    host.apps["mesc"]._switch(unit(host, target))
    host.figure.canvas.force_draw()
    assert host.data.unit_key.endswith(target)
    assert host.viewer.data[0].shape[0] == host.data.shape[0]


def test_each_unit_keeps_its_own_rois(host):
    host.figure.canvas.force_draw()
    rois = host.apps["manual_roi"]
    rois.open = True
    host.figure.canvas.force_draw()
    first = host.data.unit_key
    widget = host.context.manual_roi
    store = widget.store
    tag = first.replace("/", "_")
    assert widget.tag == tag
    assert widget._save_target().name == f"manual_labels_{tag}.zarr"
    assert widget.run_prefix == f"rois_{tag}_"

    other = "MUnit_1" if first.endswith("MUnit_0") else "MUnit_0"
    host.apps["mesc"]._switch(unit(host, other))
    host.figure.canvas.force_draw()
    assert host.context.manual_roi.store is not store

    host.apps["mesc"]._switch(unit(host, first.rsplit("/", 1)[-1]))
    host.figure.canvas.force_draw()
    assert host.context.manual_roi.store is store
    rois.open = False
    host.figure.canvas.force_draw()


def test_curate_offers_itself_for_a_mesc_file(host):
    from mbo_utilities.gui._availability import HAS_VNOISER

    curate = host.apps["curate"]
    assert curate.available(host) is HAS_VNOISER
    assert curate.target == host.data.source_path
    assert curate.open is False


def test_the_reference_buttons_unit_opens_at_its_slice(stack_mesc_path):
    """The reference popup's display button hands the tab a unit and a slice,
    applied the frame after, so a Z-stack opens on the tissue the lines were
    scanned in; a picture carries no slice and opens where it opens.
    """
    from mbo_utilities.arrays.mesc import MescArray, list_mesc_units
    from mbo_utilities.gui.mesc_reference import roi_slider

    stack, scan = list_mesc_units(stack_mesc_path)
    host = build_host(MescArray(stack_mesc_path, unit=1), size=(900, 600))
    try:
        host.figure.show()
        host.figure.canvas.force_draw()
        tab = host.apps["mesc"]
        tab._show_reference_unit(stack["key"], 3)
        assert tab._pending is not None
        tab._strip_hook()
        assert tab._pending is None
        assert host.data.unit_key == stack["key"]
        assert int(host.viewer.indices[roi_slider(host.viewer.dim_names)]) == 3

        tab._switch(scan)
        tab._show_reference_unit(stack["key"], None)
        tab._strip_hook()
        assert int(host.viewer.indices[roi_slider(host.viewer.dim_names)]) == 0
    finally:
        host.close()
        host.viewer.close()


def test_suite2p_data_options_reach_the_phase_correction(host):
    context = host.context
    source = base_array(host.data)
    assert context.has_raster_scan_support
    assert not context.is_mbo_scan
    context.border = 5
    context.max_offset = 7
    assert (source.border, source.max_offset) == (5, 7)
    assert (context.border, context.max_offset) == (5, 7)
    assert len(context.current_offset) == 1
    assert context._register_z is False
