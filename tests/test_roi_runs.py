"""gui.roi_runs: derived-set state, pure-unit.

The overlay, pick-map, scan and sidecar helpers need only numpy and a tmp
dir. Nothing here imports masknmf or opens a figure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from mbo_utilities.gui import roi_runs as rr
from mbo_utilities.results import ResultUnit


def _row(ypix, xpix, lam):
    ypix, xpix = np.asarray(ypix, np.int32), np.asarray(xpix, np.int32)
    lam = np.asarray(lam, np.float32)
    return {
        "ypix": ypix,
        "xpix": xpix,
        "lam": lam,
        "med": (float(ypix.mean()), float(xpix.mean())),
        "npix": int(ypix.size),
    }


def _result(rows, shape=(6, 6), **attrs):
    """A pixel ``ResultUnit`` from suite2p-shaped stat rows."""
    ly, lx = shape
    return ResultUnit(
        name="zplane01",
        kind="plane",
        index=1,
        roi_names=[f"roi{k}" for k in range(len(rows))],
        member_kind="pixel",
        members=[r["ypix"].astype(np.int64) * lx + r["xpix"] for r in rows],
        weights=[r["lam"] for r in rows],
        image_shape=(ly, lx),
        attrs=dict(attrs),
    )


def _set(unit, name, color, **kw):
    return rr.DerivedSet(unit, Path("run") / name, name, color, **kw)


class _HydrateParent:
    """What ``_try_hydrate_s2p_from_binary`` needs of the preview widget."""

    logger = logging.getLogger("test.hydrate")

    def __init__(self, s2p, s2p_db):
        self.s2p, self.s2p_db, self.s2p_extras = s2p, s2p_db, None


def _fake_run_dir(root, name, n_rois=0):
    d = root / name
    d.mkdir(parents=True)
    np.save(d / "stat.npy", np.empty(n_rois, object))
    np.save(d / "ops.npy", {"Ly": 4, "Lx": 4}, allow_pickle=True)
    return d


def test_finished_dirs_prefix_matches_suffixed_plane_dirs(tmp_path):
    a = _fake_run_dir(tmp_path, "zplane01_tp00001-00400")
    b = _fake_run_dir(tmp_path, "zplane02")
    partial = tmp_path / "zplane03"
    partial.mkdir()
    np.save(partial / "ops.npy", {}, allow_pickle=True)  # no stat: still writing
    assert rr.finished_dirs(tmp_path, [1]) == [a]
    assert rr.finished_dirs(tmp_path, [3]) == []
    assert rr.finished_dirs(tmp_path) == [a, b]
    assert rr.finished_dirs(a) == [a]


def test_pick_map_strongest_lam_wins():
    res = _result(
        [
            _row([0, 1], [0, 1], [5.0, 5.0]),
            _row([1, 2], [1, 2], [2.0, 2.0]),
        ]
    )
    s = _set(res, "a", rr.set_color(0))
    pick = rr.build_pick_map(s.footprints, s.shape)
    assert pick[0, 0] == 0 and pick[2, 2] == 1
    assert pick[1, 1] == 0  # contested pixel goes to the stronger component
    assert pick[3, 3] == -1
    np.testing.assert_array_equal(s.pick_map, pick)
    assert len(s) == 2 and s.z == 0 and s.engine == "suite2p" and not s.has_prob


def test_derived_rgba_lam_weights():
    s = _set(_result([_row([1, 2], [1, 2], [1.0, 2.0])]), "a", (1.0, 0.0, 0.0))
    img = rr.derived_rgba((6, 6), [s], 0.5)
    assert img.shape == (6, 6, 4) and img.dtype == np.uint8
    c = np.rint(np.asarray(rr.component_color(s, 0)) * 255).astype(int).tolist()
    assert img[1, 1].tolist() == [*c, 64]  # lam/peak = 0.5 -> 0.25 alpha
    assert img[2, 2].tolist() == [*c, 128]
    assert img[0, 0, 3] == 0


def test_derived_rgba_overlap_winner_takes_color():
    a = _set(_result([_row([1, 2], [1, 2], [0.5, 1.0])]), "a", (1.0, 0.0, 0.0))
    b = _set(_result([_row([1], [1], [1.0])]), "b", (0.0, 0.0, 1.0))
    img = rr.derived_rgba((6, 6), [a, b], 0.6)
    ca = np.rint(np.asarray(rr.component_color(a, 0)) * 255).astype(int).tolist()
    cb = np.rint(np.asarray(rr.component_color(b, 0)) * 255).astype(int).tolist()
    assert img[2, 2].tolist() == [*ca, 153]  # a's peak pixel, 0.6 alpha
    assert img[1, 1].tolist() == [*cb, 153]  # b's 0.6 beats a's 0.3


def test_derived_rgba_selection_fill_and_rim():
    yy, xx = np.mgrid[1:4, 1:4]
    s = _set(_result([_row(yy.ravel(), xx.ravel(), np.ones(9))]), "a", (1.0, 0.0, 0.0))
    img = rr.derived_rgba((6, 6), [s], 0.3, selected=(s, 0))
    fill = round(rr.SELECTED_ALPHA * 255)
    c = np.rint(np.asarray(rr.component_color(s, 0)) * 255).astype(int).tolist()
    assert img[2, 2].tolist() == [*c, fill]  # interior keeps its own color
    assert img[1, 1].tolist() == [255, 255, 255, fill]  # border is the white rim
    assert img[1, 2].tolist() == [255, 255, 255, fill]


def test_derived_rgba_skips_discarded_and_invisible():
    s = _set(
        _result(
            [
                _row([1], [1], [1.0]),
                _row([3], [3], [1.0]),
            ]
        ),
        "a",
        (1.0, 0.0, 0.0),
        discarded={0},
    )
    img = rr.derived_rgba((6, 6), [s], 0.5)
    assert img[1, 1, 3] == 0 and img[3, 3, 3] == 128
    s.visible = False
    assert not rr.derived_rgba((6, 6), [s], 0.5).any()
    # a discarded selection is not highlighted either
    assert not rr.derived_rgba((6, 6), [s], 0.5, selected=(s, 0)).any()


def _pieces(pos):
    """The line buffer's paths, split on the NaN rows between them."""
    out, run = [], []
    for point in pos:
        if np.isnan(point).any():
            if run:
                out.append(np.array(run))
                run = []
            continue
        run.append(point[:2])
    if run:
        out.append(np.array(run))
    return out


def test_footprint_edges_traces_the_pixel_border():
    yy, xx = np.mgrid[2:5, 3:7]  # rows 2..4, cols 3..6
    edges = rr.footprint_edges(yy.ravel(), xx.ravel())
    finite = edges[np.isfinite(edges[:, 0])]
    assert len(finite) % 2 == 0
    # 3x4 block: 14 unit edges around it, on the pixel border itself
    assert len(finite) // 2 == 14
    assert finite[:, 0].min() == 3.0 and finite[:, 0].max() == 7.0
    assert finite[:, 1].min() == 2.0 and finite[:, 1].max() == 5.0


def test_footprint_edges_of_one_pixel_is_its_square():
    edges = rr.footprint_edges([5], [7])
    finite = edges[np.isfinite(edges[:, 0])]
    assert len(finite) // 2 == 4
    assert finite.min(axis=0).tolist() == [7.0, 5.0]
    assert finite.max(axis=0).tolist() == [8.0, 6.0]


def test_footprint_edges_of_nothing():
    assert rr.footprint_edges([], []).shape == (0, 2)


def test_the_ring_is_the_equal_area_circle_on_pixel_centres():
    yy, xx = np.mgrid[10:20, 10:20]  # 100 px, centres 10.5..19.5
    y, x = yy.ravel(), xx.ravel()
    assert rr.footprint_center(y, x) == (15.0, 15.0)
    assert rr.footprint_radius(y, x, 1.0) == pytest.approx(np.sqrt(100 / np.pi))
    # a floor, so a mask a few pixels across is still worth looking at
    assert rr.footprint_radius([1], [1], 1.0) == rr.MIN_RING_RADIUS


def test_outline_data_gives_one_closed_path_per_component():
    comps = [
        (np.array([1, 2]), np.array([1, 2]), np.ones(2), (1.0, 0.0, 0.0), 1.0),
        (np.array([8]), np.array([8]), np.ones(1), (0.0, 0.0, 1.0), 0.4),
    ]
    pos, colors = rr.outline_data(comps, "circle")
    assert pos.shape[1] == 3 and colors.shape == (len(pos), 4)
    rings = _pieces(pos)
    assert len(rings) == 2
    for path in rings:
        assert np.allclose(path[0], path[-1])  # closed
    # fill rides along as the stroke alpha, one color per path
    finite = np.isfinite(pos[:, 0])
    assert np.allclose(np.unique(np.round(colors[finite][:, 3], 3)), [0.4, 1.0])


def test_outline_data_rings_the_halo_footprints_in_white():
    comps = [(np.array([4]), np.array([4]), np.ones(1), (1.0, 0.0, 0.0), 1.0)]
    pos, colors = rr.outline_data(
        comps, "circle", halo=[(np.array([4]), np.array([4]))]
    )
    rings = _pieces(pos)
    assert len(rings) == 2
    spans = [np.ptp(r[:, 0]) for r in rings]
    assert spans[1] > spans[0]  # the halo sits outside the component's ring
    assert np.allclose(colors[-1], (1.0, 1.0, 1.0, 1.0))


def test_outline_data_of_nothing_is_empty():
    pos, colors = rr.outline_data([], "circle")
    assert pos.shape == (0, 3) and colors.shape == (0, 4)


def test_derived_outline_follows_the_same_rows_as_derived_rgba():
    s = _set(
        _result(
            [
                _row([1], [1], [1.0]),
                _row([3], [3], [1.0]),
            ]
        ),
        "a",
        (1.0, 0.0, 0.0),
        discarded={0},
    )
    pos, colors = rr.derived_outline([s], "circle")
    assert len(_pieces(pos)) == 1  # the discarded row is skipped
    # rejected rows draw dimmed, and only they do
    assert colors[0, 3] == 1.0
    s.accepted[1] = False
    _pos, colors = rr.derived_outline([s], "circle")
    assert colors[0, 3] < 1.0
    s.visible = False
    assert not len(rr.derived_outline([s], "circle")[0])


def test_derived_outline_haloes_the_selection():
    s = _set(_result([_row([3], [3], [1.0])]), "a", (1.0, 0.0, 0.0))
    pos, colors = rr.derived_outline([s], "circle", selected=(s, 0))
    assert len(_pieces(pos)) == 2
    assert np.allclose(colors[-1], (1.0, 1.0, 1.0, 1.0))


def test_component_color_is_class_color_when_labeled():
    from mbo_utilities.annotation import class_color

    s = _set(_result([_row([1], [1], [1.0])]), "a", (1.0, 0.0, 0.0))
    hue = rr.component_color(s, 0)
    s.classes[0] = 1
    assert rr.component_color(s, 0) == class_color(1) != hue


def test_unit_trace_carries_the_read_coordinates(tmp_path):
    unit = _result(
        [_row([0], [0], [1.0])] * 2, shape=(8, 9), engine="suite2p", z=1, c=2
    )
    unit.fs = 5.0
    unit.traces = {
        "raw": np.arange(8, dtype=np.float32).reshape(2, 4),
        "neuropil": np.zeros((2, 4), np.float32),
        "spikes": np.ones((2, 4), np.float32),
    }
    unit.attrs["frames"] = (10, 14, 1)
    path = tmp_path / "movie.a.2026-10-28-10-00-00.manual.zarr"
    trace = rr.unit_trace(
        unit, 0, uid=5, member=None, source="a", path=path, z=1, c=2, engine="suite2p"
    )
    assert trace.key == ("roi", 5, 1, 2, "suite2p")
    assert trace.source == "a" and trace.path == path
    assert trace.frames == (10, 14, 1) and trace.fs == 5.0
    np.testing.assert_array_equal(trace.F, [0, 1, 2, 3])
    np.testing.assert_array_equal(trace.Fneu, 0)
    assert trace.norm is None and trace.kinds["spikes"].tolist() == [1] * 4
    assert (
        rr.unit_trace(
            unit, 2, uid=0, member=None, source="a", path=path, z=0, c=0, engine="mean"
        )
        is None
    )
    s = _set(unit, "a", rr.set_color(0))
    assert (s.read_z, s.read_c, s.frames, s.engine) == (1, 2, (10, 14, 1), "suite2p")
    member = s.trace(1, z=0, c=0)
    assert member.key == ("member", "a", 1) and member.z == 1 and member.c == 2
    plain = _set(_result([_row([0], [0], [1.0])]), "b", rr.set_color(1), z=3)
    assert (plain.read_z, plain.read_c, plain.frames) == (None, None, None)
    assert plain.trace(0, z=3, c=0) is None


class TestSuite2pHydrate:
    """Opening a run dir restores its parameters, never its run gates.

    A registration pass writes roidetect=0 into the plane dir's ops.npy
    (that is how it asks suite2p for registration only). Hydrating that key
    set the Process tab's Detection toggle to Skip, and every later suite2p
    run then reported "Suite2p disabled by user toggles; regenerating
    figures only" and found no ROIs.
    """

    @staticmethod
    def _parent():
        try:
            from mbo_utilities.gui.widgets.pipelines.settings import (
                Suite2pDB,
                Suite2pSettings,
            )
        except Exception as error:  # suite2p / cellpose not importable here
            pytest.skip(f"suite2p settings unavailable: {error}")
        return _HydrateParent(Suite2pSettings(), Suite2pDB())

    def test_a_registered_dir_does_not_switch_detection_off(self, tmp_path):
        from mbo_utilities.gui._dialogs import _try_hydrate_s2p_from_binary

        plane = tmp_path / "zplane01"
        plane.mkdir()
        np.save(
            plane / "ops.npy",
            {
                "roidetect": False,  # registration-only pass wrote this
                "do_registration": 1,
                "tau": 0.7,  # a real parameter, which must survive
                "Ly": 4,
                "Lx": 4,
            },
        )

        parent = self._parent()
        before = parent.s2p.do_detection
        assert _try_hydrate_s2p_from_binary(parent, plane)
        assert parent.s2p.do_detection == before, "the Skip/Run gate must not move"
        assert parent.s2p.tau == pytest.approx(0.7), "parameters still hydrate"
