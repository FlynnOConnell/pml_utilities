"""gui.roi_runs: run orchestration and derived-set state, pure-unit.

RoiRunManager is exercised against a stub process manager (real LocalJob
handles, fake spawn/get_running/kill); the overlay, pick-map, scan,
worker-args and sidecar helpers need only numpy and a tmp dir. Nothing
here imports masknmf or opens a figure.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mbo_utilities.gui import roi_runs as rr
from mbo_utilities.gui.widgets.process_manager import LocalJob
from mbo_utilities.results import ResultUnit


class _StubPM:
    """start_job/spawn/get_running/kill lookalike; runs nothing."""

    def __init__(self):
        self.jobs: list[LocalJob] = []
        self.spawns: list[dict] = []
        self.killed: list[int] = []
        self.running: list[SimpleNamespace] = []
        self.spawn_fails = False
        self._next_pid = 1000

    def start_job(self, task_type, description):
        job = LocalJob(
            job_id=len(self.jobs) + 1,
            task_type=task_type,
            description=description,
            start_time=time.time(),
        )
        self.jobs.append(job)
        return job

    def spawn(self, task_type, args, description, output_path=None):
        self.spawns.append(
            {
                "task_type": task_type,
                "args": args,
                "description": description,
                "output_path": output_path,
            }
        )
        if self.spawn_fails:
            return None
        self._next_pid += 1
        self.running.append(
            SimpleNamespace(pid=self._next_pid, status="running", status_message="")
        )
        return self._next_pid

    def get_running(self):
        return list(self.running)

    def kill(self, pid):
        self.killed.append(pid)
        self.running = [p for p in self.running if p.pid != pid]
        return True


@pytest.fixture
def pm():
    return _StubPM()


def _drain(mgr, pm, n=1, timeout=5.0):
    end = time.time() + timeout
    out = []
    while time.time() < end and len(out) < n:
        out += mgr.poll(pm)
        time.sleep(0.005)
    return out


def _until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


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


def test_submit_success_and_single_poll(pm):
    mgr = rr.RoiRunManager(pm)
    run = rr.RoiRun(kind="extract", tag="a", description="extract rois_a")
    seen = []
    mgr.submit(run, lambda job: seen.append(job) or ["out"])
    assert mgr.busy and run in mgr.active
    done = _drain(mgr, pm)
    assert done == [(run, ["out"])]
    assert run.finished and run.error is None
    assert seen == [run.job] and run.job.status == "completed"
    assert pm.jobs[0].task_type == "extract"
    assert not mgr.busy and mgr.poll(pm) == []


def test_submit_failure_lands_on_run_and_job(pm):
    mgr = rr.RoiRunManager(pm)
    run = rr.RoiRun(kind="demix", tag="b", description="demix rois_b")

    def boom(job):
        raise ValueError("boom")

    mgr.submit(run, boom)
    ((got, payload),) = _drain(mgr, pm)
    assert got is run and payload is None
    # the message names the frame that raised, so a failure deep in a
    # pipeline is diagnosable without opening the log
    assert run.error.startswith("ValueError: boom (test_roi_runs.py:")
    assert run.job.status == "error" and run.job.error_details == run.error


def test_heavy_runs_serialize_on_the_gpu_lock(pm):
    mgr = rr.RoiRunManager(pm)
    gate = threading.Event()
    order = []

    def slow(job):
        order.append("a")
        assert gate.wait(5)
        order.append("a-done")

    ra = mgr.submit(
        rr.RoiRun(kind="demix", tag="a", description="demix a"), slow, heavy=True
    )
    assert _until(lambda: order == ["a"])
    rb = mgr.submit(
        rr.RoiRun(kind="discover", tag="b", description="find b"),
        lambda job: order.append("b"),
        heavy=True,
    )
    assert _until(lambda: rb.job.status_message == "waiting for gpu")
    time.sleep(0.05)
    assert order == ["a"]  # b is queued behind a, not running
    gate.set()
    assert len(_drain(mgr, pm, n=2)) == 2
    assert order == ["a", "a-done", "b"]
    assert ra.finished and rb.finished and not mgr.busy


def test_spawn_records_pid_and_out_root(pm, tmp_path):
    mgr = rr.RoiRunManager(pm)
    run = rr.RoiRun(kind="suite2p", tag="plane01", description="suite2p plane01")
    mgr.spawn(run, "suite2p", {"output_dir": str(tmp_path), "planes": [1]})
    assert run.pid == pm.running[0].pid and run.out_root == tmp_path
    assert pm.spawns[0]["task_type"] == "suite2p"
    assert pm.spawns[0]["output_path"] == str(tmp_path)
    assert mgr.poll(pm) == [] and mgr.busy
    pm.running[0].status = "completed"
    assert mgr.poll(pm) == [(run, None)]
    assert run.finished and run.error is None
    assert mgr.poll(pm) == []


def test_spawn_error_and_disappearance(pm, tmp_path):
    mgr = rr.RoiRunManager(pm)
    bad = rr.RoiRun(kind="masknmf", tag="p2", description="masknmf plane02")
    mgr.spawn(bad, "masknmf", {"output_dir": str(tmp_path)})
    pm.running[0].status = "error"
    pm.running[0].status_message = "cuda oom"
    assert mgr.poll(pm) == [(bad, None)] and bad.error == "cuda oom"

    gone = rr.RoiRun(kind="suite2p", tag="p3", description="suite2p plane03")
    mgr.spawn(gone, "suite2p", {"output_dir": str(tmp_path)})
    pm.running.clear()
    assert mgr.poll(pm) == [(gone, None)] and gone.error == "process stopped"


def test_spawn_start_failure_surfaces_through_poll(pm, tmp_path):
    mgr = rr.RoiRunManager(pm)
    pm.spawn_fails = True
    run = rr.RoiRun(kind="suite2p", tag="p1", description="suite2p plane01")
    mgr.spawn(run, "suite2p", {"output_dir": str(tmp_path)})
    assert run.pid is None
    assert mgr.poll(pm) == [(run, None)]
    assert run.finished and run.error == "failed to start worker"


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


def test_spawn_disappearance_with_outputs_is_success(pm, tmp_path):
    # the console prunes completed entries after a few minutes; outputs on
    # disk mean the run finished, not that it crashed
    _fake_run_dir(tmp_path, "zplane01_tp00001-00400")
    mgr = rr.RoiRunManager(pm)
    run = rr.RoiRun(kind="suite2p", tag="p1", description="suite2p plane01", planes=[1])
    mgr.spawn(run, "suite2p", {"output_dir": str(tmp_path)})
    pm.running.clear()
    assert mgr.poll(pm) == [(run, None)]
    assert run.finished and run.error is None


def test_stop_kills_spawned_only(pm, tmp_path):
    mgr = rr.RoiRunManager(pm)
    spawned = rr.RoiRun(kind="suite2p", tag="p1", description="suite2p plane01")
    mgr.spawn(spawned, "suite2p", {"output_dir": str(tmp_path)})
    mgr.stop(spawned, pm)
    assert pm.killed == [spawned.pid]
    assert mgr.poll(pm) == [(spawned, None)] and spawned.error == "process stopped"

    gate = threading.Event()
    local = rr.RoiRun(kind="extract", tag="a", description="extract rois_a")
    mgr.submit(local, lambda job: gate.wait(5))
    mgr.stop(local, pm)  # in-process: no cancel
    assert pm.killed == [spawned.pid]
    gate.set()
    _drain(mgr, pm)


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


def _run_dir(path, kind=None, n_stat=1, n_rois=None, pipeline=None):
    path.mkdir(parents=True, exist_ok=True)
    ops = {"Ly": 8, "Lx": 9}
    if kind is not None:
        ops["roi_workflow"] = {"process": kind}
    if n_rois is not None:
        ops["n_rois"] = n_rois
    if pipeline is not None:
        ops["pipeline"] = pipeline
    np.save(path / "ops.npy", ops, allow_pickle=True)
    if n_stat is not None:
        np.save(path / "stat.npy", np.array([_row([0], [0], [1.0])] * n_stat, object))
    return path


class _Settings:
    """Stand-in for a settings dataclass: just carries a to_dict()."""

    def __init__(self, payload, **attrs):
        self._payload = payload
        self.__dict__.update(attrs)

    def to_dict(self):
        return dict(self._payload)


class _Host:
    """Stand-in for the PreviewDataWidget that owns the Process tab's settings."""

    def __init__(self, s2p=None, s2p_db=None, masknmf=None):
        if s2p is not None:
            self.s2p = s2p
        if s2p_db is not None:
            self.s2p_db = s2p_db
        if masknmf is not None:
            self._pipeline_instances = {"MaskNMF": _Settings({}, settings=masknmf)}


def test_masknmf_settings_is_none_until_the_run_tab_builds_it():
    assert rr.masknmf_settings(None) is None
    assert rr.masknmf_settings(_Host()) is None
    assert rr.masknmf_settings(_Host(masknmf=_Settings({"a": 1}))) == {"a": 1}


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
        import logging

        try:
            from mbo_utilities.gui.widgets.pipelines.settings import (
                Suite2pDB,
                Suite2pSettings,
            )
        except Exception as error:  # suite2p / cellpose not importable here
            pytest.skip(f"suite2p settings unavailable: {error}")

        class _Parent:
            logger = logging.getLogger("test.hydrate")

            def __init__(self):
                self.s2p = Suite2pSettings()
                self.s2p_db = Suite2pDB()
                self.s2p_extras = None

        return _Parent()

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
