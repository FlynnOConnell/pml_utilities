"""The results zarr (AGENTS.md §7.5): naming from filename tags, the write/read
round trip, and molding suite2p-shaped folders into it. The voltage
pipeline's PF conversion is in ``test_voltage_pipeline.py``.
"""

from datetime import datetime

import numpy as np
import pytest
import zarr
from mbo_utilities.arrays.features._dim_tags import filename_tags, parse_tag
from mbo_utilities.results import (
    ResultUnit,
    newest_results,
    read_results,
    results_from_suite2p,
    results_name,
    results_pipeline,
    results_stamp,
    unit_name,
    write_results,
)


def test_filename_tags_follow_the_dim_tag_vocabulary():
    assert [t.to_string() for t in filename_tags("mouse_V1_GCaMP6f_session1.tif")] == [
        "session01"
    ]
    assert [t.to_string() for t in filename_tags("plane_03.bin")] == ["zplane03"]
    assert [t.to_string() for t in filename_tags("Session_2-plane1.tif")] == [
        "session02",
        "zplane01",
    ]
    assert [t.to_string() for t in filename_tags("tp00001-01574_zplane01-14.tif")] == [
        "tp00001-01574",
        "zplane01-14",
    ]
    assert [t.to_string() for t in filename_tags("2026-09-16_session01.zarr")] == [
        "session01"
    ]
    # an animal id, an experiment number and a brain region are not tags
    assert filename_tags("stan112_expt12.mesc") == []
    assert parse_tag("V1") is None and parse_tag("stan112") is None
    tag = parse_tag("session01-03-2")
    assert (tag.start, tag.stop, tag.step) == (1, 3, 2)


def test_results_name_is_the_input_then_a_stamp_then_the_pipeline():
    when = datetime(2026, 9, 16, 14, 30, 22)
    assert (
        results_name("d/session1.mesc", when, pipeline="voltage")
        == "session1.2026-09-16-14-30-22.voltage.zarr"
    )
    assert results_name("run/zplane01_tp00001-01574", when, pipeline="suite2p") == (
        "zplane01_tp00001-01574.2026-09-16-14-30-22.suite2p.zarr"
    )
    assert results_name("stan112_expt12.mesc", when, extra_tags=["scan35"]) == (
        "stan112_expt12.scan35.2026-09-16-14-30-22.zarr"
    )
    # the dot separates the fields, so it cannot survive inside one
    assert results_name("a b/c.d.zarr", when) == "c_d.2026-09-16-14-30-22.zarr"
    assert unit_name("plane", 3) == "zplane03" and unit_name("scan", 35) == "scan35"
    with pytest.raises(ValueError):
        unit_name("tile", 1)


def test_results_stamp_reads_the_timestamp_back():
    when = datetime(2026, 9, 16, 14, 30, 22)
    assert (
        results_stamp(results_name("session1.mesc", when, pipeline="voltage")) == when
    )
    assert results_stamp("d/session1.2026-09-16-14-30-22.voltage.zarr") == when
    # a name from before the convention, and one whose stem is not a stamp
    assert results_stamp("2026-09-16_session01.zarr") is None
    assert results_stamp("session1.zarr") is None


def test_newest_results_picks_the_latest_run_of_a_pipeline(tmp_path):
    unit = ResultUnit(
        name="scan35",
        kind="scan",
        index=35,
        roi_names=["a"],
        traces={"raw": np.zeros((1, 4))},
        member_kind="line",
        members=[np.array([0])],
    )
    made = {}
    for stamp, pipeline in (
        (datetime(2026, 9, 16, 9, 0, 0), "voltage"),
        (datetime(2026, 9, 16, 17, 5, 0), "voltage"),
        (datetime(2026, 9, 17, 8, 0, 0), "suite2p"),
    ):
        name = results_name("session1.mesc", stamp, pipeline=pipeline)
        made[name] = write_results(tmp_path / name, [unit], pipeline=pipeline)
    assert (
        newest_results(tmp_path, "voltage").name
        == "session1.2026-09-16-17-05-00.voltage.zarr"
    )
    assert (
        newest_results(tmp_path, "suite2p").name
        == "session1.2026-09-17-08-00-00.suite2p.zarr"
    )
    assert newest_results(tmp_path).name == "session1.2026-09-17-08-00-00.suite2p.zarr"
    assert newest_results(tmp_path, "masknmf") is None
    assert newest_results(tmp_path / "nope") is None
    # a plain zarr in the folder is not a results file and never wins
    zarr.open_group(str(tmp_path / "plain.zarr"), mode="w", zarr_format=3)
    assert (
        newest_results(tmp_path, "voltage").name
        == "session1.2026-09-16-17-05-00.voltage.zarr"
    )


def test_write_and_read_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    scan = ResultUnit(
        name="scan35",
        kind="scan",
        index=35,
        fs=1075.2688,
        roi_names=["soma", "basal1"],
        traces={"denoised": rng.normal(size=(2, 50)), "dff": rng.normal(size=(2, 50))},
        member_kind="line",
        members=[np.array([0, 1, 2]), np.array([3])],
        member_traces={"raw": rng.normal(size=(4, 50))},
        events={"soma": np.array([30, 5, 12])},
        attrs={"scan_id": "35", "first_env": True},
    )
    plane = ResultUnit(
        name="zplane01",
        kind="plane",
        index=1,
        fs=10.0,
        roi_names=["0", "1", "2"],
        traces={"raw": rng.normal(size=(3, 20)), "spikes": np.zeros((3, 20))},
        members=[np.array([0, 1]), np.array([13]), np.zeros(0, int)],
        weights=[np.array([0.5, 1.0]), np.array([2.0]), np.zeros(0)],
        image_shape=(4, 5),
        iscell=np.array([[1, 0.9], [0, 0.1], [1, 0.5]]),
        images={"mean": rng.normal(size=(4, 5))},
    )
    path = write_results(
        tmp_path
        / results_name("mouse_session1.tif", datetime(2026, 9, 16, 14, 30, 22)),
        [scan, plane],
        pipeline="test",
        source={"path": "mouse_session1.tif"},
        settings={"a": (1, 2)},
        metadata={"fs": 10.0, "si": {"x": np.arange(3)}, "meanImg": np.zeros((4, 5))},
    )
    assert (
        path.name == "mouse_session1.2026-09-16-14-30-22.zarr"
        and results_pipeline(path) == "test"
    )
    assert results_pipeline(tmp_path) is None
    back = read_results(path)
    assert (
        back.pipeline == "test"
        and back.tags == ["session01"]
        and list(back.units) == ["scan35", "zplane01"]
    )
    assert back.source == {"path": "mouse_session1.tif"} and back.settings == {
        "a": [1, 2]
    }
    # metadata is stripped for export: a suite2p summary image does not ride along
    assert (
        back.metadata["fs"] == 10.0
        and back.metadata["si"] == {"x": [0, 1, 2]}
        and "meanImg" not in back.metadata
    )
    s = back["scan35"]
    assert s.kind == "scan" and s.index == 35 and s.fs == pytest.approx(1075.2688)
    assert (
        s.roi_names == ["soma", "basal1"]
        and s.member_kind == "line"
        and s.image_shape is None
    )
    assert [m.tolist() for m in s.members] == [[0, 1, 2], [3]]
    assert [w.tolist() for w in s.weights] == [[1, 1, 1], [1]]
    np.testing.assert_allclose(s.traces["denoised"], scan.traces["denoised"], rtol=1e-6)
    np.testing.assert_allclose(
        s.member_traces["raw"], scan.member_traces["raw"], rtol=1e-6
    )
    assert s.events["soma"].tolist() == [5, 12, 30] and "basal1" not in s.events
    assert s.iscell.shape == (2, 2) and s.iscell.all()
    assert s.attrs == {"scan_id": "35", "first_env": True}
    p = back["zplane01"]
    assert p.n_rois == 3 and p.n_timepoints == 20 and p.image_shape == (4, 5)
    assert [m.tolist() for m in p.members] == [[0, 1], [13], []]
    np.testing.assert_allclose(p.weights[0], [0.5, 1.0])
    np.testing.assert_allclose(p.iscell, plane.iscell)
    np.testing.assert_allclose(p.images["mean"], plane.images["mean"], rtol=1e-6)
    assert p.events == {} and p.member_traces == {}
    with pytest.raises(FileExistsError):
        write_results(path, [scan], pipeline="test")
    write_results(path, [scan], pipeline="test", overwrite=True)
    assert list(read_results(path).units) == ["scan35"]


def test_write_rejects_shapes_outside_the_schema(tmp_path):
    unit = ResultUnit(
        name="zplane01",
        kind="plane",
        index=1,
        roi_names=["0"],
        traces={"raw": np.zeros((1, 5))},
        members=[np.zeros(0, int)],
    )
    with pytest.raises(ValueError, match="no units"):
        write_results(tmp_path / "a.zarr", [], pipeline="x")
    with pytest.raises(ValueError, match="kind"):
        write_results(
            tmp_path / "a.zarr",
            [ResultUnit(name="u", kind="tile", index=0)],
            pipeline="x",
        )
    unit.traces["F"] = np.zeros((1, 5))
    with pytest.raises(ValueError, match="unknown trace kind"):
        write_results(tmp_path / "a.zarr", [unit], pipeline="x")
    del unit.traces["F"]
    unit.traces["dff"] = np.zeros((2, 5))
    with pytest.raises(ValueError, match="expected \\(1, 5\\)"):
        write_results(tmp_path / "a.zarr", [unit], pipeline="x")
    del unit.traces["dff"]
    unit.events = {"9": np.array([1])}
    with pytest.raises(ValueError, match="unknown ROIs"):
        write_results(tmp_path / "a.zarr", [unit], pipeline="x")
    unit.events = {}
    unit.attrs = {"fs": 3.0}
    with pytest.raises(ValueError, match="written by the schema"):
        write_results(tmp_path / "a.zarr", [unit], pipeline="x")


def _suite2p_plane(plane_dir, n_frames=30, pipeline=None):
    rng = np.random.default_rng(1)
    plane_dir.mkdir(parents=True)
    stat = np.array(
        [
            {
                "ypix": np.array([0, 0]),
                "xpix": np.array([0, 1]),
                "lam": np.array([0.5, 1.0], np.float32),
            },
            {
                "ypix": np.array([2]),
                "xpix": np.array([3]),
                "lam": np.array([2.0], np.float32),
            },
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "F.npy", rng.normal(size=(2, n_frames)).astype(np.float32))
    np.save(plane_dir / "Fneu.npy", np.zeros((2, n_frames), np.float32))
    np.save(plane_dir / "spks.npy", np.zeros((2, n_frames), np.float32))
    np.save(plane_dir / "iscell.npy", np.array([[1, 0.8], [0, 0.2]], np.float32))
    ops = {
        "Ly": 4,
        "Lx": 5,
        "fs": 9.5,
        "nframes": n_frames,
        "meanImg": rng.normal(size=(4, 5)).astype(np.float32),
        "max_proj": np.ones((4, 5), np.float32),
        "yoff": np.zeros(n_frames),
    }
    if pipeline:
        ops["pipeline"] = pipeline
        ops["masknmf"] = {"runtime": {"device": "cpu"}}
        np.save(plane_dir / "norm_traces.npy", np.full((2, n_frames), 3.0, np.float32))
    np.save(plane_dir / "ops.npy", ops)


def test_suite2p_and_masknmf_folders_mold_into_results(tmp_path):
    run = tmp_path / "run"
    _suite2p_plane(run / "zplane02_tp00001-00030")
    _suite2p_plane(run / "zplane01_tp00001-00030", pipeline="masknmf")
    units, root = results_from_suite2p(run)
    assert [u.name for u in units] == ["zplane01", "zplane02"] and [
        u.index for u in units
    ] == [1, 2]
    assert root["pipeline"] == "masknmf" and root["settings"] == {
        "runtime": {"device": "cpu"}
    }
    first = units[0]
    assert (
        first.fs == 9.5
        and first.image_shape == (4, 5)
        and first.roi_names == ["0", "1"]
    )
    # flat pixel index is y * Lx + x, with lam as the weight
    assert first.members[0].tolist() == [0, 1] and first.members[1].tolist() == [13]
    np.testing.assert_allclose(first.weights[1], [2.0])
    assert set(first.traces) == {"raw", "neuropil", "spikes", "dff"} and set(
        units[1].traces
    ) == {"raw", "neuropil", "spikes"}
    assert set(first.images) == {"mean", "max"} and first.attrs["plane_dir"].endswith(
        "zplane01_tp00001-00030"
    )
    path = write_results(
        run / results_name(run, datetime(2026, 9, 16, 14, 30, 22), pipeline="masknmf"),
        units,
        **root,
    )
    assert path.name == "run.2026-09-16-14-30-22.masknmf.zarr"
    back = read_results(path)
    assert back.pipeline == "masknmf" and list(back.units) == ["zplane01", "zplane02"]
    np.testing.assert_allclose(back["zplane01"].traces["dff"], 3.0)
    np.testing.assert_allclose(back["zplane02"].iscell, [[1, 0.8], [0, 0.2]])
    # per-frame vectors and images do not ride along in the root metadata
    assert (
        "yoff" not in back.metadata
        and "meanImg" not in back.metadata
        and back.metadata["fs"] == 9.5
    )
    # one plane dir on its own works too, named from its tags
    single, root = results_from_suite2p(run / "zplane02_tp00001-00030")
    assert len(single) == 1 and root["pipeline"] == "suite2p"
    assert results_name(
        run / "zplane02_tp00001-00030", datetime(2026, 9, 16, 14, 30, 22)
    ) == ("zplane02_tp00001-00030.2026-09-16-14-30-22.zarr")
    with pytest.raises(FileNotFoundError):
        results_from_suite2p(tmp_path)


def test_pixel_units_load_as_run_results_for_the_roi_widget(tmp_path):
    from mbo_utilities.roi_workflow import run_result_from_unit

    run = tmp_path / "run"
    _suite2p_plane(run / "zplane02_tp00001-00030", pipeline="masknmf")
    units, root = results_from_suite2p(run)
    path = write_results(run / "2026-09-16_run.zarr", units, **root)
    unit = read_results(path)["zplane02"]
    res = run_result_from_unit(unit, path / unit.name, "masknmf")
    assert (
        res.path == path / "zplane02"
        and res.kind == "masknmf"
        and res.z == 1
        and res.shape == (4, 5)
    )
    assert [s["ypix"].tolist() for s in res.stat] == [[0, 0], [2]] and [
        s["xpix"].tolist() for s in res.stat
    ] == [[0, 1], [3]]
    np.testing.assert_allclose(res.stat[1]["lam"], [2.0])
    assert res.stat[0]["npix"] == 2 and res.stat[0]["med"] == (0.0, 0.5)
    assert (
        res.F.shape == (2, 30)
        and res.Fneu.shape == (2, 30)
        and res.norm.shape == (2, 30)
        and res.iscell.shape == (2, 2)
    )
    assert res.uids is None and res.store_indices is None
    line = ResultUnit(
        name="scan1",
        kind="scan",
        index=1,
        roi_names=["soma"],
        traces={"denoised": np.zeros((1, 5))},
        member_kind="line",
        members=[np.array([0])],
    )
    with pytest.raises(ValueError, match="pixel"):
        run_result_from_unit(line, path / "scan1")


def test_results_files_are_run_dirs_to_the_roi_widget(tmp_path):
    from mbo_utilities.gui import roi_runs as rr
    from mbo_utilities.results import results_summary

    run = tmp_path / "zplane01_tp00001-00030"
    _suite2p_plane(run)
    units, root = results_from_suite2p(run)
    beside = write_results(tmp_path / "2026-09-16_session01.zarr", units, **root)
    scan = ResultUnit(
        name="scan35",
        kind="scan",
        index=35,
        fs=1000.0,
        roi_names=["soma", "basal1"],
        traces={"denoised": np.zeros((2, 5))},
        member_kind="line",
        members=[np.array([0, 1]), np.array([2])],
    )
    (tmp_path / "PF").mkdir()
    pf = write_results(
        tmp_path / "PF" / "2026-09-16_stan1.zarr", [scan], pipeline="voltage"
    )
    assert results_summary(beside) == {
        "pipeline": "suite2p",
        "units": ["zplane01"],
        "n_rois": 2,
        "created": read_results(beside).created,
    }
    assert results_summary(tmp_path) is None
    assert (
        rr.run_dir_complete(beside)
        and rr.run_dir_complete(beside / "zplane01")
        and rr.run_dir_complete(pf)
    )
    assert not rr.run_dir_complete(beside / "zplane09") and not rr.run_dir_complete(
        tmp_path
    )
    rows = {r["path"]: r for r in rr.scan_run_dirs(tmp_path / "raw.tif")}
    assert set(rows) == {run, beside, pf}
    assert rows[beside]["kind"] == "suite2p" and rows[beside]["n_rois"] == 2
    assert rows[pf]["kind"] == "voltage" and rows[pf]["n_rois"] == 2


def test_imread_does_not_open_a_results_file_as_an_image(tmp_path):
    from mbo_utilities.arrays.zarr import ZarrArray

    unit = ResultUnit(
        name="zplane01",
        kind="plane",
        index=1,
        roi_names=["0"],
        traces={"raw": np.zeros((1, 5))},
        members=[np.zeros(0, int)],
    )
    path = write_results(
        tmp_path / "2026-09-16_session01.zarr", [unit], pipeline="suite2p"
    )
    assert not ZarrArray.can_open(path)
    assert ZarrArray.can_open(tmp_path / "movie.zarr")
