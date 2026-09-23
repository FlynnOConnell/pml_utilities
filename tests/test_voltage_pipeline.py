"""`mbo voltage` and the Run-tab widget: the spatial JEDI pipeline from a raw line-scan .mesc.

The parity tests need the archive experiment on X: (or the vnoiser data
folder) and skip without it; the end-to-end one is marked slow (about 15
minutes: eight wavelet transforms of a 120 s scan)."""

import json
import os
from logging.handlers import BufferingHandler
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("vnoiser")

from mbo_utilities.vnoiser.params import VoltageSettings  # noqa: E402
from mbo_utilities.vnoiser.pipeline import (  # noqa: E402
    DOMAINS_FILE,
    default_pf_dir,
    read_domains,
    run_voltage_pipeline,
    scan_traces_from_mesc,
    write_domains_template,
)

ARCHIVE = next(
    (
        Path(p)
        for p in (
            "X:/data/asako/stan112/stan112_expt12",
            "C:/Users/flynn/repos/vnoiser/data/stan112/stan112_expt12",
        )
        if Path(p, "PF", "denoised_trace_scans.pkl").exists()
    ),
    None,
)
archive = pytest.mark.skipif(ARCHIVE is None, reason="archive experiment not reachable")


def pkl_settings() -> VoltageSettings:
    """The archive's PF folder of pickles; the default output is the results zarr."""
    settings = VoltageSettings()
    settings.runtime.output_format = "pkl"
    return settings
MESC = ARCHIVE / "stan112_expt12" / "stan112_expt12.mesc" if ARCHIVE else None


class _Progress(list):
    """Records the runner's ``progress_callback(fraction, message)`` calls."""

    def __call__(self, fraction, message):
        self.append((float(fraction), str(message)))


def test_read_domains_json(tmp_path):
    path = tmp_path / DOMAINS_FILE
    path.write_text(json.dumps({"domains": {"soma1": [0, 1], "apical1": [2]}, "scans": [35, "38"]}))
    spec = read_domains(path)
    assert spec == {"domains": {"soma1": [0, 1], "apical1": [2]}, "scan_ids": ["35", "38"], "first_env": []}
    path.write_text(json.dumps({"domains": {}}))
    with pytest.raises(ValueError):
        read_domains(path)


def test_default_pf_dir_follows_the_archive_layout(tmp_path):
    nested = tmp_path / "stan1" / "stan1_expt1" / "stan1_expt1" / "stan1_expt1.mesc"
    assert default_pf_dir(nested) == tmp_path / "stan1" / "stan1_expt1" / "PF"
    flat = tmp_path / "scans" / "todd.mesc"
    assert default_pf_dir(flat) == tmp_path / "scans" / "PF"


def test_settings_roundtrip_and_defaults_are_the_archives():
    settings = VoltageSettings()
    settings.denoiser.soft_levels = (0.7, 0.5, 0.2, 0.05)
    settings.events.detect = False
    back = VoltageSettings.from_dict(json.loads(json.dumps(settings.to_dict())))
    assert back == settings
    assert isinstance(back.denoiser.soft_levels, tuple)
    assert VoltageSettings.from_dict({"dfof": {"sigma_dfof": 10, "bogus": 1}}).dfof.sigma_dfof == 10
    model = VoltageSettings().denoiser.factory(1000.0)
    upstream = model.upstream(1000.0)
    assert model.describe() == upstream.describe()
    assert VoltageSettings().events.config().distance_samples == 3


def test_settings_from_provenance_of_a_written_folder(tmp_path):
    from vnoiser import ScanTraces, read_pf, run_pipeline

    rng = np.random.default_rng(0)
    n = 4000
    traces = {0: 1700 + rng.normal(0, 20, n), 1: 1700 + rng.normal(0, 20, n)}
    scan = ScanTraces("7", 1000.0, traces, {0: 10.0, 1: 10.0})
    settings = VoltageSettings()
    settings.denoiser.n_scales = 32
    settings.events.thres_bp_sd = 3.0
    run_pipeline(
        [scan], tmp_path / "PF", domains={"soma1": [0, 1]}, dfof_cfg=settings.dfof.config(),
        denoiser_factory=settings.denoiser.factory, spike_cfg=settings.events.config(),
        provenance={"settings": settings.to_dict()},
    )
    files = read_pf(tmp_path / "PF")
    assert VoltageSettings.from_provenance(files.provenance) == settings
    # a folder made without the settings block still maps back from the denoiser description
    prov = dict(files.provenance)
    del prov["settings"]
    again = VoltageSettings.from_provenance(prov)
    assert again.denoiser.n_scales == 32 and again.events.thres_bp_sd == 3.0
    assert again.denoiser.thres_type == "soft" and again.denoiser.complex_bands is True


def test_settings_scale_to_the_frame_rate():
    """The sample-count parameters keep the archive's durations at another frame rate."""
    settings = VoltageSettings()
    assert settings.at_fs(1075.2688) == settings
    fast = settings.at_fs(1000 / 4.8)
    k = (1000 / 4.8) / 1075.2688
    assert fast.dfof.sigma_dfof == pytest.approx(1500 * k)
    assert fast.dfof.sigma_baseline == pytest.approx(5000 * k)
    assert fast.dfof.n_startup == round(1000 * k)
    assert fast.denoiser.scale_min == 1.0 and fast.denoiser.scale_max == pytest.approx(1000 * k)
    assert fast.events.distance_samples == 1
    assert fast.events.bp_high == pytest.approx(0.95 * (1000 / 4.8) / 2)
    assert fast.events.bp_low == 2.0
    assert fast.runtime.reference_fs == pytest.approx(1000 / 4.8)
    assert fast.at_fs(1000 / 4.8) == fast
    # what is not counted in samples stays: the wavelet count, the FIR window in ms, the thresholds
    assert fast.denoiser.n_scales == 100 and fast.denoiser.fir_window_ms == 2000.0
    assert fast.events.thres_bp_sd == 3.5 and fast.denoiser.soft_levels == settings.denoiser.soft_levels
    # the band-pass is capped at any rate, even the reference one
    settings.events.bp_high = 600.0
    assert settings.at_fs(1075.2688).events.bp_high == pytest.approx(0.95 * 1075.2688 / 2)


def _chessboard_mesc(path, *, n_frames=1200, step_ms=5.0, extra_unit=False):
    """A .mesc with one MethodType 8 unit: three 20 x 20 patches tiled along X, JEDI-like
    dips in patch 0 at frames 300 and 700; optionally a second unit at half the rate."""
    import h5py

    rng = np.random.default_rng(0)
    page = rng.normal(1200, 15, (n_frames, 20, 60))
    # two 30 ms-wide dips in patch 0, the shape a 200 Hz scan resolves (a 1 ms spike is sub-sample)
    t = np.arange(n_frames)[:, None, None]
    for centre in (300, 700):
        page[:, :, 0:20] -= 80 * np.exp(-0.5 * ((t - centre) / 2.5) ** 2)
    pattern = {
        "centerPoints": [[10.0, 40.0, 70.0], [5.0, 5.0, 5.0], [-100.0, -100.0, -100.0]],
        "pixelSizeX": 1.0, "pixelSizeY": 1.0, "edgeSize": 20,
        "rotation": {"e": [0.0, 0.0, 0.0]},
    }
    protocol = json.dumps({"protocol": {"scanners": {"mainPatternIndex": 1}}, "scanPatterns": {"patterns": [pattern]}})
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")
        for munit, ms in (("MUnit_1", step_ms), ("MUnit_2", step_ms * 2)):
            if munit == "MUnit_2" and not extra_unit:
                continue
            u = s.create_group(munit)
            u.attrs.update({"MethodType": 8, "VecChannelsSize": 2, "TStepInMs": ms, "MeasurementDatePosix": 1_700_000_000,
                            "Comment": "", "ImageRoleDebugString": "measurement"})
            u.attrs["MultiROIProtocolJSON"] = protocol
            for c in range(2):
                u.create_dataset(f"Channel_{c}", data=np.clip(page + 50 * c, 0, 65535).astype(np.uint16))
    return page


def test_chessboard_patches_run_as_scans(tmp_path):
    """A chessboard unit's patches are ROIs like a line scan's lines: read, grouped one per domain,
    denoised with the settings scaled to its frame rate, written as a PF folder."""
    from vnoiser import read_pf

    mesc = tmp_path / "chess.mesc"
    page = _chessboard_mesc(mesc, extra_unit=True)
    scan = scan_traces_from_mesc(mesc, "MUnit_1")
    assert scan.scan_id == "1" and scan.fs_hz == pytest.approx(200.0)
    assert sorted(scan.traces) == [0, 1, 2] and scan.weights == {0: 400.0, 1: 400.0, 2: 400.0}
    raw = np.clip(page, 0, 65535).astype(np.uint16).astype(np.float64)
    # the reader flips chessboard pages in Y, which a patch mean does not see
    assert np.allclose(scan.traces[1], raw[:, :, 20:40].mean(axis=(1, 2)))
    template = write_domains_template(mesc, tmp_path / DOMAINS_FILE)
    doc = json.loads(template.read_text())
    assert doc["scans"] == ["1", "2"] and doc["domains"] == {"roi0": [0], "roi1": [1], "roi2": [2]}
    with pytest.raises(ValueError, match="frame rate"):
        run_voltage_pipeline(mesc, domains=doc["domains"], units=["MUnit_1", "MUnit_2"], out=tmp_path / "PF_mixed",
                             settings=pkl_settings())
    paths = run_voltage_pipeline(mesc, domains=doc["domains"], units=["MUnit_1"], first_env=["1"],
                                 out=tmp_path / "PF", settings=pkl_settings())
    assert "denoised_trace_scans.pkl" in paths and "detected_events_peaks.pkl" in paths
    files = read_pf(tmp_path / "PF")
    assert files.scan_ids == ["1"] and set(files.domains) >= {"roi0", "roi1", "roi2"}
    prov = files.provenance
    assert prov["source"]["units"] == {"1": "MSession_0/MUnit_1"}
    assert prov["fs_hz"]["1"] == pytest.approx(200.0)
    k = 200.0 / 1075.2688
    assert prov["dfof"]["sigma_dfof"] == pytest.approx(1500 * k)
    assert prov["events"]["bp"][1] == pytest.approx(95.0)
    assert prov["events"]["distance_samples"] == 1
    traces = tmp_path / "PF" / "traces"
    assert paths["traces/scan1_denoised.npy"] == traces / "scan1_denoised.npy"
    assert np.load(traces / "scan1_rois.npy").shape == (3, 1200)
    for name in ("dfof", "zscore", "denoised"):
        assert np.load(traces / f"scan1_{name}.npy").shape == (3, 1200)
    assert np.allclose(np.load(traces / "scan1_denoised.npy")[0], files.traces["1"]["roi0"], atol=1e-3)
    assert (traces / "domains.csv").read_text().splitlines() == ["row,domain,rois", "0,roi0,0", "1,roi1,1", "2,roi2,2"]
    assert (traces / "scans.csv").read_text().splitlines()[1].startswith("1,MSession_0/MUnit_1,200.0")
    assert (traces / "scan1_peaks.csv").read_text().splitlines()[0] == "domain,frame,time_s"
    peaks = files.peaks["1"]["roi0"]
    # the two injected dips (sign-flipped to peaks) are found; the quiet patches stay near-empty
    assert any(abs(int(p) - 300) <= 3 for p in peaks) and any(abs(int(p) - 700) <= 3 for p in peaks)
    assert len(files.peaks["1"]["roi1"]) <= 5


def test_the_viewer_opens_a_mesc_on_a_scan_its_pf_folder_holds(tmp_path):
    """`mbo scan.mesc` lands on the first unit the pipeline processed, not the file's first ROI unit."""
    from mbo_utilities.gui.run_gui import _first_linescan_unit

    mesc = tmp_path / "chess.mesc"
    _chessboard_mesc(mesc, extra_unit=True)
    assert _first_linescan_unit(mesc) == "MSession_0/MUnit_1"
    doc = json.loads(write_domains_template(mesc, tmp_path / DOMAINS_FILE).read_text())
    run_voltage_pipeline(mesc, domains=doc["domains"], units=["MUnit_2"], out=tmp_path / "PF",
                         settings=pkl_settings())
    assert _first_linescan_unit(mesc) == "MSession_0/MUnit_2"


def test_planes_pick_the_rois_and_cut_the_domains(tmp_path):
    """The input contract's 1-based ``planes`` are ROIs on an AOD unit (its Z axis): only they are
    read, every domain is cut down to them, an emptied domain is dropped."""
    from vnoiser import read_pf

    from mbo_utilities.roi_workflow import linescan_roi_read
    from mbo_utilities.arrays.mesc import MescArray

    mesc = tmp_path / "chess.mesc"
    _chessboard_mesc(mesc)
    arr = MescArray(mesc, unit="MUnit_1")
    assert arr.metadata["mesc_z_axis_meaning"] == "roi_index"
    full, _ = linescan_roi_read(arr, convert=False, dtype=np.float64)
    part, _ = linescan_roi_read(arr, convert=False, dtype=np.float64, rois=[2, 0])
    assert part.shape == (2, 1200) and np.array_equal(part, full[[2, 0]])
    with pytest.raises(ValueError, match="outside"):
        linescan_roi_read(arr, rois=[3])
    scan = scan_traces_from_mesc(mesc, "MUnit_1", rois=[0, 2])
    assert sorted(scan.traces) == [0, 2] and sorted(scan.weights) == [0, 2]
    assert np.array_equal(scan.traces[2], full[2])
    domains = {"roi0": [0], "roi1": [1], "roi2": [2], "pair": [1, 2]}
    with pytest.raises(ValueError, match="outside 1..3"):
        run_voltage_pipeline(mesc, domains=domains, units=["MUnit_1"], planes=[4], out=tmp_path / "PF_bad",
                             settings=pkl_settings())
    with pytest.raises(ValueError, match="no domain"):
        run_voltage_pipeline(mesc, domains={"roi1": [1]}, units=["MUnit_1"], planes=[1], out=tmp_path / "PF_none",
                             settings=pkl_settings())
    paths = run_voltage_pipeline(mesc, domains=domains, units=["MUnit_1"], planes=[3, 1], out=tmp_path / "PF",
                                 settings=pkl_settings())
    files = read_pf(tmp_path / "PF")
    kept = {k: v for k, v in files.domains.items() if k != "All_domains"}
    assert kept == {"roi0": [0], "roi2": [2], "pair": [2]}
    assert files.rois["roi_list"] == {"1": [0, 2]}
    assert files.provenance["source"]["planes"] == [3, 1]
    assert set(files.traces["1"]) == {"roi0", "roi2", "pair"}
    assert np.load(tmp_path / "PF" / "traces" / "scan1_rois.npy").shape == (2, 1200)
    assert (tmp_path / "PF" / "traces" / "domains.csv").read_text().splitlines() == [
        "row,domain,rois", "0,roi0,0", "1,roi2,2", "2,pair,2",
    ]
    assert "traces/scan1_denoised.npy" in paths


def test_zarr_is_the_default_output_and_holds_the_whole_run(tmp_path):
    """The default ``output_format="zarr"`` writes one
    ``<input>.<stamp>.voltage.zarr`` beside the input, no pickles and no PF
    folder, with everything else the run made in a ``voltage/`` folder inside
    it; ResultsArray, imread and the mesc lookups open it the same way."""
    from mbo_utilities import imread
    from mbo_utilities.results import (
        ResultsArray,
        newest_results,
        pipeline_files,
        read_results,
        results_stamp,
    )
    from mbo_utilities.vnoiser import voltage_run_for_mesc, voltage_unit_for_mesc

    mesc = tmp_path / "chess_session1.mesc"
    _chessboard_mesc(mesc)
    doc = json.loads(write_domains_template(mesc, tmp_path / DOMAINS_FILE).read_text())
    settings = VoltageSettings()
    assert settings.runtime.output_format == "zarr"
    paths = run_voltage_pipeline(
        mesc, domains=doc["domains"], units=["MUnit_1"], first_env=["1"],
        settings=settings, provenance={"settings": settings.to_dict()},
    )
    zarr_path = newest_results(tmp_path, "voltage")
    zarr_name = zarr_path.name
    # named after its input, stamped to the second, beside the file and nowhere else
    assert zarr_name.startswith("chess_session1.") and zarr_name.endswith(".voltage.zarr")
    assert results_stamp(zarr_path) is not None and paths[zarr_name] == zarr_path
    assert not (tmp_path / "PF").exists() and not any(p.suffix == ".work" for p in tmp_path.iterdir())
    assert not any(k.endswith(".pkl") for k in paths)
    own, traces = pipeline_files(zarr_path), zarr_path / "voltage" / "traces"
    assert own == zarr_path / "voltage"
    assert (own / "test.h5").is_file() and (own / "pipeline.json").is_file()
    assert (traces / "scan1_denoised.npy").is_file() and "voltage/test.h5" in paths
    results = read_results(paths[zarr_name])
    assert results.pipeline == "voltage" and list(results.units) == ["scan1"] and results.tags == ["session01"]
    assert results.settings["runtime"]["output_format"] == "zarr"
    assert results.source["units"] == {"1": "MSession_0/MUnit_1"} and results.provenance["fs_hz"] == {"1": 200.0}
    scan = results["scan1"]
    assert scan.kind == "scan" and scan.index == 1 and scan.fs == pytest.approx(200.0)
    assert scan.roi_names == ["roi0", "roi1", "roi2"] and scan.member_kind == "line"
    assert [m.tolist() for m in scan.members] == [[0], [1], [2]] and scan.attrs["member_ids"] == [0, 1, 2]
    assert scan.attrs["source_unit"] == "MSession_0/MUnit_1" and scan.attrs["first_env"] is True
    np.testing.assert_allclose(scan.traces["denoised"], np.load(traces / "scan1_denoised.npy"), rtol=1e-6)
    np.testing.assert_allclose(scan.traces["dff"], np.load(traces / "scan1_dfof.npy"), rtol=1e-6)
    np.testing.assert_allclose(scan.traces["zscore"], np.load(traces / "scan1_zscore.npy"), rtol=1e-6)
    np.testing.assert_allclose(scan.member_traces["raw"], np.load(traces / "scan1_rois.npy"), rtol=1e-6)
    assert any(abs(int(p) - 300) <= 3 for p in scan.events["roi0"])
    # the results file itself is what opens as a ResultsArray
    run = ResultsArray(zarr_path, source=False)
    assert run.path == paths[zarr_name] and run.pipeline == "voltage" and run.unit == "scan1"
    assert list(run.results.units) == ["scan1"] and run.results.units["scan1"].fs == pytest.approx(200.0)
    np.testing.assert_allclose(run.trace("roi0"), scan.traces["denoised"][0])
    assert run.events("roi0").tolist() == scan.events["roi0"].tolist() and run.events("roi1").size <= 5
    assert run.results.settings == results.settings
    assert isinstance(imread(zarr_path), ResultsArray) and isinstance(imread(tmp_path), ResultsArray)
    assert voltage_run_for_mesc(mesc) == zarr_path
    assert voltage_unit_for_mesc(mesc, "MSession_0/MUnit_1").unit == "scan1"
    assert voltage_unit_for_mesc(mesc, "MUnit_2") is None
    # a rerun writes its own file; the stamp orders them and the newest wins
    again = run_voltage_pipeline(
        mesc, domains=doc["domains"], units=["MUnit_1"], settings=settings, overwrite=True,
    )
    second = next(p for k, p in again.items() if k.endswith(".zarr"))
    assert results_stamp(second) >= results_stamp(zarr_path)
    assert newest_results(tmp_path, "voltage") == second and list(read_results(second).units) == ["scan1"]


def test_every_step_is_timed_and_logged(tmp_path):
    """Each ROI read, each scan's dF/F, each domain's denoising and each write is logged with its
    time and memory; the same record lands in pipeline.json (``timing``, ``processing_history``),
    in timings.json, in the results zarr's provenance and in ResultsArray.metadata; the progress
    callback runs from the first ROI to 1.0 in order."""
    from mbo_utilities import log
    from mbo_utilities.results import read_results
    from mbo_utilities.vnoiser.pipeline import TIMINGS_FILE

    mesc = tmp_path / "chess_session2.mesc"
    _chessboard_mesc(mesc)
    doc = json.loads(write_domains_template(mesc, tmp_path / DOMAINS_FILE).read_text())
    settings = VoltageSettings()
    settings.runtime.output_format = "zarr"
    logger = log.get("tests.voltage")
    records = BufferingHandler(10_000)
    logger.addHandler(records)
    progress = _Progress()
    try:
        paths = run_voltage_pipeline(
            mesc, domains=doc["domains"], units=["MUnit_1"], settings=settings,
            progress_callback=progress, logger=logger,
        )
    finally:
        logger.removeHandler(records)
    from mbo_utilities.results import ResultsArray, newest_results, pipeline_files

    zarr_path = newest_results(tmp_path, "voltage")
    own = pipeline_files(zarr_path)
    messages = [r.getMessage() for r in records.buffer]
    assert any(m.startswith("voltage: 1 scan(s) x 3 domain(s)") and "cpus" in m for m in messages)
    assert any(m.startswith("scan 1: read ROI 3/3") for m in messages)
    assert any(m.startswith("scan 1: read 3 ROIs x 1200 frames in ") and "(cpu " in m and " GB" in m for m in messages)
    assert any(m.startswith("scan 1: dF/F and z-score of roi2 (3/3) in ") for m in messages)
    assert any(m == "scan 1: denoising roi1 (2/3)" for m in messages)
    assert any(m.startswith("scan 1: denoised roi0 (1/3), ") and " events [cwt " in m and ", peaks " in m for m in messages)
    assert any(m.startswith("wrote traces/ for 1 scan(s) in ") for m in messages)
    assert any(m.startswith("voltage done in ") and "peak process memory" in m and "timings.json" in m for m in messages)

    prov = json.loads((own / "pipeline.json").read_text())
    timing, history = prov["timing"], prov["processing_history"]
    assert list(timing["totals"]) == ["read", "dfof", "denoise", "write_pf", "traces", "figures", "results"]
    assert all(seconds > 0 for seconds in timing["totals"].values())
    assert timing["wall_seconds"] >= sum(timing["totals"].values()) - 0.01
    assert timing["cpu_seconds"] > 0 and timing["peak_rss_gb"] > 0 and timing["cpu_count"] == os.cpu_count()
    assert timing["started"] <= timing["finished"]
    scan = timing["scans"]["1"]
    assert scan["read"] > 0 and scan["dfof"] > 0 and list(scan["domains"]) == ["roi0", "roi1", "roi2"]
    assert scan["denoise"] == pytest.approx(sum(d["denoise"] for d in scan["domains"].values()), abs=1e-2)
    assert scan["dfof"] == pytest.approx(sum(d["dfof"] for d in scan["domains"].values()), abs=1e-2)
    stages = ["cwt", "cluster", "reduce", "mask", "baseline", "baseline_100hz", "peaks"]
    assert list(timing["denoise_stages"]) == stages and timing["denoise_stages"]["cwt"] > 0
    assert timing["denoise_stages"]["cwt"] <= timing["totals"]["denoise"]
    assert [h["step"] for h in history[:2]] == ["voltage_read", "voltage_dfof"] and history[-1]["step"] == "voltage_results"
    denoise = [h for h in history if h["step"] == "voltage_denoise"]
    assert [h["domain"] for h in denoise] == ["roi0", "roi1", "roi2"] and all(h["scan"] == "1" for h in denoise)
    assert all({"timestamp", "duration_seconds", "cpu_seconds", "rss_gb", "peak_rss_gb", "n_events"} <= set(h) for h in denoise)
    assert all({f"{s}_s" for s in stages} <= set(h) for h in denoise)
    assert history[0]["unit"] == "MSession_0/MUnit_1" and history[0]["n_rois"] == 3 and history[0]["n_frames"] == 1200

    assert paths[f"voltage/{TIMINGS_FILE}"] == own / TIMINGS_FILE
    timings = json.loads((own / TIMINGS_FILE).read_text())
    assert timings["totals"] == timing["totals"] and timings["wall_seconds"] == timing["wall_seconds"]
    assert [s["step"] for s in timings["steps"]] == [h["step"].removeprefix("voltage_") for h in history]
    assert sum(s["seconds"] for s in timings["steps"] if s["step"] == "denoise") == pytest.approx(timing["totals"]["denoise"], abs=1e-2)

    results = read_results(zarr_path)
    assert results.provenance["timing"] == timing and results.provenance["processing_history"] == history
    md = ResultsArray(zarr_path, source=False).metadata
    assert md["timing"] == timing and md["processing_history"] == history

    fractions = [f for f, _ in progress]
    assert fractions == sorted(fractions) and 0 < fractions[0] < 1 and fractions[-1] == 1.0
    assert progress[0][1] == "scan 1: read ROI 1/3"
    assert [m for _, m in progress if m.startswith("scan 1: dF/F")] == [f"scan 1: dF/F of roi{i} ({i + 1}/3)" for i in range(3)]
    assert [m for _, m in progress if m.startswith("scan 1: denoising")] == [f"scan 1: denoising roi{i} ({i + 1}/3)" for i in range(3)]
    assert any(m == "writing the results zarr" for _, m in progress)


def test_task_and_widget_are_registered():
    pytest.importorskip("imgui_bundle")
    from types import SimpleNamespace

    from mbo_utilities.gui.tasks import TASKS
    from mbo_utilities.gui.widgets.pipelines.voltage import VoltagePipelineWidget, parse_roi_text

    assert "voltage" in TASKS
    from mbo_utilities.pipeline_registry import get_pipeline_info
    import mbo_utilities.results  # noqa: F401  registers the voltage pipeline and its output

    assert get_pipeline_info("voltage").marker_files == ["denoised_trace_scans.pkl"]
    assert VoltagePipelineWidget.axis_mode("Z") == "range"  # Z is the ROI index on an AOD unit
    assert VoltagePipelineWidget.applies_to(SimpleNamespace(metadata={"mesc_layout": "packed"}))
    # chessboard patches and ribbon boxes are ROIs too
    assert VoltagePipelineWidget.applies_to(SimpleNamespace(metadata={"mesc_layout": "tiled"}))
    assert VoltagePipelineWidget.applies_to(SimpleNamespace(metadata={"mesc_layout": "boxes"}, filenames=["a.tif"]))
    assert not VoltagePipelineWidget.applies_to(SimpleNamespace(metadata={"mesc_layout": "frames"}, filenames=["a.tif"]))
    assert not VoltagePipelineWidget.applies_to(None)
    if ARCHIVE is not None:
        zstack = SimpleNamespace(metadata={"mesc_layout": "multicube"}, filenames=[str(ARCHIVE / "stan112_expt12_zstack.mesc")])
        assert not VoltagePipelineWidget.applies_to(zstack)
        other_unit = SimpleNamespace(metadata={"mesc_layout": "multicube"}, filenames=[str(MESC)])
        assert VoltagePipelineWidget.applies_to(other_unit)
    assert parse_roi_text("0, 2:4, 1", 8) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        parse_roi_text("0, 9", 8)
    with pytest.raises(ValueError):
        parse_roi_text("", 8)


@archive
def test_domains_template_lists_the_scans_and_groups_lines(tmp_path):
    path = write_domains_template(MESC, tmp_path / DOMAINS_FILE)
    doc = json.loads(path.read_text())
    assert doc["scans"] == ["35", "38"]
    assert doc["first_env"] == ["35"]
    assert len(doc["domains"]) == 24 and doc["domains"]["roi0"] == [0]
    grouped = json.loads(write_domains_template(MESC, tmp_path / "threes.json", per_domain=3).read_text())
    assert len(grouped["domains"]) == 8 and list(grouped["domains"].values())[0] == [0, 1, 2]
    spec = read_domains(path)
    assert spec["scan_ids"] == ["35", "38"]


@archive
def test_per_line_traces_match_the_archives_packaged_raw():
    from vnoiser import load_vi

    scan = scan_traces_from_mesc(MESC, "MUnit_35")
    assert scan.scan_id == "35"
    assert scan.fs_hz == pytest.approx(1075.2688, abs=1e-3)
    assert scan.weights[1] == 20.0 and scan.weights[0] == 10.0
    _animal, _experiment, theirs = load_vi(next(ARCHIVE.glob("VI_*.pkl")))
    for roi, trace in scan.traces.items():
        assert np.abs(trace - theirs["35"].traces[roi]).max() < 1e-9, roi
    assert scan.weights == theirs["35"].weights
    window = scan_traces_from_mesc(MESC, "MUnit_35", frames=(1000, 3000))
    assert window.n_frames == 2000
    assert np.array_equal(window.traces[0], scan.traces[0][1000:3000])
    with pytest.raises(ValueError):
        scan_traces_from_mesc(MESC, "MUnit_35", frames=(0, 10**9))


@archive
def test_archive_rois_pickle_reads_as_domains():
    spec = read_domains(ARCHIVE / "PF" / "scanIDs_ROIs.pkl")
    assert spec["scan_ids"] == ["35", "38"]
    assert spec["domains"]["soma1"] == [0, 1, 2]
    assert "All_domains" in spec["domains"]


@archive
@pytest.mark.slow
def test_scan_35_from_the_mesc_reproduces_the_archive_pf(tmp_path):
    """The acceptance test: the raw .mesc that made stan112_expt12's PF
    folder, through this pipeline, gives the same traces."""
    from vnoiser import SpikeDetectConfig, read_pf
    from vnoiser.dataset import _restricted_pickle_load

    spec = read_domains(ARCHIVE / "PF" / "scanIDs_ROIs.pkl")
    params = _restricted_pickle_load(ARCHIVE / "PF" / "param_spike_detect.pkl")
    out = tmp_path / "stan112" / "stan112_expt12" / "PF"
    paths = run_voltage_pipeline(
        MESC, domains=spec["domains"], units=["MUnit_35"], first_env=["35"], out=out,
        spike_cfg=SpikeDetectConfig.from_param_pickle(params), settings=pkl_settings(),
    )
    assert "denoised_trace_scans.pkl" in paths
    ours = read_pf(out)
    theirs = read_pf(ARCHIVE / "PF")
    assert ours.fs == {"35": theirs.fs["35"]}
    assert set(ours.traces["35"]) == set(theirs.traces["35"])
    for name, trace in ours.traces["35"].items():
        diff = np.abs(trace - theirs.traces["35"][name])
        assert np.median(diff) < 1e-6 and (diff < 1e-5).mean() > 0.999, name
    for name in ("basal1", "basal3", "apical1", "apical4", "soma"):
        assert np.array_equal(ours.peaks["35"][name], theirs.peaks["35"][name]), name
    assert ours.provenance["source"]["units"] == {"35": "MSession_0/MUnit_35"}
    assert ours.provenance["source"]["mesc"] == str(MESC)


@pytest.mark.parametrize("layout", ["beside", "pf", "legacy"])
def test_widget_seeds_from_a_previous_zarr_run(tmp_path, layout):
    """A run written as a results zarr seeds the scans, domains and settings
    whether it landed beside the file (the default) or inside a ``PF`` folder,
    with its own files in the results file's own folder or, as an older run
    left them, loose beside it; the output folder follows the format, so a zarr
    run never writes ``PF``.
    """
    pytest.importorskip("imgui_bundle")
    from types import SimpleNamespace

    import h5py
    from mbo_utilities.gui.widgets.pipelines.settings import _MISSING_COLOR
    from mbo_utilities.gui.widgets.pipelines.voltage import VoltagePipelineWidget
    from mbo_utilities.results import ResultUnit, pipeline_files, results_name, write_results

    mesc = tmp_path / "session1.mesc"
    with h5py.File(mesc, "w") as f:
        unit = f.create_group("MSession_0").create_group("MUnit_3")
        unit.attrs.update({"MethodType": 6, "VecChannelsSize": 1, "TStepInMs": 1.0, "MeasurementDatePosix": 0})
        unit.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": [
                {"lowerLeftFramePix": [2 * i + 1, 1], "upperRightFramePix": [2 * i + 2, 1]} for i in range(4)
            ]}]}
        )
        unit.create_dataset("Channel_0", data=np.zeros((1, 20, 8), np.uint16))

    out = tmp_path if layout == "beside" else tmp_path / "PF"
    out.mkdir(exist_ok=True)
    scan = ResultUnit(
        name="scan3", kind="scan", index=3, fs=1000.0, roi_names=["soma", "basal"],
        traces={"denoised": np.zeros((2, 20))}, member_kind="line",
        members=[np.array([0, 1]), np.array([2])], attrs={"scan_id": "3", "first_env": True},
    )
    results = write_results(out / results_name(mesc, pipeline="voltage"), [scan], pipeline="voltage")
    if layout == "legacy":
        own = out
    else:
        own = pipeline_files(results)
        own.mkdir()
    (own / "pipeline.json").write_text(json.dumps({
        "domains": {"All_domains": [0, 1, 2], "soma": [0, 1], "basal": [2]},
        "scan_ids": ["3"], "first_env": ["3"],
        "settings": {"runtime": {"output_format": "zarr", "reference_fs": 1000.0}},
    }))

    widget = VoltagePipelineWidget(SimpleNamespace(fpath=mesc, image_widget=None))
    widget._ensure_state()
    assert widget._status_color is not _MISSING_COLOR, widget._last_status
    assert widget._last_status.startswith("Loaded the previous run")
    assert widget._domain_rows == [["soma", "0,1"], ["basal", "2"]]
    assert widget._scans == {"MSession_0/MUnit_3": True}
    assert widget._first_env == {"MSession_0/MUnit_3": True}
    assert widget.settings.runtime.reference_fs == 1000.0
    assert widget._outdir == str(tmp_path)
