"""`mbo voltage` and the Run-tab widget: the spatial JEDI pipeline from a raw line-scan .mesc.

The parity tests need the archive experiment on X: (or the vnoiser data
folder) and skip without it; the end-to-end one is marked slow (about 15
minutes: eight wavelet transforms of a 120 s scan)."""

import json
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
MESC = ARCHIVE / "stan112_expt12" / "stan112_expt12.mesc" if ARCHIVE else None


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
    assert doc["scans"] == ["1", "2"] and doc["domains"] == {"domain1": [0], "domain2": [1], "domain3": [2]}
    with pytest.raises(ValueError, match="frame rate"):
        run_voltage_pipeline(mesc, domains=doc["domains"], units=["MUnit_1", "MUnit_2"], out=tmp_path / "PF_mixed")
    paths = run_voltage_pipeline(mesc, domains=doc["domains"], units=["MUnit_1"], first_env=["1"], out=tmp_path / "PF")
    assert "denoised_trace_scans.pkl" in paths and "detected_events_peaks.pkl" in paths
    files = read_pf(tmp_path / "PF")
    assert files.scan_ids == ["1"] and set(files.domains) >= {"domain1", "domain2", "domain3"}
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
    assert np.allclose(np.load(traces / "scan1_denoised.npy")[0], files.traces["1"]["domain1"], atol=1e-3)
    assert (traces / "domains.csv").read_text().splitlines() == ["row,domain,rois", "0,domain1,0", "1,domain2,1", "2,domain3,2"]
    assert (traces / "scans.csv").read_text().splitlines()[1].startswith("1,MSession_0/MUnit_1,200.0")
    assert (traces / "scan1_peaks.csv").read_text().splitlines()[0] == "domain,frame,time_s"
    peaks = files.peaks["1"]["domain1"]
    # the two injected dips (sign-flipped to peaks) are found; the quiet patches stay near-empty
    assert any(abs(int(p) - 300) <= 3 for p in peaks) and any(abs(int(p) - 700) <= 3 for p in peaks)
    assert len(files.peaks["1"]["domain2"]) <= 5


def test_task_and_widget_are_registered():
    pytest.importorskip("imgui_bundle")
    from types import SimpleNamespace

    from mbo_utilities.gui.tasks import TASKS
    from mbo_utilities.gui.widgets.pipelines.voltage import VoltagePipelineWidget, parse_roi_text

    assert "voltage" in TASKS
    from mbo_utilities.pipeline_registry import get_pipeline_info
    import mbo_utilities.arrays.pf  # noqa: F401  registers the PF folder as the pipeline's output

    assert get_pipeline_info("voltage").marker_files == ["denoised_trace_scans.pkl"]
    assert VoltagePipelineWidget.axis_mode("Z") == "all"
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
    assert len(doc["domains"]) == 8
    assert list(doc["domains"].values())[0] == [0, 1, 2]
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
        spike_cfg=SpikeDetectConfig.from_param_pickle(params), log=lambda *_: None,
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
