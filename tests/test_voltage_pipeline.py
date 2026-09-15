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
    assert not VoltagePipelineWidget.applies_to(SimpleNamespace(metadata={"mesc_layout": "boxes"}, filenames=["a.tif"]))
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
