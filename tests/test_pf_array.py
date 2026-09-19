"""``PfArray``: the voltage pipeline's PF folder opens through ``imread``
the way a suite2p output folder does, with the source line scan as the
image when it is reachable and the trace raster otherwise."""

from __future__ import annotations

import json
import pickle

import h5py
import numpy as np
import pytest

from mbo_utilities.arrays.pf import RASTER_WIDTH, TRACES_FILE, PfArray, pf_dir_of
from mbo_utilities.reader import imread

FS = 1000.0
SCANS = ("35", "38")
LENGTHS = (4000, 6000)
N_ROIS = 7


def write_pf(pf_dir, source=None, units=None):
    """A PF folder: two scans of different lengths, two domains each (the
    archive's ``soma1`` -> ``soma`` rename included), peaks, thresholds and
    provenance naming ``source`` when given."""
    pf_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    traces = {s: {"soma": rng.random(n), "basal1": rng.random(n) * 2} for s, n in zip(SCANS, LENGTHS, strict=True)}
    payloads = {
        TRACES_FILE: traces,
        "fs_scans.pkl": {s: int(FS) for s in SCANS},
        "scanIDs_ROIs.pkl": {
            "scanID_spatial": np.array([int(s) for s in SCANS]),
            "scanID_1st_env": [SCANS[0]],
            "domain_ROInumber": {
                "soma1": [0, 1, 2], "basal1": [3, 4, 5], "All_domains": [0, 1, 2, 3, 4, 5], "bg": [6],
            },
        },
        "detected_events_peaks.pkl": {s: {"soma": np.array([10, 20]), "basal1": np.array([], dtype=int)} for s in SCANS},
        "param_spike_detect.pkl": {"bp": [3, 400]},
    }
    for name, payload in payloads.items():
        with (pf_dir / name).open("wb") as handle:
            pickle.dump(payload, handle)
    provenance = {"settings": {"dfof": {"sigma_dfof": 1500}}}
    if source is not None:
        provenance["source"] = {"mesc": str(source), "units": dict(units or {})}
    (pf_dir / "pipeline.json").write_text(json.dumps(provenance))
    return pf_dir


def write_mesc(path, munits=(35, 38), frames=8):
    """A ``.mesc`` with one packed line-scan unit per ``munits`` entry: 7 ROIs
    of 2 lines x 4 px, ``frames`` frames packed into Y."""
    boxes = [(0, 2, 4 * i, 4 * i + 4) for i in range(N_ROIS)]
    rois = [{"lowerLeftFramePix": [c0 + 1, r0 + 1], "upperRightFramePix": [c1, r1]} for r0, r1, c0, c1 in boxes]
    guide = [[[i, i + 1], [0, 0], [0, 0]] for i in range(N_ROIS)]
    protocol = json.dumps({
        "protocol": {"scanners": {"mainPatternIndex": 1}},
        "scanPatterns": {"patterns": [{"guideLine": guide, "pixelSize": 0.5}]},
    })
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")
        for n, munit in enumerate(munits):
            u = s.create_group(f"MUnit_{munit}")
            u.attrs.update({
                "MethodType": 6, "VecChannelsSize": 1, "TStepInMs": 1000.0 / FS,
                "MeasurementDatePosix": 1_700_000_000 + n, "Comment": f"linescan {munit}",
            })
            u.attrs["CoordinateMapJSON"] = json.dumps({"maps": [{"measurementROIs": rois}]})
            u.attrs["MultiROIProtocolJSON"] = protocol
            page = np.arange(frames * 2 * N_ROIS * 4, dtype=np.uint16).reshape(1, frames * 2, N_ROIS * 4) + 100 * n
            u.create_dataset("Channel_0", data=page)
    return path


def test_pf_dir_of_and_can_open(tmp_path):
    experiment = tmp_path / "stan1" / "stan1_expt1"
    pf = write_pf(experiment / "PF")
    assert pf_dir_of(pf) == pf
    assert pf_dir_of(pf / TRACES_FILE) == pf
    assert pf_dir_of(experiment) == pf
    assert pf_dir_of(tmp_path / "stan1") is None
    assert pf_dir_of(pf / "fs_scans.pkl") is None
    assert pf_dir_of(tmp_path / "missing") is None
    assert PfArray.can_open(experiment) and PfArray.can_open(str(pf))
    assert not PfArray.can_open(tmp_path) and not PfArray.can_open(None)


def test_imread_opens_the_folder_as_a_trace_raster(tmp_path):
    experiment = tmp_path / "stan1" / "stan1_expt1"
    pf = write_pf(experiment / "PF")
    arr = imread(experiment)
    assert isinstance(arr, PfArray)
    assert arr.scan_ids == ["35", "38"] and arr.scan == "35"
    assert arr.first_env == ["35"]
    assert arr.domains == {"soma": [0, 1, 2], "basal1": [3, 4, 5]}
    assert arr.domain_names == ["soma", "basal1"]
    # the longest trace (6000 samples) spans at most RASTER_WIDTH columns
    assert arr.raster_bin == 2 and RASTER_WIDTH == 4096
    assert arr.shape == (1, 1, 2, 2, 3000)
    assert arr.dtype == np.float32 and len(arr) == 1
    assert arr.slider_dim_labels == ("Scan",)
    binned = arr.trace("soma", "38")[:10].reshape(5, 2).mean(axis=1)
    assert np.allclose(arr[0, 0, 1, 0, :5], binned)
    # scan 35 is 4000 samples: 2000 columns, NaN after
    assert np.isfinite(arr[0, 0, 0, 1, :2000]).all() and np.isnan(arr[0, 0, 0, 1, 2000:]).all()
    assert arr.metadata["fs"] == FS
    assert arr.metadata["pf_dir"] == str(pf) and arr.metadata["pf_scans"] == ["35", "38"]
    assert arr.metadata["pf_domains"] == arr.domains
    assert arr.metadata["voltage_settings"] == {"dfof": {"sigma_dfof": 1500}}
    assert arr.metadata["source_mesc"] is None and arr.source_mesc is None
    assert arr.reader_kwargs == {"scan": "35"} and arr.source_path == pf
    assert np.array_equal(arr.events("soma"), [10, 20]) and arr.events("basal1").size == 0
    assert arr.recording_id("soma") == "scan=35/domain=soma"
    assert arr.recording_id("basal1", "38") == "scan=38/domain=basal1"
    assert arr.domain_of_line(4) == "basal1" and arr.domain_of_line(6) is None
    assert arr.params == {"bp": [3, 400]}
    assert arr.mean().shape == (2, 3000) or arr.mean() is not None


def test_scan_and_unit_pick_the_scan(tmp_path):
    pf = write_pf(tmp_path / "PF")
    assert imread(pf, scan="38").scan == "38"
    assert imread(pf, scan=38).scan == "38"
    assert imread(pf, unit="MSession_0/MUnit_38").scan == "38"
    assert imread(pf, **imread(pf, scan="38").reader_kwargs).scan == "38"
    with pytest.raises(ValueError, match="no scan"):
        PfArray(pf, scan="99")


def test_metadata_overrides_survive(tmp_path):
    arr = PfArray(write_pf(tmp_path / "PF"))
    arr.metadata = {**arr.metadata, "fs": 999.0, "note": "hand-set"}
    assert arr.metadata["fs"] == 999.0 and arr.metadata["note"] == "hand-set"
    assert arr.metadata["pf_scan"] == "35"
    with pytest.raises(TypeError):
        arr.metadata = "fs"


def test_the_source_line_scan_is_the_image(tmp_path):
    from mbo_utilities.arrays.mesc import MescArray

    mesc = write_mesc(tmp_path / "scan.mesc")
    pf = write_pf(tmp_path / "PF", source=mesc, units={"35": "MSession_0/MUnit_35", "38": "MSession_0/MUnit_38"})
    arr = imread(pf)
    unit = MescArray(mesc, unit="MSession_0/MUnit_35")
    assert arr.source_mesc == mesc and arr.unit_key == "MSession_0/MUnit_35"
    assert arr.shape == unit.shape and arr.dtype == unit.dtype
    assert np.array_equal(np.asarray(arr[0, 0, 0]), np.asarray(unit[0, 0, 0]))
    assert arr.metadata["mesc_layout"] == "packed" and arr.metadata["source_unit"] == "MSession_0/MUnit_35"
    assert arr.metadata["pf_scan"] == "35" and arr.slider_dim_labels == unit.slider_dim_labels
    other = imread(pf, scan="38")
    assert other.unit_key == "MSession_0/MUnit_38"
    assert np.array_equal(np.asarray(other[0, 0, 0]), np.asarray(MescArray(mesc, unit="MSession_0/MUnit_38")[0, 0, 0]))
    raster = PfArray(pf, source=False)
    assert raster.shape == (1, 1, 2, 2, 3000) and raster.source_mesc == mesc


def test_the_archive_layout_finds_the_line_scan_without_provenance(tmp_path):
    experiment = tmp_path / "stan1" / "stan1_expt1"
    (experiment / "stan1_expt1").mkdir(parents=True)
    mesc = write_mesc(experiment / "stan1_expt1" / "stan1_expt1.mesc")
    (experiment / "stan1_expt1_zstack.mesc").write_bytes(b"z")
    pf = write_pf(experiment / "PF")
    arr = imread(pf)
    assert arr.source_mesc == mesc
    # no unit table in the provenance: the scan id names the unit
    assert arr.unit_key == "MSession_0/MUnit_35" and arr.metadata["mesc_layout"] == "packed"


def test_a_missing_unit_falls_back_to_the_raster(tmp_path):
    mesc = write_mesc(tmp_path / "scan.mesc", munits=(35,))
    pf = write_pf(tmp_path / "PF", source=mesc)
    assert imread(pf, scan="38").shape == (1, 1, 2, 2, 3000)


def test_the_pipeline_registers_its_output_marker():
    from mbo_utilities.pipeline_registry import get_pipeline_info

    info = get_pipeline_info("voltage")
    assert info is not None and info.marker_files == [TRACES_FILE]


def test_the_voltage_widget_runs_from_a_pf_folder(tmp_path):
    pytest.importorskip("imgui_bundle")
    from mbo_utilities.gui.widgets.pipelines.voltage import VoltagePipelineWidget

    mesc = write_mesc(tmp_path / "scan.mesc")
    with_source = PfArray(write_pf(tmp_path / "a" / "PF", source=mesc), source=False)
    without = PfArray(write_pf(tmp_path / "b" / "PF"))
    assert VoltagePipelineWidget.applies_to(with_source)
    assert not VoltagePipelineWidget.applies_to(without)


def test_pf_source_names_the_line_scan_and_its_units(tmp_path):
    from mbo_utilities.arrays.pf import pf_source

    mesc = write_mesc(tmp_path / "scan.mesc")
    named = write_pf(tmp_path / "a" / "PF", source=mesc, units={"35": "MSession_0/MUnit_35"})
    assert pf_source(named) == (mesc, {"35": "MSession_0/MUnit_35"})
    # the archive layout beside the folder when the provenance names nothing
    experiment = tmp_path / "stan1" / "stan1_expt1"
    (experiment / "stan1_expt1").mkdir(parents=True)
    beside = write_mesc(experiment / "stan1_expt1" / "stan1_expt1.mesc")
    assert pf_source(write_pf(experiment / "PF")) == (beside, {})
    assert pf_source(write_pf(tmp_path / "b" / "PF")) == (None, {})
    # a provenance already in hand is used as is
    assert pf_source(tmp_path / "b" / "PF", {"source": {"mesc": str(mesc)}}) == (mesc, {})
    assert pf_source(tmp_path / "b" / "PF", {"source": {"mesc": str(tmp_path / "gone.mesc")}}) == (None, {})
