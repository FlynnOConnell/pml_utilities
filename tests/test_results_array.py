"""``ResultsArray``: a run's output opens through ``imread`` the way a suite2p
output folder does, whichever pipeline wrote it, with the recording as the
image when it is reachable and the trace raster otherwise.
"""

from __future__ import annotations

import json
import pickle

import h5py
import numpy as np
import pytest
from mbo_utilities.reader import imread
from mbo_utilities.results import (
    RASTER_WIDTH,
    TRACES_PKL,
    ResultsArray,
    open_results,
    pipeline_files,
    results_dir_of,
)

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
        TRACES_PKL: traces,
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


def test_results_dir_of_finds_the_folder(tmp_path):
    experiment = tmp_path / "stan1" / "stan1_expt1"
    pf = write_pf(experiment / "PF")
    assert results_dir_of(pf) == pf
    assert results_dir_of(pf / TRACES_PKL) == pf
    assert results_dir_of(experiment) == pf
    assert results_dir_of(tmp_path / "stan1") is None
    assert results_dir_of(pf / "fs_scans.pkl") is None
    assert results_dir_of(tmp_path / "missing") is None
    assert ResultsArray.can_open(experiment) and ResultsArray.can_open(str(pf))
    assert not ResultsArray.can_open(tmp_path) and not ResultsArray.can_open(None)
    # a PF folder of pickles keeps its own files; a results file holds them in its own folder
    assert pipeline_files(pf) == pf


def test_imread_opens_a_pf_folder_as_a_trace_raster(tmp_path):
    experiment = tmp_path / "stan1" / "stan1_expt1"
    pf = write_pf(experiment / "PF")
    arr = imread(experiment)
    assert isinstance(arr, ResultsArray)
    assert arr.pipeline == "voltage"
    assert list(arr.results.units) == ["scan35", "scan38"] and arr.unit == "scan35"
    scan35 = arr.results.units["scan35"]
    assert scan35.attrs["first_env"] and not arr.results.units["scan38"].attrs["first_env"]
    assert scan35.roi_names == ["soma", "basal1"]
    assert [m.tolist() for m in scan35.members] == [[0, 1, 2], [3, 4, 5]]
    # the longest trace (6000 samples) spans at most RASTER_WIDTH columns
    assert arr.raster_bin == 2 and RASTER_WIDTH == 4096
    assert arr.shape == (1, 1, 2, 2, 3000)
    assert arr.dtype == np.float32 and len(arr) == 1
    assert arr.slider_dim_labels == ("Unit",)
    binned = arr.trace("soma", unit="scan38")[:10].reshape(5, 2).mean(axis=1)
    assert np.allclose(arr[0, 0, 1, 0, :5], binned)
    # scan 35 is 4000 samples: 2000 columns, NaN after
    assert np.isfinite(arr[0, 0, 0, 1, :2000]).all() and np.isnan(arr[0, 0, 0, 1, 2000:]).all()
    assert arr.metadata["fs"] == FS
    assert arr.metadata["results_path"] == str(pf)
    assert arr.metadata["results_units"] == ["scan35", "scan38"]
    assert arr.metadata["voltage_settings"] == {"dfof": {"sigma_dfof": 1500}}
    assert arr.metadata["source_recording"] is None and arr.source_recording is None
    assert arr.reader_kwargs == {"unit": "scan35"} and arr.source_path == pf
    assert np.array_equal(arr.events("soma"), [10, 20]) and arr.events("basal1").size == 0
    assert scan35.member_roi(4) == 1 and scan35.member_roi(6) is None
    assert arr.mean().shape == (2, 3000) or arr.mean() is not None


def test_unit_picks_the_scan_by_either_name(tmp_path):
    pf = write_pf(tmp_path / "PF")
    assert imread(pf, unit="scan38").unit == "scan38"
    assert imread(pf, unit="MSession_0/MUnit_38").unit == "scan38"
    assert imread(pf, unit="MUnit_38").unit == "scan38"
    assert imread(pf, **imread(pf, unit="scan38").reader_kwargs).unit == "scan38"
    with pytest.raises(ValueError, match="no unit"):
        ResultsArray(pf, unit="scan99")


def test_metadata_overrides_survive(tmp_path):
    arr = ResultsArray(write_pf(tmp_path / "PF"))
    arr.metadata = {**arr.metadata, "fs": 999.0, "note": "hand-set"}
    assert arr.metadata["fs"] == 999.0 and arr.metadata["note"] == "hand-set"
    assert arr.metadata["results_unit"] == "scan35"
    with pytest.raises(TypeError):
        arr.metadata = "fs"


def test_the_source_line_scan_is_the_image(tmp_path):
    from mbo_utilities.arrays.mesc import MescArray

    mesc = write_mesc(tmp_path / "scan.mesc")
    pf = write_pf(tmp_path / "PF", source=mesc, units={"35": "MSession_0/MUnit_35", "38": "MSession_0/MUnit_38"})
    arr = imread(pf)
    unit = MescArray(mesc, unit="MSession_0/MUnit_35")
    assert arr.source_recording == mesc and arr.unit_key == "MSession_0/MUnit_35"
    assert arr.shape == unit.shape and arr.dtype == unit.dtype
    assert np.array_equal(np.asarray(arr[0, 0, 0]), np.asarray(unit[0, 0, 0]))
    assert arr.metadata["mesc_layout"] == "packed" and arr.metadata["source_unit"] == "MSession_0/MUnit_35"
    assert arr.metadata["results_unit"] == "scan35" and arr.slider_dim_labels == unit.slider_dim_labels
    other = imread(pf, unit="scan38")
    assert other.unit_key == "MSession_0/MUnit_38"
    assert np.array_equal(np.asarray(other[0, 0, 0]), np.asarray(MescArray(mesc, unit="MSession_0/MUnit_38")[0, 0, 0]))
    raster = ResultsArray(pf, source=False)
    assert raster.shape == (1, 1, 2, 2, 3000) and raster.source_recording == mesc


def test_the_archive_layout_finds_the_line_scan_without_provenance(tmp_path):
    experiment = tmp_path / "stan1" / "stan1_expt1"
    (experiment / "stan1_expt1").mkdir(parents=True)
    mesc = write_mesc(experiment / "stan1_expt1" / "stan1_expt1.mesc")
    (experiment / "stan1_expt1_zstack.mesc").write_bytes(b"z")
    pf = write_pf(experiment / "PF")
    arr = imread(pf)
    assert arr.source_recording == mesc
    # no unit table in the provenance: the scan id names the unit
    assert arr.unit_key == "MSession_0/MUnit_35" and arr.metadata["mesc_layout"] == "packed"
    assert open_results(pf).source["mesc"] == str(mesc)


def test_a_missing_unit_falls_back_to_the_raster(tmp_path):
    mesc = write_mesc(tmp_path / "scan.mesc", munits=(35,))
    pf = write_pf(tmp_path / "PF", source=mesc)
    assert imread(pf, unit="scan38").shape == (1, 1, 2, 2, 3000)


def test_the_pipeline_registers_its_output_marker():
    from mbo_utilities.pipeline_registry import get_pipeline_info

    info = get_pipeline_info("voltage")
    assert info is not None and info.marker_files == [TRACES_PKL]


def test_the_voltage_widget_runs_from_a_run(tmp_path):
    pytest.importorskip("imgui_bundle")
    from mbo_utilities.gui.widgets.pipelines.voltage import VoltagePipelineWidget

    mesc = write_mesc(tmp_path / "scan.mesc")
    with_source = ResultsArray(write_pf(tmp_path / "a" / "PF", source=mesc), source=False)
    without = ResultsArray(write_pf(tmp_path / "b" / "PF"))
    assert VoltagePipelineWidget.applies_to(with_source)
    assert not VoltagePipelineWidget.applies_to(without)
