"""``roi_workflow`` linescan contract tests.

A linescan MESc unit already puts each ROI on its own Z-index, ragged
widths padded to a shared ``Lx`` (see ``tests/test_mesc.py``). These tests
check that the extractor crops each ROI back to its true, unpadded extent
before averaging -- the padding bug this function exists to avoid -- and
that a non-linescan unit is rejected outright.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from mbo_utilities.arrays.mesc import MescArray
from mbo_utilities.analysis.linescan import background_image, pair_reference_zstack, stim_events
from mbo_utilities.roi_workflow import (
    extract_linescan_traces,
    extract_linescan_units,
    linescan_roi_means,
    linescan_roi_read,
)


def _protocol(pattern):
    return json.dumps(
        {
            "protocol": {"scanners": {"mainPatternIndex": 1}},
            "scanPatterns": {"patterns": [pattern]},
        }
    )


def _boxes(boxes):
    """MESc stores 1-based, lower-left/upper-right pixel corners."""
    return [
        {"lowerLeftFramePix": [c0 + 1, r0 + 1], "upperRightFramePix": [c1, r1]}
        for (r0, r1, c0, c1) in boxes
    ]


_GUIDELINE = [[[0, 1], [0, 0], [0, 0]], [[0, 1], [1, 1], [0, 0]]]


@pytest.fixture
def linescan_path(tmp_path):
    """A ``.mesc`` with one linescan unit (2 ragged ROIs) and one chessboard
    unit (real 2D tiles, used to check the modality guard)."""
    path = tmp_path / "linescan.mesc"
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")

        # MUnit_0 - MethodType 6 linescan: 8 frames of 4 lines packed into Y,
        # ROI 0 (width 12) padded to the shared Lx=18 of ROI 1 (width 18).
        u = s.create_group("MUnit_0")
        u.attrs.update(
            {"MethodType": 6, "VecChannelsSize": 1, "TStepInMs": 2.0,
             "MeasurementDatePosix": 1_700_000_000, "Comment": "linescan"}
        )
        u.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": _boxes([(0, 4, 0, 12), (0, 4, 12, 30)])}]}
        )
        u.attrs["MultiROIProtocolJSON"] = _protocol(
            {"guideLine": _GUIDELINE, "pixelSize": 0.5}
        )
        u.create_dataset(
            "Channel_0", data=np.arange(1 * 32 * 30, dtype=np.uint16).reshape(1, 32, 30)
        )

        # MUnit_1 - MethodType 8 chessboard: 4 ROIs tiled along X, real 2D
        # tiles - must be rejected, not silently flattened to a trace.
        u = s.create_group("MUnit_1")
        u.attrs.update(
            {"MethodType": 8, "VecChannelsSize": 1, "TStepInMs": 50.0,
             "MeasurementDatePosix": 1_700_000_100, "Comment": "chessboard"}
        )
        u.attrs["MultiROIProtocolJSON"] = _protocol(
            {
                "centerPoints": np.arange(12).reshape(3, 4).tolist(),
                "pixelSizeX": 0.8,
                "rotation": {"e": [0.0, 0.0, 0.0]},
            }
        )
        u.create_dataset(
            "Channel_0",
            data=np.arange(6 * 32 * 96, dtype=np.uint16).reshape(6, 32, 96),
        )

        # MUnit_2 - a second linescan unit in the same file (one ROI): the
        # per-file helpers must keep its outputs apart from MUnit_0's.
        u = s.create_group("MUnit_2")
        u.attrs.update(
            {"MethodType": 6, "VecChannelsSize": 1, "TStepInMs": 1.0,
             "MeasurementDatePosix": 1_700_000_200, "Comment": "linescan 2"}
        )
        u.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": _boxes([(0, 2, 0, 10)])}]}
        )
        u.attrs["MultiROIProtocolJSON"] = _protocol(
            {"guideLine": _GUIDELINE[:1], "pixelSize": 0.5}
        )
        u.create_dataset(
            "Channel_0", data=np.full((1, 40, 10), 300, dtype=np.uint16)
        )
    return path


def _curve(unit, idx, name, delta_ms, next_sample, values):
    g = unit.create_group(f"Curve_{idx}")
    g.attrs["Name"] = name
    g.attrs["CurveDataXRawDelta"] = float(delta_ms)
    g.create_dataset("CurveDataYIdxNextSample", data=np.asarray(next_sample, dtype=np.int64))
    g.create_dataset("CurveDataYRawData", data=np.asarray(values, dtype=np.float64))


@pytest.fixture
def stim_path(tmp_path):
    """A .mesc shaped like the real AOD spine files: a 2-channel linescan
    unit (offset-coded uint16, a 3-pulse photostim train from the pattern
    sequence, RTMC curves) and a small Z-stack it was drawn on."""
    path = tmp_path / "stim.mesc"
    fs = 1000.0  # TStepInMs = 1
    T, n_rois, w = 400, 3, 8
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")

        z = s.create_group("MUnit_0")
        z.attrs.update(
            {"MethodType": 2, "VecChannelsSize": 2, "TStepInMs": 1.0,
             "MeasurementDatePosix": 0, "Comment": "zstack",
             "MinZ": 0.0, "MaxZ": 8.0, "ZDim": 5}
        )
        z.attrs["ReferenceViewportJSON"] = json.dumps(
            {"viewports": [{"geomTransTransl": [0.0, 0.0, -100.0], "width": 20.0, "height": 20.0}]}
        )
        for c in (0, 1):
            z.create_dataset(f"Channel_{c}", data=rng.integers(1000, 1200, (5, 64, 64)).astype(np.uint16))

        # a whole-cell stack: 400 um across 32 px, contains everything and
        # resolves nothing - must never be picked while MUnit_0 exists
        big = s.create_group("MUnit_2")
        big.attrs.update(
            {"MethodType": 2, "VecChannelsSize": 2, "TStepInMs": 1.0,
             "MeasurementDatePosix": 0, "Comment": "whole cell",
             "MinZ": 0.0, "MaxZ": 90.0, "ZDim": 31}
        )
        big.attrs["ReferenceViewportJSON"] = json.dumps(
            {"viewports": [{"geomTransTransl": [-190.0, -190.0, -150.0], "width": 400.0, "height": 400.0}]}
        )
        for c in (0, 1):
            big.create_dataset(f"Channel_{c}", data=rng.integers(1000, 1200, (31, 32, 32)).astype(np.uint16))

        # the raster snapshot the lines were drawn on, at z = -98 (ROI 0's plane)
        s1 = f.create_group("MSession_1")
        bg = s1.create_group("MUnit_0")
        bg.attrs.update({"MethodType": 1, "VecChannelsSize": 2, "TStepInMs": 1.0, "MeasurementDatePosix": 1})
        bg.attrs["ReferenceViewportJSON"] = json.dumps(
            {"viewports": [{"geomTransTransl": [0.0, 0.0, -98.0], "width": 20.0, "height": 20.0}]}
        )
        for c in (0, 1):
            bg.create_dataset(f"Channel_{c}", data=rng.integers(1000, 1200, (1, 32, 32)).astype(np.uint16))

        u = s.create_group("MUnit_1")
        u.attrs["BackgroundImagePath"] = "/MSession_1/MUnit_0"
        u.attrs.update(
            {"MethodType": 6, "VecChannelsSize": 2, "TStepInMs": 1.0,
             "MeasurementDatePosix": 1, "Comment": "linescan",
             "Channel_0_Conversion_ConversionLinearOffset": -1000.0,
             "Channel_0_Conversion_ConversionLinearScale": 1.0,
             "Channel_1_Conversion_ConversionLinearOffset": -900.0,
             "Channel_1_Conversion_ConversionLinearScale": 1.0,
             "Channel_0_Name": "Green", "Channel_1_Name": "Red"}
        )
        boxes = [(0, 1, i * w, (i + 1) * w) for i in range(n_rois)]
        u.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{
                "measurementROIs": _boxes(boxes),
                "driftEndPoints": [
                    [[2.0, 6.0], [5.0, 5.0], [-98.0, -98.0]],   # z 2 -> slice 1
                    [[8.0, 12.0], [10.0, 10.0], [-94.0, -94.0]],  # z 6 -> slice 3
                    [[14.0, 18.0], [15.0, 15.0], [-94.0, -94.0]],
                ],
            }]}
        )
        u.attrs["MultiROIProtocolJSON"] = json.dumps({
            "protocol": {"scanners": {"mainPatternIndex": 2}},
            "scanPatterns": {"patterns": [
                {"centerPoints": [0, 0, 0], "pixelSizeX": 1.0, "edgeSize": 10},
                {"guideLine": [_GUIDELINE[0]] * n_rois, "pixelSize": 0.5},
                {"centerPoints": [0, 0, 0], "pixelSizeX": 1.0, "edgeSize": 10},
            ]},
        })
        # green: 100 counts above the 1000 offset, a 50-count step 20 ms after
        # the stim on ROI 1 only, stim frames read 30 low; red: flat 150
        green = np.full((T, n_rois * w), 1100.0)
        green[120:220, w:2 * w] += 50.0
        stim_frames = [100, 102, 104]
        green[stim_frames] -= 30.0
        green += rng.normal(0, 2.0, green.shape)
        u.create_dataset("Channel_0", data=green.reshape(1, T, n_rois * w).astype(np.uint16))
        u.create_dataset("Channel_1", data=np.full((1, T, n_rois * w), 1050, np.uint16))
        # PatternSeq_AO1: 3 pulses of pattern 3 (stim), 1 frame each, at 100 ms
        # one curve sample per ROI visit -> delta = 1 ms / n_rois
        per = n_rois
        idx = [100 * per, 101 * per, 102 * per, 103 * per, 104 * per, 105 * per, T * per]
        _curve(u, 0, "PatternSeq_AO1", 1.0 / per, idx, [2, 3, 2, 3, 2, 3, 2])
        t = np.arange(0, T * per, 7)
        _curve(u, 1, "RTMC X correction (total)", 1.0 / per, t, np.sin(t / 50.0))
        _curve(u, 2, "RTMC Y correction (total)", 1.0 / per, t, np.cos(t / 50.0))
        _curve(u, 3, "RTMC Z correction (total)", 1.0 / per, t, np.zeros_like(t, dtype=float))
    return path, fs, stim_frames


def test_crops_padding_before_averaging(linescan_path):
    arr = MescArray(linescan_path, unit=0)
    extents = arr.metadata["mesc_roi_extents"]
    assert (extents[0]["height"], extents[0]["width"]) == (4, 12)
    assert (extents[1]["height"], extents[1]["width"]) == (4, 18)

    out = extract_linescan_traces(arr, compute_dfof=True)
    F = np.load(out / "F.npy")
    Fneu = np.load(out / "Fneu.npy")
    dfof = np.load(out / "dfof.npy")
    stat = np.load(out / "stat.npy", allow_pickle=True)

    assert F.shape == (2, 8)
    assert np.array_equal(Fneu, np.zeros_like(F))
    assert dfof.shape == F.shape
    assert np.all(np.isfinite(dfof))
    assert [int(s["width"]) for s in stat] == [12, 18]

    # the reference: crop each ROI to its *true* extent before averaging --
    # arr[t, 0, z] without cropping would still include ROI 0's zero padding
    for z, ext in enumerate(extents):
        h, w = ext["height"], ext["width"]
        expected = np.array(
            [arr[t, 0, z][:h, :w].mean() for t in range(8)], dtype=np.float32
        )
        assert np.allclose(F[z], expected)
        # padded columns are real zeros in the unpadded array too -- confirm
        # they would have pulled the mean down if left in, so this is a
        # real assertion and not a no-op for a full-width ROI
        if w < arr.shape[-1]:
            padded_mean = np.array(
                [arr[t, 0, z][:h].mean() for t in range(8)], dtype=np.float32
            )
            assert not np.allclose(F[z], padded_mean)


def test_non_linescan_unit_rejected(linescan_path):
    arr = MescArray(linescan_path, unit=1)
    assert arr.metadata["mesc_layout"] != "packed"
    with pytest.raises(ValueError, match="linescan"):
        extract_linescan_traces(arr)


def test_dfof_window_scales_with_fs(linescan_path):
    """A window sized in seconds must actually use fs, not a fixed frame count."""
    arr = MescArray(linescan_path, unit=0)
    out_short = extract_linescan_traces(
        arr, out_dir=arr.filenames[0].parent / "rois_a", dfof_window_s=0.001
    )
    out_long = extract_linescan_traces(
        arr, out_dir=arr.filenames[0].parent / "rois_b", dfof_window_s=50.0
    )
    dfof_short = np.load(out_short / "dfof.npy")
    dfof_long = np.load(out_long / "dfof.npy")
    assert not np.allclose(dfof_short, dfof_long)


def test_roi_means_matches_written_F(linescan_path):
    arr = MescArray(linescan_path, unit=0)
    seen = []
    F = linescan_roi_means(arr, progress=lambda i, k, s: seen.append((i, k)))
    assert seen == [(0, 2), (1, 2)]
    out = extract_linescan_traces(arr, out_dir=linescan_path.parent / "rois_m")
    assert np.array_equal(F, np.load(out / "F.npy"))
    # a batch smaller than T reads the same numbers
    assert np.array_equal(F, linescan_roi_means(arr, batch_size=3))


def test_units_in_one_file_get_their_own_default_dirs(linescan_path):
    """Two linescan units of one .mesc must not overwrite each other."""
    outputs = extract_linescan_units(linescan_path)
    assert sorted(outputs) == ["MSession_0/MUnit_0", "MSession_0/MUnit_2"]
    dirs = set(outputs.values())
    assert len(dirs) == 2
    assert outputs["MSession_0/MUnit_0"] == linescan_path.parent / "rois_linescan" / "MUnit_0"
    assert outputs["MSession_0/MUnit_2"] == linescan_path.parent / "rois_linescan" / "MUnit_2"
    assert np.load(outputs["MSession_0/MUnit_0"] / "F.npy").shape == (2, 8)
    assert np.load(outputs["MSession_0/MUnit_2"] / "F.npy").shape == (1, 20)
    # the single-unit entry point nests the same way when no out_dir is given
    assert extract_linescan_traces(MescArray(linescan_path, unit=2)) == outputs["MSession_0/MUnit_2"]


def test_units_filter_and_out_root(linescan_path, tmp_path):
    out_root = tmp_path / "results"
    outputs = extract_linescan_units(linescan_path, out_root=out_root, units=["MUnit_2"])
    assert list(outputs) == ["MSession_0/MUnit_2"]
    assert outputs["MSession_0/MUnit_2"] == out_root / "MUnit_2"
    assert not (out_root / "MUnit_0").exists()
    # full keys work too
    assert list(extract_linescan_units(linescan_path, out_root=out_root, units=["MSession_0/MUnit_0"])) == [
        "MSession_0/MUnit_0"
    ]
    # naming a non-linescan unit is an error, not a silent skip
    with pytest.raises(ValueError, match="not a linescan"):
        extract_linescan_units(linescan_path, out_root=out_root, units=["MUnit_1"])


def test_nonpositive_baseline_warns(linescan_path):
    """A ROI whose baseline is <= 0 gets a warning, not a silent bad dF/F."""
    import logging

    class _Catch(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    catch = _Catch()
    logger = logging.getLogger("test_linescan_warn")
    logger.addHandler(catch)
    logger.setLevel(logging.INFO)

    with h5py.File(linescan_path, "r+") as f:
        f["MSession_0/MUnit_2/Channel_0"][...] = 0
    arr = MescArray(linescan_path, unit=2)
    out = extract_linescan_traces(arr, out_dir=linescan_path.parent / "rois_zero", logger=logger)
    assert any("baseline <= 0" in m and "[0]" in m for m in catch.messages)
    assert (out / "dfof.npy").exists()

    catch.messages.clear()
    extract_linescan_traces(
        MescArray(linescan_path, unit=0), out_dir=linescan_path.parent / "rois_ok", logger=logger
    )
    assert not any("baseline <= 0" in m for m in catch.messages)


def test_conversion_offset_applied(stim_path, tmp_path):
    """F is in the file's converted counts (zero = no photons), not raw uint16."""
    path, fs, _ = stim_path
    arr = MescArray(path, unit=1)
    conv = arr.metadata["mesc_channel_conversion"]
    assert [c["offset"] for c in conv] == [-1000.0, -900.0]
    F_raw = linescan_roi_means(arr, channel=0)
    F, kymo = linescan_roi_read(arr, channel=0, bin_frames=10)
    F_nc, _ = linescan_roi_read(arr, channel=0, convert=False)
    assert np.allclose(F, F_raw) and np.allclose(F_nc, F + 1000.0)
    assert abs(float(np.median(F[0])) - 100.0) < 5
    assert kymo.shape == (3, 40, 8) and abs(float(np.nanmean(kymo[2])) - 100.0) < 5
    F1, _ = linescan_roi_read(arr, channel=1)
    assert np.allclose(F1, 150.0)


def test_stim_events_from_pattern_sequence(stim_path):
    path, fs, stim_frames = stim_path
    ev = stim_events(path, "MSession_0/MUnit_1", fs, 400)
    assert ev["frames"].tolist() == stim_frames
    assert len(ev["onsets_s"]) == 1 and abs(ev["onsets_s"][0] - 0.100) < 1e-6
    assert len(ev["pulses_s"]) == 3 and ev["patterns"] == [2]
    assert abs(ev["durations_s"][0] - 0.005) < 1e-6
    # the zstack has no pattern sequence at all
    assert stim_events(path, "MSession_0/MUnit_0", 1.0, 5) is None


def test_reference_zstack_pairing(stim_path):
    path, _, _ = stim_path
    ref = pair_reference_zstack(path, "MSession_0/MUnit_1")
    assert ref["munit"] == "MUnit_0" and ref["xy_fraction"] == 1.0 and ref["z_fraction"] == 1.0
    assert ref["coarse"] is False and abs(ref["um_per_px"] - 20 / 64) < 1e-9
    # the fine stack keeps winning when the lines sit below its depth range:
    # depth coverage is flagged, not used to fall back to the coarse stack
    with h5py.File(path, "r+") as f:
        u = f["MSession_0/MUnit_1"]
        cm = json.loads(u.attrs["CoordinateMapJSON"])
        for seg in cm["maps"][0]["driftEndPoints"]:
            seg[2] = [-108.0, -108.0]  # 8 um below the stack's bottom slice
        u.attrs["CoordinateMapJSON"] = json.dumps(cm)
    ref = pair_reference_zstack(path, "MSession_0/MUnit_1")
    assert ref["munit"] == "MUnit_0" and ref["z_fraction"] == 0.0 and ref["coarse"] is False
    # with the fine stack gone the coarse one is used, and says so
    units = [u for u in __import__("mbo_utilities.arrays.mesc", fromlist=["list_mesc_units"]).list_mesc_units(path)
             if u["munit"] != "MUnit_0"]
    ref = pair_reference_zstack(path, "MSession_0/MUnit_1", units)
    assert ref["munit"] == "MUnit_2" and ref["coarse"] is True
    bg = background_image(path, "MSession_0/MUnit_1")
    assert bg["key"] == "MSession_1/MUnit_0" and bg["z"] == -98.0 and bg["nchannels"] == 2
    assert bg["shape"] == (1, 32, 32) and bg["width"] == 20.0
    assert background_image(path, "MSession_0/MUnit_0") is None


def test_extraction_with_stimulus(stim_path, tmp_path):
    path, fs, stim_frames = stim_path
    out = extract_linescan_traces(MescArray(path, unit=1), out_dir=tmp_path / "rois")
    assert np.load(out / "stim_frames.npy").tolist() == stim_frames
    F = np.load(out / "F.npy")
    # F.npy keeps the low stim frames as recorded; dfof is bridged over them
    assert F[0, 102] < F[0, 101] - 20
    dfof = np.load(out / "dfof.npy")
    assert abs(dfof[0, 102] - dfof[0, 101]) < 0.05
    # the 50-count step on ROI 1 shows as a ~50 % stimulus-aligned response
    stat = np.load(out / "stat.npy", allow_pickle=True)
    assert stat[1]["response_mode"] == "stim"
    assert 0.4 < stat[1]["peak_dfof"] < 0.6 and stat[0]["peak_dfof"] < 0.1
    assert stat[1]["snr"] > stat[0]["snr"]
    assert (out / "F_chan1.npy").exists() and (out / "kymographs_chan1.npy").exists()
    ops = np.load(out / "ops.npy", allow_pickle=True).item()["roi_workflow"]
    assert ops["stim_n_pulses"] == 3 and ops["reference_zstack"]["munit"] == "MUnit_0"
    assert ops["background_image"]["munit"] == "MUnit_0" and ops["background_image"]["z"] == -98.0
    names = sorted(p.name for p in out.glob("[0-9][0-9]*_*.png"))
    assert names == [
        "01a_background_snapshot_lines.png",
        "01b_reference_zstack_lines.png",
        "01c_line_zooms.png",
        "02_line_profiles.png",
        "03a_kymographs_green.png",
        "03b_kymographs_red.png",
        "04a_traces_raw.png",
        "04b_traces_dfof.png",
        "05_stim_response.png",
        "06_roi_response_metrics.png",
        "07_motion_correction.png",
    ]
    # rerunning replaces the figure set rather than accumulating
    out2 = extract_linescan_traces(MescArray(path, unit=1), out_dir=tmp_path / "rois", figures=False)
    assert out2 == out and not list(out.glob("*.png"))


def test_no_stimulus_metrics(linescan_path):
    out = extract_linescan_traces(MescArray(linescan_path, unit=0), out_dir=linescan_path.parent / "ns")
    stat = np.load(out / "stat.npy", allow_pickle=True)
    assert all(s["response_mode"] == "spontaneous" for s in stat)
    assert not (out / "stim_frames.npy").exists()
    assert not (out / "05_stim_response.png").exists()
    assert (out / "06_roi_response_metrics.png").exists()


def test_cli_linescan(linescan_path, tmp_path):
    from click.testing import CliRunner
    from mbo_utilities.cli import main

    out_root = tmp_path / "cli_out"
    r = CliRunner().invoke(main, ["linescan", str(linescan_path), "-o", str(out_root)])
    assert r.exit_code == 0, r.output
    assert (out_root / "MUnit_0" / "F.npy").exists()
    assert (out_root / "MUnit_2" / "dfof.npy").exists()
    assert "MSession_0/MUnit_0: 2 ROI(s) x 8 timepoints at 500.0 Hz" in r.output
    assert "widths px: 12, 18" in r.output

    r = CliRunner().invoke(
        main, ["linescan", str(linescan_path), "-o", str(tmp_path / "one"), "--unit", "MUnit_2", "--no-dfof"]
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "one" / "MUnit_2" / "F.npy").exists()
    assert not (tmp_path / "one" / "MUnit_2" / "dfof.npy").exists()
    assert not (tmp_path / "one" / "MUnit_0").exists()

    r = CliRunner().invoke(main, ["linescan", str(linescan_path), "--unit", "MUnit_1"])
    assert r.exit_code != 0
    assert "not a linescan" in r.output
