"""``linescan_roi_means`` applies the channel conversion and ``pair_reference_zstack`` finds the stack a line scan was drawn on."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from mbo_utilities.analysis.linescan import (
    background_image,
    pair_reference_zstack,
)
from mbo_utilities.arrays.mesc import MescArray
from mbo_utilities.roi_workflow import (
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
    unit (real 2D tiles, used to check the modality guard).
    """
    path = tmp_path / "linescan.mesc"
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")

        # MUnit_0 - MethodType 6 linescan: 8 frames of 4 lines packed into Y,
        # ROI 0 (width 12) padded to the shared Lx=18 of ROI 1 (width 18).
        u = s.create_group("MUnit_0")
        u.attrs.update(
            {
                "MethodType": 6,
                "VecChannelsSize": 1,
                "TStepInMs": 2.0,
                "MeasurementDatePosix": 1_700_000_000,
                "Comment": "linescan",
            }
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
            {
                "MethodType": 8,
                "VecChannelsSize": 1,
                "TStepInMs": 50.0,
                "MeasurementDatePosix": 1_700_000_100,
                "Comment": "chessboard",
            }
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
            {
                "MethodType": 6,
                "VecChannelsSize": 1,
                "TStepInMs": 1.0,
                "MeasurementDatePosix": 1_700_000_200,
                "Comment": "linescan 2",
            }
        )
        u.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": _boxes([(0, 2, 0, 10)])}]}
        )
        u.attrs["MultiROIProtocolJSON"] = _protocol(
            {"guideLine": _GUIDELINE[:1], "pixelSize": 0.5}
        )
        u.create_dataset("Channel_0", data=np.full((1, 40, 10), 300, dtype=np.uint16))
    return path


def _curve(unit, idx, name, delta_ms, next_sample, values):
    g = unit.create_group(f"Curve_{idx}")
    g.attrs["Name"] = name
    g.attrs["CurveDataXRawDelta"] = float(delta_ms)
    g.create_dataset(
        "CurveDataYIdxNextSample", data=np.asarray(next_sample, dtype=np.int64)
    )
    g.create_dataset("CurveDataYRawData", data=np.asarray(values, dtype=np.float64))


@pytest.fixture
def stim_path(tmp_path):
    """A .mesc shaped like the real AOD spine files: a 2-channel linescan
    unit (offset-coded uint16, a 3-pulse photostim train from the pattern
    sequence, RTMC curves) and a small Z-stack it was drawn on.
    """
    path = tmp_path / "stim.mesc"
    fs = 1000.0  # TStepInMs = 1
    T, n_rois, w = 400, 3, 8
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")

        z = s.create_group("MUnit_0")
        z.attrs.update(
            {
                "MethodType": 2,
                "VecChannelsSize": 2,
                "TStepInMs": 1.0,
                "MeasurementDatePosix": 0,
                "Comment": "zstack",
                "MinZ": 0.0,
                "MaxZ": 8.0,
                "ZDim": 5,
            }
        )
        z.attrs["ReferenceViewportJSON"] = json.dumps(
            {
                "viewports": [
                    {
                        "geomTransTransl": [0.0, 0.0, -100.0],
                        "width": 20.0,
                        "height": 20.0,
                    }
                ]
            }
        )
        for c in (0, 1):
            z.create_dataset(
                f"Channel_{c}",
                data=rng.integers(1000, 1200, (5, 64, 64)).astype(np.uint16),
            )

        # a whole-cell stack: 400 um across 32 px, contains everything and
        # resolves nothing - must never be picked while MUnit_0 exists
        big = s.create_group("MUnit_2")
        big.attrs.update(
            {
                "MethodType": 2,
                "VecChannelsSize": 2,
                "TStepInMs": 1.0,
                "MeasurementDatePosix": 0,
                "Comment": "whole cell",
                "MinZ": 0.0,
                "MaxZ": 90.0,
                "ZDim": 31,
            }
        )
        big.attrs["ReferenceViewportJSON"] = json.dumps(
            {
                "viewports": [
                    {
                        "geomTransTransl": [-190.0, -190.0, -150.0],
                        "width": 400.0,
                        "height": 400.0,
                    }
                ]
            }
        )
        for c in (0, 1):
            big.create_dataset(
                f"Channel_{c}",
                data=rng.integers(1000, 1200, (31, 32, 32)).astype(np.uint16),
            )

        # the raster snapshot the lines were drawn on, at z = -98 (ROI 0's plane)
        s1 = f.create_group("MSession_1")
        bg = s1.create_group("MUnit_0")
        bg.attrs.update(
            {
                "MethodType": 1,
                "VecChannelsSize": 2,
                "TStepInMs": 1.0,
                "MeasurementDatePosix": 1,
            }
        )
        bg.attrs["ReferenceViewportJSON"] = json.dumps(
            {
                "viewports": [
                    {
                        "geomTransTransl": [0.0, 0.0, -98.0],
                        "width": 20.0,
                        "height": 20.0,
                    }
                ]
            }
        )
        for c in (0, 1):
            bg.create_dataset(
                f"Channel_{c}",
                data=rng.integers(1000, 1200, (1, 32, 32)).astype(np.uint16),
            )

        u = s.create_group("MUnit_1")
        u.attrs["BackgroundImagePath"] = "/MSession_1/MUnit_0"
        u.attrs.update(
            {
                "MethodType": 6,
                "VecChannelsSize": 2,
                "TStepInMs": 1.0,
                "MeasurementDatePosix": 1,
                "Comment": "linescan",
                "Channel_0_Conversion_ConversionLinearOffset": -1000.0,
                "Channel_0_Conversion_ConversionLinearScale": 1.0,
                "Channel_1_Conversion_ConversionLinearOffset": -900.0,
                "Channel_1_Conversion_ConversionLinearScale": 1.0,
                "Channel_0_Name": "Green",
                "Channel_1_Name": "Red",
            }
        )
        boxes = [(0, 1, i * w, (i + 1) * w) for i in range(n_rois)]
        u.attrs["CoordinateMapJSON"] = json.dumps(
            {
                "maps": [
                    {
                        "measurementROIs": _boxes(boxes),
                        "driftEndPoints": [
                            [[2.0, 6.0], [5.0, 5.0], [-98.0, -98.0]],  # z 2 -> slice 1
                            [
                                [8.0, 12.0],
                                [10.0, 10.0],
                                [-94.0, -94.0],
                            ],  # z 6 -> slice 3
                            [[14.0, 18.0], [15.0, 15.0], [-94.0, -94.0]],
                        ],
                    }
                ]
            }
        )
        u.attrs["MultiROIProtocolJSON"] = json.dumps(
            {
                "protocol": {"scanners": {"mainPatternIndex": 2}},
                "scanPatterns": {
                    "patterns": [
                        {"centerPoints": [0, 0, 0], "pixelSizeX": 1.0, "edgeSize": 10},
                        {"guideLine": [_GUIDELINE[0]] * n_rois, "pixelSize": 0.5},
                        {"centerPoints": [0, 0, 0], "pixelSizeX": 1.0, "edgeSize": 10},
                    ]
                },
            }
        )
        # green: 100 counts above the 1000 offset, a 50-count step 20 ms after
        # the stim on ROI 1 only, stim frames read 30 low; red: flat 150
        green = np.full((T, n_rois * w), 1100.0)
        green[120:220, w : 2 * w] += 50.0
        stim_frames = [100, 102, 104]
        green[stim_frames] -= 30.0
        green += rng.normal(0, 2.0, green.shape)
        u.create_dataset(
            "Channel_0", data=green.reshape(1, T, n_rois * w).astype(np.uint16)
        )
        u.create_dataset("Channel_1", data=np.full((1, T, n_rois * w), 1050, np.uint16))
        # PatternSeq_AO1: 3 pulses of pattern 3 (stim), 1 frame each, at 100 ms
        # one curve sample per ROI visit -> delta = 1 ms / n_rois
        per = n_rois
        idx = [
            100 * per,
            101 * per,
            102 * per,
            103 * per,
            104 * per,
            105 * per,
            T * per,
        ]
        _curve(u, 0, "PatternSeq_AO1", 1.0 / per, idx, [2, 3, 2, 3, 2, 3, 2])
        t = np.arange(0, T * per, 7)
        _curve(u, 1, "RTMC X correction (total)", 1.0 / per, t, np.sin(t / 50.0))
        _curve(u, 2, "RTMC Y correction (total)", 1.0 / per, t, np.cos(t / 50.0))
        _curve(
            u,
            3,
            "RTMC Z correction (total)",
            1.0 / per,
            t,
            np.zeros_like(t, dtype=float),
        )
    return path, fs, stim_frames


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


def test_reference_zstack_pairing(stim_path):
    path, _, _ = stim_path
    ref = pair_reference_zstack(path, "MSession_0/MUnit_1")
    assert (
        ref["munit"] == "MUnit_0"
        and ref["xy_fraction"] == 1.0
        and ref["z_fraction"] == 1.0
    )
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
    assert (
        ref["munit"] == "MUnit_0"
        and ref["z_fraction"] == 0.0
        and ref["coarse"] is False
    )
    # with the fine stack gone the coarse one is used, and says so
    units = [
        u
        for u in __import__(
            "mbo_utilities.arrays.mesc", fromlist=["list_mesc_units"]
        ).list_mesc_units(path)
        if u["munit"] != "MUnit_0"
    ]
    ref = pair_reference_zstack(path, "MSession_0/MUnit_1", units)
    assert ref["munit"] == "MUnit_2" and ref["coarse"] is True
    bg = background_image(path, "MSession_0/MUnit_1")
    assert (
        bg["key"] == "MSession_1/MUnit_0" and bg["z"] == -98.0 and bg["nchannels"] == 2
    )
    assert bg["shape"] == (1, 32, 32) and bg["width"] == 20.0
    assert background_image(path, "MSession_0/MUnit_0") is None
