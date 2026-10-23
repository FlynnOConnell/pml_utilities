"""Domain traces for the vnoiser pipeline: which pixels, what is subtracted, which way up.

A one-patch chessboard (the recordings most AOD voltage sessions hold) with a
dimming cell in the middle of the patch, a background around it, a slow
artifact shared with a reference channel and MESc's conversion offset.
"""

import json

import numpy as np
import pytest
from mbo_utilities.analysis.dff import BASELINES, dfof_zscore
from mbo_utilities.analysis.roi_traces import (
    Domain,
    TraceSettings,
    extract_traces,
    invert,
    roi_geometry,
)

FS = 1000.0
N_FRAMES = 6000
SPIKES = (1500, 3000, 4500)
CELL = (slice(7, 13), slice(7, 13))
OFFSET = -1000.0


def _patch_movies(n_frames=N_FRAMES):
    """``(indicator, reference)`` pages: a 6 x 6 cell at 1800 counts dimming 10 %
    at each spike, a 1200-count background, a 2 Hz artifact in both channels.
    """
    rng = np.random.default_rng(1)
    t = np.arange(n_frames)[:, None, None]
    artifact = 40 * np.sin(2 * np.pi * 2.0 * t / FS)
    green = rng.normal(1200, 6, (n_frames, 20, 20)) + artifact
    dips = np.zeros((n_frames, 1, 1))
    for centre in SPIKES:
        dips -= 0.1 * np.exp(-0.5 * ((t - centre) / 3.0) ** 2)
    cell = 600 * (1 + dips)
    green[:, CELL[0], CELL[1]] += cell
    red = rng.normal(1100, 6, (n_frames, 20, 20)) + 2 * artifact
    return green, red


def _single_patch_mesc(path, *, n_frames=N_FRAMES):
    """A .mesc whose one measurement unit is a single 20 x 20 chessboard patch, with a picture
    unit of one frame beside it (which holds no trace).
    """
    import h5py

    green, red = _patch_movies(n_frames)
    pattern = {
        "centerPoints": [[10.0], [10.0], [-100.0]],
        "pixelSizeX": 1.0,
        "pixelSizeY": 1.0,
        "edgeSize": 20,
        "rotation": {"e": [0.0, 0.0, 0.0]},
    }
    protocol = json.dumps(
        {
            "protocol": {"scanners": {"mainPatternIndex": 1}},
            "scanPatterns": {"patterns": [pattern]},
        }
    )
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")
        u = s.create_group("MUnit_4")
        u.attrs.update(
            {
                "MethodType": 8,
                "VecChannelsSize": 2,
                "TStepInMs": 1000.0 / FS,
                "MeasurementDatePosix": 1_700_000_000,
                "Comment": "cell",
                "ImageRoleDebugString": "measurement",
                "MultiROIProtocolJSON": protocol,
            }
        )
        for c, page in enumerate((green, red)):
            u.attrs[f"Channel_{c}_Conversion_ConversionLinearScale"] = 1.0
            u.attrs[f"Channel_{c}_Conversion_ConversionLinearOffset"] = OFFSET
            u.create_dataset(
                f"Channel_{c}", data=np.clip(page, 0, 65535).astype(np.uint16)
            )
        pic = s.create_group("MUnit_5")
        pic.attrs.update(
            {
                "MethodType": 1,
                "VecChannelsSize": 1,
                "TStepInMs": 100.0,
                "MeasurementDatePosix": 1_700_000_000,
                "ImageRoleDebugString": "measurement",
            }
        )
        pic.create_dataset("Channel_0", data=np.zeros((1, 20, 20), np.uint16))
    return green, red


@pytest.fixture
def patch(tmp_path):
    from mbo_utilities.arrays.mesc import MescArray

    path = tmp_path / "cell_session1.mesc"
    green, red = _single_patch_mesc(path)
    arr = MescArray(path, unit="MUnit_4")
    # the reader flips chessboard pages in Y; the cell square is symmetric about the middle
    return path, arr, green[:, ::-1], red[:, ::-1]


def _uint(page):
    return np.clip(page, 0, 65535).astype(np.uint16).astype(np.float64)


def test_a_single_patch_is_one_roi_of_the_whole_frame(patch):
    from mbo_utilities.arrays.mesc import is_trace_unit, list_mesc_units
    from mbo_utilities.vnoiser.pipeline import scan_units

    path, arr, green, _ = patch
    assert arr.metadata["mesc_layout"] == "frames"
    assert roi_geometry(arr) == {0: (0, 20, 20)}
    units = {u["key"]: u for u in list_mesc_units(path)}
    assert is_trace_unit(units["MSession_0/MUnit_4"])
    assert not is_trace_unit(units["MSession_0/MUnit_5"])
    assert [u["key"] for u in scan_units(path)] == ["MSession_0/MUnit_4"]
    with pytest.raises(ValueError, match="no trace"):
        scan_units(path, ["MUnit_5"])
    found = extract_traces(arr, {"patch": [0]})
    np.testing.assert_allclose(found.F[0], _uint(green).mean(axis=(1, 2)), rtol=1e-12)
    np.testing.assert_allclose(found.roi_mean[0], found.F[0])


def test_ragged_boxes_on_a_flipped_layout_sit_at_the_bottom():
    """A shorter box of a Y-flipped ribbon is padded below and flipped, so its pixels are the last
    rows of the frame; ``roi_geometry`` reads that from the extents.
    """

    class Unit:
        shape = (10, 1, 2, 8, 6)
        metadata = {
            "mesc_z_axis_meaning": "roi_index",
            "mesc_roi_extents": [
                {"index": 0, "y_start": 0, "y_end": 8, "height": 8, "width": 6},
                {"index": 1, "y_start": 3, "y_end": 8, "height": 5, "width": 4},
            ],
        }

    assert roi_geometry(Unit()) == {0: (0, 8, 6), 1: (3, 5, 4)}


def test_pixel_rules_find_the_cell(patch):
    _, arr, green, _ = patch
    raw = _uint(green)
    cell = np.zeros((20, 20), bool)
    cell[CELL] = True
    domains = [
        Domain("all", [0]),
        Domain("bright", [0], pixels="bright", fraction=36 / 400),
        Domain("correlated", [0], pixels="correlated", fraction=36 / 400),
        Domain(
            "mask", [0], pixels="mask", mask={0: np.flatnonzero(cell.ravel()).tolist()}
        ),
    ]
    # a 20 Hz high-pass keeps the 2 Hz artifact both channels share out of the scores
    found = extract_traces(
        arr, domains, TraceSettings(score_seconds=6.0, score_highpass_hz=20.0)
    )
    for name in ("bright", "correlated", "mask"):
        picked = np.zeros(400, bool)
        picked[found.masks[name][0]] = True
        assert picked.reshape(20, 20)[CELL].all(), name
        np.testing.assert_allclose(
            found.F[found.row(name)], raw[:, CELL[0], CELL[1]].mean(axis=(1, 2))
        )
    assert set(found.images[0]) == {"mean", "sd", "corr"}
    # the cell dims 60 of its 1800 counts (3 %); over the whole patch that is under 0.5 %
    depth = {
        n: 1 - found.F[found.row(n)][1500] / np.median(found.F[found.row(n)])
        for n in ("all", "mask")
    }
    assert depth["mask"] > 0.03 and depth["all"] < 0.006
    with pytest.raises(ValueError, match="outside"):
        extract_traces(arr, [Domain("x", [0], pixels="mask", mask={0: [400]})])
    with pytest.raises(ValueError, match="no mask pixels"):
        Domain("x", [0], pixels="mask", mask={})


def test_background_reference_conversion_and_inversion(patch):
    _, arr, green, red = patch
    raw, ref = _uint(green), _uint(red)
    cell = raw[:, CELL[0], CELL[1]].mean(axis=(1, 2))
    mask = {0: [r * 20 + c for r in range(7, 13) for c in range(7, 13)]}
    domain = Domain("cell", [0], pixels="mask", mask=mask)
    rest = np.ones((20, 20), bool)
    rest[CELL] = False

    out = extract_traces(
        arr, [domain], TraceSettings(convert=True, background="outside")
    )
    np.testing.assert_allclose(out.pixel_mean[0], cell + OFFSET)
    np.testing.assert_allclose(out.background[0], raw[:, rest].mean(axis=1) + OFFSET)
    np.testing.assert_allclose(out.F[0], cell - raw[:, rest].mean(axis=1))

    dim = extract_traces(
        arr, [domain], TraceSettings(background="dim", background_percentile=5.0)
    )
    np.testing.assert_allclose(
        dim.background[0], np.percentile(raw.reshape(len(raw), -1), 5.0, axis=1)
    )

    bg = Domain("bg", [0], pixels="mask", mask={0: np.flatnonzero(rest).tolist()})
    named = extract_traces(arr, [domain, bg], TraceSettings(background="domain"))
    assert named.names == ["cell"] and named.info["background_domain"] == "bg"
    np.testing.assert_allclose(named.F[0], out.F[0])
    with pytest.raises(ValueError, match="not among the domains"):
        extract_traces(arr, [domain], TraceSettings(background="domain"))
    # a domain covering the whole patch has nothing outside it: no background, and the run says so
    whole = extract_traces(arr, {"all": [0]}, TraceSettings(background="outside"))
    assert whole.info["no_background"] == ["all"]
    np.testing.assert_allclose(whole.F[0], raw.mean(axis=(1, 2)))

    # the 2 Hz artifact rides on both channels; regressing the reference removes it
    reg = extract_traces(arr, [domain], TraceSettings(reference="regress"))
    assert reg.info["reference_channel"] == 1
    np.testing.assert_allclose(
        reg.reference[0], ref[:, CELL[0], CELL[1]].mean(axis=(1, 2))
    )
    quiet = slice(0, 1200)
    assert reg.F[0][quiet].std() < 0.25 * cell[quiet].std()
    ratio = extract_traces(arr, [domain], TraceSettings(reference="ratio"))
    np.testing.assert_allclose(
        ratio.F[0], cell / reg.reference[0] * reg.reference[0].mean()
    )
    with pytest.raises(ValueError, match="another of the 2 channels"):
        extract_traces(
            arr, [domain], TraceSettings(reference="ratio", reference_channel=0)
        )

    # pml_utilities' Invert Deflection on the movie is the inversion of the pixel mean
    from mbo_utilities.arrays._inverted_view import InvertedDeflectionView

    flipped = InvertedDeflectionView(arr)
    inv = extract_traces(arr, [domain], TraceSettings(invert=True))
    np.testing.assert_allclose(inv.measured[0], cell)
    np.testing.assert_allclose(inv.F[0], invert(cell))
    from_view = extract_traces(flipped, [domain])
    np.testing.assert_allclose(from_view.F[0], inv.F[0], rtol=1e-5)
    assert inv.F[0][1500] > np.median(inv.F[0])


def test_every_baseline_method(patch):
    from vnoiser.preprocess import domain_zscore

    _, arr, green, _ = patch
    raw = _uint(green)
    rest = np.ones((20, 20), bool)
    rest[CELL] = False
    # the cell minus the patch around it: the shared artifact goes, the dips stay
    cell = raw[:, CELL[0], CELL[1]].mean(axis=(1, 2)) - raw[:, rest].mean(axis=1)
    rng = np.random.default_rng(0)
    F = np.stack(
        [
            cell,
            900
            + 300 * np.exp(-np.arange(N_FRAMES) / 2000)
            + rng.normal(0, 2, N_FRAMES),
        ]
    )
    archive = domain_zscore({0: F[0], 1: F[1]}, {"a": [0], "b": [1]})
    dfof, z = dfof_zscore(F, FS)
    assert np.array_equal(dfof, archive.dfof_raw) and np.array_equal(z, archive.z)
    for method in BASELINES:
        dfof, z = dfof_zscore(F, FS, method=method, noise="mad", n_startup=0)
        assert np.isfinite(dfof).all() and np.isfinite(z).all(), method
        # a 10 % dip of the cell is a positive z event once negated
        assert z[0][1500] > 3, method
    dfof, _ = dfof_zscore(F, FS, method="exponential", n_startup=0)
    assert np.abs(dfof[1]).max() < 0.02
    dfof, _ = dfof_zscore(F, FS, method="window", f0_window_s=(0.0, 0.5), n_startup=0)
    np.testing.assert_allclose(dfof[0][:5], F[0][:5] / F[0][:500].mean() - 1)
    dfof, _ = dfof_zscore(F, FS, method="none", n_startup=0)
    np.testing.assert_array_equal(dfof, F)


def test_settings_polarity_and_round_trip():
    from mbo_utilities.vnoiser.params import VnoiserSettings

    settings = VnoiserSettings()
    assert settings.polarity == "flip_z" and not settings.traces.convert
    settings.polarity = "invert_f"
    assert settings.traces.invert and not settings.dfof.negative
    settings.polarity = "none"
    assert not settings.traces.invert and not settings.dfof.negative
    settings.traces.invert = settings.dfof.negative = True
    with pytest.raises(ValueError, match="inverted twice"):
        settings.polarity
    with pytest.raises(ValueError, match="polarity must be"):
        settings.polarity = "up"
    rec = VnoiserSettings.recommended()
    rec.traces.background = "dim"
    rec.dfof.method = "percentile"
    rec.runtime.denoise = False
    back = VnoiserSettings.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert back == rec and back.traces.convert and back.dfof.noise == "mad"
    # a run written before the traces section kept the conversion under runtime
    old = VnoiserSettings.from_dict({"runtime": {"convert": True}})
    assert old.traces.convert
    with pytest.raises(ValueError, match="dfof.method"):
        VnoiserSettings.from_dict({"dfof": {"method": "median"}})


def test_domains_file_takes_pixel_rules_and_masks(tmp_path):
    from mbo_utilities.vnoiser.pipeline import read_domains

    path = tmp_path / "domains.json"
    path.write_text(
        json.dumps(
            {
                "domains": {
                    "soma": [0, 1, 2],
                    "cell": {"rois": [3], "pixels": "correlated", "fraction": 0.2},
                    "drawn": {"rois": [3], "pixels": "mask", "mask": {"3": [5, 6, 7]}},
                },
                "scans": ["4"],
            }
        )
    )
    spec = read_domains(path)
    assert spec["domains"] == {
        "soma": [0, 1, 2],
        "cell": {"rois": [3], "pixels": "correlated", "fraction": 0.2},
        "drawn": {"rois": [3], "pixels": "mask", "mask": {"3": [5, 6, 7]}},
    }
    path.write_text(json.dumps({"domains": {"x": {"rois": [0], "pixels": "loud"}}}))
    with pytest.raises(ValueError, match="pixels must be one of"):
        read_domains(path)


def test_a_single_patch_runs_with_a_mask_a_background_and_no_denoising(patch):
    """The run the Run tab makes for a one-patch chessboard: the cell's correlated pixels, the
    rest of the patch as background, inverted by pml_utilities, stopped after the z-score.
    The results zarr holds each domain's F, dF/F and z and no denoised trace; the cell's dips
    point up in dF/F and z.
    """
    from mbo_utilities.results import Results, pipeline_files
    from mbo_utilities.vnoiser.params import VnoiserSettings
    from mbo_utilities.vnoiser.pipeline import run_vnoiser_pipeline

    path, *_ = patch
    settings = VnoiserSettings.recommended()
    settings.traces.background = "outside"
    settings.polarity = "invert_f"
    settings.runtime.denoise = False
    settings.dfof.method = "percentile"
    settings.dfof.window_s = 1.0
    paths = run_vnoiser_pipeline(
        path,
        domains={
            "cell": {"rois": [0], "pixels": "correlated", "fraction": 36 / 400},
            "patch": [0],
        },
        settings=settings,
    )
    run = next(p for k, p in paths.items() if k.endswith(".zarr"))
    results = Results.read(run)
    scan = results["scan4"]
    assert scan.roi_names == ["cell", "patch"]
    assert set(scan.traces) == {"raw", "dff", "zscore"} and not scan.events
    assert results.provenance["polarity"] == "invert_f"
    assert results.provenance["traces"]["background"] == "outside"
    assert results.source["domains"]["cell"]["pixels"] == "correlated"
    cell = scan.traces["zscore"][0]
    assert cell[1500] > 4 and scan.traces["dff"][0][1500] > 0.05
    own = pipeline_files(run) / "traces"
    masks = json.loads((own / "scan4_masks.json").read_text())
    assert (
        masks["shapes"] == {"0": [20, 20]} and len(masks["domains"]["cell"]["0"]) == 36
    )
    assert (own / "scan4_background.npy").is_file() and (
        own / "scan4_images.npz"
    ).is_file()
    assert not (own / "scan4_denoised.npy").exists()
    # the raw kind is the measured trace, never inverted: the dip stays a dip there
    raw = scan.traces["raw"][0]
    assert raw[1500] < np.median(raw)
    with pytest.raises(ValueError, match="results zarr"):
        settings.runtime.output_format = "pkl"
        run_vnoiser_pipeline(path, domains={"patch": [0]}, settings=settings)


def test_a_single_patch_denoises_and_finds_its_spikes(patch):
    from mbo_utilities.results import Results
    from mbo_utilities.vnoiser.params import VnoiserSettings
    from mbo_utilities.vnoiser.pipeline import run_vnoiser_pipeline

    path, *_ = patch
    settings = VnoiserSettings.recommended()
    paths = run_vnoiser_pipeline(
        path,
        domains={"cell": {"rois": [0], "pixels": "bright", "fraction": 36 / 400}},
        units=["MUnit_4"],
        settings=settings,
    )
    scan = Results.read(next(p for k, p in paths.items() if k.endswith(".zarr")))[
        "scan4"
    ]
    assert {"raw", "dff", "zscore", "denoised"} <= set(scan.traces)
    found = scan.events["cell"]
    for centre in SPIKES:
        assert any(abs(int(p) - centre) <= 5 for p in found), (centre, found)
    # the conversion offset was applied: the raw trace is photons over the detector's zero
    assert np.median(scan.traces["raw"][0]) == pytest.approx(1800 + OFFSET, rel=0.05)


def test_cli_runs_a_single_patch_without_a_domains_file(patch):
    """`mbo vnoiser` on a file with no domains.json: one domain per ROI, the trace and dF/F
    options from the command line, the run recorded with them.
    """
    from click.testing import CliRunner
    from mbo_utilities.cli import main
    from mbo_utilities.results import Results, newest_results

    path, *_ = patch
    result = CliRunner().invoke(
        main,
        [
            "vnoiser",
            str(path),
            "--recommended",
            "--pixels",
            "bright",
            "--fraction",
            "0.09",
            "--background",
            "dim",
            "--polarity",
            "invert_f",
            "--dfof",
            "window",
            "--f0",
            "0,0.5",
            "--no-denoise",
            "--timepoints",
            "1:5000",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "one domain per ROI (1)" in result.output
    results = Results.read(newest_results(path.parent, "vnoiser"))
    scan = results["scan4"]
    assert scan.roi_names == ["roi0"] and scan.n_timepoints == 5000
    assert results.source["domains"]["roi0"] == {
        "rois": [0],
        "pixels": "bright",
        "fraction": 0.09,
    }
    assert results.provenance["polarity"] == "invert_f"
    assert results.settings["dfof"]["method"] == "window"
    assert results.provenance["traces"]["background"] == "dim"
    assert scan.traces["zscore"][0][1500] > 3
    bad = CliRunner().invoke(main, ["vnoiser", str(path), "--no-denoise", "--pkl"])
    assert bad.exit_code != 0 and "results zarr" in bad.output
