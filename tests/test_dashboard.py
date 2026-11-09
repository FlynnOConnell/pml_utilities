"""The dashboard: a ``.mesc`` unit's ROI traces read once and cached, shown
as raw, dF/F or z-score through a filter, drawn at any zoom from a min/max
pyramid, and exported as slide figures. No window is opened.
"""

from __future__ import annotations

import json
import time

import h5py
import numpy as np
import pytest
import scipy.ndimage
from mbo_utilities.analysis.filters import (
    FilterSettings,
    apply_filter,
    describe,
    gaussian,
)
from mbo_utilities.analysis.mesc_placement import StackPlacement
from mbo_utilities.analysis.roi_traces import roi_geometry
from mbo_utilities.analysis.trace_readouts import (
    correlation_matrix,
    detect_events,
    noise_floor_hz,
    power_spectrum,
    triggered_average,
)
from mbo_utilities.annotation import RoiTrace
from mbo_utilities.annotation.display import (
    DffSettings,
    computed_zscore,
    display_trace,
)
from mbo_utilities.annotation.domains import DomainFile, Layout, domains_path
from mbo_utilities.annotation.ngff import RoiFile
from mbo_utilities.annotation.store import RoiLabelStore
from mbo_utilities.arrays.features import MotionCorrection
from mbo_utilities.dashboard import (
    DOMAIN_SOURCE,
    FigureOutline,
    FigureSettings,
    FigureTrace,
    MinMaxPyramid,
    ReadoutSettings,
    TraceView,
    _envelope,
    bar_label,
    compute_readouts,
    cut,
    export_figure,
    export_readouts,
    load_rows,
    morphology_figure,
    nice_bar,
    process,
    row_times,
    trace_figure,
    trace_units,
)
from mbo_utilities.gui.dashboard_app import DashboardApp, write_exports
from mbo_utilities.gui.dashboard_morphology import (
    best_stack,
    grey_rgba,
    load_picture,
    load_stacks,
    slice_range,
    um_per_px,
    view_of,
    write_morphology,
)
from mbo_utilities.gui.mesc_reference import ReferenceImage, polyline_distance
from mbo_utilities.reader import imread
from mbo_utilities.roi_workflow import labels_path

from tests.test_mesc_placement import (
    N_LINES,
    UM_PER_PX,
    open_scan,
    tissue,
    write_scan,
    write_stack,
)
from tests.test_mesc_placement import STACK_CORNER as PLACEMENT_CORNER
from tests.test_results_array import N_ROIS, write_mesc

FS = 100.0
SCALE, OFFSET = 2.0, -10.0


def line_rows(n_rows=3, n=2000):
    rng = np.random.default_rng(0)
    return [
        RoiTrace(
            uid=0,
            member=k,
            source="MUnit_3 lines",
            label=f"ROI {k}",
            fs=FS,
            F=(200 + 5 * rng.standard_normal(n)).astype(np.float32),
            extra={"unit": "photons"},
        )
        for k in range(n_rows)
    ]


def figure_traces(rows, kind):
    return [FigureTrace(r.name, *process(r, TraceView(kind=kind))[:2]) for r in rows]


def mesc_with_conversion(tmp_path, munits=(35, 38), frames=40):
    """``write_mesc``'s line scans with a ``raw * 2 - 10`` photon conversion."""
    path = write_mesc(tmp_path / "rec.mesc", munits=munits, frames=frames)
    with h5py.File(path, "a") as f:
        for munit in munits:
            u = f[f"MSession_0/MUnit_{munit}"]
            u.attrs["Channel_0_Conversion_ConversionLinearScale"] = SCALE
            u.attrs["Channel_0_Conversion_ConversionLinearOffset"] = OFFSET
    return path


def wait(app, timeout=30.0):
    """Poll ``app`` until its threads are done."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app._poll()
        if app.rows:
            app._schedule()
            app._schedule_readouts()
        if not (
            app._listing
            or app._loading
            or app._pending
            or app._exporting
            or app._morph_picture
            or app._morph_stacks
            or app._reading
        ):
            return
        time.sleep(0.01)
    raise TimeoutError("dashboard threads did not finish")


@pytest.fixture
def mbo_dir(tmp_path, monkeypatch):
    """A fresh ``~/.mbo`` so the cache and the ROI folder start empty."""
    monkeypatch.setenv("MBO_DIR", str(tmp_path / "mbo"))
    return tmp_path / "mbo"


def test_nice_bar_is_one_two_or_five_times_a_power_of_ten():
    assert nice_bar(30) == 5.0
    assert nice_bar(30, fraction=0.4) == 10.0
    assert nice_bar(0.7) == 0.1
    assert nice_bar(0) == 1.0


def test_bar_label_keeps_percent_on_the_number():
    assert bar_label(20, "% dF/F") == "20% dF/F"
    assert bar_label(2, "SD") == "2 SD"
    assert bar_label(0.5, "s") == "0.5 s"


def test_cut_keeps_the_window_and_the_underlay():
    tr = FigureTrace("a", np.arange(10.0), np.arange(10.0), -np.arange(10.0), "red")
    (out,) = cut([tr], 2.0, 5.0)
    np.testing.assert_array_equal(out.t, [2, 3, 4, 5])
    np.testing.assert_array_equal(out.under, [-2, -3, -4, -5])
    assert out.color == "red"


def test_envelope_keeps_each_bins_extremes_in_time_order():
    t = np.arange(1003.0)
    y = np.sin(t / 7.0)
    y[503] = 9.0
    y[1001] = -9.0
    te, ye = _envelope(t, y, 50)
    assert len(ye) <= 100
    assert ye.max() == 9.0 and ye.min() == -9.0
    assert np.all(np.diff(te) >= 0)


def test_gaussian_matches_scipy_past_the_fft_switch():
    x = np.random.default_rng(1).standard_normal((2, 5000))
    for sigma in (3.0, 200.0):
        np.testing.assert_allclose(
            gaussian(x, sigma, axis=1),
            scipy.ndimage.gaussian_filter1d(x, sigma, axis=1),
            atol=1e-12,
        )


def test_lowpass_keeps_the_slow_sine_and_drops_the_fast_one():
    fs = 1000.0
    t = np.arange(4000) / fs
    slow, fast = np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 200 * t)
    out = apply_filter(slow + fast, fs, FilterSettings("lowpass", high_hz=50.0))
    np.testing.assert_allclose(out[500:-500], slow[500:-500], atol=1e-3)
    assert len(out) == len(t)


def test_a_band_edge_past_nyquist_raises():
    with pytest.raises(ValueError, match="between 0 and 50"):
        apply_filter(np.zeros(100), 100.0, FilterSettings("lowpass", high_hz=80.0))
    with pytest.raises(ValueError, match="low < high"):
        apply_filter(
            np.zeros(100), 100.0, FilterSettings("bandpass", low_hz=20, high_hz=10)
        )
    with pytest.raises(ValueError, match="filter must be one of"):
        FilterSettings("wiener")


def test_describe_names_the_filter():
    assert describe(FilterSettings("bandpass", low_hz=1, high_hz=100)) == (
        "band-pass 1-100 Hz"
    )
    assert describe(FilterSettings("gaussian", width_ms=2)) == "2 ms Gaussian"


def test_zscore_matches_the_direct_gaussian():
    dff = np.random.default_rng(2).standard_normal(20000).cumsum()
    settings = DffSettings(slow_s=2.0)
    residual = dff - scipy.ndimage.gaussian_filter1d(dff, settings.slow_s * FS)
    expected = residual / residual.std()
    np.testing.assert_allclose(
        computed_zscore(dff, FS, settings), expected, rtol=1e-5, atol=1e-5
    )


def test_pyramid_view_keeps_the_extremes_and_covers_the_span():
    y = np.sin(np.arange(100_000) / 50.0).astype(np.float32)
    y[61_234] = 7.0
    pyramid = MinMaxPyramid(0.0, 0.001, y)
    x, v = pyramid.view(0.0, 100.0, 500)
    assert len(v) <= 4 * 500
    assert v.max() == 7.0 and v.min() == pytest.approx(-1.0, abs=1e-3)
    assert x[0] <= 100 / 500 and x[-1] >= 100 - 100 / 500
    x, v = pyramid.view(10.0, 10.1, 500)
    np.testing.assert_array_equal(v, y[int(x[0] * 1000) : int(x[0] * 1000) + len(v)])


def test_process_puts_a_windowed_row_where_it_was_recorded():
    row = line_rows(1)[0]
    row.frames = (500, 2500, 2)
    t, shown, raw = process(row, TraceView(kind="raw"))
    assert t[0] == pytest.approx(5.0)
    assert t[1] - t[0] == pytest.approx(2 / FS)
    assert shown is raw
    np.testing.assert_array_equal(row_times(row, 3), t[:3])


def test_process_filters_after_the_kind_and_keeps_the_unfiltered():
    row = line_rows(1)[0]
    view = TraceView(kind="dff", filter=FilterSettings("gaussian", width_ms=50))
    _, shown, raw = process(row, view)
    np.testing.assert_allclose(raw, display_trace(row, "dff"), rtol=1e-6)
    assert shown.std() < raw.std()


def test_trace_figure_refuses_an_empty_window():
    with pytest.raises(ValueError, match="no samples"):
        trace_figure(
            figure_traces(line_rows(1), "raw"),
            FigureSettings(start_s=100.0, stop_s=200.0),
        )


def test_export_writes_every_format_with_editable_text(tmp_path):
    traces = figure_traces(line_rows(), "zscore")
    traces[0].under = traces[0].y * 2
    settings = FigureSettings(
        start_s=2.0, stop_s=6.0, y_unit="SD", dpi=72, caption="grey: unfiltered"
    )
    written = export_figure(traces, settings, tmp_path / "fig", info={"kind": "zscore"})
    assert sorted(p.suffix for p in written) == [
        ".csv",
        ".json",
        ".pdf",
        ".png",
        ".svg",
    ]
    svg = (tmp_path / "fig.svg").read_text()
    assert "<text" in svg and "SD</text>" in svg and "grey: unfiltered" in svg
    assert "Nimbus Sans" not in svg
    header, first = (tmp_path / "fig.csv").read_text().splitlines()[:2]
    assert header == "time_s,ROI 0,ROI 1,ROI 2"
    assert float(first.split(",")[0]) == pytest.approx(2.0)
    meta = json.loads((tmp_path / "fig.json").read_text())
    assert meta["kind"] == "zscore" and meta["settings"]["stop_s"] == 6.0


def test_export_writes_long_csv_when_rows_have_different_clocks(tmp_path):
    rows = line_rows(2)
    rows[1].fs = 2 * FS
    export_figure(
        figure_traces(rows, "raw"),
        FigureSettings(dpi=72),
        tmp_path / "fig",
        formats=("csv",),
    )
    assert (tmp_path / "fig.csv").read_text().splitlines()[0] == "label,time_s,value"


def test_trace_units_lists_only_units_with_rois_over_time(tmp_path):
    path = write_mesc(tmp_path / "rec.mesc", munits=(35, 38), frames=8)
    with h5py.File(path, "a") as f:
        del f["MSession_0/MUnit_38/Channel_0"]
        f["MSession_0/MUnit_38"].create_dataset(
            "Channel_0", data=np.zeros((1, 2, N_ROIS * 4), np.uint16)
        )
    assert [u["key"] for u in trace_units(path)] == ["MSession_0/MUnit_35"]


def test_load_rows_reads_lines_and_drawn_rois_in_photons(tmp_path, mbo_dir):
    path = mesc_with_conversion(tmp_path)
    arr = imread(path, unit="MSession_0/MUnit_35")
    store = RoiLabelStore(N_ROIS, arr.ny, arr.nx)
    store.plane_axes = (("ROI", N_ROIS),)
    store.add_roi(2, np.ones((arr.ny, arr.nx), bool))
    store.rois[0].note = "soma"
    RoiFile(labels_path(path, "MSession_0_MUnit_35")).save(store)

    rows = load_rows(path, "MSession_0/MUnit_35")
    assert [r.name for r in rows][:2] == ["ROI 0", "ROI 1"]
    assert len(rows) == N_ROIS + 1
    drawn = rows[-1]
    assert (drawn.source, drawn.name, drawn.z) == ("drawn", "soma", 2)
    raw = np.asarray(arr[:, 0, 2], np.float64).reshape(arr.nt, -1).mean(1)
    np.testing.assert_allclose(rows[2].F, raw * SCALE + OFFSET, rtol=1e-5)
    np.testing.assert_allclose(drawn.F, rows[2].F, rtol=1e-5)
    assert rows[0].fs == pytest.approx(1000.0)
    assert len(list((mbo_dir / "cache" / "dashboard").glob("rec_*.npz"))) == 1

    again = load_rows(path, "MSession_0/MUnit_35")
    np.testing.assert_array_equal(again[-1].F, drawn.F)
    assert len(list((mbo_dir / "cache" / "dashboard").glob("rec_*.npz"))) == 1


def test_load_rows_adds_the_layouts_whole_roi_domains(tmp_path, mbo_dir):
    """A domain row is its lines' pixel-weighted mean; a domain that picks
    pixels inside its lines is left out; the app shows the scan's block, and
    picking a domain colours its lines on the tissue.
    """
    path = mesc_with_conversion(tmp_path)
    DomainFile(
        [
            Layout(
                ["35", "38"],
                {
                    "pair": [0, 1],
                    "cell": {"rois": [2], "pixels": "bright", "fraction": 0.3},
                },
            )
        ],
        first_env=["38"],
        scan_labels={"38": ["Odor on"]},
    ).write(domains_path(path))

    rows = load_rows(path, "MSession_0/MUnit_35")
    domains = [r for r in rows if r.source == DOMAIN_SOURCE]
    assert [(r.name, r.extra["lines"]) for r in domains] == [("pair", [0, 1])]
    sizes = {
        k: h * w
        for k, (h, w) in roi_geometry(imread(path, unit="MSession_0/MUnit_35")).items()
    }
    weighted = (sizes[0] * rows[0].F + sizes[1] * rows[1].F) / (sizes[0] + sizes[1])
    np.testing.assert_allclose(domains[0].F, weighted, rtol=1e-6)

    app = DashboardApp(path, unit="MUnit_38")
    wait(app)
    assert app.block == "block from MUnit_38: Odor on"
    pair = next(r for r in app.rows if r.source == DOMAIN_SOURCE)
    app.picked = {pair.key}
    assert set(app._line_owners()) == {0, 1}
    app.choose("MSession_0/MUnit_35")
    wait(app)
    assert app.block == ""


def test_app_opens_a_unit_switches_kind_and_exports(tmp_path, mbo_dir):
    path = mesc_with_conversion(tmp_path)
    app = DashboardApp(path, unit="MUnit_38")
    wait(app)
    assert app.unit == "MSession_0/MUnit_38"
    assert len(app.rows) == N_ROIS and len(app.picked) == 4
    assert app.folder == str(tmp_path / "figures")

    app._schedule()
    wait(app)
    assert {r.view for r in app._results.values()} == {repr(app.view)}
    app.view.kind = "zscore"
    app.view.filter = FilterSettings("gaussian", width_ms=5)
    app._schedule()
    wait(app)
    assert app.process_error == ""
    assert all(r.view == repr(app.view) for r in app._results.values())
    assert all(r.under is not None for r in app._results.values())

    assert app.stem("view", 0.0, 0.02) == "rec_MUnit_38_zscore_0-0.02s"
    jobs = app.exports([("view", 0.0, 0.02), ("burst", 0.01, 0.03)])
    _, settings, stem, info = jobs[0]
    assert settings.y_unit == "SD" and "5 ms Gaussian" in settings.caption
    assert info["unit"] == "MSession_0/MUnit_38" and len(info["rows"]) == 4
    for job in jobs:
        job[1].dpi = 72
    assert write_exports(jobs).startswith("wrote 10 files")
    assert (tmp_path / "figures" / "rec_MUnit_38_zscore_burst.png").exists()


def test_app_reports_a_unit_without_rois(tmp_path, mbo_dir):
    app = DashboardApp(mesc_with_conversion(tmp_path), unit="MUnit_99")
    wait(app)
    assert "MUnit_99 has no ROIs" in app.error and app.unit is None
    assert len(app.units) == 2


def test_app_shows_a_filter_error_instead_of_raising(tmp_path, mbo_dir):
    app = DashboardApp(mesc_with_conversion(tmp_path), unit="MUnit_35")
    wait(app)
    app.view.filter = FilterSettings("lowpass", high_hz=900.0)
    app._schedule()
    wait(app)
    assert "between 0 and 500" in app.process_error


def test_power_spectrum_puts_a_sine_in_its_bin():
    fs = 1000.0
    t = np.arange(20000) / fs
    frequency, power = power_spectrum(np.sin(2 * np.pi * 50 * t)[None], fs)
    assert frequency[np.argmax(power[0])] == pytest.approx(50.0)
    assert frequency[1] - frequency[0] == pytest.approx(1.0)


def test_noise_floor_is_where_a_slow_signal_sinks_into_white_noise():
    rng = np.random.default_rng(0)
    fs = 1000.0
    slow = apply_filter(
        rng.normal(size=200_000), fs, FilterSettings("lowpass", high_hz=30.0, order=8)
    )
    y = 6 * slow + rng.normal(size=200_000)
    frequency, power = power_spectrum(y, fs)
    assert 25.0 < noise_floor_hz(frequency, power) < 50.0


def test_correlation_matrix_gives_a_flat_trace_no_correlation():
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    r = correlation_matrix(np.stack([a, -a, np.full(500, 3.0)]))
    np.testing.assert_allclose(r[:2, :2], [[1, -1], [-1, 1]], atol=1e-12)
    np.testing.assert_array_equal(r[2], 0.0)


def test_detect_events_finds_peaks_or_troughs_a_gap_apart():
    rng = np.random.default_rng(2)
    y = rng.normal(size=10_000)
    planted = np.array([1000, 3000, 3002, 7000])
    y[planted] += 20.0
    found = detect_events(y, 1000.0, threshold_sd=8.0, gap_ms=10.0)
    assert len(found) == 3 and set(found) <= set(planted)
    np.testing.assert_array_equal(
        detect_events(-y, 1000.0, 8.0, 10.0, polarity=-1), found
    )
    assert len(detect_events(np.zeros(100), 1000.0)) == 0
    with pytest.raises(ValueError, match="polarity"):
        detect_events(y, 1000.0, polarity=0)


def test_triggered_average_recovers_a_waveform_and_leaves_out_the_edges():
    y = np.zeros((2, 1000))
    events = np.array([3, 200, 500, 800, 997])
    for e in events[1:-1]:
        y[0, e : e + 3] = [5.0, 3.0, 1.0]
    mean, sem, n = triggered_average(y, events, before=10, after=10)
    assert n == 3 and mean.shape == sem.shape == (2, 21)
    np.testing.assert_allclose(mean[0, 10:13], [5.0, 3.0, 1.0])
    np.testing.assert_array_equal(mean[1], 0.0)
    assert sem[0, 10] == 0.0


def test_triggered_average_with_no_event_left_is_empty():
    mean, sem, n = triggered_average(np.ones((1, 50)), np.array([1]), 10, 10)
    assert n == 0 and mean.shape == sem.shape == (1, 21)


def rtmc_rows(n=4000, fs=1000.0):
    """Two rows and an RTMC whose X follows the first row's slow wave."""
    t = np.arange(n) / fs
    wave = np.sin(2 * np.pi * 3 * t)
    noise = np.random.default_rng(3).normal(size=n)
    motion = MotionCorrection("RTMC", "um", {"X": (t, wave), "Y": (t[:2], np.zeros(2))})
    return t, np.stack([wave + 0.1 * noise, noise]), fs, motion


def test_compute_readouts_measures_rows_and_motion_in_one_band():
    t, traces, fs, motion = rtmc_rows()
    band = FilterSettings("lowpass", high_hz=10.0)
    filtered = apply_filter(traces, fs, band)
    r = compute_readouts(
        ["a", "b"], t, filtered, fs, 0, (0.0, 4.0), ReadoutSettings(), motion, band
    )
    assert r.correlation_labels == ["a", "b", "RTMC X", "RTMC Y"]
    assert r.correlation[0, 2] > 0.99
    np.testing.assert_array_equal(r.correlation[3], 0.0)
    assert r.power.shape[0] == 2 and len(r.floors) == 2
    assert r.lags_ms[0] == pytest.approx(-20.0)
    assert r.mean.shape == r.sem.shape == (2, len(r.lags_ms))


def test_compute_readouts_keeps_to_the_span_and_refuses_an_empty_one():
    t, traces, fs, _ = rtmc_rows()
    traces[0, 500] = 50.0
    traces[0, 3000] = 50.0
    r = compute_readouts(["a", "b"], t, traces, fs, 0, (2.0, 4.0), ReadoutSettings())
    assert r.correlation_labels == ["a", "b"]
    np.testing.assert_allclose(r.event_times, [3.0])
    with pytest.raises(ValueError, match="fewer than two samples"):
        compute_readouts(["a", "b"], t, traces, fs, 0, (9.0, 10.0), ReadoutSettings())


def test_export_readouts_writes_a_csv_each_that_reads_back(tmp_path):
    t, traces, fs, motion = rtmc_rows()
    r = compute_readouts(
        ["a", "b"], t, traces, fs, 0, (0.0, 4.0), ReadoutSettings(), motion
    )
    written = export_readouts(r, tmp_path / "out" / "rec")
    assert sorted(p.name for p in written) == [
        "rec_correlation.csv",
        "rec_events.csv",
        "rec_spectrum.csv",
        "rec_triggered.csv",
    ]
    spectrum = np.loadtxt(
        tmp_path / "out" / "rec_spectrum.csv", delimiter=",", skiprows=1
    )
    assert spectrum.shape == (len(r.frequency), 3)
    header = (tmp_path / "out" / "rec_correlation.csv").read_text().splitlines()[0]
    assert header == "a,b,RTMC X,RTMC Y"


def test_polyline_distance_measures_to_the_nearest_segment():
    vertices = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert polyline_distance(np.array([5.0, 3.0]), vertices) == pytest.approx(3.0)
    assert polyline_distance(np.array([13.0, 10.0]), vertices) == pytest.approx(3.0)
    assert polyline_distance(np.array([-4.0, -3.0]), vertices) == pytest.approx(5.0)


def test_grey_rgba_is_opaque_and_stretched():
    rgba = grey_rgba(np.linspace(0.0, 1.0, 100 * 100).reshape(100, 100))
    assert rgba.shape == (100, 100, 4) and rgba.dtype == np.uint8
    assert rgba[..., 3].min() == 255
    assert rgba[..., 0].min() == 0 and rgba[..., 0].max() == 255


@pytest.fixture(scope="module")
def placed_scan(tmp_path_factory):
    """A line scan with its picture and one stack it can be placed on."""
    path = tmp_path_factory.mktemp("morphology") / "scan.mesc"
    volume = tissue()
    with h5py.File(path, "w") as f:
        write_stack(f.create_group("MSession_0"), volume, PLACEMENT_CORNER)
    write_scan(path, volume)
    return path


def test_the_picture_comes_first_and_the_stack_carries_the_placement(placed_scan):
    arr, picture, _, domains = load_picture(placed_scan, "MSession_0/MUnit_1", 0)
    assert picture is not None and picture.placement is None
    assert len(picture.records) == N_LINES and domains == {}
    assert um_per_px(arr, picture) == pytest.approx(UM_PER_PX)
    assert slice_range(picture, None, 3) is None

    arr = open_scan(placed_scan)
    images, refused = load_stacks(arr, 0, ())
    assert len(images) == 1 and refused == []
    stack = best_stack(images)
    assert stack.placement.placed and len(stack.records) == N_LINES
    assert um_per_px(arr, stack) == pytest.approx(UM_PER_PX)
    lo, hi = slice_range(stack, None, 0)
    assert slice_range(stack, None, 100) == (0, stack.stack.shape[2] - 1)
    assert slice_range(stack, {0}, 0)[0] >= lo and slice_range(stack, {99}, 0) is None
    col0, col1, row0, row1 = view_of(stack)
    points = np.concatenate([r["pixels"] for r in stack.records])
    assert col0 < points[:, 0].min() and points[:, 0].max() < col1
    assert row0 < points[:, 1].min() and points[:, 1].max() < row1


def test_best_stack_prefers_a_confirmed_placement_then_more_rois():
    stack = {"stack_path": "s", "stack_key": "k", "shift_um": (0, 0, 0), "gap_s": 0.0}
    weak = StackPlacement(**stack, score=0.9, profile_r=0.9, profile_p=0.5)
    sure = StackPlacement(**stack, score=0.7, profile_r=0.5, profile_p=0.01)
    a = ReferenceImage("a", np.zeros((2, 2)), [{}] * 9, weak)
    b = ReferenceImage("b", np.zeros((2, 2)), [{}] * 2, sure)
    c = ReferenceImage("c", np.zeros((2, 2)), [{}] * 3, sure)
    assert best_stack([a, b, c]).key == "c"
    assert best_stack([]) is None


def test_app_opens_on_the_placed_stack_and_asks_for_the_slices_set(placed_scan):
    app = DashboardApp()
    app.path, app.unit, app.channel = placed_scan, "MSession_0/MUnit_1", 0
    app._reset_morphology()
    wait(app)
    assert app.morph_note == "" and len(app.morph_images) == 2
    image = app.morph_images[app.morph_key]
    assert image.placement is not None
    assert app._morph_scale == pytest.approx(UM_PER_PX)
    assert app._morph_shown() == (image.key, None)
    app.morph_pad = 1
    lo, hi = slice_range(image, None, 1)
    assert app._morph_shown() == (image.key, (lo, hi))


def test_morphology_exports_the_figure_and_the_positions(placed_scan, tmp_path):
    arr = open_scan(placed_scan)
    images, _ = load_stacks(arr, 0, ())
    stack = images[0]
    outlines = [FigureOutline("ROI 0", stack.records[0]["pixels"], "#ff8c3a")]
    settings = FigureSettings(dpi=72)
    status = write_morphology(
        arr,
        {stack.key: stack},
        stack,
        stack.image,
        outlines,
        view_of(stack),
        settings,
        tmp_path / "rec_morphology",
        {"image": stack.key},
        {},
    )
    assert status.startswith("wrote 5 files")
    assert (tmp_path / "rec_morphology.svg").read_text().count("ROI 0") == 1
    table = (tmp_path / "rec_morphology_positions.csv").read_text().splitlines()
    assert table[0].startswith("unit,roi,domain") and len(table) == N_LINES + 1


def test_morphology_figure_crops_to_the_view_with_rows_growing_up():
    image = np.zeros((50, 80))
    fig = morphology_figure(image, [], 0.5, (10, 40, 5, 25), FigureSettings())
    ax = fig.axes[0]
    assert ax.get_xlim() == (10, 40) and ax.get_ylim() == (5, 25)
    assert ax.images[0].origin == "lower"
    assert fig.get_size_inches()[1] / fig.get_size_inches()[0] == pytest.approx(20 / 30)


def test_app_says_when_a_unit_has_no_picture(tmp_path, mbo_dir):
    app = DashboardApp(mesc_with_conversion(tmp_path), unit="MUnit_35")
    wait(app)
    assert app.morph_images == {} and "no ROIs on a picture" in app.morph_note


def test_app_reads_out_the_plotted_rows_over_the_active_window(tmp_path, mbo_dir):
    app = DashboardApp(mesc_with_conversion(tmp_path, frames=400), unit="MUnit_35")
    wait(app)
    app.page = "Spectrum"
    app.readout.skip_s = 0.1
    app._schedule()
    wait(app)
    assert app.readout_error == "" and app.readouts is not None
    assert app.readouts.labels == [r.name for r in app._picked_rows()]
    assert app.readouts.span == pytest.approx((0.1, app.duration[1]))

    app.add_window(0.05, 0.2)
    app._schedule_readouts()
    wait(app)
    assert app.readouts.span == pytest.approx((0.05, 0.2))

    app.trigger = app._picked_rows()[1].key
    app._schedule_readouts()
    wait(app)
    assert app.readouts.trigger == 1

    app.choose(app.unit)
    assert app.readouts is None and app._reading is None
