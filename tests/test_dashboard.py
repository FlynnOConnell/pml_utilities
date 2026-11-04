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
from mbo_utilities.annotation import RoiTrace
from mbo_utilities.annotation.display import (
    DffSettings,
    computed_zscore,
    display_trace,
)
from mbo_utilities.annotation.ngff import RoiFile
from mbo_utilities.annotation.store import RoiLabelStore
from mbo_utilities.dashboard import (
    FigureSettings,
    FigureTrace,
    MinMaxPyramid,
    TraceView,
    _envelope,
    bar_label,
    cut,
    export_figure,
    load_rows,
    nice_bar,
    process,
    row_times,
    trace_figure,
    trace_units,
)
from mbo_utilities.gui.dashboard_app import DashboardApp, write_exports
from mbo_utilities.reader import imread
from mbo_utilities.roi_workflow import labels_path

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
    """Poll ``app`` until its loader and worker are idle."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app._poll()
        busy = app._listing or app._loading or app._pending or app._exporting
        if not busy:
            return
        time.sleep(0.01)
    raise TimeoutError("the dashboard's threads did not finish")


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
