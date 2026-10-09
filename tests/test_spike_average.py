"""Spike-triggered averages: the windows, where the spikes come from, one ROI
of a run's results over its movie, and the window that shows them, drawn
offscreen.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
from mbo_utilities.analysis.spike_average import (
    EVENTS,
    MEAN_F,
    THRESHOLD,
    SpikeSource,
    accepted_events,
    threshold_peaks,
    triggered_average,
)
from mbo_utilities.results import (
    CURATION_DIR,
    Results,
    ResultUnit,
    recording_id,
    results_name,
)

from tests.test_manual_roi import _offscreen_selected

WAVE = np.array([0.0, 1.0, 3.0, 1.0, 0.0])
SPIKES = np.arange(100, 4900, 97)
T, Y, X = 600, 12, 16
FLASHES = np.arange(40, 560, 40)
# the run read the recording from this frame on
FIRST = 20


@pytest.fixture
def run(tmp_path):
    """A tif recording whose top-left 4x4 brightens at ``FLASHES``, and a
    results file of two ROIs over frames ``FIRST`` on: roi0's dff peaks at
    the flashes, its events every other flash; roi1 is noise.
    """
    rng = np.random.default_rng(0)
    movie = (100 + rng.normal(0, 2, (T, Y, X))).astype(np.float32)
    movie[FLASHES, :4, :4] += 50
    tifffile.imwrite(tmp_path / "rec.tif", movie)
    n = T - FIRST
    dff = rng.normal(0, 0.05, (2, n)).astype(np.float32)
    dff[0, FLASHES - FIRST] += 3.0
    unit = ResultUnit(
        name="zplane01",
        kind="plane",
        index=1,
        fs=30.0,
        roi_names=["roi0", "roi1"],
        traces={"dff": dff, "raw": 100 + dff},
        members=[np.arange(4), np.arange(100, 104)],
        image_shape=(Y, X),
        events={"roi0": FLASHES[::2] - FIRST},
        attrs={"z": 0},
    )
    path = Results(
        pipeline="test",
        units={unit.name: unit},
        source={"path": str(tmp_path / "rec.tif"), "frames": [FIRST, T]},
    ).write(tmp_path / results_name("rec.tif", pipeline="test"))
    curated = {
        f"a|{s}": {
            "label": "yes",
            "recording": recording_id(unit, "roi0"),
            "source_aligned_index": int(s),
        }
        for s in FLASHES[1:4] - FIRST
    }
    (path / CURATION_DIR).mkdir()
    (path / CURATION_DIR / "fast_template_curation.json").write_text(
        json.dumps({"version": 5, "events": curated}), encoding="utf-8"
    )
    return path, movie


def test_average_recovers_the_waveform_under_every_spike():
    trace = np.random.default_rng(0).normal(0, 0.01, 5000)
    trace[SPIKES[:, None] + np.arange(-2, 3)] += WAVE
    mean, sem, n = triggered_average(trace, SPIKES, before=4, after=4)
    assert n == len(SPIKES)
    assert mean.shape == sem.shape == (9,)
    assert mean.argmax() == 4
    np.testing.assert_allclose(mean[2:7], WAVE, atol=0.01)


def test_windows_past_either_end_are_left_out():
    mean, _sem, n = triggered_average(np.arange(50.0), [2, 25, 48], before=5, after=5)
    assert n == 1
    np.testing.assert_array_equal(mean, np.arange(20.0, 31.0))


def test_no_spike_with_a_whole_window_raises():
    with pytest.raises(ValueError, match="none of 1 spikes"):
        triggered_average(np.zeros(10), [1], before=5, after=5)


def test_centering_takes_the_drift_between_windows_out_of_the_error():
    drift = np.linspace(0.0, 100.0, 2000)
    spikes = np.arange(50, 1950, 50)
    _mean, sem, _n = triggered_average(drift, spikes, 5, 5)
    centered, centered_sem, _n = triggered_average(drift, spikes, 5, 5, center=True)
    assert sem.min() > 1.0
    assert centered_sem.max() < 1e-6
    assert abs(centered.mean()) < 1e-9


def test_a_movie_averages_per_pixel_across_overlapping_windows_and_blocks():
    movie = np.random.default_rng(1).normal(size=(200, 3, 4))
    spikes = np.array([20, 22, 100])
    mean, _sem, n = triggered_average(movie, spikes, 3, 3, block=7)
    assert n == 3
    np.testing.assert_allclose(
        mean, np.mean([movie[s - 3 : s + 4] for s in spikes], axis=0)
    )


def test_threshold_peaks_are_the_crests_over_the_line_kept_apart():
    trace = np.zeros(100)
    trace[[10, 12, 40, 70]] = [5.0, 4.0, 2.0, 6.0]
    np.testing.assert_array_equal(threshold_peaks(trace, 3.0, distance=5), [10, 70])
    np.testing.assert_array_equal(threshold_peaks(trace, 3.0, distance=1), [10, 12, 70])
    np.testing.assert_array_equal(threshold_peaks(trace, 1.0, distance=5), [10, 40, 70])


def test_accepted_events_are_the_yes_labels_of_each_recording(tmp_path):
    events = {
        "a|sample=30": {"label": "yes", "recording": "a", "source_aligned_index": 31},
        "a|sample=10": {"label": "yes", "recording": "a", "source_aligned_index": 10},
        "a|sample=20": {"label": "no", "recording": "a", "source_aligned_index": 20},
        "b|sample=5": {"label": "yes", "recording": "b", "source_aligned_index": 5},
    }
    path = tmp_path / "fast_template_curation.json"
    path.write_text(json.dumps({"version": 5, "events": events}), encoding="utf-8")
    found = accepted_events(path)
    assert sorted(found) == ["a", "b"]
    np.testing.assert_array_equal(found["a"], [10, 31])
    np.testing.assert_array_equal(found["b"], [5])


def test_a_results_file_opens_one_roi_over_the_frames_the_run_read(run):
    path, movie = run
    source = SpikeSource.open(path)
    assert (source.unit.name, source.roi, source.c, source.fs) == (
        "zplane01",
        "roi0",
        0,
        30.0,
    )
    assert list(source.traces) == ["dff", "raw"]
    assert list(source.movies) == ["movie"]
    assert source.movies["movie"].shape == (T - FIRST, Y, X)
    np.testing.assert_allclose(source.movies["movie"][5], movie[FIRST + 5])
    assert source.motion == {}


def test_spikes_come_from_the_runs_events_and_its_curation(run):
    path, _movie = run
    source = SpikeSource.open(path)
    assert list(source.spikes) == [EVENTS, "curated fast"]
    np.testing.assert_array_equal(source.spikes[EVENTS], FLASHES[::2] - FIRST)
    np.testing.assert_array_equal(source.spikes["curated fast"], FLASHES[1:4] - FIRST)
    assert SpikeSource.open(path, roi="roi1").spikes == {}


def test_a_roi_averages_its_movie_and_trace_around_its_spikes(run):
    path, _movie = run
    source = SpikeSource.open(path)
    average = source.average(source.spikes[EVENTS], "dff", before=3, after=4)
    np.testing.assert_array_equal(average.lags, np.arange(-3, 5))
    assert average.n_spikes == len(FLASHES[::2])
    assert average.movies["movie"].shape == (8, Y, X)
    assert average.movies["movie"][:, 0, 0].argmax() == 3
    assert list(average.traces) == ["roi0 dff", MEAN_F]
    mean, sem = average.traces["roi0 dff"]["dff"]
    assert mean.argmax() == 3 and sem.shape == (8,)
    flat, none = average.traces[MEAN_F]["movie"]
    assert flat.shape == (8,) and none is None


def test_another_roi_opens_on_the_same_array(run):
    path, _movie = run
    source = SpikeSource.open(path)
    other = source.with_roi("roi1")
    assert other.arr is source.arr and other.roi == "roi1"
    with pytest.raises(KeyError, match="no ROI 'roi9'"):
        source.with_roi("roi9")


@pytest.mark.skipif(
    not _offscreen_selected(), reason="needs the offscreen rendercanvas"
)
def test_the_window_follows_its_roi_trace_and_spike_choices(run):
    from fastplotlib.widgets.nd_widget._async import run_sync
    from mbo_utilities.gui.spike_average_viewer import SpikeAverageViewer

    path, _movie = run
    viewer = SpikeAverageViewer(
        SpikeSource.open(path), before=3, after=4, size=(900, 900)
    )
    try:
        viewer.show()
        viewer.ndw.figure.canvas.draw()
        assert (viewer.kind, viewer.spikes_from) == ("dff", EVENTS)
        np.testing.assert_array_equal(viewer.spikes, FLASHES[::2] - FIRST)
        assert viewer.lag_index == 3
        viewer.seek(1)
        # the NDWidget fetches a slice on its event loop, which a test does not run
        run_sync(viewer.images["movie"]._set_indices_())
        viewer.ndw.figure.canvas.draw()
        assert viewer.lag_index == 1
        shown = viewer.average.movies["movie"]
        np.testing.assert_allclose(
            np.asarray(viewer.images["movie"].graphic.data.value),
            (shown - shown.mean(axis=0))[1],
            rtol=1e-5,
            atol=1e-5,
        )
        viewer.set_subtract_mean(False)
        assert viewer.images["movie"].graphic.vmax == pytest.approx(float(shown.max()))
        viewer.set_spikes_from("curated fast")
        assert viewer.average.n_spikes == 3
        viewer.set_spikes_from(THRESHOLD)
        np.testing.assert_array_equal(viewer.spikes, FLASHES - FIRST)
        viewer.set_threshold(100.0)
        assert "none of 0 spikes" in viewer.status
        np.testing.assert_array_equal(viewer.spikes, FLASHES - FIRST)
        viewer.set_kind("raw")
        assert viewer.status == "" and "roi0 raw" in viewer.traces.panels
        viewer.set_roi("roi1")
        viewer.ndw.figure.canvas.draw()
        assert viewer.source.roi == "roi1" and viewer.spikes_from == THRESHOLD
    finally:
        viewer.close()


@pytest.mark.slow
def test_a_masknmf_run_averages_its_registered_and_compressed_movies(tmp_path):
    pytest.importorskip("masknmf")
    from mbo_utilities.masknmf import run_plane
    from mbo_utilities.masknmf.params import STAGE_SKIP, MasknmfSettings

    rng = np.random.default_rng(0)
    movie = (200 + rng.normal(0, 5, (300, 48, 48))).astype(np.int16)
    flashes = np.arange(20, 280, 20)
    movie[flashes, 20:28, 20:28] += 300
    tifffile.imwrite(tmp_path / "movie.tif", movie)
    settings = MasknmfSettings()
    settings.registration.max_shifts = (4, 4)
    settings.compression.denoise = False
    settings.compression.block_sizes = (16, 16)
    settings.demixing.do_demixing = STAGE_SKIP
    settings.runtime.device = "cpu"
    run = run_plane(
        str(tmp_path / "movie.tif"),
        tmp_path / "out",
        settings=settings,
        frame_indices=list(range(10, 290)),
        replot=False,
    )
    # masknmf demixes on cuda only: the results file demixing writes is made here
    dff = np.zeros((1, 280), np.float32)
    dff[0, flashes - 10] = 5.0
    unit = ResultUnit(
        name="zplane01",
        kind="plane",
        index=1,
        fs=30.0,
        roi_names=["0"],
        traces={"dff": dff},
        members=[np.arange(20 * 48 + 20, 20 * 48 + 28)],
        image_shape=(48, 48),
    )
    Results(pipeline="masknmf", units={unit.name: unit}).write(
        run / results_name("movie.tif", pipeline="masknmf")
    )

    source = SpikeSource.open(run)
    assert list(source.movies) == ["registered", "compressed"]
    assert list(source.motion) == ["masknmf shift (px)"]
    spikes = threshold_peaks(source.traces["dff"], 1.0)
    np.testing.assert_array_equal(spikes, flashes - 10)
    average = source.average(spikes, "dff", 3, 3)
    assert average.n_spikes == len(flashes)
    for movie in average.movies.values():
        assert movie[:, 24, 24].argmax() == 3
    assert sorted(average.traces["masknmf shift (px)"]) == ["X", "Y"]
