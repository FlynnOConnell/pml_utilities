"""Spike-triggered averages: the windows, the spikes a threshold or a
curation file gives, and the window that shows them, drawn offscreen.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from mbo_utilities.analysis.spike_average import (
    MasknmfUnit,
    accepted_events,
    threshold_peaks,
    triggered_average,
)

from tests.test_manual_roi import _offscreen_selected

WAVE = np.array([0.0, 1.0, 3.0, 1.0, 0.0])
SPIKES = np.arange(100, 4900, 97)


def _unit() -> MasknmfUnit:
    """5000 frames of noise, two demixed traces: the first spikes at
    ``SPIKES``, the second at every other one, and the movie's top left
    pixel brightens with the first.
    """
    rng = np.random.default_rng(2)
    signals = rng.normal(0, 0.05, (5000, 2))
    signals[SPIKES[:, None] + np.arange(-2, 3), 0] += WAVE
    signals[SPIKES[::2, None] + np.arange(-2, 3), 1] += WAVE
    movie = rng.normal(0, 0.01, (5000, 6, 8)).astype(np.float32)
    movie[SPIKES, 0, 0] += 1.0
    return MasknmfUnit(
        label="synthetic",
        recording="synthetic/MUnit_0/roi=0",
        fs=1000.0,
        first_frame=0,
        movies={"registered": movie, "compressed": 2 * movie},
        traces={
            "mean F (a.u.)": {"registered": movie.mean(axis=(1, 2))},
            "RTMC shift (um)": {"X": rng.normal(size=5000), "Z": rng.normal(size=5000)},
        },
        signals=signals,
    )


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


def test_a_unit_averages_its_movies_and_traces_around_the_spikes():
    average = _unit().average(SPIKES, before=4, after=6)
    np.testing.assert_array_equal(average.lags, np.arange(-4, 7))
    assert average.n_spikes == len(SPIKES)
    assert average.movies["registered"].shape == (11, 6, 8)
    assert average.movies["registered"][:, 0, 0].argmax() == 4
    assert average.movies["compressed"][4, 0, 0] == pytest.approx(2.0, abs=0.01)
    mean, sem = average.traces["mean F (a.u.)"]["registered"]
    assert mean.shape == sem.shape == (11,)
    assert mean.argmax() == 4
    assert sorted(average.traces["RTMC shift (um)"]) == ["X", "Z"]


@pytest.mark.skipif(
    not _offscreen_selected(), reason="needs the offscreen rendercanvas"
)
def test_the_window_finds_spikes_on_a_signal_and_follows_its_controls():
    from mbo_utilities.gui.spike_average_viewer import SpikeAverageVis

    vis = SpikeAverageVis(_unit(), before=4, after=6, size=(900, 900))
    try:
        vis.show()
        vis.figure.canvas.draw()
        np.testing.assert_array_equal(vis.spikes, SPIKES)
        assert vis.player.t == 4
        vis.seek(2)
        vis.figure.canvas.draw()
        for name, movie in vis.average.movies.items():
            np.testing.assert_allclose(
                np.asarray(vis.graphics[name].data.value),
                (movie - movie.mean(axis=0))[2],
                rtol=1e-5,
                atol=1e-7,
            )
        vis.set_subtract_mean(False)
        assert vis.graphics["registered"].vmax == pytest.approx(
            float(vis.average.movies["registered"].max())
        )
        vis.set_signal(1)
        vis.figure.canvas.draw()
        np.testing.assert_array_equal(vis.spikes, SPIKES[::2])
        assert vis.average.n_spikes == len(SPIKES[::2])
        vis.set_threshold(100.0)
        vis.figure.canvas.draw()
        assert "none of 0 spikes" in vis.status
        np.testing.assert_array_equal(vis.spikes, SPIKES[::2])
        vis.set_threshold(2.0)
        assert vis.status == ""
    finally:
        vis.close()


@pytest.mark.skipif(
    not _offscreen_selected(), reason="needs the offscreen rendercanvas"
)
def test_the_window_takes_curated_spikes_in_place_of_a_threshold():
    from mbo_utilities.gui.spike_average_viewer import SpikeAverageVis

    vis = SpikeAverageVis(_unit(), spikes=SPIKES[:10], size=(900, 900))
    try:
        vis.show()
        vis.figure.canvas.draw()
        assert vis.average.n_spikes == 10
        vis.set_threshold(0.5)
        np.testing.assert_array_equal(vis.spikes, SPIKES[:10])
    finally:
        vis.close()


@pytest.mark.slow
def test_a_run_folder_averages_on_the_frames_it_read(tmp_path):
    pytest.importorskip("masknmf")
    import h5py
    import tifffile
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
    # masknmf demixes on cuda only: the demixed traces are written here
    signals = np.zeros((280, 2))
    signals[flashes - 10, 0] = 5.0
    with h5py.File(run / "results.hdf5", "a") as f:
        f["DemixingResults/temporal_demixed"] = signals

    unit = MasknmfUnit.from_run(run)
    assert unit.first_frame == 10
    assert unit.recording == "movie/roi=0"
    assert sorted(unit.traces) == ["masknmf shift (px)", "mean F (a.u.)"]
    spikes = threshold_peaks(unit.signals[:, 0], 1.0)
    np.testing.assert_array_equal(spikes, flashes - 10)
    average = unit.average(spikes, 3, 3)
    assert average.n_spikes == len(flashes)
    for movie in average.movies.values():
        assert movie[:, 24, 24].argmax() == 3
