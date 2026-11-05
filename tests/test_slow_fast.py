"""The slow / fast curation method: the detector, its session and the
curation window's panels, none of which needs vnoiser.
"""

from __future__ import annotations

import json
import logging
import traceback

import numpy as np
import pytest

from tests.test_manual_roi import _offscreen_selected

FS_HZ = 1000.0
N_SAMPLES = 20000
BUMPS_S = (1.0, 6.0, 12.0)  # the first sits in the skipped first 2 s
SPIKES_S = (5.98, 6.03, 12.01)
RID = "scan=10/domain=soma"
FIGURE_SIZE = (1200, 800)


def _synthetic_trace(seed: int = 0) -> np.ndarray:
    """20 s of events-positive dF/F: slow bumps (40 ms) with sharp spikes (1 ms) on two of them."""
    t = np.arange(N_SAMPLES) / FS_HZ
    trace = np.random.default_rng(seed).normal(0.0, 0.002, t.size)
    for center in BUMPS_S:
        trace += 0.03 * np.exp(-0.5 * ((t - center) / 0.04) ** 2)
    for center in SPIKES_S:
        trace += 0.02 * np.exp(-0.5 * ((t - center) / 0.001) ** 2)
    return trace


def _write_run(folder, dff, polarity="flip_z"):
    """A results file with one scan, one domain, its ``dff`` and ``denoised``."""
    from mbo_utilities.results import Results, ResultUnit

    unit = ResultUnit(
        name="scan10",
        kind="scan",
        index=10,
        fs=FS_HZ,
        roi_names=["soma"],
        traces={
            "dff": np.asarray(dff, dtype=np.float32)[None, :],
            "denoised": np.asarray(dff, dtype=np.float32)[None, :],
        },
        member_kind="line",
        members=[np.array([0])],
        attrs={"scan_id": 10},
    )
    path = folder / "expt1.2026-11-03-12-00-00.vnoiser.zarr"
    Results(
        pipeline="vnoiser", units={"scan10": unit}, provenance={"polarity": polarity}
    ).write(path)
    return path


@pytest.fixture
def run_path(tmp_path):
    """A flip_z run: its stored dF/F dims at events, as dfof_raw does for ASAP."""
    return _write_run(tmp_path, -_synthetic_trace())


def _session(run_path):
    from mbo_utilities.annotation.slow_fast import SlowFastSession
    from mbo_utilities.results import ResultsArray

    session = SlowFastSession()
    session.load_run(ResultsArray(run_path, source=False), "scan10", "soma")
    return session


def _near(times, wanted, tol=0.003) -> bool:
    return bool(np.any(np.abs(np.asarray(times) - wanted) <= tol))


class TestDetector:
    def test_finds_the_spikes_inside_their_slow_events(self):
        from mbo_utilities.analysis.slow_fast import (
            FAST_COLUMNS,
            SLOW_COLUMNS,
            detect_slow_fast,
        )

        events = detect_slow_fast(_synthetic_trace(), FS_HZ)
        assert set(events.slow) == set(SLOW_COLUMNS)
        assert set(events.fast) == set(FAST_COLUMNS)
        fast, slow = events.fast, events.slow
        for spike in SPIKES_S:
            assert _near(fast["time_s"], spike), spike
            k = int(fast["slow_id"][np.argmin(np.abs(fast["time_s"] - spike))])
            assert slow["start_s"][k] <= spike < slow["end_s"][k]
        # the bump in the first 2 s is dropped, the other two are found
        assert not np.any(slow["start_s"] < 2.0)
        for bump in BUMPS_S[1:]:
            assert _near(slow["peak_s"], bump, tol=0.02), bump
        assert np.all(fast["prominence_noise"] >= 5.0)
        assert np.array_equal(
            slow["n_fast"], np.bincount(fast["slow_id"], minlength=events.n_slow)
        )
        assert events.x.shape == events.slow_trace.shape == events.fast_trace.shape

    def test_merge_joins_stretches_closer_than_merge_ms(self):
        from mbo_utilities.analysis.slow_fast import SlowFastSettings, detect_slow_fast

        t = np.arange(N_SAMPLES) / FS_HZ
        trace = np.random.default_rng(1).normal(0.0, 0.002, t.size)
        for center in (8.0, 8.25):
            trace += 0.03 * np.exp(-0.5 * ((t - center) / 0.03) ** 2)
        apart = detect_slow_fast(trace, FS_HZ, SlowFastSettings(merge_ms=0.0))
        joined = detect_slow_fast(trace, FS_HZ, SlowFastSettings(merge_ms=100.0))
        around = (7.8, 8.4)
        count_apart = np.count_nonzero(
            (apart.slow["start_s"] > around[0]) & (apart.slow["end_s"] < around[1])
        )
        count_joined = np.count_nonzero(
            (joined.slow["start_s"] > around[0]) & (joined.slow["end_s"] < around[1])
        )
        assert (count_apart, count_joined) == (2, 1)

    def test_no_slow_event_gives_empty_tables(self):
        from mbo_utilities.analysis.slow_fast import detect_slow_fast

        events = detect_slow_fast(np.zeros(6000), FS_HZ)
        assert (events.n_slow, events.n_fast) == (0, 0)
        assert all(len(v) == 0 for v in (*events.slow.values(), *events.fast.values()))

    def test_settings_round_trip_and_ignore_unknown_keys(self):
        from mbo_utilities.analysis.slow_fast import SlowFastSettings

        settings = SlowFastSettings(slow_sigma_ms=16.0, slow_sd=3.5)
        assert SlowFastSettings.from_dict(settings.to_dict()) == settings
        assert SlowFastSettings.from_dict({"edge_sd": 2, "later": 1}).edge_sd == 2.0


class TestSession:
    def test_load_run_negates_a_dimming_dff(self, run_path):
        session = _session(run_path)
        assert session.loaded and session.mode == "slow_fast"
        assert session.recording_id == RID
        for spike in SPIKES_S:
            assert _near(session.times_s, spike), spike
        assert session.events.slow_trace.max() > 0

    def test_an_inverted_run_is_read_as_stored(self, tmp_path):
        from mbo_utilities.annotation.slow_fast import events_positive_dff
        from mbo_utilities.results import ResultsArray

        path = _write_run(tmp_path, _synthetic_trace(), polarity="invert_f")
        arr = ResultsArray(path, source=False)
        np.testing.assert_allclose(
            events_positive_dff(arr, "scan10", "soma"),
            _synthetic_trace().astype(np.float32),
        )

    def test_labels_and_settings_survive_a_reload(self, run_path):
        session = _session(run_path)
        session.select(1)
        session.set_label("yes")
        session.select(0)
        session.set_label("no")
        session.set_settings(slow_sd=3.2)
        path = run_path / ".curation" / "slow_fast_curation.json"
        saved = json.loads(path.read_text())["recordings"][RID]
        assert saved["settings"]["slow_sd"] == 3.2
        assert set(saved["labels"].values()) == {"yes", "no"}
        again = _session(run_path)
        assert again.settings.slow_sd == 3.2
        assert again.counts()[:2] == (1, 1)

    def test_a_label_stays_on_its_peak_across_a_settings_change(self, run_path):
        session = _session(run_path)
        spike = int(np.argmin(np.abs(session.times_s - SPIKES_S[2])))
        session.select(spike)
        session.set_label("yes")
        sample = int(session.sample_index[spike])
        session.set_settings(edge_sd=1.2)
        moved = int(np.flatnonzero(session.sample_index == sample)[0])
        assert session.label(moved) == "yes"
        assert session.current == moved

    def test_box_labels_filter_and_stepping(self, run_path):
        session = _session(run_path)
        assert session.set_labels(range(session.n), "yes") == session.n
        assert session.counts() == (session.n, 0, 0)
        session.set_view_filter("no")
        assert session.visible.size == 0
        session.set_view_filter("all")
        session.select(0)
        session.step(1)
        assert session.current == min(1, session.n - 1)
        assert session.clear_labels() == session.n
        assert session.counts() == (0, 0, session.n)

    def test_excluding_the_trace_drops_its_candidates_and_saves(self, run_path):
        session = _session(run_path)
        session.set_excluded(True)
        assert session.n == 0 and session.visible.size == 0
        assert session.slow_bounds().shape == (0, 2)
        assert _session(run_path).excluded

    def test_load_trace_takes_raw_counts(self, tmp_path):
        from mbo_utilities.annotation.slow_fast import SlowFastSession

        counts = 100.0 * (1.0 + _synthetic_trace())
        session = SlowFastSession()
        session.load_trace(
            counts,
            FS_HZ,
            recording_id="expt/MUnit_3/roi=0/inverted",
            label="MUnit_3 ROI 0",
            source_path=tmp_path / "expt.mesc",
        )
        assert session.label_path == tmp_path / ".curation" / "slow_fast_curation.json"
        assert _near(session.times_s, SPIKES_S[2])

    def test_tables_carry_the_labels(self, run_path):
        session = _session(run_path)
        session.select(0)
        session.set_label("yes")
        slow, fast = session.tables()
        assert fast["label"][0] == "yes" and len(fast["label"]) == session.n
        assert len(slow["start_s"]) == session.events.n_slow
        assert np.array_equal(slow["slow_id"], np.arange(session.events.n_slow))

    def test_export_writes_the_run_tables_beside_the_labels(self, run_path):
        import csv

        from mbo_utilities.annotation.slow_fast import TABLE_FILES, export_run_tables
        from mbo_utilities.results import ResultsArray

        session = _session(run_path)
        session.select(0)
        session.set_label("yes")
        paths = export_run_tables(ResultsArray(run_path, source=False))
        assert paths == [run_path / ".curation" / name for name in TABLE_FILES.values()]
        recordings, slow, fast = (list(csv.DictReader(p.open())) for p in paths)
        assert [r["recording_id"] for r in recordings] == [RID]
        assert recordings[0]["excluded"] == "False" and int(recordings[0]["yes"]) == 1
        assert len(slow) == session.events.n_slow and len(fast) == session.n
        assert fast[0]["label"] == "yes" and {r["label"] for r in fast[1:]} == {
            "unlabeled"
        }
        assert np.allclose([float(r["time_s"]) for r in fast], session.times_s)

    def test_an_excluded_trace_exports_without_events(self, run_path, tmp_path):
        import csv

        from mbo_utilities.annotation.slow_fast import export_tables

        session = _session(run_path)
        session.set_excluded(True)
        recordings, slow, fast = (
            list(csv.DictReader(p.open()))
            for p in export_tables([session], tmp_path / "out")
        )
        assert recordings[0]["excluded"] == "True" and recordings[0]["n_fast"] == "0"
        assert slow == [] and fast == []

    def test_kept_times_leave_out_the_no_calls(self, run_path):
        from mbo_utilities.annotation.slow_fast import kept_times

        session = _session(run_path)
        session.select(0)
        session.set_label("no")
        assert kept_times(session).size == session.n - 1

    def test_colours_are_vnoisers(self):
        pytest.importorskip("vnoiser")
        from mbo_utilities.annotation.slow_fast import LABEL_RGBA
        from mbo_utilities.vnoiser import LABEL_RGBA as VNOISER_RGBA

        assert set(LABEL_RGBA) == set(VNOISER_RGBA)
        for name, rgba in LABEL_RGBA.items():
            assert rgba == pytest.approx(VNOISER_RGBA[name])


class _Parent:
    def __init__(self, iw):
        self.image_widget = iw
        self.logger = logging.getLogger("test_slow_fast")
        self.fpath = None


class _Recorded:
    """A panel draw that records a traceback instead of raising inside the render loop."""

    def __init__(self, draw, errors: list[str]):
        self.draw = draw
        self.errors = errors

    def __call__(self, *_args):
        try:
            self.draw()
        except Exception:
            self.errors.append(traceback.format_exc())


@pytest.fixture
def curation(monkeypatch):
    if not _offscreen_selected():
        pytest.skip("needs the offscreen rendercanvas")
    from mbo_utilities.gui import event_curation
    from mbo_utilities.gui._ndviewer import MboNDViewer

    monkeypatch.setattr(event_curation, "set_last_dir", lambda *a, **k: None)
    monkeypatch.setattr(event_curation, "get_last_dir", lambda *a, **k: None)
    data = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)
    iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
    iw.show()
    widget = event_curation.EventCurationWidget(_Parent(iw))
    errors: list[str] = []
    for panel in widget.strip.panels:
        panel.draw = _Recorded(panel.draw, errors)
    widget.strip._update_calls.append(_Recorded(widget.draw_tab, errors))
    widget.errors = errors
    yield widget
    widget.close()
    iw.close()


def _frames(widget, n=3):
    widget.strip.focus("curation")
    for _ in range(n):
        widget.parent.image_widget.figure.canvas.draw()
    assert not widget.errors, widget.errors[0]


def _open_slow_fast(widget, run_path):
    widget.set_method("slow_fast")
    widget.scan(run_path)
    widget.wait(60)
    assert widget.session is not None and widget.session.loaded, widget.status


class TestWidget:
    def test_slow_fast_loads_and_draws_every_panel(self, curation, run_path):
        _open_slow_fast(curation, run_path)
        assert curation.method == "slow_fast"
        assert curation.session.mode == "slow_fast" and curation.session.n >= 3
        _frames(curation)

    def test_labels_box_and_settings_redraw(self, curation, run_path):
        _open_slow_fast(curation, run_path)
        session = curation.session
        session.set_label("yes")
        curation.set_box_mode("no")
        curation._box = ("timeline", np.arange(session.n))
        assert curation.apply_box() == session.n
        assert session.counts()[1] == session.n
        curation._apply_setting(session, "slow_sd", 4.0)
        session.set_view_filter("no")
        _frames(curation)
        assert session.settings.slow_sd == 4.0

    def test_export_tables_writes_off_the_draw_thread(self, curation, run_path):
        _open_slow_fast(curation, run_path)
        curation._export_slow_fast_tables(curation.session)
        assert (run_path / ".curation" / "slow_fast_fast.csv").exists()
        assert curation.status.startswith("wrote ") and not curation._exporting
        _frames(curation)

    def test_an_excluded_trace_draws(self, curation, run_path):
        _open_slow_fast(curation, run_path)
        curation.session.set_excluded(True)
        _frames(curation)

    def test_switching_method_keeps_the_recording(self, curation, run_path):
        pytest.importorskip("vnoiser")
        curation.scan(run_path)
        curation.wait(60)
        vnoiser_session = curation.session
        assert curation.method == "vnoiser"
        curation.set_method("slow_fast")
        curation.wait(60)
        assert curation.session.recording_id == vnoiser_session.recording_id
        curation.set_method("vnoiser")
        assert curation.session is vnoiser_session

    def test_compare_loads_and_draws_the_other_method(self, curation, run_path):
        pytest.importorskip("vnoiser")
        _open_slow_fast(curation, run_path)
        curation.compare = True
        _frames(curation, 2)
        curation.wait(60)
        assert ("fast", RID) in curation.sessions
        _frames(curation)

    def test_without_vnoiser_the_window_opens_on_slow_fast(self, monkeypatch, run_path):
        if not _offscreen_selected():
            pytest.skip("needs the offscreen rendercanvas")
        from mbo_utilities.gui import event_curation
        from mbo_utilities.gui._ndviewer import MboNDViewer

        monkeypatch.setattr(event_curation, "HAS_VNOISER", False)
        monkeypatch.setattr(event_curation, "set_last_dir", lambda *a, **k: None)
        monkeypatch.setattr(event_curation, "get_last_dir", lambda *a, **k: None)
        iw = MboNDViewer(
            data=np.zeros((2, 8, 8), np.float32), figure_kwargs={"size": FIGURE_SIZE}
        )
        iw.show()
        widget = event_curation.EventCurationWidget(_Parent(iw))
        try:
            assert widget.mode == "slow_fast"
            widget.set_method("vnoiser")
            assert widget.mode == "slow_fast" and "not installed" in widget.status
            widget.scan(run_path)
            widget.wait(60)
            assert widget.session.loaded
        finally:
            widget.close()
            iw.close()

    def test_a_run_without_dff_or_raw_fails_only_for_slow_fast(
        self, curation, tmp_path
    ):
        from mbo_utilities.results import Results, ResultUnit

        unit = ResultUnit(
            name="scan10",
            kind="scan",
            index=10,
            fs=FS_HZ,
            roi_names=["soma"],
            traces={"denoised": _synthetic_trace().astype(np.float32)[None, :]},
            member_kind="line",
            members=[np.array([0])],
            attrs={"scan_id": 10},
        )
        path = tmp_path / "expt1.2026-11-03-12-00-00.vnoiser.zarr"
        Results(pipeline="vnoiser", units={"scan10": unit}).write(path)
        curation.set_method("slow_fast")
        curation.scan(path)
        curation.wait(60)
        rec = curation.recording(RID)
        assert "slow_fast" in rec.errors and set(rec.errors) == {"slow_fast"}
