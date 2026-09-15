"""vnoiser event curation inside the viewer.

``CurationSession`` drives vnoiser's notebook dashboard headless, so every
rule (candidates, seed template, auto calls, the saved JSON) is vnoiser's
own; the tests here check the adapter reads and writes the same state the
notebook would, and that the imgui panels draw it offscreen without error.
Skipped when vnoiser is not installed.
"""

from __future__ import annotations

import json
import logging
import pickle
import traceback

import numpy as np
import pytest

from tests.test_manual_roi import _offscreen_selected

vnoiser = pytest.importorskip("vnoiser")

FS_HZ = 1000.0
EVENT_SAMPLES = (800, 2000, 3200)


def _write_spatial_recording(root, trace=None):
    """One experiment's PF folder the way vnoiser's own tests lay it out."""
    pf_dir = root / "stan1" / "stan1_expt1" / "PF"
    pf_dir.mkdir(parents=True)
    t = np.arange(4000, dtype=float) / FS_HZ
    if trace is None:
        trace = 0.1 * np.sin(2 * np.pi * 5 * t)
        trace[list(EVENT_SAMPLES)] += [4.0, 7.0, 5.0]
    payloads = {
        "denoised_trace_scans.pkl": {"10": {"soma": trace}},
        "fs_scans.pkl": {"10": FS_HZ},
        "scanIDs_ROIs.pkl": {
            "scanID_spatial": np.array([10]),
            "domain_ROInumber": {"soma": [0]},
        },
    }
    for filename, payload in payloads.items():
        with (pf_dir / filename).open("wb") as handle:
            pickle.dump(payload, handle)
    return pf_dir


@pytest.fixture(autouse=True)
def _keep_preferences(monkeypatch):
    """The widget remembers the last data path in ~/.mbo; tests must not
    leave a pytest tmp dir there for the next real session to open."""
    from mbo_utilities.gui import event_curation

    monkeypatch.setattr(event_curation, "set_last_dir", lambda *a, **k: None)
    monkeypatch.setattr(event_curation, "get_last_dir", lambda *a, **k: None)


@pytest.fixture
def data_root(tmp_path):
    _write_spatial_recording(tmp_path)
    return tmp_path


def _loaded(data_root, mode="fast"):
    from mbo_utilities.vnoiser import CurationSession

    session = CurationSession(data_root, mode=mode)
    experiment = session.experiments(session.animals[0][1])[0][1]
    session.select_experiment(experiment)
    session.load(session.recordings[0][1])
    return session


# ----------------------------------------------------------------------
# the session
# ----------------------------------------------------------------------


class TestSession:
    def test_walks_the_hierarchy_one_level_at_a_time(self, data_root):
        from mbo_utilities.vnoiser import CurationSession

        session = CurationSession(data_root, mode="fast")
        assert session.hierarchical
        assert not session.loaded
        animals = session.animals
        assert [label for label, _ in animals] == ["stan1"]
        experiments = session.experiments(animals[0][1])
        assert [label for label, _ in experiments] == ["stan1_expt1"]
        assert session.recordings == []
        session.select_experiment(experiments[0][1])
        assert [value for _, value in session.recordings] == [
            "stan1/stan1_expt1/scan=10/domain=soma"
        ]

    def test_load_reads_the_processed_trace_and_finds_the_events(self, data_root):
        session = _loaded(data_root)
        assert session.loaded
        assert session.fs == FS_HZ
        assert session.t.shape == session.denoised.shape == (4000,)
        assert session.n == 3
        assert np.allclose(session.times_s * FS_HZ, EVENT_SAMPLES, atol=2)
        assert session.label_path == data_root / "stan1" / "stan1_expt1" / "PF" / ".curation" / "fast_template_curation.json"
        assert "denoised" in session.cache_status

    def test_bad_recording_id_raises(self, data_root):
        from mbo_utilities.vnoiser import CurationSession

        session = CurationSession(data_root, mode="fast")
        session.select_experiment(session.experiments(session.animals[0][1])[0][1])
        with pytest.raises(KeyError):
            session.load("nope")

    def test_labels_go_to_vnoiser_json(self, data_root):
        session = _loaded(data_root)
        session.select(1)
        session.set_label("yes")
        assert session.counts() == (1, 0, 2)
        assert session.manual_label(1) == "yes"
        assert session.label(1) == "yes"
        payload = json.loads(session.label_path.read_text(encoding="utf-8"))
        assert payload["mode"] == "fast"
        keys = list(payload["events"])
        assert len(keys) == 1
        assert keys[0].endswith(f"sample={int(session.candidates.indices[1])}")
        assert payload["events"][keys[0]]["label"] == "yes"
        session.set_label("unlabeled")
        assert session.counts() == (0, 0, 3)

    def test_seeded_mode_auto_calls_and_colours(self, data_root):
        session = _loaded(data_root)
        shown = session.labels()
        assert set(shown) <= {"auto_yes", "auto_no", "unlabeled"}
        colours = session.colors()
        assert colours.shape == (3, 4)
        assert colours.dtype == np.float32
        info = session.event_info(session.current)
        assert info["auto_call"] in ("pass", "reject", None)
        assert info["source"] == "threshold"
        assert session.auto_pass is None
        session.set_auto_pass(session.auto_pass_range[0])
        assert all(label == "auto_yes" for label in session.labels())
        assert session.event_info(0)["auto_call"] == "pass"

    def test_manual_mode_has_no_auto_calls(self, data_root):
        session = _loaded(data_root, mode="manual")
        assert not session.seeded
        assert session.labels() == ["unlabeled"] * 3
        assert session.template is None
        session.select(0)
        session.set_label("yes")
        assert session.template is not None
        assert list(session.template_source) == [0]

    def test_threshold_rebuilds_candidates_within_the_slider_range(self, data_root):
        session = _loaded(data_root)
        lo, hi, step = session.threshold_range
        assert lo < session.threshold < hi and step > 0
        session.set_threshold(6.0)
        assert session.threshold == 6.0
        assert session.n == 1
        session.set_threshold(hi + 100)
        assert session.threshold == hi
        payload = json.loads(session.label_path.read_text(encoding="utf-8"))
        assert payload["candidate_detection"]["thresholds"] == {
            "stan1/stan1_expt1/scan=10/domain=soma": hi
        }

    def test_a1_a2_controls_do_what_the_notebook_sliders_do(self, data_root):
        """The imgui A1 / A2 controls call the session; the notebook moves
        ipywidgets sliders. Driving both on the same recording must leave
        identical candidates, auto calls, labels and saved JSON."""
        from vnoiser.curation import EventCurationDashboard

        from mbo_utilities.vnoiser import CurationSession

        def notebook(mode):
            dash = EventCurationDashboard(data_path=data_root, mode=mode, duration_s=None, auto_load=False)
            experiment = dash.dataset.experiment_options(dash.dataset.animal_options()[0][1])[0][1]
            dash._experiment_changed({"new": experiment})
            dash.recording_dropdown.value = dash.dataset.recording_options()[0][1]
            dash._load_selected_recording(None)
            return dash

        def state(dash):
            return (
                dash.candidates.indices.tolist(),
                [dash._label_for_index(i) for i in range(len(dash.candidates.indices))],
                float(dash.candidate_threshold),
                dash.auto_pass_amplitude,
                bool(dash.waveform_rejection),
                json.loads(dash.label_path.read_text(encoding="utf-8"))["candidate_detection"]
                if dash.label_path.exists()
                else None,
            )

        dash = notebook("fast")
        session = _loaded(data_root, mode="fast")
        assert state(dash) == state(session.dash)

        # A1: the notebook slider snaps to its own range; so does set_threshold
        lo, hi, _step = session.threshold_range
        assert (float(dash.threshold_slider.min), float(dash.threshold_slider.max)) == (lo, hi)
        target = lo + 0.5 * (hi - lo)
        dash.threshold_slider.value = target
        session.set_threshold(target)
        assert state(dash) == state(session.dash)
        dash.threshold_slider.value = hi + 100  # ipywidgets clips to max
        session.set_threshold(hi + 100)
        assert state(dash) == state(session.dash)

        # A2: auto-pass starts off (slider parked at its top), then a value
        # auto-passes everything at or above it
        assert dash.auto_pass_amplitude is None and session.auto_pass is None
        assert float(dash.auto_pass_slider.value) == session.auto_pass_range[1]
        amp = float(dash.auto_pass_slider.min)
        dash.auto_pass_slider.value = amp
        session.set_auto_pass(amp)
        assert state(dash) == state(session.dash)
        assert all(label == "auto_yes" for label in session.labels())
        assert session.auto_pass_count() == session.n

        # waveform rejection off leaves sub-threshold candidates unlabeled
        dash.waveform_rejection_checkbox.value = False
        session.set_waveform_rejection(False)
        assert state(dash) == state(session.dash)

        # and manual labels survive a threshold change on both (the
        # threshold sits at the top of its range now, so bring candidates back)
        dash.threshold_slider.value = target
        session.set_threshold(target)
        assert session.n > 0
        dash._select_event(0)
        dash._set_label("no")
        session.select(0)
        session.set_label("no")
        dash.threshold_slider.value = lo + 0.25 * (hi - lo)
        session.set_threshold(lo + 0.25 * (hi - lo))
        assert state(dash) == state(session.dash)
        # the label follows its event (keyed by sample), not its index
        assert [session.manual_label(i) for i in range(session.n)].count("no") == 1

    def test_a2_floor_reaches_under_every_amplitude(self, data_root):
        """The amplitude A2 compares with is baseline-subtracted, so it can be
        below the notebook slider's floor (the trace median); the session's
        range goes under the lowest one so the bottom passes everything."""
        session = _loaded(data_root, mode="fast")
        lo, hi, _step = session.threshold_range
        session.set_threshold(lo)
        assert session.n > 3
        floor = session.auto_pass_range[0]
        assert floor <= float(session.amplitudes.min())
        assert floor <= float(session.dash.auto_pass_slider.min)
        session.set_auto_pass(floor)
        assert all(label == "auto_yes" for label in session.labels())
        assert session.auto_pass_count() == session.n
        # halfway up, only the candidates at or above it pass
        mid = 0.5 * (floor + session.auto_pass_range[1])
        session.set_auto_pass(mid)
        above = int((session.amplitudes >= mid).sum())
        assert session.auto_pass_count() == above
        assert sum(label == "auto_yes" for label in session.labels()) >= above

    def test_a3_a4_controls_do_what_the_notebook_sliders_do(self, data_root):
        """The PC1 line (A3) and the cosine threshold (A4) drive vnoiser's
        own sliders; the session's setters must leave the same calls and
        the same saved JSON as moving them in the notebook."""
        from vnoiser.curation import EventCurationDashboard

        def notebook(mode):
            dash = EventCurationDashboard(data_path=data_root, mode=mode, duration_s=None, auto_load=False)
            experiment = dash.dataset.experiment_options(dash.dataset.animal_options()[0][1])[0][1]
            dash._experiment_changed({"new": experiment})
            dash.recording_dropdown.value = dash.dataset.recording_options()[0][1]
            dash._load_selected_recording(None)
            return dash

        def state(dash):
            return (
                [dash._label_for_index(i) for i in range(len(dash.candidates.indices))],
                dash.auto_pass_pc1,
                dash.auto_pass_pc1_side,
                float(dash.auto_template_threshold),
                json.loads(dash.label_path.read_text(encoding="utf-8"))["candidate_detection"]
                if dash.label_path.exists()
                else None,
            )

        dash = notebook("fast")
        session = _loaded(data_root, mode="fast")
        assert state(dash) == state(session.dash)
        assert session.auto_pass_pc1 is None and session.auto_pass_pc1_side == "right"
        assert session.auto_template_threshold == pytest.approx(0.8)

        # A3 starts off, parked beyond the rightmost point so nothing passes
        lo, hi, _step = session.auto_pass_pc1_range
        pc1 = session.pca_scores[:, 0]
        assert lo < pc1.min() and hi > pc1.max()
        assert session.auto_pass_pc1_shown == pytest.approx(hi)
        assert float(dash.pc1_slider.value) == pytest.approx(hi)
        assert session.auto_pass_pc1_count() == 0

        # a line between the two highest PC1 scores passes the top one
        order = np.argsort(pc1)
        line = 0.5 * (pc1[order[-1]] + pc1[order[-2]])
        dash.pc1_slider.value = line
        session.set_auto_pass_pc1(line)
        assert state(dash) == state(session.dash)
        assert session.auto_pass_pc1_count() == 1
        assert session.label(int(order[-1])) == "auto_yes"
        assert session.event_info(int(order[-1]))["pc1_pass"] is True
        assert session.event_info(int(order[0]))["pc1_pass"] is False

        # flip the side: the others pass instead
        dash.pc1_side.value = "left"
        session.set_auto_pass_pc1_side("left")
        assert state(dash) == state(session.dash)
        assert session.auto_pass_pc1_count() == session.n - 1
        assert session.label(int(order[0])) == "auto_yes"

        # A4: every candidate at or above the cosine threshold passes; at the
        # floor nothing is rejected, at the top only A2 / A3 can pass one
        session.set_auto_pass_pc1_side("right")
        dash.pc1_side.value = "right"
        scores = session.dash.initial_template_scores
        assert session.auto_template_count() == int((scores >= 0.8).sum())
        floor = session.auto_template_threshold_range[0]
        dash.cosine_slider.value = floor
        session.set_auto_template_threshold(floor)
        assert state(dash) == state(session.dash)
        assert session.auto_template_count() == session.n
        assert "auto_no" not in session.labels()
        dash.cosine_slider.value = 1.0
        session.set_auto_template_threshold(1.0)
        assert state(dash) == state(session.dash)
        rejected = sum(label == "auto_no" for label in session.labels())
        expected = sum(
            not session.event_info(i)["pc1_pass"] and scores[i] < 1.0 for i in range(session.n)
        )
        assert rejected == expected > 0

        # both are saved per recording and restored by a fresh session
        detection = state(session.dash)[-1]
        assert detection["auto_pass_pc1"][session.recording_id] == pytest.approx(line)
        assert detection["auto_pass_pc1_sides"][session.recording_id] == "right"
        assert detection["auto_template_thresholds"][session.recording_id] == pytest.approx(1.0)
        again = _loaded(data_root, mode="fast")
        assert again.auto_pass_pc1 == pytest.approx(line)
        assert again.auto_template_threshold == pytest.approx(1.0)
        assert again.labels() == session.labels()

        # manual mode has neither rule
        manual = _loaded(data_root, mode="manual")
        manual.set_auto_pass_pc1(line)
        manual.set_auto_template_threshold(0.0)
        assert manual.auto_pass_pc1 is None
        assert manual.auto_pass_pc1_count() == 0 and manual.auto_template_count() == 0

    def test_set_labels_labels_a_box_at_once(self, data_root):
        session = _loaded(data_root, mode="fast")
        assert session.n == 3
        assert session.set_labels([0, 2], "no") == 2
        assert [session.manual_label(i) for i in range(3)] == ["no", "unlabeled", "no"]
        saved = json.loads(session.label_path.read_text(encoding="utf-8"))
        assert sum(e["label"] == "no" for e in saved["events"].values()) == 2
        assert session.set_labels([0, 1, 2], "yes") == 3
        assert session.template_source.tolist() == [0, 1, 2]
        assert session.set_labels([1], "unlabeled") == 1
        assert session.manual_label(1) == "unlabeled"
        assert session.set_labels([], "yes") == 0
        assert session.set_labels([0], "maybe") == 0

    def test_view_filter_and_stepping(self, data_root):
        session = _loaded(data_root)
        session.select(0)
        session.set_label("no")
        session.set_view_filter("no")
        assert list(session.visible) == [0]
        session.set_view_filter("all")
        assert list(session.visible) == [0, 1, 2]
        session.step(1)
        assert session.current == 1
        session.step(-2)
        assert session.current == 2

    def test_slow_mode_keeps_its_own_file(self, data_root):
        fast = _loaded(data_root, mode="fast")
        slow = _loaded(data_root, mode="slow")
        assert slow.label_path.name == "slow_template_curation.json"
        assert fast.label_path != slow.label_path
        assert slow.candidate_window_ms == 500.0 and fast.candidate_window_ms == 100.0
        assert slow.analysis_trace.shape == slow.denoised.shape
        assert not np.array_equal(slow.analysis_trace, slow.denoised)

    def test_scan_moves_every_mode_to_the_new_path(self, data_root, tmp_path):
        other = tmp_path / "other"
        _write_spatial_recording(other)
        session = _loaded(data_root)
        session.scan(other)
        assert not session.loaded
        assert session.data_path == other
        assert session.experiment == "" and session.recording_id == ""


# ----------------------------------------------------------------------
# the scatter widget's pick
# ----------------------------------------------------------------------


class TestNearestIndex:
    def test_nearest_within_radius(self):
        from mbo_utilities.gui.imgui.scatter import nearest_index

        px = np.array([0.0, 10.0, 20.0])
        py = np.array([0.0, 0.0, 0.0])
        assert nearest_index(px, py, 11.0, 2.0, 8.0) == 1
        assert nearest_index(px, py, 15.0, 0.0, 4.0) is None
        assert nearest_index([], [], 0.0, 0.0, 8.0) is None

    def test_nan_points_are_skipped(self):
        from mbo_utilities.gui.imgui.scatter import nearest_index

        px = np.array([np.nan, 3.0])
        py = np.array([np.nan, 0.0])
        assert nearest_index(px, py, 0.0, 0.0, 8.0) == 1

    def test_packed_colours_are_abgr(self):
        from mbo_utilities.gui.imgui.lines import packed_colors

        packed = packed_colors(np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.5]]))
        assert packed.dtype == np.uint32
        assert packed[0] == 0xFF0000FF
        assert packed[1] == (128 << 24) | (255 << 8)
        assert packed_colors(np.array([0.0, 0.0, 1.0])).tolist() == [0xFFFF0000]


# ----------------------------------------------------------------------
# the panels, offscreen
# ----------------------------------------------------------------------

FIGURE_SIZE = (1200, 800)


class _Parent:
    def __init__(self, iw):
        self.image_widget = iw
        self.logger = logging.getLogger("test_event_curation")
        self.fpath = None


@pytest.fixture
def curation(data_root):
    if not _offscreen_selected():
        pytest.skip("needs the offscreen rendercanvas")
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui.event_curation import EventCurationWidget

    data = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)
    iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
    iw.show()
    widget = EventCurationWidget(_Parent(iw))
    errors: list[str] = []

    def wrap(fn):
        def body(*_args):
            try:
                fn()
            except Exception:
                errors.append(traceback.format_exc())

        return body

    for panel in widget.strip.panels:
        panel.draw = wrap(panel.draw)
    # the strip keeps drawing its own panels; the tab body joins it so one
    # frame exercises the top panel and the right tab together
    widget.strip._update_calls.append(wrap(widget.draw_tab))
    widget.errors = errors
    yield widget
    widget.close()
    iw.close()


def _frames(widget, n=2):
    for _ in range(n):
        widget.parent.image_widget.figure.canvas.draw()
    assert not widget.errors, widget.errors[0]


def _load(widget, data_root):
    """Scan the root: everything under it is cataloged and loaded."""
    widget.scan(data_root)
    widget.wait(60)
    assert widget.session is not None and widget.session.loaded, widget.status


def _each_panel(widget, frames=2):
    """The one Curation panel draws both rows (A with its cards, B to D)."""
    widget.strip.focus("curation")
    _frames(widget, frames)


class TestWidget:
    def test_registers_one_top_panel(self, curation):
        # the notebook's dashboard is a single tab: no separate Candidates tab
        assert [p.key for p in curation.strip.panels] == ["curation"]
        assert {p.right_tab for p in curation.strip.panels} == {"curation"}
        _frames(curation)

    def test_scope_narrows_what_is_shown_and_flipped(self, curation, data_root, tmp_path):
        # a second scan in the same PF folder, as an experiment with two units has
        pf = data_root / "stan1" / "stan1_expt1" / "PF"
        traces = pickle.loads((pf / "denoised_trace_scans.pkl").read_bytes())
        traces["20"] = {"soma": traces["10"]["soma"]}
        (pf / "denoised_trace_scans.pkl").write_bytes(pickle.dumps(traces))
        fs = pickle.loads((pf / "fs_scans.pkl").read_bytes())
        fs["20"] = fs["10"]
        (pf / "fs_scans.pkl").write_bytes(pickle.dumps(fs))
        meta = pickle.loads((pf / "scanIDs_ROIs.pkl").read_bytes())
        meta["scanID_spatial"] = np.array([10, 20])
        (pf / "scanIDs_ROIs.pkl").write_bytes(pickle.dumps(meta))

        curation.scope = lambda rec: "scan=20" in rec.rid.split("/")
        _load(curation, data_root)
        rids = [r.rid for r in curation.catalog]
        assert len(rids) == 2 and any("scan=10" in r for r in rids)
        # the catalog still holds both scans; only scan 20 is shown, loaded and flipped to
        assert [r.rid for r in curation.shown] == ["stan1/stan1_expt1/scan=20/domain=soma"]
        assert curation.current == "stan1/stan1_expt1/scan=20/domain=soma"
        assert [r.rid for r, _ in curation.loaded()] == [curation.current]
        assert [r.rid for r in curation.loadable()] == [curation.current]
        curation.step_recording(1)
        assert curation.current == "stan1/stan1_expt1/scan=20/domain=soma"
        _each_panel(curation)
        # widening the scope brings the other scan back
        curation.scope = None
        assert len(curation.shown) == 2 and len(curation.loadable()) == 2
        curation.step_recording(1)
        assert "scan=10" in curation.current

    def test_flipping_through_recordings_updates_everything(self, curation, data_root, tmp_path):
        pf2 = tmp_path / "stan1" / "stan1_expt2" / "PF"
        pf2.mkdir(parents=True)
        for name in ("denoised_trace_scans.pkl", "fs_scans.pkl", "scanIDs_ROIs.pkl"):
            (pf2 / name).write_bytes((data_root / "stan1" / "stan1_expt1" / "PF" / name).read_bytes())
        seen = []
        curation.on_recording = seen.append
        _load(curation, data_root)
        first = curation.current
        _each_panel(curation)
        assert seen == [first]
        curation.step_recording(1)
        assert curation.current != first and curation.loading
        curation.wait(60)
        _each_panel(curation)
        assert seen == [first, curation.current]
        assert curation.session.recording_id == curation.current
        curation.step_recording(1)
        assert curation.current == first
        _each_panel(curation)
        assert seen == [first, seen[1], first]

    def test_scan_catalogs_and_loads_everything(self, curation, data_root, tmp_path):
        # a second experiment under the same animal is found and loaded too
        pf2 = tmp_path / "stan1" / "stan1_expt2" / "PF"
        pf2.mkdir(parents=True)
        for name in ("denoised_trace_scans.pkl", "fs_scans.pkl", "scanIDs_ROIs.pkl"):
            (pf2 / name).write_bytes((data_root / "stan1" / "stan1_expt1" / "PF" / name).read_bytes())
        curation.scan(data_root)
        assert [r.rid for r in curation.catalog] == [
            "stan1/stan1_expt1/scan=10/domain=soma",
            "stan1/stan1_expt2/scan=10/domain=soma",
        ]
        assert curation.experiments == ["stan1_expt1", "stan1_expt2"]
        assert curation.current == curation.catalog[0].rid
        assert "2 recordings in 2 experiment(s)" in curation.status
        curation.wait(60)
        # the first experiment loads on its own; the rest on demand
        assert [r.rid for r, _ in curation.loaded()] == [curation.catalog[0].rid]
        curation.load_all(None)
        curation.wait(60)
        assert len(curation.loaded()) == 2
        _each_panel(curation)

    def test_bad_path_reports_instead_of_raising(self, curation, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        curation.scan(empty)
        assert curation.catalog == []
        assert curation.session is None
        assert "no vnoiser data" in curation.status
        _frames(curation)

    def test_load_runs_off_the_frame_and_draws_every_panel(self, curation, data_root):
        _load(curation, data_root)
        assert curation.session.n == 3
        _each_panel(curation, 3)
        assert curation.strip.active == "curation"

    def test_mode_switch_loads_the_same_recording(self, curation, data_root):
        _load(curation, data_root)
        fast = curation.session
        curation.set_mode("slow")
        assert curation.session is None and curation.loading
        curation.wait(60)
        slow = curation.session
        assert slow is not fast and slow.mode == "slow"
        assert slow.recording_id == fast.recording_id
        assert curation.sessions[("fast", fast.recording_id)] is fast
        _each_panel(curation)

    def test_labels_and_filters_redraw(self, curation, data_root):
        _load(curation, data_root)
        session = curation.session
        session.select(1)
        session.set_label("yes")
        session.set_view_filter("yes")
        _each_panel(curation)
        assert session.counts() == (1, 0, 2)

    def test_a3_a4_card_and_pc1_line_draw(self, curation, data_root):
        """The A3 / A4 card and the PC1 line on the PCA draw in a seeded
        mode; a release of the dragged line applies through the session."""
        _load(curation, data_root)
        session = curation.session
        _each_panel(curation)
        lo, hi, _step = session.auto_pass_pc1_range
        line = 0.5 * (lo + hi)
        session.set_auto_pass_pc1(line)
        session.set_auto_pass_pc1_side("left")
        session.set_auto_template_threshold(0.5)
        _each_panel(curation, 3)
        assert session.auto_pass_pc1 == pytest.approx(line)
        # the drag ends off-frame: the next frame applies the held value
        curation._pc1_drag = lo + 0.25 * (hi - lo)
        _each_panel(curation)
        assert curation._pc1_drag is None
        assert session.auto_pass_pc1 == pytest.approx(lo + 0.25 * (hi - lo))
        # manual mode shows neither the card nor the line
        curation.set_mode("manual")
        curation.wait(60)
        _each_panel(curation)

    def test_box_mode_labels_what_the_box_holds(self, curation, data_root):
        _load(curation, data_root)
        session = curation.session
        assert curation.box_mode is None and curation.apply_box() == 0
        curation.set_box_mode("no")
        assert curation.box_mode == "no"
        curation.set_box_mode("no")  # the same button again leaves the mode
        assert curation.box_mode is None
        curation.set_box_mode("no")
        _each_panel(curation)  # box mode with no box yet draws (plots take right-drag)
        # the rectangle comes from a right-drag inside a plot; stand one in
        # that covers the whole trace: the frame measures what it holds
        t = session.times_s
        curation._box_rect = {
            "plot": "timeline", "x0": float(t.min()) - 1.0, "y0": -100.0,
            "x1": float(t.max()) + 1.0, "y1": 100.0, "drawing": False,
        }
        _each_panel(curation)
        assert sorted(curation.boxed().tolist()) == [0, 1, 2]
        # a narrower one, still measured each frame
        curation._box_rect.update(x0=float(t[0]) - 0.01, x1=float(t[0]) + 0.01)
        _each_panel(curation)
        assert curation.boxed().tolist() == [0]
        assert curation.apply_box() == 1
        assert [session.manual_label(i) for i in range(3)] == ["no", "unlabeled", "unlabeled"]
        # the box is gone, the mode stays for the next one
        assert curation._box_rect is None and curation.boxed().size == 0 and curation.box_mode == "no"
        # switching to accept drops any box; a box on the PCA applies the same way
        curation._box_rect = {"plot": "timeline", "x0": 0, "y0": 0, "x1": 1, "y1": 1, "drawing": False}
        curation.set_box_mode("yes")
        assert curation.box_mode == "yes" and curation._box_rect is None
        curation._box = ("pca", np.array([1, 2]))
        assert curation.apply_box() == 2
        assert [session.manual_label(i) for i in range(3)] == ["no", "yes", "yes"]
        curation.exit_box_mode()
        assert curation.box_mode is None
        _each_panel(curation)

    def test_close_gives_the_strip_back(self, curation):
        strip = curation.strip
        curation.close()
        assert not strip.has("curation")
        curation.close()


class TestViewerIntegration:
    def test_widgets_menu_entry_and_tab(self):
        from mbo_utilities.gui.widgets.widget_toggles import get_entry
        from mbo_utilities.gui.widgets.curation_tab import CurationTabWidget

        entry = get_entry("vnoiser")
        assert entry is not None and entry.default is False
        assert entry.on_toggle is not None
        assert CurationTabWidget.toggle_key == "vnoiser"
        assert CurationTabWidget.placement == "tab"

    def test_tab_is_greyed_until_the_widget_is_on(self):
        from mbo_utilities.gui.widgets.curation_tab import CurationTabWidget

        class Parent:
            event_curation = None
            top_strip = None

        tab = CurationTabWidget(Parent())
        assert "Widgets > Event Curation" in tab.tab_disabled()
        assert tab.wants_focus() is False

    def test_check_install_lists_vnoiser(self):
        from mbo_utilities.install import HAS_VNOISER, VNOISER_HINT

        assert HAS_VNOISER is True
        assert "vnoiser" in VNOISER_HINT

    def test_sync_on_a_preview_widget(self):
        if not _offscreen_selected():
            pytest.skip("needs the offscreen rendercanvas")
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui.data_vis import DataVis

        data = np.random.default_rng(0).random((4, 1, 3, 32, 32)).astype(np.float32)
        vis = DataVis(NumpyArray(data, dims="TCZYX"), size=FIGURE_SIZE)
        vis.show()
        try:
            parent = vis.widget
            parent.sync_event_curation(True)
            widget = parent.event_curation
            assert widget is not None
            assert widget.strip is parent.top_strip
            assert parent.top_strip.has("curation")
            for _ in range(2):
                vis.figure.canvas.draw()
            parent.sync_event_curation(False)
            assert parent.event_curation is None
            assert not parent.top_strip.has("curation")
        finally:
            vis.close()


# ----------------------------------------------------------------------
# a trace handed over in memory (a line-scan ROI)
# ----------------------------------------------------------------------


def _fake_pipeline(monkeypatch, event_indices=(700, 1500, 2300)):
    """Stand in for the wavelet denoiser, the way vnoiser's own tests do."""
    from vnoiser.curation import EventCurationDashboard

    event_indices = np.asarray(event_indices, dtype=int)
    calls = []

    def run_pipeline(self, sample):
        calls.append(sample.metadata.get("recording_id"))
        self.recording = sample
        self.raw_trace = sample.trace.astype(float)
        self.denoised = np.zeros_like(self.raw_trace)
        self.denoised[event_indices] = np.arange(len(event_indices)) + 4.0
        self.denoiser_input = self.denoised.copy()
        self.raw_cluster_starts = event_indices.copy()
        self.pca_event_indices = np.array([], dtype=int)
        self.pca_event_score_z = np.zeros_like(self.raw_trace)
        self.pca_threshold_z = 3.0
        self._finalize_pipeline_state()

    monkeypatch.setattr(EventCurationDashboard, "_run_pipeline", run_pipeline)
    return calls


def _source(tmp_path):
    source = tmp_path / "expt.mesc"
    source.write_bytes(b"not really a mesc")
    return source


class TestLoadTrace:
    def test_runs_the_pipeline_and_files_labels_beside_the_source(self, tmp_path, monkeypatch):
        from mbo_utilities.vnoiser import CurationSession

        calls = _fake_pipeline(monkeypatch)
        source = _source(tmp_path)
        trace = np.random.default_rng(0).normal(size=3000) + 100.0
        session = CurationSession(source, mode="fast")
        session.load_trace(trace, FS_HZ, recording_id="expt/MUnit_1/roi=3", label="ROI 3", source_path=source)
        assert session.loaded and session.n == 3
        assert calls == ["expt/MUnit_1/roi=3"]
        assert session.cache_status == "computed pipeline"
        assert session.label_path == tmp_path / ".curation" / "fast_template_curation.json"
        session.select(0)
        session.set_label("yes")
        payload = json.loads(session.label_path.read_text(encoding="utf-8"))
        key = next(iter(payload["events"]))
        assert key.startswith("expt/MUnit_1/roi=3|sample=")
        assert payload["events"][key]["recording"] == "expt/MUnit_1/roi=3"

    def test_second_load_restores_the_cache_per_roi(self, tmp_path, monkeypatch):
        from mbo_utilities.vnoiser import CurationSession

        calls = _fake_pipeline(monkeypatch)
        source = _source(tmp_path)
        trace = np.random.default_rng(1).normal(size=3000)
        session = CurationSession(source, mode="fast")
        session.load_trace(trace, FS_HZ, recording_id="expt/MUnit_1/roi=0", label="ROI 0", source_path=source)
        session.load_trace(trace, FS_HZ, recording_id="expt/MUnit_1/roi=1", label="ROI 1", source_path=source)
        assert calls == ["expt/MUnit_1/roi=0", "expt/MUnit_1/roi=1"]
        session.load_trace(trace, FS_HZ, recording_id="expt/MUnit_1/roi=0", label="ROI 0", source_path=source)
        assert calls == ["expt/MUnit_1/roi=0", "expt/MUnit_1/roi=1"]
        assert session.cache_status.startswith("loaded cache")
        assert len(list((tmp_path / ".curation" / "cache").glob("*.npz"))) == 2

    def test_rejects_bad_traces(self, tmp_path):
        from mbo_utilities.vnoiser import CurationSession

        source = _source(tmp_path)
        session = CurationSession(source, mode="fast")
        with pytest.raises(ValueError):
            session.load_trace([1.0], FS_HZ, recording_id="r", label="r", source_path=source)
        with pytest.raises(ValueError):
            session.load_trace([1.0, np.nan, 2.0], FS_HZ, recording_id="r", label="r", source_path=source)

    def test_widget_loads_a_trace_off_the_frame_and_reports_focus(self, curation, tmp_path, monkeypatch):
        _fake_pipeline(monkeypatch)
        source = _source(tmp_path)
        focused = []
        curation.on_focus = focused.append
        trace = np.random.default_rng(2).normal(size=3000)
        curation.load_trace(trace, FS_HZ, recording_id="expt/MUnit_1/roi=2", label="ROI 2", source_path=source)
        curation.wait(60)
        assert curation.session.loaded and curation.session.n == 3
        assert curation.current == "expt/MUnit_1/roi=2"
        assert [r.rid for r in curation.catalog] == ["expt/MUnit_1/roi=2"]
        _frames(curation, 2)
        assert focused and abs(focused[-1] - 700 / FS_HZ) < 0.01
        curation.session.select(2)
        _frames(curation, 2)
        assert abs(focused[-1] - 2300 / FS_HZ) < 0.01
        _frames(curation, 2)
        # a mode switch replays the same trace into the new mode's session
        curation.set_mode("slow")
        curation.wait(60)
        assert curation.session.mode == "slow"
        assert curation.session.recording_id == "expt/MUnit_1/roi=2"
        _frames(curation, 2)


ASAKO_MESC = next(
    (
        p
        for p in (
            "X:/data/asako/stan112/stan112_expt12/stan112_expt12/stan112_expt12.mesc",
            "C:/Users/flynn/repos/vnoiser/data/stan112/stan112_expt12/stan112_expt12/stan112_expt12.mesc",
        )
        if __import__("pathlib").Path(p).exists()
    ),
    "X:/data/asako/stan112/stan112_expt12/stan112_expt12/stan112_expt12.mesc",
)


class TestSeparateZstackFile:
    @pytest.mark.skipif(not __import__("pathlib").Path(ASAKO_MESC).exists(), reason="local data only")
    def test_stack_in_a_sibling_file_pairs_with_the_lines(self):
        from pathlib import Path

        from mbo_utilities.analysis.linescan import pair_reference_zstack, zstack_candidates
        from mbo_utilities.arrays.mesc import list_mesc_units

        mesc = Path(ASAKO_MESC)
        zstack = mesc.parent.parent / f"{mesc.stem}_zstack.mesc"
        units = list_mesc_units(zstack)
        cands = zstack_candidates(mesc, "MSession_0/MUnit_35", units, zstack_path=zstack)
        assert [c["munit"] for c in cands] == ["MUnit_3"]
        assert cands[0]["xy_fraction"] == 1.0 and cands[0]["z_fraction"] == 1.0
        paired = pair_reference_zstack(mesc, "MSession_0/MUnit_35", units, zstack_path=zstack)
        assert paired["munit"] == "MUnit_3"
        assert zstack_candidates(mesc, "MSession_0/MUnit_35") == []

    @pytest.mark.skipif(not __import__("pathlib").Path(ASAKO_MESC).exists(), reason="local data only")
    def test_mbo_recognises_the_line_scan_unit(self):
        from mbo_utilities.gui.run_gui import _is_linescan_unit, _mesc_unit

        assert _is_linescan_unit(ASAKO_MESC, "MUnit_35")
        assert _is_linescan_unit(ASAKO_MESC, "MSession_0/MUnit_38")
        assert not _is_linescan_unit(ASAKO_MESC, "MUnit_0")
        assert not _is_linescan_unit(ASAKO_MESC, None)
        assert _mesc_unit(ASAKO_MESC, "MUnit_35")["key"] == "MSession_0/MUnit_35"

    @pytest.mark.skipif(not __import__("pathlib").Path(ASAKO_MESC).exists(), reason="local data only")
    def test_dry_run_of_the_viewer_entry_point(self, capsys):
        from mbo_utilities.gui.linescan_viewer import open_linescan_viewer

        assert open_linescan_viewer(ASAKO_MESC, ref_key="MUnit_35", dry_run=True) is None
        out = capsys.readouterr().out
        assert "1 zstack in stan112_expt12_zstack.mesc" in out
        assert "Z-stack: MSession_0/MUnit_3" in out
        assert "24 line(s) on MUnit_3" in out


def _mesc_beside_pf(tmp_path):
    """Asako's layout: <animal>/<expt>/<expt>/<expt>.mesc next to <animal>/<expt>/PF."""
    pf_dir = _write_spatial_recording(tmp_path)
    scan_dir = pf_dir.parent / "stan1_expt1"
    scan_dir.mkdir()
    mesc = scan_dir / "stan1_expt1.mesc"
    mesc.write_bytes(b"x")
    return mesc, pf_dir


class TestPfForMesc:
    def test_pf_folder_and_scan_are_found_beside_the_mesc(self, tmp_path):
        from mbo_utilities.vnoiser import pf_dir_for_mesc, pf_scan_for_mesc

        mesc, pf_dir = _mesc_beside_pf(tmp_path)
        assert pf_dir_for_mesc(mesc) == pf_dir
        scan = pf_scan_for_mesc(mesc, "MSession_0/MUnit_10")
        assert scan is not None
        assert scan.scan_id == "10" and scan.domains == {"soma": [0]}
        assert scan.domain_for_roi(0) == "soma" and scan.domain_for_roi(7) is None
        assert scan.recording_id("soma") == "stan1/stan1_expt1/scan=10/domain=soma"

    def test_unprocessed_scan_or_missing_pf_gives_none(self, tmp_path):
        from mbo_utilities.vnoiser import pf_dir_for_mesc, pf_scan_for_mesc

        mesc, _pf_dir = _mesc_beside_pf(tmp_path)
        assert pf_scan_for_mesc(mesc, "MUnit_99") is None
        lone = tmp_path / "elsewhere" / "scan.mesc"
        lone.parent.mkdir()
        lone.write_bytes(b"x")
        assert pf_dir_for_mesc(lone) is None
        assert pf_scan_for_mesc(lone, "MUnit_10") is None

    def test_domain_trace_loads_by_its_recording_id(self, tmp_path):
        from mbo_utilities.vnoiser import CurationSession, pf_scan_for_mesc

        mesc, pf_dir = _mesc_beside_pf(tmp_path)
        scan = pf_scan_for_mesc(mesc, "MUnit_10")
        session = CurationSession(pf_dir, mode="fast")
        assert not session.hierarchical
        session.load(scan.recording_id("soma"))
        assert session.loaded and session.n == 3

    def test_picker_pointed_at_a_mesc_redirects_to_its_pf(self, curation, data_root, tmp_path):
        # the curation fixture already wrote the PF folder under data_root
        pf_dir = data_root / "stan1" / "stan1_expt1" / "PF"
        scan_dir = pf_dir.parent / "stan1_expt1"
        scan_dir.mkdir()
        mesc = scan_dir / "stan1_expt1.mesc"
        mesc.write_bytes(b"x")
        curation.scan(mesc)
        assert curation.data_path == str(pf_dir)
        assert "PF folder of" in curation.status
        assert curation.catalog
        curation.wait(60)
        lone = tmp_path / "elsewhere" / "scan.mesc"
        lone.parent.mkdir()
        lone.write_bytes(b"x")
        curation.scan(lone)
        assert curation.data_path == ""
        assert "raw line scan" in curation.status
        _frames(curation)


class TestMboOpensTheLineScanViewer:
    """``mbo scan.mesc`` opens the image viewer on the file's first line-scan
    unit with no prompt (or on the line-scan unit picked); the curation and
    the Voltage pipeline follow the unit on screen. Other units and files
    still prompt once and go to the image viewer."""

    @staticmethod
    def _units(monkeypatch, kinds):
        from mbo_utilities.gui import run_gui as rg

        units = [
            {"key": f"MSession_0/MUnit_{i}", "munit": f"MUnit_{i}", "kind": kind}
            for i, kind in enumerate(kinds)
        ]
        monkeypatch.setattr("mbo_utilities.arrays.mesc.list_mesc_units", lambda p: units)
        return rg

    def test_a_file_with_line_scans_opens_the_viewer_with_no_prompt(self, tmp_path, monkeypatch):
        rg = self._units(monkeypatch, ["multicube", "frames", "packed", "packed"])
        mesc = tmp_path / "scan.mesc"
        mesc.write_bytes(b"x")
        standard = []
        monkeypatch.setattr(
            rg, "_resolve_mesc_unit",
            lambda p, u: ({"unit": u}, True) if u is not None else pytest.fail("prompted"),
        )
        monkeypatch.setattr(rg, "_launch_standard_viewer", lambda *a, **k: standard.append(a))
        rg._run_gui_impl(data_in=mesc)
        assert len(standard) == 1
        assert standard[0][0] == mesc and standard[0][-1] == "MSession_0/MUnit_2"

    def test_a_file_with_chessboard_patches_opens_the_same_way(self, tmp_path, monkeypatch):
        rg = self._units(monkeypatch, ["frames", "tiled", "packed"])
        mesc = tmp_path / "chess.mesc"
        mesc.write_bytes(b"x")
        standard = []
        monkeypatch.setattr(
            rg, "_resolve_mesc_unit",
            lambda p, u: ({"unit": u}, True) if u is not None else pytest.fail("prompted"),
        )
        monkeypatch.setattr(rg, "_launch_standard_viewer", lambda *a, **k: standard.append(a))
        rg._run_gui_impl(data_in=mesc)
        assert len(standard) == 1 and standard[0][-1] == "MSession_0/MUnit_1"

    def test_an_explicit_line_scan_unit_goes_to_the_viewer(self, tmp_path, monkeypatch):
        rg = self._units(monkeypatch, ["multicube", "packed"])
        mesc = tmp_path / "scan.mesc"
        mesc.write_bytes(b"x")
        standard = []
        monkeypatch.setattr(rg, "_launch_standard_viewer", lambda *a, **k: standard.append(a))
        rg._run_gui_impl(data_in=mesc, unit="MUnit_1")
        assert len(standard) == 1 and standard[0][-1] == "MUnit_1"
        # a non line-scan unit of the same file still goes to the image viewer
        standard = []
        monkeypatch.setattr(rg, "_resolve_mesc_unit", lambda p, u: ({"unit": u}, True))
        monkeypatch.setattr(rg, "_launch_standard_viewer", lambda *a, **k: standard.append(a))
        rg._run_gui_impl(data_in=mesc, unit="MUnit_0")
        assert standard and standard[0][-1] == "MUnit_0"

    def test_a_file_without_line_scans_prompts_once_and_opens_the_image_viewer(self, tmp_path, monkeypatch):
        rg = self._units(monkeypatch, ["multicube", "frames"])
        mesc = tmp_path / "scan.mesc"
        mesc.write_bytes(b"x")
        prompts, standard = [], []

        def resolve(p, u):
            prompts.append(u)
            return {"unit": "MSession_0/MUnit_0"}, True

        monkeypatch.setattr(rg, "_resolve_mesc_unit", resolve)
        monkeypatch.setattr(rg, "_launch_standard_viewer", lambda *a, **k: standard.append(a))
        rg._run_gui_impl(data_in=mesc)
        assert prompts == [None]
        assert standard and standard[0][-1] == "MSession_0/MUnit_0"

    def test_cancelled_picker_opens_nothing(self, tmp_path, monkeypatch):
        rg = self._units(monkeypatch, ["multicube", "frames"])
        mesc = tmp_path / "scan.mesc"
        mesc.write_bytes(b"x")
        monkeypatch.setattr(rg, "_resolve_mesc_unit", lambda p, u: ({}, False))
        monkeypatch.setattr(rg, "_launch_standard_viewer", lambda *a, **k: pytest.fail("opened"))
        assert rg._run_gui_impl(data_in=mesc) is None

    def test_linescan_command_view_flag(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        from mbo_utilities import cli
        from mbo_utilities.gui import linescan_viewer

        mesc = tmp_path / "scan.mesc"
        mesc.write_bytes(b"x")
        calls = []
        monkeypatch.setattr(linescan_viewer, "open_linescan_viewer", lambda p, **k: calls.append((p, k)))
        result = CliRunner().invoke(cli.main, ["linescan", str(mesc), "--view", "--unit", "MUnit_35"])
        assert result.exit_code == 0, result.output
        assert calls[0][1]["ref_key"] == "MUnit_35"
        assert calls[0][1]["zstack_key"] is None


# ----------------------------------------------------------------------
# the notebook's experiment folder opens the line-scan viewer
# ----------------------------------------------------------------------


def _experiment_layout(root, name="stan1_expt1", animal="stan1"):
    """``<animal>/<expt>/<expt>/<expt>.mesc`` with ``PF`` and the Z-stack beside it."""
    experiment = root / animal / name
    scan_dir = experiment / name
    scan_dir.mkdir(parents=True)
    (experiment / "PF").mkdir()
    mesc = scan_dir / f"{name}.mesc"
    mesc.write_bytes(b"x")
    (experiment / f"{name}_zstack.mesc").write_bytes(b"z")
    return experiment, mesc


class TestExperimentFolder:
    def test_experiment_pf_and_inner_folders_resolve_to_the_line_scan(self, tmp_path):
        from mbo_utilities.analysis.linescan import experiment_linescan_mesc

        experiment, mesc = _experiment_layout(tmp_path)
        for path in (experiment, experiment / "PF", experiment / experiment.name, mesc):
            assert experiment_linescan_mesc(path) == mesc, path
        # the Z-stack file is never the line scan, even when it is the only match
        (mesc).unlink()
        assert experiment_linescan_mesc(experiment) is None
        (experiment / experiment.name / "other_scan.mesc").write_bytes(b"x")
        assert experiment_linescan_mesc(experiment).name == "other_scan.mesc"

    def test_other_folders_and_files_give_none(self, tmp_path):
        from mbo_utilities.analysis.linescan import experiment_linescan_mesc

        _experiment_layout(tmp_path)
        assert experiment_linescan_mesc(tmp_path) is None  # the Data root
        assert experiment_linescan_mesc(tmp_path / "stan1") is None  # the animal
        tif = tmp_path / "movie.tif"
        tif.write_bytes(b"x")
        assert experiment_linescan_mesc(tif) is None
        assert experiment_linescan_mesc(tmp_path / "missing") is None

    def test_mbo_opens_the_experiment_folder_in_the_image_viewer(self, tmp_path, monkeypatch):
        rg = TestMboOpensTheLineScanViewer._units(monkeypatch, ["frames", "packed"])
        experiment, _mesc = _experiment_layout(tmp_path)
        standard = []
        monkeypatch.setattr(rg, "_launch_standard_viewer", lambda *a, **k: standard.append(a))
        rg._run_gui_impl(data_in=experiment)
        rg._run_gui_impl(data_in=str(experiment / "PF"))
        # the folder itself is handed over: imread opens it as a PfArray
        assert [a[0] for a in standard] == [experiment, str(experiment / "PF")]

    def test_a_folder_without_line_scans_falls_through(self, tmp_path, monkeypatch):
        rg = TestMboOpensTheLineScanViewer._units(monkeypatch, ["frames"])
        experiment, _mesc = _experiment_layout(tmp_path)
        standard = []
        monkeypatch.setattr(rg, "_launch_standard_viewer", lambda *a, **k: standard.append(a))
        rg._run_gui_impl(data_in=experiment)
        assert standard and standard[0][0] == experiment

    def test_linescan_command_takes_the_folder(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        from mbo_utilities import cli
        from mbo_utilities.gui import linescan_viewer

        experiment, mesc = _experiment_layout(tmp_path)
        calls = []
        monkeypatch.setattr(linescan_viewer, "open_linescan_viewer", lambda p, **k: calls.append((p, k)))
        result = CliRunner().invoke(cli.main, ["linescan", str(experiment), "--view"])
        assert result.exit_code == 0, result.output
        assert calls[0][0] == str(mesc)
        result = CliRunner().invoke(cli.main, ["linescan", str(tmp_path), "--view"])
        assert result.exit_code != 0 and "no <name>/<name>.mesc" in result.output


# ----------------------------------------------------------------------
# the standalone curation window
# ----------------------------------------------------------------------


class TestCurationWindow:
    """``gui/curation_viewer.py``: the dashboard in a hello_imgui window,
    no figure. Building the app and its catalog needs no window."""

    def test_panel_host_stands_in_for_the_strip(self):
        from mbo_utilities.gui._top_strip import TopPanel
        from mbo_utilities.gui.curation_viewer import PanelHost

        host = PanelHost()
        calls = []
        host.add_hook(calls.append)
        host.register(TopPanel("curation", "Curation", lambda: None, 100))
        assert host.has("curation") and host.active == "curation"
        assert host.panel("curation").label == "Curation"
        host.unregister("curation")
        assert not host.has("curation") and host.active is None
        host.remove_hook(calls.append)
        assert host.hooks == []

    def test_opens_the_notebook_data_path(self, data_root):
        from mbo_utilities.gui.curation_viewer import open_curation_viewer

        app = open_curation_viewer(data_root, run=False)
        try:
            assert app.host.has("curation")
            assert [r.rid for r in app.widget.catalog] == ["stan1/stan1_expt1/scan=10/domain=soma"]
            app.widget.wait(60)
            assert app.widget.session is not None and app.widget.session.loaded
            assert app.title.endswith(data_root.name)
        finally:
            app.widget.close()

    def test_a_raw_mesc_lists_every_line_for_the_denoiser(self, tmp_path, monkeypatch):
        from mbo_utilities.gui import curation_viewer

        mesc = tmp_path / "scan.mesc"
        mesc.write_bytes(b"x")
        traces = np.random.default_rng(0).random((3, 4000))
        from mbo_utilities.gui import event_curation

        monkeypatch.setattr(
            event_curation, "raw_linescan_traces",
            lambda p, channel=0, traces_dir=None: [
                {"key": "MSession_0/MUnit_35", "munit": "MUnit_35", "fs": FS_HZ, "traces": traces},
            ],
        )
        calls = _fake_pipeline(monkeypatch)
        app = curation_viewer.open_curation_viewer(mesc, run=False)
        try:
            widget = app.widget
            assert [r.rid for r in widget.catalog] == [f"scan/MUnit_35/roi={i}" for i in range(3)]
            assert not any(r.pre_denoised for r in widget.catalog)
            assert widget.session is None and "3 raw ROI traces" in widget.status
            # nothing runs until a recording is picked; then the denoiser does
            assert calls == []
            widget.load("scan/MUnit_35/roi=1")
            widget.wait(60)
            assert calls == ["scan/MUnit_35/roi=1"]
            assert widget.session is not None and widget.session.n == 3
            assert (mesc.parent / ".curation").is_dir()
        finally:
            widget.close()

    def test_figure_host_draws_the_dashboard_offscreen(self, data_root):
        """The notebook route: the dashboard as a fastplotlib figure's top
        window (jupyter_rfb in a real notebook, offscreen here)."""
        if not _offscreen_selected():
            pytest.skip("needs the offscreen rendercanvas")
        from mbo_utilities.gui.curation_viewer import CurationVis, open_curation_viewer

        vis = open_curation_viewer(data_root, figure=True, size=(900, 700))
        assert isinstance(vis, CurationVis)
        try:
            errors: list[str] = []
            panel = vis.host.panel("curation")
            draw = panel.draw

            def body():
                try:
                    draw()
                except Exception:
                    errors.append(traceback.format_exc())

            panel.draw = body
            vis.show()
            vis.widget.wait(60)
            assert vis.widget.session is not None and vis.widget.session.loaded
            for _ in range(3):
                vis.figure.canvas.draw()
            assert not errors, errors[0]
            # the window takes the canvas but leaves the renderer a viewport
            height = vis.figure.canvas.get_logical_size()[1]
            assert 0 < vis.window.size < height
            assert vis.figure[0, 0].viewport.rect[3] >= 1
        finally:
            vis.close()

    def test_add_trace_lists_without_loading(self, curation, tmp_path):
        rec = curation.add_trace(
            np.zeros(100), FS_HZ, recording_id="x/roi=0", label="ROI 0", source_path=tmp_path / "a.mesc",
        )
        assert rec in curation.catalog and not rec.pre_denoised
        assert not curation.loading and curation.session is None
        assert [r.rid for r in curation.loadable()] == ["x/roi=0"]


class TestClearLabels:
    def test_clear_all_hands_every_candidate_back_to_the_rules(self, data_root):
        session = _loaded(data_root, mode="fast")
        session.set_labels([0, 1], "no")
        assert session.counts()[1] == 2
        assert session.labels()[0] == "no"
        assert session.clear_labels() == 2
        assert session.counts() == (0, 0, session.n)
        assert "no" not in session.labels() and "yes" not in session.labels()
        assert session.clear_labels() == 0
        saved = json.loads(session.label_path.read_text(encoding="utf-8"))
        assert saved["events"] == {}
        assert saved["candidate_detection"]["thresholds"]


class TestLoadingLine:
    def test_loading_line_names_the_active_job(self, tmp_path):
        from mbo_utilities.gui.curation_viewer import _Dashboard

        dash = _Dashboard(None)
        widget = dash.widget
        assert not hasattr(widget, "pipeline_mesc")
        assert widget.loading_line() == ""
        widget._busy.add(("fast", "x"))
        assert widget.loading_line() == "1 queued"
        widget._active = (("fast", "x"), "MUnit_35 ROI 0", 0.0)
        widget._trace_sources["x"] = {}
        assert widget.loading_line().startswith("denoising MUnit_35 ROI 0 · ")
        widget.close()


class TestOpenArray:
    def test_curation_source_of_each_kind(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.arrays.pf import PfArray
        from mbo_utilities.gui.event_curation import curation_source

        pf_dir = _write_spatial_recording(tmp_path)
        assert curation_source(PfArray(pf_dir)) == "pf"
        assert curation_source(NumpyArray(np.zeros((2, 4, 4), dtype=np.float32))) == ""
        assert curation_source(None) == ""
        mesc, _pf = _mesc_beside_pf(tmp_path / "beside")
        unit = SimpleNamespace(metadata={"mesc_layout": "packed"}, filenames=[mesc], unit_key="MSession_0/MUnit_10")
        assert curation_source(unit) == "pf"
        lone = tmp_path / "elsewhere" / "scan.mesc"
        lone.parent.mkdir()
        lone.write_bytes(b"x")
        unit.filenames = [lone]
        assert curation_source(unit) == "raw"
        # chessboard patches and ribbon boxes are ROIs too; a plain frame series is not
        unit.metadata = {"mesc_layout": "tiled"}
        assert curation_source(unit) == "raw"
        unit.metadata = {"mesc_layout": "boxes"}
        assert curation_source(unit) == "raw"
        unit.metadata = {"mesc_layout": "frames"}
        assert curation_source(unit) == ""

    def test_a_pf_array_scans_its_folder_and_selects_its_scan(self, curation, data_root):
        from mbo_utilities.arrays.pf import PfArray

        pf_dir = data_root / "stan1" / "stan1_expt1" / "PF"
        assert curation.open_array(PfArray(pf_dir)) == "pf"
        assert curation.data_path == str(pf_dir)
        assert curation.scope is None
        assert [r.rid for r in curation.shown] == ["stan1/stan1_expt1/scan=10/domain=soma"]
        assert curation.current == "stan1/stan1_expt1/scan=10/domain=soma"
        curation.wait(60)
        assert curation.session is not None and curation.session.loaded
        _frames(curation)

    def test_a_unit_beside_a_pf_folder_shows_every_scan_and_follows_the_unit(self, curation, tmp_path):
        """The viewer shows one unit at a time; the curation lists every scan the pipeline wrote
        and moves its selection with the unit, without rescanning the folder."""
        from types import SimpleNamespace

        mesc, pf = _mesc_beside_pf(tmp_path / "beside")
        traces = pickle.loads((pf / "denoised_trace_scans.pkl").read_bytes())
        traces["20"] = {"soma": traces["10"]["soma"]}
        (pf / "denoised_trace_scans.pkl").write_bytes(pickle.dumps(traces))
        fs = pickle.loads((pf / "fs_scans.pkl").read_bytes())
        fs["20"] = fs["10"]
        (pf / "fs_scans.pkl").write_bytes(pickle.dumps(fs))
        meta = pickle.loads((pf / "scanIDs_ROIs.pkl").read_bytes())
        meta["scanID_spatial"] = np.array([10, 20])
        (pf / "scanIDs_ROIs.pkl").write_bytes(pickle.dumps(meta))

        unit = SimpleNamespace(metadata={"mesc_layout": "packed"}, filenames=[mesc], unit_key="MSession_0/MUnit_20")
        assert curation.open_array(unit) == "pf"
        assert curation.data_path == str(pf)
        assert sorted(r.rid for r in curation.shown) == [
            "stan1/stan1_expt1/scan=10/domain=soma", "stan1/stan1_expt1/scan=20/domain=soma",
        ]
        assert curation.current == "stan1/stan1_expt1/scan=20/domain=soma"
        catalog = curation.catalog
        unit.unit_key = "MSession_0/MUnit_10"
        assert curation.open_array(unit) == "pf"
        assert curation.current == "stan1/stan1_expt1/scan=10/domain=soma"
        assert curation.catalog is catalog
        # a unit the pipeline never processed keeps every scan on show and says so
        unit.unit_key = "MSession_0/MUnit_2"
        assert curation.open_array(unit) == "pf"
        assert len(curation.shown) == 2 and "scan 2 is not in" in curation.status
        _frames(curation)

    def test_a_raw_line_scan_lists_its_lines_scoped_to_the_unit(self, curation, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from mbo_utilities.gui import event_curation

        mesc = tmp_path / "scan.mesc"
        mesc.write_bytes(b"x")
        traces = np.random.default_rng(0).random((2, 4000))
        monkeypatch.setattr(
            event_curation, "raw_linescan_traces",
            lambda p, channel=0, traces_dir=None: [
                {"key": "MSession_0/MUnit_35", "munit": "MUnit_35", "fs": FS_HZ, "traces": traces},
                {"key": "MSession_0/MUnit_38", "munit": "MUnit_38", "fs": FS_HZ, "traces": traces},
            ],
        )
        unit = SimpleNamespace(metadata={"mesc_layout": "packed"}, filenames=[mesc], unit_key="MSession_0/MUnit_38")
        assert curation.open_array(unit) == "raw"
        assert len(curation.catalog) == 4
        assert [r.rid for r in curation.shown] == ["scan/MUnit_38/roi=0", "scan/MUnit_38/roi=1"]
        assert "4 raw ROI traces" in curation.status
        _frames(curation)

    def test_the_viewer_turns_the_curation_on_for_a_pf_folder(self, data_root):
        if not _offscreen_selected():
            pytest.skip("needs the offscreen rendercanvas")
        from mbo_utilities.arrays.pf import PfArray
        from mbo_utilities.gui.data_vis import DataVis

        pf_dir = data_root / "stan1" / "stan1_expt1" / "PF"
        vis = DataVis(PfArray(pf_dir), size=FIGURE_SIZE)
        vis.show()
        try:
            parent = vis.widget
            widget = parent.event_curation
            assert widget is not None and widget.data_path == str(pf_dir)
            assert parent.top_strip.has("curation")
            widget.wait(60)
            for _ in range(2):
                vis.figure.canvas.draw()
        finally:
            vis.close()
