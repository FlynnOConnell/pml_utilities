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
    widget.scan(data_root)
    session = widget.session
    widget.select_experiment(session.experiments(widget.animal)[0][1])
    widget.load(session.recordings[0][1])
    widget.wait(60)
    assert session.loaded, widget.status


class TestWidget:
    def test_registers_two_top_panels(self, curation):
        assert curation.strip.has("curation") and curation.strip.has("candidates")
        assert {p.right_tab for p in curation.strip.panels} == {"curation"}
        _frames(curation)

    def test_scan_picks_the_only_animal(self, curation, data_root):
        curation.scan(data_root)
        assert curation.animal.endswith("stan1")
        assert curation.session.hierarchical
        _frames(curation)

    def test_bad_path_reports_instead_of_raising(self, curation, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        curation.scan(empty)
        assert curation.session is not None
        assert not curation.session.has_dataset
        assert curation.session.recordings == []
        assert "no vnoiser data" in curation.status
        _frames(curation)

    def test_load_runs_off_the_frame_and_draws_every_panel(self, curation, data_root):
        _load(curation, data_root)
        assert curation.session.n == 3
        _frames(curation, 3)
        curation.strip.focus("candidates")
        _frames(curation, 3)
        assert curation.strip.active == "candidates"

    def test_mode_switch_reloads_the_same_recording(self, curation, data_root):
        _load(curation, data_root)
        fast = curation.session
        curation.set_mode("slow")
        curation.wait(60)
        slow = curation.session
        assert slow is not fast and slow.mode == "slow"
        assert slow.recording_id == fast.recording_id
        _frames(curation, 2)
        curation.strip.focus("candidates")
        _frames(curation, 2)

    def test_labels_and_filters_redraw(self, curation, data_root):
        _load(curation, data_root)
        session = curation.session
        session.select(1)
        session.set_label("yes")
        session.set_view_filter("yes")
        _frames(curation, 2)
        curation.strip.focus("candidates")
        _frames(curation, 2)
        assert session.counts() == (1, 0, 2)

    def test_close_gives_the_strip_back(self, curation):
        strip = curation.strip
        curation.close()
        assert not strip.has("curation") and not strip.has("candidates")
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
