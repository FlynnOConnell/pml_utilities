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
        assert curation.data_path == str(source)
        _frames(curation, 2)
        assert focused and abs(focused[-1] - 700 / FS_HZ) < 0.01
        curation.session.select(2)
        _frames(curation, 2)
        assert abs(focused[-1] - 2300 / FS_HZ) < 0.01
        curation.strip.focus("candidates")
        _frames(curation, 2)
        # a mode switch replays the same trace into the new mode's session
        curation.set_mode("slow")
        curation.wait(60)
        assert curation.session.mode == "slow"
        assert curation.session.recording_id == "expt/MUnit_1/roi=2"
        _frames(curation, 2)


ASAKO_MESC = (
    "C:/Users/flynn/repos/vnoiser/data/stan112/stan112_expt12/stan112_expt12/stan112_expt12.mesc"
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


class TestMboOpensTheLineScanViewer:
    """``mbo scan.mesc`` hands a picked line-scan unit to the viewer; other
    units and files still go to the image viewer."""

    def test_line_scan_unit_goes_to_the_viewer(self, tmp_path, monkeypatch):
        from mbo_utilities.gui import run_gui as rg

        mesc = tmp_path / "scan.mesc"
        mesc.write_bytes(b"x")
        opened, standard = [], []
        monkeypatch.setattr(rg, "_resolve_mesc_unit", lambda p, u: ({"unit": "MSession_0/MUnit_35"}, True))
        monkeypatch.setattr(rg, "_is_linescan_unit", lambda p, u: u == "MSession_0/MUnit_35")
        monkeypatch.setattr(rg, "_launch_linescan_viewer", lambda p, u: opened.append((p, u)))
        monkeypatch.setattr(rg, "_launch_standard_viewer", lambda *a, **k: standard.append(a))
        rg._run_gui_impl(data_in=mesc)
        assert opened == [(mesc, "MSession_0/MUnit_35")]
        assert standard == []

    def test_other_units_still_open_the_image_viewer_without_a_second_prompt(self, tmp_path, monkeypatch):
        from mbo_utilities.gui import run_gui as rg

        mesc = tmp_path / "scan.mesc"
        mesc.write_bytes(b"x")
        prompts, standard = [], []

        def resolve(p, u):
            prompts.append(u)
            return {"unit": "MSession_0/MUnit_0"}, True

        monkeypatch.setattr(rg, "_resolve_mesc_unit", resolve)
        monkeypatch.setattr(rg, "_is_linescan_unit", lambda p, u: False)
        monkeypatch.setattr(rg, "_launch_standard_viewer", lambda *a, **k: standard.append(a))
        rg._run_gui_impl(data_in=mesc)
        assert prompts == [None]
        assert standard and standard[0][-1] == "MSession_0/MUnit_0"

    def test_cancelled_picker_opens_nothing(self, tmp_path, monkeypatch):
        from mbo_utilities.gui import run_gui as rg

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
