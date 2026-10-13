from pathlib import Path

import pytest
from click.testing import CliRunner
from mbo_utilities.cli import main
from mbo_utilities.masknmf.params import Asap7Settings


class TestAsap7Settings:
    def test_defaults_are_the_detrend_notebook(self):
        s = Asap7Settings()
        assert (s.channel, s.first_frame) == (1, 200)
        assert s.reference.block_sizes == (4, 4) and not s.reference.denoise
        assert (s.reference.spatial_avg_factor, s.reference.temporal_avg_factor) == (4, 2)
        assert s.registration.max_shifts == (40, 40)
        assert s.compression.denoise and s.compression.detrend
        assert (s.compression.block_sizes, s.compression.temporal_avg_factor) == ((10, 10), 10)
        assert (s.detrend_window_seconds, s.mad_correlation_threshold, s.demixing_passes) == (0.3, 0.6, 2)

    def test_round_trip(self):
        s = Asap7Settings(channel=2, first_frame=0, detrend_window_seconds=0.5)
        s.registration.max_shifts = (5, 5)
        assert Asap7Settings.from_dict(s.to_dict()) == s

    def test_partial_dict_keeps_stage_defaults(self):
        back = Asap7Settings.from_dict({"compression": {"block_sizes": [8, 8]}})
        assert back.compression.block_sizes == (8, 8)
        assert back.compression.temporal_avg_factor == 10
        assert back.reference == Asap7Settings().reference


def test_default_save_path_is_beside_the_source():
    from mbo_utilities.masknmf.asap7 import default_save_path

    src = Path("/data/todd/scan.mesc")
    assert default_save_path(src, "MSession_0/MUnit_10") == Path("/data/todd/masknmf_asap7/MSession_0_MUnit_10_detrend")
    assert default_save_path(Path("/data/movie.tif"), None) == Path("/data/masknmf_asap7/movie_detrend")


def test_detrender_refuses_a_movie_shorter_than_two_windows():
    pytest.importorskip("masknmf")
    from mbo_utilities.masknmf.asap7 import asap7_detrender

    with pytest.raises(ValueError, match="under twice the 0.3 s detrend window"):
        asap7_detrender(n_frames=500, fs=1000.0, window_seconds=0.3, device="cpu")


def test_cli_lists_the_units_when_none_is_given(tmp_path, monkeypatch):
    path = tmp_path / "scan.mesc"
    path.write_bytes(b"")
    listed = []
    monkeypatch.setattr("mbo_utilities.cli._echo_mesc_units", lambda p, unit: listed.append(p) or {})
    result = CliRunner().invoke(main, ["asap7", str(path)])
    assert result.exit_code != 0
    assert listed == [str(path)] and "--unit" in result.output


def asap7_run(tmp_path, data):
    """A run folder as run_asap7 leaves it, over a tiff source, without the masknmf files' contents."""
    import json

    import tifffile
    from mbo_utilities.masknmf.params import ASAP7_RESULTS_FILE, ASAP7_SETTINGS_FILE

    source = tmp_path / "rec.tif"
    tifffile.imwrite(source, data)
    run = tmp_path / "masknmf_asap7" / "rec_detrend" / "20261013T120000_glutamate-calcium-spine"
    run.mkdir(parents=True)
    (run / ASAP7_RESULTS_FILE).write_bytes(b"")
    settings = Asap7Settings(first_frame=4).to_dict()
    provenance = {"source": {"path": str(source), "reader_kwargs": {}}, "n_frames": len(data) - 4, "fs": 10.0, "settings": settings}
    (run / ASAP7_SETTINGS_FILE).write_text(json.dumps(provenance))
    return run


def test_a_run_folder_reopens_its_inverted_movie_and_frame_times(tmp_path):
    import numpy as np
    from mbo_utilities.masknmf.asap7 import is_asap7_run, run_frame_times, run_raw_movie, run_registration, run_rtmc

    data = (100 + np.random.default_rng(0).random((12, 5, 6)) * 50).astype(np.int16)
    run = asap7_run(tmp_path, data)
    assert is_asap7_run(run) and not is_asap7_run(run.parent)
    raw = data[4:].astype(np.float32)
    np.testing.assert_allclose(run_raw_movie(run), 2 * raw.mean(axis=0) - raw, rtol=1e-5)
    np.testing.assert_allclose(run_frame_times(run), np.arange(4, 12) / 10.0)
    assert run_registration(run) is None
    assert run_rtmc(run) is None
