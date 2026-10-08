"""masknmf integration tests: params round-trip, stage gating, suite2p-shaped
output conversion. No masknmf install required — the compute stages are only
exercised when the package is present (the denoised-reference registration run).
"""

import json

import numpy as np
import pytest
from mbo_utilities.masknmf import outputs
from mbo_utilities.masknmf.params import (
    STAGE_FORCE,
    STAGE_RUN,
    STAGE_SKIP,
    MasknmfSettings,
    stage_action,
)


def test_settings_roundtrip_json():
    s = MasknmfSettings()
    s.registration.strategy = "pwrigid"
    s.registration.max_shifts = (7, 9)
    s.demixing.do_demixing = STAGE_FORCE
    d = json.loads(json.dumps(s.to_dict()))
    restored = MasknmfSettings.from_dict(d)
    assert restored.registration.strategy == "pwrigid"
    assert restored.registration.max_shifts == (7, 9)
    assert restored.demixing.do_demixing == STAGE_FORCE


def test_settings_from_dict_ignores_unknown_keys():
    restored = MasknmfSettings.from_dict(
        {"registration": {"strategy": "rigid", "bogus": 1}, "extra_section": {}}
    )
    assert restored.registration.strategy == "rigid"


def test_strategy_kwargs_by_strategy():
    s = MasknmfSettings()
    assert set(s.registration.strategy_kwargs()) == {"max_shifts"}
    s.registration.strategy = "pwrigid"
    assert "minimum_patch_sizes" in s.registration.strategy_kwargs()
    assert "max_rigid_shifts" in s.registration.strategy_kwargs()


def test_nmf_kwargs_detrender_never_tuple():
    # upstream NMFConfig has a buggy `None,` detrender default; ours must
    # always pass a real None through
    kw = MasknmfSettings().demixing.nmf_kwargs(0.8, ring=True)
    assert kw["detrender"] is None
    assert kw["ring_model_start_pt"] == 0
    # matches the reference NMFConfig default, not demix()'s False default
    assert kw["reassign_background"] is True
    kw = MasknmfSettings().demixing.nmf_kwargs(0.8, ring=False)
    assert kw["ring_model_start_pt"] is None


@pytest.mark.parametrize(
    "tri,cached,expected",
    [
        (STAGE_SKIP, False, "skip"),
        (STAGE_SKIP, True, "skip"),
        (STAGE_RUN, False, "compute"),
        (STAGE_RUN, True, "reuse"),
        (STAGE_FORCE, False, "compute"),
        (STAGE_FORCE, True, "compute"),
    ],
)
def test_stage_action(tri, cached, expected):
    assert stage_action(tri, cached) == expected


def _toy_footprints():
    # two ROIs on a 4x5 grid: roi0 = pixels {(0,0),(0,1)}, roi1 = {(2,3)}
    shape = (4, 5)
    pix = np.array([0, 1, 13])
    roi = np.array([0, 0, 1])
    vals = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    return np.stack([pix, roi]), vals, shape


def test_split_sparse_footprints():
    indices, values, _ = _toy_footprints()
    per_roi = outputs.split_sparse_footprints(indices, values, 2)
    assert len(per_roi) == 2
    np.testing.assert_array_equal(per_roi[0][0], [0, 1])
    np.testing.assert_array_equal(per_roi[1][0], [13])
    np.testing.assert_allclose(per_roi[1][1], [2.0])


def test_roi_stat_coordinates():
    indices, values, shape = _toy_footprints()
    per_roi = outputs.split_sparse_footprints(indices, values, 2)
    stat = outputs.roi_stat(*per_roi[1], shape)
    # flat index 13 in C-order on (4,5) -> y=2, x=3
    np.testing.assert_array_equal(stat["ypix"], [2])
    np.testing.assert_array_equal(stat["xpix"], [3])
    assert stat["npix"] == 1
    assert stat["med"] == (2.0, 3.0)


def test_roi_calibration_unweighted_over_support():
    """Masknmf's convention: mean over the support, not a lam-weighted mean."""
    indices, values, shape = _toy_footprints()
    per_roi = outputs.split_sparse_footprints(indices, values, 2)
    n_pix = shape[0] * shape[1]
    var_img = np.full(n_pix, 2.0, dtype=np.float32)
    mean_img = np.arange(n_pix, dtype=np.float32)
    baseline = np.ones(n_pix, dtype=np.float32)

    gain, f0 = outputs.roi_calibration(
        per_roi, var_img=var_img, mean_img=mean_img, baseline=baseline
    )
    # roi0 lam = [0.5, 1.0] on pixels [0, 1]; gain = mean(lam * 2) = 1.5
    np.testing.assert_allclose(gain[0], 1.5, rtol=1e-6)
    # roi1 lam = [2.0] on pixel 13; gain = 2.0 * 2 = 4.0
    np.testing.assert_allclose(gain[1], 4.0, rtol=1e-6)
    # F0 = mean_support(b * var_img + mean_img); roi0 -> mean([2+0, 2+1]) = 2.5
    np.testing.assert_allclose(f0[0], 2.5, rtol=1e-6)
    np.testing.assert_allclose(f0[1], 2.0 + 13.0, rtol=1e-6)


def test_roi_calibration_defaults_are_inert():
    """No PMD images -> gain is the mean lam and F0 is 0 (uncalibrated)."""
    indices, values, shape = _toy_footprints()
    per_roi = outputs.split_sparse_footprints(indices, values, 2)
    gain, f0 = outputs.roi_calibration(per_roi)
    np.testing.assert_allclose(gain[0], 0.75, rtol=1e-6)  # mean([0.5, 1.0])
    np.testing.assert_allclose(gain[1], 2.0, rtol=1e-6)
    assert (f0 == 0).all()


def test_calibrated_traces_zeroes_uncalibrated_rois():
    c = np.ones((10, 2), dtype=np.float32)
    gain = np.array([2.0, 3.0], dtype=np.float32)
    f0 = np.array([50.0, 0.0], dtype=np.float32)  # roi1 has no usable F0
    F, dff = outputs.calibrated_traces(c, gain, f0)
    np.testing.assert_allclose(F[0], 52.0, rtol=1e-6)
    np.testing.assert_allclose(F[1], 3.0, rtol=1e-6)
    np.testing.assert_allclose(dff[0], 4.0, rtol=1e-6)  # 2/50 -> 4%
    assert (dff[1] == 0).all()


def test_write_plane_outputs(tmp_path):
    indices, values, shape = _toy_footprints()
    n_pix = shape[0] * shape[1]
    # real demixed c is nonnegative (c_nonneg=True) with the baseline already
    # factored out into b, which is what makes dF/F well posed here
    c = np.vstack([np.abs(np.sin(np.linspace(0, 6, 50))), np.ones(50)]).T.astype(
        np.float32
    )
    info = outputs.write_plane_outputs(
        tmp_path,
        indices=indices,
        values=values,
        c=c,
        shape=shape,
        baseline=np.ones(n_pix, dtype=np.float32),
        var_img=np.full(n_pix, 2.0, dtype=np.float32),
        mean_img=np.full(n_pix, 100.0, dtype=np.float32),
    )
    assert info["n_rois"] == 2
    assert info["n_calibrated"] == 2
    stat = np.load(tmp_path / "stat.npy", allow_pickle=True)
    iscell = np.load(tmp_path / "iscell.npy")
    F = np.load(tmp_path / "F.npy")
    norm = np.load(tmp_path / "norm_traces.npy")
    assert len(stat) == 2
    assert iscell.shape == (2, 2)
    assert (iscell[:, 0] == 1).all()
    assert F.shape == (2, 50)
    # F is calibrated into movie units, so it sits on the F0 = 102 baseline
    # rather than hovering around 0 the way standardised c does.
    assert F.min() > 90.0
    # norm_traces is dF/F in percent against that F0, not a z-score: a
    # nonnegative trace stays nonnegative and is not centred on 0.
    assert norm.shape == (2, 50)
    assert norm.min() >= 0.0
    np.testing.assert_allclose(norm, (F - 102.0) / 102.0 * 100.0, atol=1e-4)
    for name in ("Fneu.npy", "spks.npy"):
        assert (tmp_path / name).exists()


def test_merge_ops_roundtrip(tmp_path):
    ops = outputs.merge_ops(tmp_path, {"Ly": 4, "Lx": 5})
    assert ops["Ly"] == 4
    ops = outputs.merge_ops(tmp_path, {"nframes": 50})
    assert ops["Ly"] == 4 and ops["nframes"] == 50
    assert ops["save_path"] == str(tmp_path)
    loaded = np.load(tmp_path / "ops.npy", allow_pickle=True).item()
    assert loaded["nframes"] == 50


def test_task_registered():
    pytest.importorskip("imgui_bundle")
    from mbo_utilities.gui.tasks import TASKS

    assert "masknmf" in TASKS


def test_plane_dirname():
    from mbo_utilities.masknmf.runner import generate_plane_dirname

    assert generate_plane_dirname(3) == "zplane03"
    assert generate_plane_dirname(3, [0, 4999]) == "zplane03_tp00001-05000"


def test_widget_modified_params():
    pytest.importorskip("imgui_bundle")
    from mbo_utilities.gui.widgets.pipelines.masknmf import (
        _collect_modified,
        _is_default,
    )

    s = MasknmfSettings()
    assert _collect_modified(s) == []
    assert _is_default(s.registration, "max_shifts")
    s.registration.max_shifts = (7, 9)
    s.demixing.maxiter = 55
    s.demixing.do_demixing = STAGE_FORCE  # tri-states excluded from the table
    rows = _collect_modified(s)
    assert [r[0] for r in rows] == ["reg.max_shifts", "demix.maxiter"]
    assert rows[1][1] == 55 and rows[1][2] == 40
    # float32 truncation from imgui must not read as modified
    s2 = MasknmfSettings()
    import numpy as np

    s2.demixing.filter_sigma = float(np.float32(s2.demixing.filter_sigma))
    assert _is_default(s2.demixing, "filter_sigma")


def test_find_masknmf_run(tmp_path):
    pytest.importorskip("imgui_bundle")
    from mbo_utilities.gui.widgets.pipelines.masknmf import find_masknmf_run

    # not a run
    assert find_masknmf_run(tmp_path) == (None, None)
    assert find_masknmf_run(None) == (None, None)

    # a run folder of another pipeline: outdir only
    older = tmp_path / "20261007T120000_masknmf_zplane01"
    older.mkdir()
    (older / "results.hdf5").touch()
    (older / "config.json").write_text(json.dumps({"pipeline": "Other", "configs": {}}))
    params, outdir = find_masknmf_run(tmp_path)
    assert params is None and outdir == str(tmp_path)

    # the newest run folder's parameters come back, from any entry point
    saved = MasknmfSettings()
    saved.demixing.maxiter = 55
    newer = tmp_path / "20261007T130000_masknmf_zplane01"
    newer.mkdir()
    (newer / "results.hdf5").touch()
    config = {"pipeline": "mbo_utilities.masknmf", "configs": saved.to_dict()}
    (newer / "config.json").write_text(json.dumps(config))
    for entry in (tmp_path, newer, newer / "config.json"):
        params, outdir = find_masknmf_run(entry)
        assert outdir == str(tmp_path)
        assert MasknmfSettings.from_dict(params).demixing.maxiter == 55


class TestBackgroundDownsampling:
    """masknmf's ring/background model average-pools the field by
    ``background_downsampling_factor`` and squeezes the result. Pooling floors,
    so a factor at or above a side length collapses that axis, the squeeze
    drops it, and ``lowrank_background_svd`` raises ``IndexError: tuple index
    out of range``. The default is 30, which any narrow field hits.
    """

    @staticmethod
    def _pooled(side: int, factor: int) -> int:
        """avg_pool2d(kernel=factor, stride=factor) output length."""
        return (side - factor) // factor + 1 if side >= factor else 0

    def _clamp(self, ly, lx, factor=None):
        from mbo_utilities.masknmf.params import MasknmfSettings
        from mbo_utilities.masknmf.runner import clamp_background_downsampling

        cfg = MasknmfSettings().demixing
        if factor is not None:
            cfg.background_downsampling_factor = factor
        return clamp_background_downsampling(cfg, ly, lx)

    def test_the_lbm_strip_that_failed(self):
        # 50x128 at the default 30: the 50 side pools to a single pixel
        assert self._pooled(50, 30) == 1
        cfg = self._clamp(50, 128)
        assert cfg.background_downsampling_factor == 12
        assert self._pooled(50, 12) >= 2 and self._pooled(128, 12) >= 2

    def test_a_wide_field_keeps_the_default(self):
        from mbo_utilities.masknmf.params import MasknmfSettings

        default = MasknmfSettings().demixing.background_downsampling_factor
        cfg = self._clamp(512, 512)
        assert cfg.background_downsampling_factor == default

    def test_never_zero_on_a_tiny_crop(self):
        cfg = self._clamp(6, 6)
        assert cfg.background_downsampling_factor == 1

    def test_the_callers_settings_are_left_alone(self):
        from mbo_utilities.masknmf.params import MasknmfSettings

        settings = MasknmfSettings()
        clamped = self._clamp(50, 128)
        assert clamped is not settings.demixing
        assert settings.demixing.background_downsampling_factor == 30

    def test_the_clamped_factor_survives_masknmfs_own_downsample(self):
        """The crash itself: masknmf pools the baseline image, squeezes, then
        reshapes on ``shape[1]``. Run its code, not a model of it.
        """
        try:
            import torch
            from masknmf.compression.decomposition import spatial_downsample
        except Exception:
            pytest.skip("masknmf/torch not importable")

        b = torch.zeros(50 * 128, 1).reshape(50, 128, 1)
        with pytest.raises(IndexError):
            bad = spatial_downsample(b, 30).squeeze()
            bad.reshape(bad.shape[0] * bad.shape[1], 1)

        factor = self._clamp(50, 128).background_downsampling_factor
        good = spatial_downsample(b, factor).squeeze()
        assert good.ndim == 2
        good.reshape(good.shape[0] * good.shape[1], 1)


class TestSplineDetrender:
    """The detrender reflect-pads by half its window; torch rejects a pad
    wider than the axis, so a movie shorter than the window has to skip it.
    """

    def _make(self, nframes, fs, window_seconds=40):
        from mbo_utilities.masknmf.runner import _spline_detrender

        return _spline_detrender(nframes, fs, window_seconds, 25, "cpu")

    def test_no_fs_means_no_detrender(self):
        assert self._make(100_000, None) is None

    def test_short_movie_at_a_high_frame_rate_skips_it(self):
        # 40 s at 429.93 Hz is a 17k-frame window; 3000 frames cannot pad it
        assert self._make(3000, 429.93) is None

    def test_long_movie_gets_one(self):
        try:
            import masknmf  # noqa: F401
        except Exception:
            pytest.skip("masknmf not importable")
        det = self._make(168_533, 429.93)
        assert det is not None and det.window == int(40 * 429.93)


def test_reference_kwargs_make_the_denoised_copy():
    reg = MasknmfSettings().registration
    assert reg.denoised_reference is False
    reg.reference_block_sizes = [6, 6]
    assert reg.reference_kwargs() == {
        "block_sizes": (6, 6),
        "max_components": 20,
        "max_consecutive_failures": 1,
        "spatial_avg_factor": 4,
        "temporal_avg_factor": 2,
    }


@pytest.fixture
def shaking_movie():
    """(T, 1, 1, Y, X) int16: two blobs moved by known integer shifts plus noise."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:48, :48]
    base = (
        400
        + 300 * np.exp(-((yy - 24) ** 2 + (xx - 20) ** 2) / 30)
        + 200 * np.exp(-((yy - 12) ** 2 + (xx - 34) ** 2) / 20)
    )
    shifts = rng.integers(-3, 4, size=(300, 2))
    mov = np.stack([np.roll(base, tuple(s), axis=(0, 1)) for s in shifts])
    mov = mov + rng.normal(0, 40, mov.shape)
    return mov.astype(np.int16)[:, None, None], shifts


@pytest.fixture
def shaking_tif(tmp_path, shaking_movie):
    import tifffile

    movie, _ = shaking_movie
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, movie[:, 0, 0])
    return path


@pytest.fixture
def denoised_registration():
    s = MasknmfSettings()
    s.registration.denoised_reference = True
    s.registration.max_shifts = (6, 6)
    s.compression.denoise = False
    s.compression.block_sizes = (16, 16)
    s.demixing.do_demixing = STAGE_SKIP
    s.runtime.device = "cpu"
    return s


@pytest.mark.slow
def test_a_run_folder_holds_no_movie_and_registers_the_raw_one(
    tmp_path, shaking_tif, shaking_movie, denoised_registration
):
    pytest.importorskip("masknmf")
    import h5py

    from mbo_utilities import imread
    from mbo_utilities.arrays.masknmf_run import MasknmfRunArray
    from mbo_utilities.masknmf import run_plane

    movie, shifts = shaking_movie
    run = run_plane(
        str(shaking_tif),
        tmp_path / "out",
        settings=denoised_registration,
        writer_kwargs={"invert_deflection": True},
        replot=False,
    )
    assert run.parent == tmp_path / "out" and run.name.endswith("_masknmf_zplane01")
    names = {p.name for p in run.iterdir()}
    assert {"results.hdf5", "config.json", "alignment.hdf5", f"{run.name}.log"} <= names
    assert not names & {"data.bin", "data_raw.bin", "ops.npy", "stat.npy"}
    with h5py.File(run / "results.hdf5") as f:
        assert {"RigidRegistrationArray", "RigidMotionCorrector", "CompressionArray"} <= set(f)
        est = f["RigidRegistrationArray"]["shifts"][()]
    assert np.corrcoef(est[:, 0], shifts[:, 0])[0, 1] > 0.9
    assert np.corrcoef(est[:, 1], shifts[:, 1])[0, 1] > 0.9
    config = json.loads((run / "config.json").read_text())
    assert config["pipeline"] == "mbo_utilities.masknmf" and config["run"]["status"] == "completed"
    assert config["inputs"]["movie"]["read_features"] == {"invert_deflection": True}

    arr = imread(run)
    assert isinstance(arr, MasknmfRunArray) and arr.shape == (300, 1, 1, 48, 48)
    data = movie[:, 0, 0].astype(np.float32)
    raw = np.asarray(arr.raw[:])
    np.testing.assert_allclose(raw, 2 * data.mean(axis=0) - data, rtol=1e-5)
    inner = (slice(None), slice(6, -6), slice(6, -6))
    reg = np.asarray(arr[:, 0, 0])
    assert reg[inner].var(axis=0).mean() < raw[inner].var(axis=0).mean()


@pytest.mark.slow
def test_a_later_run_copies_the_stages_whose_provenance_matches(
    tmp_path, shaking_tif, denoised_registration
):
    pytest.importorskip("masknmf")
    from mbo_utilities.masknmf import run_plane

    first = run_plane(str(shaking_tif), tmp_path, settings=denoised_registration, replot=False)
    second = run_plane(str(shaking_tif), tmp_path, settings=denoised_registration, replot=False)
    denoised_registration.compression.block_sizes = (12, 12)
    third = run_plane(str(shaking_tif), tmp_path, settings=denoised_registration, replot=False)
    assert len({first, second, third}) == 3
    actions = [
        {k: v["action"] for k, v in json.loads((r / "config.json").read_text())["timings"].items()}
        for r in (first, second, third)
    ]
    assert actions[0]["registration"] == actions[0]["compression"] == "compute"
    assert actions[1]["registration"] == actions[1]["compression"] == "reuse"
    assert actions[2] == {"registration": "reuse", "compression": "compute", "demixing": "skip"}
    assert (third / "alignment.hdf5").exists()


@pytest.mark.slow
def test_task_masknmf_registers_the_inverted_movie(tmp_path, shaking_tif, shaking_movie):
    pytest.importorskip("masknmf")
    import logging

    from mbo_utilities.arrays.masknmf_run import run_raw_movie
    from mbo_utilities.gui.tasks import task_masknmf

    movie, _ = shaking_movie
    s = MasknmfSettings()
    s.registration.denoised_reference = True
    s.compression.do_compression = STAGE_SKIP
    s.demixing.do_demixing = STAGE_SKIP
    s.runtime.device = "cpu"
    task_masknmf(
        {
            "input_path": str(shaking_tif),
            "output_dir": str(tmp_path / "out"),
            "planes": [1],
            "settings": s.to_dict(),
            "fix_phase": False,
            "invert_deflection": True,
        },
        logging.getLogger("test"),
    )
    run = next((tmp_path / "out").glob("*_masknmf_zplane01"))
    data = movie[:, 0, 0].astype(np.float32)
    np.testing.assert_allclose(run_raw_movie(run)[:], 2 * data.mean(axis=0) - data, rtol=1e-5)
    assert (run / "alignment.hdf5").exists()


@pytest.fixture
def firing_tif(tmp_path):
    """Three gaussian cells with sparse exponential transients, (600, 48, 48) int16."""
    import tifffile

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:48, :48]
    mov = np.full((600, 48, 48), 200.0)
    for cy, cx in ((12, 12), (30, 34), (36, 14)):
        footprint = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 12)
        spikes = (rng.random(600) < 0.03).astype(float)
        trace = np.convolve(spikes, np.exp(-np.arange(30) / 6))[:600] * 400
        mov += footprint[None] * (100 + trace[:, None, None])
    mov += rng.normal(0, 15, mov.shape)
    path = tmp_path / "cells.tif"
    tifffile.imwrite(path, mov.astype(np.int16))
    return path


@pytest.mark.slow
def test_demixing_writes_the_results_file(tmp_path, firing_tif):
    pytest.importorskip("masknmf")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("masknmf's superpixel init needs CUDA")
    from mbo_utilities import imread
    from mbo_utilities.masknmf import run_plane
    from mbo_utilities.results import Results

    s = MasknmfSettings()
    s.compression.denoise = False
    s.compression.block_sizes = (16, 16)
    s.demixing.patch_size = (24, 24)
    s.runtime.device = "cuda"
    run = run_plane(str(firing_tif), tmp_path, settings=s, metadata={"fs": 30.0})
    unit = Results.open(run).units["zplane01"]
    assert unit.n_rois == 3 and unit.traces["raw"].shape == (3, 600)
    assert set(unit.images) == {"mean", "max", "corr", "ref"}
    assert imread(run).results.units["zplane01"].n_rois == 3
    assert any(p.suffix == ".png" for p in run.iterdir())
