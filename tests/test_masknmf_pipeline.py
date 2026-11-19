"""masknmf runs from pml_utilities: the (T, Y, X) view handed to masknmf, the results zarr's
calibration, and one plane through masknmf's own pipeline on a synthetic movie (CUDA only).
"""

import numpy as np
import pytest

from mbo_utilities.masknmf import outputs


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


@pytest.fixture
def firing_tif(tmp_path):
    """Nine gaussian cells with sparse exponential transients, (600, 128, 128) int16.

    128 px: masknmf's background step downsamples the frame, which a smaller one does not survive.
    """
    import tifffile

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:128, :128]
    mov = np.full((600, 128, 128), 200.0)
    for cy in (24, 64, 104):
        for cx in (24, 64, 104):
            footprint = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 20)
            spikes = (rng.random(600) < 0.03).astype(float)
            trace = np.convolve(spikes, np.exp(-np.arange(30) / 6))[:600] * 400
            mov += footprint[None] * (100 + trace[:, None, None])
    mov += rng.normal(0, 15, mov.shape)
    path = tmp_path / "cells.tif"
    tifffile.imwrite(path, mov.astype(np.int16))
    return path


def test_plane_view_picks_the_plane_channel_and_frames():
    from mbo_utilities.arrays.numpy import NumpyArray
    from mbo_utilities.masknmf.pipeline import plane_view

    data = np.arange(6 * 2 * 3 * 4 * 5, dtype=np.int16).reshape(6, 2, 3, 4, 5)
    arr = NumpyArray(data, metadata={"fs": 30.0})
    view, record, inverted = plane_view(arr, plane=3, channel=2, frame_indices=[0, 2, 4])
    assert view.shape == (3, 4, 5)
    np.testing.assert_array_equal(view[:], data[[0, 2, 4], 1, 2])
    assert (record["z"], record["c"], record["plane"]) == (2, 1, 3)
    assert record["frames"] == [0, 5, 2]
    assert record["fs"] == pytest.approx(15.0)
    assert not inverted


def test_pipeline_class_names_masknmf_pipelines():
    pytest.importorskip("masknmf")
    from mbo_utilities.masknmf.pipeline import pipeline_class

    assert pipeline_class({"pipeline": "TwoPhotonCalciumPipeline", "configs": {}}).__name__ == "TwoPhotonCalciumPipeline"
    with pytest.raises(ValueError, match="not a masknmf pipeline"):
        pipeline_class({"pipeline": "Nope", "configs": {}})


@pytest.mark.slow
def test_one_plane_through_masknmf_writes_its_run_folder_and_results_zarr(tmp_path, firing_tif):
    pytest.importorskip("masknmf")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("masknmf's superpixel init needs CUDA")
    from mbo_utilities import imread
    from mbo_utilities.arrays.masknmf_run import is_masknmf_run, run_config, run_movie
    from mbo_utilities.masknmf.pipeline import run_planes
    from mbo_utilities.results import Results

    config = {"pipeline": "TwoPhotonCalciumPipeline", "configs": {}}
    runs = run_planes(imread(firing_tif), tmp_path / "out", config, run_args={"frame_rate": 30.0})
    assert len(runs) == 1 and is_masknmf_run(runs[0])
    assert runs[0].parent.name == "zplane01"
    assert run_movie(run_config(runs[0]))["path"] == str(firing_tif)
    zarrs = list(runs[0].glob("*.zarr"))
    assert len(zarrs) == 1
    unit = next(iter(Results.open(zarrs[0]).units.values()))
    assert unit.n_rois >= 1
    assert unit.traces["dff"].shape == (unit.n_rois, 600)
    from masknmf.visualization.imgui.parameters import Parameters

    from mbo_utilities.gui.widgets.pipelines.masknmf import find_masknmf_run

    loaded, outdir = find_masknmf_run(tmp_path / "out")
    assert outdir == str(tmp_path / "out")
    parameters = Parameters("two-photon-calcium")
    parameters.load(loaded)
    assert parameters.config()["pipeline"] == "TwoPhotonCalciumPipeline"
