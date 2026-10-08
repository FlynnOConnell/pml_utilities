"""masknmf compressed movies (``CompressionArray`` hdf5) as lazy 5D arrays, and a run folder's movies for the registration viewer."""

import json

import h5py
import numpy as np
import pytest
import tifffile

from mbo_utilities import imread
from mbo_utilities.arrays.compression import CompressedMovieArray

T, Y, X, R = 10, 6, 8, 3


@pytest.fixture
def factors():
    rng = np.random.default_rng(0)
    u = rng.random((Y * X, R)).astype(np.float32)
    v = rng.random((R, T)).astype(np.float32)
    mean = (100 + rng.random((Y, X))).astype(np.float32)
    scale = (1 + rng.random((Y, X))).astype(np.float32)
    return u, v, mean, scale


@pytest.fixture
def compressed(tmp_path, factors):
    """A CompressionArray hdf5 in masknmf's layout, with a provenance frame rate."""
    u, v, mean, scale = factors
    rows, cols = np.nonzero(u)
    path = tmp_path / "compression.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["mbo_provenance"] = '{"fs": 250.0}'
        g = f.create_group("CompressionArray")
        g.create_dataset("shape", data=np.array([T, Y, X]))
        s = g.create_group("spatial_compressed")
        s.attrs["layout"] = "sparse_coo"
        s.create_dataset("indices", data=np.array([rows, cols]))
        s.create_dataset("values", data=u[rows, cols])
        s.create_dataset("size", data=np.array([Y * X, R]))
        g.create_dataset("temporal_compressed", data=v)
        g.create_dataset("mean_image", data=mean)
        g.create_dataset("noise_variance_image", data=scale)
    return path


def movie(factors):
    u, v, mean, scale = factors
    return (u @ v).T.reshape(T, Y, X) * scale + mean


def test_imread_dispatches_to_it(compressed):
    arr = imread(compressed)
    assert isinstance(arr, CompressedMovieArray)
    assert arr.shape == (T, 1, 1, Y, X) and arr.dtype == np.float32
    assert arr.metadata["fs"] == 250.0


def test_frames_are_rebuilt_in_movie_units(compressed, factors):
    arr = CompressedMovieArray(compressed)
    np.testing.assert_allclose(arr[:, 0, 0], movie(factors), rtol=1e-5)


@pytest.mark.parametrize(
    "key", [4, -1, slice(2, 6), (slice(None), 0, 0, slice(1, 4)), [0, 3, 9]]
)
def test_keys_agree_with_numpy(compressed, factors, key):
    full = movie(factors)[:, None, None]
    np.testing.assert_allclose(CompressedMovieArray(compressed)[key], full[key], rtol=1e-5)


def test_the_representative_frame_is_the_mean_image(compressed, factors):
    np.testing.assert_allclose(np.asarray(CompressedMovieArray(compressed)), factors[2])


def test_a_file_without_the_group_is_not_claimed(tmp_path):
    path = tmp_path / "other.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("mov", data=np.zeros((2, 3, 4)))
    assert not CompressedMovieArray.can_open(path)


def test_it_matches_masknmf_on_a_real_export(tmp_path):
    masknmf = pytest.importorskip("masknmf")
    rng = np.random.default_rng(1)
    data = (100 + rng.normal(0, 5, (60, 24, 24))).astype(np.float32)
    pmd = masknmf.CompressStrategy(
        block_sizes=(12, 12), frame_batch_size=60, device="cpu"
    ).compress(data)
    path = tmp_path / "compression.hdf5"
    pmd.export(path)
    np.testing.assert_allclose(
        imread(path)[:, 0, 0], np.asarray(pmd[:]), rtol=1e-4, atol=1e-3
    )


def test_a_run_folders_movies_in_panel_order(compressed, factors):
    from mbo_utilities.gui.registration_viewer import registration_movies

    run = compressed.parent
    raw = np.arange(T * Y * X, dtype=np.int16).reshape(T, Y, X)
    tifffile.imwrite(run / "movie.tif", raw)
    compressed.rename(run / "results.hdf5")
    inputs = {"movie": {"path": str(run / "movie.tif"), "z": 0, "c": 0}}
    (run / "config.json").write_text(json.dumps({"inputs": inputs}))
    movies = registration_movies(run)
    assert list(movies) == ["raw", "denoised"]
    assert all(m.shape == (T, Y, X) for m in movies.values())
    np.testing.assert_array_equal(movies["raw"][:], raw)
    np.testing.assert_allclose(movies["denoised"][:], movie(factors), rtol=1e-5)


def test_a_folder_without_a_run_is_refused(tmp_path):
    from mbo_utilities.gui.registration_viewer import registration_movies

    with pytest.raises(FileNotFoundError):
        registration_movies(tmp_path)
