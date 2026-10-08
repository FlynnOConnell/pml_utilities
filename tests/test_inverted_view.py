"""Deflection inversion as a read-time view: values, keys, and the round trip
through ``imread`` and ``apply_read_features``.
"""

import numpy as np
import pytest
import tifffile
from mbo_utilities import imread
from mbo_utilities.arrays import InvertedDeflectionView, invert_deflection
from mbo_utilities.arrays.features import READ_FEATURE_KEYS, apply_read_features
from mbo_utilities.arrays.numpy import NumpyArray


@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    return (rng.random((12, 2, 1, 4, 5)) * 500).astype(np.int16)


@pytest.fixture
def source(raw):
    return NumpyArray(raw, dims="TCZYX", metadata={"fs": 30.0})


def flipped(raw):
    data = raw.astype(np.float32)
    return 2 * data.mean(axis=0) - data


class TestValues:
    def test_every_frame_is_flipped_about_the_mean(self, source, raw):
        view = invert_deflection(source)
        assert view.shape == source.shape and view.dtype == np.float32
        np.testing.assert_allclose(view[:], flipped(raw), rtol=1e-5)

    def test_the_mean_is_unchanged(self, source, raw):
        view = invert_deflection(source)
        np.testing.assert_allclose(view[:].mean(axis=0), raw.mean(axis=0), rtol=1e-4)

    @pytest.mark.parametrize(
        "key",
        [3, -1, slice(2, 7), (slice(None), 1), (4, 0, 0, slice(1, 3)), [0, 5, 9]],
    )
    def test_keys_agree_with_numpy(self, source, raw, key):
        np.testing.assert_allclose(
            invert_deflection(source)[key], flipped(raw)[key], rtol=1e-5
        )

    def test_a_frame_window_keeps_the_whole_recordings_mean(self, source, raw):
        np.testing.assert_allclose(
            invert_deflection(source)[2:5], flipped(raw)[2:5], rtol=1e-5
        )


class TestWrapping:
    def test_disabled_is_the_source_itself(self, source):
        assert invert_deflection(source, False) is source

    def test_wrapping_a_view_does_not_stack(self, source):
        once = invert_deflection(source)
        assert invert_deflection(once).source is source
        assert invert_deflection(once, False) is source

    def test_attribute_writes_reach_the_source(self, source):
        view = InvertedDeflectionView(source)
        view.custom_flag = 7
        assert source.custom_flag == 7

    def test_metadata_records_the_step(self, source):
        meta = invert_deflection(source).metadata
        assert meta["invert_deflection"] is True
        assert meta["processing_history"][-1] == {"step": "invert_deflection"}


class TestReadFeatures:
    def test_it_is_a_read_feature(self):
        assert "invert_deflection" in READ_FEATURE_KEYS

    def test_apply_read_features_wraps_and_consumes_the_key(self, source):
        arr, rest = apply_read_features(source, {"invert_deflection": True, "x": 1})
        assert isinstance(arr, InvertedDeflectionView) and rest == {"x": 1}

    def test_false_and_none_leave_the_array_alone(self, source):
        assert apply_read_features(source, {"invert_deflection": False})[0] is source
        assert apply_read_features(source, {"invert_deflection": None})[0] is source

    def test_imread_round_trips_through_reader_kwargs(self, tmp_path, raw):
        path = tmp_path / "movie.tif"
        tifffile.imwrite(path, raw[:, 0, 0])
        arr = imread(path, invert_deflection=True)
        assert isinstance(arr, InvertedDeflectionView)
        assert arr.reader_kwargs["invert_deflection"] is True
        again = imread(path, **arr.reader_kwargs)
        np.testing.assert_allclose(again[:], arr[:])
        np.testing.assert_allclose(
            np.asarray(arr[:]).squeeze(), flipped(raw[:, 0, 0]), rtol=1e-5
        )
