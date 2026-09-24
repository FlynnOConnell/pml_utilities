"""Temporal frame averaging: the read-time view, and the viewer lock that
installs it as a pipeline step.
"""

import numpy as np
import pytest
from mbo_utilities.arrays import FrameAveragedView, average_frames
from mbo_utilities.arrays.numpy import NumpyArray

FIGURE_SIZE = (900, 700)


@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    return (rng.random((22, 2, 1, 4, 5)) * 500).astype(np.int16)


@pytest.fixture
def source(raw):
    return NumpyArray(raw, dims="TCZYX", metadata={"fs": 30.0, "num_frames": 22})


def reference(raw, factor):
    """What numpy would give: whole bins only, rounded back to the dtype."""
    n = raw.shape[0] // factor
    mean = raw[: n * factor].reshape(n, factor, *raw.shape[1:]).mean(axis=1)
    return np.rint(mean).astype(raw.dtype)


class TestShape:
    def test_bins_are_whole_and_the_tail_is_dropped(self, source, raw):
        view = average_frames(source, 4)
        assert view.shape == (5, 2, 1, 4, 5)  # 22 // 4, the last 2 frames dropped
        assert view.nt == 5 and len(view) == 5 and view.ndim == 5
        assert view.dims == ("T", "C", "Z", "Y", "X")

    def test_factor_one_is_the_source_itself(self, source):
        assert average_frames(source, 1) is source

    def test_wrapping_a_view_replaces_it_rather_than_stacking(self, source):
        once = average_frames(source, 2)
        twice = average_frames(once, 4)
        assert twice.source is source and twice.factor == 4
        assert average_frames(once, 1) is source

    def test_a_factor_past_the_end_is_refused(self, source):
        with pytest.raises(ValueError):
            average_frames(source, 100)

    def test_zero_and_negative_are_refused(self, source):
        for bad in (0, -3):
            with pytest.raises(ValueError):
                FrameAveragedView(source, bad)


class TestReads:
    """Every key form must agree with numpy on the binned array."""

    @pytest.fixture
    def pair(self, source, raw):
        return average_frames(source, 4), reference(raw, 4)

    def test_whole_array(self, pair):
        view, ref = pair
        assert np.array_equal(np.asarray(view[:]), ref)
        assert np.array_equal(np.asarray(view), ref)

    def test_integer_and_negative_t(self, pair):
        view, ref = pair
        assert np.array_equal(view[2], ref[2])
        assert np.array_equal(view[-1], ref[-1])

    def test_slices_and_steps(self, pair):
        view, ref = pair
        assert np.array_equal(view[1:4], ref[1:4])
        assert np.array_equal(view[0:5:2], ref[0:5:2])
        assert view[3:3].shape == (0, 2, 1, 4, 5)

    def test_fancy_t(self, pair):
        view, ref = pair
        assert np.array_equal(view[[3, 1]], ref[[3, 1]])

    def test_spatial_subkey_reads_only_that_crop(self, pair):
        view, ref = pair
        assert np.array_equal(view[1:3, 0, 0, 1:3, 2:4], ref[1:3, 0, 0, 1:3, 2:4])

    def test_t_out_of_range(self, pair):
        view, _ = pair
        with pytest.raises(IndexError):
            view[99]

    def test_the_cache_does_not_change_what_is_read(self, pair):
        view, ref = pair
        for _ in range(3):
            assert np.array_equal(view[2], ref[2])


class TestDtype:
    def test_source_dtype_is_preserved_by_default(self, source, raw):
        view = average_frames(source, 4)
        assert view.dtype == raw.dtype
        assert np.asarray(view[:]).dtype == raw.dtype

    def test_float32_keeps_the_fractional_means(self, source, raw):
        view = average_frames(source, 4, dtype="float32")
        assert view.dtype == np.float32
        exact = raw[:20].reshape(5, 4, 2, 1, 4, 5).mean(axis=1, dtype=np.float32)
        np.testing.assert_allclose(np.asarray(view[:]), exact, rtol=1e-6)

    def test_an_unknown_dtype_is_refused(self, source):
        with pytest.raises(ValueError):
            FrameAveragedView(source, 2, dtype="int8")


class TestMetadata:
    """Averaging N frames divides the frame rate by N. Every downstream
    window, detrend and trace axis is in seconds, so this must be right.
    """

    def test_rate_and_frame_count_are_scaled(self, source):
        meta = average_frames(source, 4).metadata
        assert meta["fs"] == pytest.approx(7.5)
        assert meta["num_frames"] == 5
        assert meta["frame_average"] == 4

    def test_every_rate_alias_is_scaled(self, raw):
        arr = NumpyArray(
            raw, dims="TCZYX", metadata={"frame_rate": 30.0, "framerate": 30.0}
        )
        meta = average_frames(arr, 2).metadata
        assert meta["frame_rate"] == pytest.approx(15.0)
        assert meta["framerate"] == pytest.approx(15.0)

    def test_history_records_the_step(self, source):
        history = average_frames(source, 4).metadata["processing_history"]
        assert {"step": "frame_average", "factor": 4} in history

    def test_every_registered_alias_is_retimed(self, raw):
        """A single alias left claiming the original rate makes the resolver
        warn about a stale alias, and anything reading `fps` or `dt` straight
        from the dict gets the pre-binning number.
        """
        from mbo_utilities.metadata import get_param

        meta = {
            "fs": 30.0,
            "fps": 30.0,
            "fr": 30.0,
            "scanFrameRate": 30.0,
            "frameRate": 30.0,
            "sampling_frequency": 30.0,
            "frame_rate_hz": 30.0,
            "finterval": 1 / 30,
            "dt": 1 / 30,
            "frame_period": 1 / 30,
        }
        out = average_frames(NumpyArray(raw, dims="TCZYX", metadata=meta), 3).metadata
        for key in (
            "fs",
            "fps",
            "fr",
            "scanFrameRate",
            "frameRate",
            "sampling_frequency",
            "frame_rate_hz",
        ):
            assert out[key] == pytest.approx(10.0), key
        for key in ("finterval", "dt", "frame_period"):
            assert out[key] == pytest.approx(0.1), key
        assert get_param(out, "fs") == pytest.approx(10.0)

    def test_missing_rate_is_left_alone(self, raw):
        meta = average_frames(NumpyArray(raw, dims="TCZYX"), 2).metadata
        assert "fs" not in meta and meta["frame_average"] == 2


class TestPassthrough:
    def test_domain_attributes_forward_to_the_source(self, source, tmp_path):
        view = average_frames(source, 4)
        assert view.source is source
        assert view._arr is source
        # a name only the source defines still resolves
        source.some_reader_detail = "kept"
        assert view.some_reader_detail == "kept"

    def test_the_frame_count_never_leaks_from_the_source(self, source):
        view = average_frames(source, 4)
        # nt/num_frames/len come off _shape5d, not the wrapped reader
        assert view.nt == 5 and len(view) == 5
        assert view.shape[0] == 5 and source.shape[0] == 22

    def test_imwrite_bakes_the_averaging_in(self, source, raw, tmp_path):
        from mbo_utilities.reader import imread

        view = average_frames(source, 4)
        view._imwrite(tmp_path, ext=".tiff", overwrite=True)
        written = list(tmp_path.rglob("*.tif*"))
        assert written, "nothing written"
        back = np.asarray(imread(written[0])[:]).squeeze()
        assert back.shape[0] == 5
        np.testing.assert_allclose(
            back.reshape(5, -1), reference(raw, 4).squeeze().reshape(5, -1), atol=1
        )


class TestReadFeatures:
    """The features API: ``frame_average`` as an imread / imwrite kwarg, the
    reader_kwargs round-trip a worker uses to re-open the same binned array,
    and the shared applier every save / run path goes through.
    """

    def test_imread_kwarg_and_reader_kwargs_roundtrip(self, source, tmp_path):
        from mbo_utilities.reader import imread, source_reader_kwargs
        from mbo_utilities.writer import imwrite

        imwrite(source, tmp_path, ext=".tiff", overwrite=True)
        path = next(tmp_path.rglob("*.tif*"))
        view = imread(path, frame_average=4)
        assert isinstance(view, FrameAveragedView)
        assert view.shape[0] == 5
        assert source_reader_kwargs(view) == {"frame_average": 4}
        again = imread(path, **source_reader_kwargs(view))
        assert again.shape == view.shape

    def test_imread_factor_one_is_the_plain_reader(self, source, tmp_path):
        from mbo_utilities.reader import imread, source_reader_kwargs
        from mbo_utilities.writer import imwrite

        imwrite(source, tmp_path, ext=".tiff", overwrite=True)
        arr = imread(next(tmp_path.rglob("*.tif*")), frame_average=1)
        assert not isinstance(arr, FrameAveragedView)
        assert source_reader_kwargs(arr) == {}

    def test_imwrite_kwarg_bakes_the_binning_in(self, source, raw, tmp_path):
        from mbo_utilities.reader import imread
        from mbo_utilities.writer import imwrite

        imwrite(source, tmp_path, ext=".tiff", overwrite=True, frame_average=4)
        back = imread(next(tmp_path.rglob("*.tif*")))
        assert back.shape[0] == 5
        np.testing.assert_allclose(
            np.asarray(back[:]).squeeze().reshape(5, -1),
            reference(raw, 4).squeeze().reshape(5, -1),
            atol=1,
        )
        assert back.metadata["frame_average"] == 4
        assert back.metadata["fs"] == pytest.approx(7.5)

    def test_apply_read_features_pops_and_wraps(self, source):
        from mbo_utilities.arrays.features import apply_read_features

        arr, rest = apply_read_features(
            source, {"frame_average": 4, "fix_phase": True, "sharded": False}
        )
        assert isinstance(arr, FrameAveragedView) and arr.factor == 4
        assert rest == {"sharded": False}, "writer kwargs pass through untouched"
        back, _ = apply_read_features(arr, frame_average=1)
        assert back is source, "factor 1 unwraps"
        same, _ = apply_read_features(arr, frame_average=None)
        assert same is arr, "None leaves the array alone"

    def test_feature_object(self):
        from mbo_utilities.arrays.features import FrameAverageFeature

        fa = FrameAverageFeature(4)
        assert fa.enabled and fa.to_reader_kwargs() == {"frame_average": 4}
        seen = []
        fa.add_event_handler(seen.append)
        fa.factor = 2
        assert seen and seen[0].info == {"value": 2, "old_value": 4}
        assert FrameAverageFeature().to_reader_kwargs() == {}
        with pytest.raises(ValueError):
            FrameAverageFeature(0)


class TestWriterContract:
    """The writers mutate the array they are handed (roi per split, fix_phase
    from the save options) and reassign its metadata; the view has to let all
    of that through to the source without corrupting its own scaling.
    """

    def test_setting_reader_attributes_reaches_the_source(self, source):
        view = average_frames(source, 4)
        view.some_setting = "x"
        assert source.some_setting == "x"
        assert view.some_setting == "x"

    def test_metadata_assignment_does_not_double_scale(self, source):
        view = average_frames(source, 4)
        meta = dict(view.metadata)
        meta["custom"] = 1
        view.metadata = meta
        assert view.metadata["fs"] == pytest.approx(7.5)
        assert view.metadata["custom"] == 1
        assert view.metadata["num_frames"] == 5
        assert source.metadata["fs"] == pytest.approx(30.0)
        assert source.metadata["num_frames"] == 22
        assert "frame_average" not in source.metadata
        history = view.metadata["processing_history"]
        assert history.count({"step": "frame_average", "factor": 4}) == 1

    def test_offsets_map_to_the_first_source_frame(self, source):
        source.get_offset_at = lambda t, c, z: float(t)
        view = average_frames(source, 4)
        assert view.get_offset_at(3, 0, 0) == 12.0

    def test_frame_average_reads_on_any_array(self, source):
        assert getattr(source, "frame_average", 1) == 1
        assert average_frames(source, 4).frame_average == 4


class TestWorkerPaths:
    """The subprocess paths re-open the dataset from its path; the option has
    to survive that round trip the way fix_phase does.
    """

    def test_task_save_as_bins_the_output(self, source, raw, tmp_path):
        import logging

        from mbo_utilities.gui.tasks import task_save_as
        from mbo_utilities.reader import imread
        from mbo_utilities.writer import imwrite

        src_dir = tmp_path / "src"
        imwrite(source, src_dir, ext=".tiff", overwrite=True)
        out_dir = tmp_path / "out"
        task_save_as(
            {
                "input_path": str(next(src_dir.rglob("*.tif*"))),
                "output_path": str(out_dir),
                "ext": ".tiff",
                "frame_average": 4,
                "fix_phase": False,
                "use_fft": False,
            },
            logging.getLogger("test"),
        )
        back = imread(next(out_dir.rglob("*.tif*")))
        assert back.shape[0] == 5
        assert back.metadata["frame_average"] == 4

    def test_save_as_worker_forwards_the_option(self, source, tmp_path):
        from mbo_utilities.gui._save_as import _save_as_worker
        from mbo_utilities.reader import imread
        from mbo_utilities.writer import imwrite

        src_dir = tmp_path / "src"
        imwrite(source, src_dir, ext=".tiff", overwrite=True)
        out_dir = tmp_path / "out"
        _save_as_worker(
            str(next(src_dir.rglob("*.tif*"))),
            outpath=out_dir,
            ext=".tiff",
            overwrite=True,
            fix_phase=False,
            use_fft=False,
            border=10,
            max_offset=4,
            mean_subtraction=False,
            frame_average=2,
        )
        assert imread(next(out_dir.rglob("*.tif*"))).shape[0] == 11

    def test_imwrite_phase_kwargs_reach_the_array(self, raw, tmp_path):
        """fix_phase= used to fall through imwrite's **kwargs unread; the
        pipeline menus rely on it landing on the reader.
        """
        from mbo_utilities.arrays._phasecorr_view import with_phasecorr
        from mbo_utilities.writer import imwrite

        rng = np.random.default_rng(1)
        movie = (rng.random((6, 1, 1, 64, 128)) * 500).astype(np.int16)
        view = with_phasecorr(NumpyArray(movie, dims="TCZYX"))
        assert view.fix_phase is False
        imwrite(
            view, tmp_path, ext=".tiff", overwrite=True, fix_phase=True, use_fft=True
        )
        assert view.fix_phase is True and view.use_fft is True
