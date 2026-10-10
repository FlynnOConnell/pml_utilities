import h5py
import numpy as np
from mbo_utilities.arrays.features import MotionCorrection
from mbo_utilities.masknmf.params import REG_DENOISE_PIPELINE, RegDenoiseSettings
from mbo_utilities.masknmf.reg_denoise import alignment, is_reg_denoise


class TestRegDenoiseSettings:
    def test_defaults_are_the_reference_run(self):
        s = RegDenoiseSettings()
        assert (s.channel, s.first_frame) == (1, 200)
        assert s.reference.block_sizes == (4, 4)
        assert s.registration.max_shifts == (40, 40)
        assert not s.compression.denoise

    def test_round_trip(self):
        s = RegDenoiseSettings(channel=2, first_frame=0)
        s.registration.max_shifts = (5, 5)
        s.compression.denoise = True
        back = RegDenoiseSettings.from_dict(s.to_dict())
        assert back == s

    def test_partial_dict_keeps_stage_defaults(self):
        back = RegDenoiseSettings.from_dict({"reference": {"max_components": 8}})
        assert back.reference.max_components == 8
        assert back.reference.block_sizes == (4, 4)
        assert back.compression == RegDenoiseSettings().compression


class TestIsRegDenoise:
    def test_marked_file(self, tmp_path):
        p = tmp_path / "results.hdf5"
        with h5py.File(p, "w") as f:
            f.attrs["mbo_pipeline"] = REG_DENOISE_PIPELINE
        assert is_reg_denoise(p)

    def test_other_hdf5(self, tmp_path):
        p = tmp_path / "results.hdf5"
        with h5py.File(p, "w") as f:
            f.create_group("DemixingResults")
        assert not is_reg_denoise(p)

    def test_not_hdf5(self, tmp_path):
        p = tmp_path / "results.hdf5"
        p.write_bytes(b"not hdf5")
        assert not is_reg_denoise(p)
        assert not is_reg_denoise(tmp_path / "missing.hdf5")


class TestMotionAt:
    def test_holds_last_sample(self):
        mc = MotionCorrection(
            "RTMC", "um", {"X": (np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0]))}
        )
        out = mc.at(np.array([0.0, 0.5, 1.0, 1.9, 2.5]))
        np.testing.assert_array_equal(out["X"], [5, 5, 6, 6, 7])

    def test_before_first_sample_uses_first(self):
        mc = MotionCorrection("RTMC", "um", {"Y": (np.array([1.0]), np.array([3.0]))})
        np.testing.assert_array_equal(mc.at(np.array([0.0, 2.0]))["Y"], [3, 3])


class TestAlignment:
    def test_shifted_frames_align_worse(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(32, 32)).astype(np.float32)
        still = np.repeat(base[None], 200, axis=0) + rng.normal(
            scale=0.1, size=(200, 32, 32)
        )
        moving = np.stack([np.roll(f, rng.integers(-4, 5), axis=1) for f in still])
        assert alignment(still) > alignment(moving)
