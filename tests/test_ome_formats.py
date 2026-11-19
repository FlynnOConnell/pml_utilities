"""OME-TIFF and OME-Zarr written by other tools: axes, voxel size and save-as round trips."""

import numpy as np
import pytest
import tifffile
import zarr

import mbo_utilities as mbo

FS, DX, DZ = 14.0, 2.0, 16.0
AXES = {
    "t": {"name": "t", "type": "time", "unit": "second"},
    "c": {"name": "c", "type": "channel"},
    "z": {"name": "z", "type": "space", "unit": "micrometer"},
    "y": {"name": "y", "type": "space", "unit": "micrometer"},
    "x": {"name": "x", "type": "space", "unit": "micrometer"},
}
SCALE = {"t": 1 / FS, "c": 1.0, "z": DZ, "y": DX, "x": DX}


def _tczyx():
    """(T, C, Z, Y, X) int16 where every (t, c, z) frame holds a distinct value."""
    t, c, z = np.meshgrid(np.arange(6), np.arange(2), np.arange(3), indexing="ij")
    return np.broadcast_to((t * 100 + c * 10 + z)[..., None, None], (6, 2, 3, 8, 10)).astype(np.int16).copy()


def _write_ngff(path, data, axes, version="0.5"):
    multiscale = {
        "axes": [AXES[a] for a in axes],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [SCALE[a] for a in axes]}]}],
    }
    if version == "0.5":
        group = zarr.open_group(path, mode="w", zarr_format=3)
        group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale]}
        group.create_array("0", data=data, dimension_names=list(axes))
    else:
        group = zarr.open_group(path, mode="w", zarr_format=2)
        group.attrs["multiscales"] = [multiscale | {"version": "0.4"}]
        group.create_array("0", data=data)
    return path


class TestForeignOmeZarr:
    @pytest.mark.parametrize("version", ["0.5", "0.4"])
    def test_tczyx_axes_and_voxel_size(self, tmp_path, version):
        data = _tczyx()
        arr = mbo.imread(_write_ngff(tmp_path / "a.zarr", data, "tczyx", version))
        assert arr.shape == data.shape
        assert (arr.fs, arr.dx, arr.dz) == pytest.approx((FS, DX, DZ))
        assert arr.metadata["num_zplanes"] == 3 and arr.metadata["num_color_channels"] == 2
        assert np.array_equal(arr[:], data)

    def test_a_zyx_stack_is_z_not_time(self, tmp_path):
        data = _tczyx()[0, 0]
        arr = mbo.imread(_write_ngff(tmp_path / "zyx.zarr", data, "zyx"))
        assert arr.shape == (1, 1, 3, 8, 10)
        assert arr.dz == pytest.approx(DZ)

    def test_a_cyx_image_is_channels(self, tmp_path):
        data = _tczyx()[0, :, 0]
        arr = mbo.imread(_write_ngff(tmp_path / "cyx.zarr", data, "cyx"))
        assert arr.shape == (1, 2, 1, 8, 10)
        assert np.array_equal(arr[0, 1, 0], data[1])

    def test_declared_dims_win(self, tmp_path):
        path = tmp_path / "plain.zarr"
        zarr.create_array(path, data=_tczyx()[0, 0])
        assert mbo.imread(path).shape == (3, 1, 1, 8, 10)
        assert mbo.imread(path, dims="ZYX").shape == (1, 1, 3, 8, 10)

    def test_asarray_is_one_frame(self, tmp_path):
        arr = mbo.imread(_write_ngff(tmp_path / "a.zarr", _tczyx(), "tczyx"))
        assert np.asarray(arr).shape == (8, 10)

    def test_gapped_selections_on_two_axes(self, tmp_path):
        data = _tczyx()
        arr = mbo.imread(_write_ngff(tmp_path / "a.zarr", data, "tczyx"))
        assert np.array_equal(arr[[0, 2, 5], 1, [0, 2]], data[[0, 2, 5]][:, 1][:, [0, 2]])

    def test_several_stores_stack_onto_z(self, tmp_path):
        data = _tczyx()[:, 0]  ## (T, Z, Y, X)
        for z in range(3):
            _write_ngff(tmp_path / "stack" / f"plane{z:02d}.zarr", data[:, z], "tyx")
        arr = mbo.imread(tmp_path / "stack")
        assert arr.shape == (6, 1, 3, 8, 10)
        assert np.array_equal(arr[:, 0], data)
        assert np.array_equal(arr[2, 0, 1], data[2, 1])

    @pytest.mark.parametrize("ext", [".tiff", ".zarr", ".h5"])
    def test_save_as_keeps_values_and_scale(self, tmp_path, ext):
        data = _tczyx()
        arr = mbo.imread(_write_ngff(tmp_path / "a.zarr", data, "tczyx"))
        out = tmp_path / "out"
        mbo.imwrite(arr, out, ext=ext, timepoints=[2, 3, 4], planes=[1, 3], overwrite=True)
        back = mbo.imread(next(out.rglob("*.zarr" if ext == ".zarr" else "*.tif" if ext == ".tiff" else "*.h5")))
        assert np.array_equal(back[:], data[1:4][:, :, [0, 2]])
        assert (back.fs, back.dx, back.dz) == pytest.approx((FS, DX, 2 * DZ))


class TestForeignOmeTiff:
    def _write(self, path):
        tifffile.imwrite(
            path, _tczyx(), ome=True, photometric="minisblack",
            metadata={"axes": "TCZYX", "PhysicalSizeX": DX, "PhysicalSizeY": DX, "PhysicalSizeZ": DZ,
                      "TimeIncrement": 1 / FS, "TimeIncrementUnit": "s"},
        )
        return path

    def test_reads_axes_and_voxel_size(self, tmp_path):
        arr = mbo.imread(self._write(tmp_path / "a.ome.tif"))
        assert arr.shape == (6, 2, 3, 8, 10)
        assert (arr.fs, arr.dx, arr.dz) == pytest.approx((FS, DX, DZ))
        assert np.array_equal(arr[:], _tczyx())

    def test_tiff_save_keeps_channel_and_plane_order(self, tmp_path):
        arr = mbo.imread(self._write(tmp_path / "a.ome.tif"))
        out = tmp_path / "out"
        mbo.imwrite(arr, out, ext=".tiff", planes=[1, 3], overwrite=True)
        back = mbo.imread(next(out.rglob("*.tif")))
        assert np.array_equal(back[:], _tczyx()[:, :, [0, 2]])
        assert "DimensionOrder" not in back.metadata
