"""Every reader through the same AGENTS.md §5 and §6 checks, from a small real file of its format.

A reader that breaks a rule is marked ``xfail(strict=True)`` with the reason, so the
break shows in every run and the mark has to go when it is fixed. Add a format by
adding a ``make_<name>`` function and a row to ``SOURCES``.
"""

import numpy as np
import pytest
import tifffile
import zarr

import mbo_utilities as mbo
from mbo_utilities.metadata import get_param
from mbo_utilities.reader import source_reader_kwargs

from tests.scanimage_tiff import scanimage_header, write_scanimage_files

FS, DX, DZ = 10.0, 2.0, 5.0
MBO_ALIASES = {"nframes", "num_frames", "n_frames", "num_planes", "nplanes", "frame_rate"}  # §6.4.2; a format's own keys (SizeT, frames) are allowed


def _tczyx():
    """(T, C, Z, Y, X) int16, every (t, c, z) frame a distinct value."""
    t, c, z = np.meshgrid(np.arange(6), np.arange(2), np.arange(3), indexing="ij")
    return np.broadcast_to((t * 100 + c * 10 + z)[..., None, None], (6, 2, 3, 8, 10)).astype(np.int16).copy()


def make_numpy(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, _tczyx()[:, 0])
    return mbo.imread(path), (6, 1, 3, 8, 10), {}


def make_ome_tiff(tmp_path):
    path = tmp_path / "a.ome.tif"
    tifffile.imwrite(
        path, _tczyx(), ome=True, photometric="minisblack",
        metadata={"axes": "TCZYX", "PhysicalSizeX": DX, "PhysicalSizeY": DX, "PhysicalSizeZ": DZ,
                  "TimeIncrement": 1 / FS, "TimeIncrementUnit": "s"},
    )
    return mbo.imread(path), (6, 2, 3, 8, 10), {"fs": FS, "dz": DZ, "dx": DX}


def make_imagej_tiff(tmp_path):
    path = tmp_path / "a.tif"
    tifffile.imwrite(
        path, _tczyx().transpose(0, 2, 1, 3, 4), imagej=True,
        resolution=(1 / DX, 1 / DX), metadata={"axes": "TZCYX", "spacing": DZ, "finterval": 1 / FS, "unit": "um"},
    )
    return mbo.imread(path), (6, 2, 3, 8, 10), {"fs": FS, "dz": DZ, "dx": DX}


def make_ome_zarr(tmp_path):
    path = tmp_path / "a.zarr"
    axes = [{"name": "t", "type": "time", "unit": "second"}, {"name": "c", "type": "channel"}] + [
        {"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"
    ]
    group = zarr.open_group(path, mode="w", zarr_format=3)
    group.attrs["ome"] = {"version": "0.5", "multiscales": [{"axes": axes, "datasets": [
        {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1 / FS, 1.0, DZ, DX, DX]}]}]}]}
    group.create_array("0", data=_tczyx(), dimension_names=list("tczyx"))
    return mbo.imread(path), (6, 2, 3, 8, 10), {"fs": FS, "dz": DZ, "dx": DX}


def make_h5(tmp_path):
    import h5py

    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as f:
        f["mov"] = _tczyx()[:, 0]
    return mbo.imread(path), (6, 1, 3, 8, 10), {}


def make_scanimage_piezo(tmp_path):
    header = scanimage_header(
        hStackManager__enable=True, hStackManager__stackMode="fast", hStackManager__numSlices=3,
        hStackManager__framesPerSlice=float("inf"), hStackManager__actualStackZStepSize=DZ,
        hStackManager__numFramesPerVolume=3, hStackManager__numFramesPerVolumeWithFlyback=3,
        hFastZ__enable=True, hRoiManager__scanVolumeRate=FS,
    )
    write_scanimage_files(tmp_path / "piezo", [12, 6], header)
    return mbo.imread(tmp_path / "piezo", fix_phase=False), (6, 1, 3, 16, 12), {"fs": FS, "dz": DZ}


def make_scanimage_lbm(tmp_path):
    header = scanimage_header(hChannels__channelSave=list(range(1, 7)), hRoiManager__scanFrameRate=FS)
    write_scanimage_files(tmp_path / "lbm", [24], header)
    arr = mbo.imread(tmp_path / "lbm", fix_phase=False)
    arr.metadata = {"dz": DZ}
    return arr, (4, 1, 6, 16, 12), {"fs": FS, "dz": DZ}


SOURCES = {
    "numpy": make_numpy,
    "ome_tiff": make_ome_tiff,
    "imagej_tiff": make_imagej_tiff,
    "ome_zarr": make_ome_zarr,
    "h5": make_h5,
    "scanimage_piezo": make_scanimage_piezo,
    "scanimage_lbm": make_scanimage_lbm,
}
KNOWN_BREAKS = {  # (source, check) -> why; strict xfail, so a fix forces this entry out
}


@pytest.fixture(params=list(SOURCES))
def source(request, tmp_path):
    arr, shape, physical = SOURCES[request.param](tmp_path)
    return request.param, arr, shape, physical


def _known(name, check):
    reason = KNOWN_BREAKS.get((name, check))
    if reason is not None:
        pytest.xfail(reason)


def test_shape_is_5d(source):
    name, arr, shape, _ = source
    assert arr.shape == shape and tuple(arr.dims) == tuple("TCZYX") and arr.ndim == 5


def test_metadata_counts_agree_with_shape(source):
    name, arr, shape, _ = source
    _known(name, "counts")
    md = arr.metadata
    for key, axis in (("num_timepoints", 0), ("num_color_channels", 1), ("num_zplanes", 2)):
        assert get_param(md, key) == shape[axis], key


def test_physical_sizes(source):
    name, arr, _, physical = source
    for key, value in physical.items():
        assert getattr(arr, key) == pytest.approx(value), key


def test_metadata_getter_is_pure(source):
    name, arr, _, _ = source
    first = dict(arr.metadata)
    first["_probe"] = 1
    assert "_probe" not in arr.metadata


def test_reader_stamps_no_aliases(source):
    name, arr, _, _ = source
    _known(name, "aliases")
    assert not MBO_ALIASES & set(arr.metadata), sorted(MBO_ALIASES & set(arr.metadata))


def test_asarray_is_one_frame(source):
    name, arr, shape, _ = source
    assert np.asarray(arr).shape == shape[-2:]


def test_reopens_from_source_path(source):
    name, arr, shape, _ = source
    again = mbo.imread(arr.source_path, **source_reader_kwargs(arr))
    assert again.shape == shape
    assert np.array_equal(again[-1], arr[-1])


@pytest.mark.parametrize("ext", [".tiff", ".zarr", ".h5"])
def test_save_as_round_trip(source, ext, tmp_path):
    name, arr, shape, physical = source
    planes = [1, shape[2]] if shape[2] > 1 else [1]
    out = tmp_path / "out"
    mbo.imwrite(arr, out, ext=ext, timepoints=[2, 3, 4], planes=planes, overwrite=True)
    back = mbo.imread(next(out.rglob({".tiff": "*.tif", ".zarr": "*.zarr", ".h5": "*.h5"}[ext])))
    assert np.array_equal(back[:], arr[1:4][:, :, [p - 1 for p in planes]])
    if "fs" in physical:
        assert back.fs == pytest.approx(physical["fs"])
    if "dz" in physical and len(planes) > 1:
        assert back.dz == pytest.approx(physical["dz"] * (planes[1] - planes[0]))
