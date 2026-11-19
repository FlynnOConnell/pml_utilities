"""Raw ScanImage acquisitions of every stack type: where each (t, c, z) comes from among the pages, the metadata, and the save-as round trip.

The files are real ScanImage BigTIFFs written by ``tests/scanimage_tiff.py``; each strip pixel names its raw page.
"""

from pathlib import Path

import mbo_utilities as mbo
import numpy as np
import pytest
import tifffile
import zarr
from mbo_utilities.arrays import LBMArray, LBMPiezoArray, PiezoArray, SinglePlaneArray
from mbo_utilities.metadata import get_metadata
from mbo_utilities.metadata.io import query_tiff_pages
from mbo_utilities.metadata.scanimage import detect_stack_type

from tests.scanimage_tiff import (
    FLY_TO_VALUE,
    scanimage_header,
    strip_value,
    write_scanimage_files,
    write_scanimage_tiff,
)

ROI_HEIGHTS = [8, 8, 8]
FLY_TO = 3
WIDTH = 12


def _ports(*ports: str) -> dict:
    """Virtual channel N digitized on ``ports[N - 1]``."""
    return {f"SI.hScan2D.virtualChannelSettings__{i + 1}.source": p for i, p in enumerate(ports)}


def _open(paths, **kwargs):
    return mbo.imread(paths, fix_phase=False, **kwargs)


def _page(arr, t: int, c: int, z: int) -> int:
    """The raw page a stitched pixel of (t, c, z) came from: strip 0's value decoded."""
    return int(arr[t, c, z][0, 0]) // 8


def _single_plane(tmp_path, pages=12, keys=None, **header):
    return write_scanimage_tiff(
        tmp_path / "plane.tif",
        pages,
        scanimage_header(keys, **header),
        roi_heights=ROI_HEIGHTS,
        roi_width=WIDTH,
        fly_to_lines=FLY_TO,
    )


def _fast_z_header(slices=3, flyback=0, discard=False, keys=None, **extra):
    return scanimage_header(
        keys,
        hStackManager__enable=True,
        hStackManager__stackMode="fast",
        hStackManager__numSlices=slices,
        hStackManager__framesPerSlice=float("inf"),
        hStackManager__numVolumes=100000,
        hStackManager__actualStackZStepSize=5,
        hStackManager__numFramesPerVolume=slices,
        hStackManager__numFramesPerVolumeWithFlyback=slices + flyback,
        hFastZ__enable=True,
        hFastZ__discardFlybackFrames=discard,
        hFastZ__numDiscardFlybackFrames=flyback,
        hRoiManager__scanVolumeRate=30.0 / (slices + flyback),
        **extra,
    )


def _slow_header(slices=3, frames_per_slice=4, average=2, **extra):
    return scanimage_header(
        None,
        hStackManager__enable=True,
        hStackManager__numSlices=slices,
        hStackManager__framesPerSlice=frames_per_slice,
        hStackManager__stackZStepSize=2.5,
        hScan2D__logAverageFactor=average,
        hRoiManager__scanVolumeRate=30.0 / (slices * frames_per_slice),
        **extra,
    )


def _lbm_header(beams=4, ports=("AI0",), base=None):
    """``base`` (a plain acquisition by default) saving ``beams`` channels from each port."""
    channels = list(range(1, beams * len(ports) + 1))
    sources = [p for p in ports for _ in range(beams)]
    return (base or scanimage_header()) | {"SI.hChannels.channelSave": channels} | _ports(*sources)


class TestPageCount:
    def test_a_header_bigger_than_a_page_adds_no_phantom_page(self, tmp_path):
        path = write_scanimage_tiff(tmp_path / "a.tif", 5, scanimage_header(), roi_heights=[4], roi_width=4)
        assert query_tiff_pages(path) == 5
        assert _open(path).shape[0] == 5


class TestSinglePlane:
    def test_shape_metadata_and_stitching(self, tmp_path):
        arr = _open(_single_plane(tmp_path))
        assert isinstance(arr, SinglePlaneArray)
        assert arr.shape == (12, 1, 1, 8, 3 * WIDTH)
        assert arr.fs == 30.0
        assert arr.dx == pytest.approx(100.0 / WIDTH)
        frame = arr[5, 0, 0]
        for strip in range(3):
            assert np.all(frame[:, strip * WIDTH : (strip + 1) * WIDTH] == strip_value(5, strip))
        assert FLY_TO_VALUE not in frame

    def test_strips_come_from_the_geometry_when_the_flyto_timing_rounds_wrong(self, tmp_path):
        path = _single_plane(tmp_path, hScan2D__flytoTimePerScanfield=0)
        with tifffile.TiffFile(path) as tf:
            assert tf.pages[0].shape == (3 * 8 + 2 * FLY_TO, WIDTH)
        arr = _open(path)
        assert arr.shape[3] == 8
        assert FLY_TO_VALUE not in arr[0, 0, 0]

    def test_y_and_x_keys_crop_the_stitched_frame(self, tmp_path):
        arr = _open(_single_plane(tmp_path, keys=_ports("AI0", "AI1"), hChannels__channelSave=[1, 2]))
        full = arr[0:3, :, 0]
        assert full.shape == (3, 2, 8, 3 * WIDTH)
        assert np.array_equal(arr[0:3, 0, 0, 1:5, 10:14], full[:, 0, 1:5, 10:14])
        assert arr[1, 1, 0, 2, WIDTH] == strip_value(3, 1)
        assert np.array_equal(arr[1, 0, 0, 2, :], full[1, 0, 2])

    def test_one_roi_and_a_list_of_rois(self, tmp_path):
        path = _single_plane(tmp_path)
        assert _open(path, roi=2)[3, 0, 0].max() == strip_value(3, 1)
        first, third = _open(path, roi=[1, 3])[3, 0, 0]
        assert first.max() == strip_value(3, 0) and third.max() == strip_value(3, 2)

    def test_two_colors_interleave_on_c(self, tmp_path):
        arr = _open(_single_plane(tmp_path, keys=_ports("AI0", "AI1", "AI2", "AI3"), hChannels__channelSave=[2, 4]))
        assert arr.shape[:3] == (6, 2, 1)
        assert [_page(arr, 4, c, 0) for c in range(2)] == [8, 9]

    def test_four_pmt_channels_without_virtual_channels_are_colors_not_beamlets(self, tmp_path):
        path = _single_plane(tmp_path, hChannels__channelSave=[1, 2, 3, 4])
        assert detect_stack_type(get_metadata(path)) == "single_plane"
        arr = _open(path)
        assert arr.shape[:3] == (3, 4, 1)
        assert [_page(arr, 2, c, 0) for c in range(4)] == [8, 9, 10, 11]

    def test_a_disabled_stack_reports_no_dz(self, tmp_path):
        meta = get_metadata(_single_plane(tmp_path, hStackManager__numSlices=21, hStackManager__stackZStepSize=0.5))
        assert meta.get("dz") is None


class TestLBM:
    def test_beamlets_are_z(self, tmp_path):
        path = write_scanimage_tiff(tmp_path / "lbm.tif", 4 * 5, _lbm_header(beams=4))
        arr = _open(path)
        assert isinstance(arr, LBMArray)
        assert arr.shape[:3] == (5, 1, 4)
        assert arr.dz is None and arr.fs == 30.0
        assert [_page(arr, 2, 0, z) for z in range(4)] == [8, 9, 10, 11]

    def test_two_ports_are_colors_of_the_same_beamlets(self, tmp_path):
        path = write_scanimage_tiff(tmp_path / "lbm.tif", 6 * 3, _lbm_header(beams=3, ports=("AI0", "AI1")))
        arr = _open(path)
        assert arr.shape[:3] == (3, 2, 3)
        assert _page(arr, 1, 1, 2) == 6 + 3 + 2


class TestPiezo:
    def test_fast_z_slices_are_z_and_volumes_are_counted_from_the_pages(self, tmp_path):
        paths = write_scanimage_files(tmp_path / "acq", [10, 7], _fast_z_header(slices=3))
        arr = _open(tmp_path / "acq")
        assert isinstance(arr, PiezoArray)
        assert arr.shape[:3] == (5, 1, 3)  # 17 pages: 5 volumes, the partial sixth left out
        assert arr.fs == pytest.approx(10.0) and arr.dz == 5.0
        assert [_page(arr, t, 0, z) for t in (0, 3) for z in range(3)] == [0, 1, 2, 9, 10, 11]
        assert _open(paths[0]).shape[0] == 3

    def test_a_folder_mixing_two_acquisitions_is_refused(self, tmp_path):
        write_scanimage_files(tmp_path / "acq", [6], _fast_z_header(slices=3))
        write_scanimage_tiff(tmp_path / "acq" / "zz_other.tif", 6, _fast_z_header(slices=3), roi_width=20)
        with pytest.raises(ValueError, match="another acquisition"):
            _open(tmp_path / "acq")

    def test_flyback_frames_are_dropped(self, tmp_path):
        path = write_scanimage_tiff(tmp_path / "fb.tif", 12, _fast_z_header(slices=3, flyback=1))
        arr = _open(path)
        assert arr.shape[:3] == (3, 1, 3)
        assert [_page(arr, 2, 0, z) for z in range(3)] == [8, 9, 10]
        assert arr.fs == pytest.approx(7.5)

    def test_discarded_flyback_frames_are_not_in_the_file(self, tmp_path):
        path = write_scanimage_tiff(tmp_path / "fb.tif", 9, _fast_z_header(slices=3, flyback=1, discard=True))
        arr = _open(path)
        assert arr.shape[:3] == (3, 1, 3)
        assert _page(arr, 2, 0, 0) == 6

    def test_fast_z_with_the_stack_manager_off_is_still_a_volume(self, tmp_path):
        header = _fast_z_header(slices=2) | {"SI.hStackManager.enable": False, "SI.hStackManager.stackMode": "slow"}
        arr = _open(write_scanimage_tiff(tmp_path / "old.tif", 8, header))
        assert arr.shape[:3] == (4, 1, 2)

    def test_slow_stack_repeats_follow_their_slice_on_t(self, tmp_path):
        # 4 frames per slice averaged 2x at acquisition: 2 saved repeats per slice, 6 pages per volume
        arr = _open(write_scanimage_tiff(tmp_path / "slow.tif", 12, _slow_header()))
        assert arr.shape[:3] == (4, 1, 3)
        assert arr.frames_per_slice == 2 and arr.can_average
        assert arr.dz == 2.5 and arr.fs is None
        assert [_page(arr, t, 0, 1) for t in range(4)] == [2, 3, 8, 9]

    def test_average_frames_averages_each_slice(self, tmp_path):
        path = write_scanimage_tiff(tmp_path / "slow.tif", 12, _slow_header())
        arr = _open(path, average_frames=True)
        assert _open(path, **arr.reader_kwargs).shape == arr.shape  # what a save worker re-opens
        assert arr.shape[:3] == (2, 1, 3)
        assert arr.dtype == np.float32
        assert arr[1, 0, 2][0, 0] == np.mean([strip_value(10), strip_value(11)])
        assert arr.num_timepoints == 2 and arr.metadata["num_timepoints"] == 2
        assert arr.fs == pytest.approx(30.0 / 12)
        arr.average_frames = False
        assert arr.shape[0] == 4 and arr.metadata["num_timepoints"] == 4

    def test_one_volume_of_repeats_is_sampled_at_the_frame_rate(self, tmp_path):
        arr = _open(write_scanimage_tiff(tmp_path / "slow.tif", 6, _slow_header()))
        assert arr.shape[:3] == (2, 1, 3)
        assert arr.fs == 30.0

    def test_two_colors(self, tmp_path):
        header = _fast_z_header(slices=2, keys=_ports("AI0", "AI1"), hChannels__channelSave=[1, 2])
        arr = _open(write_scanimage_tiff(tmp_path / "c2.tif", 12, header))
        assert arr.shape[:3] == (3, 2, 2)
        assert _page(arr, 1, 1, 1) == (1 * 2 + 1) * 2 + 1

    def test_reads_whole_selections_in_order(self, tmp_path):
        arr = _open(write_scanimage_tiff(tmp_path / "fz.tif", 30, _fast_z_header(slices=3)))
        block = arr[[7, 2], :, ::2]
        assert block.shape == (2, 1, 2, 16, 12)
        assert block[:, 0, :, 0, 0].tolist() == [[strip_value(21), strip_value(23)], [strip_value(6), strip_value(8)]]


class TestPollen:
    def test_piezo_positions_on_t_and_beamlets_on_z(self, tmp_path):
        header = _lbm_header(beams=4, base=_slow_header(frames_per_slice=2, average=1))
        path = write_scanimage_tiff(tmp_path / "pollen.tif", 4 * 3 * 2, header)
        arr = _open(path)
        assert isinstance(arr, LBMPiezoArray)
        assert arr.shape[:3] == (6, 1, 4)
        assert arr.fs is None and arr.dz is None and arr.num_beamlets == 4
        assert _page(arr, 3, 0, 2) == 3 * 4 + 2  # slice 1, repeat 1
        averaged = _open(path, average_frames=True)
        assert averaged.shape[:3] == (3, 1, 4)
        assert averaged[1, 0, 2][0, 0] == np.mean([strip_value(2 * 4 + 2), strip_value(3 * 4 + 2)])


def _piezo(tmp_path) -> Path:
    header = _fast_z_header(slices=3, keys=_ports("AI0", "AI1"), hChannels__channelSave=[1, 2])
    write_scanimage_files(tmp_path / "acq", [24, 12], header, roi_heights=ROI_HEIGHTS, roi_width=WIDTH, fly_to_lines=FLY_TO)
    return tmp_path / "acq"


def _lbm(tmp_path) -> Path:
    path = write_scanimage_tiff(tmp_path / "lbm.tif", 4 * 6, _lbm_header(beams=4), roi_heights=[8, 8], roi_width=WIDTH, fly_to_lines=FLY_TO)
    return path


def _written(outdir: Path, ext: str) -> Path:
    if ext == ".zarr":
        return next(outdir.rglob("*.zarr"))
    if ext == ".bin":
        return outdir
    return next(outdir.rglob("*.tif" if ext == ".tiff" else f"*{ext}"))


class TestSaveAs:
    @pytest.mark.parametrize("ext", [".tiff", ".zarr", ".h5"])
    def test_piezo_volume_round_trips_with_selections(self, tmp_path, ext):
        arr = _open(_piezo(tmp_path))
        assert arr.shape[:3] == (6, 2, 3)
        out = tmp_path / "out"
        mbo.imwrite(arr, out, ext=ext, timepoints=[2, 3, 4, 5], planes=[1, 3], channels=2, overwrite=True)
        back = mbo.imread(_written(out, ext))
        expected = arr[1:5, 1:2, [0, 2]]
        assert back.shape == expected.shape
        assert np.array_equal(np.asarray(back[:]), expected)
        assert back.dz == pytest.approx(10.0)  # planes 1 and 3 of a 5 um stack
        assert back.fs == pytest.approx(10.0)

    @pytest.mark.parametrize("ext", [".bin", ".npy"])
    def test_piezo_planes_write_one_file_each(self, tmp_path, ext):
        arr = _open(_piezo(tmp_path))
        out = tmp_path / "out"
        mbo.imwrite(arr, out, ext=ext, planes=[2, 3], channels=1, overwrite=True)
        if ext == ".bin":
            dirs = sorted(d for d in out.iterdir() if d.is_dir())
            assert [d.name[:8] for d in dirs] == ["zplane02", "zplane03"]
            back = mbo.imread(dirs[1] / "data_raw.bin")
            assert np.array_equal(np.asarray(back[:]).reshape(6, 8, -1), arr[:, 0, 2])
            ops = np.load(dirs[1] / "ops.npy", allow_pickle=True).item()
            assert ops["fs"] == pytest.approx(10.0) and ops["nframes"] == 6
        else:
            files = sorted(out.glob("*.npy"))
            assert len(files) == 2 and "zplane03" in files[1].name

    def test_piezo_mrois_split_into_their_own_folders(self, tmp_path):
        arr = _open(_piezo(tmp_path), roi=0)
        out = tmp_path / "out"
        mbo.imwrite(arr, out, ext=".tiff", planes=1, channels=1, overwrite=True)
        assert sorted(d.name for d in out.iterdir()) == ["roi01", "roi02", "roi03"]
        back = mbo.imread(next((out / "roi02").rglob("*.tif")))
        assert back.shape == (6, 1, 1, 8, WIDTH)
        assert np.all(np.asarray(back[:])[:, 0, 0] // 8 == np.asarray([(t * 3) * 2 for t in range(6)])[:, None, None])

    def test_averaged_slow_stack_writes_float32(self, tmp_path):
        arr = _open(write_scanimage_tiff(tmp_path / "slow.tif", 12, _slow_header()), average_frames=True)
        out = tmp_path / "out"
        mbo.imwrite(arr, out, ext=".zarr", overwrite=True)
        back = mbo.imread(_written(out, ".zarr"))
        assert back.dtype == np.float32
        assert np.array_equal(np.asarray(back[:]), arr[:])

    @pytest.mark.parametrize("ext", [".tiff", ".zarr", ".h5"])
    def test_lbm_round_trips_with_the_users_dz(self, tmp_path, ext):
        arr = _open(_lbm(tmp_path))
        arr.metadata = {"dz": 16.0}
        out = tmp_path / "out"
        mbo.imwrite(arr, out, ext=ext, planes=[2, 4], timepoints=[1, 3, 5], overwrite=True)
        back = mbo.imread(_written(out, ext))
        assert np.array_equal(np.asarray(back[:]), arr[0:6:2, :, [1, 3]])
        assert back.dz == pytest.approx(32.0)
        assert back.fs == pytest.approx(15.0)

    def test_unevenly_spaced_planes_write_no_dz(self, tmp_path):
        arr = _open(_lbm(tmp_path))
        arr.metadata = {"dz": 16.0}
        out = tmp_path / "out"
        mbo.imwrite(arr, out, ext=".zarr", planes=[1, 2, 4], overwrite=True)
        assert mbo.imread(_written(out, ".zarr")).dz is None

    def test_tiff_stamps_imagej_spacing_and_interval(self, tmp_path):
        arr = _open(_piezo(tmp_path))
        out = tmp_path / "out"
        mbo.imwrite(arr, out, ext=".tiff", channels=1, overwrite=True)
        with tifffile.TiffFile(_written(out, ".tiff")) as tf:
            ij = tf.imagej_metadata
        assert ij["slices"] == 3 and ij["frames"] == 6
        assert ij["spacing"] == pytest.approx(5.0)
        assert ij["finterval"] == pytest.approx(0.1)

    def test_zarr_scale_is_seconds_and_microns(self, tmp_path):
        arr = _open(_piezo(tmp_path))
        out = tmp_path / "out"
        mbo.imwrite(arr, out, ext=".zarr", channels=1, overwrite=True)
        group = zarr.open_group(_written(out, ".zarr"), mode="r")
        scale = group.attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        assert scale[0] == pytest.approx(0.1) and scale[1] == pytest.approx(5.0)
