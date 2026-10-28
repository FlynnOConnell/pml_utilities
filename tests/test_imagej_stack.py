"""Single-file TIFF stacks: which axis the page dimension lands on.

ImageJ writes ``slices=N`` for any plain stack, movies included; only a real
hyperstack carries ``hyperstack=true`` and a z-stack a ``spacing``; tifffile's
``imagej=True`` default writes ``channels=N``; a plain file says nothing. The
reader puts a plain single-channel stack on T (where traces, binning and the
pipelines need it) and ``dims=`` names the axis the pages are on: ``"TYX"``,
``"ZYX"`` or ``"CYX"``, whatever the file's own tags say.
"""

import numpy as np
import pytest
import tifffile
from mbo_utilities.arrays import FrameAveragedView
from mbo_utilities.reader import imread, source_reader_kwargs


@pytest.fixture
def frames():
    return (np.random.default_rng(0).random((12, 8, 9)) * 100).astype(np.uint16)


@pytest.fixture
def plain_stack(tmp_path, frames):
    """What ImageJ writes for File > Save As > Tiff on a plain stack."""
    path = tmp_path / "plain.tif"
    description = "ImageJ=1.54f\nimages=12\nslices=12\nunit=micron\nloop=false\n"
    tifffile.imwrite(path, frames, description=description)
    return path


@pytest.fixture
def z_hyperstack(tmp_path, frames):
    path = tmp_path / "zstack.tif"
    tifffile.imwrite(
        path,
        frames,
        imagej=True,
        metadata={"axes": "ZYX", "spacing": 2.0, "unit": "um"},
    )
    return path


def test_plain_stack_reads_as_time(plain_stack, frames):
    arr = imread(plain_stack)
    assert arr._shape5d() == (12, 1, 1, 8, 9)
    assert arr.metadata["num_frames"] == 12
    np.testing.assert_array_equal(np.asarray(arr[3, 0, 0]), frames[3])


def test_plain_stack_can_be_binned(plain_stack):
    view = imread(plain_stack, frame_average=4)
    assert isinstance(view, FrameAveragedView)
    assert view.shape[0] == 3


def test_labelled_z_hyperstack_stays_on_z(z_hyperstack):
    arr = imread(z_hyperstack)
    assert arr._shape5d() == (1, 1, 12, 8, 9)


def test_dims_override_roundtrips(plain_stack, z_hyperstack):
    as_z = imread(plain_stack, dims="ZYX")
    assert as_z._shape5d() == (1, 1, 12, 8, 9)
    assert source_reader_kwargs(as_z) == {"dims": "ZYX"}
    as_t = imread(z_hyperstack, dims="TYX")
    assert as_t._shape5d() == (12, 1, 1, 8, 9)
    again = imread(z_hyperstack, **source_reader_kwargs(as_t))
    assert again._shape5d() == as_t._shape5d()
    assert source_reader_kwargs(imread(plain_stack)) == {}


@pytest.fixture
def plain_2d(tmp_path, frames):
    path = tmp_path / "image.tif"
    tifffile.imwrite(path, frames[0])
    return path


@pytest.fixture
def plain_3d(tmp_path, frames):
    """tifffile's default for a 3D array: a shaped TIFF, no ImageJ tags."""
    path = tmp_path / "pages.tif"
    tifffile.imwrite(path, frames)
    return path


@pytest.fixture
def bare_pages(tmp_path, frames):
    """One IFD per page and no description at all."""
    path = tmp_path / "bare.tif"
    with tifffile.TiffWriter(path) as writer:
        for frame in frames:
            writer.write(frame, contiguous=False, metadata=None)
    return path


@pytest.fixture
def imagej_default(tmp_path, frames):
    """``imwrite(x, imagej=True)`` on a 3D array: ``channels=N, hyperstack``."""
    path = tmp_path / "ij_default.tif"
    tifffile.imwrite(path, frames, imagej=True)
    return path


def test_a_2d_image_is_one_frame_whatever_dims_says(plain_2d, frames):
    assert imread(plain_2d)._shape5d() == (1, 1, 1, 8, 9)
    assert imread(plain_2d, dims="ZYX")._shape5d() == (1, 1, 1, 8, 9)
    np.testing.assert_array_equal(np.asarray(imread(plain_2d)[0, 0, 0]), frames[0])


def test_a_plain_3d_tiff_reads_every_page_as_time(plain_3d, bare_pages, frames):
    for path in (plain_3d, bare_pages):
        arr = imread(path)
        assert arr._shape5d() == (12, 1, 1, 8, 9)
        assert arr.fs is None
        np.testing.assert_array_equal(np.asarray(arr[5, 0, 0]), frames[5])


def test_a_plain_3d_tiff_takes_its_page_axis_from_dims(plain_3d, bare_pages, frames):
    for path in (plain_3d, bare_pages):
        as_z = imread(path, dims="ZYX")
        assert as_z._shape5d() == (1, 1, 12, 8, 9)
        assert source_reader_kwargs(as_z) == {"dims": "ZYX"}
        np.testing.assert_array_equal(np.asarray(as_z[0, 0, 5]), frames[5])
        as_c = imread(path, dims="CYX")
        assert as_c._shape5d() == (1, 12, 1, 8, 9)
        np.testing.assert_array_equal(np.asarray(as_c[0, 5, 0]), frames[5])
        assert imread(path, dims="TYX")._shape5d() == (12, 1, 1, 8, 9)


def test_tifffiles_imagej_default_is_channels_until_dims_says_otherwise(
    imagej_default, frames
):
    assert imread(imagej_default)._shape5d() == (1, 12, 1, 8, 9)
    as_t = imread(imagej_default, dims="TYX")
    assert as_t._shape5d() == (12, 1, 1, 8, 9)
    np.testing.assert_array_equal(np.asarray(as_t[5, 0, 0]), frames[5])
    as_z = imread(imagej_default, dims="ZYX")
    assert as_z._shape5d() == (1, 1, 12, 8, 9)
    np.testing.assert_array_equal(np.asarray(as_z[0, 0, 5]), frames[5])


def test_imagej_slices_read_as_time_and_dims_keeps_them_on_z(plain_stack, frames):
    assert imread(plain_stack)._shape5d() == (12, 1, 1, 8, 9)
    as_z = imread(plain_stack, dims="ZYX")
    assert as_z._shape5d() == (1, 1, 12, 8, 9)
    np.testing.assert_array_equal(np.asarray(as_z[0, 0, 5]), frames[5])
    assert imread(plain_stack, dims="CYX")._shape5d() == (1, 12, 1, 8, 9)
