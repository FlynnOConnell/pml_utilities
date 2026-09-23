"""ImageJ single-file stacks: which axis the page dimension lands on.

ImageJ writes ``slices=N`` for any plain stack, movies included; only a real
hyperstack carries ``hyperstack=true`` and a z-stack a ``spacing``. The reader
puts a plain single-channel stack on T (where traces, binning and the
pipelines need it) and honours ``dims=`` as an explicit override.
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
