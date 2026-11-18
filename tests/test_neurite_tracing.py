"""Tracing on small synthetic stacks whose answer is known."""

import numpy as np
import pytest
import skimage.draw

from mbo_utilities.analysis import neurite_tracing as tracing

VOXEL_UM = (3.0, 0.5, 0.5)


def stack_with_lines(segments: list, shape: tuple = (5, 96, 96), seed: int = 0) -> np.ndarray:
    """A noisy stack with a bright two-pixel-wide line per ``(z, (r0, c0, r1, c1))``."""
    rng = np.random.default_rng(seed)
    volume = rng.normal(10.0, 1.0, shape).astype(np.float32)
    for z, (r0, c0, r1, c1) in segments:
        rr, cc = skimage.draw.line(r0, c0, r1, c1)
        for offset in (0, 1):
            volume[z, np.clip(rr + offset, 0, shape[1] - 1), cc] = 100.0
    return volume


def test_a_branch_gives_three_paths_that_meet():
    volume = stack_with_lines([(2, (48, 10, 48, 50)), (2, (48, 50, 15, 85)), (2, (48, 50, 85, 85))])
    trace = tracing.trace_volume(volume=volume, voxel_um=VOXEL_UM, channel=0, quantiles_hysteresis=(0.98, 0.99))
    long_paths = [points for points in trace.paths.values() if len(points) > 10]
    assert len(long_paths) == 3
    ends = np.concatenate([points[[0, -1]] for points in long_paths])
    near_junction = np.linalg.norm(ends[:, 1:] - np.array([48.5, 50]), axis=1) < 4
    assert near_junction.sum() == 3
    assert len(trace.links) == 0


def test_a_gap_is_proposed_as_one_link():
    ## NOTE: the ridge filter's response spreads a few pixels past a line's end, so a gap under ~8 px closes by itself
    volume = stack_with_lines([(2, (48, 5, 48, 40)), (2, (48, 56, 48, 90))])
    trace = tracing.trace_volume(volume=volume, voxel_um=VOXEL_UM, channel=1, link_distance_um=10.0)
    assert len(trace.paths) == 2
    assert trace.links.shape == (1, 2, 3)
    gap_columns = sorted(trace.links[0, :, 2])
    assert gap_columns[0] == pytest.approx(40, abs=3) and gap_columns[1] == pytest.approx(56, abs=3)


def test_a_gap_wider_than_the_link_distance_is_not_proposed():
    volume = stack_with_lines([(2, (48, 5, 48, 30)), (2, (48, 60, 48, 90))])
    trace = tracing.trace_volume(volume=volume, voxel_um=VOXEL_UM, channel=0, link_distance_um=10.0)
    assert len(trace.paths) == 2
    assert len(trace.links) == 0


def test_an_empty_stack_gives_an_empty_trace():
    volume = np.random.default_rng(1).normal(10.0, 1.0, (5, 64, 64)).astype(np.float32)
    trace = tracing.trace_volume(volume=volume, voxel_um=VOXEL_UM, channel=0, n_voxels_min=10_000)
    assert trace.paths == {} and trace.links.shape == (0, 2, 3) and trace.next_id == 0


def test_save_and_load_keep_points_links_and_metadata(tmp_path):
    volume = stack_with_lines([(2, (48, 5, 48, 40)), (2, (48, 56, 48, 90))])
    trace = tracing.trace_volume(volume=volume, voxel_um=VOXEL_UM, channel=1, link_distance_um=10.0)
    filepath = str(tmp_path / "stack.trace.json")
    tracing.save_trace(trace=trace, filepath=filepath)
    loaded = tracing.load_trace(filepath=filepath)
    assert loaded.voxel_um == VOXEL_UM and loaded.channel == 1
    assert len(loaded.paths) == len(trace.paths)
    for points, points_loaded in zip(trace.paths.values(), loaded.paths.values()):
        np.testing.assert_allclose(points_loaded, points)
    np.testing.assert_allclose(loaded.links, trace.links)
