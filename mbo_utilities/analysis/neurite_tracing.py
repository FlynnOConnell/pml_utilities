"""Automatic tracing of one channel of a z-stack into paths that meet at junctions.

Each plane is ridge filtered (Sato et al. 1998, doi:10.1016/S1361-8415(98)80009-1), the
ridge volume is thresholded at two levels with hysteresis (Canny 1986,
doi:10.1109/TPAMI.1986.4767851), and the mask is thinned to a one-voxel skeleton (Lee et al.
1994). skan cuts the skeleton into paths at every junction. Path ends in separate pieces that
lie close together become link proposals, which the annotator accepts or rejects.

The default parameters are an untuned first guess, made on one in vivo two-photon stack.
"""

import numpy as np
import scipy.ndimage
import scipy.spatial
import skimage.filters
import skimage.morphology

from mbo_utilities.annotation.neurite_trace import Trace

try:
    import skan
except ImportError as e:
    raise ImportError(
        'neurite tracing needs skan: uv pip install "pml_utilities[annotate] @ '
        'git+https://github.com/FlynnOConnell/pml_utilities.git"'
    ) from e

NEIGHBOURS_3D = np.ones((3, 3, 3), bool)
SIGMAS_PX = (1.0, 2.0)
QUANTILES_HYSTERESIS = (0.985, 0.997)
N_VOXELS_MIN = 30
LINK_DISTANCE_UM = 8.0


def trace_volume(
    volume: np.ndarray,
    voxel_um: tuple,
    channel: int,
    sigmas_px: tuple = SIGMAS_PX,
    quantiles_hysteresis: tuple = QUANTILES_HYSTERESIS,
    n_voxels_min: int = N_VOXELS_MIN,
    link_distance_um: float = LINK_DISTANCE_UM,
) -> Trace:
    """
    Trace one channel of a stack.

    Parameters
    ----------
    volume : np.ndarray
        One channel. Shape (Z, Y, X).
    voxel_um : tuple
        ``(dz, dy, dx)`` in microns.
    channel : int
        Which channel ``volume`` is, kept with the trace.
    sigmas_px : tuple
        Line widths the ridge filter looks for, in pixels.
    quantiles_hysteresis : tuple
        ``(low, high)`` quantiles of the ridge volume used as the two threshold levels.
    n_voxels_min : int
        Pieces of the mask smaller than this are dropped before thinning.
    link_distance_um : float
        Path ends in separate pieces closer than this are proposed as links.

    Returns
    -------
    Trace
        Paths ``0 .. n - 1`` in skan's order, and the link proposals.
    """
    steps = skeleton_steps(
        volume=volume, sigmas_px=sigmas_px, quantiles_hysteresis=quantiles_hysteresis, n_voxels_min=n_voxels_min
    )
    return trace_of_skeleton(
        skeleton=steps["skeleton"], voxel_um=voxel_um, channel=channel, link_distance_um=link_distance_um
    )


def skeleton_steps(
    volume: np.ndarray,
    sigmas_px: tuple = SIGMAS_PX,
    quantiles_hysteresis: tuple = QUANTILES_HYSTERESIS,
    n_voxels_min: int = N_VOXELS_MIN,
) -> dict:
    """
    Ridge filter, threshold and thin one channel of a stack, keeping the volume after each step.

    Parameters
    ----------
    volume : np.ndarray
        One channel. Shape (Z, Y, X).
    sigmas_px : tuple
        Line widths the ridge filter looks for, in pixels.
    quantiles_hysteresis : tuple
        ``(low, high)`` quantiles of the ridge volume used as the two threshold levels.
    n_voxels_min : int
        Pieces of the mask smaller than this are dropped before thinning.

    Returns
    -------
    dict
        ``ridge`` float, ``levels`` ``(low, high)``, ``mask_hysteresis`` and ``mask`` (small pieces dropped)
        and ``skeleton`` bool, each volume shape (Z, Y, X).
    """
    ## NOTE: per plane, because the plane spacing (3 um on the stacks this was made on) is wider than any neurite
    ridge = np.stack([skimage.filters.sato(plane, sigmas=sigmas_px, black_ridges=False) for plane in volume])
    low, high = np.quantile(ridge, quantiles_hysteresis)
    mask_hysteresis = skimage.filters.apply_hysteresis_threshold(ridge, low, high)
    mask = skimage.morphology.remove_small_objects(mask_hysteresis, max_size=n_voxels_min, connectivity=3)
    skeleton = skimage.morphology.skeletonize(mask).astype(bool)
    return {"ridge": ridge, "levels": (low, high), "mask_hysteresis": mask_hysteresis, "mask": mask, "skeleton": skeleton}


def trace_of_skeleton(skeleton: np.ndarray, voxel_um: tuple, channel: int, link_distance_um: float = LINK_DISTANCE_UM) -> Trace:
    """
    Cut a skeleton into paths at its junctions, and propose links between the ends of its pieces.

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean, one voxel thick. Shape (Z, Y, X).
    voxel_um : tuple
        ``(dz, dy, dx)`` in microns.
    channel : int
        Which channel the skeleton came from, kept with the trace.
    link_distance_um : float
        Path ends in separate pieces closer than this are proposed as links.

    Returns
    -------
    Trace
        Paths ``0 .. n - 1`` in skan's order, and the link proposals.
    """
    if not skeleton.any():
        return Trace(voxel_um=tuple(voxel_um), channel=channel)
    graph = skan.Skeleton(skeleton)
    paths = {index: graph.path_coordinates(index).astype(np.float32) for index in range(graph.n_paths)}
    links = links_between_pieces(skeleton=skeleton, voxel_um=voxel_um, link_distance_um=link_distance_um)
    return Trace(paths=paths, links=links, voxel_um=tuple(voxel_um), channel=channel)


def links_between_pieces(skeleton: np.ndarray, voxel_um: tuple, link_distance_um: float) -> np.ndarray:
    """
    Pairs of skeleton ends in separate pieces that lie closer than ``link_distance_um``.

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean, one voxel thick. Shape (Z, Y, X).
    voxel_um : tuple
        ``(dz, dy, dx)`` in microns.
    link_distance_um : float
        Largest gap proposed.

    Returns
    -------
    np.ndarray
        Shape (n_links, 2, 3) float32, the two ends as (z, y, x) in voxels.
    """
    n_neighbours = scipy.ndimage.convolve(skeleton.astype(np.int32), NEIGHBOURS_3D.astype(np.int32), mode="constant")
    ends = np.argwhere(skeleton & (n_neighbours - 1 == 1))  ## (n_ends, 3)
    pieces = scipy.ndimage.label(skeleton, structure=NEIGHBOURS_3D)[0]
    piece_of_end = pieces[tuple(ends.T)]
    tree = scipy.spatial.cKDTree(ends * np.asarray(voxel_um))
    pairs = np.array(sorted(tree.query_pairs(link_distance_um)), dtype=np.int64).reshape(-1, 2)
    pairs = pairs[piece_of_end[pairs[:, 0]] != piece_of_end[pairs[:, 1]]]
    return ends[pairs].astype(np.float32)

