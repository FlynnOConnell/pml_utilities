"""Mean traces of drawn masks: the pixel mean per frame and a neuropil ring around it."""

from __future__ import annotations

import numpy as np

from mbo_utilities.annotation.store import RoiLabelStore
from mbo_utilities.arrays.plane import PlaneMovie

__all__ = ["feather_mask", "masks_label_image", "measure_mean", "plane_masks"]


def plane_masks(
    store: RoiLabelStore, z: int, indices: list[int]
) -> tuple[list[np.ndarray], list[int]]:
    """``(masks, kept)`` for the ROIs of ``indices`` that live on plane ``z``:
    one ``(Y, X)`` bool mask per ROI, in ``kept`` order (their store indices).
    """
    kept = [i for i in indices if store.rois[i].plane == int(z)]
    return [store.mask(i) for i in kept], kept


def masks_label_image(masks: list[np.ndarray]) -> np.ndarray:
    """``(Y, X)`` uint16 numbering ``masks`` ``1..K`` (suite2p / cellpose
    mask convention), the smallest on top where they overlap.
    """
    out = np.zeros(masks[0].shape, np.uint16)
    for k in sorted(range(len(masks)), key=lambda k: -int(masks[k].sum())):
        out[masks[k]] = k + 1
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("empty mask")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def feather_mask(mask: np.ndarray, edge_width: int = 3) -> np.ndarray:
    """Soft-edged weights for a binary mask, 1 in the interior falling off
    over ``edge_width`` px toward the boundary (lbm_suite2p_python's
    feathering, kept inside the mask so trace weights never leave it).
    """
    from scipy.ndimage import distance_transform_edt

    inside = distance_transform_edt(np.asarray(mask, bool))
    return np.clip(inside / max(int(edge_width), 1), 0.0, 1.0).astype(np.float32)


def _factorized(arr) -> bool:
    """True for masknmf factorized arrays, whose ``__getitem__`` reconstructs
    only the requested crop; frame batching then just adds overhead.
    """
    return callable(getattr(arr, "getitem_tensor", None))


def _sparse_weights(masks: list[np.ndarray], box) -> scipy.sparse.csr_matrix:  # noqa: F821
    """(K, h*w) CSR averaging each ``(Y, X)`` bool mask inside ``box``."""
    from scipy.sparse import csr_matrix

    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    indptr = np.zeros(len(masks) + 1, np.int64)
    indices, data = [], []
    for k, m in enumerate(masks):
        pix = np.flatnonzero(m[y0:y1, x0:x1])
        indptr[k + 1] = indptr[k] + pix.size
        indices.append(pix)
        data.append(np.full(pix.size, 1.0 / max(pix.size, 1), np.float32))
    if not masks:
        return csr_matrix((0, h * w), dtype=np.float32)
    return csr_matrix(
        (np.concatenate(data), np.concatenate(indices), indptr),
        shape=(len(masks), h * w),
        dtype=np.float32,
    )


def _neuropil_annulus(
    cell: np.ndarray, exclude: np.ndarray, inner: int, outer: int, min_pixels: int
) -> np.ndarray:
    """``(Y, X)`` bool ring around ``cell`` excluding every pixel of
    ``exclude`` (every ROI on the plane, selected or not).

    Grows ``outer`` until ``min_pixels`` free pixels are found (or the frame
    is exhausted), like suite2p's ``create_neuropil_masks``.
    """
    from scipy.ndimage import binary_dilation

    limit = max(cell.shape)
    while True:
        inner_zone = binary_dilation(cell, iterations=max(inner, 1))
        outer_zone = binary_dilation(cell, iterations=max(outer, inner + 1))
        ring = outer_zone & ~inner_zone & ~exclude
        if ring.sum() >= min_pixels or outer >= limit:
            return ring
        outer *= 2


def measure_mean(
    movie: PlaneMovie,
    masks: list[np.ndarray],
    exclude: np.ndarray | None = None,
    *,
    neuropil: bool = True,
    inner: int = 2,
    outer: int = 8,
    min_npix: int = 350,
    batch: int = 500,
) -> tuple[np.ndarray, np.ndarray | None]:
    """``(F, Fneu)`` as ``(K, T)`` float32: the uniform mean of each mask's
    pixels per frame and of a neuropil ring around it that avoids every
    pixel of ``exclude`` (every drawn ROI on the plane; the masks themselves
    when None); ``Fneu`` is None without a neuropil.

    The one measurement behind a ``mean`` row: the quick trace, the pixel
    average (no ring) and the ``mean`` engine all call it, so a row is the
    same numbers whoever made it. Reads only the union bounding box of the
    masks and rings, in ``batch``-frame blocks (one read on a masknmf
    factorized array, which reconstructs just the crop).
    """
    T, ly, lx = movie.shape
    K = len(masks)
    if _factorized(movie.arr):
        batch = T
    if exclude is None:
        exclude = np.zeros((ly, lx), bool)
        for m in masks:
            exclude |= m
    rings = (
        [_neuropil_annulus(m, exclude, inner, outer, min_npix) for m in masks]
        if neuropil
        else []
    )
    union = np.zeros((ly, lx), bool)
    for m in masks + rings:
        union |= m
    box = _bbox(union)
    y0, y1, x0, x1 = box
    Wc = _sparse_weights(masks, box)
    Wn = _sparse_weights(rings, box) if neuropil else None
    F = np.zeros((K, T), np.float32)
    Fneu = np.zeros((K, T), np.float32) if neuropil else None
    for t0 in range(0, T, batch):
        blk = movie.frames(t0, min(T, t0 + batch), slice(y0, y1), slice(x0, x1))
        flat = blk.reshape(blk.shape[0], -1).astype(np.float32, copy=False)
        F[:, t0 : t0 + flat.shape[0]] = Wc @ flat.T
        if Wn is not None:
            Fneu[:, t0 : t0 + flat.shape[0]] = Wn @ flat.T
    return F, Fneu
