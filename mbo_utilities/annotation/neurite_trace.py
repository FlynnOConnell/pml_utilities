"""The trace zarr: paths traced through one z-stack, and the links proposed across their gaps.

``<stem>.trace.zarr`` sits beside the stack's rois zarr and is a zarr v3 group
written whole on every save (into a temporary name, then swapped in):

```
paths/                the truth; the rois zarr's ragged encoding
  offsets               (n_paths + 1,) int64: path k is points[offsets[k]:offsets[k+1]]
  points                (n_points, 3) float32: z, y, x in voxels
links                 (n_links, 2, 3) float32: the two path ends each proposal would join
labels/               OME-NGFF 0.5 labels group, derived from paths/, never read back
  paths/              one label image: multiscales + image-label
    0                   uint32 (Z, Y, X): path k's voxels hold k + 1
attrs: mbo_trace, source {path, reader_kwargs, unit}, shape [Z, Y, X],
       voxel_um [dz, dy, dx], channel (0-based)
```

Reading it needs only zarr and numpy, so another package (masknmf-toolbox)
reads it from this layout without importing mbo_utilities.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mbo_utilities.annotation.ngff import write_label_image
from mbo_utilities.arrays.features._dim_tags import dim_to_ome_axis

__all__ = ["TRACE_SUFFIX", "TRACE_VERSION", "Trace", "read_trace", "write_trace"]

TRACE_SUFFIX = ".trace.zarr"
TRACE_VERSION = 1
TRACE_ATTR = "mbo_trace"
LABEL_IMAGE = "paths"


@dataclass
class Trace:
    """Paths through a stack, and proposed links across the gaps between them."""

    paths: dict = field(default_factory=dict)  # path id -> (n_points, 3) float32 z, y, x in voxels
    links: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 3), np.float32))  # (n_links, 2, 3)
    voxel_um: tuple = (1.0, 1.0, 1.0)  # (dz, dy, dx)
    channel: int = 0

    @property
    def next_id(self) -> int:
        return max(self.paths, default=-1) + 1


def _path_voxels(points: np.ndarray) -> np.ndarray:
    """``(n, 3)`` int64 voxels along a path, consecutive points joined by straight lines."""
    n_steps = np.maximum(np.ceil(np.abs(np.diff(points, axis=0)).max(axis=1, initial=0)), 1).astype(np.int64)
    segment = np.repeat(np.arange(len(n_steps)), n_steps)
    fraction = (np.arange(n_steps.sum()) - np.repeat(np.cumsum(n_steps) - n_steps, n_steps)) / n_steps[segment]
    along = points[segment] + fraction[:, None] * (points[segment + 1] - points[segment])
    return np.rint(np.vstack([along, points[-1:]])).astype(np.int64)


def _label_volume(trace: Trace, shape: tuple[int, int, int]) -> np.ndarray:
    """``(Z, Y, X)`` uint32 with path k's voxels set to k + 1, in ``paths`` order."""
    volume = np.zeros(shape, np.uint32)
    for k, points in enumerate(trace.paths.values()):
        voxels = _path_voxels(np.asarray(points, np.float64))
        inside = np.all((voxels >= 0) & (voxels < shape), axis=1)
        z, y, x = voxels[inside].T
        volume[z, y, x] = k + 1
    return volume


def write_trace(trace: Trace, path, *, shape: tuple[int, int, int], source: dict | None = None) -> None:
    """Write the whole trace zarr at ``path``: the paths, the links and the derived label image.

    ``shape`` is the stack's ``(Z, Y, X)``; ``source`` names it (``path``,
    ``reader_kwargs``, ``unit``) as the rois zarr does.
    """
    import zarr

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    old = path.with_name(path.name + ".old")
    for leftover in (tmp, old):
        if leftover.exists():
            shutil.rmtree(leftover)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(tmp), mode="w", zarr_format=3)
    paths = [np.asarray(points, np.float32).reshape(-1, 3) for points in trace.paths.values()]
    offsets = np.zeros(len(paths) + 1, np.int64)
    offsets[1:] = np.cumsum([len(points) for points in paths])
    group = root.create_group("paths")
    group.create_array("offsets", data=offsets)
    group.create_array("points", data=np.concatenate(paths) if paths else np.zeros((0, 3), np.float32))
    root.create_array("links", data=np.asarray(trace.links, np.float32).reshape(-1, 2, 3))
    write_label_image(
        root.create_group("labels"),
        LABEL_IMAGE,
        _label_volume(trace=trace, shape=tuple(int(n) for n in shape)),
        [dim_to_ome_axis(d) for d in "ZYX"],
        scale=list(trace.voxel_um),
    )
    root.attrs.update(
        {
            TRACE_ATTR: TRACE_VERSION,
            "source": dict(source or {}),
            "shape": [int(n) for n in shape],
            "voxel_um": [float(v) for v in trace.voxel_um],
            "channel": int(trace.channel),
        }
    )
    if path.exists():
        path.rename(old)
    tmp.rename(path)
    if old.exists():
        shutil.rmtree(old)


def read_trace(path) -> Trace:
    """Read a trace zarr back; its paths get ids ``0 .. n - 1`` in file order."""
    import zarr

    root = zarr.open_group(str(path), mode="r")
    if TRACE_ATTR not in root.attrs:
        raise ValueError(f"{path} is not a trace zarr (no {TRACE_ATTR!r} attr)")
    offsets = root["paths/offsets"][:]
    points = root["paths/points"][:]
    return Trace(
        paths={k: points[offsets[k] : offsets[k + 1]] for k in range(len(offsets) - 1)},
        links=root["links"][:],
        voxel_um=tuple(float(v) for v in root.attrs["voxel_um"]),
        channel=int(root.attrs["channel"]),
    )
