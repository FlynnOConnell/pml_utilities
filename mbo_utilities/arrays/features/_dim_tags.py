"""
dimension tags for filename generation.

extensible system for labeling output files with dimension ranges.
tags follow array dims order and use format: {label}{start}[-{stop}[-{step}]]

examples:
    tp00001-10000_zplane01-14.tif  (TZYX)
    zplane01-14.tif                (ZYX, single timepoint)
    tp00001-10000.tif              (TYX, single plane)

the same vocabulary reads tags back out of a filename (``filename_tags``):
``mouse_V1_session1.tif`` carries ``session01``, ``plane_03.bin`` carries
``zplane03``; results files are named from them (``mbo_utilities.results``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# spatial dims that don't appear in filenames
SPATIAL_DIMS = {"Y", "X"}

# map dimension name aliases to canonical single-letter keys
DIM_ALIASES: dict[str, str] = {
    "timepoints": "T",
    "timepoint": "T",
    "time": "T",
    "frames": "T",
    "frame": "T",
    "t": "T",
    "z-planes": "Z",
    "z-plane": "Z",
    "z-slices": "Z",
    "z-slice": "Z",
    "zplane": "Z",
    "zplanes": "Z",
    "plane": "Z",
    "planes": "Z",
    "slice": "Z",
    "slices": "Z",
    "depth": "Z",
    "z": "Z",
    "channel": "C",
    "channels": "C",
    "color": "C",
    "colors": "C",
    "ch": "C",
    "c": "C",
    "view": "V",
    "views": "V",
    "volume": "V",
    "volumes": "V",
    "v": "V",
    "roi": "R",
    "rois": "R",
    "region": "R",
    "regions": "R",
    "r": "R",
    "beamlet": "B",
    "beamlets": "B",
    "b": "B",
    "camera": "A",
    "cameras": "A",
    "cm": "A",
    "a": "A",
    "session": "S",
    "sessions": "S",
    "s": "S",
    "y": "Y",
    "x": "X",
    "height": "Y",
    "width": "X",
}


@dataclass(frozen=True)
class TagDefinition:
    """definition for a dimension tag type."""

    label: str  # filename prefix: "tp", "zplane", "cam"
    description: str  # human readable
    zero_pad: int = 2  # pad with how many digits could this value be


# built-in tag definitions
TAG_REGISTRY: dict[str, TagDefinition] = {
    "T": TagDefinition("tp", "timepoint", zero_pad=5),
    "Z": TagDefinition("zplane", "z-plane", zero_pad=2),
    "C": TagDefinition("ch", "channel", zero_pad=2),
    "V": TagDefinition("view", "view", zero_pad=2),
    "R": TagDefinition("roi", "region", zero_pad=2),
    "S": TagDefinition("session", "session", zero_pad=2),
    # future extensions
    "B": TagDefinition("beamlet", "beamlet", zero_pad=2),
    "A": TagDefinition("cm", "camera", zero_pad=2),
}

# one filename token: label, start, optional stop and step, bounded by separators
_TAG_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z]+)[_-]?(\d+)(?:-(\d+)(?:-(\d+))?)?(?![A-Za-z0-9])"
)


def parse_tag(token: str) -> DimensionTag | None:
    """The tag a filename token spells, or None when its label is not in the vocabulary.

    ``session1`` and ``session_01`` are ``session01``; ``plane3`` is ``zplane03``;
    ``tp00001-10000`` keeps its range. ``stan112`` is not a tag, and neither is a
    one-letter label (``V1`` is a brain region, not a view).
    """
    match = _TAG_TOKEN.fullmatch(token)
    if match is None:
        return None
    label, start, stop, step = match.groups()
    key = DIM_ALIASES.get(label.lower())
    if key is None:
        key = next((k for k, d in TAG_REGISTRY.items() if d.label == label.lower()), None)
    if key is None or key not in TAG_REGISTRY:
        return None
    return DimensionTag(
        TAG_REGISTRY[key], int(start), None if stop is None else int(stop), 1 if step is None else int(step),
    )


def filename_tags(path) -> list[DimensionTag]:
    """Every tag in a file or folder name, in the order written.

    A tag is a ``label``-``number`` token whose label :func:`parse_tag` knows,
    so ``mouse_V1_GCaMP6f_session01.tif`` yields ``[session01]``,
    ``plane_03.bin`` yields ``[zplane03]`` and ``tp00001-01574_zplane01-14.tif``
    yields both of its tags.
    """
    from pathlib import Path

    name = Path(path).name
    stem = name[: -len(".zarr")] if name.endswith(".zarr") else Path(name).stem
    tags = [parse_tag(m.group(0)) for m in _TAG_TOKEN.finditer(stem)]
    return [t for t in tags if t is not None]


@dataclass
class DimensionTag:
    """single dimension tag instance with range information."""

    definition: TagDefinition
    start: int  # 1-based start index
    stop: int | None  # 1-based stop index (None = single value)
    step: int = 1  # step size (default 1)

    def to_string(self) -> str:
        """format as filename component.

        examples:
            zplane01       (single value)
            zplane01-14    (range, step=1)
            zplane01-14-2  (range with step)
        """
        pad = self.definition.zero_pad
        label = self.definition.label

        if self.stop is None or self.start == self.stop:
            # single value
            return f"{label}{self.start:0{pad}d}"

        if self.step == 1:
            # range without step
            return f"{label}{self.start:0{pad}d}-{self.stop:0{pad}d}"

        # range with step
        return f"{label}{self.start:0{pad}d}-{self.stop:0{pad}d}-{self.step}"

    @classmethod
    def from_dim_size(
        cls,
        definition: TagDefinition,
        size: int,
        selection: slice | list | Sequence[int] | None = None,
    ) -> "DimensionTag":
        """create from dimension size and optional selection.

        parameters
        ----------
        definition : TagDefinition
            tag type definition
        size : int
            total size of dimension
        selection : slice | list | Sequence[int] | None
            optional selection (indices are 1-based)

        returns
        -------
        DimensionTag
            tag with appropriate range
        """
        if selection is None:
            # full range
            if size == 1:
                return cls(definition, start=1, stop=None, step=1)
            return cls(definition, start=1, stop=size, step=1)

        if isinstance(selection, slice):
            # convert slice to range
            start = (selection.start or 0) + 1  # convert to 1-based
            stop = selection.stop if selection.stop else size
            step = selection.step or 1
            if start == stop or (stop - start) <= step:
                return cls(definition, start=start, stop=None, step=1)
            return cls(definition, start=start, stop=stop, step=step)

        # list/sequence of indices (assumed 1-based)
        indices = list(selection)
        if len(indices) == 1:
            return cls(definition, start=indices[0], stop=None, step=1)

        # check if contiguous with consistent step
        if len(indices) >= 2:
            step = indices[1] - indices[0]
            is_uniform = all(
                indices[i + 1] - indices[i] == step for i in range(len(indices) - 1)
            )
            if is_uniform and step > 0:
                return cls(definition, start=indices[0], stop=indices[-1], step=step)

        # non-uniform selection - use start-stop with step showing non-uniform
        return cls(definition, start=indices[0], stop=indices[-1], step=1)


class OutputFilename:
    """builds output filename from dimension tags."""

    def __init__(self, tags: list[DimensionTag], suffix: str = ""):
        self.tags = tags
        self.suffix = suffix

    def build(self, ext: str = ".tif") -> str:
        """build filename string.

        returns
        -------
        str
            filename like "tp00001-10000_zplane01-14.tif"; ``ext=""`` names a
            folder ("ch01_zplane02")
        """
        if ext and not ext.startswith("."):
            ext = "." + ext

        parts = [tag.to_string() for tag in self.tags]
        if self.suffix:
            parts.append(self.suffix)
        if not parts:
            parts = ["data"]
        return "_".join(parts) + ext

    @classmethod
    def from_array(
        cls,
        arr,
        planes: int | list | Sequence[int] | None = None,
        frames: int | list | Sequence[int] | None = None,
        channels: int | list | Sequence[int] | None = None,
        suffix: str = "",
    ) -> "OutputFilename":
        """create from array and optional selections.

        parameters
        ----------
        arr : array-like
            array with shape and dims
        planes : int | list | None
            z-plane selection (1-based indices)
        frames : int | list | None
            timepoint selection (1-based indices)
        channels : int | list | None
            channel selection (1-based indices)
        suffix : str
            filename suffix (default "" = no suffix)

        returns
        -------
        OutputFilename
            builder with tags derived from array dims
        """
        from mbo_utilities.arrays.features._dim_labels import get_dims

        dims = get_dims(arr)
        shape = arr.shape
        tags = []

        for i, dim_name in enumerate(dims):
            if dim_name in SPATIAL_DIMS:
                continue

            # normalize dimension name to canonical key (T, Z, C, etc.)
            dim_key = DIM_ALIASES.get(dim_name.lower(), dim_name.upper())

            if dim_key not in TAG_REGISTRY:
                continue

            definition = TAG_REGISTRY[dim_key]
            dim_size = shape[i]

            # apply selection if matching dimension
            selection = None
            if dim_key == "Z" and planes is not None:
                selection = [planes] if isinstance(planes, int) else planes
            elif dim_key == "T" and frames is not None:
                selection = [frames] if isinstance(frames, int) else frames
            elif dim_key == "C" and channels is not None:
                selection = [channels] if isinstance(channels, int) else channels

            tag = DimensionTag.from_dim_size(definition, dim_size, selection)
            tags.append(tag)

        return cls(tags, suffix=suffix)

    def __repr__(self) -> str:
        return f"OutputFilename({self.build()!r})"


def normalize_dims(dims: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """convert descriptive dim names to canonical single-letter form.

    parameters
    ----------
    dims : tuple[str, ...] | list[str]
        dimension labels like ("timepoints", "z-planes", "Y", "X")

    returns
    -------
    tuple[str, ...]
        canonical form like ("T", "Z", "Y", "X")

    examples
    --------
    >>> normalize_dims(("timepoints", "z-planes", "Y", "X"))
    ('T', 'Z', 'Y', 'X')
    >>> normalize_dims(("T", "C", "Z", "Y", "X"))
    ('T', 'C', 'Z', 'Y', 'X')
    """
    result = []
    for d in dims:
        # check aliases first, then uppercase single letter
        canonical = DIM_ALIASES.get(d.lower(), d.upper())
        result.append(canonical)
    return tuple(result)


def get_ome_axis_type(dim: str) -> str:
    """get ome-zarr ngff axis type for a dimension.

    parameters
    ----------
    dim : str
        canonical dimension label (T, C, Z, Y, X, etc.)

    returns
    -------
    str
        "time", "channel", or "space"
    """
    dim = dim.upper()
    if dim == "T":
        return "time"
    elif dim == "C":
        return "channel"
    else:
        return "space"  # Z, Y, X, V, R, B, S all spatial


def get_ome_axis_unit(dim: str) -> str | None:
    """get ome-zarr ngff axis unit for a dimension.

    parameters
    ----------
    dim : str
        canonical dimension label

    returns
    -------
    str | None
        unit string or None if no unit (e.g., channel)
    """
    dim = dim.upper()
    if dim == "T":
        return "second"
    elif dim in ("Z", "Y", "X"):
        return "micrometer"
    return None  # C, V, R, B have no unit


def dim_to_ome_axis(dim: str) -> dict:
    """build ome-zarr ngff axis definition from canonical dimension label.

    parameters
    ----------
    dim : str
        canonical dimension label (T, C, Z, Y, X, etc.)

    returns
    -------
    dict
        ome-zarr axis definition with name, type, and optional unit
    """
    dim = dim.upper()
    axis = {
        "name": dim.lower(),
        "type": get_ome_axis_type(dim),
    }
    unit = get_ome_axis_unit(dim)
    if unit:
        axis["unit"] = unit
    return axis


def dims_to_ome_axes(dims: tuple[str, ...] | list[str] | str) -> list[dict]:
    """convert dimension labels to ome-zarr ngff axes list.

    parameters
    ----------
    dims : tuple[str, ...] | list[str] | str
        dimension labels like ("T", "C", "Z", "Y", "X") or "TCZYX"

    returns
    -------
    list[dict]
        list of ome-zarr axis definitions

    examples
    --------
    >>> dims_to_ome_axes(("T", "Z", "Y", "X"))
    [{'name': 't', 'type': 'time', 'unit': 'second'},
     {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
     {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
     {'name': 'x', 'type': 'space', 'unit': 'micrometer'}]
    """
    if isinstance(dims, str):
        dims = tuple(dims.upper())
    # normalize first if needed
    canonical = normalize_dims(dims)
    return [dim_to_ome_axis(d) for d in canonical]
