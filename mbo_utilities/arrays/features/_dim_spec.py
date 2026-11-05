"""Per-dimension role, size, scale and unit of an array: spatial Y/X, iteratable T/Z/C, batch camera/trial."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DimRole(StrEnum):
    """Role of a dimension in the array structure."""

    SPATIAL = "spatial"
    """spatial display dimensions (Y, X) - always form the 2D image plane."""

    ITERATABLE = "iteratable"
    """iteratable dimensions (T, Z, C) - can be scrolled/indexed with sliders."""

    BATCH = "batch"
    """batch dimensions (camera, trial) - produce separate output files."""

    @classmethod
    def from_dim_name(cls, name: str) -> DimRole:
        name_upper = name.upper()
        if name_upper in ("Y", "X"):
            return cls.SPATIAL
        if name_upper in ("CAM", "CAMERA", "TRIAL", "RUN"):
            return cls.BATCH
        return cls.ITERATABLE


DIM_UNITS: dict[str, str | None] = {
    "T": "second",
    "Z": "micrometer",
    "Y": "micrometer",
    "X": "micrometer",
    "C": None,
    "V": "micrometer",  # piezo volume
    "R": "micrometer",  # roi
    "B": "micrometer",  # beamlet
    "S": "micrometer",  # slice
}


@dataclass
class DimensionSpec:
    """One dimension: upper-case name, role, size, physical size per index (``dz``, ``1 / fs``, ``dx``; None when the source stored none) and unit (``DIM_UNITS`` when not given)."""

    name: str
    role: DimRole
    size: int
    scale: float | None = 1.0
    unit: str | None = field(default=None)

    def __post_init__(self):
        self.name = self.name.upper()
        if self.unit is None:
            self.unit = DIM_UNITS.get(self.name)

    @property
    def is_spatial(self) -> bool:
        return self.role == DimRole.SPATIAL

    @property
    def is_iteratable(self) -> bool:
        return self.role == DimRole.ITERATABLE

    @property
    def is_batch(self) -> bool:
        return self.role == DimRole.BATCH


@dataclass
class DimensionSpecs:
    """An array's dimensions in axis order, indexable by position or name."""

    specs: list[DimensionSpec]

    def __getitem__(self, idx: int | str) -> DimensionSpec:
        if isinstance(idx, int):
            return self.specs[idx]
        spec = self.get(idx)
        if spec is None:
            raise KeyError(f"No dimension named '{idx}'")
        return spec

    def get(
        self, name: str, default: DimensionSpec | None = None
    ) -> DimensionSpec | None:
        return next((s for s in self.specs if s.name == name.upper()), default)

    @classmethod
    def from_array(
        cls,
        arr,
        dims: tuple[str, ...] | None = None,
        metadata: dict | None = None,
    ) -> DimensionSpecs:
        """
        Create dimension specs from an array.

        Parameters
        ----------
        arr : array-like
            array with shape attribute
        dims : tuple of str, optional
            dimension names (e.g., ("T", "Z", "Y", "X"))
            if None, inferred from array
        metadata : dict, optional
            metadata for scale values (dx, dy, dz, fs, etc.)

        Returns
        -------
        DimensionSpecs
        """
        from mbo_utilities.arrays.features._dim_labels import get_dims
        from mbo_utilities.metadata.params import get_param

        if dims is None:
            dims = get_dims(arr)

        shape = arr.shape
        metadata = metadata or getattr(arr, "metadata", {}) or {}

        specs = []
        for i, dim_name in enumerate(dims):
            name = dim_name.upper()
            role = DimRole.from_dim_name(name)
            size = shape[i] if i < len(shape) else 1

            # pixel size defaults to the registry's 1.0; a z-step or rate nobody stored is None
            scale = 1.0
            if name == "X":
                scale = get_param(metadata, "dx", default=1.0) or 1.0
            elif name == "Y":
                scale = get_param(metadata, "dy", default=1.0) or 1.0
            elif name == "Z":
                scale = get_param(metadata, "dz") or None
            elif name == "T":
                fs = get_param(metadata, "fs")
                scale = 1.0 / fs if fs and fs > 0 else None

            specs.append(
                DimensionSpec(
                    name=name,
                    role=role,
                    size=size,
                    scale=scale,
                )
            )

        return cls(specs)

    @property
    def spatial_dims(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs if s.is_spatial)

    @property
    def iteratable_dims(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs if s.is_iteratable)

    @property
    def batch_dims(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs if s.is_batch)

    @property
    def num_timepoints(self) -> int:
        spec = self.get("T")
        return spec.size if spec else 1

    @property
    def num_zplanes(self) -> int:
        spec = self.get("Z")
        return spec.size if spec else 1

    @property
    def num_channels(self) -> int:
        spec = self.get("C")
        return spec.size if spec else 1

    @property
    def dx(self) -> float:
        spec = self.get("X")
        return spec.scale if spec else 1.0

    @property
    def dy(self) -> float:
        spec = self.get("Y")
        return spec.scale if spec else 1.0

    @property
    def dz(self) -> float | None:
        """z-step size (None without a Z dimension or a stored value)."""
        spec = self.get("Z")
        return spec.scale if spec else None

    @property
    def fs(self) -> float | None:
        """Sampling rate in Hz (None without a T dimension or a stored rate)."""
        spec = self.get("T")
        if spec and spec.scale:
            return 1.0 / spec.scale
        return None

    @property
    def finterval(self) -> float | None:
        """Sampling interval in seconds (None without a T dimension or a stored rate)."""
        spec = self.get("T")
        return spec.scale if spec else None
