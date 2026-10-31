"""Framework-agnostic ROI annotation model.

``RoiLabelStore`` holds everything the manual ROI GUI edits: one
:class:`RoiRecord` per ROI (its pixels and weights, the plane it was drawn
on, its class, note and colour) and the user-defined class-label set, with
no GUI imports, so the same model can back a widget, a script, or a batch
tool. It is an :class:`Observable`: every mutation emits one ``"rois"``
event, so a view redraws from the event instead of polling.

An ROI's pixels are flat ``y * X + x`` indices into a ``(Y, X)`` frame, the
encoding a results zarr's ``rois/member`` uses, with one weight per pixel
(ones for a hand-drawn mask, a promoted component's ``lam``). ROIs may
overlap; the per-plane label image (:meth:`label_image`, cached) paints the
largest first so the smallest ROI wins a contested pixel, which is what
:meth:`roi_at` answers. Deleting an ROI shifts the indices above it; its
``uid`` is the identity that survives.

Planes: one plane per combination of the data's scrolling dims (plain z for
most data; arrays without depth get a single plane). When channels or other
scroll axes key their own masks, ``plane_axes`` records the layout
``((dim, size), ...)`` with z last, so ``nz`` is the product and a z-only
store keeps ``plane == z``. The store owns that arithmetic: :meth:`plane_of`
folds a slider position into a plane, :meth:`plane_pos` unfolds one, and
:meth:`roi_z` / :meth:`roi_c` say which z-plane and channel an ROI was drawn
on. T is always shared.
"""

from __future__ import annotations

import colorsys
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from mbo_utilities.annotation.events import Observable
from mbo_utilities.arrays.features._dim_labels import find_slider_name

__all__ = [
    "UNLABELED",
    "CLASS_COLORS",
    "ROI_COLORS",
    "class_color",
    "RoiRecord",
    "RoiLabelStore",
]

UNLABELED = -1

# tab10, as masknmf's classification gui colors the same label set
CLASS_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
)


def class_color(index: int) -> tuple[float, float, float]:
    """Rgb in 0-1 for a class label index; wraps past the palette end."""
    return CLASS_COLORS[index % len(CLASS_COLORS)]


def _make_roi_colors() -> np.ndarray:
    """180 saturated hues in shuffled order: consecutive ROIs contrast, and full saturation stays readable over gnuplot2's pale end."""
    hues = np.random.default_rng(0).permutation(180)
    return np.array(
        [
            [int(round(c * 255)) for c in colorsys.hsv_to_rgb(h / 180.0, 1.0, 1.0)]
            for h in hues
        ],
        dtype=np.uint8,
    )


ROI_COLORS = (
    _make_roi_colors()
)  # (180, 3) uint8, indexed roi_index % 180 for unlabeled rois


@dataclass
class RoiRecord:
    """One ROI: its pixels and everything the user said about it.

    ``pixels`` are flat ``y * X + x`` indices into the plane, ``weights`` one
    float per pixel; ``plane`` is the flat index into the store's planes,
    which the store decodes into a z-plane and a channel
    (:meth:`RoiLabelStore.roi_z`, :meth:`roi_c`). ``polygon`` is the outline
    as drawn, ``(m, 2)`` ``(x, y)``, None for a mask that came from a run or
    a file.
    """

    plane: int
    pixels: np.ndarray
    weights: np.ndarray
    class_index: int = UNLABELED
    note: str = ""
    uid: int = 0  # persistent id, never reused; 0 = unassigned
    source: str = ""  # "" = drawn by hand; "<run name>:<row>" = promoted
    color: tuple[int, int, int] | None = None  # explicit group color, uint8 rgb
    polygon: np.ndarray | None = None

    def __post_init__(self):
        self.pixels = np.asarray(self.pixels, np.int64).ravel()
        self.weights = (
            np.ones(self.pixels.size, np.float32)
            if self.weights is None
            else np.asarray(self.weights, np.float32).ravel()
        )
        if self.weights.size != self.pixels.size:
            raise ValueError(
                f"{self.weights.size} weights for {self.pixels.size} pixels"
            )
        if self.polygon is not None:
            self.polygon = np.asarray(self.polygon, np.float32).reshape(-1, 2)

    @property
    def area(self) -> int:
        return int(self.pixels.size)

    def copy(self) -> RoiRecord:
        return replace(
            self,
            pixels=self.pixels.copy(),
            weights=self.weights.copy(),
            polygon=None if self.polygon is None else self.polygon.copy(),
        )


class RoiLabelStore(Observable):
    """ROI records + the class-label name set over a ``(P, Y, X)`` frame.

    Every mutation emits a ``"rois"`` event whose ``info["action"]`` is one
    of ``add``, ``delete``, ``clear``, ``restore``, ``class``, ``note``,
    ``color``, ``labels``, ``tint``; ``index`` and ``uid`` name the ROI when
    one is involved. ``axis_roles`` says which plane axis is the z-plane and
    which the channel (``{"z": "ROI", "c": "Channel"}``) when their names do
    not say so; ``tint`` is a display-only color per uid that wins over every
    other color while set and is never saved.
    """

    events = ("rois",)

    def __init__(
        self,
        nz: int,
        ny: int,
        nx: int,
        label_names: tuple[str, ...] = (),
        min_pixels: int = 1,
        rois: list[RoiRecord] | None = None,
        next_uid: int = 0,
    ):
        super().__init__()
        self._shape = (int(nz), int(ny), int(nx))
        self.rois: list[RoiRecord] = list(rois) if rois else []
        for r in self.rois:
            if not 0 <= r.plane < self._shape[0]:
                raise ValueError(f"plane {r.plane} outside {self._shape[0]} planes")
            if r.pixels.size and (
                r.pixels.min() < 0 or r.pixels.max() >= self._shape[1] * self._shape[2]
            ):
                raise ValueError(f"pixels outside a {ny}x{nx} frame")
        self.label_names: tuple[str, ...] = tuple(str(n) for n in label_names)
        # ((dim name, size), ...) over the scrolling dims keying planes, z last; empty is plain z planes
        self.plane_axes: tuple[tuple[str, int], ...] = ()
        self.axis_roles: dict[str, str] = {}
        self.tint: dict[int, tuple[int, int, int]] = {}
        self.min_pixels = int(min_pixels)
        self._label_images: dict[int, np.ndarray] = {}
        self.next_uid = max(
            int(next_uid), max((r.uid for r in self.rois), default=0) + 1, 1
        )
        seen: set[int] = set()
        for r in self.rois:
            if r.uid <= 0 or r.uid in seen:
                r.uid = self.next_uid
                self.next_uid += 1
            seen.add(r.uid)

    @property
    def nz(self) -> int:
        return self._shape[0]

    @property
    def ny(self) -> int:
        return self._shape[1]

    @property
    def nx(self) -> int:
        return self._shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def axis_name(self, role: str) -> str | None:
        """The plane-axis name playing ``role`` (``"z"`` or ``"c"``), or None
        when the volume is not keyed by that axis: ``axis_roles`` first, then
        the name's own alias (``Zplane``, ``Cam``).
        """
        names = [name for name, _ in self.plane_axes]
        named = self.axis_roles.get(role)
        if named in names:
            return named
        return find_slider_name(names, role)

    def axis_size(self, role: str) -> int:
        """The number of positions the ``role`` axis keys; 1 when it is not keyed."""
        name = self.axis_name(role)
        return dict(self.plane_axes).get(name, 1) if name is not None else 1

    def plane_of(self, pos: Mapping[str, int]) -> int:
        """Flat plane for a slider position ``{dim name: index}``; dims the
        volume is not keyed by are ignored, missing ones sit at 0.
        """
        if not self.plane_axes:
            return 0
        sizes = [n for _, n in self.plane_axes]
        idx = [
            int(np.clip(int(pos.get(name, 0)), 0, n - 1)) for name, n in self.plane_axes
        ]
        return int(np.ravel_multi_index(idx, sizes))

    def plane_pos(self, plane: int) -> dict[str, int]:
        """``{dim name: index}`` behind one flat plane."""
        if not self.plane_axes:
            return {}
        sizes = [n for _, n in self.plane_axes]
        vals = np.unravel_index(int(np.clip(plane, 0, self.nz - 1)), sizes)
        return {name: int(v) for (name, _n), v in zip(self.plane_axes, vals)}

    def plane_label(self, plane: int) -> str:
        """``"3"`` for plain planes, ``"c2·z3"`` when more dims key them
        (1-based); a worded axis keeps its space, ``"Channel 1 · ROI 3"``.
        """
        pos = self.plane_pos(plane)
        if not pos:
            return "1"
        if len(pos) == 1:
            return f"{next(iter(pos.values())) + 1}"
        if all(len(name) == 1 for name in pos):
            return "·".join(f"{name}{v + 1}" for name, v in pos.items())
        return " · ".join(f"{name} {v + 1}" for name, v in pos.items())

    def roi_pos(self, index: int) -> dict[str, int]:
        return self.plane_pos(self.rois[index].plane)

    def roi_z(self, index: int) -> int:
        """0-based z-plane ROI ``index`` was drawn on (0 when z is not keyed)."""
        name = self.axis_name("z")
        return self.roi_pos(index).get(name, 0) if name is not None else 0

    def roi_c(self, index: int) -> int:
        """0-based channel ROI ``index`` was drawn on (0 when c is not keyed)."""
        name = self.axis_name("c")
        return self.roi_pos(index).get(name, 0) if name is not None else 0

    def add_roi(
        self,
        plane: int,
        mask: np.ndarray | None = None,
        source: str = "",
        *,
        pixels: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        polygon: np.ndarray | None = None,
    ) -> int | None:
        """Add an ROI on ``plane`` from a boolean ``(Y, X)`` ``mask`` or flat
        ``pixels`` (with optional ``weights``).

        Returns the new ROI's index, or None when it has fewer than
        ``min_pixels`` pixels (nothing is added then). Overlap with other
        ROIs is allowed. ``source`` names where the mask came from ("" =
        drawn by hand); ``polygon`` keeps the outline as drawn.
        """
        plane = int(plane)
        if not 0 <= plane < self.nz:
            raise IndexError(f"plane {plane} outside {self.nz} planes")
        if pixels is None:
            if mask is None:
                raise ValueError("add_roi needs a mask or pixels")
            mask = np.asarray(mask, bool)
            if mask.shape != (self.ny, self.nx):
                raise ValueError(f"mask is {mask.shape}, frame is {(self.ny, self.nx)}")
            pixels = np.flatnonzero(mask)
        pixels, first = np.unique(
            np.asarray(pixels, np.int64).ravel(), return_index=True
        )
        if pixels.size < self.min_pixels:
            return None
        if weights is not None:
            weights = np.asarray(weights, np.float32)
            weights = (
                weights.reshape(self.ny, self.nx).ravel()[pixels]
                if weights.size == self.ny * self.nx and mask is not None
                else weights.ravel()[first]
            )
        record = RoiRecord(
            plane=plane,
            pixels=pixels,
            weights=weights,
            uid=self.next_uid,
            source=str(source),
            polygon=polygon,
        )
        self.rois.append(record)
        self.next_uid += 1
        self._label_images.pop(plane, None)
        index = len(self.rois) - 1
        self._emit("rois", action="add", index=index, uid=record.uid)
        return index

    def delete_roi(self, index: int) -> bool:
        """Drop one ROI; the indices above it shift down, the uids stay."""
        if not 0 <= index < len(self.rois):
            return False
        record = self.rois.pop(index)
        self._label_images.pop(record.plane, None)
        self._emit("rois", action="delete", index=index, uid=record.uid)
        return True

    def clear(self) -> None:
        self.rois.clear()
        self._label_images.clear()
        self._emit("rois", action="clear")

    def snapshot(self) -> RoiLabelStore:
        """Deep copy (records, names, ``next_uid``) that later mutations of
        either store cannot reach; no handlers come along.
        """
        out = RoiLabelStore(
            *self._shape,
            label_names=self.label_names,
            min_pixels=self.min_pixels,
            rois=[r.copy() for r in self.rois],
            next_uid=self.next_uid,
        )
        out.plane_axes = self.plane_axes
        out.axis_roles = dict(self.axis_roles)
        return out

    def restore(self, snapshot: RoiLabelStore) -> None:
        """Take a :meth:`snapshot`'s records, names and ``next_uid`` back
        in place (ctrl+z); the handlers and the plane layout stay this
        store's. One ``restore`` event.
        """
        if snapshot.shape != self._shape:
            raise ValueError(f"snapshot is {snapshot.shape}, store is {self._shape}")
        self.rois = [r.copy() for r in snapshot.rois]
        self.label_names = snapshot.label_names
        self.next_uid = max(snapshot.next_uid, self.next_uid)
        self._label_images.clear()
        self._emit("rois", action="restore")

    def add_label_name(self, name: str) -> int:
        """Add a class name to the label set; returns its index (existing
        names return their current index instead of duplicating).
        """
        name = str(name).strip()
        if not name:
            raise ValueError("label name must be non-empty")
        if name in self.label_names:
            return self.label_names.index(name)
        self.label_names = (*self.label_names, name)
        self._emit("rois", action="labels")
        return len(self.label_names) - 1

    def set_class(self, index: int, class_index: int) -> None:
        """Assign a class label to ROI ``index``; UNLABELED (-1) clears."""
        if not UNLABELED <= class_index < len(self.label_names):
            raise IndexError(f"class index {class_index} out of range")
        self.rois[index].class_index = int(class_index)
        self._emit("rois", action="class", index=index, uid=self.rois[index].uid)

    def set_note(self, index: int, note: str) -> None:
        self.rois[index].note = str(note)
        self._emit("rois", action="note", index=index, uid=self.rois[index].uid)

    def set_color(self, index: int, rgb: tuple[int, int, int] | None) -> None:
        """Give ROI ``index`` an explicit display color; None reverts it to
        the class / hue color.
        """
        self.rois[index].color = None if rgb is None else tuple(int(v) for v in rgb)
        self._emit("rois", action="color", index=index, uid=self.rois[index].uid)

    def set_tint(self, colors: Mapping[int, tuple[int, int, int]] | None) -> None:
        """Display colors keyed by uid that win over every other color while
        set (a colormap over some per-ROI value); None clears them.
        """
        self.tint = (
            {}
            if not colors
            else {int(u): tuple(int(v) for v in rgb) for u, rgb in colors.items()}
        )
        self._emit("rois", action="tint")

    def uid_index(self, uid: int) -> int | None:
        """ROI index currently holding ``uid``, or None when it is gone."""
        for i, r in enumerate(self.rois):
            if r.uid == uid:
                return i
        return None

    def footprint(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(ypix, xpix, weights)`` of one ROI (int32, int32, float32)."""
        record = self.rois[index]
        ypix, xpix = np.divmod(record.pixels, self.nx)
        return ypix.astype(np.int32), xpix.astype(np.int32), record.weights

    def mask(self, index: int) -> np.ndarray:
        """``(Y, X)`` bool mask of one ROI."""
        out = np.zeros(self.ny * self.nx, bool)
        out[self.rois[index].pixels] = True
        return out.reshape(self.ny, self.nx)

    def label_image(self, plane: int) -> np.ndarray:
        """``(Y, X)`` uint32 of one plane: ROI ``i`` paints ``i + 1``, 0 is
        background; where ROIs overlap the smallest is on top. Cached until
        the plane's ROIs change.
        """
        plane = int(plane)
        got = self._label_images.get(plane)
        if got is None:
            got = np.zeros(self.ny * self.nx, np.uint32)
            on_plane = self.rois_on_plane(plane)
            for i in sorted(on_plane, key=lambda i: -self.rois[i].area):
                got[self.rois[i].pixels] = i + 1
            got = got.reshape(self.ny, self.nx)
            self._label_images[plane] = got
        return got

    def roi_at(self, plane: int, row: int, col: int) -> int:
        """ROI index under a pixel (the smallest when several), or -1 for
        background / out of range.
        """
        if not (0 <= plane < self.nz and 0 <= row < self.ny and 0 <= col < self.nx):
            return -1
        return int(self.label_image(plane)[row, col]) - 1

    def class_counts(self) -> list[int]:
        """Number of ROIs per class name, in label-set order"""
        counts = [0] * len(self.label_names)
        for r in self.rois:
            if 0 <= r.class_index < len(counts):
                counts[r.class_index] += 1
        return counts

    def rois_on_plane(self, plane: int) -> list[int]:
        return [i for i, r in enumerate(self.rois) if r.plane == int(plane)]

    @property
    def counts(self) -> list[int]:
        """per-ROI pixel counts, in ROI order"""
        return [r.area for r in self.rois]

    def roi_rgb(self, index: int) -> tuple[int, int, int]:
        """Display color of one ROI: its tint while a colormap is on, else
        its explicit group color when set, else its class color when
        classified, else its own hue from ``ROI_COLORS`` (uint8 rgb)
        """
        record = self.rois[index]
        tinted = self.tint.get(record.uid)
        if tinted is not None:
            return tinted
        if record.color is not None:
            return tuple(int(v) for v in record.color)
        if record.class_index >= 0:
            return tuple(int(round(c * 255)) for c in class_color(record.class_index))
        return tuple(int(v) for v in ROI_COLORS[index % len(ROI_COLORS)])

    def color_lut(self) -> np.ndarray:
        """(num_rois + 1, 3) uint8 lookup table indexed by label value;
        row 0 (background) is black
        """
        lut = np.zeros((len(self.rois) + 1, 3), np.uint8)
        for i in range(len(self.rois)):
            lut[i + 1] = self.roi_rgb(i)
        return lut
