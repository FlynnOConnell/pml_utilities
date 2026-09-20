"""Where the shown MESc unit sits: a popup over the images around it.

An AOD scan (a line scan, chessboard or ribbon) has no picture of its own:
what the viewer shows is its lines by their samples, or its patches packed
side by side. The outer view puts it back in the field. It shows, in one
popup with a combo over the images:

- the unit's own quick projections over time (mean, max, std) of the slice
  on screen, from evenly spaced frames capped at ``MAX_ELEMENTS`` samples so
  a long recording opens in a moment; a line scan projects every line, lines
  by samples;
- the snapshot its ROIs were placed on (``BackgroundImagePath``) and every
  Z-stack whose field holds them, with the ROIs drawn in MESc's colours: the
  one the viewer's ROI slider is on thicker, every one solid on the snapshot
  (the caption says how far off its plane the slider's ROI sits), the ones on
  another slice of a stack faint (or hidden, with the ROI Overlay panel's
  off-plane switch); a stack contributes its mean and max over depth and each
  slice an ROI sits on.

The popup is ``gui.imgui.summary.SummaryImageViewer``, masknmf's full-FOV
popup and the one behind the ROI widget's Open full FOV: pan, zoom,
colormap, contrast, pixel values. :func:`outer_images` is GUI-free and pinned
on synthetic files; :class:`OuterView` is the popup, opened from the MESc
tab and redrawn every frame from the top strip's hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from imgui_bundle import imgui

from mbo_utilities.annotation.store import CLASS_COLORS
from mbo_utilities.arrays.features._dim_labels import find_slider_name, slider_roles
from mbo_utilities.arrays.mesc_geometry import image_overlays, zstack_contents
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.imgui.summary import SummaryImageViewer

__all__ = [
    "GHOST_ALPHA",
    "GHOST_THICKNESS",
    "MAX_ELEMENTS",
    "ON_THICKNESS",
    "SELECTED_THICKNESS",
    "OuterImage",
    "OuterView",
    "current_z",
    "on_plane",
    "z_slider",
    "outer_images",
    "unit_projections",
]

ON_THICKNESS = 2.5
SELECTED_THICKNESS = 4.0
GHOST_THICKNESS = 1.0
GHOST_ALPHA = 0.3
# samples a quick projection reads at most: 200 MB of float32
MAX_ELEMENTS = 50_000_000


def z_slider(names) -> str | None:
    """The viewer slider that walks the array's Z axis: the one labelled so
    (``Z-plane``) when there is one, else the one in Z's position (``ROI``
    on an AOD unit, ``slider_roles``); None without either."""
    name = find_slider_name(names, "z")
    if name is None:
        name = next((n for n, role in slider_roles(names).items() if role == "z"), None)
    return name


def current_z(iw) -> int:
    """The Z-stack slice on screen, or the ROI an AOD unit's slider is on:
    the z slider's index, 0 when there is none."""
    zdim = z_slider(iw.dim_names)
    return int(iw.indices[zdim]) if zdim is not None else 0


def on_plane(record: dict, z: int) -> bool:
    """Whether an overlay record's ROI draws solid on the plane shown: every
    ROI placed on a snapshot does (MESc drew them all there; how far off its
    plane each sits is the depth column's business), and on a Z-stack the
    ones sitting on slice ``z``."""
    if record["slice"] is None:
        return True
    return record["on_plane"] and record["slice"] == z


@dataclass
class OuterImage:
    """One image of the outer view: what the combo calls it, its pixels, the
    shown unit's overlay records on it (``mesc_geometry.image_overlays``,
    none on a projection of the unit itself) and, for one slice of a
    Z-stack, which slice (the ROIs on it draw solid, the rest faint)."""

    key: str
    image: np.ndarray
    records: list[dict] = field(default_factory=list)
    slice: int | None = None


def unit_projections(arr, c: int, z: int, max_elements: int = MAX_ELEMENTS) -> dict[str, np.ndarray]:
    """Quick ``mean``, ``max`` and ``std`` over time of one slice of a 5D
    array, from evenly spaced frames capped at ``max_elements`` samples;
    a unit with one frame has none. A slice one row tall on a unit with
    several (a line scan: one line per ROI on Z) projects every ROI, lines
    by samples, since one line on its own is no picture."""
    nt, nc, nz, ny, nx = arr.shape
    if nt <= 1:
        return {}
    c = min(int(c), nc - 1)
    whole = ny == 1 and nz > 1
    per_frame = (nz if whole else 1) * ny * nx
    n = max(1, min(nt, max_elements // max(per_frame, 1)))
    step = max(1, -(-nt // n))
    if whole:
        block = np.asarray(arr[::step, c], np.float32).reshape(-1, nz * ny, nx)
    else:
        block = np.asarray(arr[::step, c, min(int(z), nz - 1)], np.float32)
    return {"mean": block.mean(axis=0), "max": block.max(axis=0), "std": block.std(axis=0)}


def outer_images(mesc, open_unit: Callable[[str], object], c: int, z: int) -> list[OuterImage]:
    """Everything the outer view shows for ``mesc``, in combo order: its own
    projections, then each reference image carrying its ROIs — the snapshot
    it was drawn on, then every Z-stack holding it (mean and max over depth,
    then the slices its ROIs sit on). ``open_unit`` opens a sibling unit by
    key; the caller keeps the cache."""
    path = mesc.filenames[0]
    key = mesc.unit_key
    munit = key.rsplit("/", 1)[-1]
    out = [
        OuterImage(f"{munit} {name}", image)
        for name, image in unit_projections(mesc, c, z).items()
    ]
    info = next((u for u in mesc.units if u["key"] == key), None)
    refs = [info["background_unit"]] if info is not None and info.get("background_unit") else []
    refs += [
        stack for stack, held in zstack_contents(path, mesc.units).items()
        if key in held and stack not in refs
    ]
    for ref_key in refs:
        records = [r for r in image_overlays(path, ref_key, mesc.units) if r["unit"] == key]
        if not records:
            continue
        ref = open_unit(ref_key)
        _nt, nc, nz, ny, nx = ref.shape
        cc = min(int(c), nc - 1)
        name = ref_key.rsplit("/", 1)[-1]
        if nz > 1:
            # mean and max over depth, a few slices at a time
            chunk = max(1, min(nz, MAX_ELEMENTS // max(ny * nx, 1)))
            total = np.zeros((ny, nx), np.float64)
            peak = np.full((ny, nx), -np.inf, np.float32)
            for k0 in range(0, nz, chunk):
                block = np.asarray(ref[0, cc, k0 : k0 + chunk], np.float32)
                total += block.sum(axis=0)
                np.maximum(peak, block.max(axis=0), out=peak)
            out.append(OuterImage(f"{name} stack mean", (total / nz).astype(np.float32), records))
            out.append(OuterImage(f"{name} stack max", peak, records))
            # the slices its ROIs sit on; one scanned outside the stack adds none
            for k in sorted({r["slice"] for r in records if r["slice"] is not None and r["on_plane"]}):
                out.append(
                    OuterImage(f"{name} slice {k + 1}", np.asarray(ref[0, cc, k], np.float32), records, k)
                )
        else:
            out.append(OuterImage(f"{name} snapshot", np.asarray(ref[0, cc, 0], np.float32), records))
            for pname, image in unit_projections(ref, cc, 0).items():
                out.append(OuterImage(f"{name} snapshot {pname}", image, records))
    return out


class OuterView:
    """The outer-view popup of one viewer, rebuilt for the unit it shows.

    ``open`` builds the images once per unit and slice on screen and keeps
    them until another is asked for; ``draw`` goes in a per-frame hook. The
    ROI the viewer's ROI slider is on is read as the overlay draws, so the
    highlight follows the slider with no work of its own.
    """

    def __init__(self, parent):
        self.parent = parent
        self.images: dict[str, OuterImage] = {}
        self.unit = ""
        self._built: tuple | None = None
        self.viewer = SummaryImageViewer(
            parent.image_widget.figure,
            title="Outer view",
            window_id="mesc_outer_view",
            roi_provider=self.contours,
            extra_toolbar=self._toolbar,
            show_rois=True,
            on_pick=self.pick,
        )

    @property
    def is_open(self) -> bool:
        return self.viewer.is_open

    def open(self, mesc, open_unit: Callable[[str], object]) -> bool:
        """Show the popup for ``mesc``; False when there is nothing to show
        (one frame, and no snapshot or Z-stack holds it)."""
        iw = self.parent.image_widget
        cdim = find_slider_name(iw.dim_names, "c")
        c = int(iw.indices[cdim]) if cdim is not None else 0
        z = current_z(iw)
        # a line scan projects every line whatever ROI the slider is on, so
        # moving it never rebuilds; any other unit projects the slice shown
        _nt, _nc, nz, ny, _nx = mesc.shape
        built = (str(mesc.filenames[0]), mesc.unit_key, c, None if ny == 1 and nz > 1 else z)
        if built != self._built:
            images = outer_images(mesc, open_unit, c, z)
            self.images = {im.key: im for im in images}
            self.unit = mesc.unit_key.rsplit("/", 1)[-1]
            self._built = built
            first = next((im.key for im in images if im.records), images[0].key if images else None)
            self.viewer.set_images({k: im.image for k, im in self.images.items()}, selected=first)
        if not self.images:
            return False
        self.viewer.open()
        return True

    def draw(self) -> None:
        self.viewer.draw()

    def close(self) -> None:
        self.viewer.close()

    def cleanup(self) -> None:
        self.viewer.cleanup()
        self.images = {}
        self._built = None

    def contours(self, key: str) -> list[tuple]:
        """The shown unit's ROIs on image ``key`` as ``(points, rgba,
        thickness)`` for the popup: MESc's colours, the slider's ROI thicker,
        off-plane ones faint or left out."""
        im = self.images.get(key)
        if im is None or not im.records:
            return []
        roi = current_z(self.parent.image_widget)
        ghosts = bool(getattr(self.parent, "_mesc_overlay_ghosts", True))
        out = []
        for r in im.records:
            on = on_plane(r, im.slice) if im.slice is not None else (r["slice"] is None or r["on_plane"])
            if not on and not ghosts:
                continue
            rgb = (r["color"] or CLASS_COLORS[r["roi"] % len(CLASS_COLORS)])[:3]
            thickness = (SELECTED_THICKNESS if r["roi"] == roi else ON_THICKNESS) if on else GHOST_THICKNESS
            # the records are [col, row]; the popup draws [row, col]
            out.append((r["pixels"][:, ::-1], (*rgb, 1.0 if on else GHOST_ALPHA), thickness))
        return out

    def pick(self, key: str, py: float, px: float) -> int | None:
        """A click on image ``key`` at ``(py, px)``: the ROI whose line or
        patch runs within a few screen pixels of it becomes the viewer's ROI
        slider's, and is returned; None when none is near."""
        im = self.images.get(key)
        if im is None or not im.records:
            return None
        reach = max(6.0 / max(self.viewer.zoom, 1e-6), 1.0)
        best, best_d = None, reach
        p = np.array([px, py], dtype=float)
        for r in im.records:
            pts = np.asarray(r["pixels"], dtype=float)
            a, b = pts[:-1], pts[1:]
            ab = b - a
            t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
            d = float(np.min(np.linalg.norm(p - (a + t[:, None] * ab), axis=1)))
            if d < best_d:
                best, best_d = r, d
        if best is None:
            return None
        iw = self.parent.image_widget
        zdim = z_slider(iw.dim_names)
        if zdim is not None:
            iw.indices[zdim] = int(best["roi"])
        return int(best["roi"])

    def _toolbar(self, viewer) -> None:
        """The popup's extra controls: the off-plane switch (shared with the
        ROI Overlay panel) and where the slider's ROI sits on this image."""
        parent = self.parent
        imgui.same_line(0, 12)
        changed, ghosts = imgui.checkbox("off-plane", bool(getattr(parent, "_mesc_overlay_ghosts", True)))
        set_tooltip("Keep ROIs scanned at another depth visible, faint.", show_mark=False)
        if changed:
            parent._mesc_overlay_ghosts = ghosts
            overlay = getattr(parent, "_mesc_overlay", None)
            if overlay is not None:
                overlay.show_ghosts = ghosts
                overlay.refresh()
        im = self.images.get(viewer.current_key)
        if im is None:
            return
        if not im.records:
            imgui.text_disabled(f"{self.unit}: this slice over time")
            return
        roi = current_z(parent.image_widget)
        r = next((r for r in im.records if r["roi"] == roi), None)
        if r is None:
            imgui.text_disabled(f"{self.unit}: {len(im.records)} ROIs on this image")
            return
        on = on_plane(r, im.slice) if im.slice is not None else (r["slice"] is None or r["on_plane"])
        text = f"{self.unit} ROI {roi + 1}: {r['dz_um']:+.1f} um"
        if r["slice"] is not None:
            text += f", slice {r['slice'] + 1}"
        imgui.text_disabled(
            text + ("" if on else (" (off this slice)" if im.slice is not None else " (outside the stack)"))
        )
