"""The picture a MESc AOD scan's lines or patches were drawn on, with them drawn.

An AOD scan (a line scan, chessboard or ribbon) has no picture of its own:
the viewer shows its lines by their samples, or its patches packed side by
side. The reference image puts each line or patch back where the operator
drew it, in one popup with a combo over the images:

- the picture MEScan took just before the scan (the unit's
  ``BackgroundImagePath``), the max over its frames when it has several;
- every Z-stack whose field and depth range hold the scan's lines or
  patches, projected over the slices they were scanned on, not over the
  whole stack: a 98-slice stack's full max is a wall of tissue with the
  ROI's own plane lost in it.

Every ROI is drawn in the colour MESc used, the one the viewer's ROI slider
is on thicker; a click on a line or patch moves the slider to it. Nothing
here says how deep a line sits: the picture is a reference for where it was
drawn, no more.

Each image says which unit it is and, for a Z-stack, which slice its ROIs sit
on, so the popup's one button can open that unit in the viewer at that slice.
That is the only place the Z-stack is offered: the table says nothing about
it, because a stack is worth seeing only next to the lines drawn on it.

The popup is ``gui.imgui.summary.SummaryImageViewer``, masknmf's full-FOV
popup and the one behind the ROI widget's Open full FOV: pan, zoom,
colormap, contrast, pixel values. :func:`reference_images` is GUI-free and
pinned on synthetic files; :class:`ReferenceView` is the popup, opened from
the MESc tab and redrawn every frame from the top strip's hook.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from imgui_bundle import imgui

from mbo_utilities.annotation.store import CLASS_COLORS
from mbo_utilities.arrays.features._dim_labels import find_slider_name, slider_roles
from mbo_utilities.arrays.mesc_geometry import image_overlays, zstack_contents
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.imgui.summary import SummaryImageViewer

__all__ = [
    "MAX_ELEMENTS",
    "ON_THICKNESS",
    "SELECTED_THICKNESS",
    "ReferenceImage",
    "ReferenceView",
    "current_roi",
    "roi_slider",
    "reference_images",
]

ON_THICKNESS = 2.5
SELECTED_THICKNESS = 4.0
# samples a projection reads at most: 200 MB of float32
MAX_ELEMENTS = 50_000_000


def roi_slider(names) -> str | None:
    """The viewer slider that walks the array's Z axis: the one labelled so
    (``Z-plane``) when there is one, else the one in Z's position (``ROI``
    on an AOD unit, ``slider_roles``); None without either.
    """
    name = find_slider_name(names, "z")
    if name is None:
        name = next((n for n, role in slider_roles(names).items() if role == "z"), None)
    return name


def current_roi(iw) -> int:
    """The ROI an AOD unit's slider is on: the Z slider's index, 0 when
    there is none.
    """
    zdim = roi_slider(iw.dim_names)
    return int(iw.indices[zdim]) if zdim is not None else 0


@dataclass
class ReferenceImage:
    """One image of the popup: what the combo calls it, its pixels, the shown
    unit's overlay records on it (``mesc_geometry.image_overlays``), the unit
    it is of, and the slice to open that unit at (None for a picture).
    """

    key: str
    image: np.ndarray
    records: list[dict] = field(default_factory=list)
    unit: str = ""
    slice: int | None = None


def reference_images(
    mesc, open_unit: Callable[[str], object], c: int
) -> list[ReferenceImage]:
    """Every image the popup shows for ``mesc``, in combo order: the picture
    its ROIs were drawn on, then every Z-stack holding them, each a max
    projection in channel ``c`` carrying the unit's ROIs. A picture projects
    over its frames, a Z-stack over the slices the ROIs were scanned on.
    Empty for a unit with no ROIs, or none placed on an image of the file.
    ``open_unit`` opens a sibling unit by key; the caller keeps the cache.
    """
    path = mesc.filenames[0]
    key = mesc.unit_key
    info = next((u for u in mesc.units if u["key"] == key), None)
    refs = (
        [info["background_unit"]]
        if info is not None and info.get("background_unit")
        else []
    )
    refs += [
        stack
        for stack, held in zstack_contents(path, mesc.units).items()
        if key in held and stack not in refs
    ]
    out = []
    for ref_key in refs:
        records = [
            r for r in image_overlays(path, ref_key, mesc.units) if r["unit"] == key
        ]
        if not records:
            continue
        ref = open_unit(ref_key)
        nt, nc, nz, ny, nx = ref.shape
        cc = min(int(c), nc - 1)
        name = ref_key.rsplit("/", 1)[-1]
        if nz > 1:
            # one entry per ROI, so the middle slice is where most of them are
            ks = sorted(r["slice"] for r in records)
            lo, hi, at = ks[0], ks[-1], ks[len(ks) // 2]
            label = (
                f"{name} slice {lo + 1}"
                if lo == hi
                else f"{name} slices {lo + 1}-{hi + 1}"
            )
        else:
            lo, hi, label, at = 0, 0, f"{name} picture", None
        # evenly spaced frames, a few slices at a time, within MAX_ELEMENTS samples
        step = max(1, -(-nt * ny * nx // MAX_ELEMENTS))
        chunk = max(
            1, min(hi - lo + 1, MAX_ELEMENTS // max(-(-nt // step) * ny * nx, 1))
        )
        peak = np.full((ny, nx), -np.inf, np.float32)
        for k0 in range(lo, hi + 1, chunk):
            block = np.asarray(
                ref[::step, cc, k0 : min(k0 + chunk, hi + 1)], np.float32
            )
            np.maximum(peak, block.reshape(-1, ny, nx).max(axis=0), out=peak)
        out.append(ReferenceImage(label, peak, records, ref_key, at))
    return out


class ReferenceView:
    """The reference-image popup of one viewer, rebuilt for the unit it shows.

    ``open`` builds the images once per unit and channel on screen and keeps
    them until another is asked for; ``draw`` goes in a per-frame hook. The
    ROI the viewer's ROI slider is on is read as the overlay draws, so the
    highlight follows the slider with no work of its own. ``on_show(unit,
    slice)`` is called by the toolbar's one button: the host displays that
    unit, a Z-stack at that slice.
    """

    def __init__(
        self, parent, on_show: Callable[[str, int | None], None] | None = None
    ):
        self.parent = parent
        self.on_show = on_show
        self.images: dict[str, ReferenceImage] = {}
        self.unit = ""
        self.n_rois = 0
        self._built: tuple | None = None
        self.viewer = SummaryImageViewer(
            parent.image_widget.figure,
            title="Reference image",
            window_id="mesc_reference",
            roi_provider=self.contours,
            extra_toolbar=self._toolbar,
            show_rois=True,
            on_pick=self.pick,
        )

    @property
    def is_open(self) -> bool:
        return self.viewer.is_open

    def open(self, mesc, open_unit: Callable[[str], object]) -> bool:
        """Show the popup for ``mesc``, on the picture its ROIs were drawn on;
        False when no image of the file carries them.
        """
        iw = self.parent.image_widget
        cdim = find_slider_name(iw.dim_names, "c")
        c = int(iw.indices[cdim]) if cdim is not None else 0
        built = (str(mesc.filenames[0]), mesc.unit_key, c)
        if built != self._built:
            images = reference_images(mesc, open_unit, c)
            self.images = {im.key: im for im in images}
            self.unit = mesc.unit_key.rsplit("/", 1)[-1]
            self.n_rois = next(
                (u["n_outlines"] for u in mesc.units if u["key"] == mesc.unit_key), 0
            )
            self._built = built
            self.viewer.set_images(
                {k: im.image for k, im in self.images.items()},
                selected=images[0].key if images else None,
            )
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
        thickness)`` for the popup: MESc's colours, the slider's ROI thicker.
        """
        im = self.images.get(key)
        if im is None:
            return []
        roi = current_roi(self.parent.image_widget)
        out = []
        for r in im.records:
            rgb = (r["color"] or CLASS_COLORS[r["roi"] % len(CLASS_COLORS)])[:3]
            # the records are [col, row]; the popup draws [row, col]
            out.append(
                (
                    r["pixels"][:, ::-1],
                    (*rgb, 1.0),
                    SELECTED_THICKNESS if r["roi"] == roi else ON_THICKNESS,
                )
            )
        return out

    def pick(self, key: str, py: float, px: float) -> int | None:
        """A click on image ``key`` at ``(py, px)``: the ROI whose line or
        patch runs within a few screen pixels of it becomes the viewer's ROI
        slider's, and is returned; None when none is near.
        """
        im = self.images.get(key)
        if im is None:
            return None
        reach = max(6.0 / max(self.viewer.zoom, 1e-6), 1.0)
        best, best_d = None, reach
        p = np.array([px, py], dtype=float)
        for r in im.records:
            pts = np.asarray(r["pixels"], dtype=float)
            a, b = pts[:-1], pts[1:]
            ab = b - a
            t = np.clip(
                np.einsum("ij,ij->i", p - a, ab)
                / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12),
                0.0,
                1.0,
            )
            d = float(np.min(np.linalg.norm(p - (a + t[:, None] * ab), axis=1)))
            if d < best_d:
                best, best_d = r, d
        if best is None:
            return None
        iw = self.parent.image_widget
        zdim = roi_slider(iw.dim_names)
        if zdim is not None:
            iw.indices[zdim] = int(best["roi"])
        return int(best["roi"])

    def _toolbar(self, viewer) -> None:
        """The popup's one button, which displays the image's own unit in the
        viewer (a Z-stack at the slice its ROIs sit on), and its caption: how
        many of the unit's ROIs are on this image and which one is thick. A
        Z-stack holds only the ROIs scanned inside it, so the count says of
        how many when some are missing.
        """
        im = self.images.get(viewer.current_key)
        if im is None:
            return
        name = im.unit.rsplit("/", 1)[-1]
        if self.on_show is not None:
            imgui.same_line(0, 12)
            at = "" if im.slice is None else f" at slice {im.slice + 1}"
            if imgui.button(f"display {name}{at}"):
                self.on_show(im.unit, im.slice)
            set_tooltip(
                f"Show {name} in the viewer{at}, in place of {self.unit}."
                + ("" if im.slice is None else " The Z-plane slider opens there."),
                show_mark=False,
            )
        roi = current_roi(self.parent.image_widget)
        n = len(im.records)
        what = "patches" if im.records[0]["kind"] == "patch" else "lines"
        drawn = f"{n} {what}" if n == self.n_rois else f"{n} of {self.n_rois} {what}"
        imgui.same_line(0, 12)
        imgui.text_disabled(f"{self.unit}: {drawn} drawn here, ROI {roi + 1} thick")
        set_tooltip(
            "Click a line or patch to move the ROI slider to it."
            + (
                ""
                if n == self.n_rois
                else "\nThe rest were scanned above or below this Z-stack."
            ),
            show_mark=False,
        )
