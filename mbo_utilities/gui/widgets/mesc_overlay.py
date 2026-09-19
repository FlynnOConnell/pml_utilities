"""ROI overlay: the lines and patches a MESc multi-ROI scan covered, drawn on
the image they were set up on.

The MESc GUI draws every ROI of a scan on the raster snapshot it was placed
on (the unit's ``BackgroundImagePath``), whatever its depth. This panel does
the same when that snapshot, or a Z-stack containing the ROIs, is the
displayed unit: they are drawn as soon as the unit is on screen, from
``arrays.mesc_geometry.image_overlays`` in the colours MESc used, solid when
on the plane on screen (a snapshot: within 1 um of it; a Z-stack: the slice
the slider is on) and faint otherwise; a checkbox hides them for the session.
The panel appears only when the file pairs ROIs with the image by metadata
alone, so a drawn line is one MESc recorded for exactly this picture.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imgui_bundle import imgui

from mbo_utilities import log
from mbo_utilities.arrays.features._dim_labels import find_slider_name
from mbo_utilities.arrays.mesc_geometry import image_overlays
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.mesc_units import mesc_array_of

logger = log.get("gui.mesc_overlay")

_ACCENT = imgui.ImVec4(0.8, 0.8, 0.2, 1.0)
ON_THICKNESS = 2.5
GHOST_THICKNESS = 1.0
GHOST_ALPHA = 0.3
# in front of the image (0) and the manual-ROI masks (1 .. 1.75)
OVERLAY_Z = 2.0


def overlay_records(parent: Any) -> list[dict]:
    """:func:`image_overlays` for the displayed unit, cached on the parent per unit."""
    mesc = mesc_array_of(parent.image_widget.data[0])
    if mesc is None or mesc.modality not in (1, 2):
        return []
    key = (str(mesc.filenames[0]), mesc.unit_key)
    cached = getattr(parent, "_mesc_overlay_records", None)
    if cached is None or cached[0] != key:
        cached = (key, image_overlays(mesc.filenames[0], mesc.unit_key, mesc.units))
        parent._mesc_overlay_records = cached
    return cached[1]


def close_overlay(parent: Any) -> None:
    """Drop the live overlay graphics, if any; a unit swap or file load calls this."""
    overlay = getattr(parent, "_mesc_overlay", None)
    if overlay is not None:
        overlay.close()
    parent._mesc_overlay = None


class MescOverlay:
    """The line graphics of one image unit's ROI outlines on the viewer's subplot.

    Kept on the parent (``parent._mesc_overlay``) so it outlives the widget
    rebuild every unit swap triggers.
    """

    def __init__(self, iw, unit_key: str, records: list[dict], show_ghosts: bool):
        from mbo_utilities.annotation.store import CLASS_COLORS

        self.iw = iw
        self.unit_key = unit_key
        self.records = records
        self.show_ghosts = show_ghosts
        self.subplot = iw.figure[0, 0]
        self.zdim = find_slider_name(iw.dim_names, "z")
        units = sorted({r["unit"] for r in records})
        self.colors = np.array(
            [
                (*(r["color"] or CLASS_COLORS[units.index(r["unit"]) % len(CLASS_COLORS)])[:3], 1.0)
                for r in records
            ],
            dtype=np.float32,
        )
        self.lines = self.subplot.add_line_collection(
            [r["pixels"].astype(np.float32) for r in records],
            colors=self.colors,
            thickness=ON_THICKNESS,
            name="mesc_roi_overlay",
            offset=(0, 0, OVERLAY_Z),
        )
        if self.zdim is not None:
            iw.ndwidget.indices.add_event_handler(self._on_indices)
        self.refresh()

    def _on_indices(self, _indices) -> None:
        self.refresh()

    def current_z(self) -> int:
        return int(self.iw.indices[self.zdim]) if self.zdim is not None else 0

    def shown(self, record: dict, z: int) -> bool:
        """Whether this ROI sits on the plane on screen."""
        if record["slice"] is None:
            return record["on_plane"]
        return record["on_plane"] and record["slice"] == z

    def refresh(self) -> None:
        z = self.current_z()
        for record, graphic, color in zip(self.records, self.lines.graphics, self.colors):
            on = self.shown(record, z)
            graphic.visible = on or self.show_ghosts
            graphic.colors = (*color[:3], 1.0 if on else GHOST_ALPHA)
            graphic.thickness = ON_THICKNESS if on else GHOST_THICKNESS

    def close(self) -> None:
        if self.zdim is not None:
            self.iw.ndwidget.indices.remove_event_handler(self._on_indices)
        self.subplot.delete_graphic(self.lines)


class MescOverlayWidget(Widget):
    """Checkbox that draws the ROIs MESc recorded for the displayed snapshot or Z-stack."""

    name = "ROI Overlay"
    priority = 5
    toggle_key = "preview.mesc_overlay"

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        iw = getattr(parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        # `--roi 0` fans one unit over several subplots; the overlay draws on one
        return len(data) == 1 and bool(overlay_records(parent))

    def draw(self) -> None:
        parent = self.parent
        records = overlay_records(parent)
        mesc = mesc_array_of(parent.image_widget.data[0])
        overlay = getattr(parent, "_mesc_overlay", None)
        if overlay is not None and overlay.unit_key != mesc.unit_key:
            close_overlay(parent)
            overlay = None
        wanted = bool(getattr(parent, "_mesc_overlay_on", True))
        ghosts = bool(getattr(parent, "_mesc_overlay_ghosts", True))

        imgui.spacing()
        imgui.text_colored(_ACCENT, "ROI Overlay")
        imgui.spacing()
        changed, wanted = imgui.checkbox("Overlay ROIs", wanted)
        set_tooltip(
            "Draw the lines and patches MESc scanned on this image, in the "
            "colours it used. Solid: on the plane shown. Faint: at another depth."
        )
        if changed:
            parent._mesc_overlay_on = wanted
        if wanted and overlay is None:
            overlay = parent._mesc_overlay = MescOverlay(
                parent.image_widget, mesc.unit_key, records, ghosts
            )
        elif not wanted and overlay is not None:
            close_overlay(parent)
            overlay = None
        if overlay is None:
            return

        changed, ghosts = imgui.checkbox("Show off-plane ROIs", ghosts)
        set_tooltip("Keep ROIs scanned at another depth visible, faint.")
        if changed:
            parent._mesc_overlay_ghosts = overlay.show_ghosts = ghosts
            overlay.refresh()

        z = overlay.current_z()
        by_unit: dict[str, list[dict]] = {}
        for r in records:
            by_unit.setdefault(r["munit"], []).append(r)
        for munit, rs in by_unit.items():
            on = sum(overlay.shown(r, z) for r in rs)
            kind = "lines" if rs[0]["kind"] == "line" else "patches"
            imgui.text_disabled(f"{munit} · {len(rs)} {kind} · {on} on this plane")
            set_tooltip(
                "\n".join(
                    f"ROI {r['roi'] + 1}: {r['dz_um']:+.1f} um"
                    + ("" if r["slice"] is None else f", slice {r['slice'] + 1}")
                    + ("" if r["on_plane"] else " (outside)")
                    for r in rs
                ),
                show_mark=False,
            )

    def cleanup(self) -> None:
        close_overlay(self.parent)
        self.parent._mesc_overlay_records = None
