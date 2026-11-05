"""The mask / contour overlay on the image, region drawing, picking, color by."""

from __future__ import annotations

import numpy as np
from fastplotlib.graphics.selectors._polygon import point_in_polygon
from imgui_bundle import imgui
from masknmf.visualization.imgui import (
    CLICK_SLOP,
    GROUP_COLORS,
)
from masknmf.visualization.imgui import THEME as MTHEME
from masknmf.visualization.rois import MARKED_COLOR, FootprintSet

from mbo_utilities.gui.roi._shared import (
    COLOR_BY,
    CONTOUR_SHAPES,
)
from mbo_utilities.gui.roi_runs import (
    component_color,
    outline_data,
)
from mbo_utilities.roi_workflow import (
    feather_mask,
)


class OverlayMixin:
    """The mask / contour overlay on the image, region drawing, picking, color by."""

    def apply_color_by(self):
        """Tint every drawn ROI by ``color_by`` through ``color_cmap`` (a store column, or its traces' peak); the id palette restores the class / hue colors."""
        if self.color_by == COLOR_BY[0]:
            self.model.colorize(None)
            return
        if self.color_by == "peak":
            values = {}
            for record in self.store.rois:
                peaks = [
                    self._trace_stat(t.key)[2] for t in self.traces.for_roi(record.uid)
                ]
                if peaks:
                    values[record.uid] = max(peaks)
        else:
            values = self.model.column(self.color_by)
        self.model.colorize(
            values,
            cmap=self.color_cmap,
            categorical=self.color_by in ("class", "z", "c"),
        )

    def set_color_by(self, by: str, cmap: str | None = None):
        if by not in COLOR_BY:
            raise ValueError(f"color by one of {COLOR_BY}, not {by!r}")
        self.color_by = by
        if cmap is not None:
            self.color_cmap = cmap
        self.apply_color_by()
        self.status = (
            "colors: class / group / hue"
            if by == COLOR_BY[0]
            else f"colored by {by} ({self.color_cmap})"
        )

    def _feather(self, index: int) -> tuple:
        """``(ypix, xpix, lam)`` of one drawn mask, soft-edged; cached by uid since a mask's pixels never change."""
        record = self.store.rois[index]
        got = self._feathers.get(record.uid)
        if got is None:
            w = feather_mask(self.store.mask(index))
            ypix, xpix, _weights = self.store.footprint(index)
            got = (ypix, xpix, w[ypix, xpix])
            self._feathers[record.uid] = got
        return got

    def refresh_overlay(self):
        """Masknmf's masks and contours over the plane on screen, drawn and algo ROIs alike.

        Every mask at the masks opacity (weighted by its trace's peak), the
        selection and the group at the sel masks opacity with a white rim,
        marked rows red; other contours white, the selection's and the
        group's in their color.
        """
        rows = [row for row in range(len(self.rows)) if self._row_plane(row) == self.z]
        key = (self.z, tuple(rows))
        if self._footprints is None or self._footprints[0] != key:
            footprints = FootprintSet([self._row_footprint(row) for row in rows])
            self._footprints = (key, rows, footprints)
        _key, rows, footprints = self._footprints
        footprints.colors = (
            np.array([self._row_rgb(row) for row in rows], np.float32) if rows else None
        )
        peaks = np.asarray(self.order.columns.get("peak", np.zeros(0)), np.float64)
        peaks = (
            peaks[rows] if len(peaks) == len(self.rows) else np.full(len(rows), np.nan)
        )
        finite = np.isfinite(peaks) & (peaks > 0)
        top = float(peaks[finite].max()) if finite.any() else 1.0
        footprints.peaks = np.where(finite, peaks, top).astype(
            np.float32
        )  # no trace: the field's peak
        position = {self.rows[row]: i for i, row in enumerate(rows)}
        group = self._group_colors()
        picks = [position[pair] for pair in self.buffer if pair in position]
        pair = self._selection_pair()
        selected = position.get(pair) if pair is not None else None
        visible = self.show_masks or self.show_selected_masks
        self.overlay.visible = visible and bool(rows)
        if self.overlay.visible:
            self.overlay.data = footprints.rgba(
                (self.ny, self.nx),
                self.opacity if self.show_masks else 0.0,
                selected if self.show_selected_masks else None,
                [],
                {i: group.get(self.rows[rows[i]], footprints.color(i)) for i in picks}
                if self.show_selected_masks
                else {},
                self.selected_opacity,
                by_peak=self.masks_by_peak,
            )
        highlighted = set(picks) | ({selected} if selected is not None else set())
        comps = []
        for i, row in enumerate(rows):
            on = i in highlighted
            if (
                on
                and not self.show_selected_contours
                or not on
                and not self.show_contours
            ):
                continue
            ypix, xpix, lam = footprints.footprints[i]
            if on:
                comps.append(
                    (
                        ypix,
                        xpix,
                        lam,
                        footprints.color(i),
                        self.selected_contour_opacity,
                    )
                )
            else:
                comps.append((ypix, xpix, lam, (1.0, 1.0, 1.0), self.contour_opacity))
        positions, colors = outline_data(comps, self.contour_shape)
        if not len(positions):
            self.outline.visible = False
            return
        self.outline.data = positions
        self.outline.colors = colors
        self.outline.visible = True

    def _center_on(self, ypix, xpix):
        """Pan the camera onto one mask, holding the zoom the user set; it widens only when the mask cannot fit (or there is no view yet)."""
        if not len(ypix):
            return
        y0, y1 = float(np.min(ypix)), float(np.max(ypix))
        x0, x1 = float(np.min(xpix)), float(np.max(xpix))
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        cam = self.subplot.camera
        need_w, need_h = (x1 - x0) * 1.5, (y1 - y0) * 1.5  # some air around the mask
        width, height = float(cam.width), float(cam.height)
        if width <= 0 or height <= 0 or need_w > width or need_h > height:
            half = max(need_w, need_h, 40.0) / 2
            cam.show_rect(cx - half, cx + half, cy - half, cy + half)
            return
        pos = cam.local.position
        cam.local.position = (cx, cy, float(pos[2]))

    def _center_selection(self):
        if self.selected >= 0:
            ypix, xpix, _lam = self._feather(self.selected)
            self._center_on(ypix, xpix)
        elif self.selected_derived is not None:
            si, k = self.selected_derived
            ypix, xpix, _lam = self.derived[si].footprints[k]
            self._center_on(ypix, xpix)

    def toggle_follow(self):
        """Center (f): the view follows the selection, and a label steps to the next ROI, like masknmf's classification GUI."""
        self.follow = not self.follow
        if self.follow:
            self._center_selection()
            self.status = "centering on the selection; labeling advances"
        else:
            self.status = f"{self.n_rois} ROIs"

    def open_full_fov(self):
        """Masknmf's summary-image popup over the frame on screen and the labels."""
        frame = np.asarray(self.image.data.value, np.float32)
        if frame.ndim == 3:
            frame = frame[..., :3].mean(axis=-1)
        images = {"current frame": frame}
        if self.n_rois:
            images["ROI labels"] = self.labels.astype(np.float32)
        self.summary.set_images(images, selected="current frame")
        self.summary.open()

    def _drawing(self) -> bool:
        """The region has the pointer: its vertices are being placed or dragged."""
        return (
            self.region_selector is not None
            and self.region_selector._move_info.mode is not None
        )

    def _start_region(self):
        """Draw (a): arm a polygon on the image, or drop the one there is (the selection stays)."""
        if self.region_selector is not None:
            self._drop_region()
            return
        self.filter_select = False
        self._snapshot()
        self.region_selector = self.iw.graphics[0].add_polygon_selector(
            fill_color=(0.0, 0.0, 0.0, 0.0),
            edge_color=MTHEME.accent[:3],
            vertex_color=MTHEME.accent[:3],
            edge_thickness=2,
            vertex_size=8,
        )
        self.status = "click on the image to add points; click the first point to close"

    def _poll_region(self):
        """The region selects the ROIs in view on this slice whose centers it holds, as it is drawn or dragged; under three vertices, or a hand-edited selection, drops it."""
        selector = self.region_selector
        if selector is None:
            return
        polygon = np.asarray(selector.selection)[:, :2]
        moving = selector._move_info.mode is not None
        if polygon.shape[0] < 3:
            if not moving:
                self._drop_region()
            return
        if (
            self._region_hits is not None
            and not moving
            and self.buffer != self._region_hits
        ):
            self._drop_region()
            return
        key = (polygon.tobytes(), self.region_outside, self.order.range_limits, moving)
        if key == self._region_key:
            return
        self._region_key = key
        view = [
            int(row) for row in self.order.order if self._row_plane(int(row)) == self.z
        ]
        inside = [point_in_polygon(self._row_center(row), polygon) for row in view]
        hits = [
            self.rows[row]
            for row, hit in zip(view, inside)
            if hit != self.region_outside
        ]
        if hits != self._region_hits:
            self._region_hits = hits
            self.buffer = list(hits)
            self.selected = -1
            self.selected_derived = None
            if hits:
                si, k = hits[-1]
                if si < 0:
                    self.selected = k
                else:
                    self.selected_derived = (si, k)
                self.order.goto(self._row_index[hits[-1]])
            where = "outside" if self.region_outside else "inside"
            self.status = f"region: {len(hits)} ROI(s) {where}; add it as an ROI (r), esc drops it"
            self.refresh_overlay()

    def _drop_region(self):
        selector, self.region_selector = self.region_selector, None
        self._region_key = None
        self._region_hits = None
        if selector is None:
            return
        if selector._move_info.mode is not None:
            selector._end_move_mode()
        try:
            self.subplot.delete_graphic(selector)
        except (KeyError, ValueError):
            pass

    def _commit_region(self):
        """Add ROI (r): the settled region becomes a drawn ROI on the slice on screen, and the selection."""
        selector = self.region_selector
        if selector is None or selector._move_info.mode is not None:
            return
        points = np.asarray(selector.selection)[:, :2]
        self._drop_region()
        self.buffer_clear()
        self.add_roi(points)

    def _row_plane(self, row: int) -> int:
        si, k = self.rows[row]
        return self.store.rois[k].plane if si < 0 else self.derived[si].z

    def _row_footprint(self, row: int) -> tuple:
        """``(ypix, xpix, lam)`` of one table row."""
        si, k = self.rows[row]
        if si < 0:
            return self._feather(k)
        return self.derived[si].footprints[k]

    def _row_center(self, row: int) -> tuple[float, float]:
        """``(x, y)`` center of one row's footprint, where the region tests it."""
        ypix, xpix, _lam = self._row_footprint(row)
        if not len(ypix):
            return (-1.0, -1.0)
        return float(np.mean(xpix)) + 0.5, float(np.mean(ypix)) + 0.5

    def _row_rgb(self, row: int) -> tuple[float, float, float]:
        """One row's color in 0-1 for its mask, table row and trace: its group color in a group, red when marked, else its own."""
        group = self._group_colors()
        pair = self.rows[row]
        if pair in group:
            return group[pair]
        if self.order.columns["del"][row]:
            return MARKED_COLOR
        si, k = pair
        if si < 0:
            return tuple(v / 255.0 for v in self.store.roi_rgb(k))
        return tuple(float(v) for v in component_color(self.derived[si], k))

    def _group_colors(self) -> dict:
        """One contrasting color per group member, shared by its trace, mask and table row (masknmf's)."""
        members = [*self.buffer, *self.pixel_group]
        if not members:
            return {}
        if len(members) == 1:
            return {
                members[0]: GROUP_COLORS[0]
            }  # a lone pixel average has no color of its own
        return {m: GROUP_COLORS[i % len(GROUP_COLORS)] for i, m in enumerate(members)}

    def _pointer_down(self, ev):
        if getattr(ev, "button", 1) != 1:
            return
        self._press_drawn = self._drawing()
        self._press = (ev.x, ev.y)

    def _pointer_up(self, ev):
        """A click on the image picks: no drag, no region taking the pointer, no imgui window over it."""
        press, self._press = self._press, None
        if getattr(ev, "button", 1) != 1 or press is None or self._press_drawn:
            return
        if abs(ev.x - press[0]) + abs(ev.y - press[1]) > CLICK_SLOP or self._drawing():
            return
        if (
            imgui.get_current_context() is not None
            and imgui.get_io().want_capture_mouse
        ):
            return
        pos = self.subplot.map_screen_to_world((ev.x, ev.y))
        if pos is None:
            return
        mods = frozenset(getattr(ev, "modifiers", ()) or ())
        self._pick(int(pos[1]), int(pos[0]), mods)

    def set_masks(self, show: bool):
        """Every mask (m); the selection's has its own switch."""
        self.show_masks = bool(show)
        self.refresh_overlay()

    def set_contours(self, show: bool):
        """Every other ROI's contour (c)."""
        self.show_contours = bool(show)
        self.refresh_overlay()

    def cycle_contour_shape(self):
        """Contours along each ROI's border, or a ring around it (o)."""
        self.contour_shape = CONTOUR_SHAPES[
            (CONTOUR_SHAPES.index(self.contour_shape) + 1) % len(CONTOUR_SHAPES)
        ]
        self.refresh_overlay()
        self.status = f"contours: {self.contour_shape}"
