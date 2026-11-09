"""The dashboard's morphology panel: a unit's ROIs on their tissue, beside the
traces, in the traces' colours.

The picture the ROIs were drawn on shows at once. The Z-stacks arrive when
``analysis.mesc_placement`` has measured them, and the one the recording fits
best replaces it: a stack's stored position is not the scan's frame, so a stack
the picture does not match carries no ROIs. A stack shows the max over the
slices its ROIs sit on, or only the picked rows' slices, a few more each way
when asked. A click on a line or patch picks or drops its row.

:class:`MorphologyMixin` holds the drawing, in the manner of ``gui/roi``: it has
no state of its own, every attribute is :class:`~.dashboard_app.DashboardApp`'s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from imgui_bundle import hello_imgui, imgui, implot
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities import log
from mbo_utilities.analysis.mesc_placement import StackPlacement
from mbo_utilities.arrays.mesc import MescArray
from mbo_utilities.arrays.mesc_geometry import viewport_geometry
from mbo_utilities.dashboard import (
    DOMAIN_SOURCE,
    FigureOutline,
    bar_label,
    export_morphology,
    morphology_figure,
    nice_bar,
)
from mbo_utilities.gui.imgui.lines import line, plot_style, vec4
from mbo_utilities.gui.mesc_reference import (
    ReferenceImage,
    _projection,
    picture_image,
    placement_summary,
    polyline_distance,
    roi_domains,
    stack_images,
    write_positions,
)

logger = log.get("gui.dashboard_morphology")

__all__ = [
    "PANEL_WIDTH",
    "MorphologyMixin",
    "best_stack",
    "grey_rgba",
    "load_picture",
    "load_stacks",
    "project",
    "slice_range",
    "um_per_px",
    "view_of",
]

PANEL_WIDTH = 420
PICK_PIXELS = 8.0
# the panel opens on the ROIs with this share of their extent around them
VIEW_MARGIN = 0.35
MIN_VIEW_MARGIN_PX = 24.0
CONTRAST_PERCENTILES = (1.0, 99.7)
UNPICKED_RGB = (0.62, 0.62, 0.62)
BAR_RGB = (1.0, 1.0, 1.0)
PICKED_WEIGHT = 3.0
UNPICKED_WEIGHT = 1.5


def open_unit(path, key: str) -> MescArray:
    return MescArray(path, unit=key)


def load_picture(path, unit: str, channel: int) -> tuple:
    """``(arr, picture, motion, domains)`` of one unit, read on a thread:
    ``picture`` is None when the unit has no ROIs on a picture.
    """
    arr = open_unit(path, unit)
    return (
        arr,
        picture_image(arr, open_unit, channel),
        arr.motion_correction,
        roi_domains(arr),
    )


def load_stacks(
    arr: MescArray, channel: int, stack_files
) -> tuple[list[ReferenceImage], list[StackPlacement]]:
    """Every stack ``arr``'s scan was placed on, then those refused."""
    return stack_images(arr, open_unit, channel, stack_files)


def _trust(image: ReferenceImage) -> tuple:
    p = image.placement
    r = -1.0 if p.profile_r is None else p.profile_r
    return p.confirmed, len(image.records), r, p.score


def best_stack(images: list[ReferenceImage]) -> ReferenceImage | None:
    """The stack to open on: confirmed by the recording, then holding the most
    ROIs, then fitting them best.
    """
    return max(images, key=_trust, default=None)


def grey_rgba(image: np.ndarray) -> np.ndarray:
    """``(Y, X)`` as ``(Y, X, 4)`` uint8 grey between ``CONTRAST_PERCENTILES``."""
    low, high = np.percentile(image, CONTRAST_PERCENTILES)
    scaled = np.clip((image - low) / max(high - low, 1e-12), 0.0, 1.0)
    grey = (scaled * 255).astype(np.uint8)
    return np.ascontiguousarray(np.dstack([grey, grey, grey, np.full_like(grey, 255)]))


def slice_range(image: ReferenceImage, rois, pad: int) -> tuple[int, int] | None:
    """The first and last slice of a stack ``image`` under ``rois`` (None: all
    of its ROIs), ``pad`` more each way; None on a picture or with no ROI there.
    """
    if image.placement is None:
        return None
    slices = [r["slice"] for r in image.records if rois is None or r["roi"] in rois]
    if not slices:
        return None
    last = image.stack.shape[2] - 1
    return max(min(slices) - pad, 0), min(max(slices) + pad, last)


def project(image: ReferenceImage, span: tuple[int, int] | None) -> np.ndarray:
    """``image``'s pixels over the slices ``span``, its own when None."""
    if span is None:
        return image.image
    return _projection(image.stack, image.c, *span)


def view_of(image: ReferenceImage) -> tuple[float, float, float, float]:
    """``(col0, col1, row0, row1)`` around ``image``'s ROIs."""
    points = np.concatenate([r["pixels"] for r in image.records])
    low, high = points.min(axis=0), points.max(axis=0)
    margin = np.maximum(VIEW_MARGIN * (high - low).max(), MIN_VIEW_MARGIN_PX)
    return low[0] - margin, high[0] + margin, low[1] - margin, high[1] + margin


def um_per_px(arr: MescArray, image: ReferenceImage) -> float:
    """Microns per pixel of ``image``, a stack's or the unit's picture's."""
    if image.placement is not None:
        path, key = image.placement.stack_path, image.placement.stack_key
    else:
        info = next(u for u in arr.units if u["key"] == arr.unit_key)
        path, key = arr.filenames[0], info["background_unit"]
    return viewport_geometry(path, key)["width"] / image.image.shape[1]


class MorphologyMixin:
    """The panel's loading, drawing and export."""

    def _reset_morphology(self) -> None:
        """Forget the unit shown and read the new one's picture; a stale
        thread's answer is dropped by its future no longer being held.
        """
        self.morph_images, self.morph_key = {}, None
        self.morph_refused, self.morph_domains = [], {}
        self.morph_note, self._morph_arr, self.motion = "", None, None
        self._morph_textures.clear()
        self._morph_pixels, self._morph_on = {}, None
        self._morph_picture = self._morph_stacks = self._morph_project = None
        self._morph_fit = None
        if self.path is not None:
            self._morph_picture = self._morpher.submit(
                load_picture, self.path, self.unit, self.channel
            )

    def _poll_morphology(self) -> None:
        if self._morph_picture is not None and self._morph_picture.done():
            future, self._morph_picture = self._morph_picture, None
            if future.exception() is not None:
                self.morph_note = f"no morphology: {future.exception()}"
                logger.error(self.morph_note)
                return
            arr, picture, self.motion, self.morph_domains = future.result()
            self._morph_arr = arr
            if picture is None:
                self.morph_note = "this unit has no ROIs on a picture"
                return
            self.morph_images = {picture.key: picture}
            self._show_image(picture.key)
            self._measure_stacks()
        if self._morph_stacks is not None and self._morph_stacks.done():
            future, self._morph_stacks = self._morph_stacks, None
            if future.exception() is not None:
                self.morph_note = f"placement failed: {future.exception()}"
                logger.error(self.morph_note)
                return
            images, self.morph_refused = future.result()
            self.morph_images.update({im.key: im for im in images})
            self.morph_note = ""
            best = best_stack(images)
            if best is not None:
                self._show_image(best.key)
        if self._morph_project is not None and self._morph_project.done():
            future, self._morph_project = self._morph_project, None
            if future.exception() is not None:
                self.morph_note = f"projection failed: {future.exception()}"
                return
            shown, pixels = future.result()
            self._morph_pixels[shown] = pixels

    def _measure_stacks(self) -> None:
        self.morph_note = "measuring Z-stacks..."
        self._morph_stacks = self._morpher.submit(
            load_stacks, self._morph_arr, self.channel, tuple(self.stack_files)
        )

    def _show_image(self, key: str) -> None:
        self.morph_key = key
        self._morph_fit = view_of(self.morph_images[key])
        self._morph_scale = um_per_px(self._morph_arr, self.morph_images[key])

    def _morph_shown(self) -> tuple:
        """``(image key, slices)`` the settings ask for; slices None is the
        image as it came, a picture or a stack's ROIs' own slices.
        """
        image = self.morph_images[self.morph_key]
        rois = None
        if self.morph_picked_only:
            rois = set(self._line_owners())
        span = slice_range(image, rois, self.morph_pad)
        return image.key, None if span == slice_range(image, None, 0) else span

    def _morph_texture(self):
        """The texture to draw: the slices asked for once they landed, until
        then what the image was showing.
        """
        shown = self._morph_shown()
        image = self.morph_images[shown[0]]
        self._morph_pixels.setdefault((image.key, None), image.image)
        if shown in self._morph_pixels:
            self._morph_on = shown
        else:
            if self._morph_project is None:
                self._morph_project = self._morpher.submit(_project_job, shown, image)
            if self._morph_on is None or self._morph_on[0] != image.key:
                self._morph_on = (image.key, None)
        if self._morph_on not in self._morph_textures:
            self._morph_textures[self._morph_on] = (
                hello_imgui.create_texture_gpu_from_rgba_data(
                    grey_rgba(self._morph_pixels[self._morph_on])
                )
            )
        return self._morph_textures[self._morph_on]

    def _roi_rows(self) -> dict:
        """``{line: row}`` of the unit's own lines or patches."""
        return {r.extra["line"]: r for r in self.rows if "line" in (r.extra or {})}

    def _line_owners(self) -> dict:
        """``{line: row}`` for the lines drawn in colour: a picked line in its
        own row's, a line of a picked domain in the domain's.
        """
        owners = {
            line: r
            for r in self._picked_rows()
            if r.source == DOMAIN_SOURCE
            for line in r.extra["lines"]
        }
        return owners | {
            line: r for line, r in self._roi_rows().items() if r.key in self.picked
        }

    def _toggle_row(self, row) -> None:
        self.picked = (
            self.picked - {row.key}
            if row.key in self.picked
            else self.picked | {row.key}
        )

    def _roi_at(self, image: ReferenceImage) -> dict | None:
        """The record of the line or patch within ``PICK_PIXELS`` of the mouse."""
        mouse = imgui.get_mouse_pos()
        point = np.array([mouse.x, mouse.y])
        best, reach = None, PICK_PIXELS
        for record in image.records:
            pixels = [
                implot.plot_to_pixels(float(x), float(y)) for x, y in record["pixels"]
            ]
            gap = polyline_distance(point, np.array([(p.x, p.y) for p in pixels]))
            if gap < reach:
                best, reach = record, gap
        return best

    def _draw_outlines(self, image: ReferenceImage) -> None:
        """Unpicked ROIs in grey, then the picked ones and the picked domains' over them in their rows' colours."""
        rows = self._line_owners()
        for picked in (False, True):
            for record in image.records:
                row = rows.get(record["roi"])
                if (row is not None) != picked:
                    continue
                col, y = record["pixels"].T
                line(
                    f"##roi{record['roi']}",
                    y,
                    x=col,
                    color=self._row_rgb(row) if picked else UNPICKED_RGB,
                    weight=PICKED_WEIGHT if picked else UNPICKED_WEIGHT,
                    legend=False,
                )

    def _draw_scale_bar(self, scale: float) -> None:
        limits = implot.get_plot_limits()
        width = limits.x.max - limits.x.min
        bar_um = nice_bar(width * scale)
        right = limits.x.max - 0.04 * width
        left = right - bar_um / scale
        low = limits.y.min + 0.05 * (limits.y.max - limits.y.min)
        line(
            "##bar",
            np.array([low, low]),
            x=np.array([left, right]),
            color=BAR_RGB,
            weight=2.0,
            legend=False,
        )
        implot.push_style_color(implot.Col_.inlay_text, vec4(BAR_RGB))
        implot.plot_text(
            bar_label(bar_um, "um"), (left + right) / 2, low, imgui.ImVec2(0, -10)
        )
        implot.pop_style_color()

    def _hover_text(self, record: dict) -> str:
        row = self._roi_rows().get(record["roi"])
        parts = [row.name if row is not None else f"ROI {record['roi']}"]
        if record["roi"] in self.morph_domains:
            parts.append(self.morph_domains[record["roi"]])
        if record["slice"] is not None:
            parts.append(f"slice {record['slice'] + 1}")
        extra = {} if row is None else row.extra or {}
        if "length_um" in extra:
            parts.append(f"{extra['length_um']:.1f} um")
        return ", ".join(parts)

    def _morph_caption(self, image: ReferenceImage) -> str:
        """How many of the ROIs ``image`` holds and, on a stack, how far to trust it."""
        n, total = len(image.records), len(self._roi_rows())
        kind = "patches" if image.records[0]["kind"] == "patch" else "lines"
        held = f"{n} {kind}" if n == total else f"{n} of {total} {kind}"
        if image.placement is None:
            return f"{held} as drawn"
        check = "confirmed" if image.placement.confirmed else "unconfirmed"
        caption = f"{held}, match {image.placement.score:.2f}, {check}"
        span = self._morph_on[1]
        return (
            caption
            if span is None
            else f"slices {span[0] + 1}-{span[1] + 1}, {caption}"
        )

    def _morph_status(self) -> str:
        """The stacks left out or still coming, below the caption."""
        if self.morph_note:
            return self.morph_note
        refused = [
            f"{p.stack_key.rsplit('/', 1)[-1]} not placed ({p.score:.2f})"
            for p in self.morph_refused
        ]
        return "  ".join(refused) or (
            "" if len(self.morph_images) > 1 else "no Z-stack placed"
        )

    def _draw_morph_toolbar(self, image: ReferenceImage) -> None:
        keys = list(self.morph_images)
        imgui.set_next_item_width(-1)
        changed, i = imgui.combo("##image", keys.index(self.morph_key), keys)
        if changed:
            self._show_image(keys[i])
        if imgui.small_button("add Z-stack file") and self._stack_dialog is None:
            self._stack_dialog = pfd.open_file(
                "Z-stack file", str(self.path.parent), ["MESc", "*.mesc"]
            )
        imgui.set_item_tooltip("Measure this scan against the stacks of another .mesc.")
        if image.placement is None:
            return
        imgui.same_line()
        _, self.morph_picked_only = imgui.checkbox("picked", self.morph_picked_only)
        imgui.set_item_tooltip("Project only the slices under the picked rows.")
        imgui.same_line()
        imgui.set_next_item_width(60)
        _, pad = imgui.input_int("slices each way", self.morph_pad, 0, 0)
        self.morph_pad = max(pad, 0)

    def _draw_morphology(self) -> None:
        if self._stack_dialog is not None and self._stack_dialog.ready():
            picked = self._stack_dialog.result()
            self._stack_dialog = None
            if picked and self._morph_stacks is None:
                self.stack_files.append(picked[0])
                self._measure_stacks()
        if not self.morph_images:
            imgui.text_disabled(self.morph_note or "reading the picture...")
            return
        image = self.morph_images[self.morph_key]
        self._draw_morph_toolbar(image)
        texture = self._morph_texture()
        height, width = self._morph_pixels[self._morph_on].shape[:2]
        captions = 2 * imgui.get_text_line_height_with_spacing()
        size = imgui.ImVec2(-1, imgui.get_content_region_avail().y - captions)
        flags = implot.Flags_.equal | implot.Flags_.no_title | implot.Flags_.no_legend
        with plot_style():
            if implot.begin_plot("##morphology", size, flags):
                hidden = implot.AxisFlags_.no_decorations
                implot.setup_axes("", "", hidden, hidden)
                if self._morph_fit is not None:
                    implot.setup_axes_limits(*self._morph_fit, implot.Cond_.always)
                    self._morph_fit = None
                ## NOTE: uv (0, 0) lands on the plot's top left; rows grow with +y, so v is turned over
                implot.plot_image(
                    "##image",
                    imgui.ImTextureRef(texture.texture_id()),
                    implot.Point(0, 0),
                    implot.Point(width, height),
                    imgui.ImVec2(0, 1),
                    imgui.ImVec2(1, 0),
                )
                self._draw_outlines(image)
                self._draw_scale_bar(self._morph_scale)
                if implot.is_plot_hovered():
                    record = self._roi_at(image)
                    if record is not None:
                        imgui.set_tooltip(self._hover_text(record))
                        row = self._roi_rows().get(record["roi"])
                        if imgui.is_mouse_clicked(0) and row is not None:
                            self._toggle_row(row)
                limits = implot.get_plot_limits()
                self.morph_view = (
                    limits.x.min,
                    limits.x.max,
                    limits.y.min,
                    limits.y.max,
                )
                implot.end_plot()
        imgui.text_disabled(self._morph_caption(image))
        if image.placement is not None:
            imgui.set_item_tooltip(placement_summary(image.placement))
        imgui.text_disabled(self._morph_status())

    def morphology_job(self) -> tuple:
        """What the worker needs to write the panel as shown, taken on the
        draw thread; :func:`write_morphology` is the worker's half.
        """
        image = self.morph_images[self.morph_key]
        rows = self._line_owners()
        settings = self._settings(*self.x_view)
        outlines, named = [], set()
        for r in image.records:
            if r["roi"] not in rows:
                continue
            row = rows[r["roi"]]
            # a domain's name goes on its first line only
            outlines.append(
                FigureOutline(
                    "" if row.key in named else row.name,
                    np.asarray(r["pixels"]),
                    self._color(row, "dark"),
                )
            )
            named.add(row.key)
        munit = str(self.unit).rsplit("/", 1)[-1]
        stem = Path(self.folder) / f"{self.path.stem}_{munit}_morphology"
        info = {
            "source": str(self.path),
            "unit": self.unit,
            "image": image.key,
            "slices": self._morph_on[1],
            "rois": [o.label for o in outlines],
        }
        return (
            self._morph_arr,
            dict(self.morph_images),
            image,
            self._morph_pixels[self._morph_on],
            outlines,
            self.morph_view,
            settings,
            stem,
            info,
            self.morph_domains,
        )


def _project_job(shown: tuple, image: ReferenceImage) -> tuple:
    return shown, project(image, shown[1])


def write_morphology(
    arr, images, image, pixels, outlines, view, settings, stem, info, domains
) -> str:
    """Write the figure of ``pixels`` under ``outlines`` and, with a placed
    stack among ``images``, every ROI's position as ``<stem>_positions.csv``.
    """
    fig = morphology_figure(
        grey_rgba(pixels)[..., 0], outlines, um_per_px(arr, image), view, settings
    )
    written = export_morphology(fig, settings, stem, info)
    if any(im.placement is not None for im in images.values()):
        path = Path(f"{stem}_positions.csv")
        write_positions(path, arr, images.values(), domains)
        written.append(path)
    return f"wrote {len(written)} files to {written[0].parent}"
