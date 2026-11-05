"""A MESc AOD scan's lines or patches on its picture and on the Z-stacks around it.

An AOD scan (a line scan, chessboard or ribbon) has no picture of its own:
the viewer shows its lines by their samples, or its patches packed side by
side. The reference image puts each line or patch back in the tissue, in one
popup with a combo over the images:

- the picture MEScan took just before the scan (the unit's
  ``BackgroundImagePath``), the max over its frames when it has several, with
  every ROI where the operator drew it;
- every Z-stack the scan could be placed on (``analysis.mesc_placement``), in
  this file or a stack file beside it, projected over the slices the ROIs sit
  on, not over the whole stack: a 98-slice stack's full max is a wall of
  tissue with the ROI's own plane lost in it.

A stack's position in the file is not the scan's frame, so its ROIs are drawn
at the measured shift, and a stack the picture does not match carries none.
The caption gives the slice under the ROI on the slider, its domain when a run
named one, and the match; measuring takes seconds, so the stacks arrive after
the picture.

Every ROI is drawn in the colour MESc used, the one the viewer's ROI slider
is on thicker; a click on a line or patch moves the slider to it.

The popup is ``gui.imgui.summary.SummaryImageViewer``, masknmf's full-FOV
popup and the one behind the ROI widget's Open full FOV: pan, zoom,
colormap, contrast, pixel values. :func:`picture_image` and
:func:`stack_images` are GUI-free and pinned on synthetic files;
:class:`ReferenceView` is the popup, opened from the MESc tab and redrawn
every frame from the top strip's hook.
"""

from __future__ import annotations

import csv
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities import log
from mbo_utilities.analysis.mesc_placement import (
    StackPlacement,
    place_scan,
    zstack_files,
)
from mbo_utilities.annotation.store import CLASS_COLORS
from mbo_utilities.arrays.features._dim_labels import find_slider_name, slider_roles
from mbo_utilities.arrays.mesc import list_mesc_units
from mbo_utilities.arrays.mesc_geometry import image_overlays
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.imgui.summary import SummaryImageViewer
from mbo_utilities.results import unit_for_source

logger = log.get("gui.mesc_reference")

__all__ = [
    "MAX_ELEMENTS",
    "HALO_THICKNESS",
    "ON_ALPHA",
    "ON_THICKNESS",
    "SELECTED_THICKNESS",
    "ReferenceImage",
    "ReferenceView",
    "current_roi",
    "picture_image",
    "roi_domains",
    "roi_slider",
    "stack_images",
    "write_positions",
]

ON_THICKNESS = 2.0
ON_ALPHA = 0.45
SELECTED_THICKNESS = 4.0
HALO_THICKNESS = 9.0
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
    unit's overlay records on it (``mesc_geometry.image_overlays``), and for a
    Z-stack the placement that put them there.
    """

    key: str
    image: np.ndarray
    records: list[dict] = field(default_factory=list)
    placement: StackPlacement | None = None


def _projection(ref, c: int, lo: int, hi: int) -> np.ndarray:
    """Max of ``ref`` in channel ``c`` over slices ``lo`` to ``hi``, from evenly
    spaced frames, a few slices at a time, within ``MAX_ELEMENTS`` samples.
    """
    nt, nc, _nz, ny, nx = ref.shape
    cc = min(int(c), nc - 1)
    step = max(1, -(-nt * ny * nx // MAX_ELEMENTS))
    chunk = max(1, min(hi - lo + 1, MAX_ELEMENTS // max(-(-nt // step) * ny * nx, 1)))
    peak = np.full((ny, nx), -np.inf, np.float32)
    for k0 in range(lo, hi + 1, chunk):
        block = np.asarray(ref[::step, cc, k0 : min(k0 + chunk, hi + 1)], np.float32)
        np.maximum(peak, block.reshape(-1, ny, nx).max(axis=0), out=peak)
    return peak


def _scan_info(mesc) -> dict | None:
    """The list entry of a scan whose outlines pair with its ROIs: pairing a
    different count by order would be a guess.
    """
    info = next((u for u in mesc.units if u["key"] == mesc.unit_key), None)
    return info if info is not None and info["n_outlines"] == info["nrois"] else None


def picture_image(mesc, open_unit: Callable, c: int) -> ReferenceImage | None:
    """The picture ``mesc``'s ROIs were drawn on, carrying every one of them;
    None for a unit with no ROIs or no picture. ``open_unit(path, key)`` opens
    a unit; the caller keeps the cache.
    """
    info = _scan_info(mesc)
    key = None if info is None else info["background_unit"]
    if key is None:
        return None
    path = mesc.filenames[0]
    records = image_overlays(path, mesc.unit_key, key)
    if not records:
        return None
    name = key.rsplit("/", 1)[-1]
    return ReferenceImage(
        f"{name} picture", _projection(open_unit(path, key), c, 0, 0), records
    )


def stack_images(
    mesc, open_unit: Callable, c: int, stack_files=()
) -> tuple[list[ReferenceImage], list[StackPlacement]]:
    """Every Z-stack ``mesc``'s scan was placed on, in the file, beside it or in
    ``stack_files``, each projected over the slices its ROIs sit on; then the
    placements refused, whose stacks carry no ROIs.
    """
    if _scan_info(mesc) is None:
        return [], []
    source = str(mesc.filenames[0])
    paths = dict.fromkeys(
        [source, *(str(p) for p in zstack_files(source)), *map(str, stack_files)]
    )
    images, refused = [], []
    for path in paths:
        for u in list_mesc_units(path):
            placement = (
                place_scan(mesc, path, u["key"], c) if u["kind"] == "zstack" else None
            )
            if placement is None:
                continue
            if not placement.placed:
                refused.append(placement)
                continue
            records = image_overlays(
                source, mesc.unit_key, u["key"], path, placement.shift_um
            )
            if not records:
                continue
            slices = sorted(r["slice"] for r in records)
            lo, hi = slices[0], slices[-1]
            name = u["munit"] if path == source else f"{Path(path).stem} {u['munit']}"
            label = f"slice {lo + 1}" if lo == hi else f"slices {lo + 1}-{hi + 1}"
            image = _projection(open_unit(path, u["key"]), c, lo, hi)
            images.append(ReferenceImage(f"{name} {label}", image, records, placement))
    return images, refused


def roi_domains(mesc) -> dict[int, str]:
    """``{roi: domain}`` from the newest run's line membership; empty without one."""
    results = mesc.results
    name = None if results is None else unit_for_source(results, mesc.unit_key)
    if name is None or results.units[name].member_kind != "line":
        return {}
    unit = results.units[name]
    return {
        int(line): domain
        for domain, lines in zip(unit.roi_names, unit.members)
        for line in lines
    }


def write_positions(path, mesc, images, domains: dict[int, str]) -> None:
    """One CSV row per ROI on each placed stack: its ends in that stack's microns,
    and its ROI and slice counted from 1 as the caption counts them.
    """
    rows = []
    for im in images:
        if im.placement is None:
            continue
        p = im.placement
        for r in im.records:
            pos = mesc.line_positions[r["roi"]]
            start = (np.array(pos["start_um"]) + p.shift_um).round(2).tolist()
            end = (np.array(pos["end_um"]) + p.shift_um).round(2).tolist()
            rows.append(
                {
                    "unit": mesc.unit_key,
                    "roi": r["roi"] + 1,
                    "domain": domains.get(r["roi"], ""),
                    "stack_file": p.stack_path,
                    "stack_unit": p.stack_key,
                    "slice": r["slice"] + 1,
                    **dict(zip(("x0_um", "y0_um", "z0_um"), start)),
                    **dict(zip(("x1_um", "y1_um", "z1_um"), end)),
                    "match": round(p.score, 3),
                    "confirmed": p.confirmed,
                    "with_rtmc": p.with_rtmc,
                }
            )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def placement_summary(placement: StackPlacement) -> str:
    """One image's placement in words, for the caption's tooltip."""
    dx, dy, dz = placement.shift_um
    minutes = abs(placement.gap_s) / 60
    when = "before" if placement.gap_s > 0 else "after"
    text = (
        f"ROIs moved {dx:+.1f}, {dy:+.1f}, {dz:+.1f} um from where the file stores "
        f"them. The picture matches this stack at {placement.score:.2f}. "
        f"The stack was taken {minutes:.0f} min {when} the scan."
    )
    if placement.profile_r is None:
        return f"{text}\nThe recording cannot check this placement."
    verdict = "confirms" if placement.confirmed else "does not confirm"
    rtmc = "with" if placement.with_rtmc else "without"
    return (
        f"{text}\nThe recording {verdict} it, {rtmc} the scan's mean RTMC: time "
        f"means against the stack r {placement.profile_r:.2f}, p {placement.profile_p:.3f}."
    )


class ReferenceView:
    """The reference-image popup of one viewer, rebuilt for the unit it shows.

    ``open`` shows the picture at once and measures the Z-stacks on a thread,
    once per unit and channel on screen; ``draw`` goes in a per-frame hook and
    takes them in when they land. The ROI the viewer's ROI slider is on is read
    as the overlay draws, so the highlight follows the slider with no work of
    its own.
    """

    def __init__(self, parent):
        self.parent = parent
        self.images: dict[str, ReferenceImage] = {}
        self.unit = ""
        self.n_rois = 0
        self.domains: dict[int, str] = {}
        self.refused: list[StackPlacement] = []
        self.stack_files: list[str] = []
        self.placing = False
        self.error: str | None = None
        self._built: tuple | None = None
        self._placed: tuple | None = None
        self._source: tuple | None = None
        self._dialog = None
        self._save_dialog = None
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

    def open(self, mesc, open_unit: Callable) -> bool:
        """Show the popup for ``mesc``, on the picture its ROIs were drawn on;
        False when it has none.
        """
        iw = self.parent.image_widget
        cdim = find_slider_name(iw.dim_names, "c")
        c = int(iw.indices[cdim]) if cdim is not None else 0
        built = (str(mesc.filenames[0]), mesc.unit_key, c, tuple(self.stack_files))
        self._source = (mesc, open_unit)
        if built != self._built:
            picture = picture_image(mesc, open_unit, c)
            self.images = {} if picture is None else {picture.key: picture}
            self.unit = mesc.unit_key.rsplit("/", 1)[-1]
            self.n_rois = next(
                (u["n_outlines"] for u in mesc.units if u["key"] == mesc.unit_key), 0
            )
            self.domains = roi_domains(mesc)
            self.refused, self.error, self._placed = [], None, None
            self._built = built
            self.viewer.set_images(
                {k: im.image for k, im in self.images.items()},
                selected=None if picture is None else picture.key,
            )
            self.placing = picture is not None
            if self.placing:
                threading.Thread(
                    target=self._place,
                    args=(built, mesc, open_unit, c),
                    name="mesc-placement",
                    daemon=True,
                ).start()
        if not self.images:
            return False
        self.viewer.open()
        return True

    def _place(self, built: tuple, mesc, open_unit: Callable, c: int) -> None:
        """Measure the stacks off the draw thread; a stale unit's answer is dropped."""
        try:
            found = (*stack_images(mesc, open_unit, c, built[3]), None)
        except Exception as e:
            logger.exception(f"placing {mesc.unit_key} failed")
            found = ([], [], str(e))
        if built == self._built:
            self._placed = found

    def draw(self) -> None:
        if self._dialog is not None and self._dialog.ready():
            picked = self._dialog.result()
            self._dialog = None
            if picked:
                self.stack_files.append(picked[0])
                self.open(*self._source)
        if self._save_dialog is not None and self._save_dialog.ready():
            picked = self._save_dialog.result()
            self._save_dialog = None
            if picked:
                write_positions(
                    picked, self._source[0], self.images.values(), self.domains
                )
        if self._placed is not None:
            images, self.refused, self.error = self._placed
            self._placed = None
            self.placing = False
            self.images.update({im.key: im for im in images})
            self.viewer.set_images({k: im.image for k, im in self.images.items()})
        self.viewer.draw()

    def close(self) -> None:
        self.viewer.close()

    def cleanup(self) -> None:
        self.viewer.cleanup()
        self.images = {}
        self._built = None

    def contours(self, key: str) -> list[tuple]:
        """The shown unit's ROIs on image ``key`` as ``(points, rgba,
        thickness)`` for the popup: MESc's colours, dimmed, and the slider's
        ROI last at full opacity over a white halo.
        """
        im = self.images.get(key)
        if im is None:
            return []
        roi = current_roi(self.parent.image_widget)
        out, selected = [], []
        for r in im.records:
            rgb = (r["color"] or CLASS_COLORS[r["roi"] % len(CLASS_COLORS)])[:3]
            # the records are [col, row]; the popup draws [row, col]
            pts = r["pixels"][:, ::-1]
            if r["roi"] == roi:
                selected += [
                    (pts, (1.0, 1.0, 1.0, 1.0), HALO_THICKNESS),
                    (pts, (*rgb, 1.0), SELECTED_THICKNESS),
                ]
            else:
                out.append((pts, (*rgb, ON_ALPHA), ON_THICKNESS))
        return out + selected

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

    def caption(self, im: ReferenceImage) -> str:
        """What the toolbar says about image ``im``: how many of the unit's
        ROIs are on it and, on a Z-stack, the slice under the slider's ROI.
        """
        roi = current_roi(self.parent.image_widget)
        n = len(im.records)
        what = "patches" if im.records[0]["kind"] == "patch" else "lines"
        drawn = f"{n} {what}" if n == self.n_rois else f"{n} of {self.n_rois} {what}"
        name = f"ROI {roi + 1}"
        if roi in self.domains:
            name += f" ({self.domains[roi]})"
        if im.placement is None:
            return f"{self.unit}: {drawn} drawn here, {name} thick"
        record = next((r for r in im.records if r["roi"] == roi), None)
        where = "not in this stack" if record is None else f"slice {record['slice'] + 1}"
        check = "confirmed" if im.placement.confirmed else "unconfirmed"
        return f"{self.unit}: {name} {where}, {drawn}, match {im.placement.score:.2f}, {check}"

    def status(self) -> str:
        """The stacks' state: still measuring, failed, or which were refused."""
        if self.placing:
            return "measuring Z-stacks..."
        if self.error is not None:
            return f"placement failed: {self.error}"
        return "  ".join(
            f"{p.stack_key.rsplit('/', 1)[-1]} not placed ({p.score:.2f})"
            for p in self.refused
        )

    def _toolbar(self, viewer) -> None:
        """A row under the viewer's own for the button that takes a stack file
        the search missed and the image's caption, then the stacks' state, on
        a row of its own so the image does not jump when it clears.
        """
        if imgui.button("add Z-stack file") and self._dialog is None:
            start = str(Path(self._built[0]).parent)
            self._dialog = pfd.open_file("Z-stack file", start, ["MESc", "*.mesc"])
        set_tooltip(
            "Measure this scan against the stacks of another .mesc file.",
            show_mark=False,
        )
        imgui.same_line()
        imgui.begin_disabled(all(im.placement is None for im in self.images.values()))
        if imgui.button("save positions") and self._save_dialog is None:
            source = Path(self._built[0])
            self._save_dialog = pfd.save_file(
                "ROI positions",
                str(source.with_name(f"{source.stem}_{self.unit}_positions.csv")),
                ["CSV", "*.csv"],
            )
        imgui.end_disabled()
        set_tooltip(
            "Every ROI's slice and microns on each placed Z-stack, as one CSV.",
            show_mark=False,
        )
        im = self.images.get(viewer.current_key)
        if im is not None:
            imgui.text_disabled(self.caption(im))
            tip = "Click a line or patch to move the ROI slider to it."
            if im.placement is not None:
                tip = f"{placement_summary(im.placement)}\n{tip}"
            set_tooltip(tip, show_mark=False)
        imgui.text_disabled(self.status())
