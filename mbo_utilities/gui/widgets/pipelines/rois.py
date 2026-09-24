"""The ROIs pipeline: run the hand-drawn ROIs from the Process tab.

Which ROIs (the selection or group, the rows the ROIs tab lists, every ROI
on the slice on screen, all of them, or the whole frame as one mask), where
their pixels are read (the plane each was drawn on, the slice the sliders
show, or a fixed z-plane / channel, over a frame window) and with which
engine (``mean``, suite2p's extractor, masknmf demixing). The full image
with ``mean`` is the frame's mean trace; with suite2p or masknmf it is a
full detection of that z-plane and channel. Everything it edits lives on
the manual ROI widget (``run_where`` / ``run_z`` / ``run_c`` /
``run_tp`` / ``engine`` / ``run_tag``), so the row buttons on the ROIs
tab and the ``t`` key run one ROI exactly the way this tab is set. The
trace table under the run button is the same table as the Traces tab, cut
down to the ROIs about to run: one row per ROI, z-plane, channel and engine.

Region detection lives here too: find cells inside a drawn region, loading
back as an algo overlay.

Layout follows the other pipeline tabs: a facts block under an accent
title, a two-column settings table whose caption column is exactly as wide
as its longest caption, the green centred Run button, then ``separator_text``
sections for detection and the trace table, which takes the rest of the tab.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import hello_imgui, imgui, imgui_ctx

from mbo_utilities.annotation import ENGINES, FULL_IMAGE
from mbo_utilities.arrays.features._slicing import parse_timepoint_selection
from mbo_utilities.gui._imgui_helpers import (
    right_aligned_text,
    selected_button_style,
    set_tooltip,
    settings_row,
    settings_table,
)
from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.gui.widgets.pipelines.settings import _MISSING_COLOR
from mbo_utilities.pipeline_registry import PipelineInfo

__all__ = ["RoiPipelineWidget", "TARGETS", "WHERE"]

_TITLE_COLOR = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)
_DIM_COLOR = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
_OK_COLOR = imgui.ImVec4(0.40, 0.90, 0.40, 1.0)
_RUN_GREEN = (
    imgui.ImVec4(0.13, 0.55, 0.13, 1.0),
    imgui.ImVec4(0.18, 0.65, 0.18, 1.0),
    imgui.ImVec4(0.1, 0.45, 0.1, 1.0),
)

# which drawn ROIs a run takes: (key, label, tooltip)
TARGETS = (
    (
        "selected",
        "selected",
        "The selected ROI, or every ROI in the ctrl / shift click group",
    ),
    ("listed", "listed", "Every drawn ROI the ROIs tab lists (its filters apply)"),
    ("plane", "this slice", "Every ROI drawn on the slice the sliders show"),
    ("all", "all", "Every drawn ROI, whatever plane it is on"),
    (
        "full",
        "full image",
        "The whole frame as one mask: its mean trace with mean, a full "
        "suite2p / masknmf detection of that z-plane and channel otherwise",
    ),
)
# where the pixels come from: (key, label, tooltip)
WHERE = (
    (
        "drawn",
        "as drawn",
        "Each mask is read on the z-plane and channel it was drawn on",
    ),
    (
        "screen",
        "slice on screen",
        "Each mask is read on the z-plane and channel the sliders show "
        "when the run starts: scroll to another channel and run again",
    ),
    ("fixed", "fixed", "Each mask is read on the z-plane and channel picked here"),
)
# the settings table's captions; the caption column is as wide as the longest
_CAPTIONS = ("Which ROIs", "Read from", "Frames", "Engine", "Settings")


def _em(x: float) -> float:
    return hello_imgui.em_size(x)


class RoiPipelineWidget(PipelineWidget):
    """Traces of hand-drawn ROIs: which ROIs, where they are read, with which engine."""

    name = "ROIs"
    is_available = True
    install_command = "uv pip install mbo_utilities"
    info = PipelineInfo(
        name="rois",
        description="Traces of hand-drawn ROIs (mean, suite2p or masknmf) per z-plane and channel",
        category="processor",
        output_patterns=["**/rois_*/F.npy", "**/rois_*/stat.npy"],
        output_extensions=["npy", "json"],
        marker_files=["rois.json"],
    )
    # T is a frame window; Z and C are one read each (or "as drawn")
    axes_consumed = {"T": "range", "Z": "select-one", "C": "select-one"}

    @classmethod
    def applies_to(cls, arr: Any) -> bool:
        """Anything the ROI tool can draw on: a frame with a time axis."""
        if arr is None:
            return False
        shape = tuple(getattr(arr, "shape", ()) or ())
        return len(shape) >= 3 and int(shape[0]) > 1

    def __init__(self, parent: Any):
        super().__init__(parent)
        self.target = TARGETS[0][0]
        self._frames_text = ""
        self._frames_error = ""
        self._last_status = ""

    @property
    def roi(self):
        """The manual ROI widget, or None while Manual ROI Labeling is off."""
        return getattr(self.parent, "manual_roi", None)

    def _turn_on(self) -> None:
        sync = getattr(self.parent, "sync_manual_roi", None)
        if sync is None:
            self._last_status = "Manual ROI Labeling is unavailable for this view."
            return
        sync(True)
        if self.roi is None:
            self._last_status = "Manual ROI Labeling could not be built for this view."

    def target_indices(self) -> list[int]:
        """Store indices of the drawn ROIs the run would take."""
        roi = self.roi
        if roi is None or self.target == "full":
            return []
        if self.target == "selected":
            return roi.selection_indices()
        if self.target == "listed":
            return roi.listed_drawn()
        if self.target == "plane":
            return roi.store.rois_on_plane(roi.z)
        return list(range(roi.n_rois))

    def draw_config(self) -> None:
        roi = self.roi
        imgui.spacing()
        if roi is None:
            imgui.text_colored(_TITLE_COLOR, "Manual ROI Labeling is off")
            imgui.spacing()
            imgui.text_disabled(
                "Draw ROIs on the image first: the tool adds the ROI cards and tabs."
            )
            imgui.spacing()
            if imgui.button(
                "Turn on Manual ROI Labeling##rois_on", imgui.ImVec2(_em(16), 0)
            ):
                self._turn_on()
            if self._last_status:
                imgui.text_colored(_MISSING_COLOR, self._last_status)
            return
        self._draw_facts(roi)
        imgui.spacing()
        imgui.separator_text("Run")
        self._draw_settings(roi)
        imgui.spacing()
        self._draw_run(roi)
        imgui.spacing()
        imgui.separator_text("Find cells in a region")
        self._draw_find_row(roi)
        imgui.spacing()
        self._draw_traces(roi)

    def _draw_facts(self, roi) -> None:
        """What is there to run: counts as a key / value block, the keys in
        one dim column so the numbers line up.
        """
        imgui.text_colored(_TITLE_COLOR, "Drawn ROIs")
        on_plane = len(roi.store.rois_on_plane(roi.z))
        grouped = len([1 for si, _k in roi.buffer if si < 0])
        picked = (
            f"{grouped} grouped"
            if grouped > 1
            else f"ROI {roi.selected} selected"
            if roi.selected >= 0
            else "none selected"
        )
        traced = len([t for t in roi.traces if t.stands_for_roi])
        other = len(roi.traces) - traced
        drawn = f"{roi.n_rois}"
        if roi.store.nz > 1:
            drawn += f" · {on_plane} on this slice ({roi._plane_label(roi.z)})"
        rows = (
            ("drawn", drawn),
            ("selection", picked),
            (
                "traces",
                f"{traced} of drawn ROIs" + (f" · {other} other" if other else ""),
            ),
        )
        flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_pad_outer_x
        with imgui_ctx.begin_table("##rois_facts", 2, flags) as table:
            if not table:
                return
            imgui.table_setup_column(
                "key", imgui.TableColumnFlags_.width_fixed, _em(5.5)
            )
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            for key, value in rows:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(key)
                imgui.table_next_column()
                imgui.text(value)

    def _draw_settings(self, roi) -> None:
        """The run settings as one table: captions in a fixed column exactly
        as wide as the longest of them, controls in the stretch column, every
        caption on the frame baseline of its row's widgets.
        """
        with settings_table("##rois_settings", _CAPTIONS) as table:
            if not table:
                return
            settings_row("Which ROIs")
            self._draw_target_row(roi)
            settings_row("Read from")
            self._draw_where_row(roi)
            settings_row("Frames")
            self._draw_frames_row(roi)
            settings_row("Engine")
            self._draw_engine_row(roi)
            kind = roi.pipeline_for()
            if kind is not None:
                settings_row("Settings")
                self._draw_pipeline_row(roi, kind)
        imgui.text_disabled(f"reads {roi._where_label()}")
        set_tooltip(
            "Where the next run reads its pixels: the plane each ROI was drawn on, "
            "the slice the sliders show, or the fixed z-plane / channel above; and "
            "the frame window when one is set.",
            show_mark=False,
        )

    def _draw_target_row(self, roi) -> None:
        for i, (key, label, tip) in enumerate(TARGETS):
            if i:
                imgui.same_line(0, _em(0.6))
            if imgui.radio_button(f"{label}##rois_target_{key}", self.target == key):
                self.target = key
            if imgui.is_item_hovered():
                imgui.set_tooltip(tip)
        count = (
            "the whole frame"
            if self.target == "full"
            else f"{len(self.target_indices())} ROI(s)"
        )
        imgui.same_line(0, _em(0.6))
        right_aligned_text(count)

    def _draw_where_row(self, roi) -> None:
        for i, (key, label, tip) in enumerate(WHERE):
            if i:
                imgui.same_line(0, _em(0.6))
            if imgui.radio_button(f"{label}##rois_where_{key}", roi.run_where == key):
                roi.run_where = key
            if imgui.is_item_hovered():
                imgui.set_tooltip(tip)
        if roi.run_where != "fixed":
            return
        # the pickers take a second line: the radios fill the first
        nz, nc = roi.store.axis_size("z"), roi.store.axis_size("c")
        if nz <= 1 and nc <= 1:
            imgui.text_disabled("one z-plane and one channel: same as drawn")
            return
        for role, size, attr in (("z", nz, "run_z"), ("c", nc, "run_c")):
            if size <= 1:
                continue
            if role == "c" and nz > 1:
                imgui.same_line(0, _em(0.8))
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(roi.axis_label(role))
            imgui.same_line(0, _em(0.3))
            imgui.set_next_item_width(_em(4))
            current = getattr(roi, attr)
            idx = 0 if current is None else min(int(current), size - 1)
            changed, idx = imgui.combo(
                f"##rois_fixed_{role}", idx, [str(i + 1) for i in range(size)]
            )
            if changed or current is None:
                setattr(roi, attr, int(idx))

    def _draw_frames_row(self, roi) -> None:
        movie = roi.movie()
        max_frames = int(movie.shape[0]) if movie is not None else 1
        imgui.set_next_item_width(_em(8))
        changed, self._frames_text = imgui.input_text_with_hint(
            "##rois_frames", f"1:{max_frames}", self._frames_text
        )
        if changed:
            text = self._frames_text.strip()
            if not text:
                roi.run_tp, self._frames_error = None, ""
            else:
                try:
                    # the selection string every Save As and pipeline row takes
                    indices = [
                        int(t)
                        for t in parse_timepoint_selection(
                            text, max_frames
                        ).final_indices
                    ]
                    roi.run_tp = None if indices == list(range(max_frames)) else indices
                    self._frames_error = ""
                except (ValueError, IndexError) as e:
                    self._frames_error = str(e)
        set_tooltip(
            "Frames to read, 1-based: start:stop, start:stop:step, or start:stop,"
            "exclude_start:exclude_stop; empty for every frame. What is read is "
            "stamped on the trace row and on the run's ops.npy.",
            show_mark=False,
        )
        imgui.same_line(0, _em(0.6))
        imgui.align_text_to_frame_padding()
        if self._frames_error:
            imgui.text_colored(_MISSING_COLOR, "invalid")
            if imgui.is_item_hovered():
                imgui.set_tooltip(self._frames_error)
        elif roi.run_tp is not None:
            imgui.text_disabled(f"{len(roi.run_tp)} of {max_frames}")
        else:
            imgui.text_disabled(f"all {max_frames}")

    def _draw_engine_row(self, roi) -> None:
        from mbo_utilities.gui.manual_roi import ENGINE_HELP

        imgui.set_next_item_width(_em(6.5))
        changed, sel = imgui.combo(
            "##rois_engine", ENGINES.index(roi.engine), list(ENGINES)
        )
        if changed:
            roi.engine = ENGINES[sel]
        set_tooltip(
            "Which signal to pull out of the masks:\n\n"
            + "\n\n".join(ENGINE_HELP[e] for e in ENGINES),
            show_mark=False,
        )
        imgui.same_line(0, _em(0.6))
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("tag")
        imgui.same_line(0, _em(0.3))
        imgui.set_next_item_width(_em(6.5))
        _, roi.run_tag = imgui.input_text_with_hint("##rois_tag", "manual", roi.run_tag)
        set_tooltip(
            f"Names the output folder: {roi.run_prefix}{roi.effective_tag}/ beside the data. "
            "Leave it empty for the default.",
            show_mark=False,
        )

    def _draw_pipeline_row(self, roi, kind: str) -> None:
        """Whose parameters a suite2p / masknmf run takes, and the way there."""
        label, detail = roi._pipeline_summary(kind)
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(label)
        set_tooltip(detail, show_mark=False)
        imgui.same_line(0, _em(0.6))
        if imgui.small_button(f"Open##rois_open_{kind}"):
            roi.open_pipeline_params(kind)
        set_tooltip(
            f"Switch this tab to {kind}: runs started here use those settings, "
            "including the skip / run / force stage toggles",
            show_mark=False,
        )

    def _draw_run(self, roi) -> None:
        full = self.target == "full"
        indices = self.target_indices()
        movie = roi.movie()
        no_movie = movie is None or int(movie.shape[0]) < 2
        reason = ""
        if no_movie:
            reason = "No (T, Y, X) movie behind this view."
        elif full:
            if roi.engine != "mean" and roi.fpath is None:
                reason = "No data path to run a full detection on."
        elif not indices:
            reason = "No drawn ROI matches the choice above."
        elif roi.fpath is None:
            reason = "No data path to write beside; use Trace for an in-memory mean."
        ready = not reason
        can_trace = not no_movie and (full or bool(indices))
        # the two buttons as one centred group, the way the other tabs centre Run
        run_w, trace_w, gap = _em(11), _em(7), _em(0.6)
        avail = imgui.get_content_region_avail().x
        if avail > run_w + gap + trace_w:
            imgui.set_cursor_pos_x(
                imgui.get_cursor_pos_x() + (avail - run_w - gap - trace_w) * 0.5
            )
        for slot, color in zip(
            (imgui.Col_.button, imgui.Col_.button_hovered, imgui.Col_.button_active),
            _RUN_GREEN,
        ):
            imgui.push_style_color(slot, color)
        if not ready:
            imgui.begin_disabled()
        clicked = imgui.button(f"Run {roi.engine}##rois_run", imgui.ImVec2(run_w, 0))
        if not ready:
            imgui.end_disabled()
        imgui.pop_style_color(3)
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            if full and roi.engine == "mean":
                what = f"Mean trace of the whole frame ({roi._where_label()}); the row lands in the Traces tab"
            elif full:
                what = f"Full {roi.engine} detection of the z-plane and channel ({roi._where_label()}) as a detached worker"
            else:
                what = (
                    f"Run {len(indices)} ROI(s) through {roi.engine} ({roi._where_label()}) "
                    f"-> rois_{roi.effective_tag}/; the rows land in the Traces tab"
                )
            imgui.set_tooltip(reason or what)
        if clicked and ready:
            if full and roi.engine == "mean":
                roi.trace_full()
            elif full:
                roi.run_full_plane(roi.engine)
            else:
                roi.run_rois(indices, roi.run_tag)
            self._last_status = roi.status
        imgui.same_line(0, gap)
        if not can_trace:
            imgui.begin_disabled()
        traced = imgui.button("Trace##rois_trace", imgui.ImVec2(trace_w, 0))
        if not can_trace:
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "No (T, Y, X) movie behind this view."
                if no_movie
                else "No drawn ROI matches the choice above."
                if not (full or indices)
                else f"Mean trace of {'the whole frame' if full else f'{len(indices)} ROI(s)'} in memory "
                f"({roi._where_label()}), no files written"
            )
        if traced and can_trace:
            if full:
                roi.trace_full()
            else:
                roi.trace_rois(indices)
            self._last_status = roi.status
        if roi._run_error:
            imgui.text_colored(_MISSING_COLOR, roi._run_error)
        elif roi.manager.busy:
            imgui.text_colored(_OK_COLOR, roi._status_message()[1])
        elif self._last_status:
            imgui.text_colored(_DIM_COLOR, self._last_status)

    def _draw_find_row(self, roi) -> None:
        have_region = roi.region is not None
        with selected_button_style(roi.region_mode):
            if imgui.button(
                ("Region" if have_region else "Draw region") + "##rois_region",
                imgui.ImVec2(_em(7), 0),
            ):
                roi.set_region_mode(not roi.region_mode)
        if imgui.is_item_hovered():
            y0, y1, x0, x1 = roi.region if have_region else (0, 0, 0, 0)
            imgui.set_tooltip(
                f"Region {y1 - y0}x{x1 - x0} px - drag again to replace it (r)"
                if have_region
                else "Drag a box on the image to mark where to look (r)"
            )
        for kind in ("suite2p", "masknmf"):
            imgui.same_line(0, _em(0.6))
            if not have_region:
                imgui.begin_disabled()
            if imgui.button(f"{kind}##rois_find", imgui.ImVec2(_em(6), 0)):
                roi.discover_region(kind)
                self._last_status = roi.status
            if not have_region:
                imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip(
                    f"Draw a region first, then {kind} looks for cells inside it"
                    if not have_region
                    else f"Look for cells inside the region with {kind}, unseeded; they "
                    "arrive as an algo overlay to promote or discard"
                )
        if have_region:
            y0, y1, x0, x1 = roi.region
            imgui.same_line(0, _em(0.6))
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(f"{y1 - y0} x {x1 - x0} px")

    def _draw_traces(self, roi) -> None:
        if self.target == "full":
            keys = [t.key for t in roi.traces if t.source == FULL_IMAGE]
        else:
            uids = {roi.store.rois[i].uid for i in self.target_indices()}
            keys = [t.key for t in roi.traces if t.stands_for_roi and t.uid in uids]
        imgui.separator_text(f"Traces of these ROIs ({len(keys)})")
        set_tooltip(
            "One row per ROI, z-plane, channel and engine; click plots it on the "
            "Traces panel. The Traces tab lists every row.",
            show_mark=False,
        )
        # the table takes what is left of the tab
        roi.draw_trace_table(keys=keys, table_id="##rois_pipeline_traces")
