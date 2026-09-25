"""The viewer's tab bar, as widgets.

Every tab in the main tab bar is a ``placement = "tab"`` widget with an entry
in the Widgets menu. The bodies live where the work lives — the Preview and
Signal Quality tabs delegate to ``PreviewDataWidget``, Process to the pipelines
package, Cloud to ``gui/_cloud.py`` — so these classes are only the seam
between a tab and its panel.

Manual ROI labelling is split: its control sections sit over the ROI table
in the ROIs tab here and the trace table is the Traces tab, beside Image and
Signal Quality, while its trace plot and the plot's controls are a panel on
the figure's top strip (``gui/manual_roi.py``), which has the width for it.
Nothing selects a tab or a panel for the user; the one exception is the ROIs
tab the moment Manual ROI Labeling is switched on. Runs report through the
process manager (the status button and its console), not a tab of their own.

Tab order is ``priority``; the viewer draws them in that order.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.pipelines import draw_pipeline_windows

__all__ = [
    "PreviewTabWidget",
    "RoiTableTabWidget",
    "TraceTableTabWidget",
    "RunTabWidget",
    "SignalQualityTabWidget",
]


class PreviewTabWidget(Widget):
    """The Image tab: every panel widget, stacked."""

    name = "Image"
    tab_label = "Image"
    placement = "tab"
    toggle_key = "preview"
    priority = 10

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def draw(self) -> None:
        # draw_preview_section opens its own child window
        self.parent.draw_preview_section()


class SignalQualityTabWidget(Widget):
    """The Signal Quality tab: the metric table, once the z-stats have been
    computed. Its plot is the top strip's Signal Quality panel, which has the
    canvas's full width.
    """

    name = "Signal Quality"
    tab_label = "Signal Quality"
    placement = "tab"
    toggle_key = "signal_quality"
    priority = 20

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def tab_disabled(self) -> str | None:
        if all(self.parent._zstats_done):
            return None
        return ""

    def draw(self) -> None:
        with imgui_ctx.begin_child(
            "##StatsContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            self.parent.draw_stats_section()


class RunTabWidget(Widget):
    """The Process tab: registration / segmentation pipelines."""

    name = "Process"
    tab_label = "Process"
    placement = "tab"
    toggle_key = "run"
    priority = 30

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._has_pipeline: bool | None = None
        # pipelines popped out as floating windows draw from the strip's
        # frame hook, so they stay up whatever tab is selected
        self._draw_windows = partial(draw_pipeline_windows, parent)
        strip = getattr(parent, "top_strip", None)
        if strip is not None:
            strip.add_hook(self._draw_windows)

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def cleanup(self) -> None:
        strip = getattr(self.parent, "top_strip", None)
        if strip is not None:
            strip.remove_hook(self._draw_windows)

    def _pipeline_available(self) -> bool:
        if self._has_pipeline is None:
            from mbo_utilities.gui.widgets.pipelines import any_pipeline_available

            self._has_pipeline = any_pipeline_available()
        return self._has_pipeline

    def tab_disabled(self) -> str | None:
        if self._pipeline_available():
            return None
        # Show each registered pipeline's install command so the user can pick
        # whichever applies to their workflow, instead of hard-coding Suite2p.
        from mbo_utilities.gui.widgets.pipelines import get_available_pipelines

        pipelines = get_available_pipelines()
        if not pipelines:
            return (
                "No pipelines registered.\nInstall with: uv pip install mbo_utilities"
            )
        lines = ["No pipeline is installed.\nInstall one of:"]
        for cls in pipelines:
            lines.append(f"  {cls.name}: {cls.install_command}")
        return "\n".join(lines)

    def wants_focus(self) -> bool:
        """One-shot programmatic focus, used by scripts/capture_docs.py."""
        if getattr(self.parent, "_force_run_tab", False):
            self.parent._force_run_tab = False
            return True
        return False

    def draw(self) -> None:
        from mbo_utilities.gui.widgets.pipelines import draw_run_tab

        with imgui_ctx.begin_child(
            "##RunContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            draw_run_tab(self.parent)


class RoiTableTabWidget(Widget):
    """The ROIs tab: the manual-ROI control sections (NAVIGATE, DRAW, VIEW,
    LABELS), the status row and the ROI table with its per-row trace actions.

    The panel itself is built by ``PreviewDataWidget.sync_manual_roi`` when
    the menu entry is switched on; switching it on is the one time this tab
    selects itself.
    """

    name = "ROIs"
    tab_label = "ROIs"
    placement = "tab"
    toggle_key = "manual_roi"
    priority = 45

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def tab_disabled(self) -> str | None:
        if getattr(self.parent, "manual_roi", None) is None:
            return "Enable Widgets > Manual ROI Labeling to draw ROIs."
        return None

    def wants_focus(self) -> bool:
        # one-shot programmatic focus: the menu toggle or --widget manualroi
        roi = getattr(self.parent, "manual_roi", None)
        if roi is None:
            return False
        if getattr(roi, "focus_tab", False):
            roi.focus_tab = False
            return True
        return False

    def draw(self) -> None:
        roi = getattr(self.parent, "manual_roi", None)
        if roi is None:
            imgui.text_disabled("Manual ROI Labeling is off.")
            return
        with imgui_ctx.begin_child(
            "##RoiTableContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            roi.draw_rois()


class TraceTableTabWidget(Widget):
    """The Traces tab: every collected trace with stats; the rows selected
    here are what the top strip's Traces panel plots.
    """

    name = "Traces"
    tab_label = "Traces"
    placement = "tab"
    toggle_key = "manual_roi"
    priority = 46

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def tab_disabled(self) -> str | None:
        roi = getattr(self.parent, "manual_roi", None)
        if roi is None:
            return "Enable Widgets > Manual ROI Labeling to draw ROIs."
        if not roi.has_traces():
            return "No traces yet: use the trace button on a row of the ROIs tab, or run extract / demix."
        return None

    def draw(self) -> None:
        roi = getattr(self.parent, "manual_roi", None)
        if roi is None:
            imgui.text_disabled("Manual ROI Labeling is off.")
            return
        with imgui_ctx.begin_child(
            "##TraceTableContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            roi.draw_trace_table()
