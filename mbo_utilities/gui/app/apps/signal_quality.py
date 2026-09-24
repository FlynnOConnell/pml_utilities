"""How much signal each plane holds: the summary stats table and plot."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui._stats import draw_stats_section
from mbo_utilities.gui.app._app import App


class SignalQualityApp(App):
    """The summary stats of the open data, per plane, as a table over a plot.

    The host computes them once per dataset (cached ones are read back from
    the store); this only draws them, with a progress bar while they run.
    """

    id = "signal_quality"
    title = "Signal Quality"
    window = True
    order = 50
    window_size = (900, 640)

    def available(self, host) -> bool:
        return host.zstats is not None

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        stats = host.zstats
        if any(stats.done):
            draw_stats_section(stats)
        elif any(stats.running):
            imgui.progress_bar(
                sum(stats.progress) / len(stats.progress),
                imgui.ImVec2(-imgui.FLT_MIN, 0),
                "computing summary stats",
            )
        else:
            imgui.text_disabled("No summary stats for this data.")
