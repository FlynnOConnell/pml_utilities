"""Pollen calibration: the beamlet offsets and depths measured from a pollen stack."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.viewers.pollen_calibration import PollenCalibrationViewer
from mbo_utilities.lazy_array import base_array


class PollenApp(App):
    """The pollen calibration panel, over a ScanImage pollen stack.

    Built for the stack on screen and dropped with it.
    """

    id = "pollen"
    title = "Pollen"
    dock = "right"
    order = 0
    size = 380
    start_open = True

    def __init__(self):
        super().__init__()
        self.viewer: PollenCalibrationViewer | None = None

    def available(self, host) -> bool:
        return (
            host.data is not None
            and getattr(base_array(host.data), "stack_type", None) == "pollen"
        )

    def data_changed(self, host) -> None:
        self.close()

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        if self.viewer is None:
            self.viewer = PollenCalibrationViewer(host.viewer, host.data.source_path)
        self.viewer.draw()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.cleanup()
            self.viewer = None
