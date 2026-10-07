"""Voltage preprocessing widget: the MaskNMF pipeline up to the denoised movie.

Registration estimates its shifts on a quick denoised copy of the movie and
applies them to the raw frames, then the registered movie is compressed and
denoised; demixing starts off. Run it on a frame window (Set slice) to try
settings, and View movies to see raw, the alignment copy, registered and
denoised side by side. Everything else is the MaskNMF widget.
"""

from pathlib import Path

from imgui_bundle import imgui

from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.widgets.pipelines.masknmf import (
    _BTN_W,
    MaskNMFPipelineWidget,
)


class VoltagePreprocessingWidget(MaskNMFPipelineWidget):
    """MaskNMF registration and denoising on a denoised copy, with a movie viewer."""

    name = "Voltage Preprocessing"
    run_label = "Run Preprocessing"

    def __init__(self, parent):
        super().__init__(parent)
        self._viewer = None

    def default_settings(self):
        from mbo_utilities.masknmf.params import STAGE_SKIP

        settings = super().default_settings()
        settings.registration.denoised_reference = True
        settings.demixing.do_demixing = STAGE_SKIP
        settings.runtime.keep_raw = True
        return settings

    def _draw_run(self) -> None:
        super()._draw_run()
        if not self._outdir:
            imgui.begin_disabled()
        clicked = imgui.button("View movies##voltage_prep_view", imgui.ImVec2(_BTN_W * 1.5, 0))
        if not self._outdir:
            imgui.end_disabled()
        set_tooltip(
            "Raw, the alignment copy, registered and denoised side by side, from "
            "the plane folder last written under the output folder.",
            show_mark=False,
        )
        if not clicked:
            return
        written = [(p.stat().st_mtime, p.parent) for p in Path(self._outdir).glob("**/ops.npy")]
        if not written:
            self._last_status = f"No plane folder in {self._outdir} yet."
            return
        from mbo_utilities.gui.registration_viewer import registration_viewer

        plane = max(written)[1]
        self._viewer = registration_viewer(plane)
        self._viewer.show()
        self._last_status = f"Showing {plane.name}."
