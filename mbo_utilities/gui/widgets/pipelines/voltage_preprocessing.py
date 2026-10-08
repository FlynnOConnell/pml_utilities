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

    def default_settings(self):
        from mbo_utilities.masknmf.params import STAGE_SKIP

        settings = super().default_settings()
        settings.registration.denoised_reference = True
        settings.demixing.do_demixing = STAGE_SKIP
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
            "the run folder last written under the output folder.",
            show_mark=False,
        )
        if not clicked:
            return
        written = [(p.stat().st_mtime, p.parent) for p in Path(self._outdir).glob("**/config.json")]
        if not written:
            self._last_status = f"No run folder in {self._outdir} yet."
            return
        from mbo_utilities.gui.launch import launch_window

        run = max(written)[1]
        # its own process: a second figure built inside this imgui frame crashes imgui
        pid = launch_window("mbo_utilities.gui.registration_viewer", [str(run)], f"movies_{run.name}")
        self._last_status = f"Opened {run.name} in its own window (PID {pid})."
