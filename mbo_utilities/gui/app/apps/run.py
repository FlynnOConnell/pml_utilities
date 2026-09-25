"""The Process tab: the registered pipelines, run on the open data."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets.pipelines import (
    PipelineWidget,
    get_available_pipelines,
    pipeline_installed,
    start_preload,
)

WARN = imgui.ImVec4(1.0, 0.7, 0.2, 1.0)
INFO = imgui.ImVec4(0.6, 0.8, 1.0, 1.0)


class RunApp(App):
    """The pipelines that apply to the open data, each with its settings and run button.

    Every registered pipeline is listed: the ones that run on the open data
    first, then the installed ones it does not apply to, then the ones not
    installed, and picking a non-runnable one says why. The pipeline
    widgets draw themselves against the host's ``context``; they are built
    on first use, one per pipeline, and kept across datasets, as is which
    one is selected (by name, since the runnable set changes with the
    data). ``show(name)`` selects one and brings the tab up, for a panel
    that sends the user here.
    """

    id = "run"
    title = "Process"
    dock = "right"
    order = 2
    size = 380

    def __init__(self):
        super().__init__()
        start_preload()
        self.selected: str | None = None
        self.pipelines: dict[str, PipelineWidget] = {}
        # applies_to can open the source file, so it is asked once per array
        self._applies: tuple | None = None

    def available(self, host) -> bool:
        return host.context is not None

    def show(self, name: str) -> None:
        """Select the pipeline called ``name`` and bring the tab up on it."""
        self.selected = name
        self.open = True
        self.focus = True

    def data_changed(self, host) -> None:
        self._applies = None

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        classes = get_available_pipelines()
        if not classes:
            imgui.text_colored(WARN, "No pipelines registered.")
            imgui.text("Install a pipeline package:")
            imgui.text_colored(INFO, "uv pip install mbo_utilities")
            return
        arr = host.data
        if self._applies is None or self._applies[0] is not arr:
            applies = {}
            for cls in classes:
                try:
                    applies[cls] = bool(cls.applies_to(arr))
                except Exception:
                    applies[cls] = False
            self._applies = (arr, applies)
        applies = self._applies[1]
        runnable, other, missing = [], [], []
        for cls in classes:
            if not pipeline_installed(cls):
                missing.append(cls)
            elif applies.get(cls, False):
                runnable.append(cls)
            else:
                other.append(cls)
        # Isoview ahead of Suite2p when both apply: Suite2p applies to any array
        runnable.sort(key=lambda cls: 0 if cls.name == "Isoview" else 1)
        entries = [(cls, cls.name, "ok") for cls in runnable]
        entries += [(cls, f"{cls.name} (not applicable)", "na") for cls in other]
        entries += [(cls, f"{cls.name} (not installed)", "missing") for cls in missing]
        idx = next(
            (i for i, (cls, _, _) in enumerate(entries) if cls.name == self.selected), 0
        )
        if len(entries) > 1:
            imgui.set_next_item_width(220)
            changed, new_idx = imgui.combo(
                "Pipeline##run_tab", idx, [label for _, label, _ in entries]
            )
            if changed:
                idx = new_idx
            imgui.separator()
        cls, _label, state = entries[idx]
        self.selected = cls.name
        if state == "missing":
            imgui.text(f"{cls.name} is not installed.")
            imgui.text_colored(INFO, cls.install_command)
            return
        if state == "na":
            imgui.text_colored(WARN, f"{cls.name} does not apply to the loaded data.")
            return
        pipeline = self.pipelines.get(cls.name)
        if pipeline is None:
            pipeline = self.pipelines[cls.name] = cls(host.context)
        pipeline.draw()

    def close(self) -> None:
        for pipeline in self.pipelines.values():
            pipeline.cleanup()
        self.pipelines.clear()
