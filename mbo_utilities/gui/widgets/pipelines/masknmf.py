"""masknmf pipeline widget.

pml_utilities prepares the data and masknmf processes it: the Current-dataset
block, the output folder and the Frames & Planes selection here pick what each
plane's ``(T, Y, X)`` view holds, and masknmf's own Parameters popup
(``masknmf.visualization.imgui.parameters.Parameters``) sets the pipeline and
its stages. Run spawns the "masknmf" worker task, which runs masknmf's pipeline
per plane into masknmf's run folder plus a results zarr. View results opens
the last run in masknmf's viewer, Spike average in the spike-triggered average
window.
"""

import re
from pathlib import Path
from typing import Any

from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui._selection_ui import (
    draw_frame_average_input,
    draw_selection_table,
    resolve_dim_labels,
    set_selection,
    source_timepoints,
)
from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.gui.widgets.pipelines.settings import (
    _dataset_size_bytes,
    _draw_md_field,
    _format_size,
)
from mbo_utilities.install import MASKNMF_HINT
from mbo_utilities.metadata import get_param
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.reader import widget_reader_kwargs

_TITLE_COLOR = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)
_SUB_COLOR = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)
_DIM_COLOR = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
_MODIFIED_COLOR = imgui.ImVec4(1.0, 0.72, 0.40, 1.0)

_RUN_W = 220
_BTN_W = 90

PIPELINE_INITIAL = "two-photon-calcium"

_HAS_MASKNMF: bool | None = None


def _check_masknmf_available() -> bool:
    global _HAS_MASKNMF
    if _HAS_MASKNMF is None:
        import importlib.util

        _HAS_MASKNMF = importlib.util.find_spec("masknmf") is not None
    return _HAS_MASKNMF


def find_masknmf_run(fpath) -> tuple[dict | None, str | None]:
    """
    The newest masknmf run at or under ``fpath``.

    Returns
    -------
    tuple[dict | None, str | None]
        The run's ``config.json`` and the output folder Run writes the next run into (the folder holding its
        ``zplaneNN/`` or ``roiNN/zplaneNN/``); both None when ``fpath`` is neither a run folder nor a folder holding one.
    """
    from mbo_utilities.arrays.masknmf_run import is_masknmf_run, run_config

    if fpath is None:
        return None, None
    if isinstance(fpath, (list, tuple)):
        if not fpath:
            return None, None
        fpath = fpath[0]
    p = Path(str(fpath))
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        return None, None
    if is_masknmf_run(p):
        run = p
    else:
        runs = sorted(c.parent for c in p.glob("**/config.json") if is_masknmf_run(c.parent))
        if not runs:
            return None, None
        run = runs[-1]
    outdir = run.parent.parent if run.parent.name.startswith("zplane") else run.parent
    if re.fullmatch(r"roi\d+", outdir.name):
        outdir = outdir.parent
    return run_config(run), str(outdir)


class MaskNMFPipelineWidget(PipelineWidget):
    """masknmf processing widget."""

    name = "MaskNMF"
    run_label = "Run MaskNMF"
    install_command = MASKNMF_HINT

    @property
    def is_available(self) -> bool:
        return _check_masknmf_available()

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._parameters = None
        self._outdir = ""
        self._outdir_dialog = None
        self._fix_phase = True
        self._use_fft = True
        self._invert_deflection = False
        self._last_status = ""
        self._show_slice_popup = False
        self._last_fpath = None
        self._masknmf_last_max_tp = None
        self._masknmf_last_num_planes = None
        self._masknmf_last_num_channels = None
        self._masknmf_tp_parsed = None
        self._masknmf_z_parsed = self._masknmf_c_parsed = None
        # the windows View results and Spike average started, until they exit
        self._windows: list = []

    @property
    def parameters(self):
        """masknmf's Parameters popup and its values, built on first use since masknmf imports torch."""
        if self._parameters is None:
            from masknmf.visualization.imgui.parameters import Parameters

            self._parameters = Parameters(PIPELINE_INITIAL)
        return self._parameters

    def _array(self):
        iw = self.parent.image_widget
        if iw is None or not iw.data:
            return None
        return iw.data[0]

    def _dims(self) -> tuple[int, int, int]:
        """(max_frames, num_planes, num_channels) of the loaded data."""
        arr = self._array()
        max_frames = 1000 if arr is None else int(arr.num_timepoints)
        return max_frames, int(self.parent.nz or 1), int(self.parent.nc or 1)

    def _hydrate_from_run(self, fpath) -> None:
        """Fill the parameters and output folder from a previous run's tree (opening results hydrates the Run tab)."""
        config, outdir = find_masknmf_run(fpath)
        if outdir is None:
            return
        self._outdir = outdir
        try:
            self.parameters.load(config)
        except ValueError as e:
            self._last_status = f"Previous run found; its parameters do not load: {e}"
            return
        self._last_status = "Loaded parameters from previous run"

    def _ensure_slice_state(self) -> None:
        """Seed/reset slicing state when the dataset changes (suite2p parity)."""
        fpath = self.parent.fpath
        max_frames, num_planes, num_channels = self._dims()
        fpath_changed = fpath != self._last_fpath
        if fpath_changed:
            self._invert_deflection = bool(self.parent.invert_deflection)
            self._hydrate_from_run(fpath)
        if fpath_changed or self._masknmf_last_max_tp != max_frames:
            self._last_fpath = fpath
            self._masknmf_last_max_tp = max_frames
            set_selection(self, "_masknmf_tp", max_frames)
        if self._masknmf_last_num_planes != num_planes:
            self._masknmf_last_num_planes = num_planes
            set_selection(self, "_masknmf_z", num_planes)
        if self._masknmf_last_num_channels != num_channels:
            self._masknmf_last_num_channels = num_channels
            set_selection(self, "_masknmf_c", num_channels)

    def _selected_planes(self) -> list[int]:
        parsed = self._masknmf_z_parsed
        return [i + 1 for i in parsed.final_indices] if parsed else []

    def _selected_channels(self) -> list[int]:
        parsed = self._masknmf_c_parsed
        return [i + 1 for i in parsed.final_indices] if parsed else []

    def _tp_indices(self) -> list[int] | None:
        parsed = self._masknmf_tp_parsed
        if parsed is not None and parsed.final_indices:
            return list(parsed.final_indices)
        return None

    def draw_config(self) -> None:
        self._ensure_slice_state()
        imgui.spacing()
        self._draw_dataset_block()
        from mbo_utilities.gui._gpu_status import draw_gpu_status

        draw_gpu_status(str(self.parameters.runtime().get("device", "auto")))
        imgui.separator()
        self._draw_output_row()
        self._draw_slice_row()
        imgui.spacing()
        self._draw_pipeline_row()
        self.parameters.draw()
        imgui.spacing()
        self._draw_modified_table()
        imgui.spacing()
        self._draw_run()

    def _draw_pipeline_row(self) -> None:
        """The masknmf pipeline, and the button that opens its Parameters popup."""
        from masknmf.pipelines import scraper

        slugs = sorted(scraper.pipeline_registry())
        imgui.text_colored(_SUB_COLOR, "Pipeline")
        imgui.set_next_item_width(max(imgui.get_content_region_avail().x - _BTN_W * 1.3 - 12, 100))
        index = slugs.index(self.parameters.slug)
        changed, index = imgui.combo("##masknmf_pipeline", index, slugs)
        if changed:
            self.parameters.select(slugs[index])
        set_tooltip("The masknmf pipeline each plane runs through.", show_mark=False)
        imgui.same_line()
        if imgui.button("Parameters##masknmf_parameters", imgui.ImVec2(_BTN_W * 1.3, 0)):
            self.parameters.open()
        set_tooltip("masknmf's parameters: a column per stage, the runtime under them.", show_mark=False)

    def _draw_modified_table(self) -> None:
        """The parameters that differ from the pipeline's defaults, one row each."""
        rows = self.parameters.modified()
        if not rows:
            imgui.text_colored(_DIM_COLOR, "All parameters at their defaults.")
            return
        imgui.text_colored(_MODIFIED_COLOR, f"{len(rows)} parameter(s) changed")
        flags = imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.sizing_stretch_prop
        if imgui.begin_table("##masknmf_modified", 3, flags):
            imgui.table_setup_column("parameter")
            imgui.table_setup_column("value")
            imgui.table_setup_column("default")
            imgui.table_headers_row()
            for name, value, default in rows:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(name)
                imgui.table_next_column()
                imgui.text_colored(_MODIFIED_COLOR, value)
                imgui.table_next_column()
                imgui.text_disabled(default)
            imgui.end_table()

    def _draw_dataset_block(self) -> None:
        imgui.text_colored(_TITLE_COLOR, "Current dataset")
        arr = self._array()
        if arr is None:
            imgui.text_disabled("No dataset loaded.")
            return
        fpath = self.parent.fpath
        path_str = str(fpath) if fpath else ""
        short_name = Path(path_str).name or path_str if path_str else "(in-memory)"
        imgui.indent(8)
        imgui.text(f"Name: {short_name}")
        if path_str and imgui.is_item_hovered():
            imgui.set_tooltip(path_str)
        filenames = getattr(arr, "filenames", None) or ([path_str] if path_str else [])
        if filenames:
            imgui.text(f"Size on disk: {_format_size(_dataset_size_bytes(filenames))}")
        imgui.text("Shape: " + " × ".join(str(s) for s in arr.shape))
        md = dict(arr.metadata or {})
        md.update(self.parent._custom_metadata)
        _draw_md_field("Frame rate", get_param(md, "fs"), "Hz")
        _, num_planes, _ = self._dims()
        if num_planes > 1:
            _draw_md_field("Z-step", get_param(md, "dz"), "µm")
        imgui.unindent(8)

    def _draw_output_row(self) -> None:
        if self._outdir_dialog is not None and self._outdir_dialog.ready():
            result = self._outdir_dialog.result()
            if result:
                self._outdir = result
                set_last_dir("masknmf_outdir", result)
            self._outdir_dialog = None

        imgui.text_colored(_SUB_COLOR, "Output folder")
        imgui.set_next_item_width(
            max(imgui.get_content_region_avail().x - _BTN_W - 12, 100)
        )
        _, self._outdir = imgui.input_text("##masknmf_outdir", self._outdir)
        set_tooltip(
            "Save path. Each plane runs into masknmf's own run folder under zplaneNN/.",
            show_mark=False,
        )
        imgui.same_line()
        if imgui.button("Browse##masknmf_outdir_btn", imgui.ImVec2(_BTN_W, 0)):
            start = str(get_last_dir("masknmf_outdir") or Path.home())
            self._outdir_dialog = pfd.select_folder("Select output folder", start)

    def _draw_slice_row(self) -> None:
        max_frames, num_planes, num_channels = self._dims()
        if imgui.button("Set slice##masknmf_slice", imgui.ImVec2(_BTN_W, 0)):
            self._show_slice_popup = True
        set_tooltip("Timepoints / planes / channels to process.", show_mark=False)
        imgui.same_line()
        tp = self._tp_indices()
        n_tp = len(tp) if tp is not None else max_frames
        planes = self._selected_planes()
        channels = self._selected_channels()
        preview = f"{n_tp} frames · {len(planes)} plane(s)"
        if num_channels > 1:
            preview += f" · {len(channels)} channel(s)"
        imgui.text_colored(_DIM_COLOR, preview)

        if self._show_slice_popup:
            imgui.open_popup("Frames & Planes##masknmf_slice")
            self._show_slice_popup = False
        imgui.set_next_window_size(imgui.ImVec2(520, 0), imgui.Cond_.first_use_ever)
        if imgui.begin_popup("Frames & Planes##masknmf_slice"):
            tp_label, z_label, c_label = resolve_dim_labels(self.parent)
            draw_selection_table(
                self,
                max_frames,
                num_planes,
                tp_attr="_masknmf_tp",
                z_attr="_masknmf_z",
                id_suffix="_masknmf",
                num_channels=num_channels,
                c_attr="_masknmf_c",
                tp_label=tp_label,
                z_label=z_label,
                c_label=c_label,
            )
            imgui.spacing()
            _, self._fix_phase = imgui.checkbox(
                "Fix scan phase##masknmf_fixphase", self._fix_phase
            )
            if self._fix_phase:
                imgui.same_line()
                _, self._use_fft = imgui.checkbox("FFT##masknmf_fft", self._use_fft)
            _, self._invert_deflection = imgui.checkbox(
                "Invert deflection##masknmf_invert", self._invert_deflection
            )
            set_tooltip(
                "Flip every frame about the mean image before processing, so an "
                "indicator that dims on activity (ASAP, Voltron) reads positive. "
                "Starts as the viewer's Invert Deflection.",
                show_mark=False,
            )
            # temporal binning lives on the parent so "Apply to dataset" in
            # Window Functions seeds it, the same way the save-as menu is
            self.parent._masknmf_frame_average = draw_frame_average_input(
                "Frame average##masknmf_frame_average",
                self.parent._masknmf_frame_average,
                viewer_factor=int(self.parent.frame_average or 1),
                max_frames=source_timepoints(self.parent),
            )
            imgui.spacing()
            if imgui.button("Close##masknmf_slice_close", imgui.ImVec2(_BTN_W, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

    def _draw_run(self) -> None:
        planes = self._selected_planes()
        fpath = self.parent.fpath
        has_save_path = bool(self._outdir)
        channels = self._selected_channels()
        problems = self.parameters.problems()
        ready = has_save_path and bool(fpath) and bool(planes) and bool(channels) and not problems

        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
        imgui.push_style_color(
            imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0)
        )
        imgui.push_style_color(
            imgui.Col_.button_active, imgui.ImVec4(0.1, 0.45, 0.1, 1.0)
        )
        if not ready:
            imgui.begin_disabled()
        run_avail = imgui.get_content_region_avail().x
        if run_avail > _RUN_W:
            imgui.set_cursor_pos_x(
                imgui.get_cursor_pos_x() + (run_avail - _RUN_W) * 0.5
            )
        clicked = imgui.button(self.run_label, imgui.ImVec2(_RUN_W, 0))
        if not ready:
            imgui.end_disabled()
        imgui.pop_style_color(3)

        if not ready and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "\n".join(problems)
                if problems
                else "Load a dataset and set the output folder"
                if not has_save_path
                else "Load a dataset"
            )
        if self._last_status:
            imgui.text_colored(_DIM_COLOR, self._last_status)

        if clicked and ready:
            self._submit(planes)

        if not self._outdir:
            imgui.begin_disabled()
        view = imgui.button("View results##masknmf_view", imgui.ImVec2(_BTN_W * 1.5, 0))
        set_tooltip(
            "The last run under the output folder that demixed, in masknmf's own "
            "viewer: raw, registered and demixed movies with the shifts.",
            show_mark=False,
        )
        imgui.same_line()
        sta = imgui.button("Spike average##masknmf_sta", imgui.ImVec2(_BTN_W * 1.5, 0))
        set_tooltip(
            "The last run that demixed, averaged around the spikes of one of its "
            "components: the registered and compressed movies, the component's "
            "trace, RTMC and masknmf shifts. Its own window.",
            show_mark=False,
        )
        if not self._outdir:
            imgui.end_disabled()
        for window in list(self._windows):
            state = window.poll()
            if state is not None:
                self._windows.remove(window)
                if state:
                    self._last_status = state
        if not (view or sta):
            return
        written = sorted(
            (p.stat().st_mtime, p.parent)
            for p in Path(self._outdir).glob("**/config.json")
        )
        from mbo_utilities.arrays.masknmf_run import run_demixing

        written = [(t, run) for t, run in written if run_demixing(run)]
        if not written:
            self._last_status = f"No run folder that demixed in {self._outdir} yet."
            return
        from mbo_utilities.gui.launch import LaunchedWindow, launch_window

        run = written[-1][1]
        # its own process: a second figure built inside this imgui frame crashes imgui
        if sta:
            from mbo_utilities.gui.spike_average_viewer import launch_spike_average

            pid, log_name = launch_spike_average(run)
            self._windows.append(LaunchedWindow(pid, log_name, "Spike average"))
        else:
            log_name = f"masknmf_view_{run.name}"
            pid = launch_window("masknmf.cli", ["view", str(run_demixing(run))], log_name)
            self._windows.append(LaunchedWindow(pid, log_name, "masknmf viewer"))
        self._last_status = f"Opening {run.name} in its own window (PID {pid})."

    def _submit(self, planes: list[int]) -> None:
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        pm = get_process_manager()
        channels = self._selected_channels()
        multi_channel = len(channels) > 1
        # multi-channel SOURCES need the channel selection even when a
        # single channel is picked, else the bin write takes every channel
        has_channels = self._dims()[2] > 1
        tp_indices = self._tp_indices()

        started = []
        for channel in channels:
            output_dir = (
                str(Path(self._outdir) / f"ch{channel:02d}")
                if multi_channel
                else self._outdir
            )
            args = {
                "input_path": str(self.parent.fpath),
                "reader_kwargs": widget_reader_kwargs(self.parent.image_widget),
                "output_dir": output_dir,
                "planes": planes,
                "config": self.parameters.config(),
                "runtime": self.parameters.runtime(),
                "run_args": self.parameters.run_args(),
                "fix_phase": self._fix_phase,
                "use_fft": self._use_fft,
                "invert_deflection": self._invert_deflection,
                "frame_average": int(self.parent._masknmf_frame_average or 1),
                "tp_indices": tp_indices,
                "channel": channel if (multi_channel or has_channels) else None,
                "custom_metadata": dict(self.parent._custom_metadata),
            }
            if len(planes) == 1:
                description = f"{self.name} plane{planes[0]:02d}"
            else:
                description = f"{self.name}: {len(planes)} plane(s)"
            if multi_channel:
                description += f" ch{channel}"

            pid = pm.spawn(
                task_type="masknmf",
                args=args,
                description=description,
                output_path=output_dir,
            )
            if pid:
                started.append(pid)

        if started:
            self._last_status = f"Started {len(started)} job(s) (PID {started[0]})"
        else:
            self._last_status = "Failed to start worker."

    def cleanup(self) -> None:
        self._outdir_dialog = None
