"""Writing the open array, or a selection of it, to another format."""

from __future__ import annotations

from mbo_utilities.arrays import FrameAveragedView, ScanImageArray, TiffArray
from mbo_utilities.gui._save_as import SaveAs, SaveSource, draw_saveas_popup
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.app.apps.viewer import PROJECTIONS
from mbo_utilities.lazy_array import base_array


class SaveAsApp(App):
    """The Save As dialog over the open array, with the running save's progress.

    Everything the dialog reads about the display comes off the viewer
    itself: its window function is the projection, its spatial function
    carries the blur and the mean image. So the video options' Sync from
    preview takes what is on screen without asking another app.
    """

    id = "save_as"
    title = "Save As"
    menu = "File"
    shortcut = "s"
    window = True
    owns_window = True
    order = 10

    def __init__(self):
        self.save = SaveAs()
        super().__init__()

    @property
    def open(self) -> bool:
        return self.save.modal_open or self.save.open_requested

    @open.setter
    def open(self, value: bool) -> None:
        if value and not self.save.modal_open:
            self.save.open_requested = True
        elif not value:
            self.save.open_requested = False
            self.save.modal_open = False

    def available(self, host) -> bool:
        return host.data is not None and host.viewer is not None

    def data_changed(self, host) -> None:
        data = host.data
        self.save.frame_average = (
            data.factor if isinstance(data, FrameAveragedView) else 1
        )
        self.save.selected_rois = set()
        self.save.split_rois = False

    def progress(self, host) -> list[dict]:
        save = self.save
        save.clear_stale_progress()
        items = []
        if save.running or save.done:
            items.append(
                {
                    "key": "saveas",
                    "text": "Save complete"
                    if save.done
                    else f"Saving z-plane {save.current_index}",
                    "progress": 1.0 if save.done else max(0.01, save.progress),
                    "done": save.done,
                }
            )
        if save.register_running or save.register_done:
            items.append(
                {
                    "key": "register_z",
                    "text": f"Axial registration: {save.register_msg or 'starting'}",
                    "progress": max(0.01, save.register_progress),
                    "done": save.register_done,
                }
            )
        return items

    def draw_window(self, host) -> None:
        data = host.data
        averaged = isinstance(data, FrameAveragedView)
        funcs = host.viewer.window_funcs or {}
        func, window = next(iter(funcs.values()), (PROJECTIONS["mean"], 1))
        keywords = getattr(host.viewer.spatial_func, "keywords", {})
        draw_saveas_popup(
            self.save,
            SaveSource(
                viewer=host.viewer,
                fpath=data.source_path,
                planes=data.shape[2],
                scanimage=isinstance(base_array(data), (ScanImageArray, TiffArray)),
                frame_average=data.factor if averaged else 1,
                source_frames=(data.source if averaged else data).shape[0],
                projection={v: k for k, v in PROJECTIONS.items()}.get(func, "mean"),
                window=window,
                sigma=keywords.get("sigma", 0.0),
                mean_subtraction=keywords.get("mean") is not None,
                zstats=host.zstats,
                metadata=host.metadata_edits.values,
            ),
        )
