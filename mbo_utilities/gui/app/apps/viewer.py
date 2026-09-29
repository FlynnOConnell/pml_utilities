"""The open array on the canvas: the n-d viewer, its sliders and its contrast."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.arrays.features import find_slider_name
from mbo_utilities.gui._colormaps import DEFAULT_COLORMAPS
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.run_gui import _squeeze_for_viewer


class ViewerApp(App):
    """The open array's images, with a slider for every T, C and Z it has.

    fastplotlib's ``NDWidget`` builds its own figure, so the viewer is made
    first and the host is built on that figure: the viewer's subplots are
    never slots, and its sliders own the bottom edge. The playhead is the
    one time on screen, so the T slider and the playhead are reconciled
    once a frame: whichever moved wins. C and Z are read back onto the
    host for every other app to follow.
    """

    id = "viewer"
    title = "Image"
    dock = "right"
    order = 1
    size = 300
    start_open = True

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        # the T index the viewer showed at the end of the last frame
        self._shown = 0

    def available(self, host) -> bool:
        return host.data is not None

    def data_changed(self, host) -> None:
        self.viewer.data[0] = _squeeze_for_viewer(host.data)
        self._shown = 0

    def frame(self, host) -> None:
        names = self.viewer.dim_names
        index = self.viewer.current_index
        t_name = find_slider_name(names, "t")
        c_name = find_slider_name(names, "c")
        z_name = find_slider_name(names, "z")
        if t_name is not None:
            if index[t_name] != self._shown:
                host.seek_frame(index[t_name], source=self)
            elif index[t_name] != host.frame:
                self.viewer.indices[t_name] = host.frame
            self._shown = host.frame
        host.channel = index[c_name] if c_name else 0
        host.zplane = index[z_name] if z_name else 0

    def draw_options(self, host) -> None:
        nt, nc, nz, ny, nx = host.data.shape
        imgui.text(f"T {nt}  C {nc}  Z {nz}  {ny} x {nx}  {host.data.dtype}")
        imgui.set_next_item_width(-imgui.FLT_MIN)
        cmaps = list(DEFAULT_COLORMAPS)
        current = self.viewer.cmap[0]
        if imgui.begin_combo("##cmap", current or "rgb"):
            for name in cmaps:
                if imgui.selectable(name, name == current)[0]:
                    self.viewer.cmap = name
            imgui.end_combo()
        if imgui.button("Contrast: data"):
            self.viewer.reset_vmin_vmax()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Fit the contrast to a sample of the whole array.")
        imgui.same_line()
        if imgui.button("Contrast: frame"):
            self.viewer.reset_vmin_vmax_frame()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Fit the contrast to the frame on screen.")
