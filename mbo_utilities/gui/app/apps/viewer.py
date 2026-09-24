"""The open array on the canvas: the n-d viewer, its sliders and how frames are shown."""

from __future__ import annotations

from functools import partial

import numpy as np
from imgui_bundle import hello_imgui, imgui
from scipy.ndimage import gaussian_filter

from mbo_utilities import log
from mbo_utilities.arrays import FrameAveragedView, average_frames
from mbo_utilities.arrays.features import find_slider_name
from mbo_utilities.gui._colormaps import DEFAULT_COLORMAPS
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.app._keys import pressed
from mbo_utilities.gui.run_gui import _squeeze_for_viewer
from mbo_utilities.lazy_array import base_array

logger = log.get("gui.app.viewer")

PROJECTIONS = {"mean": np.mean, "max": np.max, "std": np.std}
HEADER = imgui.ImVec4(0.8, 0.8, 0.2, 1.0)
GOOD = imgui.ImVec4(0.6, 0.8, 0.6, 1.0)
BAD = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)


def blur(frame, sigma: float) -> np.ndarray:
    """One displayed frame smoothed by a gaussian of ``sigma`` pixels."""
    return gaussian_filter(np.asarray(frame, dtype=np.float32), sigma)


class ViewerApp(App):
    """The open array's images, with a slider for every T, C and Z it has.

    fastplotlib's ``NDWidget`` builds its own figure, so the viewer is made
    first and the host is built on that figure: the viewer's subplots are
    never slots, and its sliders own the bottom edge. The playhead is the
    one time on screen, so the T slider and the playhead are reconciled
    once a frame: whichever moved wins. C and Z are read back onto the
    host for every other app to follow.

    The panel is the preview window's Image tab: a sliding projection over
    T, a gaussian blur, temporal binning of the data itself, scan-phase
    correction and contrast. Display settings reset when another recording
    opens and survive rebinning the same one.
    """

    id = "viewer"
    title = "Image"
    dock = "right"
    order = 1
    size = 300
    start_open = True
    keybinds = (
        ("Left / Right", "Previous / next frame"),
        ("Up / Down", "Next / previous z-plane"),
        ("Shift+arrows", "Step 10 at a time"),
        ("Space", "Play / pause"),
        ("v", "Fit the contrast to the frame"),
        ("Shift+V", "Refit on channel or plane change"),
        ("c", "Fix scan phase"),
        ("Shift+C", "Sub-pixel scan phase"),
    )

    def __init__(self, viewer, data):
        super().__init__()
        self.viewer = viewer
        # the recording under every view, to tell a rebin from another file
        self.recording = base_array(data)
        self.projection = "mean"
        self.window = 1
        self.sigma = 0.0
        self.auto_contrast = False
        # the T index and (c, z) the viewer showed at the end of the last frame
        self._shown = 0
        self._plane = (0, 0)

    def available(self, host) -> bool:
        return host.data is not None

    def data_changed(self, host) -> None:
        recording = base_array(host.data)
        if recording is not self.recording:
            self.sigma = 0.0
            self.auto_contrast = False
        self.recording = recording
        self.projection = "mean"
        self.window = 1
        # the swap clears the viewer's window and spatial funcs
        self.viewer.data[0] = _squeeze_for_viewer(host.data)
        if self.sigma > 0:
            self.viewer.spatial_func = partial(blur, sigma=self.sigma)
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
        if (host.channel, host.zplane) != self._plane:
            self._plane = (host.channel, host.zplane)
            if self.auto_contrast:
                self.viewer.reset_vmin_vmax_frame()

    def on_keys(self, host) -> None:
        names = self.viewer.dim_names
        t_name = find_slider_name(names, "t")
        z_name = find_slider_name(names, "z")
        data = host.data
        source = data.source if isinstance(data, FrameAveragedView) else data
        if pressed("v"):
            self.viewer.reset_vmin_vmax_frame()
        if pressed("Shift+V"):
            self.auto_contrast = not self.auto_contrast
        if hasattr(source, "phase_correction"):
            if pressed("c"):
                source.fix_phase = not source.fix_phase
                self.viewer.indices = self.viewer.current_index
            if pressed("Shift+C") and source.fix_phase:
                source.use_fft = not source.use_fft
                self.viewer.indices = self.viewer.current_index
        if t_name is not None and pressed("Space"):
            sliders = self.viewer._sliders_ui
            sliders._playing[t_name] = not sliders._playing[t_name]
            sliders._last_frame_time[t_name] = 0
        # a panel under the mouse keeps the arrows for its own sliders
        if imgui.get_io().want_capture_mouse:
            return
        for key, name, step in (
            ("Left", t_name, -1),
            ("Right", t_name, 1),
            ("Down", z_name, -1),
            ("Up", z_name, 1),
        ):
            if name is None:
                continue
            last = data.shape[0 if name == t_name else 2] - 1
            for chord, size in ((key, 1), (f"Shift+{key}", 10)):
                if pressed(chord, repeat=True):
                    at = self.viewer.current_index[name] + step * size
                    self.viewer.indices[name] = min(max(at, 0), last)

    def draw_options(self, host) -> None:
        data = host.data
        nt, nc, nz, ny, nx = data.shape
        imgui.text(f"T {nt}  C {nc}  Z {nz}  {ny} x {nx}  {data.dtype}")
        t_name = find_slider_name(self.viewer.dim_names, "t")
        width = hello_imgui.em_size(6)

        if imgui.collapsing_header("Display", imgui.TreeNodeFlags_.default_open):
            imgui.set_next_item_width(-imgui.FLT_MIN)
            # cmap names come back namespaced, gnuplot2 as gnuplot:gnuplot2
            current = self.viewer.cmap[0]
            shown = "rgb" if current is None else current.name.split(":")[-1]
            if imgui.begin_combo("##cmap", shown):
                for name in DEFAULT_COLORMAPS:
                    if imgui.selectable(name, name == shown)[0]:
                        self.viewer.cmap = name
                imgui.end_combo()
            if imgui.button("Contrast: data"):
                self.viewer.reset_vmin_vmax()
            set_tooltip("Fit the contrast to a sample of the whole array.")
            imgui.same_line()
            if imgui.button("Contrast: frame"):
                self.viewer.reset_vmin_vmax_frame()
            set_tooltip("Fit the contrast to the frame on screen.")
            _, self.auto_contrast = imgui.checkbox(
                "Refit on channel or plane change", self.auto_contrast
            )
            set_tooltip("Fit the contrast to the frame whenever C or Z moves.")

        if nt > 1 and t_name is not None:
            if imgui.collapsing_header("Projection", imgui.TreeNodeFlags_.default_open):
                names = list(PROJECTIONS)
                imgui.set_next_item_width(width)
                changed, i = imgui.combo(
                    "Projection", names.index(self.projection), names
                )
                set_tooltip(
                    "What each frame shows over the window around it: the "
                    "mean, the peak or the standard deviation."
                )
                imgui.set_next_item_width(width)
                window_changed, window = imgui.input_int(
                    "Window", self.window, step=1, step_fast=2
                )
                set_tooltip(
                    "Frames in the window, centred on the frame on screen; 1 "
                    "shows the raw frame. Even sizes round up to the next odd."
                )
                if changed or window_changed:
                    self.projection = names[i]
                    self.window = min(max(window, 1), nt)
                    self.viewer.window_funcs = (
                        {
                            t_name: (
                                PROJECTIONS[self.projection],
                                max(3, self.window | 1),
                            )
                        }
                        if self.window > 1
                        else None
                    )

        if imgui.collapsing_header("Blur"):
            imgui.set_next_item_width(width)
            changed, sigma = imgui.input_float(
                "Sigma", self.sigma, step=0.1, step_fast=1.0, format="%.1f"
            )
            set_tooltip("Gaussian blur of the frame on screen, sigma in pixels.")
            if changed:
                self.sigma = max(0.0, sigma)
                self.viewer.spatial_func = (
                    partial(blur, sigma=self.sigma) if self.sigma > 0 else None
                )

        source = data.source if isinstance(data, FrameAveragedView) else data
        factor = data.factor if isinstance(data, FrameAveragedView) else 1
        piezo = hasattr(source, "frames_per_slice") and hasattr(source, "can_average")
        if source.shape[0] > 1 or piezo:
            if imgui.collapsing_header("Frames"):
                if factor > 1:
                    imgui.text_colored(GOOD, f"{factor} frames averaged")
                    imgui.text(f"Frames: {source.shape[0]} -> {nt}")
                    if data.fs and source.fs:
                        imgui.text(f"Rate: {source.fs:.4g} -> {data.fs:.4g} Hz")
                imgui.set_next_item_width(width)
                changed, want = imgui.input_int("Average", factor, step=1, step_fast=2)
                set_tooltip(
                    "Average this many frames into one in the data itself: the "
                    "viewer, the traces and every run started from here read "
                    "the averaged frames and the averaged rate. 1 is the raw data."
                )
                if changed:
                    want = min(max(want, 1), source.shape[0])
                    if want != factor:
                        logger.info(f"frame averaging {want}")
                        host.set_data(average_frames(source, want))
                        return
                if piezo:
                    imgui.text(f"Frames per slice: {source.frames_per_slice}")
                    if source.log_average_factor > 1:
                        imgui.text_colored(
                            GOOD,
                            f"Averaged at acquisition ({source.log_average_factor})",
                        )
                    elif source.can_average:
                        changed, on = imgui.checkbox(
                            "Average each slice", source.average_frames
                        )
                        set_tooltip(
                            f"Read the {source.frames_per_slice} frames taken "
                            "at each z position as their mean."
                        )
                        if changed:
                            source.average_frames = on
                            host.set_data(source)
                            return

        if hasattr(source, "phase_correction"):
            if imgui.collapsing_header("Scan phase", imgui.TreeNodeFlags_.default_open):
                changed, on = imgui.checkbox("Fix phase", source.fix_phase)
                set_tooltip(
                    "Shift every other line to line up with its neighbours, "
                    "undoing the lag of a bidirectional scan."
                )
                if changed:
                    source.fix_phase = on
                imgui.begin_disabled(not source.fix_phase)
                fft_changed, fft = imgui.checkbox("Sub-pixel (slower)", source.use_fft)
                set_tooltip("Find the shift to a fraction of a pixel with an FFT.")
                imgui.end_disabled()
                if fft_changed:
                    source.use_fft = fft
                # a PhaseCorrectedView finds shifts but keeps no per-frame cache
                lookup = getattr(source, "get_offset_at", None)
                offset = lookup and lookup(host.frame, host.channel, host.zplane)
                if offset is not None:
                    imgui.text("Shift on screen:")
                    imgui.same_line()
                    imgui.text_colored(
                        BAD if abs(offset) > source.max_offset else GOOD,
                        f"{offset:.3f} px",
                    )
                imgui.set_next_item_width(width)
                border_changed, border = imgui.input_int(
                    "Border", source.border, step=1, step_fast=2
                )
                set_tooltip("Pixels at each edge left out when finding the shift.")
                imgui.set_next_item_width(width)
                limit_changed, limit = imgui.input_int(
                    "Max shift", source.max_offset, step=1, step_fast=2
                )
                set_tooltip("The largest shift searched for, in pixels.")
                if border_changed:
                    source.border = max(0, border)
                if limit_changed:
                    source.max_offset = max(1, limit)
                if changed or fft_changed or border_changed or limit_changed:
                    self.viewer.indices = self.viewer.current_index
