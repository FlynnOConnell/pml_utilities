"""The open movie's current frame, drawn as a texture in a window of its own."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.app._texture import Texture

CMAPS = ("gray", "viridis", "magma", "plasma")


class ImageViewerApp(App):
    """A pan and zoom view of one frame, on the gpu but outside the scene.

    The frame it shows is the host's cursor, so scrubbing the movie moves
    this view too without either app knowing about the other.
    """

    id = "image_viewer"
    title = "Image Viewer"
    window = True
    order = 30
    window_size = (560, 620)

    def __init__(self):
        super().__init__()
        self.texture: Texture | None = None
        self.cmap = "gray"
        self.zoom = 1.0
        self.pan = imgui.ImVec2(0.0, 0.0)
        self.auto = True
        self.lo = 0.0
        self.hi = 1.0

    def available(self, host) -> bool:
        return host.data is not None

    def close(self) -> None:
        if self.texture is not None:
            self.texture.destroy()
            self.texture = None

    def draw_options(self, host) -> None:
        imgui.set_next_item_width(8 * imgui.get_font_size())
        changed, index = imgui.combo("##cmap", CMAPS.index(self.cmap), list(CMAPS))
        if changed:
            self.cmap = CMAPS[index]
        imgui.same_line()
        _, self.auto = imgui.checkbox("auto", self.auto)
        imgui.same_line()
        if imgui.button("fit"):
            self.zoom = 0.0
            self.pan = imgui.ImVec2(0.0, 0.0)
        imgui.same_line()
        imgui.set_next_item_width(-imgui.FLT_MIN)
        _, self.zoom = imgui.slider_float("##zoom", self.zoom, 0.0, 8.0, "zoom %.2f")
        if not self.auto:
            frame = host.data[host.index]
            span = float(frame.max() - frame.min()) or 1.0
            imgui.set_next_item_width(-imgui.FLT_MIN)
            _, self.lo = imgui.slider_float(
                "##lo", self.lo, float(frame.min()), float(frame.max()), "low %.3g"
            )
            imgui.set_next_item_width(-imgui.FLT_MIN)
            _, self.hi = imgui.slider_float(
                "##hi",
                self.hi,
                float(frame.min()),
                float(frame.min()) + span,
                "high %.3g",
            )

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        frame = np.asarray(host.data[host.index])
        if self.auto:
            self.lo, self.hi = (
                float(np.nanpercentile(frame, 1.0)),
                float(np.nanpercentile(frame, 99.0)),
            )
        if self.texture is None:
            self.texture = Texture(host.figure)
        self.texture.set(frame, self.lo, max(self.hi, self.lo + 1e-9), self.cmap)

        origin = imgui.get_cursor_screen_pos()
        imgui.invisible_button("##canvas", size)
        if imgui.is_item_active():
            self.pan = imgui.ImVec2(
                self.pan.x + imgui.get_io().mouse_delta.x,
                self.pan.y + imgui.get_io().mouse_delta.y,
            )
        if imgui.is_item_hovered() and imgui.get_io().mouse_wheel:
            self.zoom = max(
                0.1, self.zoom * (1.1 if imgui.get_io().mouse_wheel > 0 else 0.9)
            )

        fit = min(size.x / self.texture.width, size.y / self.texture.height)
        scale = fit if self.zoom <= 0.0 else self.zoom
        width, height = self.texture.width * scale, self.texture.height * scale
        p_min = imgui.ImVec2(
            origin.x + (size.x - width) * 0.5 + self.pan.x,
            origin.y + (size.y - height) * 0.5 + self.pan.y,
        )
        p_max = imgui.ImVec2(p_min.x + width, p_min.y + height)
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(
            origin, imgui.ImVec2(origin.x + size.x, origin.y + size.y), True
        )
        draw_list.add_image(self.texture.ref, p_min, p_max)
        draw_list.pop_clip_rect()

        if imgui.is_item_hovered() and width > 0 and height > 0:
            mouse = imgui.get_io().mouse_pos
            col = int((mouse.x - p_min.x) / scale)
            row = int((mouse.y - p_min.y) / scale)
            if 0 <= row < frame.shape[0] and 0 <= col < frame.shape[1]:
                imgui.set_tooltip(f"({row}, {col}) {float(frame[row, col]):.4g}")
