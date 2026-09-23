"""A 2D array on the gpu, drawable inside an imgui window."""

from __future__ import annotations

import numpy as np
import wgpu
from cmap import Colormap


class Texture:
    """One wgpu texture and its imgui registration.

    An app that draws pixels into its own imgui window (rather than into a
    subplot) colours them here and blits the result with
    ``draw_list.add_image(texture.ref, p_min, p_max)``. Re-uploading the same
    shape rewrites the texture in place, so scrubbing a movie does not churn
    gpu allocations.
    """

    def __init__(self, figure):
        self.backend = figure.imgui_renderer.backend
        self.ref = None
        self.width = 0
        self.height = 0
        self._texture = None
        self._view = None
        self._key = None

    def set(self, array: np.ndarray, lo: float, hi: float, cmap: str = "gray") -> None:
        """Colour ``array`` between ``lo`` and ``hi`` and put it on the gpu."""
        # the frame's address, so scrubbing back to a frame reuses its upload
        key = (array.ctypes.data, array.shape, float(lo), float(hi), cmap)
        if key == self._key:
            return
        self._key = key
        normed = np.clip(
            (np.asarray(array, dtype=np.float32) - lo) / max(hi - lo, 1e-12), 0.0, 1.0
        )
        rgba = np.ascontiguousarray((Colormap(cmap)(normed) * 255).astype(np.uint8))
        height, width = rgba.shape[:2]
        if self._texture is None or (width, height) != (self.width, self.height):
            self.destroy()
            self.width, self.height = width, height
            self._texture = self.backend._device.create_texture(
                size=(width, height, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            self._view = self._texture.create_view()
            self.ref = self.backend.register_texture(self._view)
        self.backend._device.queue.write_texture(
            {"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
            rgba.tobytes(),
            {"offset": 0, "bytes_per_row": width * 4, "rows_per_image": height},
            (width, height, 1),
        )

    def destroy(self) -> None:
        if self.ref is not None:
            self.backend.unregister_texture(self.ref)
            self.ref = None
        self._view = None
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None
        self._key = None
