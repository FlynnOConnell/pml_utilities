"""Orthogonal views of a ZYX volume through one movable point.

The XY, YZ and XZ panels show the slices through the point, each with a draggable
crosshair; the 3D panel shows the same three slices as planes inside the volume.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from mbo_utilities import log

logger = log.get("gui.volume_crosshair")

_VIEWS = ("xy", "yz", "xz")

# per view: (volume axis on the panel's columns, volume axis on the panel's rows)
_VIEW_AXES = {"xy": ("x", "y"), "yz": ("z", "y"), "xz": ("x", "z")}
_AXIS_INDEX = {"z": 0, "y": 1, "x": 2}
_AXIS_COLORS = {"z": "#4fc3f7", "y": "#ffb74d", "x": "#e57373"}


class VolumeCrosshair:
    """Orthogonal slice views of a ZYX volume that follow one point.

    Drag a crosshair line or click a slice panel to move the point.

    Parameters
    ----------
    volume : np.ndarray
        Volume, shape (Z, Y, X).
    voxel_size_zyx : tuple[float, float, float]
        Voxel size along z, y and x, in any one unit; it sets the aspect of the panels.
    cmap : str
        Colormap of every panel.
    vmin, vmax : float, optional
        Contrast limits; ``None`` uses the 1st and 99.5th percentiles of the volume.
    size : tuple[int, int]
        Figure size in pixels.
    """

    def __init__(
        self,
        volume: np.ndarray,
        voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        size: tuple[int, int] = (1000, 900),
    ):
        import fastplotlib as fpl

        if volume.ndim != 3:
            raise ValueError(f"volume must be ZYX, got shape {volume.shape}")

        self._volume = np.asarray(volume, dtype=np.float32)
        self._voxel_size = dict(zip("zyx", (float(v) for v in voxel_size_zyx)))
        self._position = [n // 2 for n in self._volume.shape]
        if vmin is None or vmax is None:
            low, high = np.nanpercentile(self._volume[:, ::4, ::4], (1, 99.5))
            vmin = low if vmin is None else vmin
            vmax = high if vmax is None else vmax

        self.figure = fpl.Figure(
            shape=(2, 2),
            names=[["xy", "yz"], ["xz", "3d"]],
            cameras=[["2d", "2d"], ["2d", "3d"]],
            controller_types=[["panzoom", "panzoom"], ["panzoom", "orbit"]],
            size=size,
        )

        self._images = {}
        self._selectors = {}
        for view in _VIEWS:
            subplot = self.figure[view]
            col_axis, row_axis = _VIEW_AXES[view]
            image = subplot.add_image(
                self._slice(view), cmap=cmap, vmin=vmin, vmax=vmax, name=view
            )
            image.world_object.local.scale_x = self._voxel_size[col_axis]
            image.world_object.local.scale_y = self._voxel_size[row_axis]
            image.add_event_handler(partial(self._on_click, view), "click")
            self._images[view] = image

            for selector_axis, volume_axis in (("x", col_axis), ("y", row_axis)):
                extent = self._volume.shape[_AXIS_INDEX[volume_axis]]
                scale = self._voxel_size[volume_axis]
                selector = fpl.LinearSelector(
                    selection=self._position[_AXIS_INDEX[volume_axis]] * scale,
                    limits=(0, (extent - 1) * scale),
                    axis=selector_axis,
                    edge_color=_AXIS_COLORS[volume_axis],
                )
                subplot.add_graphic(selector)
                selector.add_event_handler(
                    partial(self._on_selection, volume_axis), "selection"
                )
                self._selectors[(view, volume_axis)] = selector

        subplot_3d = self.figure["3d"]
        self._planes = {}
        data = None
        for axis in "zyx":
            plane = subplot_3d.add_image_volume(
                self._volume if data is None else data,
                mode="slice",
                plane=self._plane_equation(axis),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                name=f"plane_{axis}",
            )
            plane.world_object.local.scale = (
                self._voxel_size["x"],
                self._voxel_size["y"],
                self._voxel_size["z"],
            )
            # the three planes share one texture on the gpu
            data = plane.data
            self._planes[axis] = plane

        self._outlines = {
            axis: subplot_3d.add_line(
                self._outline(axis),
                colors=_AXIS_COLORS[axis],
                thickness=2,
                name=f"outline_{axis}",
            )
            for axis in "zyx"
        }

        for view in _VIEWS:
            self.figure[view].auto_scale()

    @property
    def position(self) -> tuple[int, int, int]:
        """Return the (z, y, x) voxel index of the crosshair."""
        return tuple(self._position)

    @position.setter
    def position(self, zyx: tuple[int, int, int]):
        clipped = [
            int(np.clip(v, 0, n - 1)) for v, n in zip(zyx, self._volume.shape)
        ]
        if clipped == self._position:
            return
        self._position = clipped

        for view, image in self._images.items():
            image.data = self._slice(view)
        for (view, axis), selector in self._selectors.items():
            target = self._position[_AXIS_INDEX[axis]] * self._voxel_size[axis]
            if selector.selection != target:
                selector.selection = target
        for axis, plane in self._planes.items():
            plane.plane = self._plane_equation(axis)
            self._outlines[axis].data = self._outline(axis)

    def show(self):
        """Show the figure."""
        return self.figure.show()

    def _slice(self, view: str) -> np.ndarray:
        z, y, x = self._position
        if view == "xy":
            return self._volume[z]
        if view == "yz":
            return np.ascontiguousarray(self._volume[:, :, x].T)  ## (Y, Z)
        return self._volume[:, y, :]  ## (Z, X)

    def _plane_equation(self, axis: str) -> tuple[float, float, float, float]:
        """Return the slice plane ``(a, b, c, d)`` at the crosshair, in world coordinates."""
        normal = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
        offset = self._position[_AXIS_INDEX[axis]] * self._voxel_size[axis]
        return (*normal, -float(offset))

    def _outline(self, axis: str) -> np.ndarray:
        """Return the closed outline of the slice plane on ``axis``, in world coordinates. Shape (5, 3)."""
        extent = dict(zip("zyx", self._volume.shape))
        a, b = (other for other in "xyz" if other != axis)
        corners = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        points = []
        for corner_a, corner_b in corners:
            voxel = {
                axis: self._position[_AXIS_INDEX[axis]],
                a: corner_a * extent[a] - 0.5,
                b: corner_b * extent[b] - 0.5,
            }
            points.append([voxel[k] * self._voxel_size[k] for k in "xyz"])
        return np.array(points, dtype=np.float32)

    def _on_selection(self, axis: str, ev):
        zyx = list(self._position)
        zyx[_AXIS_INDEX[axis]] = round(ev.info["value"] / self._voxel_size[axis])
        self.position = zyx

    def _on_click(self, view: str, ev):
        col, row = ev.pick_info["index"]
        col_axis, row_axis = _VIEW_AXES[view]
        zyx = list(self._position)
        zyx[_AXIS_INDEX[col_axis]] = col
        zyx[_AXIS_INDEX[row_axis]] = row
        self.position = zyx

