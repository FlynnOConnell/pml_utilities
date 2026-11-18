"""Annotate dendrites, spines and other structures in a Z-stack, in 3D.

A standalone fastplotlib window over pml_utilities: the stack is a lazy array
in ``MboNDViewer``, the annotation is ``ManualRoiWidget`` (polygon, labels,
table, undo, the rois zarr). Added here: a brush, filling the planes between
two ROIs, XZ and YZ views that take the brush too, and a 3D view of the stack with each class as a
surface. XZ, YZ and the 3D view are images of the same stack in the viewer's ``NDWidget``, each with
its own display dims, so the sliders move all four. A trace of one channel (``analysis/neurite_tracing.py``) is drawn
over XY and 3D as paths and link proposals to pick, delete, accept or reject (``stack_trace_layer.py``),
and is saved beside the rois zarr as ``<stem>.trace.zarr``.

    mbo stack-annotate <stack.mesc or any file imread opens>
        [--unit MSession_0/MUnit_1] [--voxel dz dy dx]
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Optional

import cv2
import fastplotlib as fpl
import numpy as np
import scipy.ndimage
import skimage.measure
from imgui_bundle import imgui

import mbo_utilities
from mbo_utilities.analysis import neurite_tracing as tracing
from mbo_utilities.annotation import UNLABELED
from mbo_utilities.annotation.neurite_trace import TRACE_SUFFIX, Trace, read_trace, write_trace
from mbo_utilities.annotation.store import class_color
from mbo_utilities.arrays.mesc import MescArray, list_mesc_units
from mbo_utilities.arrays.mesc_geometry import viewport_geometry, zstack_depth_info
from mbo_utilities.gui import stack_trace_layer as trace_layer
from mbo_utilities.gui._imgui_helpers import selected_button_style
from mbo_utilities.gui._ndviewer import MboNDViewer, _ref_to_index, sliders_height
from mbo_utilities.gui._theme import em
from mbo_utilities.gui.manual_roi import ManualRoiWidget
from mbo_utilities.roi_workflow import ROIS_SUFFIX

WINDOW_SIZE = (1700, 1000)
PANEL_WIDTH_PX = 460
COLOR_UNLABELED = (0.7, 0.7, 0.7)
COLOR_GUIDE = (1.0, 0.85, 0.2, 0.6)
ALPHA_ORTHO_LABELS = 150
ALPHA_STROKE = 140
SOURCE_INTERPOLATED = "interpolated"
CLICK_SLOP_PX = 4
VIEW_DIRECTION_3D = (0.5, 0.7, 0.6)
SUBPLOT_XY, SUBPLOT_YZ, SUBPLOT_XZ, SUBPLOT_3D = (0, 0), (0, 1), (1, 0), (1, 1)
SUBPLOTS_BRUSH = {"xy": SUBPLOT_XY, "xz": SUBPLOT_XZ, "yz": SUBPLOT_YZ}
DIM_Z, DIM_Y, DIM_X, DIM_RGBA = "z", "y", "x", "rgba"
DISPLAY_DIMS = {"xz": (DIM_Z, DIM_X), "yz": (DIM_Y, DIM_Z), "volume": (DIM_Z, DIM_Y, DIM_X)}


@dataclass
class Brush:
    """The brush tool: a stroke painted on one of the three views until the button lifts."""

    active: bool = False
    radius: int = 3
    painting: bool = False
    erasing: bool = False
    view: str = "xy"
    ## NOTE: the side views' strokes are in screen-square pixels, so their depth axis is n_z * z_scale long
    strokes: dict = field(default_factory=dict)  ## view -> uint8: xy (Y, X), xz (depth, X), yz (Y, depth)
    previews: dict = field(default_factory=dict)
    last: Optional[tuple] = None


@dataclass
class ClassDisplay:
    """How one class is shown: its surface's visibility and opacity in 3D, and its colour in every view."""

    visible: bool = True
    alpha: float = 1.0
    rgb: Optional[tuple] = None  ## 0 to 1; None keeps the ROI tool's class colour


@dataclass
class StackAnnotator:
    """The window's state: the stack, the ROI widget and the graphics added around it."""

    filepath_stack: str
    arr: Any  ## 5D lazy array (T, C, Z, Y, X)
    scale_um: tuple  ## (dz, dy, dx)
    viewer: Any = None
    widget: Any = None
    brush: Brush = field(default_factory=Brush)
    channel: int = -1
    nd_images: dict = field(default_factory=dict)  ## "xz", "yz", "volume", "labels_xz", "labels_yz" -> NDImage
    graphic_plane: Any = None
    graphics_mesh: dict = field(default_factory=dict)  ## class volume value -> surface
    class_display: dict = field(default_factory=dict)  ## class volume value -> ClassDisplay
    tint_applied: dict = field(default_factory=dict)  ## ROI uid -> RGB bytes last given to the store
    guides: dict = field(default_factory=dict)
    class_volume: Optional[np.ndarray] = None  ## (Z, Y, X) uint8: 0 none, 1 unlabeled, 2 + class
    labels_rgba: Optional[np.ndarray] = None  ## (Z, Y, X, 4) uint8: the class volume in colour, sliced by XZ and YZ
    stack: Optional[np.ndarray] = None  ## (C, Z, Y, X) float32, the first time point in memory
    limits: list = field(default_factory=list)  ## per channel, (vmin, vmax) of XZ, YZ and the 3D view
    trace: Optional[trace_layer.TraceLayer] = None
    trace_job: Optional[Future] = None
    executor: ThreadPoolExecutor = field(default_factory=partial(ThreadPoolExecutor, max_workers=1))
    edit_order: list = field(default_factory=list)  ## "rois", "trace" or "retrace" per undo step, oldest first
    traces_replaced: list = field(default_factory=list)  ## (trace, its undo steps) per "retrace", oldest first
    n_objects: dict = field(default_factory=dict)
    classes_drawn: tuple = ()  ## each ROI's class when the surfaces were last built
    show_volume: bool = True
    alpha_volume: float = 1.0
    show_surfaces: bool = True
    dirty_labels: bool = True
    press: Optional[tuple] = None

    @property
    def z_scale(self) -> float:
        return self.scale_um[0] / self.scale_um[2]

    @property
    def crosshair(self) -> tuple:
        """``(y, x)`` the side views cut through: the NDWidget's y and x indices."""
        indices = self.viewer.ndwidget.indices
        return _ref_to_index(indices[DIM_Y]), _ref_to_index(indices[DIM_X])


def open_stack(filepath_stack: str, unit: Optional[str], voxel_um: Optional[tuple]) -> tuple:
    """
    Open a stack as a 5D lazy array with its voxel size.

    Parameters
    ----------
    filepath_stack : str
        A ``.mesc`` file, or any file ``mbo_utilities.imread`` opens.
    unit : str, optional
        The MESc unit key. ``None`` takes the file's first Z-stack unit.
    voxel_um : tuple, optional
        ``(dz, dy, dx)`` in microns. ``None`` reads it from a ``.mesc`` and is 1 µm otherwise.

    Returns
    -------
    tuple
        ``(arr, (dz, dy, dx))``.
    """
    if Path(filepath_stack).suffix.lower() != ".mesc":
        arr = mbo_utilities.imread(filepath_stack)
        return arr, tuple(voxel_um) if voxel_um is not None else (1.0, 1.0, 1.0)
    if unit is None:
        units_stack = [u["key"] for u in list_mesc_units(filepath_stack) if u["kind"] == "zstack"]
        if not units_stack:
            raise ValueError(f"{filepath_stack} holds no Z-stack unit; name one with --unit")
        unit = units_stack[0]
    arr = MescArray(filepath_stack, unit=unit)
    if voxel_um is not None:
        return arr, tuple(voxel_um)
    viewport = viewport_geometry(filepath_stack, unit)
    depth = zstack_depth_info(filepath_stack, unit)
    if viewport is None or depth is None:
        raise ValueError(f"{unit} stores no viewport or depth; give --voxel dz dy dx")
    n_z, n_y, n_x = arr.shape[-3:]
    dz = (depth["max_z"] - depth["min_z"]) / max(n_z - 1, 1)
    return arr, (float(dz), viewport["height"] / n_y, viewport["width"] / n_x)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Distance to the mask's edge in pixels, negative inside."""
    return scipy.ndimage.distance_transform_edt(~mask) - scipy.ndimage.distance_transform_edt(mask)


def masks_between(mask_first: np.ndarray, mask_last: np.ndarray, n_between: int) -> list:
    """
    Shapes for the planes between two masks.

    Each shape blends the two signed distances after both are moved onto the
    centroid interpolated for that plane, so masks that do not overlap (a
    dendrite crossing planes at an angle) still join.

    Parameters
    ----------
    mask_first : np.ndarray
        Boolean mask on the first plane. Shape (Y, X).
    mask_last : np.ndarray
        Boolean mask on the last plane. Shape (Y, X).
    n_between : int
        Number of planes between the two.

    Returns
    -------
    list
        ``n_between`` boolean masks, first to last.
    """
    distances = [signed_distance(mask=mask) for mask in (mask_first, mask_last)]
    centroids = [np.array(scipy.ndimage.center_of_mass(mask)) for mask in (mask_first, mask_last)]
    masks = []
    for index in range(1, n_between + 1):
        fraction = index / (n_between + 1)
        centroid = (1 - fraction) * centroids[0] + fraction * centroids[1]
        moved = [
            scipy.ndimage.shift(distance, centroid - origin, order=1, mode="nearest")
            for distance, origin in zip(distances, centroids)
        ]
        masks.append(((1 - fraction) * moved[0] + fraction * moved[1]) < 0)
    return masks


def class_volume_of(store, channel_pos: dict, name_z: Optional[str], n_z: int) -> np.ndarray:
    """
    Every ROI of one channel as a class volume.

    Parameters
    ----------
    store : RoiLabelStore
        The annotation.
    channel_pos : dict
        Slider positions of the plane dims other than z.
    name_z : str, optional
        The z slider's name. ``None`` for a single-plane stack.
    n_z : int
        Number of z-planes.

    Returns
    -------
    np.ndarray
        Shape (Z, Y, X) uint8: 0 nothing, 1 unlabeled, ``2 + class_index`` otherwise.
    """
    volume = np.zeros((n_z, store.ny * store.nx), np.uint8)
    for index, roi in enumerate(store.rois):
        pos = store.roi_pos(index)
        if any(pos.get(name, 0) != value for name, value in channel_pos.items()):
            continue
        z = pos.get(name_z, 0) if name_z is not None else 0
        volume[z, roi.pixels] = 1 if roi.class_index == UNLABELED else roi.class_index + 2
    return volume.reshape(n_z, store.ny, store.nx)


def class_rgb(value: int) -> tuple:
    """Colour of a class volume value."""
    return COLOR_UNLABELED if value == 1 else class_color(value - 2)


def class_lut(rgbs: list) -> np.ndarray:
    """RGBA bytes per class volume value, value 0 transparent; ``rgbs[value - 1]`` is 0 to 1. Shape (len(rgbs) + 1, 4)."""
    lut = np.zeros((len(rgbs) + 1, 4), np.uint8)
    lut[1:, :3] = np.round(np.array(rgbs) * 255)
    lut[1:, 3] = ALPHA_ORTHO_LABELS
    return lut


def surface_of(mask: np.ndarray, z_scale: float) -> tuple:
    """
    Surface of a boolean volume.

    Parameters
    ----------
    mask : np.ndarray
        Shape (Z, Y, X).
    z_scale : float
        Slice spacing in pixels.

    Returns
    -------
    tuple
        ``(positions, faces)``: (n, 3) float32 ``(x, y, z)`` in pixels and (m, 3) int32.
    """
    z, y, x = np.nonzero(mask)
    lows = np.array([z.min(), y.min(), x.min()])
    highs = np.array([z.max(), y.max(), x.max()]) + 1
    crop = mask[lows[0] : highs[0], lows[1] : highs[1], lows[2] : highs[2]]
    vertices, faces, _normals, _values = skimage.measure.marching_cubes(
        np.pad(crop, 1).astype(np.float32), level=0.5
    )
    vertices = (vertices - 1 + lows) * np.array([z_scale, 1.0, 1.0])
    ## NOTE: reversing the axis order mirrors the surface, so the faces turn the other way
    return vertices[:, ::-1].astype(np.float32), faces[:, ::-1].astype(np.int32)


def view_position(state: StackAnnotator) -> tuple:
    """``(z, channel, positions of the plane dims other than z)`` on screen."""
    widget = state.widget
    pos = widget._view_pos()
    z = pos.get(widget.zdim, 0) if widget.zdim is not None else 0
    channel = pos.get(widget.cdim, 0) if widget.cdim is not None else 0
    return z, channel, {name: value for name, value in pos.items() if name != widget.zdim}


def hide_from_pointer(graphic) -> None:
    """Take an overlay out of picking, so tooltips and clicks reach the image under it."""
    ## NOTE: fastplotlib has no public switch for this; the material is pygfx's
    world_object = graphic.world_object
    for part in world_object.children or (world_object,):
        part.material.pick_write = False


def guide_line(subplot, points: np.ndarray):
    """A thin guide line that the pointer passes through."""
    graphic = subplot.add_line(np.asarray(points, np.float32), thickness=1.0, colors=COLOR_GUIDE)
    hide_from_pointer(graphic=graphic)
    return graphic


def move_crosshair(state: StackAnnotator, y: int, x: int, z: Optional[int] = None) -> None:
    """Move the side views to cut through ``(y, x)``, and the slice to ``z`` when given."""
    indices = {DIM_Y: y + 1, DIM_X: x + 1}
    if z is not None:
        indices[DIM_Z] = z + 1
    state.viewer.ndwidget.indices.set(indices)


def move_guides(state: StackAnnotator, indices: dict) -> None:
    """Put the guide lines of XZ, YZ and the 3D view on the NDWidget's indices."""
    z, y, x = (_ref_to_index(indices[dim]) for dim in (DIM_Z, DIM_Y, DIM_X))
    depth = (z + 0.5) * state.z_scale
    state.guides["xz_z"].data[:, 1] = depth
    state.guides["xz_x"].data[:, 0] = x
    state.guides["yz_z"].data[:, 0] = depth
    state.guides["yz_y"].data[:, 1] = y
    state.graphic_plane.data[:, 2] = z * state.z_scale


def build(filepath_stack: str, unit: Optional[str] = None, voxel_um: Optional[tuple] = None) -> StackAnnotator:
    """
    Open the window on a stack.

    Parameters
    ----------
    filepath_stack : str
        The stack file.
    unit : str, optional
        MESc unit key.
    voxel_um : tuple, optional
        ``(dz, dy, dx)`` in microns, overriding the file's.

    Returns
    -------
    StackAnnotator
        The state, with the figure shown.
    """
    arr, scale_um = open_stack(filepath_stack=filepath_stack, unit=unit, voxel_um=voxel_um)
    state = StackAnnotator(filepath_stack=str(filepath_stack), arr=arr, scale_um=scale_um)
    n_channels, n_z, n_y, n_x = arr.shape[1:]
    if n_z < 2:
        raise ValueError(f"{filepath_stack} has one z-plane; the side views and the 3D view need a stack")

    state.viewer = MboNDViewer(
        arr.squeeze(),
        slider_dim_names=[name for name, size in zip("tcz", arr.shape[:3]) if size > 1],
        figure_shape=(2, 2),
        cmap="gray",
        figure_kwargs={
            "size": WINDOW_SIZE,
            "cameras": ["2d", "2d", "2d", "3d"],
            "controller_types": ["panzoom", "panzoom", "panzoom", "orbit"],
            "controller_ids": None,
        },
    )
    figure = state.viewer.figure
    state.viewer.show()
    state.widget = StackRoiWidget(state.viewer, fpath=filepath_stack, auto_trace=False)
    state.widget.annotator = state
    ## NOTE: a stack has no time axis, so the Traces panel the widget registers would sit empty
    state.widget.strip.unregister("traces")

    z_scale = state.z_scale
    ndw = state.viewer.ndwidget
    for subplot, title in zip((SUBPLOT_XY, SUBPLOT_XZ, SUBPLOT_YZ, SUBPLOT_3D), ("XY", "XZ", "YZ", "3D")):
        figure[subplot].title = title
    for subplot in (SUBPLOT_XZ, SUBPLOT_YZ):
        figure[subplot].camera.local.scale_y = figure[SUBPLOT_XY].camera.local.scale_y

    ## NOTE: the viewer's sliders count from 1, so y and x do too and every view maps them with _ref_to_index
    ndw.indices.push_dims({DIM_Y: (1, n_y + 1, 1), DIM_X: (1, n_x + 1, 1)})
    figure.imgui_windows["bottom"].size = sliders_height(len(ndw.indices))
    state.stack = np.asarray(arr[0], np.float32)  ## (C, Z, Y, X)
    stack, dims_stack = state.stack, ("c", DIM_Z, DIM_Y, DIM_X)
    if n_channels == 1:
        stack, dims_stack = stack[0], dims_stack[1:]
    state.limits = [(float(np.percentile(volume, 1)), float(np.percentile(volume, 99.9))) for volume in state.stack]
    limits = state.limits[0]
    state.labels_rgba = np.zeros((n_z, n_y, n_x, 4), np.uint8)
    dims_labels = (DIM_Z, DIM_Y, DIM_X, DIM_RGBA)
    for view, subplot in (("xz", SUBPLOT_XZ), ("yz", SUBPLOT_YZ), ("volume", SUBPLOT_3D)):
        state.nd_images[view] = add_view(
            nd_subplot=ndw[subplot],
            data=stack,
            dims=dims_stack,
            display_dims=DISPLAY_DIMS[view],
            name=view,
            graphic_kwargs={"cmap": "gray", "vmin": limits[0], "vmax": limits[1]},
        )
    for view, subplot in (("xz", SUBPLOT_XZ), ("yz", SUBPLOT_YZ)):
        state.nd_images[f"labels_{view}"] = add_view(
            nd_subplot=ndw[subplot],
            data=state.labels_rgba,
            dims=dims_labels,
            display_dims=(*DISPLAY_DIMS[view], DIM_RGBA),
            name=f"labels_{view}",
            graphic_kwargs={"vmin": 0, "vmax": 255, "alpha_mode": "blend"},
        )
        hide_from_pointer(graphic=state.nd_images[f"labels_{view}"].graphic)
    for view, scale in (("xz", (1, z_scale, 1)), ("yz", (z_scale, 1, 1)), ("volume", (1, 1, z_scale))):
        state.nd_images[view].graphic.scale = scale
        if f"labels_{view}" in state.nd_images:
            state.nd_images[f"labels_{view}"].graphic.scale = scale

    depth = n_z * z_scale
    subplot_xz, subplot_yz, subplot_3d = figure[SUBPLOT_XZ], figure[SUBPLOT_YZ], figure[SUBPLOT_3D]
    state.guides = {
        "xz_z": guide_line(subplot=subplot_xz, points=[[0, 0, 2], [n_x, 0, 2]]),
        "xz_x": guide_line(subplot=subplot_xz, points=[[0, 0, 2], [0, depth, 2]]),
        "yz_z": guide_line(subplot=subplot_yz, points=[[0, 0, 2], [0, n_y, 2]]),
        "yz_y": guide_line(subplot=subplot_yz, points=[[0, 0, 2], [depth, 0, 2]]),
    }
    state.graphic_plane = guide_line(
        subplot=subplot_3d, points=[[0, 0, 0], [n_x, 0, 0], [n_x, n_y, 0], [0, n_y, 0], [0, 0, 0]]
    )
    n_depth = int(np.ceil(depth))
    for view, shape in (("xy", (n_y, n_x)), ("xz", (n_depth, n_x)), ("yz", (n_y, n_depth))):
        state.brush.strokes[view] = np.zeros(shape, np.uint8)
        preview = figure[SUBPLOTS_BRUSH[view]].add_image(
            np.zeros((*shape, 4), np.uint8), vmin=0, vmax=255, alpha_mode="blend", offset=(0, 0, 3)
        )
        hide_from_pointer(graphic=preview)
        state.brush.previews[view] = preview

    state.widget.model.add_event_handler(partial(mark_labels_dirty, state), "rois")
    state.widget.strip.add_hook(partial(draw_frame, state))
    ndw.indices.add_event_handler(partial(move_guides, state))
    ndw.indices.add_event_handler(partial(show_trace_plane, state))
    figure.renderer.add_event_handler(partial(on_pointer, state), "pointer_down", "pointer_move", "pointer_up")
    ## NOTE: the next two handlers work around wgpu's imgui backend and go when it is fixed upstream
    figure.canvas.add_event_handler(on_key_ctrl_letter, "key_down", "key_up")
    ## NOTE: order -100 runs before imgui's own pointer handler (-99), so the click it reports sees these modifiers
    figure.canvas.add_event_handler(sync_imgui_modifiers, "pointer_down", "pointer_up", "pointer_move", order=-100)
    figure.add_imgui_window(
        partial(draw_panel, state=state),
        location="right",
        size=PANEL_WIDTH_PX,
        title="Annotation",
        window_flags=imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_title_bar,
    )
    move_crosshair(state=state, y=n_y // 2, x=n_x // 2)
    if Path(filepath_trace(state=state)).exists():
        show_trace(state=state, trace=read_trace(filepath_trace(state=state)))
    for subplot in (subplot_xz, subplot_yz):
        subplot.auto_scale()
    subplot_3d.camera.show_object(subplot_3d.scene, view_dir=VIEW_DIRECTION_3D, up=(0, 0, -1))
    return state


def filepath_trace(state: StackAnnotator) -> str:
    """The trace zarr beside the rois zarr, named like it: ``<stem>.trace.zarr``, ``<stem>_<unit>.trace.zarr``."""
    return str(state.widget._save_target()).removesuffix(ROIS_SUFFIX) + TRACE_SUFFIX


def save_trace(state: StackAnnotator) -> None:
    """Write the trace on screen to its trace zarr."""
    filepath = filepath_trace(state=state)
    write_trace(state.trace.trace, filepath, shape=state.stack.shape[1:], source=state.widget._source_block())
    state.trace.unsaved = False
    state.widget.status = f"saved {Path(filepath).name}"


def trace_clicks_allowed(state: StackAnnotator) -> bool:
    """A click picks trace lines unless the brush or the polygon tool has it, or it lands on the panel."""
    return not state.brush.active and not state.widget._drawing() and not over_imgui()


def show_trace(state: StackAnnotator, trace: Trace) -> None:
    """Replace the trace on screen."""
    if state.trace is not None:
        trace_layer.remove_layer(layer=state.trace)
    figure = state.viewer.figure
    z, _channel, _pos = view_position(state=state)
    state.trace = trace_layer.build_layer(
        trace=trace,
        subplots={"xy": figure[SUBPLOT_XY], "3d": figure[SUBPLOT_3D]},
        z_scale=state.z_scale,
        z=z,
        clicks_allowed=partial(trace_clicks_allowed, state),
    )


def start_trace(state: StackAnnotator) -> None:
    """Trace the channel on screen in the background; ``draw_frame`` shows the result."""
    _z, channel, _pos = view_position(state=state)
    state.trace_job = state.executor.submit(
        tracing.trace_volume, volume=state.stack[channel], voxel_um=state.scale_um, channel=channel
    )
    state.widget.status = f"tracing channel {channel + 1} ..."


def show_trace_plane(state: StackAnnotator, indices: dict) -> None:
    if state.trace is not None:
        trace_layer.show_plane(layer=state.trace, z=_ref_to_index(indices[DIM_Z]))


def edit_trace(state: StackAnnotator, action: str) -> None:
    """Delete the picked paths, or accept or reject the picked links, as one undo step."""
    layer = state.trace
    if action == "delete":
        n_done = trace_layer.delete_picked_paths(layer=layer)
    else:
        n_done = trace_layer.resolve_picked_links(layer=layer, accept=action == "accept")
    if n_done:
        state.edit_order.append("trace")
    noun = "path(s)" if action == "delete" else "link(s)"
    past = {"delete": "deleted", "accept": "accepted", "reject": "rejected"}[action]
    state.widget.status = f"{past} {n_done} {noun}" if n_done else f"pick {noun} first (click, shift + click)"


def undo_trace(state: StackAnnotator, kind: str) -> None:
    """Undo one trace step: an edit, or a new trace, which brings the one it replaced back with its own undo steps."""
    if kind == "trace":
        trace_layer.undo(layer=state.trace)
        state.widget.status = "trace edit undone"
        return
    trace, undo_steps = state.traces_replaced.pop()
    show_trace(state=state, trace=trace)
    state.trace.undo_steps = undo_steps
    state.trace.unsaved = True
    state.widget.status = "previous trace back"


def add_view(nd_subplot, data: np.ndarray, dims: tuple, display_dims: tuple, name: str, graphic_kwargs: dict):
    """
    One more image of the stack in the viewer's NDWidget.

    Parameters
    ----------
    nd_subplot : NDWSubplot
        The NDWidget subplot that shows it.
    data : np.ndarray
        The array, with one axis per name in ``dims``.
    dims : tuple
        Name of every axis. Axes left out of ``display_dims`` follow the slider of that name.
    display_dims : tuple
        The axes drawn, in the order (rows, columns), (planes, rows, columns), or with ``DIM_RGBA`` last.
    name : str
        The graphic's name.
    graphic_kwargs : dict
        Passed to the image.

    Returns
    -------
    NDImage
        The NDWidget's graphic.
    """
    return nd_subplot.add_nd_image(
        data,
        dims,
        display_dims,
        rgb_dim=DIM_RGBA if DIM_RGBA in dims else None,
        compute_histogram=False,
        slider_maps={dim: _ref_to_index for dim in dims if dim not in display_dims},
        name=name,
        graphic_kwargs=graphic_kwargs,
    )


def mark_labels_dirty(state: StackAnnotator, event) -> None:
    ## NOTE: apply_class_display sets the tint from inside the refresh, and a tint moves no pixels
    if event.info.get("action") != "tint":
        state.dirty_labels = True


def colour_side_labels(state: StackAnnotator) -> None:
    """Give the labels drawn over XZ and YZ the class volume in each class's colour."""
    n_values = len(state.widget.store.label_names) + 1
    lut = class_lut(rgbs=[display_rgb(state=state, value=value) for value in range(1, n_values + 1)])
    ## NOTE: written in place and fetched again, because giving an NDImage new data rebuilds its graphic and resets the camera
    state.labels_rgba[:] = lut[state.class_volume]
    state.viewer.ndwidget.indices.set({})


def refresh_labels(state: StackAnnotator) -> None:
    """Rebuild the class volume, its surfaces in the 3D view and the object counts."""
    widget = state.widget
    _z, _channel, channel_pos = view_position(state=state)
    n_z = state.arr.shape[-3]
    state.class_volume = class_volume_of(
        store=widget.store, channel_pos=channel_pos, name_z=widget.zdim, n_z=n_z
    )
    subplot_3d = widget.figure[SUBPLOT_3D]
    for graphic in state.graphics_mesh.values():
        subplot_3d.delete_graphic(graphic)
    state.graphics_mesh = {}
    state.n_objects = {}
    for value in np.unique(state.class_volume)[1:]:
        mask = state.class_volume == value
        state.n_objects[int(value)] = int(scipy.ndimage.label(mask, structure=np.ones((3, 3, 3)))[1])
        positions, faces = surface_of(mask=mask, z_scale=state.z_scale)
        graphic = subplot_3d.add_mesh(
            positions, faces, colors=display_rgb(state=state, value=int(value)), mode="basic"
        )
        ## NOTE: the stack's max projection fills the depth buffer, so a surface inside it only shows with the depth test off
        graphic.world_object.material.depth_test = False
        graphic.world_object.render_order = 1
        state.graphics_mesh[int(value)] = graphic
    apply_class_display(state=state)


def display_rgb(state: StackAnnotator, value: int) -> tuple:
    """Colour of a class volume value: the one picked in the panel, else the ROI tool's."""
    display = state.class_display.get(value)
    return class_rgb(value) if display is None or display.rgb is None else display.rgb


def apply_class_display(state: StackAnnotator) -> None:
    """Give the surfaces, the side views and the XY overlay each class's visibility, opacity and colour from the panel."""
    for value, graphic in state.graphics_mesh.items():
        display = state.class_display.get(value, ClassDisplay())
        graphic.visible = state.show_surfaces and display.visible
        graphic.alpha = display.alpha
        graphic.colors = display_rgb(state=state, value=value)
    store = state.widget.store
    tint = {}
    for roi in store.rois:
        display = state.class_display.get(1 if roi.class_index == UNLABELED else roi.class_index + 2)
        if display is not None and display.rgb is not None:
            tint[roi.uid] = tuple(int(round(channel * 255)) for channel in display.rgb)
    ## NOTE: the store's tint is the ROI panel's colour-by too, so it is set only when ours changes
    if tint != state.tint_applied:
        store.set_tint(tint)
        state.tint_applied = tint
    colour_side_labels(state=state)


def apply_channel_limits(state: StackAnnotator, channel: int) -> None:
    """Give XZ, YZ and the 3D view the contrast of ``channel``."""
    vmin, vmax = state.limits[channel]
    for view in ("xz", "yz", "volume"):
        graphic = state.nd_images[view].graphic
        graphic.vmin, graphic.vmax = vmin, vmax


def draw_frame(state: StackAnnotator) -> None:
    """Per-frame work: the brush key, and whatever the last events left stale."""
    io = imgui.get_io()
    if not io.want_text_input and state.widget._keys_live() and imgui.is_key_pressed(imgui.Key.b, False):
        state.brush.active = not state.brush.active
    _z, channel, _pos = view_position(state=state)
    if channel != state.channel:
        state.channel = channel
        state.dirty_labels = True
        apply_channel_limits(state=state, channel=channel)
    ## NOTE: labelling a group blocks the store's events, so that change is found by comparing the classes
    classes = tuple(roi.class_index for roi in state.widget.store.rois)
    if classes != state.classes_drawn:
        state.classes_drawn = classes
        state.dirty_labels = True
    if state.dirty_labels:
        refresh_labels(state=state)
    state.dirty_labels = False
    if state.trace_job is not None and state.trace_job.done():
        trace = state.trace_job.result()
        state.trace_job = None
        if state.trace is not None:
            state.traces_replaced.append((state.trace.trace, state.trace.undo_steps))
            state.edit_order.append("retrace")
        show_trace(state=state, trace=trace)
        state.trace.unsaved = True
        state.widget.status = f"traced: {len(trace.paths)} paths, {len(trace.links)} links"


def stamp(state: StackAnnotator, point: tuple) -> None:
    """Extend the stroke on the brush's view to ``point`` (column, row) and show it."""
    brush = state.brush
    stroke = brush.strokes[brush.view]
    start = brush.last if brush.last is not None else point
    cv2.line(stroke, start, point, 1, thickness=2 * brush.radius + 1)
    brush.last = point
    colour = (255, 80, 80) if brush.erasing else (255, 255, 255)
    preview = np.zeros((*stroke.shape, 4), np.uint8)
    preview[stroke > 0] = (*colour, ALPHA_STROKE)
    brush.previews[brush.view].data = preview


def clear_stroke(brush: Brush) -> None:
    stroke = brush.strokes[brush.view]
    stroke[:] = 0
    brush.previews[brush.view].data = np.zeros((*stroke.shape, 4), np.uint8)


def replace_pixels(widget, index: int, mask: np.ndarray) -> Optional[int]:
    """
    Give ROI ``index`` a new mask, keeping its class and note.

    Returns the ROI's new index, or ``None`` when the mask is empty and the ROI is gone.
    """
    ## NOTE: RoiLabelStore cannot edit an ROI's pixels, so the ROI is deleted and added again and its uid changes
    store = widget.store
    roi = store.rois[index]
    plane, class_index, note = roi.plane, roi.class_index, roi.note
    store.delete_roi(index)
    index_new = store.add_roi(plane, mask) if mask.any() else None
    if index_new is None:
        return None
    store.set_class(index_new, class_index)
    store.set_note(index_new, note)
    return index_new


def commit_stroke(state: StackAnnotator) -> None:
    """
    Turn a finished stroke on XY into annotation.

    Painting joins the selected ROI when it is on the plane on screen, else
    starts a new one. Erasing takes the stroke out of the selected ROI.
    """
    widget, brush = state.widget, state.brush
    stroke = brush.strokes["xy"].astype(bool)
    store = widget.store
    selected = widget.selected
    on_plane = 0 <= selected < len(store.rois) and store.rois[selected].plane == widget.z
    if brush.erasing and not on_plane:
        widget.status = "select an ROI on this plane to erase from"
    elif brush.erasing or on_plane:
        mask = store.mask(selected)
        mask = mask & ~stroke if brush.erasing else mask | stroke
        widget._push_undo()
        widget.select_roi(replace_pixels(widget=widget, index=selected, mask=mask))
    else:
        widget._push_undo()
        widget.select_roi(store.add_roi(widget.z, stroke))
    clear_stroke(brush=brush)


def masks_of_side_stroke(
    stroke: np.ndarray, view: str, crosshair: tuple, radius: int, shape_stack: tuple, z_scale: float
) -> dict:
    """
    A stroke painted on XZ or YZ as one mask per z-plane it crosses.

    A closed loop is filled. Along the axis the view hides, the stroke is the
    brush diameter thick and centred on the crosshair.

    Parameters
    ----------
    stroke : np.ndarray
        The painted pixels. Shape (depth, X) for ``"xz"``, (Y, depth) for ``"yz"``, depth in pixels of ``z_scale`` per plane.
    view : str
        ``"xz"`` or ``"yz"``.
    crosshair : tuple
        ``(y, x)`` the view cuts through.
    radius : int
        Brush radius in pixels.
    shape_stack : tuple
        ``(n_z, n_y, n_x)``.
    z_scale : float
        Slice spacing in pixels.

    Returns
    -------
    dict
        z -> boolean mask of shape (Y, X), for the planes the stroke crosses.
    """
    n_z, n_y, n_x = shape_stack
    filled = scipy.ndimage.binary_fill_holes(stroke)
    if view == "yz":
        filled = filled.T
    plane_of_row = np.minimum((np.arange(filled.shape[0]) / z_scale).astype(int), n_z - 1)
    n_rows_painted = np.zeros((n_z, filled.shape[1]), np.int64)  ## (Z, X) for xz, (Z, Y) for yz
    np.add.at(n_rows_painted, plane_of_row, filled)
    ## NOTE: a plane counts where the stroke covers half of it, or half a brush when the brush is thinner than a plane,
    ## so the round end of a stroke that grazes the next plane leaves no sliver there
    n_rows_needed = np.minimum(np.ceil(np.bincount(plane_of_row, minlength=n_z) / 2), radius + 1)
    lines = n_rows_painted >= n_rows_needed[:, None]
    center = crosshair[0] if view == "xz" else crosshair[1]
    slab = slice(max(center - radius, 0), center + radius + 1)
    masks = {}
    for z in np.flatnonzero(lines.any(axis=1)):
        mask = np.zeros((n_y, n_x), bool)
        if view == "xz":
            mask[slab, :] = lines[z][None, :]
        else:
            mask[:, slab] = lines[z][:, None]
        masks[int(z)] = mask
    return masks


def select_on_plane(state: StackAnnotator, index: Optional[int]) -> None:
    """Select ROI ``index`` and stay on the plane on screen, which ``select_roi`` leaves for the ROI's own."""
    widget = state.widget
    if widget.zdim is None:
        widget.select_roi(index)
        return
    z = int(state.viewer.indices[widget.zdim])
    widget.select_roi(index)
    if int(state.viewer.indices[widget.zdim]) != z:
        state.viewer.indices[widget.zdim] = z


def commit_stroke_side(state: StackAnnotator) -> None:
    """
    Turn a finished stroke on XZ or YZ into annotation on every plane it crosses.

    The stroke works in the selected ROI's class, unlabeled with none selected.
    On each plane, painting joins the first ROI of that class it touches or
    starts a new one, and erasing takes the stroke out of every ROI of that class.
    """
    widget, brush = state.widget, state.brush
    store = widget.store
    _z, _channel, channel_pos = view_position(state=state)
    masks = masks_of_side_stroke(
        stroke=brush.strokes[brush.view].astype(bool),
        view=brush.view,
        crosshair=state.crosshair,
        radius=brush.radius,
        shape_stack=state.arr.shape[-3:],
        z_scale=state.z_scale,
    )
    clear_stroke(brush=brush)
    is_selected = 0 <= widget.selected < len(store.rois)
    class_index = store.rois[widget.selected].class_index if is_selected else UNLABELED
    uid_select = store.rois[widget.selected].uid if is_selected else None
    uids_replaced = {}
    widget._push_undo()
    for z, mask in masks.items():
        plane = store.plane_of({**channel_pos, widget.zdim: z}) if widget.zdim is not None else widget.z
        uids_touched = [
            store.rois[index].uid
            for index in store.rois_on_plane(plane)
            if store.rois[index].class_index == class_index and mask.flat[store.rois[index].pixels].any()
        ]
        if not brush.erasing and not uids_touched:
            index_new = store.add_roi(plane, mask)
            if index_new is not None and class_index != UNLABELED:
                store.set_class(index_new, class_index)
        for uid in uids_touched if brush.erasing else uids_touched[:1]:
            index = store.uid_index(uid)
            mask_new = store.mask(index) & ~mask if brush.erasing else store.mask(index) | mask
            index_new = replace_pixels(widget=widget, index=index, mask=mask_new)
            uids_replaced[uid] = None if index_new is None else store.rois[index_new].uid
        if uid_select is None and plane == widget.z and index_new is not None:
            uid_select = store.rois[index_new].uid
    uid_select = uids_replaced.get(uid_select, uid_select)
    select_on_plane(state=state, index=None if uid_select is None else store.uid_index(uid_select))
    widget.status = f"{'erased on' if brush.erasing else 'painted'} {len(masks)} plane(s)"


def rois_picked(widget) -> list:
    """Indices of the drawn ROIs in the group, else of the one selected ROI."""
    indices = [k for si, k in widget.buffer if si < 0]
    if not indices and widget.selected >= 0:
        indices = [widget.selected]
    return indices


def delete_picked(state: StackAnnotator) -> None:
    """Delete every ROI in the group in one undo step."""
    widget = state.widget
    store = widget.store
    uids = [store.rois[index].uid for index in rois_picked(widget=widget)]
    if not uids:
        widget.status = "select the ROIs to delete (ctrl / shift + click)"
        return
    widget._push_undo()
    widget.buffer = []
    widget.select_roi(-1)
    for uid in uids:
        store.delete_roi(store.uid_index(uid))
    widget.status = f"deleted {len(uids)} ROI(s)"


def merge_picked(state: StackAnnotator) -> None:
    """
    Merge the ROIs in the group that share a plane into one ROI per plane, in one undo step.

    The merged ROI keeps the class and note of the first of them picked.
    """
    widget = state.widget
    store = widget.store
    uids_by_plane = {}
    for index in rois_picked(widget=widget):
        uids_by_plane.setdefault(store.rois[index].plane, []).append(store.rois[index].uid)
    groups = [uids for uids in uids_by_plane.values() if len(uids) > 1]
    if not groups:
        widget.status = "select two or more ROIs on one plane to merge (ctrl / shift + click)"
        return
    widget._push_undo()
    widget.buffer = []
    widget.select_roi(-1)
    for uids in groups:
        mask = np.any([store.mask(store.uid_index(uid)) for uid in uids], axis=0)
        for uid in uids[1:]:
            store.delete_roi(store.uid_index(uid))
        index_merged = replace_pixels(widget=widget, index=store.uid_index(uids[0]), mask=mask)
    select_on_plane(state=state, index=index_merged)
    widget.status = f"merged {sum(len(uids) for uids in groups)} ROIs into {len(groups)}"


def fill_between(state: StackAnnotator) -> None:
    """Add ROIs on the planes between the two ROIs in the group, in the first one's class."""
    widget = state.widget
    store = widget.store
    indices = sorted((k for si, k in widget.buffer if si < 0), key=lambda k: store.roi_z(k))
    if len(indices) != 2 or store.roi_c(indices[0]) != store.roi_c(indices[1]):
        widget.status = "pick two ROIs of one channel (ctrl + click) on different planes"
        return
    planes = [store.rois[k].plane for k in indices]
    n_between = planes[1] - planes[0] - 1
    if n_between < 1:
        widget.status = "the two ROIs have no plane between them"
        return
    class_index = store.rois[indices[0]].class_index
    masks = masks_between(
        mask_first=store.mask(indices[0]), mask_last=store.mask(indices[1]), n_between=n_between
    )
    widget._push_undo()
    for offset, mask in enumerate(masks, start=1):
        index_new = store.add_roi(planes[0] + offset, mask, source=SOURCE_INTERPOLATED)
        if index_new is not None and class_index != UNLABELED:
            store.set_class(index_new, class_index)
    widget.status = f"filled {n_between} plane(s)"


def on_key_ctrl_letter(event) -> None:
    """Give imgui the letter of a ctrl + letter press, so ctrl + z and ctrl + a reach the ROI tool."""
    ## NOTE: Qt names a letter held with ctrl by its control character (ctrl + z is "\x1a"), which wgpu's imgui backend drops
    key = event["key"]
    if "Control" in event["modifiers"] and len(key) == 1 and 1 <= ord(key) <= 26:
        imgui.get_io().add_key_event(imgui.Key(imgui.Key.a.value + ord(key) - 1), event["event_type"] == "key_down")


def sync_imgui_modifiers(event) -> None:
    """Set imgui's shift, ctrl and alt to the modifiers a pointer event carries."""
    ## NOTE: imgui holds a modifier from its key-down to its key-up, so one whose key-up went to another window stays
    ## held and every table click reads as a shift or ctrl click; the window system's own state on each click is right
    io = imgui.get_io()
    modifiers = event.get("modifiers") or ()
    for name, key in (("Shift", imgui.Key.mod_shift), ("Control", imgui.Key.mod_ctrl), ("Alt", imgui.Key.mod_alt)):
        io.add_key_event(key, name in modifiers)


def over_imgui() -> bool:
    return imgui.get_current_context() is not None and imgui.get_io().want_capture_mouse


def view_under(state: StackAnnotator, event) -> tuple:
    """``(view, (column, row))`` of the brush view under the pointer, in that view's stroke pixels, else ``(None, None)``."""
    for view, subplot in SUBPLOTS_BRUSH.items():
        pos = state.widget.figure[subplot].map_screen_to_world((event.x, event.y))
        n_rows, n_columns = state.brush.strokes[view].shape
        if pos is not None and 0 <= pos[0] < n_columns and 0 <= pos[1] < n_rows:
            return view, (int(pos[0]), int(pos[1]))
    return None, None


def on_pointer(state: StackAnnotator, event) -> None:
    """Brush strokes on XY, XZ and YZ, and clicks that move the crosshair and the slice."""
    if getattr(event, "button", 1) not in (0, 1):
        return
    widget, brush = state.widget, state.brush
    figure = widget.figure
    if event.type == "pointer_move":
        if brush.painting:
            pos = figure[SUBPLOTS_BRUSH[brush.view]].map_screen_to_world((event.x, event.y))
            if pos is not None:
                stamp(state=state, point=(int(pos[0]), int(pos[1])))
        return
    if event.type == "pointer_down":
        state.press = (event.x, event.y)
        view, point = view_under(state=state, event=event)
        modifiers = getattr(event, "modifiers", ()) or ()
        ## NOTE: ctrl + click is the ROI tool's group toggle, so the brush leaves it alone
        if brush.active and view is not None and "Control" not in modifiers and not over_imgui() and not widget._drawing():
            brush.view = view
            brush.painting = True
            brush.erasing = "Shift" in modifiers
            brush.last = None
            figure[SUBPLOTS_BRUSH[view]].controller.enabled = False
            stamp(state=state, point=point)
        return
    press, state.press = state.press, None
    is_click = press is not None and abs(event.x - press[0]) + abs(event.y - press[1]) <= CLICK_SLOP_PX
    if brush.painting:
        brush.painting = False
        figure[SUBPLOTS_BRUSH[brush.view]].controller.enabled = True
        ## NOTE: shift + click without a drag is the ROI tool's add-to-group, so that stroke is dropped
        if brush.erasing and is_click:
            clear_stroke(brush=brush)
        elif brush.view == "xy":
            commit_stroke(state=state)
        else:
            commit_stroke_side(state=state)
        return
    if not is_click or over_imgui():
        return
    view, point = view_under(state=state, event=event)
    if view is None:
        return
    y, x = state.crosshair
    z = None
    if view == "xy":
        x, y = point
    elif view == "xz":
        x, z = point[0], int(point[1] / state.z_scale)
    else:
        y, z = point[1], int(point[0] / state.z_scale)
    move_crosshair(state=state, y=y, x=x, z=None if z is None else min(z, state.arr.shape[-3] - 1))


def draw_class_display(state: StackAnnotator) -> None:
    """One row per class that exists: show its surface, pick its colour, set its surface's opacity."""
    names = {value: name for value, name in enumerate(state.widget.store.label_names, start=2)}
    if 1 in state.n_objects:
        names[1] = "unlabeled"
    changed_any = False
    for value, name in sorted(names.items()):
        display = state.class_display.setdefault(value, ClassDisplay())
        imgui.push_id(f"class_display_{value}")
        changed_visible, display.visible = imgui.checkbox("##show", display.visible)
        imgui.same_line()
        changed_rgb, rgb = imgui.color_edit3(
            "##colour", list(display_rgb(state=state, value=value)), imgui.ColorEditFlags_.no_inputs
        )
        if changed_rgb:
            display.rgb = tuple(rgb)
        imgui.same_line()
        imgui.set_next_item_width(em(8))
        changed_alpha, display.alpha = imgui.slider_float("##opacity", display.alpha, 0.0, 1.0, "opacity %.2f")
        imgui.same_line()
        imgui.text(f"{name}   {state.n_objects.get(value, 0)} in 3D")
        imgui.pop_id()
        changed_any = changed_any or changed_visible or changed_rgb or changed_alpha
    if changed_any:
        apply_class_display(state=state)


def draw_panel(state: StackAnnotator) -> None:
    """The right-hand panel: the tools added here, then the ROI tool's own tab."""
    widget, brush = state.widget, state.brush
    widget._note_hover()
    dz, dy, dx = state.scale_um
    imgui.text_disabled(f"{Path(state.filepath_stack).name}   voxel {dz:.2f} x {dy:.3f} x {dx:.3f} um (z, y, x)")

    with selected_button_style(brush.active):
        if imgui.button("Brush (b)"):
            brush.active = not brush.active
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Drag on XY to paint. The stroke joins the selected ROI on this plane, or starts a new one.\n"
            "Drag on XZ or YZ to paint every plane the stroke crosses, in the selected ROI's class. A closed loop\n"
            "is filled, and the stroke is one brush diameter thick around the crosshair.\n"
            "Shift + drag erases. Ctrl + click and shift + click still group ROIs, and ctrl + click moves the crosshair."
        )
    imgui.same_line()
    imgui.set_next_item_width(em(8))
    _changed, brush.radius = imgui.slider_int("radius (px)", brush.radius, 0, 30)
    if imgui.button("Fill between planes"):
        fill_between(state=state)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Ctrl + click two ROIs on different planes, then fill the planes between them.")
    imgui.same_line()
    if imgui.button("Merge"):
        merge_picked(state=state)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Merge the ROIs picked in the image or the table (ctrl / shift + click) that share a plane.\n"
            "The merged ROI keeps the label of the first one picked."
        )
    imgui.same_line()
    if imgui.button("Delete"):
        delete_picked(state=state)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Delete every ROI picked in the image or the table. Ctrl + z brings them back.")

    changed_volume, state.show_volume = imgui.checkbox("stack in 3D", state.show_volume)
    if changed_volume:
        state.nd_images["volume"].graphic.visible = state.show_volume
    imgui.same_line()
    imgui.set_next_item_width(em(8))
    changed_alpha, state.alpha_volume = imgui.slider_float("##stack_opacity", state.alpha_volume, 0.0, 1.0, "opacity %.2f")
    if changed_alpha:
        state.nd_images["volume"].graphic.alpha = state.alpha_volume
    imgui.same_line()
    changed_surfaces, state.show_surfaces = imgui.checkbox("surfaces", state.show_surfaces)
    if changed_surfaces:
        apply_class_display(state=state)
    draw_trace_controls(state=state)
    imgui.separator()
    widget.draw_rois()


def draw_trace_controls(state: StackAnnotator) -> None:
    """Trace the channel on screen, save, show or hide paths and links, and edit what is picked."""
    imgui.separator_text("Trace")
    _z, channel, _pos = view_position(state=state)
    imgui.begin_disabled(state.trace_job is not None)
    if imgui.button(f"Trace channel {channel + 1}"):
        start_trace(state=state)
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(
            "Ridge filter, two-level threshold and thinning, then paths split at every junction (about 8 s).\n"
            "Replaces the trace on screen; ctrl + z brings the old one back."
        )
    layer = state.trace
    if layer is None:
        return
    imgui.same_line()
    if imgui.button("Save trace"):
        save_trace(state=state)
    imgui.same_line()
    changed_paths, layer.show_paths = imgui.checkbox("paths", layer.show_paths)
    imgui.same_line()
    changed_links, layer.show_links = imgui.checkbox("links", layer.show_links)
    if changed_paths or changed_links:
        trace_layer.refresh_visibility(layer=layer)
    buttons = (
        ("Delete paths", "delete", "Delete the picked paths, and the links that end on them."),
        ("Accept links", "accept", "Turn the picked links into paths that join the two ends."),
        ("Reject links", "reject", "Drop the picked links."),
    )
    for label, action, tooltip in buttons:
        if imgui.button(label):
            edit_trace(state=state, action=action)
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{tooltip}\nClick a line in XY or 3D to pick it, shift + click to pick more.")
        imgui.same_line()
    imgui.new_line()
    trace = layer.trace
    imgui.text_disabled(
        f"channel {trace.channel + 1}: {len(trace.paths)} paths, {len(trace.links)} links;"
        f" picked {len(layer.picked_paths)} paths, {len(layer.picked_links)} links{'   unsaved' if layer.unsaved else ''}"
    )


class StackRoiWidget(ManualRoiWidget):
    """The ROI tool, with each class's display controls under its own LABELS section and one undo for ROIs and trace."""

    annotator: Optional[StackAnnotator] = None

    ## NOTE: _push_undo is private to pml_utilities; every ROI edit passes through it, so it marks the shared order
    def _push_undo(self) -> None:
        super()._push_undo()
        if self.annotator is not None:
            self.annotator.edit_order.append("rois")

    def undo(self) -> None:
        """Ctrl + z: undo the newest step, whether it changed the ROIs or the trace."""
        order = self.annotator.edit_order if self.annotator is not None else []
        if order and order[-1] != "rois":
            undo_trace(state=self.annotator, kind=order.pop())
            return
        if order:
            order.pop()
        super().undo()

    ## NOTE: _draw_labels is private to pml_utilities; it is the only way into the LABELS section
    def _draw_labels(self, g) -> None:
        super()._draw_labels(g)
        if self.annotator is not None:
            draw_class_display(state=self.annotator)

