"""A trace drawn over the stack's XY and 3D views: click to pick paths and links, then delete, accept or reject.

Paths are a fastplotlib line collection per view, each line carrying ``("path", id)`` in its
metadata, so a click finds its path through ``subplot.get_pick_info``. Link proposals are a second,
dashed collection carrying ``("link", row)``. XY shows the paths that pass through the plane on
screen; 3D shows all of them.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import fastplotlib as fpl
import numpy as np

from mbo_utilities.analysis.neurite_tracing import Trace

COLOR_PATH = (0.3, 0.85, 1.0)
COLOR_LINK = (1.0, 0.55, 0.1)
COLOR_PICKED = (1.0, 1.0, 1.0)
THICKNESS = 2.0
HEIGHT_XY = 2.5  ## above the XY image and the ROI overlays, below the brush preview
VIEWS = ("xy", "3d")
UNDO_DEPTH = 50


@dataclass
class TraceLayer:
    """A trace, its graphics and what is picked."""

    trace: Trace
    subplots: dict  ## view -> Subplot
    z_scale: float
    clicks_allowed: Callable  ## () -> bool; False while a click on XY belongs to another tool
    z: int = 0  ## the plane XY shows
    paths: dict = field(default_factory=dict)  ## view -> LineCollection of every path ever drawn, deleted ones hidden
    links: dict = field(default_factory=dict)  ## view -> LineCollection of the link proposals
    position: dict = field(default_factory=dict)  ## path id -> index of its line in each view's collection
    z_ranges: list = field(default_factory=list)  ## index -> (lowest, highest) plane of that path
    next_id: int = 0
    picked_paths: set = field(default_factory=set)  ## path ids
    picked_links: set = field(default_factory=set)  ## rows of trace.links
    coloured: set = field(default_factory=set)  ## path ids drawn in COLOR_PICKED
    show_paths: bool = True
    show_links: bool = True
    undo_steps: list = field(default_factory=list)  ## (paths, links) before each edit
    unsaved: bool = False


def world_points(points: np.ndarray, view: str, z_scale: float) -> np.ndarray:
    """``(z, y, x)`` voxels as ``(x, y, z)`` world positions: depth in pixels for 3D, just above the image for XY."""
    depth = points[:, 0] * z_scale if view == "3d" else np.full(len(points), HEIGHT_XY, np.float32)
    return np.column_stack([points[:, 2], points[:, 1], depth]).astype(np.float32)


def draw_over_volume(graphic) -> None:
    """Draw a line in front of the volume, whose max projection fills the depth buffer."""
    ## NOTE: fastplotlib has no public switch for this; the material is pygfx's
    graphic.world_object.material.depth_test = False
    graphic.world_object.render_order = 1


def build_layer(trace: Trace, subplots: dict, z_scale: float, z: int, clicks_allowed: Callable) -> TraceLayer:
    """
    Draw a trace.

    Parameters
    ----------
    trace : Trace
        The paths and link proposals.
    subplots : dict
        ``"xy"`` and ``"3d"`` subplots. XY shows the image with pixel (row, col) at world (col, row).
    z_scale : float
        Plane spacing in pixels.
    z : int
        The plane XY shows.
    clicks_allowed : Callable
        Called with no arguments on each click; a click picks nothing while it returns False.

    Returns
    -------
    TraceLayer
        The layer, with its graphics added.
    """
    layer = TraceLayer(
        trace=trace, subplots=subplots, z_scale=z_scale, clicks_allowed=clicks_allowed, z=z, next_id=trace.next_id
    )
    add_path_lines(layer=layer, ids=list(trace.paths))
    draw_links(layer=layer)
    refresh_visibility(layer=layer)
    return layer


def remove_layer(layer: TraceLayer) -> None:
    """Delete the layer's graphics."""
    for view in VIEWS:
        for collections in (layer.paths, layer.links):
            if view in collections:
                layer.subplots[view].delete_graphic(collections[view])
    layer.paths, layer.links = {}, {}


def add_path_lines(layer: TraceLayer, ids: list) -> None:
    """Give paths ``ids`` a line in each view, starting each view's collection with the first ones."""
    if not ids:
        return
    for view in VIEWS:
        points = [world_points(points=layer.trace.paths[i], view=view, z_scale=layer.z_scale) for i in ids]
        if view not in layer.paths:
            collection = layer.subplots[view].add_line_collection(
                points, colors=COLOR_PATH, thickness=THICKNESS, metadatas=[("path", i) for i in ids], name="paths"
            )
            collection.add_event_handler(partial(on_click, layer, view), "click")
            layer.paths[view] = collection
        else:
            for path_id, data in zip(ids, points):
                line = fpl.LineGraphic(data, colors=COLOR_PATH, thickness=THICKNESS, metadata=("path", path_id))
                layer.paths[view].add_graphic(line)
        if view == "3d":
            for line in layer.paths[view].graphics[-len(ids) :]:
                draw_over_volume(graphic=line)
    for path_id in ids:
        layer.position[path_id] = len(layer.z_ranges)
        planes = layer.trace.paths[path_id][:, 0]
        layer.z_ranges.append((float(planes.min()), float(planes.max())))
    layer.next_id = max(layer.next_id, max(ids) + 1)


def draw_links(layer: TraceLayer) -> None:
    """Draw the link proposals again, one dashed line each."""
    for view, collection in layer.links.items():
        layer.subplots[view].delete_graphic(collection)
    layer.links = {}
    layer.picked_links = set()
    if len(layer.trace.links) == 0:
        return
    for view in VIEWS:
        collection = layer.subplots[view].add_line_collection(
            [world_points(points=ends, view=view, z_scale=layer.z_scale) for ends in layer.trace.links],
            colors=COLOR_LINK,
            thickness=THICKNESS,
            dash_pattern="--",
            metadatas=[("link", row) for row in range(len(layer.trace.links))],
            name="links",
        )
        collection.add_event_handler(partial(on_click, layer, view), "click")
        if view == "3d":
            for line in collection.graphics:
                draw_over_volume(graphic=line)
        layer.links[view] = collection


def refresh_visibility(layer: TraceLayer) -> None:
    """Show the paths still in the trace, in XY only those that pass through plane ``layer.z``, and the links."""
    if layer.paths:
        alive = np.zeros(len(layer.z_ranges), bool)
        alive[[layer.position[i] for i in layer.trace.paths]] = True
        z_ranges = np.array(layer.z_ranges)  ## (n_lines, 2)
        on_plane = (z_ranges[:, 0] - 0.5 <= layer.z) & (layer.z <= z_ranges[:, 1] + 0.5)
        layer.paths["3d"].visibles = alive & layer.show_paths
        layer.paths["xy"].visibles = alive & on_plane & layer.show_paths
    for collection in layer.links.values():
        collection.visible = layer.show_links
    refresh_colours(layer=layer)


def refresh_colours(layer: TraceLayer) -> None:
    """Draw the picked paths and links in COLOR_PICKED and the rest in their own colour."""
    stale = [layer.position[i] for i in layer.coloured - layer.picked_paths]
    fresh = [layer.position[i] for i in layer.picked_paths - layer.coloured]
    for collection in layer.paths.values():
        if stale:
            collection.colors[stale] = COLOR_PATH
        if fresh:
            collection.colors[fresh] = COLOR_PICKED
    layer.coloured = set(layer.picked_paths)
    for collection in layer.links.values():
        collection.colors[:] = COLOR_LINK
        if layer.picked_links:
            collection.colors[sorted(layer.picked_links)] = COLOR_PICKED


def show_plane(layer: TraceLayer, z: int) -> None:
    """Show in XY the paths through plane ``z``."""
    if z != layer.z:
        layer.z = z
        refresh_visibility(layer=layer)


def on_click(layer: TraceLayer, view: str, event) -> None:
    """Pick the path or link under a click: a plain click picks it alone, shift or ctrl adds or removes it."""
    if not layer.clicks_allowed():
        return
    ## NOTE: a handler on a collection is told the collection, so the line hit is looked up by position
    info = layer.subplots[view].get_pick_info((event.x, event.y))
    if info is None or not isinstance(info["graphic"].metadata, tuple):
        return
    kind, key = info["graphic"].metadata
    picked = layer.picked_paths if kind == "path" else layer.picked_links
    modifiers = event.modifiers or ()
    if "Shift" in modifiers or "Control" in modifiers:
        picked.symmetric_difference_update({key})
    else:
        layer.picked_paths.clear()
        layer.picked_links.clear()
        picked.add(key)
    refresh_colours(layer=layer)


def push_undo(layer: TraceLayer) -> None:
    """Remember the trace before an edit."""
    layer.undo_steps.append((dict(layer.trace.paths), layer.trace.links.copy()))
    del layer.undo_steps[:-UNDO_DEPTH]


def undo(layer: TraceLayer) -> bool:
    """Put the trace back as it was before the last edit; False when there is nothing to undo."""
    if not layer.undo_steps:
        return False
    layer.trace.paths, layer.trace.links = layer.undo_steps.pop()
    layer.picked_paths = set()
    draw_links(layer=layer)
    refresh_visibility(layer=layer)
    layer.unsaved = True
    return True


def path_ends(paths: dict) -> set:
    """The first and last point of every path, as tuples."""
    return {tuple(points[index]) for points in paths.values() for index in (0, -1)}


def delete_picked_paths(layer: TraceLayer) -> int:
    """Delete the picked paths and the links that ended on them; returns how many paths went."""
    ids = [i for i in layer.picked_paths if i in layer.trace.paths]
    if not ids:
        return 0
    push_undo(layer=layer)
    layer.trace.paths = {i: points for i, points in layer.trace.paths.items() if i not in layer.picked_paths}
    ends = path_ends(paths=layer.trace.paths)
    keep = [tuple(a) in ends and tuple(b) in ends for a, b in layer.trace.links]
    layer.trace.links = layer.trace.links[np.array(keep, bool).reshape(-1)]
    layer.picked_paths = set()
    draw_links(layer=layer)
    refresh_visibility(layer=layer)
    layer.unsaved = True
    return len(ids)


def resolve_picked_links(layer: TraceLayer, accept: bool) -> int:
    """Accept the picked links as new two-point paths, or reject them; returns how many."""
    rows = sorted(layer.picked_links)
    if not rows:
        return 0
    push_undo(layer=layer)
    ids_new = []
    if accept:
        for row in rows:
            layer.trace.paths[layer.next_id] = layer.trace.links[row].copy()
            ids_new.append(layer.next_id)
            layer.next_id += 1
    layer.trace.links = np.delete(layer.trace.links, rows, axis=0)
    add_path_lines(layer=layer, ids=ids_new)
    draw_links(layer=layer)
    refresh_visibility(layer=layer)
    layer.unsaved = True
    return len(rows)
