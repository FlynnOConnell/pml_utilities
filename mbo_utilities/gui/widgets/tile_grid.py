"""TileGridViewer - fast preview of every tile of a tiled acquisition.

Tiled acquisitions fold spatial tiles into the T axis, so the main viewer
only shows one tile at a time on the scrollwheel. This widget lays all
tiles of a chosen z-block out in a grid (columns/rows from the per-tile
stage positions in ``metadata["tiles"]``) for a quick glimpse across the
whole acquisition.

Thumbnails are read straight from the raw data: the full depth strided
down in Y/X to ~128 px and max-projected. Those reads run on a background
thread (see ``_prefetch_loop``), prioritised in display order and started as
soon as the side-panel section appears — so the current view's tiles are
already decompressed by the time the grid is opened. The draw loop only
consumes finished thumbs (cache miss -> placeholder), so opening never blocks
and the grid stays responsive while the rest fill in. Clicking a cell drives
the main viewer's Tile slider.

An orientation control adds 90°-multiple rotations + X/Y/Z flips (the same
ops the BigStitcher export bakes onto VW00), applied as a live preview only
— nothing is saved. For a faithful out-of-plane view (rotate about X/Y, flip
Z) the displayed tile is the matching source MIP (xy/xz/yz) with a 2D
transpose + flips. One read per tile fills all three axes, so every
orientation — the default included — shows the same real projection.

Activates for any array that is tiled and carries per-tile metadata
(see :class:`mbo_utilities.arrays.isoview.IsoviewArray`).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from imgui_bundle import imgui

from mbo_utilities import log
from mbo_utilities.gui._imgui_helpers import (
    button_width,
    draw_toolbar_row,
    selected_button_style,
)
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.summary_image import (
    _DEFAULT_COLORMAP,
    _DEFAULT_COLORMAPS,
    _auto_range,
    _GpuImage,
    center_popup_on_open,
    draw_section_header,
)

logger = log.get("gui.tile_grid")

_WHITE = (1.0, 1.0, 1.0, 1.0)

_THUMB_MAX = 128  # longest thumbnail edge in px; the y/x stride lands near it
_THUMB_CACHE_MAX = 4096  # ~32 KiB each: every tile of a 16x16x8 dataset is ~64 MiB
_MIP_CACHE_FLOOR = (
    8192  # source MIPs, ~96 KiB per tile and axis; a whole-dataset working set
)
_MIP_CACHE_MAX_HARD = 24576
_GPU_CACHE_MAX = 320  # about one z-block of textures plus slack
_PREFETCH_WORKERS = max(
    2, min(8, (os.cpu_count() or 4) - 1)
)  # reads release the GIL in the codec

_CONTRAST_DISPLAY = 0
_CONTRAST_AUTO = 1
_CONTRAST_MANUAL = 2
_CONTRAST_LABELS = ("Display", "Auto", "Manual")


def _cluster_axis(values: list[float]) -> list[float]:
    """Group stage-coordinate values into ordered grid bins.

    Values within half the smallest gap collapse to one bin (absorbs
    acquisition jitter); the bin's mean is its representative center.
    """
    uniq = sorted({float(v) for v in values if v is not None})
    if not uniq:
        return []
    gaps = [b - a for a, b in zip(uniq, uniq[1:]) if b - a > 1e-9]
    tol = min(gaps) * 0.5 if gaps else 1.0
    clusters: list[list[float]] = [[uniq[0]]]
    for v in uniq[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _bin_index(centers: list[float], v: float | None) -> int:
    if not centers or v is None:
        return 0
    return int(np.argmin([abs(v - c) for c in centers]))


def _index_centers(entries: list, idx_pos: int, stage_pos: int, n: int) -> list[float]:
    """Representative stage coordinate for each integer tile index.

    Averages the stage coordinate of every tile sharing an index; falls
    back to the index itself when no stage coordinate is available (keeps
    the z-block um readout working when digit indices drive placement).
    """
    centers = []
    for i in range(n):
        vals = [
            e[stage_pos]
            for e in entries
            if e[idx_pos] == i and e[stage_pos] is not None
        ]
        centers.append(sum(vals) / len(vals) if vals else float(i))
    return centers


def _unwrap(arr):
    """Peel display proxies (`_SqueezeSingletonDims`, `_ScrubTimingProxy`)
    so reads hit the real array with unmodified 5D indexing.
    """
    seen: set[int] = set()
    while True:
        if id(arr) in seen:
            return arr
        seen.add(id(arr))
        inner = getattr(arr, "_arr", None)
        if inner is None or inner is arr:
            return arr
        arr = inner


class _GuardedGetSubplot:
    """fastplotlib's right-click gate, answering False while the Tile Grid window has the mouse."""

    def __init__(self, widget, orig):
        self.widget = widget
        self.orig = orig

    def __call__(self):
        if self.widget._popup_open and imgui.get_io().want_capture_mouse:
            return False
        return self.orig()


class TileGridViewer(Widget):
    """Preview all tiles of a tiled acquisition laid out per z-block."""

    name = "Tile Grid"
    priority = 66
    toggle_key = "preview"

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._popup_open: bool = False
        self._zblock: int = 0
        self._c_index: int = 0
        self._cmaps: list[str] = list(_DEFAULT_COLORMAPS)
        self._cmap_idx: int = self._cmaps.index(_DEFAULT_COLORMAP)
        self._contrast_mode: int = _CONTRAST_DISPLAY
        self._manual_lo: float = 0.0
        self._manual_hi: float = 300.0
        # 0 = whole-tile MIP, 1 = one plane read lazily
        self._view_mode: int = 0
        self._plane: int = 0
        self._nplanes: int = 1
        # (camera, tile) -> {"X", "Y"}
        self._tile_flips: dict[
            tuple, set
        ] = {}  # (camera, tile) -> 90deg ccw steps, 0-3
        self._tile_rot: dict[
            tuple, int
        ] = {}  # cameras whose default orientation was applied
        self._flip_seeded: set = set()  # ti -> (tile_x, tile_y, tile_z)
        self._tile_xyz: dict[int, tuple] = {}  # every camera mounted 90deg on its side
        self._is_rotated: bool = False
        self._orient_menu_ti: int | None = None
        self._orient_menu_open: bool = False
        # (camera, z-block) -> {(ri, ci): ti}
        self._layout: dict[tuple, dict] = {}
        self._tile_spc: dict[int, int] = {}
        self._edit_layout: bool = False
        self._pick: tuple | None = None

        self._sig: str | None = None
        self._grid: dict | None = None
        self._view_names: list[str] = []
        self._tile_labels: dict[int, str] = {}

        self._mip_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()  # (ti, c, axis)
        self._thumb_cache: OrderedDict[tuple, np.ndarray] = (
            OrderedDict()  # (ti, c, mode, plane, rot, fx, fy)
        )
        self._gpu_cache: OrderedDict[tuple, _GpuImage] = OrderedDict()
        self._range: tuple[float, float] | None = None
        self._range_sig: tuple | None = None

        # _mip_cache is the only state shared with the prefetch threads; _cache_gen bumps on
        # reset so an in-flight read cannot commit after a dataset change
        self._cache_lock = threading.Lock()
        self._cache_gen: int = 0
        self._mip_cache_max: int = _MIP_CACHE_FLOOR
        self._prefetch_threads: list[threading.Thread] = []
        self._prefetch_stop = threading.Event()
        self._prefetch_wake = threading.Event()
        self._prefetch_req_lock = threading.Lock()
        self._prefetch_req: list[tuple] = []
        self._prefetch_pos: int = 0
        self._prefetch_sig: tuple | None = None
        self._prefetch_arr: Any = None

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return cls._find(parent) is not None

    @staticmethod
    def _find(parent: Any):
        for raw in parent._get_data_arrays():
            arr = _unwrap(raw)
            if bool(getattr(arr, "is_tiled", False)) and getattr(
                arr, "tile_metadata", None
            ):
                return arr
        return None

    def _active_array(self):
        return self._find(self.parent)

    def _backend(self):
        try:
            return self.parent._figure.imgui_renderer.backend
        except AttributeError:
            return None

    def _ensure_grid(self, arr) -> bool:
        sig = str(getattr(arr, "scan_root", "")) or str(id(arr))
        if self._sig == sig and self._grid is not None:
            return True
        self._sig = sig
        self._reset_caches()
        self._layout = {}
        self._tile_spc = {}
        self._tile_flips = {}
        self._tile_rot = {}
        self._flip_seeded = set()
        self._tile_xyz = {}
        self._pick = None
        self._view_names = list(getattr(arr, "view_names", []) or [])
        self._is_rotated = (
            str((getattr(arr, "metadata", {}) or {}).get("camera_orientation", ""))
            .strip()
            .lower()
            == "rotated"
        )
        self._nplanes = max(1, int(arr.shape[2]))
        self._plane = min(self._plane, self._nplanes - 1)

        tiles = arr.tile_metadata or {}
        entries = []
        self._tile_labels = {
            int(ti): (
                str(t.get("specimen_name"))
                if t.get("specimen_name")
                else f"SPM{int(t.get('specimen', ti)):02d}"
            )
            for ti, t in tiles.items()
        }
        for ti, t in tiles.items():
            entries.append(
                (
                    int(ti),
                    int(t.get("specimen", ti)),
                    t.get("stage_x"),
                    t.get("stage_y"),
                    t.get("stage_z"),
                    t.get("tile_x"),
                    t.get("tile_y"),
                    t.get("tile_z"),
                )
            )
        if not entries:
            self._grid = None
            return False
        self._tile_xyz = {e[0]: (e[5], e[6], e[7]) for e in entries}

        # the digit grid (specimen_name's trailing XYZ) wins when every tile has one; else
        # the stage coordinates are clustered
        placed: dict[int, dict[tuple[int, int], tuple[int, int]]] = {}
        use_digits = all(
            e[5] is not None and e[6] is not None and e[7] is not None for e in entries
        )
        # NOTE: the verified BigStitcher layout is BDV-x = -stage_y, BDV-y = +stage_x: tile_x runs
        # top to bottom (rows) and tile_y runs right to left (columns reversed), so TL010 sits
        # left of TL000 and TL100 under it; "Move tiles" handles a rig whose sign differs
        if use_digits:
            ncols = max(e[6] for e in entries) + 1
            nrows = max(e[5] for e in entries) + 1
            nz = max(e[7] for e in entries) + 1
            cols = _index_centers(entries, 6, 3, ncols)[::-1]
            rows = _index_centers(entries, 5, 2, nrows)
            zblocks = _index_centers(entries, 7, 4, nz)
            for ti, spc, x, y, z, tx, ty, tz in entries:
                ci = ncols - 1 - int(ty)
                placed.setdefault(int(tz), {})[(int(tx), ci)] = (ti, spc)
        else:
            cols_asc = _cluster_axis([e[3] for e in entries])
            cols = cols_asc[::-1]
            rows = _cluster_axis([e[2] for e in entries])
            zblocks = _cluster_axis([e[4] for e in entries])
            ncols = len(cols_asc)
            for ti, spc, x, y, z, *_rest in entries:
                zi = _bin_index(zblocks, z)
                ri = _bin_index(rows, x)
                ci = ncols - 1 - _bin_index(cols_asc, y)
                placed.setdefault(zi, {})[(ri, ci)] = (ti, spc)

        self._grid = {
            "cols": cols,
            "rows": rows,
            "zblocks": zblocks,
            "placed": placed,
            "ntiles": len(entries),
            # the larger working set of the two modes, so prefetched tiles survive until scrubbed to:
            # MIP mode needs xy/xz/yz per tile, plane mode every plane of the z-block plus one elsewhere
        }
        ntiles = len(entries)
        tiles_per_block = max((len(v) for v in placed.values()), default=1)
        mip_need = ntiles * 3
        plane_need = tiles_per_block * max(1, self._nplanes) + ntiles
        self._mip_cache_max = min(
            _MIP_CACHE_MAX_HARD,
            max(_MIP_CACHE_FLOOR, mip_need, plane_need) + 64,
        )
        self._zblock = min(self._zblock, max(0, len(zblocks) - 1))
        return True

    def _reset_caches(self) -> None:
        for gpu in self._gpu_cache.values():
            gpu.destroy()
        self._gpu_cache.clear()
        self._thumb_cache.clear()
        with self._cache_lock:
            self._mip_cache.clear()
            self._cache_gen += 1
        with self._prefetch_req_lock:
            self._prefetch_req = []
            self._prefetch_pos = 0
        self._prefetch_sig = None
        self._range = None
        self._range_sig = None

    def _mip_params(self, arr) -> tuple[int, int, float, float]:
        """``(xy_stride, z_stride, pix_xy_um, pix_z_um)`` for the MIP reads.

        ``z_stride`` is chosen so a decimated Z pixel spans about the same
        physical distance as a decimated Y/X pixel — that keeps xz/yz
        thumbnails close to square-pixelled. Falls back to no Z decimation
        and unit pixels when the voxel sizes aren't in the metadata.
        """
        ny, nx = int(arr.shape[3]), int(arr.shape[4])
        s = max(1, -(-max(ny, nx) // _THUMB_MAX))
        dxy = float(arr.dx or 0.0)
        dz = float(arr.dz or 0.0)
        if dxy > 0 and dz > 0:
            sz = max(1, int(round(s * dxy / dz)))
            return s, sz, s * dxy, sz * dz
        return s, 1, 1.0, 1.0

    def _cached_source(self, key: tuple) -> np.ndarray | None:
        """Non-blocking ``_mip_cache`` lookup for the draw thread (no read)."""
        with self._cache_lock:
            v = self._mip_cache.get(key)
            if v is not None:
                self._mip_cache.move_to_end(key)
            return v

    def _load_source_plane(self, arr, ti, c, z) -> np.ndarray | None:
        """Single decimated Z-plane for one tile, read on the prefetch thread
        and cached per ``(ti, c, "plane", z)``. Only the requested plane is
        pulled from the lazy array, so scrubbing planes stays cheap even on
        raw stacks. The slow read runs outside the cache lock.
        """
        z = int(z)
        key = (ti, c, "plane", z)
        with self._cache_lock:
            cached = self._mip_cache.get(key)
            if cached is not None:
                self._mip_cache.move_to_end(key)
                return cached
            gen = self._cache_gen
        s, _sz, _, _ = self._mip_params(arr)
        plane = np.squeeze(np.asarray(arr[ti, c, z, ::s, ::s]))
        if plane.ndim != 2:
            return None
        plane = np.ascontiguousarray(plane)
        with self._cache_lock:
            if gen != self._cache_gen:
                return None
            self._mip_cache[key] = plane
            while len(self._mip_cache) > self._mip_cache_max:
                self._mip_cache.popitem(last=False)
            return self._mip_cache.get(key)

    def _load_source_mip(self, arr, ti, c, axis) -> np.ndarray | None:
        """Decimated MIP of one tile for ``axis``, read on the prefetch thread and cached per ``(ti, c, axis)``.

        One read of the full depth (Y/X strided to thumbnail size) fills xy,
        xz and yz, so a rotated tile shows the same projection as the
        unrotated one. The read runs outside the cache lock.
        """
        key = (ti, c, axis)
        with self._cache_lock:
            cached = self._mip_cache.get(key)
            if cached is not None:
                self._mip_cache.move_to_end(key)
                return cached
            gen = self._cache_gen
        s, sz, _, _ = self._mip_params(arr)
        block = np.asarray(arr[ti, c, :, ::s, ::s])
        if block.ndim != 3:
            block = np.squeeze(block)
        if block.ndim != 3:
            return None
        xy = np.ascontiguousarray(block.max(axis=0))
        xz = np.ascontiguousarray(block.max(axis=1)[::sz])
        yz = np.ascontiguousarray(block.max(axis=2)[::sz])
        with self._cache_lock:
            if gen != self._cache_gen:
                return None
            self._mip_cache[(ti, c, "xy")] = xy
            self._mip_cache[(ti, c, "xz")] = xz
            self._mip_cache[(ti, c, "yz")] = yz
            while len(self._mip_cache) > self._mip_cache_max:
                self._mip_cache.popitem(last=False)
            return self._mip_cache.get(key)

    def _ensure_prefetch_threads(self) -> None:
        if self._prefetch_stop.is_set():
            return
        self._prefetch_threads = [t for t in self._prefetch_threads if t.is_alive()]
        while len(self._prefetch_threads) < _PREFETCH_WORKERS:
            t = threading.Thread(
                target=self._prefetch_loop,
                name=f"tilegrid-prefetch-{len(self._prefetch_threads)}",
                daemon=True,
            )
            self._prefetch_threads.append(t)
            t.start()

    def _prefetch_jobs(self) -> list[tuple]:
        """Ordered ``(ti, c, view_mode, plane)`` for the current view.

        The current z-block's tiles come first (display order). Once the popup
        is open the queue is widened so scrubbing never blacks out:
        - MIP mode: every other z-block, by distance from the current one.
        - Plane mode: every plane of the current z-block (current plane first,
          nearest outward), then the current plane of the other z-blocks.
        While closed only the current z-block at the current plane is queued, so
        an unopened widget doesn't read the whole dataset. ``plane`` is -1 in
        MIP mode (the read covers all planes).
        """
        g = self._grid
        if not g:
            return []
        self._seed_camera_defaults(self._c_index)
        layout = self._get_layout(g)
        c = self._c_index
        vm = self._view_mode
        cur_tiles = [layout[cell] for cell in sorted(layout.keys())]

        if vm != 1:
            jobs = [(ti, c, 0, -1) for ti in cur_tiles]
            if self._popup_open:
                jobs.extend(self._other_zblock_jobs(-1))
            return jobs

        nplanes = max(1, self._nplanes)
        cur_p = min(max(0, self._plane), nplanes - 1)
        if not self._popup_open:
            return [(ti, c, 1, cur_p) for ti in cur_tiles]
        plane_order = sorted(range(nplanes), key=lambda p: (abs(p - cur_p), p))
        jobs = [(ti, c, 1, p) for p in plane_order for ti in cur_tiles]
        jobs.extend(self._other_zblock_jobs(cur_p))
        return jobs

    def _other_zblock_jobs(self, plane: int) -> list[tuple]:
        """``(ti, c, view_mode, plane)`` for every tile outside the current z-block, nearest block first."""
        placed = self._grid["placed"]
        cur_z = self._zblock
        jobs = []
        for z in sorted(placed.keys(), key=lambda zz: (abs(zz - cur_z), zz)):
            if z == cur_z:
                continue
            cells = placed[z]
            for cell in sorted(cells.keys()):
                jobs.append((cells[cell][0], self._c_index, self._view_mode, plane))
        return jobs

    def _prefetch_current_view(self, arr) -> None:
        """Start/keep the workers fed with the current view. A no-op until the
        view changes (same signature), so it is cheap to call every frame —
        including while the popup is closed, which readies the first z-block
        before the grid is opened. Opening expands the queue to every z-block.
        """
        self._prefetch_arr = arr
        self._ensure_prefetch_threads()
        sig = (
            self._sig,
            self._zblock,
            self._c_index,
            self._view_mode,
            self._plane if self._view_mode == 1 else -1,
            self._popup_open,
        )
        if sig == self._prefetch_sig:
            return
        self._prefetch_sig = sig
        jobs = self._prefetch_jobs()
        with self._prefetch_req_lock:
            self._prefetch_req = jobs
            self._prefetch_pos = 0
        self._prefetch_wake.set()

    def _next_prefetch_job(self) -> tuple | None:
        """Hand the next queued job to exactly one worker (cursor advance)."""
        with self._prefetch_req_lock:
            if self._prefetch_pos >= len(self._prefetch_req):
                return None
            job = self._prefetch_req[self._prefetch_pos]
            self._prefetch_pos += 1
            return job

    def _prefetch_loop(self) -> None:
        while not self._prefetch_stop.is_set():
            job = self._next_prefetch_job()
            if job is None:
                self._prefetch_wake.wait(timeout=0.5)
                self._prefetch_wake.clear()
                continue
            arr = self._prefetch_arr
            if arr is None:
                continue
            ti, c, vm, plane = job
            try:
                if vm == 1:
                    self._load_source_plane(arr, ti, c, plane)
                else:
                    # the tile stays a placeholder; the worker lives on for the rest of the queue
                    self._load_source_mip(arr, ti, c, "xy")
            except Exception:
                logger.debug(f"tile grid: read of tile {ti} failed", exc_info=True)

    def _contrast_range(
        self, arr, loaded: list[np.ndarray], plan_key: tuple
    ) -> tuple[float, float]:
        if self._contrast_mode == _CONTRAST_MANUAL:
            lo, hi = self._manual_lo, self._manual_hi
            return lo, (hi if hi > lo else lo + 1.0)
        if self._contrast_mode == _CONTRAST_DISPLAY:
            lo = float(getattr(arr, "_cached_vmin", 0.0) or 0.0)
            hi = float(getattr(arr, "_cached_vmax", 1000.0) or 1000.0)
            return lo, (hi if hi > lo else lo + 1.0)
        sig = (self._zblock, self._c_index, plan_key, len(loaded))
        if sig == self._range_sig and self._range is not None:
            return self._range
        if loaded:
            sample = np.concatenate([t.ravel() for t in loaded])
            self._range = _auto_range(sample)
            # the viewer's display range until thumbs arrive, so tiles do not clip to the colormap top
            self._range_sig = sig
        elif self._range is None:
            lo = float(getattr(arr, "_cached_vmin", 0.0) or 0.0)
            hi = float(getattr(arr, "_cached_vmax", 1000.0) or 1000.0)
            self._range = (lo, hi if hi > lo else lo + 1.0)
        return self._range

    def _ensure_gpu(
        self, key: tuple[int, int], thumb: np.ndarray, lo: float, hi: float
    ) -> _GpuImage | None:
        backend = self._backend()
        if backend is None:
            return None
        cmap = self._cmaps[self._cmap_idx]
        gpu = self._gpu_cache.get(key)
        if gpu is None or gpu.arr is not thumb:
            if gpu is not None:
                gpu.destroy()
                self._gpu_cache.pop(key, None)
            gpu = _GpuImage(backend, thumb, cmap, lo, hi)
            self._gpu_cache[key] = gpu
            while len(self._gpu_cache) > _GPU_CACHE_MAX:
                _, evicted = self._gpu_cache.popitem(last=False)
                evicted.destroy()
        else:
            gpu.reupload_if_changed(cmap, lo, hi)
            self._gpu_cache.move_to_end(key)
        return gpu

    def _tile_slider_name(self) -> str | None:
        from mbo_utilities.arrays.features import find_slider_name

        return find_slider_name(self.parent.image_widget._slider_dim_names or (), "t")

    def _current_tile(self) -> int | None:
        name = self._tile_slider_name()
        if name is None:
            return None
        return int(self.parent.image_widget.indices[name])

    def _jump_to_tile(self, ti: int) -> None:
        name = self._tile_slider_name()
        if name is not None:
            self.parent.image_widget.indices[name] = int(ti)

    def _suppress_fpl_right_click_menu(self) -> None:
        """Keep fastplotlib's right-click menu closed while the Tile Grid window holds the mouse.

        fpl's ``StandardRightClickMenu.update`` opens on any right-click
        inside a subplot whatever imgui wants, so its ``get_subplot`` gate is
        wrapped once with ``_GuardedGetSubplot``.
        """
        fig = getattr(self.parent, "_figure", None)
        rcm = getattr(fig, "_right_click_menu", None)
        if rcm is None or getattr(rcm, "_mbo_guarded", False):
            return
        rcm.get_subplot = _GuardedGetSubplot(self, rcm.get_subplot)
        rcm._mbo_guarded = True

    def draw(self) -> None:
        arr = self._active_array()
        if arr is None or not self._ensure_grid(arr):
            return
        self._prefetch_current_view(arr)
        self._suppress_fpl_right_click_menu()
        draw_section_header("Tile Grid")
        imgui.indent(8)
        try:
            if imgui.button("Open grid##tilegrid_open"):
                self._popup_open = True
            g = self._grid
            ncols = max(1, len(g["cols"]))
            nrows = max(1, len(g["rows"]))
            nz = max(1, len(g["zblocks"]))
            imgui.text_colored(_WHITE, f"{g['ntiles']} tiles")
            imgui.text_colored(_WHITE, f"grid: {ncols} x {nrows}")
            imgui.text_colored(_WHITE, f"z-blocks: {nz}")
            sr = str(getattr(arr, "scan_root", "") or "")
            src = sr.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if src:
                imgui.text_colored(_WHITE, f"source: {src}")
            imgui.text_colored(
                _WHITE,
                "orientation: rotated (90deg seeded)"
                if self._is_rotated
                else "orientation: normal",
            )
        finally:
            imgui.unindent(8)

        if self._popup_open:
            self._draw_popup(arr)

    def _zblock_range(self, i: int) -> tuple[float, float]:
        """Stage-Z interval covered by z-block ``i``: from this block's
        position to the next block's (the last block mirrors the prior gap).
        """
        zb = self._grid["zblocks"]
        if not zb:
            return (0.0, 0.0)
        if len(zb) == 1:
            return (zb[0], zb[0])
        if i < len(zb) - 1:
            return (zb[i], zb[i + 1])
        return (zb[i], zb[i] + (zb[i] - zb[i - 1]))

    def _draw_toolbar(self) -> None:
        nz = len(self._grid["zblocks"])
        row1 = [("Mode", 100.0, self._toolbar_mode)]
        if self._view_mode == 1 and self._nplanes > 1:
            row1.append(("Plane", 260.0, self._toolbar_plane))
        if nz > 1:
            row1.append(("Z-block", 260.0, self._toolbar_zblock))
        draw_toolbar_row(row1)

        row2 = []
        if len(self._view_names) > 1:
            row2.append(("View", 140.0, self._toolbar_view))
        row2.append(("Cmap", 120.0, self._toolbar_cmap))
        row2.append(("Contrast", 100.0, self._toolbar_contrast))
        if self._contrast_mode == _CONTRAST_MANUAL:
            row2.append(("Min", 90.0, self._toolbar_min))
            row2.append(("Max", 90.0, self._toolbar_max))
            row2.append((None, button_width("0-300"), self._toolbar_preset))
        draw_toolbar_row(row2)

        draw_toolbar_row(
            [
                (None, button_width("Move tiles") + 20.0, self._toolbar_move),
                (None, button_width("Reset layout"), self._toolbar_reset_layout),
                (None, button_width("Reset flips"), self._toolbar_reset_flips),
            ]
        )
        imgui.text_colored(
            imgui.ImVec4(0.2, 0.9, 1.0, 1.0),
            "right-click a tile to rotate / flip"
            + ("   |   click to pick/place" if self._edit_layout else ""),
        )

    def _toolbar_mode(self) -> None:
        ch, v = imgui.combo("##tilegrid_mode", self._view_mode, ["MIP", "Plane"])
        if ch:
            self._view_mode = int(v)

    def _toolbar_plane(self) -> None:
        ch, v = imgui.slider_int(
            "##tilegrid_plane",
            self._plane,
            0,
            self._nplanes - 1,
            f"%d / {self._nplanes - 1}",
        )
        if ch:
            self._plane = int(v)

    def _toolbar_zblock(self) -> None:
        nz = len(self._grid["zblocks"])
        lo, hi = self._zblock_range(self._zblock)
        ch, v = imgui.slider_int(
            "##tilegrid_zblock",
            self._zblock,
            0,
            nz - 1,
            f"%d/{nz - 1}  z {lo:.0f}-{hi:.0f}um",
        )
        if ch:
            self._zblock = int(v)

    def _toolbar_view(self) -> None:
        ch, v = imgui.combo("##tilegrid_view", self._c_index, list(self._view_names))
        if ch:
            self._c_index = v

    def _toolbar_cmap(self) -> None:
        ch, v = imgui.combo("##tilegrid_cmap", self._cmap_idx, list(self._cmaps))
        if ch:
            self._cmap_idx = v

    def _toolbar_contrast(self) -> None:
        ch, v = imgui.combo(
            "##tilegrid_contrast", self._contrast_mode, list(_CONTRAST_LABELS)
        )
        if ch:
            self._contrast_mode = v

    def _toolbar_min(self) -> None:
        ch, v = imgui.drag_float(
            "##tilemin", self._manual_lo, 1.0, 0.0, 65535.0, "%.0f"
        )
        if ch:
            self._manual_lo = float(v)

    def _toolbar_max(self) -> None:
        ch, v = imgui.drag_float(
            "##tilemax", self._manual_hi, 1.0, 0.0, 65535.0, "%.0f"
        )
        if ch:
            self._manual_hi = float(v)

    def _toolbar_preset(self) -> None:
        if imgui.small_button("0-300##tilepreset"):
            self._manual_lo, self._manual_hi = 0.0, 300.0

    def _toolbar_move(self) -> None:
        _, self._edit_layout = imgui.checkbox("Move tiles", self._edit_layout)
        if not self._edit_layout:
            self._pick = None

    def _toolbar_reset_layout(self) -> None:
        if imgui.small_button("Reset layout"):
            self._layout.pop((self._c_index, self._zblock), None)
            self._pick = None
        if imgui.is_item_hovered():
            imgui.set_tooltip("restore tile positions (this view)")

    def _toolbar_reset_flips(self) -> None:
        if imgui.small_button("Reset flips"):
            self._tile_flips.clear()
            self._tile_rot.clear()
            self._flip_seeded.clear()
        if imgui.is_item_hovered():
            imgui.set_tooltip("restore default tile flips/rotations")

    def _camera_number(self, c: int):
        """The camera index of view-combo entry ``c`` from its ``VW{angle}`` label; None for a fused pair."""
        if not 0 <= c < len(self._view_names):
            return None
        name = self._view_names[c]
        if str(name).endswith("_fused"):
            return None
        from mbo_utilities.arrays.isoview.array import camera_from_view_label

        return camera_from_view_label(name)

    def _fused_view(self, c: int):
        """``"VW00"`` or ``"VW90"`` for a fused-pair channel such as ``VW90_VW270_CH00_fused``, else None."""
        if not 0 <= c < len(self._view_names):
            return None
        name = str(self._view_names[c])
        if not name.endswith("_fused"):
            return None
        if name.startswith("VW00"):
            return "VW00"
        if name.startswith("VW90"):
            return "VW90"
        return None

    def _camera_is_mirrored(self, c: int) -> bool:
        """The single (un-fused) opposing-side views are column-mirrored in the
        display — they image the sample from the opposite side. FUSED views
        already combined the opposing pair, so they are NOT mirrored (mirroring a
        fused view double-flips it vs the export).
        """
        if self._fused_view(c) is not None:
            return False
        return self._camera_number(c) in (1, 2)

    def _camera_default_hflip(self, c: int) -> bool:
        """VW90 cameras (CM02, CM03) default to H-flipping their tile_y==0
        tiles to make the beads tile (verified); VW00 gets no default flip.
        """
        return self._camera_number(c) in (2, 3)

    def _camera_default_rot(self, c: int) -> int:
        """Rotated datasets mount every camera 90deg on its side, so every
        tile's content needs a 90deg (CCW) rotation to tile upright. Verified
        on VW00 by bead overlap cross-correlation; VW90 follows the same mount.
        0 for Normal datasets.
        """
        return 1 if self._is_rotated else 0

    def _seed_camera_defaults(self, c: int) -> None:
        """Apply a camera's default per-tile orientation once: a 90deg rotation
        for Rotated datasets, plus the VW90 H-flips. Seeds into ``_tile_rot`` /
        ``_tile_flips`` so the user can still toggle them; re-applied after
        "Reset flips" (which clears the seeded marks).
        """
        if c in self._flip_seeded:
            return
        self._flip_seeded.add(c)
        rot = self._camera_default_rot(c)
        if rot:
            # the single VW90 / VW270 views and the fused VW90 view are h-flipped onto VW00; VW00 is not
            for ti in self._tile_xyz:
                self._tile_rot[(c, ti)] = rot
        is_fused_vw90 = self._fused_view(c) == "VW90"
        if not (self._camera_default_hflip(c) or is_fused_vw90):
            return  # every tile: verified on the v13 export
        if self._is_rotated and is_fused_vw90:
            # one parity column: tile_y == 0 for a column-mirrored single view, 1 for the fused pair
            tiles = list(self._tile_xyz)
        else:
            flip_col = 0 if self._camera_default_hflip(c) else 1
            tiles = [ti for ti, xyz in self._tile_xyz.items() if xyz[1] == flip_col]
        for ti in tiles:
            self._tile_flips.setdefault((c, ti), set()).add("X")

    @staticmethod
    def _export_view_label(name) -> str | None:
        """VW label for a channel name, matching the BigStitcher export's
        per-setup ``VW{angle}`` label (so per-tile orientation keys line up).
        """
        s = str(name)
        if "fused" in s:
            return "VW00" if "VW00" in s else ("VW90" if "VW90" in s else None)
        from mbo_utilities.arrays.isoview.array import (
            camera_from_view_label,
            camera_view_label,
        )

        cam = camera_from_view_label(s)
        return camera_view_label(cam) if cam is not None else None

    def export_tile_orientations(self, arr) -> dict:
        """``{"tx_ty_tz": {view_label: [rot, fx, fy]}}`` per-tile in-plane
        orientation for the BigStitcher export (only non-identity entries).

        ``rot`` = CCW 90deg steps, ``fx``/``fy`` = lateral mirrors — the same
        ops this widget applies to each tile's MIP, so the exported tiles match
        what is shown here. The export bakes them into the per-tile pose as an
        in-plane (rotation axis = world Z) transform.
        """
        # every channel, or a view never switched to (VW90) exports native and mis-stitches
        self._ensure_grid(arr)
        names = list(getattr(arr, "view_names", []) or [])
        for c in range(len(names)):
            self._seed_camera_defaults(c)
        out: dict = {}
        for c, ti in set(self._tile_rot) | set(self._tile_flips):
            rot = self._tile_rot.get((c, ti), 0) % 4
            tf = self._tile_flips.get((c, ti), set())
            fx, fy = "X" in tf, "Y" in tf
            if not rot and not fx and not fy:
                continue
            xyz = self._tile_xyz.get(ti)
            if xyz is None or not (0 <= c < len(names)):
                continue
            vlabel = self._export_view_label(names[c])
            if vlabel is None:
                continue
            gridkey = f"{int(xyz[0])}_{int(xyz[1])}_{int(xyz[2])}"
            out.setdefault(gridkey, {})[vlabel] = [int(rot), int(fx), int(fy)]
        return out

    def _get_layout(self, g) -> dict:
        """Editable ``{(ri, ci): ti}`` for the current (camera, z-block), from
        the stage-derived placement. Keyed per camera so each camera keeps its
        own arrangement and manual moves don't leak across cameras.

        The single (un-fused) opposing-side views are column-mirrored; fused
        views are NOT (see ``_camera_is_mirrored``), so a fused view renders with
        only its per-tile rot/flips, WYSIWYG with the export.
        """
        c = self._c_index
        z = self._zblock
        key = (c, z)
        if key not in self._layout:
            src = g["placed"].get(z, {})
            ncols = max(1, len(g["cols"]))
            mirror = self._camera_is_mirrored(c)
            lay = {}
            for (ri, ci), (ti, spc) in src.items():
                cc = (ncols - 1 - ci) if mirror else ci
                lay[(ri, cc)] = ti
                self._tile_spc[ti] = spc
            self._layout[key] = lay
        return self._layout[key]

    def _swap_cells(self, layout: dict, a: tuple, b: tuple) -> None:
        """Swap the tiles in two cells (move into an empty cell if one side
        is empty).
        """
        if a == b:
            return
        ta, tb = layout.get(a), layout.get(b)
        for cell, t in ((b, ta), (a, tb)):
            if t is None:
                layout.pop(cell, None)
            else:
                layout[cell] = t

    def _overlap_frac(self, arr, stride_key: str, centers: list, extent_um: float):
        """Fraction of a tile that overlaps its neighbor along one display axis.

        ``(extent − stride) / extent`` in µm. Stride from the declared
        ``tile_stride_*`` (cols=tile_y, rows=tile_x); falls back to the spacing
        of the grid's stage-coordinate centers. Returns ``None`` when there's
        no positive overlap or no usable stride.
        """
        meta = getattr(arr, "metadata", {}) or {}
        s = meta.get(stride_key)
        if not s and centers and len(centers) > 1:
            diffs = sorted(
                abs(centers[i + 1] - centers[i]) for i in range(len(centers) - 1)
            )  # above 1 it is a µm spacing, not an index
            d = diffs[len(diffs) // 2]
            s = d if d > 1.0 else None
        if not s or extent_um <= 0 or float(s) >= extent_um:
            return None
        return (extent_um - float(s)) / extent_um

    def _draw_grid(self, arr) -> None:
        g = self._grid
        ncols = max(1, len(g["cols"]))
        nrows = max(1, len(g["rows"]))
        c = self._c_index
        self._seed_camera_defaults(c)
        _s, _sz, pix_xy, _pix_z = self._mip_params(arr)
        # a single plane is much dimmer than a MIP, so the auto range is per mode and plane
        layout = self._get_layout(g)

        plane_key = self._plane if self._view_mode == 1 else -1
        loaded = [
            v
            for k, v in self._thumb_cache.items()
            if k[1] == c and k[2] == self._view_mode and k[3] == plane_key
        ]
        lo, hi = self._contrast_range(arr, loaded, (self._view_mode, plane_key))

        avail = imgui.get_content_region_avail()
        spacing = 4.0
        cell = max(48.0, (avail.x - spacing * (ncols - 1)) / ncols)
        cur_tile = self._current_tile()

        grey = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.25, 0.25, 0.25, 1.0))
        dark = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.08, 0.08, 0.08, 1.0))
        yellow = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.85, 0.2, 1.0))

        imgui.begin_child("##tilegrid_canvas", imgui.ImVec2(0, 0), child_flags=0)
        draw_list = imgui.get_window_draw_list()
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing, imgui.ImVec2(spacing, spacing)
        )
        try:
            for ri in range(nrows):
                for ci in range(ncols):
                    if ci > 0:
                        imgui.same_line()
                    pos = imgui.get_cursor_screen_pos()
                    cmax = imgui.ImVec2(pos.x + cell, pos.y + cell)
                    clicked = imgui.invisible_button(
                        f"##cell_{ri}_{ci}", imgui.ImVec2(cell, cell)
                    )
                    clicked_r = imgui.is_item_clicked(imgui.MouseButton_.right)
                    entry = layout.get((ri, ci))
                    picked = self._pick == (ri, ci)
                    if clicked and self._edit_layout:
                        if self._pick is None:
                            if entry is not None:
                                self._pick = (ri, ci)
                        else:
                            self._swap_cells(layout, self._pick, (ri, ci))
                            self._pick = None
                    elif clicked and entry is not None:
                        self._jump_to_tile(entry)
                    if entry is None:
                        draw_list.add_rect(pos, cmax, yellow if picked else grey)
                        continue
                    ti = entry
                    spc = self._tile_spc.get(ti, ti)
                    okey = (c, ti)
                    tf = self._tile_flips.get(okey, set())
                    fx = "X" in tf
                    fy = "Y" in tf
                    rot = self._tile_rot.get(okey, 0) % 4
                    if clicked_r:
                        # the draw thread only consumes what the prefetch thread decompressed
                        self._orient_menu_ti = ti
                        self._orient_menu_open = True
                    plane_key = self._plane if self._view_mode == 1 else -1
                    gkey = (ti, c, self._view_mode, plane_key, rot, fx, fy)
                    thumb = self._thumb_cache.get(gkey)
                    if thumb is None:
                        if self._view_mode == 1:
                            src = self._cached_source((ti, c, "plane", self._plane))
                        else:
                            src = self._cached_source((ti, c, "xy"))
                        # rotate first, then flips act on the view
                        if src is not None:
                            # Y flip = vertical mirror
                            t = src
                            # X flip = horizontal mirror
                            if rot:
                                t = np.rot90(t, rot)
                            if fy:
                                t = t[::-1, :]
                            if fx:
                                t = t[:, ::-1]
                            thumb = np.ascontiguousarray(t)
                            self._thumb_cache[gkey] = thumb
                            while len(self._thumb_cache) > _THUMB_CACHE_MAX:
                                self._thumb_cache.popitem(last=False)
                    gpu = (
                        self._ensure_gpu(gkey, thumb, lo, hi)
                        # xy MIP: both axes are lateral
                        if thumb is not None
                        else None
                    )
                    if gpu is not None:
                        h, w = thumb.shape
                        psx = psy = pix_xy
                        pw = max(w * psx, 1e-6)
                        ph = max(h * psy, 1e-6)
                        sc = min(cell / pw, cell / ph)
                        dw, dh = pw * sc, ph * sc
                        ix = pos.x + (cell - dw) * 0.5
                        # seam guides with each present neighbour, for the unrotated layout
                        iy = pos.y + (cell - dh) * 0.5
                        draw_list.add_image(
                            gpu.ref,
                            imgui.ImVec2(ix, iy),
                            imgui.ImVec2(ix + dw, iy + dh),
                        )
                        fh = self._overlap_frac(arr, "tile_stride_y", g["cols"], pw)
                        fv = self._overlap_frac(arr, "tile_stride_x", g["rows"], ph)
                        seam = imgui.color_convert_float4_to_u32(
                            imgui.ImVec4(0.2, 0.9, 1.0, 0.55)
                        )
                        if fh is not None:
                            if (ri, ci + 1) in layout:
                                x = ix + dw * (1.0 - fh)
                                draw_list.add_line(
                                    imgui.ImVec2(x, iy),
                                    imgui.ImVec2(x, iy + dh),
                                    seam,
                                    1.0,
                                )
                            if (ri, ci - 1) in layout:
                                x = ix + dw * fh
                                draw_list.add_line(
                                    imgui.ImVec2(x, iy),
                                    imgui.ImVec2(x, iy + dh),
                                    seam,
                                    1.0,
                                )
                        if fv is not None:
                            if (ri + 1, ci) in layout:
                                y = iy + dh * (1.0 - fv)
                                draw_list.add_line(
                                    imgui.ImVec2(ix, y),
                                    imgui.ImVec2(ix + dw, y),
                                    seam,
                                    1.0,
                                )
                            if (ri - 1, ci) in layout:
                                y = iy + dh * fv
                                draw_list.add_line(
                                    imgui.ImVec2(ix, y),
                                    imgui.ImVec2(ix + dw, y),
                                    seam,
                                    1.0,
                                )
                    else:
                        draw_list.add_rect_filled(pos, cmax, dark)
                    label = self._tile_labels.get(ti, f"SPM{spc:02d}")
                    if imgui.is_item_hovered():
                        tip = f"{label}  (tile {ti})"
                        tip += (
                            "\nclick: pick / place    right-click: rotate / flip"
                            if self._edit_layout
                            else "\nright-click: rotate / flip"
                        )
                        imgui.set_tooltip(tip)
                    is_current = ti == cur_tile
                    cyan = imgui.color_convert_float4_to_u32(
                        imgui.ImVec4(0.2, 0.9, 1.0, 1.0)
                    )
                    border = cyan if picked else (yellow if is_current else grey)
                    draw_list.add_rect(
                        pos,
                        cmax,
                        border,
                        thickness=3.0 if (picked or is_current) else 1.0,
                    )
                    self._draw_cell_label(draw_list, pos, label)
                    marks = []
                    if rot:
                        marks.append(f"rot {rot * 90}")
                    if tf:
                        marks.append(
                            "flip "
                            + "".join(
                                m for m, ax in (("H", "X"), ("V", "Y")) if ax in tf
                            )
                        )
                    if marks:
                        self._draw_cell_label(
                            draw_list,
                            imgui.ImVec2(pos.x, pos.y + 15.0),
                            # the popup opens outside the scrolling child so imgui clamps it inside the window
                            "  ".join(marks),
                        )
        finally:
            imgui.pop_style_var(1)
            imgui.end_child()

        if self._orient_menu_open:
            imgui.open_popup("tile_orient_menu")
            self._orient_menu_open = False
        ti = self._orient_menu_ti
        if ti is not None:
            if imgui.begin_popup("tile_orient_menu"):
                self._draw_tile_orient_menu(ti, self._tile_labels.get(ti, f"tile {ti}"))
                imgui.end_popup()
            else:
                self._orient_menu_ti = None

    @staticmethod
    def _toggle_button(label: str, active: bool, w: float) -> bool:
        """Button that stays highlighted while ``active``; returns True on click."""
        with selected_button_style(active):
            return imgui.button(label, imgui.ImVec2(w, 0))

    def _draw_tile_orient_menu(self, ti: int, label: str) -> None:
        """The right-click popup's body: rotate and flip buttons for one tile, two per row, sized from their text."""
        imgui.text_disabled(
            f"Tile {label}  ({self._view_names[self._c_index] if self._c_index < len(self._view_names) else 'view'})"
        )
        imgui.separator()
        style = imgui.get_style()
        sp = style.item_spacing.x
        bw = (
            max(
                imgui.calc_text_size("Rotate CCW").x,
                imgui.calc_text_size("Rotate CW").x,
            )
            + style.frame_padding.x * 2.0
            + 8.0
        )

        okey = (self._c_index, ti)
        if imgui.button("Rotate CCW##ccw", imgui.ImVec2(bw, 0)):
            self._tile_rot[okey] = (self._tile_rot.get(okey, 0) + 1) % 4
        imgui.same_line()
        if imgui.button("Rotate CW##cw", imgui.ImVec2(bw, 0)):
            self._tile_rot[okey] = (self._tile_rot.get(okey, 0) - 1) % 4

        s = self._tile_flips.setdefault(okey, set())
        if self._toggle_button("Flip H##fh", "X" in s, bw):
            (s.discard if "X" in s else s.add)("X")
        imgui.same_line()
        if self._toggle_button("Flip V##fv", "Y" in s, bw):
            (s.discard if "Y" in s else s.add)("Y")

        if imgui.button("Reset##tileorient", imgui.ImVec2(bw * 2 + sp, 0)):
            self._tile_rot[okey] = 0
            s.clear()

    def _draw_cell_label(self, draw_list, pos, label: str) -> None:
        txt = str(label)
        tw = imgui.calc_text_size(txt)
        bg = imgui.color_convert_float4_to_u32(imgui.ImVec4(0, 0, 0, 0.55))
        draw_list.add_rect_filled(
            pos,
            imgui.ImVec2(pos.x + tw.x + 4, pos.y + tw.y + 2),
            bg,
        )
        draw_list.add_text(
            imgui.ImVec2(pos.x + 2, pos.y + 1),
            imgui.color_convert_float4_to_u32(imgui.ImVec4(*_WHITE)),
            txt,
        )

    def _draw_popup(self, arr) -> None:
        center_popup_on_open(default_em=(64.0, 60.0), min_em=(40.0, 30.0))
        opened, self._popup_open = imgui.begin(
            "Tile Grid###tilegrid_popup",
            self._popup_open,
            flags=imgui.WindowFlags_.no_saved_settings,
        )
        if not opened:
            imgui.end()
            return
        self._draw_toolbar()
        imgui.separator()
        self._draw_grid(arr)
        imgui.end()

    def cleanup(self) -> None:
        self._prefetch_stop.set()
        self._prefetch_wake.set()
        for t in self._prefetch_threads:
            t.join(timeout=1.0)
        self._prefetch_threads = []
        self._reset_caches()
