"""Isoview pipeline widget (Consolidate / Correct / Fuse).

Three modes share one widget, gated by the loaded ``IsoviewArray.kind``
and the presence of the optional ``isoview`` package:

- **Consolidate** — collapse a corrected or fused output tree into one
  OME-Zarr group via :func:`mbo_utilities.arrays.isoview.consolidate_isoview`.
  Always available; in-tree, no extra dependency.
- **Correct** — runs the IsoView ``correct_stack`` pipeline against raw
  ``.stack`` data (pixel correction + segmentation + per-camera output).
  Requires ``pip install isoview``.
- **Fuse** — runs the IsoView ``multi_fuse`` pipeline against a corrected
  tree (camera-pair registration + blending). Requires ``pip install isoview``.

Side-panel layout matches Suite2p: dataset summary + output path + mode
picker + small Settings button + a big green Run button. All other
parameters live in a popup formatted like Suite2p's Pipeline Settings.
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil
from imgui_bundle import hello_imgui, imgui

from mbo_utilities.arrays.features._slicing import parse_timepoint_selection
from mbo_utilities.gui._imgui_helpers import (
    PopupAutoSize,
    draw_boxed_label,
    selected_button_style,
    set_tooltip,
    text_wrapped_cell,
    tooltip_marks_right,
)
from mbo_utilities.gui._selection_ui import draw_selection_table, resolve_dim_labels
from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.gui.widgets.pipelines.settings import (
    _dataset_size_bytes,
    _draw_dataset_files_popup,
    _format_size,
)

_TITLE_COLOR = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)
_SUBSECTION_COLOR = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)

_RUN_W = 220
_BTN_W = 90

_MAX_WORKERS = (
    16  # RAM, not cores, is the limit: a worker holds a whole camera-pair volume
)

# multiples of one raw camera volume: correction holds the stack, a float32 copy and masks; fusion two volumes, two masks and the output
_CORRECT_RAM_PER_WORKER_X = 5.0
_FUSE_RAM_PER_WORKER_X = 6.0


def _iso_volume_gb(arr: Any) -> float:
    """One camera-volume size in GiB from an input array's last 3 dims (Z,Y,X)."""
    shp = tuple(int(s) for s in arr.shape)
    if len(shp) < 3:
        return 0.0
    return (shp[-3] * shp[-2] * shp[-1] * arr.dtype.itemsize) / (1024**3)


def _default_workers() -> int:
    """Conservative default: half the logical CPUs, capped at 4."""
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu // 2))


def _input_w() -> float:
    """Default width for numeric inputs in the popup."""
    return hello_imgui.em_size(8)


_ZARR_CODECS = ["zstd", "gzip", "blosc-zstd", "blosc-lz4", "none"]
_TIFF_CODECS = [
    "deflate",
    "zstd",
    "lzw",
    "lzma",
    "none",
]  # NOTE: Bio-Formats cannot read zstd TIFFs, so deflate leads


def _codecs_for_format(fmt: str) -> list[str]:
    return _TIFF_CODECS if fmt == "tif" else _ZARR_CODECS


def _default_codec_for_format(fmt: str) -> str:
    return "deflate" if fmt == "tif" else "zstd"


_WORST_ROW_LABEL = "Compression level"


def _natural_col_w() -> float:
    """Min width per sub-section box: widest input row + (?) + paddings."""
    style = imgui.get_style()
    spacing_x = style.item_spacing.x
    frame_pad_x = style.frame_padding.x
    qm_w = imgui.calc_text_size("(?)").x
    label_w = imgui.calc_text_size(_WORST_ROW_LABEL).x
    body = _input_w() + spacing_x + label_w + spacing_x + qm_w
    inner_pad = 2 * frame_pad_x + 4
    return body + inner_pad + 8


def _hint(text: str) -> None:
    """Wrapped subtitle/help text in the same blue Suite2p uses."""
    imgui.push_text_wrap_pos(0.0)
    imgui.text_colored(_SUBSECTION_COLOR, text)
    imgui.pop_text_wrap_pos()


@contextmanager
def _green_button():
    """Push the Suite2p green-Run-button color scheme (idle/hover/active)."""
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
    imgui.push_style_color(
        imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0)
    )
    imgui.push_style_color(
        imgui.Col_.button_active, imgui.ImVec4(0.10, 0.45, 0.10, 1.0)
    )
    try:
        yield
    finally:
        imgui.pop_style_color(3)


_MODE_CONSOLIDATE = "Consolidate"
_MODE_CORRECT = "Correct"
_MODE_FUSE = "Fuse"
_MODE_STITCHER = "BigStitcher XML"

_RUN_LABELS = {
    _MODE_CONSOLIDATE: "Run consolidation",
    _MODE_CORRECT: "Run correct_stack",
    _MODE_FUSE: "Run multi_fuse",
    _MODE_STITCHER: "Export XML",
}

_MODE_PREFIX = {
    _MODE_CORRECT: ".corrected",
    _MODE_FUSE: ".fused",
    _MODE_STITCHER: ".stitcher",
}


# NOTE: CM01..CM03 onto CM00 for a normal mount, verified in the align widget
_CM_ALIGN_DEFAULT: dict[int, dict] = {
    1: {"rotations": [], "flips": ["X"]},
    2: {"rotations": [{"sign": "+", "axis": "Y", "deg": 90}], "flips": []},
    3: {"rotations": [{"sign": "+", "axis": "Y", "deg": 90}], "flips": ["Z"]},
}

# NOTE: every camera mounted 90 deg on its side; CM03 = rot X +90 then flip Y equals rot X -90 + flip Z
_CM_ALIGN_DEFAULT_ROTATED: dict[int, dict] = {
    1: {"rotations": [], "flips": ["Y"]},
    2: {"rotations": [{"sign": "-", "axis": "X", "deg": 90}], "flips": []},
    3: {"rotations": [{"sign": "+", "axis": "X", "deg": 90}], "flips": ["Y"]},
}


def _norm_variant(text: str) -> str:
    """Normalize a user variant: '' -> '' ; 'x' or '_x' -> '_x'."""
    s = (text or "").lstrip("_")
    return f"_{s}" if s else ""


_ISOVIEW_IMPORTANT_FIELDS: set[str] = {
    "segment_threshold",
    "blending_method",
    "blending_range",
    "transition_plane",
    "front_flag",
}


def _input_zyx(header: str, ident: str, z: int, y: int, x: int, tip: str) -> list[int]:
    imgui.text(header)
    imgui.set_next_item_width(_input_w())
    _, vals = imgui.input_int3(ident, [int(z), int(y), int(x)])
    set_tooltip(tip)
    return [max(0, int(v)) for v in vals]


def _transition_plane(val: int) -> int | None:
    return None if int(val) < 0 else int(val)


def _unwrap_array(arr: Any) -> Any:
    """Peel display wrappers (`_SqueezeSingletonDims`, `_ScrubTimingProxy`)
    so we can isinstance-check the underlying IsoviewArray.
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


def _isoview_pkg_available() -> bool:
    """True iff the upstream ``isoview`` package is importable.

    Checked via find_spec so the test stays cheap. Required by the
    Correct and Fuse modes; Consolidate works without it.
    """
    return importlib.util.find_spec("isoview") is not None


def maybe_spawn_raw_projections(parent: Any) -> None:
    """Precompute raw XY projections in a background worker when a raw
    isoview dataset is loaded, so the segmentation / dead-pixel previews
    have images to show.

    Idempotent per (session, dataset): spawns at most once per raw dir and
    not while a worker for that dir is already running. The worker itself
    skips projections already on disk, so re-runs are cheap. Does not need
    the upstream ``isoview`` package.
    """
    iw = parent.image_widget
    if iw is None or not iw.data:
        return
    arr = _unwrap_array(iw.data[0])
    from mbo_utilities.arrays.isoview import IsoviewArray

    if not isinstance(arr, IsoviewArray) or arr.kind != "raw":
        return
    raw_dir = str(arr.scan_root)
    if raw_dir in _RAW_PROJ_SPAWNED:
        return

    from mbo_utilities.gui.widgets.process_manager import get_process_manager

    pm = get_process_manager()
    for p in pm.get_all():
        if (
            p.task_type == "isoview_raw_projections"
            and (p.args or {}).get("raw_dir") == raw_dir
            and p.is_alive()
        ):
            _RAW_PROJ_SPAWNED.add(raw_dir)
            return

    pid = pm.spawn(
        task_type="isoview_raw_projections",
        args={"raw_dir": raw_dir},
        description=f"Raw projections: {Path(raw_dir).name}",
        output_path=raw_dir,
    )
    if pid:
        _RAW_PROJ_SPAWNED.add(raw_dir)
        parent.logger.info(f"raw projections worker spawned PID {pid}")


def maybe_refresh_raw_projections(parent: Any) -> None:
    """Rebuild the preview widget list once a background raw-projections
    worker finishes, so the Projections widget appears without reloading.

    The widget list is built only on dataset load; a worker that writes
    projections afterwards leaves :class:`ProjectionsViewer` excluded
    until something refreshes. Runs once per raw dir.
    """
    pending = _RAW_PROJ_SPAWNED - _RAW_PROJ_REFRESHED
    if not pending:
        return

    from mbo_utilities.gui.widgets.process_manager import get_process_manager

    pm = get_process_manager()
    procs = pm.get_all()
    for raw_dir in list(pending):
        alive = any(
            p.task_type == "isoview_raw_projections"
            and (p.args or {}).get("raw_dir") == raw_dir
            and p.is_alive()
            for p in procs
        )
        if alive:
            continue
        _RAW_PROJ_REFRESHED.add(raw_dir)
        parent._refresh_widgets()


_RAW_PROJ_SPAWNED: set[str] = set()
_RAW_PROJ_REFRESHED: set[str] = set()


def _available_modes(arr: Any) -> list[str]:
    """Modes the widget can offer for ``arr``.

    Filtered by both ``kind`` and whether the upstream ``isoview``
    package is installed (Correct/Fuse need it; Consolidate doesn't).
    """
    if arr is None:
        return []
    modes: list[str] = []
    has_pkg = _isoview_pkg_available()
    if arr.kind == "raw":
        if has_pkg:
            modes.append(_MODE_CORRECT)
            modes.append(_MODE_STITCHER)
    elif arr.kind == "corrected":
        if has_pkg:
            modes.append(_MODE_FUSE)
            modes.append(_MODE_STITCHER)
        modes.append(_MODE_CONSOLIDATE)
    elif arr.kind == "fused":
        if has_pkg:
            modes.append(_MODE_STITCHER)
        modes.append(_MODE_CONSOLIDATE)
    return modes


class IsoviewPipelineWidget(PipelineWidget):
    """Run IsoView pipelines (correct / fuse / consolidate)."""

    name = "Isoview"
    install_command = "uv pip install isoview"
    is_available = True

    @classmethod
    def applies_to(cls, arr: Any) -> bool:
        if arr is None:
            return False
        from mbo_utilities.arrays.isoview import IsoviewArray

        underlying = _unwrap_array(arr)
        return isinstance(underlying, IsoviewArray) and underlying.kind in (
            "raw",
            "corrected",
            "fused",
        )

    def __init__(self, parent: Any):
        super().__init__(parent)

        self._selected_mode: str | None = None
        self._initialized_for_scan_root: Path | None = None
        self._last_status: str = ""

        self._output_dir: str = ""
        self._output_format: str = "zarr"
        self._compression: str = "zstd"
        self._compression_level: int = 3
        self._overwrite: bool = False
        self._workers: int = _default_workers()
        self._pyramid: bool = True
        self._pyramid_max_layers: int = 4

        self._zarr_sharded: bool = True
        self._zarr_chunk_z: int = 0
        self._zarr_chunk_y: int = 0
        self._zarr_chunk_x: int = 0
        self._zarr_shard_z: int = 0
        self._zarr_shard_y: int = 0
        self._zarr_shard_x: int = 0

        self._consolidate_output_path: str = ""
        self._consolidate_pyramid: bool = True
        self._consolidate_pyramid_max_layers: int = 4
        self._consolidate_compressor: str = "zstd"
        self._consolidate_compression_level: int = 3
        self._consolidate_overwrite: bool = False

        self._stitcher_suffix: str = ""

        self._stitcher_bake_tile_positions: bool = True

        self._stitcher_reverse_z: bool = True

        self._stitcher_link_existing: bool = True

        self._stitcher_cameras: str = ""

        self._stitcher_zarr_version: int = 3

        self._stitcher_orient: dict[str, dict] = {}
        self._stitcher_orient_target: str = "VW00"
        self._stitcher_orient_profile: str = "default"

        self._fuse_output_suffix: str = ""

        self._consolidate_suffix: str = "_isoview-corrected.zarr"

        self._correct_output_suffix: str = ""
        self._correct_segment_mode: int = 1
        self._correct_apply_seg_mask: bool = False
        self._correct_do_tenengrad: bool = False
        self._correct_background_percentile: float = 5.0
        self._correct_median_kernel_size: int = 3
        self._correct_median_kernel_enabled: bool = True
        self._correct_subtract_background: bool = False
        self._correct_mask_percentile: float = 1.0
        self._correct_subsample_factor: int = 100
        self._correct_gauss_kernel: int = 5
        self._correct_gauss_sigma: float = 2.0
        self._correct_segment_threshold: float = 0.4
        self._correct_splitting: int = 10

        self._fuse_output_suffix: str = ""
        self._fuse_view_params: dict[int, dict] = {}
        self._fuse_view_ids: list[int] = []
        self._fuse_active_view: int | None = None

        self._fuse_subtract_background: bool = True
        self._fuse_background_mode: str = "pooled"
        self._fuse_background_percentile: float = 5.0

        self._mic_overrides_enabled: bool = False
        self._mic_pixel_spacing_z: float = -1.0
        self._mic_objective_mag: float = -1.0
        self._mic_pixel_spacing_camera: float = -1.0

        self._show_settings_popup: bool = False
        self._popup_just_opened: bool = False
        self._reopen_settings_after_crop: bool = False
        self._crop_was_open: bool = False

        self._iso_slicing_open: bool = False
        self._iso_last_max_tp: int = 0
        self._iso_last_fpath: str = ""
        self._iso_tp_selection = ""
        self._iso_tp_error = ""
        self._iso_tp_parsed = None
        self._iso_settings_sizer: PopupAutoSize | None = None
        self._iso_vol_gb = 0.0

        self._iso_selected_cameras: set[int] | None = None

        self._iso_files_sizer: PopupAutoSize | None = None

    def _get_array(self) -> Any | None:
        iw = self.parent.image_widget
        if iw is None or not iw.data:
            return None
        return _unwrap_array(iw.data[0])

    @staticmethod
    def _default_fuse_view_params() -> dict:
        """Per-view defaults that seed the Fuse editor for a new dataset.

        Mirrors the prior scalar defaults so behavior is unchanged when
        a user never touches the View selector.
        """
        return {
            "blending_method": "geometric",
            "blending_range": 4,
            "transition_plane": -1,
            "front_flag": 1,
            "flip_z": False,
            "flip_horizontal": True,
            "flip_vertical": False,
            "rotation": 0,
            "search_x_start": -50,
            "search_x_stop": 50,
            "search_x_step": 10,
            "search_y_start": -50,
            "search_y_stop": 50,
            "search_y_step": 10,
        }

    def _fv(self, name: str):
        """Read ``name`` from the active view's Fuse params dict."""
        view = self._fuse_active_view
        if view is None or view not in self._fuse_view_params:
            return self._default_fuse_view_params()[name]
        return self._fuse_view_params[view][name]

    def _set_fv(self, name: str, value) -> None:
        """Write ``name`` into the active view's Fuse params dict."""
        view = self._fuse_active_view
        if view is None:
            return
        params = self._fuse_view_params.setdefault(
            view,
            self._default_fuse_view_params(),
        )
        params[name] = value

    def _ensure_defaults(self, arr: Any) -> None:
        """Compute per-dataset defaults once. Re-runs when the loaded
        IsoviewArray's scan_root changes (different dataset opened).
        """
        sr = Path(arr.scan_root)
        if self._initialized_for_scan_root == sr:
            return
        self._initialized_for_scan_root = sr

        meta = arr.metadata
        cv = meta.get("camera_view_map") or {0: 0, 1: 0, 2: 90, 3: 90}
        view_ids = sorted({int(v) for v in cv.values()}) or [0, 90]
        self._fuse_view_ids = view_ids
        self._fuse_view_params = {
            vid: self._default_fuse_view_params() for vid in view_ids
        }
        if self._metadata_orient_profile(arr) == "rotated":
            for _p in self._fuse_view_params.values():
                _p["flip_horizontal"] = False
                _p["flip_vertical"] = (
                    True  # NOTE: a side mount mirrors the opposing camera about Y; bead MIP cross-correlation 0.29 for flip_v, ~0 for flip_h
                )
        if self._fuse_active_view is None or self._fuse_active_view not in view_ids:
            self._fuse_active_view = view_ids[0]

        self._consolidate_suffix = f"_isoview-{arr.kind}.zarr"
        self._consolidate_output_path = str(
            sr.parent / f"{sr.name}{self._consolidate_suffix}"
        )

        if arr.kind == "corrected":
            self._output_dir = str(sr.parent)
        else:
            self._output_dir = ""  # NOTE: isoview derives input_dir + .corrected<suffix> from an empty path; a set path freezes the suffix

        self._apply_orient_profile_all(self._metadata_orient_profile(arr))

        modes = _available_modes(arr)
        if self._selected_mode not in modes:
            if arr.kind == "corrected" and _MODE_FUSE in modes:
                self._selected_mode = _MODE_FUSE
            elif modes:
                self._selected_mode = modes[0]
            else:
                self._selected_mode = None

    def _output_path_attr(self) -> str:
        """The instance attribute holding the active mode's output path."""
        if self._selected_mode == _MODE_CONSOLIDATE:
            return "_consolidate_output_path"
        if self._selected_mode == _MODE_STITCHER:
            return "_stitcher_output_path"
        return "_output_dir"

    @staticmethod
    def _iso_input_tree(arr: Any) -> Path | None:
        """The .corrected*/.fused* tree the array was loaded from.

        Matches custom suffixes (e.g. ``.fused_v2``) by accepting a
        ``.corrected``/``.fused`` token followed by end-of-name or ``_``.
        """
        scan_root = Path(arr.scan_root)
        for ancestor in (scan_root, *scan_root.parents):
            n = ancestor.name
            for suf in (".corrected", ".fused"):
                idx = n.find(suf)
                if idx >= 0:
                    rest = n[idx + len(suf) :]
                    if rest == "" or rest.startswith("_"):
                        return ancestor
        return None

    @staticmethod
    def _iso_raw_stem(tree: Path) -> str:
        """Raw dataset stem of a tree dir, e.g. ``zebrafish.fused_v2`` -> ``zebrafish``."""
        from mbo_utilities.arrays.isoview.array import (
            _CORRECTED_TAIL_RE,
            _FUSED_TAIL_RE,
        )

        stem = tree.name
        for rx in (_CORRECTED_TAIL_RE, _FUSED_TAIL_RE):
            m = rx.search(stem)
            if m:
                return stem[: m.start()]
        return stem

    @staticmethod
    def _iso_tree_suffix(tree: Path) -> str:
        """Variant suffix carried by a tree dir, e.g. ``zebrafish.fused_v2`` -> ``v2``."""
        n = tree.name
        for pre in (".corrected", ".fused"):
            idx = n.find(pre)
            if idx >= 0:
                return n[idx + len(pre) :].lstrip("_")
        return ""

    def _iso_stitcher_dest(self, arr: Any) -> Path | None:
        """BigStitcher output dir = ``<rawstem>.stitcher[_<variant>]``, where
        the variant is the Output-options field (empty -> bare ``.stitcher``).
        Matches isoview's ``derive_output_name(raw, ".stitcher", variant)``.
        Works without the raw root.
        """
        if arr.kind == "raw":
            tree = Path(arr.scan_root)
        else:
            tree = self._iso_input_tree(arr)
        if tree is None:
            return None
        stem = self._iso_raw_stem(tree)
        return tree.parent / f"{stem}.stitcher{_norm_variant(self._stitcher_suffix)}"

    def _current_output_path(self) -> str:
        if self._selected_mode == _MODE_STITCHER:
            arr = self._get_array()
            dest = self._iso_stitcher_dest(arr) if arr is not None else None
            return str(dest) if dest is not None else ""
        explicit = getattr(self, self._output_path_attr(), "") or ""
        if explicit:
            return explicit
        if self._selected_mode == _MODE_CORRECT:
            arr = self._get_array()
            sr = arr.scan_root if arr is not None else None
            if sr is not None:
                tail = ".corrected" + _norm_variant(self._correct_output_suffix)
                return f"{Path(sr).resolve()}{tail}"
        return ""

    def _spawn(
        self, task_type: str, args: dict, description: str, output_path: str | None
    ) -> None:
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        pm = get_process_manager()
        pid = pm.spawn(
            task_type=task_type,
            args=args,
            description=description,
            output_path=output_path,
        )
        if pid:
            self._last_status = f"Started (PID {pid})"
            self.parent.logger.info(f"{task_type} spawned PID {pid}")
        else:
            self._last_status = "Failed to spawn worker."

    def _draw_settings_popup(self, arr: Any) -> None:
        """Full pipeline-settings popup, formatted like Suite2p's.

        Layout: two rows of bordered child boxes, each row sized to the
        widest natural column. Row contents depend on the active mode:

        - Consolidate: Output options + Microscope overrides.
        - Correct: Output options + Segmentation + Pixel correction
          (advanced) + Microscope overrides.
        - Fuse: Output options + Fusion + Registration search +
          Microscope overrides.

        Edits are live; the popup just hides controls until needed.
        """
        if not self._show_settings_popup:
            return

        self._iso_vol_gb = _iso_volume_gb(arr)

        popup_title = "Isoview Pipeline Settings##isoview_pipeline_settings_popup"
        viewport = imgui.get_main_viewport()
        _vp_size = viewport.size

        if self._iso_settings_sizer is None:
            self._iso_settings_sizer = PopupAutoSize(popup_title)
        self._iso_settings_sizer.before_open()  # NOTE: right before begin_popup_modal; begin_child in between would consume the buffered window pos

        if self._popup_just_opened:
            self._popup_just_opened = False

        _MAX_COLS = 3
        style = imgui.get_style()
        spacing_x = style.item_spacing.x
        window_pad_x = style.window_padding.x
        col_floor = _natural_col_w()
        min_popup_w = (
            _MAX_COLS * col_floor + spacing_x * (_MAX_COLS - 1) + 2 * window_pad_x + 16
        )
        min_popup_w = min(min_popup_w, _vp_size.x * 0.98)
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(min_popup_w, 300.0),
            imgui.ImVec2(_vp_size.x * 0.98, _vp_size.y * 0.98),
        )

        opened, visible = imgui.begin_popup_modal(
            popup_title,
            p_open=True,
            flags=self._iso_settings_sizer.flags(imgui.WindowFlags_.no_saved_settings),
        )
        if not opened:
            return
        try:
            if not visible:
                self._show_settings_popup = False
                imgui.close_current_popup()
                return

            mode = self._selected_mode
            imgui.text_colored(
                _SUBSECTION_COLOR,
                f"Mode: {mode}    Dataset: {Path(arr.base_path).name}",
            )
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if mode == _MODE_CONSOLIDATE:
                self._draw_consolidate_popup_rows()
            elif mode == _MODE_CORRECT:
                self._draw_correct_popup_rows()
            elif mode == _MODE_FUSE:
                self._draw_fuse_popup_rows()
            elif mode == _MODE_STITCHER:
                self._draw_stitcher_popup_rows()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            if imgui.button("Close##iso_settings_close", imgui.ImVec2(_BTN_W, 0)):
                self._show_settings_popup = False
                imgui.close_current_popup()
        finally:
            imgui.end_popup()

    def _draw_popup_columns(
        self,
        columns: list[tuple[str, Any] | tuple[str, Any, bool]],
    ) -> None:
        """Render a row of equal-width bordered child boxes.

        ``columns`` is a list of ``(title, draw_fn)`` or
        ``(title, draw_fn, collapsing)``. Each box gets a yellow
        ``_TITLE_COLOR`` header, matching Suite2p's pipeline popup. When
        ``collapsing=True`` the title renders as a collapsing header
        (collapsed by default each open) — same idiom Suite2p uses for
        its row-2 advanced columns. Boxes auto-resize their height;
        widths split the row evenly.
        """
        if not columns:
            return
        n = len(columns)
        avail_x = imgui.get_content_region_avail().x
        style = imgui.get_style()
        spacing_x = style.item_spacing.x
        col_w = max(_natural_col_w(), (avail_x - spacing_x * (n - 1)) / n)
        box_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
        for i, col in enumerate(columns):
            if len(col) == 3:
                title, draw_fn, collapsing = col
            else:
                title, draw_fn = col
                collapsing = False
            if i > 0:
                imgui.same_line()
            imgui.begin_child(
                f"##iso_col_{title}",
                imgui.ImVec2(col_w, 0),
                box_flags,
            )
            try:
                if collapsing:
                    imgui.set_next_item_open(False, imgui.Cond_.appearing)
                    imgui.push_style_color(imgui.Col_.text, _TITLE_COLOR)
                    expanded = imgui.collapsing_header(f"{title}##iso_col_hdr_{title}")
                    imgui.pop_style_color()
                    if expanded:
                        imgui.spacing()
                        draw_fn()
                else:
                    imgui.text_colored(_TITLE_COLOR, title)
                    imgui.spacing()
                    draw_fn()
            finally:
                imgui.end_child()

    def _draw_consolidate_popup_rows(self) -> None:
        self._draw_popup_columns(
            [
                ("I/O options", self._draw_consolidate_io_box),
                ("Codec", self._draw_consolidate_codec_box),
                (
                    "Acquisition (override XML)",
                    self._draw_microscope_overrides_box,
                    True,
                ),
            ]
        )

    def _draw_correct_popup_rows(self) -> None:
        self._draw_popup_columns(
            [
                ("I/O options", self._draw_correct_io_box),
                ("Segmentation", self._draw_correct_segmentation_box),
                ("Dead pixel correction", self._draw_correct_advanced_box),
            ]
        )
        imgui.spacing()
        self._draw_popup_columns(
            [
                (
                    "Acquisition (override XML)",
                    self._draw_microscope_overrides_box,
                    True,
                ),
            ]
        )

    def _draw_fuse_popup_rows(self) -> None:
        self._draw_fuse_view_selector()
        self._draw_popup_columns(
            [
                ("I/O options", self._draw_fuse_io_box),
                ("Fusion", self._draw_fuse_fusion_box),
                ("Background", self._draw_fuse_background_box),
            ]
        )
        imgui.spacing()
        self._draw_popup_columns(
            [
                ("Registration search", self._draw_fuse_search_box, True),
                (
                    "Acquisition (override XML)",
                    self._draw_microscope_overrides_box,
                    True,
                ),
            ]
        )

    def _draw_fuse_view_selector(self) -> None:
        """Per-view editor switcher.

        Only the boxes tagged "Target view: VW##" follow this combo — Fusion,
        View transforms, and Registration search. I/O options and
        Microscope overrides are dataset-wide.
        """
        if not self._fuse_view_ids:
            return
        labels = [f"VW{v:02d}" for v in self._fuse_view_ids]
        if self._fuse_active_view not in self._fuse_view_ids:
            self._fuse_active_view = self._fuse_view_ids[0]
        idx = self._fuse_view_ids.index(self._fuse_active_view)
        imgui.text_colored(_SUBSECTION_COLOR, "Reference view:")
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        imgui.set_next_item_width(_input_w())
        with tooltip_marks_right():
            changed, new_idx = imgui.combo("##fuse_view_selector", idx, labels)
            if changed:
                self._fuse_active_view = self._fuse_view_ids[new_idx]
            set_tooltip(
                "Each view (camera pair) carries its own blending, "
                "transforms, and registration-search params.\n"
            )
        imgui.spacing()

    def _draw_view_scope_badge(self, scope: str) -> None:
        """One-line scope hint at the top of a Fuse-mode column box.

        ``scope`` is ``"per_view"`` (binds to the View selector),
        ``"all_views"`` (edits all views in one panel), or ``"shared"``
        (dataset-wide, unaffected by the selector).
        """
        if scope == "per_view":
            view = self._fuse_active_view
            if view is not None:
                text = f"Target view: VW{(view + 180) % 360:02d}"
            else:
                text = "per-view (no view loaded)"
            color = _SUBSECTION_COLOR
        elif scope == "all_views":
            text = "edits all views"
            color = imgui.ImVec4(0.7, 0.7, 0.7, 1.0)
        else:
            text = "shared across views"
            color = imgui.ImVec4(0.7, 0.7, 0.7, 1.0)
        imgui.text_colored(color, text)
        imgui.spacing()

    def _draw_stitcher_popup_rows(self) -> None:
        """BigStitcher Parameters popup, laid out in Suite2p-style boxes."""
        self._draw_popup_columns(
            [
                ("Orientation", self._draw_stitcher_orientation_box),
                ("View transforms", self._draw_stitcher_transforms_box),
                (
                    "Acquisition (override XML)",
                    self._draw_microscope_overrides_box,
                    True,
                ),
            ]
        )

    def _orient_toggle(self, label: str, active: bool, width: float = 0.0) -> bool:
        """Highlighted selection button; returns True when clicked."""
        with selected_button_style(active):
            return imgui.button(label, imgui.ImVec2(width, 0))

    def _orient_target_state(self, target: str | None = None) -> dict:
        """Editable rotations/flips for one orientation target (lazy-init)."""
        target = target or self._stitcher_orient_target
        return self._stitcher_orient.setdefault(target, {"rotations": [], "flips": []})

    def _parse_stitcher_cameras(self) -> list:
        """Camera indices from the Cameras field (e.g. '0,2' -> [0, 2])."""
        return [
            int(c)
            for c in self._stitcher_cameras.replace(",", " ").split()
            if c.strip().isdigit()
        ]

    def _is_per_camera_export(self) -> bool:
        """True when the export is per-camera (raw, corrected with a Cameras
        filter, or corrected linked in place) rather than fused VW00/VW90.
        """
        arr = self._get_array()
        kind = arr.kind if arr is not None else None
        if kind == "raw":
            return True
        if kind == "corrected":
            return bool(self._parse_stitcher_cameras()) or self._stitcher_link_existing
        return False

    def _orient_targets(self) -> list:
        """Orientation targets. Per-camera (raw/.corrected): the views
        themselves (VW00/VW90/VW180/VW270, or the Cameras filter). Fused: the
        fused views.
        """
        if self._is_per_camera_export():
            from mbo_utilities.arrays.isoview.array import camera_view_label

            cams = self._parse_stitcher_cameras()
            if not cams:
                cams = [0, 1, 2, 3]
            return [camera_view_label(c) for c in sorted(set(cams))]
        return ["VW00", "VW90"]

    def _metadata_orient_profile(self, arr: Any) -> str:
        """Default orientation profile from the XML camera_orientation:
        Normal -> 'default', Rotated -> 'rotated'. When the field is absent,
        per-camera falls back to the Normal CM->CM00 alignment (prior
        behaviour) and fused to 'none'.
        """
        meta = arr.metadata
        co = str(meta.get("camera_orientation") or "").strip().lower()
        if not co:
            for v in arr.camera_metadata.values():
                c = str((v or {}).get("camera_orientation") or "").strip().lower()
                if c:
                    co = c
                    break
        if co == "normal":
            return "default"
        if co == "rotated":
            return "rotated"
        return "default" if self._is_per_camera_export() else "none"

    def _apply_orient_profile_all(self, name: str) -> None:
        """Seed the export orientation from the verified CM->CM00 alignment.

        ONE table for both export shapes. Per-camera seeds all four cameras
        (0-3 = VW00/VW180/VW90/VW270); fused seeds only the two it has
        (cameras 0 and 2 = VW00, VW90) from the SAME table — fused is the
        per-camera alignment minus the two opposing views. CM00 (VW00) is the
        reference and has no table entry (identity). ``none`` clears all. The
        rotated upright Z-90 is applied at export time (views.py) for both.
        """
        from mbo_utilities.arrays.isoview.array import camera_view_label

        self._stitcher_orient_profile = name
        table = {
            "default": _CM_ALIGN_DEFAULT,
            "rotated": _CM_ALIGN_DEFAULT_ROTATED,
        }.get(name)
        cams = range(4) if self._is_per_camera_export() else (0, 2)
        for t in ("VW00", "VW90", "VW180", "VW270"):
            st = self._orient_target_state(t)
            st["rotations"], st["flips"] = [], []
        for cam in cams:
            seed = table.get(cam) if table else None
            if not seed:
                continue
            st = self._orient_target_state(camera_view_label(cam))
            st["rotations"] = [dict(r) for r in seed["rotations"]]
            st["flips"] = list(seed["flips"])

    def _toggle_button_row(self, id_prefix, items, is_active, on_click) -> None:
        """Selectable toggle buttons that wrap to the box width (no clipping).

        ``items`` is a sequence of ``(key, label)``. Buttons flow left to
        right and wrap to the next line when the next one would overrun the
        box's content region, so a row never clips its last entry.
        """
        style = imgui.get_style()
        right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        for i, (key, label) in enumerate(items):
            w = imgui.calc_text_size(label).x + 2 * style.frame_padding.x
            if i > 0:
                nxt = imgui.get_item_rect_max().x + style.item_spacing.x + w
                if nxt <= right:
                    imgui.same_line()
            if self._orient_toggle(f"{label}##{id_prefix}_{key}", is_active(key)):
                on_click(key)

    def _toggle_flip(self, st: dict, axis: str) -> None:
        if axis in st["flips"]:
            st["flips"].remove(axis)
        else:
            st["flips"].append(axis)

    def _compose_orientation_ops(self, target: str | None = None) -> list:
        """Orientation op list for one target.

        An orientation committed via the Align views Apply button overrides
        the pipeline default. Rotations (row order, innermost first) then
        flips. Rotation -> ["rot", axis, signed degrees]; flip -> ["flip", axis].
        """
        target = target or self._stitcher_orient_target
        from mbo_utilities.gui import _isoview_orient_state as orient_state

        applied = orient_state.applied_ops(self._get_array(), target)
        if applied is not None:
            return applied
        st = self._orient_target_state(target)
        ops: list = []
        for rot in st["rotations"]:
            deg = int(rot["deg"])
            if rot["sign"] == "-":
                deg = -deg
            ops.append(["rot", rot["axis"], deg])
        for axis in st["flips"]:
            ops.append(["flip", axis])
        return ops

    def _draw_stitcher_orientation_box(self) -> None:
        """Reorientation for the export.

        Pick a global Profile (acquisition mounting) that cascades to every
        target, then optionally select a target and fine-tune its rotations
        + flips. Baked as an affine; BigStitcher interpolates it.
        """
        per_camera = self._is_per_camera_export()
        targets = self._orient_targets()
        if self._stitcher_orient_target not in targets:
            self._stitcher_orient_target = targets[0] if targets else "VW00"

        imgui.text("Profile")
        self._toggle_button_row(
            "orient_profile",
            (("none", "None"), ("default", "Normal"), ("rotated", "Rotated")),
            lambda name: self._stitcher_orient_profile == name,
            lambda name: self._apply_orient_profile_all(name),
        )
        if per_camera:
            _hint(
                "Normal/Rotated align VW90/180/270 onto VW00; None clears "
                "them. Applies to every view."
            )
        else:
            _hint("Seeds VW00; set VW90's ~90 below. Applies to the views.")
        imgui.spacing()

        imgui.text("Target")
        self._toggle_button_row(
            "orient_target",
            tuple((t, t) for t in targets),
            lambda t: self._stitcher_orient_target == t,
            lambda t: setattr(self, "_stitcher_orient_target", t),
        )
        imgui.spacing()

        cur = self._stitcher_orient_target
        st = self._orient_target_state(cur)

        imgui.text("Rotations")
        axes = ["X", "Y", "Z"]
        degs = ["90", "180", "270"]
        remove_idx = -1
        for i, rot in enumerate(st["rotations"]):
            imgui.push_id(i)
            if imgui.button(
                f"{rot['sign']}##sign", imgui.ImVec2(hello_imgui.em_size(2.2), 0)
            ):
                rot["sign"] = "-" if rot["sign"] == "+" else "+"
            imgui.same_line()
            imgui.set_next_item_width(hello_imgui.em_size(3.6))
            ai = axes.index(rot["axis"]) if rot["axis"] in axes else 0
            changed, ai = imgui.combo("##axis", ai, axes)
            if changed:
                rot["axis"] = axes[ai]
            imgui.same_line()
            imgui.set_next_item_width(hello_imgui.em_size(5.0))
            cur_deg = str(int(rot["deg"]))
            di = degs.index(cur_deg) if cur_deg in degs else 0
            changed, di = imgui.combo("##deg", di, degs)
            if changed:
                rot["deg"] = int(degs[di])
            imgui.same_line()
            imgui.text("deg")
            imgui.same_line()
            if imgui.small_button("x##rm"):
                remove_idx = i
            imgui.pop_id()
        if remove_idx >= 0:
            del st["rotations"][remove_idx]
        if imgui.small_button("+ add##orient_add_rot"):
            st["rotations"].append({"sign": "-", "axis": "X", "deg": 90})
        imgui.same_line()
        if imgui.small_button("clear##orient_clear_rot"):
            st["rotations"] = []
        imgui.spacing()

        imgui.text("Flip")
        self._toggle_button_row(
            "orient_flip",
            (("X", "Horizontal (X)"), ("Y", "Vertical (Y)"), ("Z", "Depth (Z)")),
            lambda axis: axis in st["flips"],
            lambda axis: self._toggle_flip(st, axis),
        )
        imgui.spacing()

        any_set = False
        for t in targets:
            ops = self._compose_orientation_ops(t)
            if not ops:
                continue
            any_set = True
            parts = []
            for op in ops:
                parts.append(
                    f"{op[2]:+d} {op[1]}" if op[0] == "rot" else f"flip {op[1]}"
                )
            imgui.text_disabled(f"{t}: " + ", ".join(parts))
        if not any_set:
            imgui.text_disabled("no transforms")

    def _draw_stitcher_transforms_box(self) -> None:
        with tooltip_marks_right():
            _, self._stitcher_bake_tile_positions = imgui.checkbox(
                "Bake tile positions",
                self._stitcher_bake_tile_positions,
            )
            set_tooltip(
                "Lay tiles on the grid from the zarr metadata "
                "(stride offset, else stage). Off: tiles land at the origin."
            )

            _, self._stitcher_reverse_z = imgui.checkbox(
                "Reverse Z",
                self._stitcher_reverse_z,
            )
            set_tooltip("Join adjacent z-blocks contiguously (camera scans -Z).")

            if self._get_array().kind == "raw":
                self._stitcher_link_existing = False
            else:
                ch_link, self._stitcher_link_existing = imgui.checkbox(
                    "Link existing (.corrected)",
                    self._stitcher_link_existing,
                )
                if ch_link:
                    self._apply_orient_profile_all(self._stitcher_orient_profile)
                set_tooltip(
                    "Reference the .corrected zarrs in place, no conversion. "
                    "Much faster; .corrected only."
                )

            imgui.begin_disabled(self._stitcher_link_existing)
            imgui.set_next_item_width(_input_w())
            _, new_workers = imgui.input_int("Workers", self._workers, 1, 2)
            self._workers = max(1, min(_MAX_WORKERS, new_workers))
            set_tooltip(
                "Parallel volume writers (threads) for zarr conversion. "
                "RAM-bound: each holds one full camera volume + its pyramid."
            )
            imgui.end_disabled()

            imgui.set_next_item_width(_input_w())
            prev_pc = self._is_per_camera_export()
            ch_cam, self._stitcher_cameras = imgui.input_text(
                "Views",
                self._stitcher_cameras,
            )
            if ch_cam and self._is_per_camera_export() != prev_pc:
                self._apply_orient_profile_all(self._stitcher_orient_profile)
            set_tooltip(
                "Per-view export from .corrected, e.g. 0,2. Empty: fused VW00/VW90."
            )

            self._stitcher_zarr_version = 3
            if not self._stitcher_link_existing:
                imgui.begin_disabled()
                imgui.text("generate zarr dataset for bigstitcher")
                imgui.set_next_item_width(_input_w())
                imgui.combo("Zarr version", 0, ["v3"])
                imgui.end_disabled()

    def _draw_consolidate_io_box(self) -> None:
        """Consolidate I/O options for the Parameters popup."""
        with tooltip_marks_right():
            _, self._consolidate_pyramid = imgui.checkbox(
                "Pyramid",
                self._consolidate_pyramid,
            )
            set_tooltip(
                "Build the OME-NGFF pyramid for fast zoomed-out viewing. "
                "Adds write time + disk."
            )

            if self._consolidate_pyramid:
                imgui.set_next_item_width(_input_w())
                _, self._consolidate_pyramid_max_layers = imgui.input_int(
                    "Pyramid levels",
                    self._consolidate_pyramid_max_layers,
                    1,
                    1,
                )
                set_tooltip(
                    "Max additional pyramid levels beyond /0.\n"
                    "Stops early when Y or X would fall below 64 voxels."
                )

    def _draw_consolidate_codec_box(self) -> None:
        with tooltip_marks_right():
            compressors = [
                "zstd",
                "gzip",
                "blosc-zstd",
                "blosc-lz4",
                "none",
            ]
            idx = (
                compressors.index(self._consolidate_compressor)
                if self._consolidate_compressor in compressors
                else 0
            )
            imgui.set_next_item_width(_input_w())
            changed, new_idx = imgui.combo("Compressor", idx, compressors)
            if changed:
                self._consolidate_compressor = compressors[new_idx]
            set_tooltip(
                "Inner-chunk codec. zstd = best size/speed; none = fastest, biggest."
            )

            imgui.set_next_item_width(_input_w())
            _, self._consolidate_compression_level = imgui.input_int(
                "Level",
                self._consolidate_compression_level,
                1,
                1,
            )
            set_tooltip(
                "0–9. Higher = smaller files, slower writes; gains taper past ~5."
            )

    def _draw_codec_controls(self) -> None:
        """Compressor + level for the active output format (zarr or tif).

        klb carries its own codec, so nothing is shown for it. Called from
        inside a ``tooltip_marks_right()`` block.
        """
        if self._output_format not in ("zarr", "tif"):
            return
        compressors = _codecs_for_format(self._output_format)
        if self._compression not in compressors:
            self._compression = compressors[0]
        cidx = compressors.index(self._compression)
        imgui.set_next_item_width(_input_w())
        c_changed, c_new = imgui.combo("Compressor", cidx, compressors)
        if c_changed:
            self._compression = compressors[c_new]
        if self._output_format == "tif":
            set_tooltip("TIFF codec. deflate default; Fiji can't read zstd TIFFs.")
        else:
            set_tooltip(
                "Inner-chunk codec. zstd = best size/speed; "
                "none = fastest writes, biggest files."
            )

        imgui.set_next_item_width(_input_w())
        _, self._compression_level = imgui.input_int(
            "Level",
            self._compression_level,
            1,
            1,
        )
        set_tooltip("0–9. Higher = smaller files, slower writes; gains taper past ~5.")

    def _draw_zarr_chunk_controls(self) -> None:
        """Sharding toggle + inner-chunk / shard sizes (zarr only).

        Called from inside a ``tooltip_marks_right()`` block. Each size is
        three Z/Y/X int inputs; set all three to use them, leave any at 0
        for auto (inner chunk = one Y×X plane, shard = Z-slab).
        """
        if self._output_format != "zarr":
            return

        _, self._zarr_sharded = imgui.checkbox("Sharding", self._zarr_sharded)
        set_tooltip(
            "Group inner chunks into shard files (bounded file count). "
            "Off = one chunk per file."
        )

        self._zarr_chunk_z, self._zarr_chunk_y, self._zarr_chunk_x = _input_zyx(
            "Chunk [Z, Y, X]",
            "##chunk_zyx",
            self._zarr_chunk_z,
            self._zarr_chunk_y,
            self._zarr_chunk_x,
            "Inner chunk decompressed per read. Set all three; 0 = auto "
            "(one Y×X plane).",
        )
        if self._zarr_sharded:
            self._zarr_shard_z, self._zarr_shard_y, self._zarr_shard_x = _input_zyx(
                "Shards [Z, Y, X]",
                "##shard_zyx",
                self._zarr_shard_z,
                self._zarr_shard_y,
                self._zarr_shard_x,
                "Outer shard file size. Set all three; 0 = auto "
                "(memory-bounded Z-slab).",
            )

    @staticmethod
    def _zyx_tuple(z: int, y: int, x: int):
        """(z, y, x) tuple when all three are >0, else None (auto)."""
        if z > 0 and y > 0 and x > 0:
            return [int(z), int(y), int(x)]
        return None

    def _zarr_layout_args(self) -> dict:
        """zarr_chunks / zarr_shards / zarr_sharded for the task args.

        Empty when the output format isn't zarr (the keys are harmless but
        meaningless for tif/klb, so skip them).
        """
        if self._output_format != "zarr":
            return {}
        args: dict = {"zarr_sharded": bool(self._zarr_sharded)}
        chunks = self._zyx_tuple(
            self._zarr_chunk_z, self._zarr_chunk_y, self._zarr_chunk_x
        )
        if chunks is not None:
            args["zarr_chunks"] = chunks
        if self._zarr_sharded:
            shards = self._zyx_tuple(
                self._zarr_shard_z, self._zarr_shard_y, self._zarr_shard_x
            )
            if shards is not None:
                args["zarr_shards"] = shards
        return args

    def _draw_correct_io_box(self) -> None:
        """Correct-mode I/O options for the Parameters popup. The
        ``output_suffix`` lives in the Run-tab Output section now;
        only the codec / worker / pyramid / overwrite knobs stay here.
        """
        with tooltip_marks_right():
            formats = ["zarr", "tif", "klb"]
            idx = (
                formats.index(self._output_format)
                if self._output_format in formats
                else 0
            )
            imgui.set_next_item_width(_input_w())
            changed, new_idx = imgui.combo("Format", idx, formats)
            if changed:
                self._output_format = formats[new_idx]
                self._compression = _default_codec_for_format(self._output_format)
            set_tooltip(
                "Container per (timepoint, camera). zarr = chunked + "
                "pyramids; tif/klb for legacy tools."
            )

            self._draw_codec_controls()
            self._draw_zarr_chunk_controls()

            imgui.set_next_item_width(_input_w())
            _, new_workers = imgui.input_int(
                "Workers",
                self._workers,
                1,
                2,
            )
            self._workers = max(1, min(_MAX_WORKERS, new_workers))
            set_tooltip(
                "Parallel timepoint/tile workers. RAM-bound, not "
                "CPU-bound: each holds a full 3D volume.\n"
                f"Start 2–4 and watch RAM (soft cap {_MAX_WORKERS})."
            )
            self._draw_worker_ram_estimate(_CORRECT_RAM_PER_WORKER_X)

            _, self._pyramid = imgui.checkbox("Pyramid", self._pyramid)
            set_tooltip(
                "Downsampled levels for fast zoomed-out viewing "
                "(zarr / OME-TIFF). Adds write time + disk."
            )

            if self._pyramid:
                imgui.set_next_item_width(_input_w())
                _, self._pyramid_max_layers = imgui.input_int(
                    "Pyramid levels",
                    self._pyramid_max_layers,
                    1,
                    1,
                )
                set_tooltip(
                    "Extra downsample levels beyond full-res. More = "
                    "smoother zoom-out + more disk; stops when an axis "
                    "< 64 px."
                )

    def _draw_correct_segmentation_box(self) -> None:
        with tooltip_marks_right():
            seg_labels = ["none", "generate + save masks"]
            imgui.set_next_item_width(_input_w())
            s_changed, s_new = imgui.combo(
                "Segment mode",
                self._correct_segment_mode,
                seg_labels,
            )
            if s_changed:
                self._correct_segment_mode = s_new
            set_tooltip(
                "  0  off — skip segmentation entirely\n"
                "  1  build a binary foreground mask per (t, c) and "
                "save alongside the volume."
            )

            _, self._correct_apply_seg_mask = imgui.checkbox(
                "Apply mask to volume",
                self._correct_apply_seg_mask,
            )
            set_tooltip(
                "Zero background pixels in the saved volume (irreversible).\n"
                "Off keeps full intensities; the mask is saved separately."
            )

            _, self._correct_do_tenengrad = imgui.checkbox(
                "Tenengrad diagnostic",
                self._correct_do_tenengrad,
            )
            set_tooltip(
                "Extra per-Z sharpness plot per camera pair (small added "
                "time).\nUse it to set transition_plane / blending_range "
                "for Fuse."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_gauss_kernel = imgui.input_int(
                "Gaussian kernel",
                self._correct_gauss_kernel,
                1,
                1,
            )
            set_tooltip(
                "Pre-blur window (px) before thresholding.\n"
                "Larger = smoother masks, fewer specks, a bit slower."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_gauss_sigma = imgui.input_float(
                "Gaussian sigma",
                self._correct_gauss_sigma,
                0.1,
                1.0,
                "%.2f",
            )
            set_tooltip(
                "Pre-blur strength (px).\n"
                "Higher = smoother foreground; too high merges nearby objects."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_segment_threshold = imgui.input_float(
                "##segment_threshold",
                self._correct_segment_threshold,
                0.01,
                0.1,
                "%.3f",
            )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("segment_threshold", "Threshold")
            set_tooltip(
                "Foreground cutoff on the smoothed image (0–1).\n"
                "Lower keeps more as signal; raise if background leaks in."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_mask_percentile = imgui.input_float(
                "Mask percentile",
                self._correct_mask_percentile,
                0.1,
                1.0,
                "%.2f",
            )
            set_tooltip(
                "Baseline percentile for the mask (0–100).\n"
                "Higher = stricter foreground (drops dim signal)."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_splitting = imgui.input_int(
                "Splitting",
                self._correct_splitting,
                1,
                5,
            )
            set_tooltip(
                "Y-axis slabs for the segmentation filter.\n"
                "Raise to cut peak RAM (10→20 roughly halves the filter's "
                "working buffer), ~5–10% slower from slab overlap."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_subsample_factor = imgui.input_int(
                "Subsample factor",
                self._correct_subsample_factor,
                1,
                10,
            )
            set_tooltip(
                "Percentile sampling stride — uses every Nth voxel "
                "(100 ≈ 1%).\nHigher = faster, slightly noisier. Shared by "
                "segmentation + background."
            )

    def _draw_correct_advanced_box(self) -> None:
        with tooltip_marks_right():
            _, self._correct_median_kernel_enabled = imgui.checkbox(
                "Median filter",
                self._correct_median_kernel_enabled,
            )
            set_tooltip(
                "Replace hot/dead pixels with a per-plane median.\n"
                "Off if the camera is already corrected."
            )
            if self._correct_median_kernel_enabled:
                imgui.set_next_item_width(_input_w())
                _, self._correct_median_kernel_size = imgui.input_int(
                    "Kernel (N×N)",
                    self._correct_median_kernel_size,
                    1,
                    1,
                )
                set_tooltip(
                    "Median window (px). Larger removes bigger clusters "
                    "but softens detail and is slower."
                )

            imgui.set_next_item_width(_input_w())
            _, self._correct_background_percentile = imgui.input_float(
                "Background pct",
                self._correct_background_percentile,
                0.5,
                5.0,
                "%.2f",
            )
            set_tooltip(
                "Percentile for the camera background floor (0–100).\n"
                "Raise if a dark haze remains; too high clips dim signal."
            )

            _, self._correct_subtract_background = imgui.checkbox(
                "Subtract background",
                self._correct_subtract_background,
            )
            set_tooltip(
                "Subtract the per-camera Background_##.tif frame pixel-by-pixel\n"
                "(CM## − Background_##.tif, clamped at 0)."
            )

    def _draw_fuse_io_box(self) -> None:
        """Fuse-mode I/O options for the Parameters popup.

        Dataset-wide: format / codec / workers / pyramid. The output-folder
        variant lives in the Run-tab Output options block.
        """
        self._draw_view_scope_badge("shared")
        with tooltip_marks_right():
            formats = ["zarr", "tif", "klb"]
            idx = (
                formats.index(self._output_format)
                if self._output_format in formats
                else 0
            )
            imgui.set_next_item_width(_input_w())
            changed, new_idx = imgui.combo("Format", idx, formats)
            if changed:
                self._output_format = formats[new_idx]
                self._compression = _default_codec_for_format(self._output_format)
            set_tooltip(
                "Container per (timepoint, camera). zarr = chunked + "
                "pyramids; tif/klb for legacy tools."
            )

            self._draw_codec_controls()
            self._draw_zarr_chunk_controls()

            imgui.set_next_item_width(_input_w())
            _, new_workers = imgui.input_int(
                "Workers",
                self._workers,
                1,
                2,
            )
            self._workers = max(1, min(_MAX_WORKERS, new_workers))
            set_tooltip(
                "Parallel workers. RAM-bound: each holds both camera "
                "volumes + the output; adaptive blending uses more.\n"
                f"Start 2–4 (soft cap {_MAX_WORKERS})."
            )
            self._draw_worker_ram_estimate(_FUSE_RAM_PER_WORKER_X)

            _, self._pyramid = imgui.checkbox("Pyramid", self._pyramid)
            set_tooltip(
                "Downsampled levels for fast zoomed-out viewing "
                "(zarr / OME-TIFF). Adds write time + disk."
            )

            if self._pyramid:
                imgui.set_next_item_width(_input_w())
                _, self._pyramid_max_layers = imgui.input_int(
                    "Pyramid levels",
                    self._pyramid_max_layers,
                    1,
                    1,
                )
                set_tooltip(
                    "Extra downsample levels beyond full-res. More = "
                    "smoother zoom-out + more disk; stops when an axis "
                    "< 64 px."
                )

    def _draw_fuse_fusion_box(self) -> None:
        """Per-view fusion box: blending + view transforms under one
        shared Target-view badge.
        """
        self._draw_view_scope_badge("per_view")
        self._draw_fuse_blending_box()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text_colored(_TITLE_COLOR, "View transforms")
        imgui.spacing()
        self._draw_fuse_transforms_box()

    def _draw_fuse_blending_box(self) -> None:
        with tooltip_marks_right():
            methods = ["geometric", "adaptive", "average"]
            idx = (
                methods.index(self._fv("blending_method"))
                if self._fv("blending_method") in methods
                else 0
            )
            imgui.set_next_item_width(_input_w())
            changed, new_idx = imgui.combo("##blending_method", idx, methods)
            if changed:
                self._set_fv("blending_method", methods[new_idx])
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("blending_method", "Blending method")
            set_tooltip(
                "  geometric  fixed transition plane (fastest)\n"
                "  adaptive   per-XY tenengrad blend (sharper, more "
                "RAM/time)\n"
                "  average    plain mean (no blending)."
            )

            imgui.set_next_item_width(_input_w())
            _, new_br = imgui.input_int(
                "##blending_range",
                int(self._fv("blending_range")),
                1,
                5,
            )
            self._set_fv("blending_range", new_br)
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("blending_range", "Blending range")
            set_tooltip(
                "Z-plane width of the blend seam (geometric).\n"
                "Wider = smoother seam, more mixing of both cameras."
            )

            imgui.set_next_item_width(_input_w())
            _, new_tp = imgui.input_int(
                "##transition_plane",
                int(self._fv("transition_plane")),
                1,
                5,
            )
            self._set_fv("transition_plane", new_tp)
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("transition_plane", "Transition plane")
            set_tooltip(
                "Z-index where the cameras hand off (geometric).\n"
                "−1 = volume center; read it off the Tenengrad plot."
            )

            imgui.set_next_item_width(_input_w())
            _, new_ff = imgui.input_int(
                "##front_flag",
                int(self._fv("front_flag")),
                1,
                1,
            )
            self._set_fv("front_flag", new_ff)
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("front_flag", "Front camera")
            set_tooltip(
                "Which camera carries the sharper signal at low Z.\n"
                "  1 = reference camera (cam0)\n"
                "  2 = transformed camera (cam1)"
            )

    def _draw_fuse_background_box(self) -> None:
        self._draw_view_scope_badge("shared")
        with tooltip_marks_right():
            _, self._fuse_subtract_background = imgui.checkbox(
                "Subtract background",
                self._fuse_subtract_background,
            )
            set_tooltip(
                "Subtract an estimated background floor from both cameras "
                "before blending.\nOff: keep raw intensities."
            )

            if self._fuse_subtract_background:
                modes = ["pooled", "per_camera"]
                idx = (
                    modes.index(self._fuse_background_mode)
                    if self._fuse_background_mode in modes
                    else 0
                )
                imgui.set_next_item_width(_input_w())
                changed, new_idx = imgui.combo("Mode", idx, modes)
                if changed:
                    self._fuse_background_mode = modes[new_idx]
                set_tooltip(
                    "  pooled      one floor from both cameras\n"
                    "  per_camera  each camera subtracts its own floor — "
                    "removes the seam when the two cameras sit on "
                    "different baselines."
                )

                imgui.set_next_item_width(_input_w())
                _, self._fuse_background_percentile = imgui.input_float(
                    "Background pct",
                    self._fuse_background_percentile,
                    0.5,
                    5.0,
                    "%.2f",
                )
                set_tooltip(
                    "Percentile for the background floor (0–100).\n"
                    "Raise if a dark haze remains; too high clips dim signal."
                )

    def _draw_fuse_transforms_box(self) -> None:
        with tooltip_marks_right():
            _, new_fz = imgui.checkbox("Flip Z", bool(self._fv("flip_z")))
            self._set_fv("flip_z", new_fz)
            set_tooltip(
                "Flip the second camera along Z before fusion.\n"
                "Use for opposing camera pairs."
            )

            _, new_fh = imgui.checkbox(
                "Flip horizontal",
                bool(self._fv("flip_horizontal")),
            )
            self._set_fv("flip_horizontal", new_fh)
            set_tooltip("Flip the second camera horizontally before fusion.")

            _, new_fv = imgui.checkbox(
                "Flip vertical",
                bool(self._fv("flip_vertical")),
            )
            self._set_fv("flip_vertical", new_fv)
            set_tooltip("Flip the second camera vertically before fusion.")

    def _draw_fuse_search_box(self) -> None:
        self._draw_view_scope_badge("per_view")
        with tooltip_marks_right():
            imgui.text_colored(_SUBSECTION_COLOR, "X offsets")
            for label, key in (
                ("Start##sx", "search_x_start"),
                ("Stop##sx", "search_x_stop"),
                ("Step##sx", "search_x_step"),
            ):
                imgui.set_next_item_width(_input_w())
                _, new_val = imgui.input_int(
                    label,
                    int(self._fv(key)),
                    1,
                    10,
                )
                self._set_fv(key, new_val)
                set_tooltip(
                    "Pixel offset grid searched along X to align the pair "
                    "(start, stop, step).\nWider range or smaller step = "
                    "more robust, slower (cost ∝ range/step)."
                )

            imgui.spacing()
            imgui.text_colored(_SUBSECTION_COLOR, "Y offsets")
            for label, key in (
                ("Start##sy", "search_y_start"),
                ("Stop##sy", "search_y_stop"),
                ("Step##sy", "search_y_step"),
            ):
                imgui.set_next_item_width(_input_w())
                _, new_val = imgui.input_int(
                    label,
                    int(self._fv(key)),
                    1,
                    10,
                )
                self._set_fv(key, new_val)
                set_tooltip(
                    "Pixel offset grid searched along Y to align the pair "
                    "(start, stop, step).\nWider range or smaller step = "
                    "more robust, slower (cost ∝ range/step)."
                )

    def _draw_microscope_overrides_box(self) -> None:
        with tooltip_marks_right():
            _, self._mic_overrides_enabled = imgui.checkbox(
                "Override XML defaults",
                self._mic_overrides_enabled,
            )
            set_tooltip(
                "When off, values are read from the dataset XML.\n"
                "Enable to override one or more values below."
            )
            imgui.spacing()

            if not self._mic_overrides_enabled:
                imgui.begin_disabled()
            try:
                _hint("−1 = auto from XML")
                imgui.spacing()

                imgui.set_next_item_width(_input_w())
                _, self._mic_pixel_spacing_z = imgui.input_float(
                    "Z spacing (µm)",
                    self._mic_pixel_spacing_z,
                    0.01,
                    0.1,
                    "%.3f",
                )
                set_tooltip("Physical Z-step in micrometers.")

                imgui.set_next_item_width(_input_w())
                _, self._mic_objective_mag = imgui.input_float(
                    "Objective mag",
                    self._mic_objective_mag,
                    0.5,
                    5.0,
                    "%.2f",
                )
                set_tooltip("Detection objective magnification (e.g. 16.0).")

                imgui.set_next_item_width(_input_w())
                _, self._mic_pixel_spacing_camera = imgui.input_float(
                    "Camera pixel (µm)",
                    self._mic_pixel_spacing_camera,
                    0.1,
                    1.0,
                    "%.2f",
                )
                set_tooltip(
                    "Physical camera pixel size in micrometers "
                    "(e.g. 6.5 for the C11440-22C)."
                )
            finally:
                if not self._mic_overrides_enabled:
                    imgui.end_disabled()

    def _microscope_kwargs(self) -> dict:
        """Pack microscope overrides as a kwargs dict for the task args.

        Sentinels of ``-1`` map to ``None`` (auto from XML); only set
        keys the user explicitly overrode.
        """
        out: dict = {}
        if not self._mic_overrides_enabled:
            return out
        if self._mic_pixel_spacing_z >= 0:
            out["pixel_spacing_z"] = float(self._mic_pixel_spacing_z)
        if self._mic_objective_mag > 0:
            out["detection_objective_mag"] = float(self._mic_objective_mag)
        if self._mic_pixel_spacing_camera > 0:
            out["pixel_spacing_camera"] = float(self._mic_pixel_spacing_camera)
        return out

    def _submit_active(self, arr: Any) -> None:
        if self._selected_mode == _MODE_CONSOLIDATE:
            self._submit_consolidate(arr)
        elif self._selected_mode == _MODE_CORRECT:
            self._submit_correct(arr)
        elif self._selected_mode == _MODE_FUSE:
            self._submit_fuse(arr)
        elif self._selected_mode == _MODE_STITCHER:
            self._submit_stitcher(arr)

    def _submit_consolidate(self, arr: Any) -> None:
        sr = Path(arr.scan_root)
        out_path = str(sr.parent / f"{sr.name}{self._consolidate_suffix}")
        self._consolidate_output_path = out_path
        if not out_path or not (self._consolidate_suffix or "").strip():
            self._last_status = "Suffix is required."
            return
        args = {
            "input_path": str(arr.scan_root),
            "output_path": out_path,
            "kind": arr.kind,
            "overwrite": self._consolidate_overwrite,
            "pyramid": self._consolidate_pyramid,
            "pyramid_max_layers": self._consolidate_pyramid_max_layers,
            "compressor": self._consolidate_compressor,
            "compression_level": self._consolidate_compression_level,
        }
        self._spawn(
            "isoview",
            args,
            description=f"Isoview consolidate ({arr.kind}) → {Path(out_path).name}",
            output_path=out_path,
        )

    def _tile_spm_key(self, arr: Any, ti: Any) -> str | int:
        """The fused ``spm_key`` for tile index ``ti``.

        Mirrors isoview's tiled naming: the ``specimen_name`` grid token
        (e.g. ``"TL000"``) when it parses as a grid position, else the
        integer specimen id (isoview formats that as ``SPM##``).
        """
        from mbo_utilities.arrays.isoview.array import _parse_tile_grid_position

        tm = arr.tile_metadata.get(ti) or {}
        name = tm.get("specimen_name")
        if name and _parse_tile_grid_position(str(name)) is not None:
            return str(name)
        spc = tm.get("specimen", ti)
        try:
            return int(spc)
        except (TypeError, ValueError):
            return ti

    @staticmethod
    def _ops_to_inplane(ops):
        """Reduce an Align-views op list to ``[rot, fx, fy]`` (Z-rotation 90deg
        steps + X/Y flips). Out-of-plane (X/Y rotation) and Z-flip components are
        dropped — they aren't part of the in-plane tile recipe.
        """
        rot = fx = fy = 0
        for op in ops or []:
            kind, axis = str(op[0]).lower(), str(op[1]).upper()
            if kind == "rot" and axis == "Z":
                rot = (rot + int(op[2]) // 90) % 4
            elif kind == "flip" and axis == "X":
                fx ^= 1
            elif kind == "flip" and axis == "Y":
                fy ^= 1
        return [rot, fx, fy]

    def _overlay_align_views(self, tile_orientations, arr):
        """Compose the Align views per-view orientation onto every tile entry in
        ``tile_orientations`` (rotations add -> position, flips XOR -> pose,
        matching how the export splits them), so view-level corrections drive the
        export too. Returns the original when Align views has nothing applied.
        """
        from mbo_utilities.gui import _isoview_orient_state as orient_state

        applied = orient_state.get_all_applied(arr)
        if not applied:
            return tile_orientations
        view_ip = {}
        for view in applied:
            ip = self._ops_to_inplane(orient_state.applied_ops(arr, view))
            if ip != [0, 0, 0]:
                view_ip[view] = ip
        if not view_ip:
            return tile_orientations
        out = {
            gk: {v: list(t) for v, t in views.items()}
            for gk, views in (tile_orientations or {}).items()
        }
        for views in out.values():
            for view, ip in view_ip.items():
                base = views.get(view, [0, 0, 0])
                views[view] = [(base[0] + ip[0]) % 4, base[1] ^ ip[1], base[2] ^ ip[2]]
        return out or None

    def _submit_stitcher(self, arr: Any) -> None:
        """Spawn ``generate_bigstitcher_xml`` against the loaded tree.

        Prefers the ``.corrected*`` or ``.fused*`` tree the array was
        loaded from — upstream ProcessingConfig now accepts those
        directly, so the raw acquisition root is no longer required.
        Falls back to a sibling raw root only for legacy datasets that
        predate the embedded ``ome.isoview`` metadata.

        Forwards ``specimens`` and ``timepoints`` from the loaded array
        so ProcessingConfig skips its auto-detect step entirely.
        """
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root

        scan_root = Path(arr.scan_root)
        input_dir: Path | None = None
        if arr.kind == "raw":
            input_dir = scan_root
        for ancestor in (
            () if input_dir is not None else (scan_root, *scan_root.parents)
        ):
            n = ancestor.name
            for suf in (".corrected", ".fused"):
                idx = n.find(suf)
                if idx < 0:
                    continue
                rest = n[idx + len(suf) :]
                if rest == "" or rest.startswith("_"):
                    input_dir = ancestor
                    break
            if input_dir is not None:
                break

        if input_dir is None:
            input_dir = _sibling_raw_root(arr)
        if input_dir is None or not input_dir.is_dir():
            self._last_status = (
                "Cannot locate a corrected or fused tree to export from."
            )
            return

        meta = arr.metadata
        specimen = meta.get("specimen", 0)

        from mbo_utilities.arrays.isoview.array import _is_method_dir

        method = None
        if arr.kind == "fused":
            sr = Path(arr.scan_root)
            if _is_method_dir(sr):
                method = sr.name

        cameras = self._parse_stitcher_cameras()
        source = "raw" if arr.kind == "raw" else "fused"
        if self._stitcher_link_existing and arr.kind == "corrected":
            source = "corrected"
        per_camera = self._is_per_camera_export()
        cam_orient = None
        orient_to_cm00 = True
        if per_camera:
            from mbo_utilities.arrays.isoview.array import camera_view_label

            export_cams = cameras if cameras else [0, 1, 2, 3]
            cam_orient = {
                str(c): self._compose_orientation_ops(camera_view_label(c))
                for c in export_cams
            }
            orient_to_cm00 = False
        included_tiles = None
        if arr.is_tiled:
            sel = self._selected_timepoints()
            if sel is not None:
                tps = list(arr._timepoints)
                included_tiles = [
                    self._tile_spm_key(arr, tps[i]) for i in sel if 0 <= i < len(tps)
                ] or None
        tile_orientations = None
        for _w in self.parent._widgets:
            if type(_w).__name__ == "TileGridViewer":
                tile_orientations = _w.export_tile_orientations(arr) or None
                break
        tile_orientations = self._overlay_align_views(tile_orientations, arr)
        args = {
            "input_path": str(input_dir),
            "method": method,
            "output_suffix": self._iso_tree_suffix(input_dir),
            "stitcher_suffix": self._stitcher_suffix,
            "bake_tile_positions": self._stitcher_bake_tile_positions,
            "orientation": self._compose_orientation_ops("VW00"),
            "orientation_vw90": self._compose_orientation_ops("VW90"),
            "camera_orientations": cam_orient,
            "overwrite": self._overwrite,
            "specimens": [int(specimen)],
            "timepoints": list(arr._timepoints),
            "included_tiles": included_tiles,
            "cameras": cameras or None,
            "source": source,
            "orient_to_cm00": orient_to_cm00,
            "reverse_z": self._stitcher_reverse_z,
            "link_existing": self._stitcher_link_existing,
            "zarr_version": self._stitcher_zarr_version,
            "tile_orientations": tile_orientations,
            "workers": self._workers,
        }
        args.update(self._microscope_kwargs())
        self._spawn(
            "isoview_bigstitcher",
            args,
            description=f"BigStitcher XML: {input_dir.name}",
            output_path=self._current_output_path() or str(input_dir),
        )

    def _submit_correct(self, arr: Any) -> None:
        median_kernel = None
        if self._correct_median_kernel_enabled:
            k = max(1, int(self._correct_median_kernel_size))
            median_kernel = [k, k]
        resolved_out = self._current_output_path()
        args = {
            "input_path": str(arr.scan_root),
            "output_dir": resolved_out or None,
            "output_suffix": self._correct_output_suffix,
            "output_format": self._output_format,
            "compression": (None if self._compression == "none" else self._compression),
            "compression_level": self._compression_level,
            "overwrite": self._overwrite,
            "workers": self._workers,
            "pyramid": self._pyramid,
            "pyramid_max_layers": self._pyramid_max_layers,
            "segment_mode": self._correct_segment_mode,
            "apply_segmentation_mask": self._correct_apply_seg_mask,
            "do_tenengrad": self._correct_do_tenengrad,
            "background_percentile": self._correct_background_percentile,
            "median_kernel": median_kernel,
            "subtract_background_image": bool(self._correct_subtract_background),
            "mask_percentile": self._correct_mask_percentile,
            "subsample_factor": self._correct_subsample_factor,
            "gauss_kernel": self._correct_gauss_kernel,
            "gauss_sigma": self._correct_gauss_sigma,
            "segment_threshold": self._correct_segment_threshold,
            "splitting": self._correct_splitting,
        }
        is_tiled = arr.is_tiled
        args["tiled_override"] = is_tiled
        sel = self._selected_timepoints()
        if sel is not None:
            if is_tiled:
                tiles = list(arr._timepoints)
                # NOTE: a tiled tree has one TM; a timepoints list would collapse to [0] and drop the selection
                args["specimens"] = [
                    tiles[i] for i in sel if 0 <= i < len(tiles)
                ] or sel
            else:
                args["timepoints"] = sel
        cams = self._selected_cameras(arr)
        if cams is not None:
            args["cameras"] = cams
        args.update(self._zarr_layout_args())
        args.update(self._microscope_kwargs())
        from mbo_utilities.gui import _isoview_crop_state as crop_state

        args.update(crop_state.to_config_args(arr))
        self._spawn(
            "isoview_correct",
            args,
            description=f"correct_stack: {Path(arr.scan_root).name}",
            output_path=resolved_out or str(arr.scan_root),
        )

    def _draw_worker_ram_estimate(self, per_worker_x: float) -> None:
        """Estimated-RAM line under the Workers control. Per-worker footprint is
        ~``per_worker_x`` x one raw camera volume; total = that x worker count.
        Colored amber when it nears, red when it exceeds, *available* RAM —
        compared against free memory (not total) so baseline usage counts, same
        basis as ``_ram_capped_fuse_workers``.
        """
        vol_gb = self._iso_vol_gb
        if vol_gb <= 0:
            return
        per_worker = vol_gb * per_worker_x
        total = per_worker * max(1, self._workers)
        avail_gb = psutil.virtual_memory().available / (1024**3)
        label = f"~{total:.0f} GB RAM  ({per_worker:.1f} GB/worker x {self._workers})"
        if total > avail_gb:
            imgui.text_colored(imgui.ImVec4(0.92, 0.32, 0.32, 1.0), label)
        elif total > 0.8 * avail_gb:
            imgui.text_colored(imgui.ImVec4(0.95, 0.75, 0.32, 1.0), label)
        else:
            imgui.text_disabled(label)
        set_tooltip(
            f"Rough estimate: each worker holds ~{per_worker_x:.0f}x one raw "
            f"camera volume ({vol_gb:.1f} GB). Free RAM: {avail_gb:.0f} GB."
        )

    def _ram_capped_fuse_workers(self, arr: Any, requested: int) -> int:
        """Clamp the fuse worker count to what available RAM can hold.

        Each tiled-fuse worker loads two camera volumes + two masks + the
        fused output, so budget ~6× one Z×Y×X volume per worker. Returns
        ``requested`` unchanged when psutil/shape are unavailable or memory
        is ample; only ever lowers it.
        """
        requested = max(1, int(requested))
        per_worker_gb = max(1.0, _iso_volume_gb(arr) * _FUSE_RAM_PER_WORKER_X)
        avail_gb = psutil.virtual_memory().available / (1024**3)
        ram_cap = max(1, int((avail_gb * 0.8) // per_worker_gb))
        if ram_cap < requested:
            self.parent.logger.info(
                f"multi_fuse: capping workers {requested} -> {ram_cap} "
                f"(~{per_worker_gb:.0f} GiB/worker, {avail_gb:.0f} GiB free)"
            )
            return ram_cap
        return requested

    def _submit_fuse(self, arr: Any) -> None:
        scan_root = Path(arr.scan_root)
        input_dir: Path | None = None
        for ancestor in (scan_root, *scan_root.parents):
            n = ancestor.name
            idx = n.find(".corrected")
            if idx < 0:
                continue
            rest = n[idx + len(".corrected") :]
            if rest == "" or rest.startswith("_"):
                input_dir = ancestor
                break
        if input_dir is None or not input_dir.is_dir():
            self.parent.logger.error(
                "multi_fuse: could not locate a .corrected tree above "
                f"{scan_root}. Open a dataset whose scan root sits "
                "under a *.corrected[_suffix]/ directory."
            )
            return
        active_view = self._fuse_active_view
        if active_view is None or active_view not in self._fuse_view_params:
            active_view = (self._fuse_view_ids or [0])[0]
        active = self._fuse_view_params.get(
            active_view,
            self._default_fuse_view_params(),
        )

        by_view: dict[str, dict[int, Any]] = {
            "blending_method_by_view": {},
            "blending_range_by_view": {},
            "transition_plane_by_view": {},
            "front_flag_by_view": {},
            "flip_z_by_view": {},
            "flip_horizontal_by_view": {},
            "flip_vertical_by_view": {},
            "rotation_by_view": {},
            "search_offsets_x_by_view": {},
            "search_offsets_y_by_view": {},
        }
        for vid, p in self._fuse_view_params.items():
            by_view["blending_method_by_view"][vid] = p["blending_method"]
            by_view["blending_range_by_view"][vid] = int(p["blending_range"])
            by_view["transition_plane_by_view"][vid] = _transition_plane(
                p["transition_plane"]
            )
            by_view["front_flag_by_view"][vid] = int(p["front_flag"])
            by_view["flip_z_by_view"][vid] = bool(p["flip_z"])
            by_view["flip_horizontal_by_view"][vid] = bool(p["flip_horizontal"])
            by_view["flip_vertical_by_view"][vid] = bool(p["flip_vertical"])
            by_view["rotation_by_view"][vid] = int(p["rotation"])
            by_view["search_offsets_x_by_view"][vid] = [
                int(p["search_x_start"]),
                int(p["search_x_stop"]),
                int(p["search_x_step"]),
            ]
            by_view["search_offsets_y_by_view"][vid] = [
                int(p["search_y_start"]),
                int(p["search_y_stop"]),
                int(p["search_y_step"]),
            ]

        workers = self._ram_capped_fuse_workers(arr, self._workers)

        args = {
            "input_path": str(input_dir),
            "output_dir": self._output_dir or None,
            "output_format": self._output_format,
            "compression": (None if self._compression == "none" else self._compression),
            "compression_level": self._compression_level,
            "overwrite": self._overwrite,
            "workers": workers,
            "pyramid": self._pyramid,
            "pyramid_max_layers": self._pyramid_max_layers,
            "output_suffix": (self._fuse_output_suffix or "").strip() or None,
            "subtract_background": bool(self._fuse_subtract_background),
            "background_mode": self._fuse_background_mode,
            "background_percentile": float(self._fuse_background_percentile),
            "blending_method": active["blending_method"],
            "blending_range": int(active["blending_range"]),
            "transition_plane": _transition_plane(active["transition_plane"]),
            "front_flag": int(active["front_flag"]),
            "flip_z": bool(active["flip_z"]),
            "flip_horizontal": bool(active["flip_horizontal"]),
            "flip_vertical": bool(active["flip_vertical"]),
            "rotation": int(active["rotation"]),
            "search_offsets_x": [
                int(active["search_x_start"]),
                int(active["search_x_stop"]),
                int(active["search_x_step"]),
            ],
            "search_offsets_y": [
                int(active["search_y_start"]),
                int(active["search_y_stop"]),
                int(active["search_y_step"]),
            ],
            **by_view,
        }
        is_tiled = arr.is_tiled
        args["tiled_override"] = is_tiled
        tps = self._selected_timepoints()
        if is_tiled:
            if tps is not None:
                tiles = list(arr._timepoints)
                # NOTE: a tiled tree has one TM; a timepoints list would collapse to [0] and drop the selection
                args["specimens"] = [
                    tiles[i] for i in tps if 0 <= i < len(tiles)
                ] or tps
        else:
            from mbo_utilities.arrays.isoview.array import (
                _SPM_PATTERN,
                _extract_timepoint,
                _find_tm_folders,
            )

            corrected_root = scan_root if _SPM_PATTERN.match(scan_root.name) else None
            if corrected_root is None:
                for d in scan_root.iterdir() if scan_root.is_dir() else []:
                    if d.is_dir() and _SPM_PATTERN.match(d.name):
                        corrected_root = d
                        break
            available_tms: list[int] = []
            if corrected_root is not None:
                available_tms = [
                    _extract_timepoint(d.name) for d in _find_tm_folders(corrected_root)
                ]
            if available_tms:
                if tps is None:
                    args["timepoints"] = available_tms
                else:
                    args["timepoints"] = [
                        available_tms[i] for i in tps if 0 <= i < len(available_tms)
                    ]
            elif tps is not None:
                args["timepoints"] = tps
        args.update(self._zarr_layout_args())
        args.update(self._microscope_kwargs())
        from mbo_utilities.gui import _isoview_crop_state as crop_state

        args.update(crop_state.to_config_args(arr))
        self._spawn(
            "isoview_fuse",
            args,
            description=f"multi_fuse ({active['blending_method']}): {input_dir.name}",
            output_path=self._output_dir or str(input_dir),
        )

    def _init_iso_selection_state(self, arr: Any) -> int:
        """Initialize / reset timepoint-slicing state for the current array.

        Returns max_frames (1-based). Resets the selection string back to
        ``1:N`` when the user loads a different dataset or the timepoint
        count changes (e.g. switching kinds).
        """
        max_frames = max(1, int(arr.num_timepoints))
        current_fpath = str(arr.scan_root)
        if self._iso_last_fpath != current_fpath:
            self._iso_last_fpath = current_fpath
            self._iso_last_max_tp = 0
        if self._iso_last_max_tp != max_frames:
            self._iso_last_max_tp = max_frames
            self._iso_tp_selection = f"1:{max_frames}"
            self._iso_tp_parsed = None
            self._iso_tp_error = ""

        if self._iso_tp_parsed is None and not self._iso_tp_error:
            try:
                self._iso_tp_parsed = parse_timepoint_selection(
                    self._iso_tp_selection,
                    max_frames,
                )
            except ValueError as e:
                self._iso_tp_error = str(e)

        return max_frames

    def _emp_label(self, field: str, text: str) -> None:
        """Render an external label for an inline widget.

        Important params (members of :data:`_ISOVIEW_IMPORTANT_FIELDS`)
        get a boxed bold label; everything else is plain text. Pair with
        ``imgui.<widget>("##field", ...)`` + ``imgui.same_line(0,
        item_inner_spacing.x)`` to replicate imgui's default inline-label
        placement, then call this.
        """
        if field in _ISOVIEW_IMPORTANT_FIELDS:
            bold_font = self.parent._bold_font
            draw_boxed_label(text, font=bold_font)
        else:
            imgui.text(text)

    def _draw_isoview_missing_banner(self) -> None:
        warn = imgui.ImVec4(0.95, 0.55, 0.25, 1.0)
        cmd = imgui.ImVec4(0.6, 0.8, 1.0, 1.0)
        imgui.spacing()
        imgui.push_text_wrap_pos(0.0)
        try:
            imgui.text_colored(warn, "isoview not installed")
            imgui.text_colored(cmd, "pip install isoview")
        finally:
            imgui.pop_text_wrap_pos()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    def _draw_iso_current_dataset(self, arr: Any) -> None:
        """Current dataset block — path + Files (N) popup + shape + size +
        Data state / dx / dy / dz / fs.
        """
        from mbo_utilities.metadata import get_param

        imgui.text_colored(_SUBSECTION_COLOR, "Current dataset")
        set_tooltip(
            "The IsoviewArray currently loaded in the viewer.\n"
            "Data state selects which pipelines apply (raw → Correct, "
            "corrected → Fuse/Consolidate, fused → Consolidate)."
        )
        imgui.spacing()
        imgui.spacing()

        filenames = list(arr.filenames)
        n_files = len(filenames)
        if self._iso_files_sizer is None:
            self._iso_files_sizer = PopupAutoSize(
                "Dataset files##current_dataset_files_popup",
                auto_resize=False,
            )

        path_str = str(arr.scan_root)
        if path_str:
            short_name = Path(path_str).name or path_str
        else:
            short_name = "(in-memory)"
        size_bytes = _dataset_size_bytes(filenames)

        shape = tuple(arr.shape)
        dims = "TVZYX" if len(shape) == 5 else None
        shape_text = " × ".join(str(s) for s in shape)
        if dims and len(dims) == len(shape):
            shape_text = f"{shape_text}\n[{','.join(dims)}]"

        md = dict(arr.metadata)
        _warn_color = imgui.ImVec4(0.95, 0.45, 0.35, 1.0)
        rows: list[tuple[str, str, str | None, Any]] = [
            ("Name", short_name, path_str or None, None),
            ("Size on disk", _format_size(size_bytes), None, None),
            ("Shape", shape_text, None, None),
            ("Data state", str(arr.kind), None, None),
        ]
        if not arr.is_tiled:
            custom = self.parent._custom_metadata
            fs_val = custom.get("fs")
            if fs_val is None:
                fs_val = md.get(
                    "fs"
                )  # NOTE: the exact key, not get_param: its aliases would hand back the camera rate fps
            if fs_val is None:
                rows.append(
                    (
                        "Frame rate",
                        "not encoded (Shift+M to set)",
                        "isoview metadata does not currently encode the frame "
                        "rate.\nPress Shift+M to open the metadata editor and set it.",
                        _warn_color,
                    )
                )
            else:
                rows.append(("Frame rate", f"{fs_val} Hz", None, None))
        for label, key, unit in (
            ("dx", "dx", "µm"),
            ("dy", "dy", "µm"),
            ("dz", "dz", "µm"),
        ):
            v = get_param(md, key)
            if v is None:
                continue
            rows.append((label, f"{v} {unit}".rstrip(), None, None))

        imgui.indent(8)
        try:
            tflags = (
                imgui.TableFlags_.sizing_stretch_prop
                | imgui.TableFlags_.no_borders_in_body
                | imgui.TableFlags_.pad_outer_x
            )
            if imgui.begin_table("iso_current_dataset_table", 2, tflags):
                imgui.table_setup_column(
                    "field", imgui.TableColumnFlags_.width_fixed, 90.0
                )
                imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
                for label, value, hover, color in rows:
                    imgui.table_next_row()
                    imgui.table_set_column_index(0)
                    if color is not None:
                        text_wrapped_cell(label, color)
                    else:
                        imgui.text_unformatted(label)
                    imgui.table_set_column_index(1)
                    text_wrapped_cell(value, color)
                    if hover and imgui.is_item_hovered():
                        imgui.set_tooltip(hover)
                imgui.end_table()

            imgui.spacing()
            if imgui.button(f"Files ({n_files})##iso_dataset_files"):
                self._iso_files_sizer.before_open()
                imgui.open_popup("Dataset files##current_dataset_files_popup")
            _draw_dataset_files_popup(filenames, None, sizer=self._iso_files_sizer)
        finally:
            imgui.unindent(8)

        imgui.spacing()
        imgui.spacing()

    def _suffix_attr_for_mode(self) -> str | None:
        """The instance attribute holding the active mode's output suffix.

        ``None`` when the mode has no suffix concept (shouldn't happen
        for any mode the widget currently supports).
        """
        return {
            _MODE_CORRECT: "_correct_output_suffix",
            _MODE_FUSE: "_fuse_output_suffix",
            _MODE_STITCHER: "_stitcher_suffix",
            _MODE_CONSOLIDATE: "_consolidate_suffix",
        }.get(self._selected_mode)

    def _output_basename_for_mode(self, arr: Any) -> str:
        """Folder/file basename the suffix attaches to.

        - CORRECT: arr is raw, basename = ``arr.scan_root.name``.
        - FUSE / STITCHER: basename = name of the raw root sibling
          (strips ``.corrected`` / ``.fused``).
        - CONSOLIDATE: basename = ``arr.scan_root.name`` (the suffix
          becomes the trailing portion of the .zarr filename).
        """
        sr = Path(arr.scan_root)
        if self._selected_mode in (_MODE_CORRECT, _MODE_CONSOLIDATE):
            return sr.name
        tree = self._iso_input_tree(arr)
        if tree is not None:
            return self._iso_raw_stem(tree)
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root

        raw = _sibling_raw_root(arr)
        return raw.name if raw is not None else sr.name

    def _overwrite_attr_for_mode(self) -> str | None:
        """Instance attr holding the active mode's overwrite flag."""
        return {
            _MODE_CORRECT: "_overwrite",
            _MODE_FUSE: "_overwrite",
            _MODE_STITCHER: "_overwrite",
            _MODE_CONSOLIDATE: "_consolidate_overwrite",
        }.get(self._selected_mode)

    def _resolved_output_dir(self, arr: Any) -> Path | None:
        """Absolute path the current mode will write to (for the exists check).

        Mirrors the per-mode output naming so the Output-options block can
        warn + offer Overwrite only when something is actually there.
        """
        mode = self._selected_mode
        if mode == _MODE_CONSOLIDATE:
            return (
                Path(self._consolidate_output_path)
                if self._consolidate_output_path
                else None
            )
        if mode == _MODE_STITCHER:
            return self._iso_stitcher_dest(arr)
        prefix = _MODE_PREFIX.get(mode)
        if prefix is None:
            return None
        attr = self._suffix_attr_for_mode()
        variant = _norm_variant(getattr(self, attr, "") if attr else "")
        basename = self._output_basename_for_mode(arr)
        if mode == _MODE_CORRECT:
            parent = Path(arr.scan_root).parent
        else:
            tree = self._iso_input_tree(arr)
            parent = tree.parent if tree is not None else Path(arr.scan_root).parent
        return parent / f"{basename}{prefix}{variant}"

    def _draw_iso_output_options(self, arr: Any) -> None:
        """Universal Output options block: one variant input + a
        filename preview. Every other I/O knob (codec / workers /
        pyramid / overwrite / format) lives in the Parameters popup.

        For the tree-producing steps the prefix (.corrected/.fused/
        .stitcher) is forced; the field is just the optional variant,
        with a leading underscore added automatically.
        """
        imgui.text_colored(_SUBSECTION_COLOR, "Output options")
        prefix = _MODE_PREFIX.get(self._selected_mode)
        set_tooltip(
            f"Output folder = <dataset>{prefix or ''}[_<suffix>]. The "
            f"{prefix} prefix is automatic; type only a suffix (blank = "
            f"bare {prefix}). A leading underscore is added for you."
            if prefix
            else "Filename suffix appended to the source name.",
            align="right",
        )
        imgui.spacing()

        attr = self._suffix_attr_for_mode()
        if attr is None:
            return

        imgui.set_next_item_width(_input_w())
        cur = getattr(self, attr) or ""
        _, new_val = imgui.input_text("Suffix##iso_suffix", cur)
        setattr(self, attr, new_val)

        basename = self._output_basename_for_mode(arr)
        if prefix:
            preview = f"{basename}{prefix}{_norm_variant(new_val)}/"
        else:
            preview = f"{basename}{new_val}"
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.65, 1.0))
        imgui.push_text_wrap_pos(0.0)
        try:
            imgui.text_unformatted(f"Result: {preview}")
        finally:
            imgui.pop_text_wrap_pos()
            imgui.pop_style_color()

        ow_attr = self._overwrite_attr_for_mode()
        if ow_attr is not None:
            out_dir = self._resolved_output_dir(arr)
            if out_dir is not None and out_dir.exists():
                imgui.spacing()
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.8, 0.2, 1.0), "Directory already exists"
                )
                _, ow = imgui.checkbox(
                    "Overwrite##iso_overwrite", bool(getattr(self, ow_attr))
                )
                setattr(self, ow_attr, ow)
            else:
                setattr(self, ow_attr, False)
        imgui.spacing()
        imgui.spacing()

    def _draw_iso_crop_readout(self, arr: Any) -> None:
        """Crop-bounds Run-tab block — corrected (fuse) data only.

        Mirrors the segmentation / dead-pixel readouts: Edit opens the
        floating crop window, Clear resets, and the summary shows the
        current per-camera bounds. Trailing separator emitted inline.
        """
        if arr.kind != "corrected":
            return
        if self._selected_mode != _MODE_FUSE:
            return
        from mbo_utilities.gui import _isoview_crop_state as crop_state
        from mbo_utilities.gui.widgets import isoview_crop as crop_window

        imgui.text_colored(_SUBSECTION_COLOR, "Crop bounds")
        set_tooltip(
            "Per-camera crop applied at fuse time. Edit opens the "
            "side-by-side crop window with draggable bounds.",
            align="right",
        )
        imgui.spacing()
        if imgui.button("Edit...##iso_crop_edit", imgui.ImVec2(_BTN_W, 0)):
            crop_window.open_window(self.parent)
        imgui.same_line()
        if imgui.button("Clear##iso_crop_clear", imgui.ImVec2(_BTN_W, 0)):
            crop_state.clear(arr)
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.65, 1.0))
        imgui.push_text_wrap_pos(0.0)
        try:
            imgui.text_unformatted(crop_state.summary(arr))
        finally:
            imgui.pop_text_wrap_pos()
            imgui.pop_style_color()
        imgui.spacing()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    def _draw_iso_segment_readout(self, arr: Any) -> None:
        """Segmentation threshold Run-tab block — only meaningful on
        raw data (where ``correct_stack`` will run). Hidden otherwise.
        Trailing separator emitted inline so a hidden readout leaves
        no orphan separator behind.
        """
        if arr.kind != "raw":
            return
        if self._selected_mode == _MODE_STITCHER:
            return
        from mbo_utilities.gui.widgets import isoview_segment as seg_window

        imgui.text_colored(_SUBSECTION_COLOR, "Segmentation threshold")
        set_tooltip(
            "Threshold + gaussian smoothing + mask percentile that drive "
            "foreground/background separation in correct_stack. Edit "
            "opens a live preview with sliders and per-view masks.",
            align="right",
        )
        imgui.spacing()
        if imgui.button("Edit...##iso_seg_edit", imgui.ImVec2(_BTN_W, 0)):
            seg_window.open_window(self.parent)
        imgui.same_line()
        if imgui.button("Reset##iso_seg_reset", imgui.ImVec2(_BTN_W, 0)):
            self._correct_segment_threshold = 0.4
            self._correct_mask_percentile = 1.0
            self._correct_gauss_sigma = 2.0
            self._correct_gauss_kernel = 5
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.65, 1.0))
        imgui.push_text_wrap_pos(0.0)
        try:
            imgui.text_unformatted(
                f"threshold={self._correct_segment_threshold:.3f}  "
                f"mask_pct={self._correct_mask_percentile:.2f}  "
                f"sigma={self._correct_gauss_sigma:.2f}  "
                f"kernel={self._correct_gauss_kernel}"
            )
        finally:
            imgui.pop_text_wrap_pos()
            imgui.pop_style_color()
        imgui.spacing()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    def _draw_iso_deadpixel_readout(self, arr: Any) -> None:
        """Dead-pixel Run-tab block — raw only."""
        if arr.kind != "raw":
            return
        if self._selected_mode == _MODE_STITCHER:
            return
        from mbo_utilities.gui.widgets import isoview_deadpixel as dp_window

        imgui.text_colored(_SUBSECTION_COLOR, "Dead-pixel correction")
        set_tooltip(
            "Median-filter window + background percentile for "
            "dead-pixel detection in correct_stack. Edit opens a live "
            "preview with the flagged pixels highlighted.",
            align="right",
        )
        imgui.spacing()
        if imgui.button("Edit...##iso_dp_edit", imgui.ImVec2(_BTN_W, 0)):
            dp_window.open_window(self.parent)
        imgui.same_line()
        if imgui.button("Reset##iso_dp_reset", imgui.ImVec2(_BTN_W, 0)):
            self._correct_background_percentile = 5.0
            self._correct_median_kernel_size = 3
            self._correct_median_kernel_enabled = True
            self._correct_subtract_background = False
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.65, 1.0))
        imgui.push_text_wrap_pos(0.0)
        try:
            on = "on" if self._correct_median_kernel_enabled else "off"
            imgui.text_unformatted(
                f"median={on}  kernel={self._correct_median_kernel_size}  "
                f"bg_pct={self._correct_background_percentile:.2f}"
            )
        finally:
            imgui.pop_text_wrap_pos()
            imgui.pop_style_color()
        imgui.spacing()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    def _draw_iso_data_slicing(self, max_frames: int) -> None:
        """Data-slicing section — Set slice button + count preview.

        The row is labeled with the loaded array's T-axis slider name
        ("Tile" for tiled trees, "Timepoint" otherwise); the selected
        indices forwarded to the pipeline are unchanged.
        """
        tp_label, _, _ = resolve_dim_labels(self.parent)

        imgui.text_colored(_SUBSECTION_COLOR, "Data slicing")
        set_tooltip(
            f"Restrict which {tp_label.lower()} feed the run. Click Set "
            "slice to define a range using start:stop[:step] syntax "
            "(1-based, inclusive). Leaving the full range selects all.",
            align="right",
        )
        imgui.spacing()
        if imgui.button("Set slice##iso_slice", imgui.ImVec2(_BTN_W, 0)):
            self._iso_slicing_open = True

        n_tp = (
            self._iso_tp_parsed.count if self._iso_tp_parsed is not None else max_frames
        )
        imgui.text(f"{tp_label}: {n_tp}/{max_frames}")

        if self._selected_mode == _MODE_CORRECT:
            self._draw_iso_camera_subset()

        self._draw_iso_slicing_popup(max_frames, tp_label)
        imgui.spacing()
        imgui.spacing()

    def _draw_iso_slicing_popup(self, max_frames: int, tp_label: str) -> None:
        """Slicing popup — pick which timepoints to process."""
        popup_id = f"{tp_label}##iso_slice"
        if self._iso_slicing_open:
            imgui.open_popup(popup_id)
            self._iso_slicing_open = False

        imgui.set_next_window_size(
            imgui.ImVec2(520, 0),
            imgui.Cond_.first_use_ever,
        )
        if imgui.begin_popup(popup_id):
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.8, 0.2, 1.0),
                tp_label,
            )
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 30.0)
                imgui.text_unformatted(
                    "Format: start:stop or start:stop:step\n"
                    "Exclude: 1:100,50:60 = 1-100 excluding 50-60\n"
                    "1-based, inclusive."
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()
            imgui.dummy(imgui.ImVec2(0, 4))

            draw_selection_table(
                self,
                max_frames,
                1,
                tp_attr="_iso_tp",
                z_attr="_iso_z",
                id_suffix="_iso",
                num_channels=1,
                c_attr="_iso_c",
                tp_label=tp_label,
            )

            imgui.spacing()
            if imgui.button("Close##iso_slice_close", imgui.ImVec2(80, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

    def _available_cameras(self, arr: Any) -> list[int]:
        """Camera indices present in the loaded array. raw/corrected view_keys
        are plain camera ints; fused view_keys are tuples (take the first).
        """
        vk = arr._view_keys
        out: list[int] = []
        for v in vk:
            if isinstance(v, (tuple, list)) and v:
                out.append(int(v[0]))
            else:
                try:
                    out.append(int(v))
                except (TypeError, ValueError):
                    continue
        return sorted(set(out))

    def _selected_cameras(self, arr: Any) -> list[int] | None:
        """Selected camera indices, or None when the full set is selected (in
        which case the pipeline auto-detects every camera).
        """
        avail = self._available_cameras(arr)
        if len(avail) <= 1:
            return None
        sel = self._iso_selected_cameras
        if sel is None:
            return None
        chosen = [c for c in avail if c in sel]
        if not chosen or len(chosen) == len(avail):
            return None
        return chosen

    def _draw_iso_camera_subset(self) -> None:
        """Camera checkboxes — process only the checked cameras (correct mode)."""
        arr = self._get_array()
        cams = self._available_cameras(arr) if arr is not None else []
        if len(cams) <= 1:
            return
        if self._iso_selected_cameras is None or not (
            self._iso_selected_cameras <= set(cams)
        ):
            self._iso_selected_cameras = set(cams)
        imgui.spacing()
        _, _, view_label = resolve_dim_labels(self.parent)
        view_label = view_label if view_label.endswith("s") else f"{view_label}s"
        imgui.text_colored(_SUBSECTION_COLOR, view_label)
        set_tooltip("Process only the checked views.", align="right")
        from mbo_utilities.arrays.isoview.array import camera_view_label

        for i, c in enumerate(cams):
            checked = c in self._iso_selected_cameras
            changed, new = imgui.checkbox(
                f"{camera_view_label(c)}##iso_cam_{c}", checked
            )
            if changed:
                if new:
                    self._iso_selected_cameras.add(c)
                else:
                    self._iso_selected_cameras.discard(c)
            if i < len(cams) - 1:
                imgui.same_line()
        n_sel = len([c for c in cams if c in self._iso_selected_cameras])
        imgui.text_disabled(f"{n_sel}/{len(cams)} {view_label.lower()}")

    def _selected_timepoints(self) -> list[int] | None:
        """Return a 0-based list of selected timepoint indices, or None
        when the full range is selected (in which case the pipeline
        auto-detects all available timepoints).
        """
        parsed = self._iso_tp_parsed
        if parsed is None:
            return None
        count = int(parsed.count)
        max_tp = int(self._iso_last_max_tp)
        if count >= max_tp and not parsed.exclude_str:
            return None
        return [int(i) for i in parsed.final_indices]

    def draw_config(self) -> None:
        """Render the Suite2p-style side panel.

        Layout (top → bottom):
          1. Current dataset (path + Files popup + shape + size + dx/dy/dz/fs)
          2. Action (mode combo — shown only when multiple apply)
          3. Output options (format / compressor / workers / pyramid / overwrite)
          4. Data slicing (timepoint subset selector)
          5. Parameters and settings (small "Open" button → popup with
             algorithm-only knobs)
          6. Run button (big, green, centered)

        The output path is derived from the raw-data folder name in
        ``_ensure_defaults`` — no user-facing path picker.
        """
        arr = self._get_array()
        if arr is None:
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.5, 0.3, 1.0),
                "No isoview array loaded.",
            )
            return

        self._ensure_defaults(arr)

        if not _isoview_pkg_available():
            self._draw_isoview_missing_banner()

        modes = _available_modes(arr)
        if not modes:
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.5, 0.3, 1.0),
                f"No isoview action available for kind={arr.kind!r}.",
            )
            return
        if self._selected_mode not in modes:
            self._selected_mode = modes[0]

        crop_open_now = self.parent._iso_crop_window_open
        if (
            self._reopen_settings_after_crop
            and self._crop_was_open
            and not crop_open_now
        ):
            self._show_settings_popup = True
            self._popup_just_opened = True
            imgui.open_popup(
                "Isoview Pipeline Settings##isoview_pipeline_settings_popup"
            )
            self._reopen_settings_after_crop = False
        self._crop_was_open = crop_open_now

        self._draw_settings_popup(arr)

        max_frames = self._init_iso_selection_state(arr)

        imgui.spacing()
        self._draw_iso_current_dataset(arr)

        imgui.separator()
        imgui.spacing()

        label = "Processing Step"
        avail = imgui.get_content_region_avail().x
        label_w = imgui.calc_text_size(label).x
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + max(0.0, (avail - label_w) * 0.5)
        )
        imgui.text_colored(_SUBSECTION_COLOR, label)
        set_tooltip(
            "Which IsoView pipeline to run on this dataset.\n"
            "Correct/Fuse require `pip install isoview`.",
            align="right",
        )
        imgui.spacing()
        idx = modes.index(self._selected_mode) if self._selected_mode in modes else 0
        combo_w = 180.0
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + max(0.0, (avail - combo_w) * 0.5)
        )
        imgui.set_next_item_width(combo_w)
        if len(modes) > 1:
            changed, new_idx = imgui.combo("##iso_mode", idx, modes)
            if changed:
                self._selected_mode = modes[new_idx]
        else:
            imgui.begin_disabled()
            imgui.combo("##iso_mode", 0, modes)
            imgui.end_disabled()
        imgui.spacing()

        self._draw_iso_output_options(arr)
        imgui.separator()
        imgui.spacing()

        if not (self._selected_mode == _MODE_STITCHER and not arr.is_tiled):
            self._draw_iso_data_slicing(max_frames)
            imgui.separator()
            imgui.spacing()

        self._draw_iso_crop_readout(arr)

        self._draw_iso_segment_readout(arr)

        self._draw_iso_deadpixel_readout(arr)

        imgui.text_colored(_SUBSECTION_COLOR, "Parameters and settings")
        set_tooltip(
            "Open the algorithm-parameter popup: segmentation, blending, "
            "registration search, microscope overrides. File-I/O knobs "
            "live in the Output options section above.",
            align="right",
        )
        imgui.spacing()
        if imgui.button("Open##iso_pipe_settings", imgui.ImVec2(_BTN_W, 0)):
            self._show_settings_popup = True
            self._popup_just_opened = True
            imgui.open_popup(
                "Isoview Pipeline Settings##isoview_pipeline_settings_popup"
            )

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        has_path = bool(self._current_output_path())
        run_label = _RUN_LABELS.get(self._selected_mode or "", "Run")
        avail = imgui.get_content_region_avail().x
        btn_w = min(_RUN_W, max(80.0, avail))
        if avail > btn_w:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail - btn_w) * 0.5)
        with _green_button():
            if not has_path:
                imgui.begin_disabled()
            clicked = imgui.button(run_label, imgui.ImVec2(btn_w, 0))
            if not has_path:
                imgui.end_disabled()
        if not has_path and imgui.is_item_hovered(
            imgui.HoveredFlags_.allow_when_disabled
        ):
            imgui.set_tooltip("No output path resolved for this dataset.")
        if clicked and has_path:
            self._submit_active(arr)

        if self._last_status:
            imgui.spacing()
            color = (
                imgui.ImVec4(0.8, 0.4, 0.4, 1.0)
                if self._last_status.startswith(("Failed", "Output", "Dialog"))
                else imgui.ImVec4(0.5, 0.85, 0.5, 1.0)
            )
            imgui.text_colored(color, self._last_status)
