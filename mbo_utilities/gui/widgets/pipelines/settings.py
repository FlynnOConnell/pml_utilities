import json
import os
import pathlib
import threading
import time
from contextlib import contextmanager
from dataclasses import MISSING, dataclass, field
from pathlib import Path

import numpy as np
import psutil
from imgui_bundle import (
    hello_imgui,
    imgui,
    imgui_ctx,
)
from imgui_bundle import (
    icons_fontawesome_6 as fa,
)
from imgui_bundle import (
    portable_file_dialogs as pfd,
)

from mbo_utilities.arrays.features import get_dims
from mbo_utilities.arrays.features._slicing import parse_timepoint_selection
from mbo_utilities.gui._imgui_helpers import (
    PopupAutoSize,
    _popup_states,
    draw_boxed_label,
    set_tooltip,
    text_wrapped_cell,
    tooltip_marks_right,
)
from mbo_utilities.gui._selection_ui import (
    draw_frame_average_input,
    draw_selection_table,
    resolve_dim_labels,
    source_timepoints,
)
from mbo_utilities.gui.widgets.pipelines._s2p_schema import is_default as _is_default
from mbo_utilities.metadata import get_param
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.reader import widget_reader_kwargs

_GPU_DEV_CACHE: list | None = None


def _cached_gpu_devices() -> list:
    """nvidia-smi compute devices, probed once per session. [] on no GPU."""
    global _GPU_DEV_CACHE
    if _GPU_DEV_CACHE is None:
        from mbo_utilities.gpu import gpu_devices

        _GPU_DEV_CACHE = gpu_devices()
    return _GPU_DEV_CACHE


def _selected_compute_gpu(torch_device: str) -> str:
    """Text reporting the device the next run uses — read-only.

    The GPU is chosen in File > Options > Compute GPU; this only reports it,
    derived from CUDA_VISIBLE_DEVICES (set live from Options) + cached
    nvidia-smi names. No subprocess, no disk.
    """
    td = torch_device or "cuda"
    if td.startswith("cpu"):
        return "Compute: CPU"
    if td.startswith("mps"):
        return "Compute: MPS"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        return "Compute: CPU  (GPU off in File > Options)"
    if cvd and cvd.split(",")[0].strip().isdigit():
        idx = int(cvd.split(",")[0].strip())
        names = {
            int(d.get("index", i)): str(d.get("name", f"GPU {i}"))
            for i, d in enumerate(_cached_gpu_devices())
        }
        return f"Compute GPU: {names.get(idx, f'GPU {idx}')}"
    return "Compute GPU: auto (default device)"


_MODIFIED_COLOR = imgui.ImVec4(1.0, 0.72, 0.40, 1.0)

_MBO_ONLY_COLOR = imgui.ImVec4(0.55, 0.85, 0.55, 1.0)

_S2P_TITLE_COLOR = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)

_SUBSECTION_COLOR = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)

_LOAD_WARN_COLOR = imgui.ImVec4(1.00, 0.85, 0.30, 1.0)
_LOAD_BAD_COLOR = imgui.ImVec4(1.00, 0.40, 0.40, 1.0)


def _has_gpu_torch() -> bool:
    """True if the installed PyTorch is a GPU build (CUDA / ROCm / Intel XPU),
    or macOS (MPS).

    Reads ``torch/version.py`` — the build's ``cuda``/``hip``/``xpu`` fields,
    which are ``None`` on a CPU-only wheel — WITHOUT importing torch (an
    import is multi-second and would freeze the GUI thread). The version
    *string* alone is ambiguous (a plain '2.12.0' can be either build), so
    the accelerator fields are the reliable signal. ~0.3 ms; reflects what a
    fresh worker process will actually load. Only the definitive CPU-only
    case warns; anything indeterminate does not.
    """
    import sys

    if sys.platform == "darwin":
        return True
    import importlib.util
    import re

    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return True
    text = (Path(spec.origin).parent / "version.py").read_text()
    for accelerator in ("cuda", "hip", "xpu"):
        m = re.search(rf"^{accelerator}\b[^=\n]*=\s*(.+?)\s*$", text, re.M)
        if m and m.group(1).strip() not in ("None", ""):
            return True
    return False


def _detect_physical_cores() -> int:
    """Best-effort physical-core count. Falls back to os.cpu_count() // 2
    assuming HT/SMT is enabled (the common case on dev machines).
    """
    n = psutil.cpu_count(logical=False)
    if n:
        return int(n)
    cpu = os.cpu_count() or 4
    return max(1, cpu // 2)


def _default_workers() -> int:
    """Sensible default workers count. Aim for workers*2 (threads default)
    <= physical_cores so a fresh-install user lands in the green zone.
    """
    physical = _detect_physical_cores()
    return max(1, min(4, physical // 2))


def _parallel_load_status(workers: int, threads_per_worker: int) -> tuple:
    """Return (color_or_None, tooltip) for current workers/threads_per_worker.

    color is None when safe (caller renders the marker dimmed). Yellow when
    load fits in logical but not physical cores; red when it exceeds logical.
    """
    physical = _detect_physical_cores()
    logical = os.cpu_count() or physical
    eff_workers = workers if workers > 0 else max(1, physical // 2)
    eff_threads = threads_per_worker if threads_per_worker > 0 else 1
    load = eff_workers * eff_threads

    if load <= physical:
        return None, f"{load} threads on {physical} cores"
    if load <= logical:
        return _LOAD_WARN_COLOR, f"{load} threads > {physical} physical (borderline)"
    return _LOAD_BAD_COLOR, f"{load} threads > {logical} logical (oversubscribed)"


def _parallel_profiles() -> dict[str, tuple[int, int]]:
    """Profile name -> (workers, threads_per_worker), sized to this host.

    Parallel profiles favor plane parallelism (more workers, one thread
    each); Sequential runs one plane on every physical core; Max
    oversubscribes with logical cores. Drives the "Profile" selector,
    which fills the workers / threads fields in one click.
    """
    physical = _detect_physical_cores()
    logical = os.cpu_count() or physical
    return {
        "Sequential": (1, max(1, physical)),
        "Low": (max(1, physical // 4), 1),
        "Medium": (max(1, physical // 2), 1),
        "High": (max(1, physical), 1),
        "Max": (max(1, logical), 1),
    }


_PROFILE_HELP = (
    "workers = parallel plane processes\n"
    "threads/worker = BLAS/OMP threads within one plane\n"
    "load = workers x threads; green when load <= physical cores\n"
    "\n"
    "- Low/Medium/High: cores go to workers, threads=1\n"
    "  (maximize planes running at once)\n"
    "- Sequential: 1 worker x all cores (one plane at a time)\n"
    "- Max: workers = logical cores, so load > physical\n"
    "\n"
    "Best when planes >= workers. If planes < workers the extra\n"
    "workers sit idle - prefer fewer workers, more threads/worker."
)


_IMPORTANT_FIELDS: set[str] = {
    "algorithm",
    "tau",
    "threshold_scaling",
    "two_step_registration",
    "cellprob_threshold",
    "flow_threshold",
    "cellpose_img",
    "highpass_time",
}


@contextmanager
def _ghost_button():
    """Faint white button tint at rest, stronger on hover and active."""
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(1, 1, 1, 0.05))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1, 1, 1, 0.10))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(1, 1, 1, 0.16))
    try:
        yield
    finally:
        imgui.pop_style_color(3)


_HAS_LSP: bool | None = None


def _check_lsp_available() -> bool:
    """Check if lbm_suite2p_python is available.

    Caches a positive result for the session (find_spec is cheap but a
    True answer can't change once we've seen it), but ALWAYS re-probes
    after a negative or unknown answer — that way `pip install
    lbm_suite2p_python` mid-session is picked up on the next click
    instead of leaving the user stuck behind a stale False cache.
    """
    global _HAS_LSP
    if _HAS_LSP is True:
        return True
    import importlib.util

    found = importlib.util.find_spec("lbm_suite2p_python") is not None
    if found:
        _HAS_LSP = True
    return found


_HAS_RASTERMAP: bool | None = None


def _check_rastermap_available() -> bool:
    """True if `rastermap` is importable. Caches a positive result."""
    global _HAS_RASTERMAP
    if _HAS_RASTERMAP is True:
        return True
    import importlib.util

    found = importlib.util.find_spec("rastermap") is not None
    if found:
        _HAS_RASTERMAP = True
    return found


def build_cell_filters(
    min_um_enabled: bool,
    min_um: float,
    max_um_enabled: bool,
    max_um: float,
    *,
    baseline_filter_enabled: bool = False,
    baseline_reject_negative_F0: bool = False,
    baseline_min_F0_abs_enabled: bool = False,
    baseline_min_F0_abs: float = 0.0,
    baseline_min_F0_rel_enabled: bool = False,
    baseline_min_F0_rel: float = 0.0,
    correct_neuropil: bool = True,
) -> list[dict]:
    """Compose lsp's `cell_filters` list from the GUI's paired checkbox+value fields.

    Emits one entry per active criterion. lsp runs filters in apply order
    (each one writes its own 14<letter>_filter_<name>.png) so the order
    here is the order the user sees in the output dir.

    Possible entries:
      - `{"name": "max_diameter", ...}` — diameter bounds.
      - `{"name": "negative_baseline", "correct_neuropil": ...}` — drop ROIs
        whose rolling F0 dips below zero anywhere.
      - `{"name": "min_baseline_abs", "correct_neuropil": ..., "min_F0_abs":
        ...}` — drop ROIs whose median rolling F0 is below an absolute
        photon-count floor.
      - `{"name": "min_baseline_rel", "correct_neuropil": ..., "min_F0_rel":
        ...}` — drop ROIs whose minimum rolling F0 is below this fraction
        of median(F_raw).

    The dff window/percentile that each baseline filter scores against is
    resolved at the lsp layer from ops/pipeline kwargs (same values feeding
    the dF/F plot), so we don't thread those per-filter here.
    """
    out: list[dict] = []

    diameter: dict = {}
    if min_um_enabled and min_um > 0:
        diameter["min_diameter_um"] = float(min_um)
    if max_um_enabled and max_um > 0:
        diameter["max_diameter_um"] = float(max_um)
    if diameter:
        diameter["name"] = "max_diameter"
        out.append(diameter)

    if baseline_filter_enabled:
        cn = bool(correct_neuropil)
        if baseline_reject_negative_F0:
            out.append({"name": "negative_baseline", "correct_neuropil": cn})
        if baseline_min_F0_abs_enabled and baseline_min_F0_abs > 0:
            out.append(
                {
                    "name": "min_baseline_abs",
                    "correct_neuropil": cn,
                    "min_F0_abs": float(baseline_min_F0_abs),
                }
            )
        if baseline_min_F0_rel_enabled and baseline_min_F0_rel > 0:
            out.append(
                {
                    "name": "min_baseline_rel",
                    "correct_neuropil": cn,
                    "min_F0_rel": float(baseline_min_F0_rel),
                }
            )

    return out


def _build_planar_sub(extras) -> dict:
    """Compose the planar sub-dict for lsp's rastermap_kwargs.

    Sentinel values (0 for ints, -1 for locality) signal "omit — use
    lsp's cell-count-aware default for this field". An empty dict is
    valid and means "use defaults for ALL params" — still on.
    """
    sub: dict = {}
    if extras.rastermap_planar_n_clusters > 0:
        sub["n_clusters"] = int(extras.rastermap_planar_n_clusters)
    if extras.rastermap_planar_n_pcs > 0:
        sub["n_PCs"] = int(extras.rastermap_planar_n_pcs)
    if extras.rastermap_planar_locality >= 0:
        sub["locality"] = float(extras.rastermap_planar_locality)
    return sub


def _build_volumetric_sub(extras) -> dict:
    """Compose the volumetric sub-dict for lsp's rastermap_kwargs."""
    sub: dict = {}
    if extras.rastermap_volumetric_n_clusters > 0:
        sub["n_clusters"] = int(extras.rastermap_volumetric_n_clusters)
    if extras.rastermap_volumetric_n_pcs > 0:
        sub["n_PCs"] = int(extras.rastermap_volumetric_n_pcs)
    return sub


def build_rastermap_kwargs(extras) -> dict | None:
    """Compose lsp's unified rastermap_kwargs for `pipeline()`.

    Returns
    -------
      None — when rastermap_mode is Skip, OR when neither sub-mode is
             enabled (lsp treats None as "both off").
      {"planar": {...}}                   — planar only.
      {"volumetric": {...}}               — volumetric only.
      {"planar": {...}, "volumetric": ...} — both.
    """
    if extras.rastermap_mode == 0:
        return None
    out: dict = {}
    if extras.rastermap_planar:
        out["planar"] = _build_planar_sub(extras)
    if extras.rastermap_volumetric:
        out["volumetric"] = _build_volumetric_sub(extras)
    return out or None


def build_planar_rastermap_kwargs(extras) -> dict | None:
    """Compose the planar-only kwargs for `run_plane()`.

    `run_plane` doesn't have a volumetric mode, so we hand it just the
    planar sub-dict (or None to disable). Empty dict = "on, use defaults".
    """
    if extras.rastermap_mode == 0 or not extras.rastermap_planar:
        return None
    return _build_planar_sub(extras)


def collect_modified_params(
    s2p, s2p_db, s2p_extras=None
) -> list[tuple[str, object, object, str]]:
    """Return modified parameters across both pipelines.

    Each tuple is `(field, current, default, source)` where `source` is
    `"s2p"` for upstream-mapped fields (Suite2pSettings / Suite2pDB) and
    `"lsp"` for MboSuite2pExtras fields (lbm_suite2p_python kwargs). s2p
    defaults come from `suite2p.parameters.SETTINGS` via the schema map;
    lsp defaults come from the dataclass field spec.

    Sorted: s2p group first (alphabetical), then lsp group (alphabetical).
    """
    from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
        _MBO_DB_TO_S2P,
        _MBO_TO_S2P,
        get_default,
        is_default,
    )

    s2p_rows: list[tuple[str, object, object, str]] = []
    for name in _MBO_TO_S2P:
        cur = getattr(s2p, name)
        if not is_default(name, cur):
            s2p_rows.append((name, cur, get_default(name), "s2p"))
    for name in _MBO_DB_TO_S2P:
        cur = getattr(s2p_db, name)
        if not is_default(name, cur):
            s2p_rows.append((name, cur, get_default(name), "s2p"))
    s2p_rows.sort(key=lambda t: t[0])

    lsp_rows: list[tuple[str, object, object, str]] = []
    if s2p_extras is not None:
        import dataclasses as _dc

        for ef in _dc.fields(s2p_extras):
            if ef.default is not _dc.MISSING:
                default = ef.default
            elif ef.default_factory is not _dc.MISSING:
                default = ef.default_factory()
            else:
                continue
            cur = getattr(s2p_extras, ef.name)
            if cur != default:
                lsp_rows.append((ef.name, cur, default, "lsp"))
        lsp_rows.sort(key=lambda t: t[0])

    return s2p_rows + lsp_rows


@dataclass
class Suite2pSettings:
    """
    Suite2p pipeline processing settings — mirrors upstream
    `suite2p.default_settings()`.

    Attribute names match upstream canonical names. The dataclass is kept
    flat for GUI ergonomics; `to_dict()` produces the upstream-shaped
    nested dict (run / io / registration / detection / ... ).

    Paths, plane counts, and I/O config live on `Suite2pDB`. Mbo-only
    helper fields (dff_*, accept_all_cells, etc.) live on
    `MboSuite2pExtras`.
    """

    torch_device: str = "cuda"
    tau: float = 1.0
    fs: float = 10.0
    diameter_y: float = 12.0
    diameter_x: float = 12.0
    algorithm: str = "sparsery"

    do_registration: int = 1
    do_regmetrics: bool = False  # ~40 s per plane above 1500 frames
    do_detection: int = 1
    do_deconvolution: bool = True
    multiplane_parallel: bool = False

    combined: bool = True
    save_mat: bool = False
    save_NWB: bool = False
    save_ops_orig: bool = True
    delete_bin: bool = False
    move_bin: bool = False

    align_by_chan: int = 1  # 1-based; serialized as the bool align_by_chan2
    nimg_init: int = 400
    maxregshift: float = 0.1
    do_bidiphase: bool = False
    bidiphase: float = 0.0
    reg_batch_size: int = 100
    nonrigid: bool = True
    maxregshiftNR: int = 5
    block_size_y: int = 128
    block_size_x: int = 128
    smooth_sigma_time: float = 0.0
    smooth_sigma: float = 1.15
    spatial_taper: float = 3.45
    th_badframes: float = 1.0
    norm_frames: bool = True
    snr_thresh: float = 1.2
    subpixel: int = 10
    two_step_registration: bool = False
    reg_tif: bool = False
    reg_tif_chan2: bool = False

    denoise: bool = False
    det_block_size_y: int = 64
    det_block_size_x: int = 64
    nbins: int = 5000
    bin_size: int | None = None  # None: suite2p uses tau * fs
    highpass_time: int = 100
    threshold_scaling: float = 1.0
    npix_norm_min: float = 0.0
    npix_norm_max: float = 100.0
    max_overlap: float = 0.75
    soma_crop: bool = True
    chan2_threshold: float = 0.25
    cellpose_chan2: bool = False

    highpass_neuropil: int = 25
    max_ROIs: int = 5000
    spatial_scale: int = 0
    active_percentile: float = 0.0

    connected: bool = True
    max_iterations: int = 20
    smooth_masks: bool = False

    cellpose_model: str = "cpsam"
    cellpose_img: str = "max_proj / meanImg"
    highpass_spatial: float = 0.0
    flow_threshold: float = 0.0  # LBM default, flow check off; suite2p ships 0.4
    cellprob_threshold: float = -4.0  # LBM default; suite2p ships 0.0
    cellpose_niter: int = 0

    classifier_path: str | None = None
    use_builtin_classifier: bool = False
    preclassify: float = 0.0

    snr_threshold: float = 0.0
    extract_batch_size: int = 500
    neuropil_extract: bool = True
    neuropil_coefficient: float = 0.7
    inner_neuropil_radius: int = 2
    min_neuropil_pixels: int = 350
    lam_percentile: float = 50.0
    allow_overlap: bool = False
    circular_neuropil: bool = False

    baseline: str = "maximin"
    win_baseline: float = 60.0
    sig_baseline: float = 10.0
    prctile_baseline: float = 8.0

    def to_dict(self) -> dict:
        """Return the upstream-shaped nested settings dict."""
        return {
            "torch_device": self.torch_device,
            "tau": self.tau,
            "fs": self.fs,
            "diameter": [float(self.diameter_y), float(self.diameter_x)],
            "run": {
                "do_registration": int(self.do_registration),
                "do_regmetrics": self.do_regmetrics,
                "do_detection": bool(self.do_detection),
                "do_deconvolution": self.do_deconvolution,
                "multiplane_parallel": self.multiplane_parallel,
            },
            "io": {
                "combined": self.combined,
                "save_mat": self.save_mat,
                "save_NWB": self.save_NWB,
                "save_ops_orig": self.save_ops_orig,
                "delete_bin": self.delete_bin,
                "move_bin": self.move_bin,
            },
            "registration": {
                "align_by_chan2": bool(self.align_by_chan == 2),
                "nimg_init": self.nimg_init,
                "maxregshift": self.maxregshift,
                "do_bidiphase": self.do_bidiphase,
                "bidiphase": self.bidiphase,
                "batch_size": self.reg_batch_size,
                "nonrigid": self.nonrigid,
                "maxregshiftNR": self.maxregshiftNR,
                "block_size": (int(self.block_size_y), int(self.block_size_x)),
                "smooth_sigma_time": self.smooth_sigma_time,
                "smooth_sigma": self.smooth_sigma,
                "spatial_taper": self.spatial_taper,
                "th_badframes": self.th_badframes,
                "norm_frames": self.norm_frames,
                "snr_thresh": self.snr_thresh,
                "subpixel": self.subpixel,
                "two_step_registration": self.two_step_registration,
                "reg_tif": self.reg_tif,
                "reg_tif_chan2": self.reg_tif_chan2,
            },
            "detection": {
                "algorithm": self.algorithm,
                "denoise": self.denoise,
                "block_size": (int(self.det_block_size_y), int(self.det_block_size_x)),
                "nbins": self.nbins,
                "bin_size": self.bin_size,
                "highpass_time": self.highpass_time,
                "threshold_scaling": self.threshold_scaling,
                "npix_norm_min": self.npix_norm_min,
                "npix_norm_max": self.npix_norm_max,
                "max_overlap": self.max_overlap,
                "soma_crop": self.soma_crop,
                "chan2_threshold": self.chan2_threshold,
                "cellpose_chan2": self.cellpose_chan2,
                "sparsery_settings": {
                    "highpass_neuropil": self.highpass_neuropil,
                    "max_ROIs": self.max_ROIs,
                    "spatial_scale": self.spatial_scale,
                    "active_percentile": self.active_percentile,
                },
                "sourcery_settings": {
                    "connected": self.connected,
                    "max_iterations": self.max_iterations,
                    "smooth_masks": self.smooth_masks,
                },
                "cellpose_settings": {
                    "cellpose_model": self.cellpose_model,
                    "img": self.cellpose_img,
                    "highpass_spatial": self.highpass_spatial,
                    "flow_threshold": self.flow_threshold,
                    "cellprob_threshold": self.cellprob_threshold,
                    "params": (
                        {"niter": int(self.cellpose_niter)}
                        if self.cellpose_niter and self.cellpose_niter > 0
                        else None  # NOTE: None keeps cellpose's own niter default; {} would not
                    ),
                },
            },
            "classification": {
                "classifier_path": self.classifier_path,
                "use_builtin_classifier": self.use_builtin_classifier,
                "preclassify": self.preclassify,
            },
            "extraction": {
                "snr_threshold": self.snr_threshold,
                "batch_size": self.extract_batch_size,
                "neuropil_extract": self.neuropil_extract,
                "neuropil_coefficient": self.neuropil_coefficient,
                "inner_neuropil_radius": self.inner_neuropil_radius,
                "min_neuropil_pixels": self.min_neuropil_pixels,
                "lam_percentile": self.lam_percentile,
                "allow_overlap": self.allow_overlap,
                "circular_neuropil": self.circular_neuropil,
            },
            "dcnv_preprocess": {
                "baseline": self.baseline,
                "win_baseline": self.win_baseline,
                "sig_baseline": self.sig_baseline,
                "prctile_baseline": self.prctile_baseline,
            },
        }


@dataclass
class Suite2pDB:
    """
    Suite2p input/output database — mirrors upstream `suite2p.default_db()`.
    These are paths, plane counts, and I/O flags that identify *what* to
    process (not *how*).
    """

    data_path: list | None = None
    look_one_level_down: bool = False
    input_format: str = "tif"
    keep_movie_raw: bool = False
    nplanes: int = 1
    nrois: int = 1
    nchannels: int = 1
    swap_order: bool = False
    functional_chan: int = 1
    subfolders: list | None = None
    save_path0: str = ""
    fast_disk: str = ""
    save_folder: str = "suite2p"
    h5py_key: str = "data"
    nwb_driver: str = ""
    nwb_series: str = ""
    force_sktiff: bool = False
    bruker_bidirectional: bool = False

    def __post_init__(self):
        if self.data_path is None:
            self.data_path = []
        if self.subfolders is None:
            self.subfolders = []

    def to_dict(self) -> dict:
        return {
            "data_path": list(self.data_path or []),
            "look_one_level_down": self.look_one_level_down,
            "input_format": self.input_format,
            "keep_movie_raw": self.keep_movie_raw,
            "nplanes": self.nplanes,
            "nrois": self.nrois,
            "nchannels": self.nchannels,
            "swap_order": self.swap_order,
            "functional_chan": self.functional_chan,
            "subfolders": list(self.subfolders or []),
            "save_path0": self.save_path0,
            "fast_disk": self.fast_disk,
            "save_folder": self.save_folder,
            "h5py_key": self.h5py_key,
            "nwb_driver": self.nwb_driver,
            "nwb_series": self.nwb_series,
            "force_sktiff": self.force_sktiff,
            "bruker_bidirectional": self.bruker_bidirectional,
        }


@dataclass
class MboSuite2pExtras:
    """
    Fork/mbo-specific fields that have no upstream home. Kept on a
    dedicated object so `Suite2pSettings` can stay in lockstep with
    upstream's schema.

    These are consumed by `lbm_suite2p_python.run_plane`'s own kwargs or
    by mbo's pre/post-processing (dF/F, target_timepoints clipping).
    """

    accept_all_cells: bool = False
    norm_method: str = "dff"
    dff_window_size: int = 300
    dff_percentile: int = 20
    dff_smooth_window: int = 0

    save_json: bool = False

    min_diameter_um_enabled: bool = False
    min_diameter_um: float = 0.0
    max_diameter_um_enabled: bool = False
    max_diameter_um: float = 0.0

    baseline_filter_enabled: bool = True
    baseline_reject_negative_F0: bool = False
    baseline_min_F0_abs_enabled: bool = False
    baseline_min_F0_abs: float = 1.0
    baseline_min_F0_rel_enabled: bool = False
    baseline_min_F0_rel: float = 0.0

    correct_neuropil: bool = False

    rastermap_mode: int = 0

    rastermap_planar: bool = False
    rastermap_volumetric: bool = False

    rastermap_planar_n_clusters: int = 0
    rastermap_planar_n_pcs: int = 0
    rastermap_planar_locality: float = -1.0

    rastermap_volumetric_n_clusters: int = 40
    rastermap_volumetric_n_pcs: int = 0

    target_timepoints: int = -1
    frames_include: int = -1

    workers: int = field(default_factory=_default_workers)
    threads_per_worker: int = 2  # NOTE: caps BLAS/OMP/numba/torch per worker; without it SVD registration is ~10x slower
    skip_volumetric: bool = False

    aspect: float = 1.0

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()  # type: ignore
        }


def _init_s2p_selection_state(self):
    """Reset the suite2p slice state when the file or its extent changes; returns (max_frames, num_planes, num_channels)."""
    max_frames = source_timepoints(self)
    num_planes, num_channels = self.nz, self.nc
    current_fpath = self.fpath[0] if isinstance(self.fpath, list) else self.fpath
    current_fpath = str(current_fpath) if current_fpath else ""
    if self._s2p_last_fpath != current_fpath:
        self._s2p_last_fpath = current_fpath
        self._s2p_last_max_tp = None
        self._s2p_last_num_planes = None
        self._s2p_last_num_channels = None
        src_fs = self._custom_metadata.get("fs")
        if src_fs is None:
            src_fs = get_param(self.image_widget.data[0].metadata, "fs")
        if src_fs is not None and self.s2p is not None:
            self.s2p.fs = float(src_fs)
    if self._s2p_last_max_tp != max_frames:
        self._s2p_last_max_tp = max_frames
        self._s2p_tp_selection = f"1:{max_frames}"
        self._s2p_tp_parsed = None
        self._s2p_tp_error = ""
    if self._s2p_last_num_planes != num_planes:
        self._s2p_last_num_planes = num_planes
        self._s2p_z_start, self._s2p_z_stop, self._s2p_z_step = 1, num_planes, 1
    if self._s2p_last_num_channels != num_channels:
        self._s2p_last_num_channels = num_channels
        self._s2p_c_start, self._s2p_c_stop, self._s2p_c_step = 1, num_channels, 1

    if num_planes > 1:
        self._selected_planes = set(
            range(self._s2p_z_start, self._s2p_z_stop + 1, self._s2p_z_step)
        )
    else:
        self._selected_planes = {1}

    if self._s2p_tp_parsed is None and not self._s2p_tp_error:
        try:
            self._s2p_tp_parsed = parse_timepoint_selection(
                self._s2p_tp_selection, max_frames
            )
        except ValueError as e:
            self._s2p_tp_error = str(e)

    return max_frames, num_planes, num_channels


def _draw_s2p_selection_preview(self, max_frames, num_planes, num_channels=1):
    """Draw frame / plane / channel selection counts as one line per field
    (matches the Current dataset info layout — non-colored, separate rows).
    """
    if self._s2p_tp_parsed:
        n_frames = self._s2p_tp_parsed.count
    else:
        n_frames = max_frames

    n_planes = len(range(self._s2p_z_start, self._s2p_z_stop + 1, self._s2p_z_step))
    n_channels = len(range(self._s2p_c_start, self._s2p_c_stop + 1, self._s2p_c_step))

    imgui.text(f"Frames: {n_frames}")
    if num_planes > 1:
        imgui.text(f"Planes: {n_planes}")
    if num_channels > 1:
        imgui.text(f"Channels: {n_channels}")


def _draw_s2p_slicing_popup(self):
    """Slicing popup — pick which timepoints, z-planes, and channels to process."""
    if self._s2p_slicing_open:
        imgui.open_popup("Frames & Planes##s2p_slice")
        self._s2p_slicing_open = False

    imgui.set_next_window_size(imgui.ImVec2(520, 0), imgui.Cond_.first_use_ever)
    if imgui.begin_popup("Frames & Planes##s2p_slice"):
        max_frames = self._s2p_last_max_tp
        num_planes = self._s2p_last_num_planes
        num_channels = self._s2p_last_num_channels

        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Frames & Planes")
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 30.0)
            imgui.text_unformatted(
                "Format: start:stop or start:stop:step\n"
                "Exclude: 1:100,50:60 = 1-100 excluding 50-60"
            )
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        imgui.dummy(imgui.ImVec2(0, 4))

        tp_label, z_label, c_label = resolve_dim_labels(self)
        tp_parsed, *_rest = draw_selection_table(
            self,
            max_frames,
            num_planes,
            tp_attr="_s2p_tp",
            z_attr="_s2p_z",
            id_suffix="_s2p",
            num_channels=num_channels,
            c_attr="_s2p_c",
            tp_label=tp_label,
            z_label=z_label,
            c_label=c_label,
        )

        if tp_parsed:
            self.s2p_extras.target_timepoints = (
                tp_parsed.count if tp_parsed.count < max_frames else -1
            )

        imgui.spacing()
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


def _draw_data_options_content(self):
    """Draw data options content showing settings that affect Suite2p processing."""
    INPUT_WIDTH = 100
    has_phase_support = self.has_raster_scan_support
    nz = self.nz

    is_raw = self.is_mbo_scan
    has_z_reg = nz > 1 and is_raw
    source_nt = source_timepoints(self)
    has_frame_average = source_nt > 1

    has_any_options = has_phase_support or has_z_reg or has_frame_average

    if has_phase_support:
        imgui.text("Scan-Phase Correction")
        imgui.spacing()

        phase_changed, phase_value = imgui.checkbox("Fix Phase", self._s2p_fix_phase)
        set_tooltip("Apply bidirectional scan-phase correction to output data")
        if phase_changed:
            self._s2p_fix_phase = phase_value

        fft_changed, fft_value = imgui.checkbox("Sub-Pixel (FFT)", self._s2p_use_fft)
        set_tooltip("Use FFT-based sub-pixel registration (slower but more accurate)")
        if fft_changed:
            self._s2p_use_fft = fft_value

        imgui.set_next_item_width(INPUT_WIDTH)
        border_changed, border_val = imgui.input_int("Border (px)", self.border, step=1)
        set_tooltip("Pixels to exclude from edges when computing phase offset")
        if border_changed:
            self.border = max(0, border_val)

        imgui.set_next_item_width(INPUT_WIDTH)
        max_offset_changed, max_offset_val = imgui.input_int(
            "Max Offset", self.max_offset, step=1
        )
        set_tooltip("Maximum allowed pixel shift for phase correction")
        if max_offset_changed:
            self.max_offset = max(1, max_offset_val)

        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Current offsets:")
        for i, ofs in enumerate(self.current_offset):
            imgui.text(f"  Array {i + 1}: {ofs:.3f} px")

    if has_frame_average:
        if has_phase_support:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
        imgui.text("Temporal Binning")
        imgui.spacing()
        self._s2p_frame_average = draw_frame_average_input(
            "Frame Average##s2p_frame_average",
            self._s2p_frame_average,
            viewer_factor=self.frame_average,
            max_frames=source_nt,
        )

    if has_z_reg:
        if has_phase_support or has_frame_average:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        imgui.text("Axial Registration")
        imgui.spacing()

        if has_z_reg:
            reg_changed, reg_value = imgui.checkbox(
                "Register Z-Planes Axially", self._register_z
            )
            set_tooltip(
                "Compute per-plane rigid shifts via phase correlation and\n"
                "apply them on write. Corrects z-drift between adjacent planes.\n"
                "Uses GPU (cupy) if available, otherwise CPU."
            )
            if reg_changed:
                self._register_z = reg_value

            if self._register_z:
                imgui.indent(16)
                imgui.set_next_item_width(80)
                _changed, _val = imgui.input_int(
                    "Max frames", self._axial_max_frames, 50, 100
                )
                if _changed:
                    self._axial_max_frames = max(10, _val)
                set_tooltip(
                    "Frames subsampled (evenly spaced) for the time-mean used\n"
                    "to compute plane shifts. 200 is usually plenty."
                )

                imgui.set_next_item_width(80)
                _changed, _val = imgui.input_int(
                    "Max shift (px)", self._axial_max_reg_xy, 10, 50
                )
                if _changed:
                    self._axial_max_reg_xy = max(1, _val)
                set_tooltip("Max shift search radius in pixels. Default 30.")
                imgui.unindent(16)
        else:
            imgui.begin_disabled()
            imgui.checkbox("Register Z-Planes Axially", False)
            imgui.end_disabled()
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
                "Requires multi-plane raw data",
            )

    if not has_any_options:
        imgui.text_disabled("No available options for this data type.")


_MISSING_COLOR = imgui.ImVec4(1.0, 0.45, 0.45, 1.0)


def _format_size(size_bytes: int | None) -> str:
    """Render a byte count as human-friendly string."""
    if size_bytes is None or size_bytes < 0:
        return "—"
    units = ("B", "KB", "MB", "GB", "TB")
    n = float(size_bytes)
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    if i == 0:
        return f"{int(n)} {units[i]}"
    return f"{n:.2f} {units[i]}"


def _compute_dataset_size_sync(filenames: list) -> int:
    """Recursive sum of bytes for ``filenames``.

    Handles directory-format containers (``.zarr``, ``.ome.zarr``,
    suite2p plane dirs) by walking their contents — ``Path.stat()`` on
    a directory only reports the dir-entry size, not the tree.
    """
    from stat import S_ISDIR

    total = 0
    for f in filenames:
        try:
            p = Path(f)
            st = p.stat()
            if S_ISDIR(st.st_mode):
                for child in p.rglob("*"):
                    try:
                        cst = child.stat()
                        if not S_ISDIR(cst.st_mode):
                            total += cst.st_size
                    except OSError:
                        continue
            else:
                total += st.st_size
        except OSError:
            continue
    return total


def _dataset_size_disk_cache_path(key: tuple[str, ...]) -> Path:
    """Per-dataset cache path; hashed to keep the filename short."""
    import hashlib

    h = hashlib.blake2b(
        "\n".join(key).encode("utf-8", errors="replace"), digest_size=16
    ).hexdigest()
    override = os.environ.get("MBO_CACHE_DIR")
    if override:
        base = Path(override)
    else:
        from mbo_utilities.preferences import get_mbo_dirs

        base = get_mbo_dirs()["cache"]
    return base / f"dataset_size_{h}.json"


def _dataset_size_load_disk(key: tuple[str, ...]) -> int | None:
    p = _dataset_size_disk_cache_path(key)
    try:
        with p.open() as f:
            data = json.load(f)
        if data.get("key") == list(key) and isinstance(data.get("size"), int):
            return int(data["size"])
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _dataset_size_save_disk(key: tuple[str, ...], size: int) -> None:
    p = _dataset_size_disk_cache_path(key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump({"key": list(key), "size": int(size)}, f)
        tmp.replace(p)
    except OSError:
        pass


_DATASET_SIZE_CACHE: dict[tuple[str, ...], int] = {}
_DATASET_SIZE_PENDING: set[tuple[str, ...]] = set()


def _dataset_size_worker(key: tuple[str, ...], files: list) -> None:
    size = _compute_dataset_size_sync(files)
    _dataset_size_save_disk(key, size)
    _DATASET_SIZE_CACHE[key] = size
    _DATASET_SIZE_PENDING.discard(key)


def _dataset_size_bytes(filenames: list) -> int | None:
    """Total bytes of ``filenames``, None while a daemon thread still computes it.

    Read once per dataset: in memory, then the on-disk cache under ``~/.mbo/cache``,
    else a daemon walks the files and the next frame finds the answer.
    """
    if not filenames:
        return None
    key = tuple(str(f) for f in filenames)
    if key in _DATASET_SIZE_CACHE:
        return _DATASET_SIZE_CACHE[key]
    disk = _dataset_size_load_disk(key)
    if disk is not None:
        _DATASET_SIZE_CACHE[key] = disk
        return disk
    if key in _DATASET_SIZE_PENDING:
        return None
    _DATASET_SIZE_PENDING.add(key)
    threading.Thread(
        target=_dataset_size_worker,
        args=(key, list(filenames)),
        daemon=True,
        name="dataset-size-walker",
    ).start()
    return None


def _truncate_to_width(text: str, max_width: float) -> str:
    """Front-truncate `text` (drop chars from the start, replace with …)
    so its rendered width fits within `max_width`. Keeps the tail of the
    string intact, which is what you want for paths (filename stays
    visible). Returns the original string when it already fits.
    """
    if max_width <= 0:
        return ""
    if imgui.calc_text_size(text).x <= max_width:
        return text
    ellipsis = "…"
    if imgui.calc_text_size(ellipsis).x >= max_width:
        return ""
    lo, hi = 0, len(text)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = ellipsis + text[-mid:] if mid > 0 else ellipsis
        if imgui.calc_text_size(cand).x <= max_width:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _draw_md_field(label: str, value, unit: str = "") -> None:
    """Render `label: value unit` on its own line. If value is None,
    render in red and attach a 'press Shift+M' tooltip.
    """
    if value is None:
        imgui.text_colored(_MISSING_COLOR, f"{label}: —")
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{label} not detected. Press Shift+M to set metadata.")
        return
    if isinstance(value, float):
        text = f"{label}: {value:.3f}".rstrip("0").rstrip(".")
    else:
        text = f"{label}: {value}"
    if unit:
        text += f" {unit}"
    imgui.text(text)


def _draw_dataset_files_popup(
    filenames: list,
    frames_per_file: list[int] | None = None,
    sizer: PopupAutoSize | None = None,
) -> None:
    """Modal popup listing each file in concatenation order.

    When `frames_per_file` is provided and length-matches `filenames`, a
    "Frames" column is added and the header shows the total frame count.
    The header row carries a ghost-styled icon button that copies the
    listing to the clipboard as JSON.
    """
    imgui.set_next_window_size(imgui.ImVec2(700, 450), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size_constraints(
        imgui.ImVec2(420, 240), imgui.ImVec2(1600, 1200)
    )
    _flags = (
        sizer.flags(imgui.WindowFlags_.no_saved_settings)
        if sizer is not None
        else imgui.WindowFlags_.no_saved_settings
    )
    opened = imgui.begin_popup_modal(
        "Dataset files##current_dataset_files_popup",
        flags=_flags,
    )[0]
    if not opened:
        return

    has_frames = isinstance(frames_per_file, (list, tuple)) and len(
        frames_per_file
    ) == len(filenames)

    if has_frames:
        imgui.text(
            f"{len(filenames)} files (concatenation order, "
            f"{sum(frames_per_file)} total frames):"
        )
    else:
        imgui.text(f"{len(filenames)} files (concatenation order):")
    imgui.same_line()
    avail = imgui.get_content_region_avail().x
    btn_w = imgui.get_frame_height()
    if avail > btn_w + 4:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - btn_w - 2)
    with _ghost_button():
        if imgui.button(
            f"{fa.ICON_FA_COPY}##dataset_files_copy",
            imgui.ImVec2(btn_w, btn_w),
        ):
            payload: dict = {
                "num_files": len(filenames),
                "file_paths": [str(f) for f in filenames],
            }
            if has_frames:
                payload["frames_per_file"] = list(frames_per_file)
                payload["num_timepoints"] = sum(frames_per_file)
            imgui.set_clipboard_text(json.dumps(payload, indent=2))
    set_tooltip(
        "Copy file list as JSON (file_paths, frames_per_file, totals).",
        show_mark=False,
    )

    imgui.separator()

    _reserved = imgui.get_frame_height_with_spacing() + 8
    if imgui.begin_child(
        "##current_dataset_files_list",
        imgui.ImVec2(-1, -_reserved),
        imgui.ChildFlags_.borders,
    ):
        n_cols = 3 if has_frames else 2
        if imgui.begin_table(
            "##dataset_files_table",
            n_cols,
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.sizing_stretch_prop,
        ):
            imgui.table_setup_column("#", imgui.TableColumnFlags_.width_fixed, 40)
            imgui.table_setup_column("File")
            if has_frames:
                imgui.table_setup_column(
                    "Frames", imgui.TableColumnFlags_.width_fixed, 80
                )
            imgui.table_headers_row()

            for i, f in enumerate(filenames):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(f"{i + 1}")
                imgui.table_set_column_index(1)
                imgui.text_wrapped(str(f))
                if has_frames:
                    imgui.table_set_column_index(2)
                    imgui.text(f"{frames_per_file[i]}")

            imgui.end_table()
        imgui.end_child()

    imgui.separator()
    if imgui.button("Close", imgui.ImVec2(80, 0)):
        imgui.close_current_popup()
    imgui.end_popup()


def _draw_current_dataset_section(self) -> None:
    """Draw the 'Current dataset' info block at the top of the run tab.

    Shows source path, a Files (N) button that opens a popup with the
    concatenation order, the array shape, total bytes on disk, plane /
    frame / channel counts, frame rate, and dx / dy / dz. Missing scalar
    metadata fields are tinted red with a Shift+M hint.
    """
    arr = self.image_widget.data[0] if self.image_widget.data else None

    imgui.text_colored(_SUBSECTION_COLOR, "Current dataset")
    imgui.spacing()
    imgui.spacing()
    if arr is None:
        imgui.text_disabled("(no file loaded)")
        imgui.spacing()
        imgui.spacing()
        return

    if isinstance(self.fpath, (list, tuple)):
        path_str = str(self.fpath[0]) if self.fpath else ""
    else:
        path_str = str(self.fpath) if self.fpath else ""
    avail_x = imgui.get_content_region_avail().x
    display_path = _truncate_to_width(path_str or "(in-memory)", avail_x)
    imgui.text_unformatted(display_path)
    if display_path != (path_str or "(in-memory)") and imgui.is_item_hovered():
        imgui.set_tooltip(path_str)

    filenames = list(getattr(arr, "filenames", []) or [])
    if not filenames:
        if isinstance(self.fpath, (list, tuple)):
            filenames = list(self.fpath)
        elif self.fpath:
            filenames = [self.fpath]

    md = dict(arr.metadata)
    custom = self._custom_metadata
    fs, dx, dy, dz = (
        custom[key] if custom.get(key) is not None else get_param(md, key)
        for key in ("fs", "dx", "dy", "dz")
    )

    shape = tuple(arr.shape)
    _dims = get_dims(arr)

    size_bytes = _dataset_size_bytes(filenames)

    n_files = len(filenames)
    _files_sizer = self._files_popup_sizer
    if _files_sizer is None:
        _files_sizer = PopupAutoSize(
            "Dataset files##current_dataset_files_popup",
            auto_resize=False,
        )
        self._files_popup_sizer = _files_sizer
    if imgui.button(f"Files ({n_files})##current_dataset_files"):
        _files_sizer.before_open()
        imgui.open_popup("Dataset files##current_dataset_files_popup")
    imgui.text(f"Size on disk: {_format_size(size_bytes)}")

    fpf = md.get("frames_per_file")
    if not (isinstance(fpf, (list, tuple)) and len(fpf) == n_files):
        fpf = None
    _draw_dataset_files_popup(filenames, fpf, sizer=_files_sizer)

    shape_text = " × ".join(str(s) for s in shape)
    if _dims and len(_dims) == len(shape):
        shape_text = f"{shape_text} [{','.join(_dims)}]"
    imgui.push_text_wrap_pos(0.0)
    try:
        imgui.text(f"Shape: {shape_text}")
    finally:
        imgui.pop_text_wrap_pos()

    _draw_md_field("Frame rate", fs, "Hz")
    _draw_md_field("dx", dx, "µm")
    _draw_md_field("dy", dy, "µm")
    _draw_md_field("dz", dz, "µm")

    imgui.spacing()
    imgui.spacing()


_INPUT_WIDTH = 80

_WORST_INPUT_LABEL = {
    "LBM-Suite2p-Python": "Max diameter (um)",
    "Registration": "Sampling Rate (Hz)",
    "ROI Detection": "Chan2 Detection Threshold",
    "Classification": "Preclassify threshold",
    "Extraction": "Inner Neuropil Radius",
    "Deconvolution": "Neuropil coefficient",
}


_WORST_CHECKBOX_LABEL = {
    "LBM-Suite2p-Python": "Skip volumetric figures",
    "Registration": "Export Registered TIFF",
    "ROI Detection": "Auto bin size (tau*fs)",
    "Classification": "Use built-in classifier",
    "Extraction": "Extract Neuropil",
    "Deconvolution": "",
}


_TITLE_RADIO_LABELS = {
    "tri": "Skip Run Force",
    "bool": "Skip Run",
    None: "",
}


_SELF_TITLED_COLUMNS = {"LBM-Suite2p-Python", "Registration"}


@contextmanager
def _hi(field: str, value):
    push_color = not _is_default(field, value)
    if push_color:
        imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
    try:
        yield
    finally:
        if push_color:
            imgui.pop_style_color()


@contextmanager
def _hi_extras(name: str, value):
    """Tint the next widget when an MboSuite2pExtras value differs from its default."""
    f = MboSuite2pExtras.__dataclass_fields__[name]
    default = f.default if f.default is not MISSING else f.default_factory()
    push_color = value != default
    if push_color:
        imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
    try:
        yield
    finally:
        if push_color:
            imgui.pop_style_color()


@contextmanager
def _mbo():
    """Tint a widget green (_MBO_ONLY_COLOR) to flag an mbo-only field
    sitting among suite2p widgets with no green group title above it
    (e.g. cellpose_niter). Always green regardless of value.
    """
    imgui.push_style_color(imgui.Col_.text, _MBO_ONLY_COLOR)
    try:
        yield
    finally:
        imgui.pop_style_color()


def _emp_label(self, field: str, value, text: str) -> None:
    """Render an external label for a widget — wrapped in a thin box
    with a bold font when `field` is in _IMPORTANT_FIELDS, plain text
    otherwise. Modified-color tint applies when `value` differs from
    the upstream default. Pair with `imgui.<widget>("##field", ...)`
    + `imgui.same_line(0, item_inner_spacing.x)` to replicate imgui's
    default inline-label placement.
    """
    push_color = not _is_default(field, value)
    is_important = field in _IMPORTANT_FIELDS
    if push_color:
        imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
    if is_important:
        draw_boxed_label(text, font=self._bold_font)
    else:
        imgui.text(text)
    if push_color:
        imgui.pop_style_color()


def _draw_lsp_settings(self):
    _box_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y

    imgui.begin_child("##lsp_box", imgui.ImVec2(-1, 0), _box_flags)
    imgui.text_colored(_MBO_ONLY_COLOR, "LBM-Suite2p-Python Settings")
    imgui.spacing()

    _, self._s2p_background = imgui.checkbox("Run in background", self._s2p_background)
    set_tooltip("Run as a separate process that continues after closing the GUI.")
    with _hi_extras("accept_all_cells", self.s2p_extras.accept_all_cells):
        _, self.s2p_extras.accept_all_cells = imgui.checkbox(
            "Accept all cells", self.s2p_extras.accept_all_cells
        )
    set_tooltip(
        "Keep every detected ROI regardless of classifier output. "
        "Forwarded to lbm_suite2p_python.pipeline(accept_all_cells=...)."
    )

    imgui.spacing()
    imgui.spacing()

    imgui.text_colored(_SUBSECTION_COLOR, "Parallel processing")
    imgui.same_line()
    _load_color, _load_tooltip = _parallel_load_status(
        self.s2p_extras.workers, self.s2p_extras.threads_per_worker
    )
    if _load_color is None:
        imgui.text_disabled("(?)")
    else:
        imgui.text_colored(_load_color, "(?)")
    if imgui.is_item_hovered():
        imgui.set_tooltip(_load_tooltip)

    _profiles = _parallel_profiles()
    _profile_names = list(_profiles) + ["Custom"]
    _cur_wt = (self.s2p_extras.workers, self.s2p_extras.threads_per_worker)
    _active = next((n for n, v in _profiles.items() if v == _cur_wt), "Custom")
    imgui.set_next_item_width(_INPUT_WIDTH)
    _pchanged, _pidx = imgui.combo(
        "Profile", _profile_names.index(_active), _profile_names
    )
    if _pchanged and _profile_names[_pidx] in _profiles:
        self.s2p_extras.workers, self.s2p_extras.threads_per_worker = _profiles[
            _profile_names[_pidx]
        ]
    set_tooltip(
        "Sizes workers x threads/worker to this machine. Sequential = "
        "one plane on all cores; Low/Medium/High/Max = more parallel "
        "planes. Custom = manual values below."
    )
    if imgui.small_button("Profile help##profile_help_btn"):
        imgui.open_popup("profile_help_popup")
    if imgui.begin_popup("profile_help_popup"):
        imgui.text(_PROFILE_HELP)
        imgui.end_popup()

    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi_extras("workers", self.s2p_extras.workers):
        _, self.s2p_extras.workers = imgui.input_int("Workers", self.s2p_extras.workers)
    set_tooltip("Zplane worker processes. 1 = sequential, 0 = auto.")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi_extras("threads_per_worker", self.s2p_extras.threads_per_worker):
        _, self.s2p_extras.threads_per_worker = imgui.input_int(
            "Threads/worker", self.s2p_extras.threads_per_worker
        )
    set_tooltip("BLAS / OMP / torch threads per worker. 0 = library default.")
    with _hi_extras("skip_volumetric", self.s2p_extras.skip_volumetric):
        _, self.s2p_extras.skip_volumetric = imgui.checkbox(
            "Skip volumetric figures", self.s2p_extras.skip_volumetric
        )
    set_tooltip("Skip merge, volume_stats, and volume plots.")

    imgui.spacing()
    imgui.spacing()

    imgui.text_colored(_SUBSECTION_COLOR, "Cell Filters")
    for _attr, _label, _tip in (
        (
            "min_diameter_um",
            "Min diameter (um)",
            "Drop ROIs whose fitted diameter is BELOW this in microns.",
        ),
        (
            "max_diameter_um",
            "Max diameter (um)",
            "Drop ROIs whose fitted diameter is ABOVE this in microns.",
        ),
    ):
        _enabled_attr = f"{_attr}_enabled"
        _enabled = getattr(self.s2p_extras, _enabled_attr)
        with _hi_extras(_enabled_attr, _enabled):
            _changed_e, _enabled = imgui.checkbox(f"##{_attr}_en", _enabled)
        if _changed_e:
            setattr(self.s2p_extras, _enabled_attr, _enabled)
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        imgui.begin_disabled(not _enabled)
        imgui.set_next_item_width(_INPUT_WIDTH)
        _val = getattr(self.s2p_extras, _attr)
        with _hi_extras(_attr, _val):
            _changed_v, _new_v = imgui.input_float(
                f"{_label}##{_attr}", _val, 0.0, 0.0, "%.1f"
            )
        imgui.end_disabled()
        if _changed_v:
            _new_v = max(0.0, _new_v)
            setattr(self.s2p_extras, _attr, _new_v)
            if _new_v == 0.0:
                setattr(self.s2p_extras, _enabled_attr, False)
        set_tooltip(_tip)

    imgui.spacing()
    imgui.spacing()
    imgui.text_colored(_SUBSECTION_COLOR, "Baseline Filter")

    with _hi_extras(
        "baseline_reject_negative_F0", self.s2p_extras.baseline_reject_negative_F0
    ):
        _, self.s2p_extras.baseline_reject_negative_F0 = imgui.checkbox(
            "Reject negative F0", self.s2p_extras.baseline_reject_negative_F0
        )
    set_tooltip(
        "Drop cells whose baseline ever goes below zero — strongest "
        "single predictor of catastrophic dF/F outliers."
    )

    for _attr, _label, _fmt, _tip in (
        (
            "baseline_min_F0_abs",
            "Min F0",
            "%.2f",
            "Drop cells whose median baseline falls below this absolute "
            "fluorescence value. Default 1.0; typical 0.5-5.0.",
        ),
    ):
        _enabled_attr = f"{_attr}_enabled"
        _enabled = getattr(self.s2p_extras, _enabled_attr)
        with _hi_extras(_enabled_attr, _enabled):
            _changed_e, _enabled = imgui.checkbox(f"##{_attr}_en", _enabled)
        if _changed_e:
            setattr(self.s2p_extras, _enabled_attr, _enabled)
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        imgui.begin_disabled(not _enabled)
        imgui.set_next_item_width(_INPUT_WIDTH)
        _val = getattr(self.s2p_extras, _attr)
        with _hi_extras(_attr, _val):
            _changed_v, _new_v = imgui.input_float(
                f"{_label}##{_attr}", _val, 0.0, 0.0, _fmt
            )
        imgui.end_disabled()
        if _changed_v:
            _new_v = max(0.0, _new_v)
            setattr(self.s2p_extras, _attr, _new_v)
            if _new_v == 0.0:
                setattr(self.s2p_extras, _enabled_attr, False)
        set_tooltip(_tip)

    imgui.spacing()
    imgui.spacing()

    imgui.text_colored(_SUBSECTION_COLOR, "Trace Normalization")
    _norm_methods = ["dff", "zscore"]
    _norm_labels = ["dF/F0", "Z-score"]
    _norm_idx = (
        _norm_methods.index(self.s2p_extras.norm_method)
        if self.s2p_extras.norm_method in _norm_methods
        else 0
    )
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi_extras("norm_method", self.s2p_extras.norm_method):
        _nm_changed, _nm_idx = imgui.combo(
            "Method##norm_method", _norm_idx, _norm_labels
        )
        if _nm_changed:
            self.s2p_extras.norm_method = _norm_methods[_nm_idx]
    set_tooltip(
        "Trace saved to norm_traces.npy.\n\n"
        "dF/F0 = (F - F0) / F0, F0 = rolling-percentile baseline.\n"
        "  Relative fluorescence change; default choice for transient "
        "detection and comparing event sizes within a cell.\n\n"
        "Z-score = (F - mean) / std, per ROI over time.\n"
        "  Unitless, centered at 0; use to compare activity patterns "
        "across cells with different brightness or noise."
    )
    _is_zscore = self.s2p_extras.norm_method == "zscore"
    with _hi_extras("correct_neuropil", self.s2p_extras.correct_neuropil):
        _, self.s2p_extras.correct_neuropil = imgui.checkbox(
            "Correct neuropil", self.s2p_extras.correct_neuropil
        )
    set_tooltip(
        "Use F - coef*Fneu in the normalized trace (default on). Off uses "
        "raw F. Shared with the baseline cell filter so both score the "
        "same trace."
    )
    imgui.begin_disabled(_is_zscore)
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi_extras("dff_window_size", self.s2p_extras.dff_window_size):
        _, self.s2p_extras.dff_window_size = imgui.input_int(
            "Window", self.s2p_extras.dff_window_size
        )
    set_tooltip("Frames for rolling percentile baseline (dF/F only, default: 300).")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi_extras("dff_percentile", self.s2p_extras.dff_percentile):
        _, self.s2p_extras.dff_percentile = imgui.input_int(
            "Percentile", self.s2p_extras.dff_percentile
        )
    set_tooltip("Percentile for baseline F0 estimation (dF/F only, default: 20).")
    imgui.end_disabled()
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi_extras("dff_smooth_window", self.s2p_extras.dff_smooth_window):
        _, self.s2p_extras.dff_smooth_window = imgui.input_int(
            "Smooth", self.s2p_extras.dff_smooth_window
        )
    set_tooltip("Smooth the normalized trace with a rolling window (0 = disabled).")

    imgui.spacing()
    imgui.spacing()

    _rm_available = _check_rastermap_available()
    if not _rm_available:
        self.s2p_extras.rastermap_mode = 0
        self.s2p_extras.rastermap_planar = False
        self.s2p_extras.rastermap_volumetric = False

    _rm_title_color = (
        _SUBSECTION_COLOR if _rm_available else imgui.ImVec4(0.95, 0.4, 0.4, 1.0)
    )
    imgui.text_colored(_rm_title_color, "Rastermap")
    if not _rm_available:
        set_tooltip(
            "Rastermap not installed. Install with `pip install rastermap`.",
            show_mark=False,
        )
    imgui.same_line()
    _rm_mode = self.s2p_extras.rastermap_mode
    imgui.begin_disabled(not _rm_available)
    if imgui.radio_button("Skip##rastermap_mode", _rm_mode == 0):
        self.s2p_extras.rastermap_mode = 0
        self.s2p_extras.rastermap_planar = False
        self.s2p_extras.rastermap_volumetric = False
    imgui.same_line()
    if imgui.radio_button("Run##rastermap_mode", _rm_mode == 1):
        self.s2p_extras.rastermap_mode = 1
    imgui.same_line()
    if imgui.radio_button("Force##rastermap_mode", _rm_mode == 2):
        self.s2p_extras.rastermap_mode = 2
    imgui.end_disabled()

    _rm_active = self.s2p_extras.rastermap_mode > 0

    set_tooltip(
        "Sorts cells so similar activity is adjacent in the figure "
        "(bands = ensembles, diagonals = sequences). "
        "Run reuses cached model.npy when params match; Force always recomputes."
    )

    imgui.begin_disabled(not _rm_active)

    with _hi_extras("rastermap_planar", self.s2p_extras.rastermap_planar):
        _, self.s2p_extras.rastermap_planar = imgui.checkbox(
            "Planar (per-plane)", self.s2p_extras.rastermap_planar
        )
    set_tooltip(
        "One rastermap per plane (12_rastermap.png each). "
        "Requires `rastermap` installed."
    )
    if self.s2p_extras.rastermap_planar:
        imgui.indent(16)
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi_extras(
            "rastermap_planar_n_clusters",
            self.s2p_extras.rastermap_planar_n_clusters,
        ):
            _, self.s2p_extras.rastermap_planar_n_clusters = imgui.input_int(
                "n_clusters##rm_planar",
                self.s2p_extras.rastermap_planar_n_clusters,
            )
        set_tooltip(
            "Superclusters; each row averages ~n_cells/n_clusters neurons. "
            "Default 0 = adaptive (None for <200 cells, 100 otherwise)."
        )
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi_extras(
            "rastermap_planar_n_pcs",
            self.s2p_extras.rastermap_planar_n_pcs,
        ):
            _, self.s2p_extras.rastermap_planar_n_pcs = imgui.input_int(
                "n_PCs##rm_planar",
                self.s2p_extras.rastermap_planar_n_pcs,
            )
        set_tooltip(
            "Principal components for similarity. Default 0 = min(128, "
            "n_cells - 1). Lower (16-32) for <50 cells; rarely needs tuning."
        )
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi_extras(
            "rastermap_planar_locality",
            self.s2p_extras.rastermap_planar_locality,
        ):
            _, self.s2p_extras.rastermap_planar_locality = imgui.input_float(
                "locality##rm_planar",
                self.s2p_extras.rastermap_planar_locality,
                0.0,
                0.0,
                "%.2f",
            )
        set_tooltip(
            "How time-local the sort is. 0 = global match anywhere, "
            "1 = strict same-time match. Range 0.0-1.0; "
            "default -1 = adaptive (0.0 for >=200 cells, 0.1 otherwise)."
        )
        imgui.unindent(16)

    with _hi_extras("rastermap_volumetric", self.s2p_extras.rastermap_volumetric):
        _, self.s2p_extras.rastermap_volumetric = imgui.checkbox(
            "Volumetric", self.s2p_extras.rastermap_volumetric
        )
    set_tooltip(
        "One rastermap pooling cells across all planes "
        "(rastermap_3d.png). Each row can now span multiple z-planes."
    )
    if self.s2p_extras.rastermap_volumetric:
        imgui.indent(16)
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi_extras(
            "rastermap_volumetric_n_clusters",
            self.s2p_extras.rastermap_volumetric_n_clusters,
        ):
            _, self.s2p_extras.rastermap_volumetric_n_clusters = imgui.input_int(
                "n_clusters##rm_vol",
                self.s2p_extras.rastermap_volumetric_n_clusters,
            )
        set_tooltip(
            "Superclusters across the whole volume. Default 40; "
            "drop to 10-20 for <200 cells, raise to 60-100 for >2000."
        )
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi_extras(
            "rastermap_volumetric_n_pcs",
            self.s2p_extras.rastermap_volumetric_n_pcs,
        ):
            _, self.s2p_extras.rastermap_volumetric_n_pcs = imgui.input_int(
                "n_PCs##rm_vol",
                self.s2p_extras.rastermap_volumetric_n_pcs,
            )
        set_tooltip(
            "Principal components for similarity. Default 0 = min(200, "
            "n_cells - 1). Only lower with <100 total cells."
        )
        imgui.unindent(16)

    imgui.end_disabled()
    imgui.end_child()


def _draw_main_settings(self):
    _box_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
    imgui.begin_child("##s2p_main_box", imgui.ImVec2(-1, 0), _box_flags)
    imgui.text_colored(_S2P_TITLE_COLOR, "Suite2p Main Settings")
    imgui.spacing()

    from mbo_utilities.preferences import set_s2p_torch_device

    cur = self.s2p.torch_device or "cuda"
    cur_class = "cuda" if cur.startswith("cuda") else cur
    if cur != cur_class:
        self.s2p.torch_device = cur_class
        set_s2p_torch_device(cur_class)
    class_options = ["cuda", "cpu", "mps"]
    class_idx = class_options.index(cur_class) if cur_class in class_options else 0
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("torch_device", self.s2p.torch_device):
        class_changed, new_class_idx = imgui.combo(
            "Torch Device", class_idx, class_options
        )
    if class_changed:
        self.s2p.torch_device = class_options[new_class_idx]
        set_s2p_torch_device(self.s2p.torch_device)
    set_tooltip(
        "Compute backend for suite2p.\n"
        "\n"
        "cuda:\n"
        "  - Registration: GPU\n"
        "  - Detection: GPU only if the algorithm is Cellpose;\n"
        "    other detection algorithms run on CPU\n"
        "  - Extraction / deconvolution: CPU\n"
        "\n"
        "cpu:\n"
        "  - All steps run on CPU\n"
        "\n"
        "Falls back to CPU if CUDA allocation fails.\n"
        "GPU device is set in File > Options > Compute GPU."
    )

    imgui.text_colored(
        imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
        _selected_compute_gpu(self.s2p.torch_device),
    )
    set_tooltip("Compute GPU is selected in File > Options.")

    imgui.spacing()
    imgui.spacing()

    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("tau", self.s2p.tau):
        _, self.s2p.tau = imgui.input_float("##tau", self.s2p.tau)
    imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
    _emp_label(self, "tau", self.s2p.tau, "Tau (s)")
    set_tooltip(
        "Calcium indicator decay timescale in seconds.\n"
        "GCaMP6f=0.7, GCaMP6m=1.0-1.3 (LBM default), GCaMP6s=1.25-1.5"
    )
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("fs", self.s2p.fs):
        _, self.s2p.fs = imgui.input_float("Sampling Rate (Hz)", self.s2p.fs)
    set_tooltip("Per-plane sampling rate in Hz; drives baseline window sizing.")

    imgui.end_child()


def _draw_registration_settings(self):
    with _hi("keep_movie_raw", self.s2p_db.keep_movie_raw):
        _, self.s2p_db.keep_movie_raw = imgui.checkbox(
            "Keep Raw Binary", self.s2p_db.keep_movie_raw
        )
    set_tooltip(
        "Keep data_raw.bin (unregistered) after processing. "
        "Suite2p db.keep_movie_raw — default False."
    )
    _keep_reg = not self.s2p.delete_bin
    with _hi("delete_bin", self.s2p.delete_bin):
        _kr_changed, _new_keep_reg = imgui.checkbox("Keep Registered Binary", _keep_reg)
    if _kr_changed:
        self.s2p.delete_bin = not _new_keep_reg
    set_tooltip(
        "Keep data.bin (registered) after processing. Inverse of "
        "suite2p io.delete_bin — Keep Registered = True maps to "
        "delete_bin=False (the upstream default)."
    )

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("align_by_chan", self.s2p.align_by_chan):
        _, self.s2p.align_by_chan = imgui.input_int(
            "Align by chan", self.s2p.align_by_chan
        )
    set_tooltip(
        "Channel index used for alignment (1-based). "
        "Serialized to settings['registration']['align_by_chan2'] = (align_by_chan == 2)."
    )
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("nimg_init", self.s2p.nimg_init):
        _, self.s2p.nimg_init = imgui.input_int("# refImg frames", self.s2p.nimg_init)
    set_tooltip("Number of subsampled frames used to build the reference image.")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("maxregshift", self.s2p.maxregshift):
        _, self.s2p.maxregshift = imgui.input_float(
            "Max reg shift", self.s2p.maxregshift
        )
    set_tooltip(
        "Max allowed registration shift, as a fraction of frame max(width, height)."
    )

    with _hi("do_bidiphase", self.s2p.do_bidiphase):
        _, self.s2p.do_bidiphase = imgui.checkbox(
            "Compute bidiphase", self.s2p.do_bidiphase
        )
    set_tooltip(
        "Auto-compute the bidirectional scan-phase offset and apply it "
        "to all frames (2P only)."
    )
    imgui.set_next_item_width(_INPUT_WIDTH)
    imgui.begin_disabled(self.s2p.do_bidiphase)
    with _hi("bidiphase", self.s2p.bidiphase):
        _, self.s2p.bidiphase = imgui.input_float(
            "Bidiphase offset", self.s2p.bidiphase, 0.0, 0.0, "%.2f"
        )
    imgui.end_disabled()
    set_tooltip(
        "User-provided bidirectional phase offset; applied to all frames "
        "when 'Compute bidiphase' is off."
    )

    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("reg_batch_size", self.s2p.reg_batch_size):
        _, self.s2p.reg_batch_size = imgui.input_int(
            "# frames/batch", self.s2p.reg_batch_size
        )
    set_tooltip("Frames per registration batch — reduce if running out of GPU memory.")

    if imgui.tree_node("Non-rigid Registration"):
        with _hi("nonrigid", self.s2p.nonrigid):
            _, self.s2p.nonrigid = imgui.checkbox("Use nonrigid", self.s2p.nonrigid)
        set_tooltip("Split FOV into blocks and compute registration offsets per block.")

        imgui.begin_disabled(not self.s2p.nonrigid)

        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("maxregshiftNR", self.s2p.maxregshiftNR):
            _, self.s2p.maxregshiftNR = imgui.input_int(
                "NR max shift", self.s2p.maxregshiftNR
            )
        set_tooltip("Max pixel shift of block relative to rigid shift.")

        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("block_size_y", self.s2p.block_size_y):
            _, self.s2p.block_size_y = imgui.input_int(
                "Block height", self.s2p.block_size_y
            )
        set_tooltip("Block height for non-rigid registration (keep multiple of 2/3/5).")
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("block_size_x", self.s2p.block_size_x):
            _, self.s2p.block_size_x = imgui.input_int(
                "Block width", self.s2p.block_size_x
            )
        set_tooltip("Block width for non-rigid registration (keep multiple of 2/3/5).")

        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("snr_thresh", self.s2p.snr_thresh):
            _, self.s2p.snr_thresh = imgui.input_float(
                "NR SNR thresh", self.s2p.snr_thresh
            )
        set_tooltip(
            "If a nonrigid block falls below this, it gets smoothed until above. "
            "1.0 = no smoothing; 1.5 recommended for 1P."
        )

        imgui.end_disabled()
        imgui.tree_pop()

    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("smooth_sigma_time", self.s2p.smooth_sigma_time):
        _, self.s2p.smooth_sigma_time = imgui.input_float(
            "Time smoothing", self.s2p.smooth_sigma_time
        )
    set_tooltip(
        "Gaussian smoothing in time (frames) before computing shifts — useful for low-SNR."
    )
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("smooth_sigma", self.s2p.smooth_sigma):
        _, self.s2p.smooth_sigma = imgui.input_float(
            "XY smoothing", self.s2p.smooth_sigma
        )
    set_tooltip(
        "Gaussian smoothing sigma (pixels) in XY before registration.\n"
        "  ~1.0 — 2-photon recordings (default 1.15)\n"
        "  3-5  — 1-photon recordings"
    )
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("spatial_taper", self.s2p.spatial_taper):
        _, self.s2p.spatial_taper = imgui.input_float(
            "Edge taper", self.s2p.spatial_taper
        )
    set_tooltip(
        "Edge tapering width in pixels (zero out edges before FFT) — "
        "important for vignetted windows. Bump higher for 1-photon "
        "recordings. Upstream default: 3.45."
    )
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("th_badframes", self.s2p.th_badframes):
        _, self.s2p.th_badframes = imgui.input_float(
            "Bad frame thresh", self.s2p.th_badframes
        )
    set_tooltip(
        "Threshold for excluding low-quality frames when cropping — smaller excludes more."
    )
    with _hi("norm_frames", self.s2p.norm_frames):
        _, self.s2p.norm_frames = imgui.checkbox(
            "Normalize frames", self.s2p.norm_frames
        )
    set_tooltip("Normalize frames when detecting shifts.")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("subpixel", self.s2p.subpixel):
        _, self.s2p.subpixel = imgui.input_int("Subpixel reg", self.s2p.subpixel)
    set_tooltip("Subpixel precision level (1/subpixel step).")
    with _hi("two_step_registration", self.s2p.two_step_registration):
        _, self.s2p.two_step_registration = imgui.checkbox(
            "##two_step_registration", self.s2p.two_step_registration
        )
    imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
    _emp_label(
        self,
        "two_step_registration",
        self.s2p.two_step_registration,
        "Run reg twice",
    )
    set_tooltip(
        "Perform registration twice for low-SNR data (set keep_movie_raw=True)."
    )
    with _hi("do_regmetrics", self.s2p.do_regmetrics):
        _, self.s2p.do_regmetrics = imgui.checkbox(
            "Compute reg metrics", self.s2p.do_regmetrics
        )
    set_tooltip("Compute registration QC metrics (requires ≥1500 frames).")


def _draw_main_above_registration(self):
    _draw_main_settings(self)

    imgui.spacing()
    imgui.spacing()

    _box_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
    imgui.begin_child("##s2p_reg_box", imgui.ImVec2(-1, 0), _box_flags)
    imgui.text_colored(_S2P_TITLE_COLOR, "Registration")
    _cur_reg = self.s2p.do_registration
    imgui.same_line()
    if imgui.radio_button("Skip##do_registration_inner", _cur_reg == 0):
        self.s2p.do_registration = 0
    imgui.same_line()
    if imgui.radio_button("Run##do_registration_inner", _cur_reg == 1):
        self.s2p.do_registration = 1
    imgui.same_line()
    if imgui.radio_button("Force##do_registration_inner", _cur_reg == 2):
        self.s2p.do_registration = 2
    imgui.separator()
    imgui.spacing()
    _reg_skip = self.s2p.do_registration == 0
    imgui.begin_disabled(_reg_skip)
    _draw_registration_settings(self)
    imgui.end_disabled()
    imgui.end_child()


def _draw_roi_detection_settings(self):
    imgui.text_colored(_SUBSECTION_COLOR, "Shared")
    imgui.spacing()

    with _hi("denoise", self.s2p.denoise):
        _, self.s2p.denoise = imgui.checkbox("Denoise", self.s2p.denoise)
    set_tooltip(
        "Run PCA denoising on the binned movie before cell detection. "
        "Suite2p detection.denoise — default False. The block size below "
        "is only used when this is on."
    )
    imgui.begin_disabled(not self.s2p.denoise)
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("det_block_size_y", self.s2p.det_block_size_y):
        _, self.s2p.det_block_size_y = imgui.input_int(
            "Denoise block Y", self.s2p.det_block_size_y
        )
    set_tooltip(
        "Block height for PCA denoising (settings['detection']['block_size'][0])."
    )
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("det_block_size_x", self.s2p.det_block_size_x):
        _, self.s2p.det_block_size_x = imgui.input_int(
            "Denoise block X", self.s2p.det_block_size_x
        )
    set_tooltip(
        "Block width for PCA denoising (settings['detection']['block_size'][1])."
    )
    imgui.end_disabled()

    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("nbins", self.s2p.nbins):
        _, self.s2p.nbins = imgui.input_int("Max binned frames", self.s2p.nbins)
    set_tooltip("Max binned frames for cell detection (reduce if RAM-limited).")

    bin_auto = self.s2p.bin_size is None
    auto_changed, new_bin_auto = imgui.checkbox("Auto bin size (tau*fs)", bin_auto)
    set_tooltip(
        "When checked, suite2p picks the cell-detection bin size as "
        "tau*fs at runtime. Uncheck to set an explicit value."
    )
    if auto_changed:
        if new_bin_auto:
            self.s2p.bin_size = None
        else:
            self.s2p.bin_size = max(1, int(round(self.s2p.tau * self.s2p.fs)))
    imgui.begin_disabled(self.s2p.bin_size is None)
    imgui.set_next_item_width(_INPUT_WIDTH)
    _bin_display = self.s2p.bin_size if self.s2p.bin_size is not None else 0
    with _hi("bin_size", self.s2p.bin_size):
        _bin_changed, _bin_new = imgui.input_int("Bin size", _bin_display)
    if _bin_changed and self.s2p.bin_size is not None:
        self.s2p.bin_size = max(1, _bin_new)
    set_tooltip("settings['detection']['bin_size'] in frames.")
    imgui.end_disabled()

    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("highpass_time", self.s2p.highpass_time):
        _, self.s2p.highpass_time = imgui.input_int(
            "##highpass_time", self.s2p.highpass_time
        )
    imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
    _emp_label(self, "highpass_time", self.s2p.highpass_time, "Highpass time")
    set_tooltip("Running mean subtraction window across bins for temporal high-pass.")

    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("npix_norm_min", self.s2p.npix_norm_min):
        _, self.s2p.npix_norm_min = imgui.input_float(
            "Min npix norm", self.s2p.npix_norm_min, 0.0, 0.0, "%.2f"
        )
    set_tooltip(
        "Minimum npix_norm filter for ROIs (per-ROI npix normalized "
        "by the highest-variance ROIs' mean npix). Default 0.0."
    )
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("npix_norm_max", self.s2p.npix_norm_max):
        _, self.s2p.npix_norm_max = imgui.input_float(
            "Max npix norm", self.s2p.npix_norm_max, 0.0, 0.0, "%.2f"
        )
    set_tooltip("Maximum npix_norm filter for ROIs. Default 100.0.")

    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("max_overlap", self.s2p.max_overlap):
        _, self.s2p.max_overlap = imgui.input_float("Max overlap", self.s2p.max_overlap)
    set_tooltip("ROIs with more overlap than this fraction are discarded.")

    with _hi("soma_crop", self.s2p.soma_crop):
        _, self.s2p.soma_crop = imgui.checkbox("Soma crop", self.s2p.soma_crop)
    set_tooltip("Crop dendrites from ROI to determine npix_norm and compactness.")

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    algo_options = ["cellpose", "sparsery", "sourcery"]
    if self.s2p.algorithm not in algo_options:
        self.s2p.algorithm = "cellpose"
    cur_algo_idx = algo_options.index(self.s2p.algorithm)
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("algorithm", self.s2p.algorithm):
        algo_changed, new_algo_idx = imgui.combo(
            "##algorithm", cur_algo_idx, algo_options
        )
    imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
    _emp_label(self, "algorithm", self.s2p.algorithm, "Algorithm")
    if algo_changed:
        self.s2p.algorithm = algo_options[new_algo_idx]
    set_tooltip(
        "Cell-detection algorithm.\n"
        "  cellpose: anatomical detection via deep learning (LBM default)\n"
        "  sparsery: functional detection, sparse activity (upstream default)\n"
        "  sourcery: functional detection, legacy"
    )

    if self.s2p.algorithm == "cellpose" and not _has_gpu_torch():
        if not self._cellpose_cpu_warned:
            imgui.open_popup("No GPU for cellpose")
            self._cellpose_cpu_warned = True
    else:
        self._cellpose_cpu_warned = False

    imgui.set_next_window_pos(
        imgui.get_main_viewport().get_center(),
        imgui.Cond_.appearing,
        pivot=imgui.ImVec2(0.5, 0.5),
    )
    imgui.set_next_window_size(
        imgui.ImVec2(hello_imgui.em_size(26), 0.0), imgui.Cond_.appearing
    )
    _cp_opened, _ = imgui.begin_popup_modal(
        "No GPU for cellpose", flags=imgui.WindowFlags_.no_saved_settings
    )
    if _cp_opened:
        imgui.text_colored(_LOAD_BAD_COLOR, "No GPU PyTorch detected.")
        imgui.text("Cellpose will run very slowly on the CPU.")
        imgui.spacing()
        _torch_cmd = (
            "uv pip uninstall torch torchvision\n\n"
            "uv pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu126"
        )
        imgui.text_wrapped("You need a version of PyTorch with GPU capabilities. See:")
        _torch_url = "https://pytorch.org/get-started/locally/"
        if imgui.text_link(_torch_url):
            import webbrowser

            webbrowser.open(_torch_url)
        imgui.text_wrapped(
            "for the command that matches your operating system and "
            "CUDA version. For example:"
        )
        imgui.spacing()
        imgui.push_style_color(imgui.Col_.text, _SUBSECTION_COLOR)
        imgui.text_wrapped(_torch_cmd)
        imgui.pop_style_color()
        with _ghost_button():
            if imgui.small_button(f"{fa.ICON_FA_COPY}##torch_gpu_copy"):
                imgui.set_clipboard_text(_torch_cmd)
        set_tooltip("Copy commands to clipboard", show_mark=False)
        imgui.spacing()
        if imgui.button("OK", imgui.ImVec2(120, 0)):
            imgui.close_current_popup()
        imgui.end_popup()

    imgui.spacing()

    if self.s2p.algorithm == "cellpose":
        imgui.text_colored(_SUBSECTION_COLOR, "Cellpose")
        imgui.spacing()

        cellpose_model_buf = self.s2p.cellpose_model or ""
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("cellpose_model", self.s2p.cellpose_model):
            _changed, _new_model = imgui.input_text(
                "Cellpose model", cellpose_model_buf
            )
        if _changed:
            self.s2p.cellpose_model = _new_model
        set_tooltip(
            "Cellpose model name (e.g. 'cpsam', 'cyto3'). Forwarded to "
            "settings['detection']['cellpose_settings']['cellpose_model']."
        )

        img_options = ["max_proj / meanImg", "meanImg", "max_proj"]
        if self.s2p.cellpose_img not in img_options:
            self.s2p.cellpose_img = img_options[0]
        cur_img_idx = img_options.index(self.s2p.cellpose_img)
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("cellpose_img", self.s2p.cellpose_img):
            img_changed, new_img_idx = imgui.combo(
                "##cellpose_img", cur_img_idx, img_options
            )
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        _emp_label(self, "cellpose_img", self.s2p.cellpose_img, "Cellpose image")
        if img_changed:
            self.s2p.cellpose_img = img_options[new_img_idx]
        set_tooltip(
            "Image cellpose runs detection on. Upstream allowed values:\n"
            "  max_proj / meanImg  (default; combined)\n"
            "  meanImg             (mean image only)\n"
            "  max_proj            (max projection only)"
        )

        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("diameter_y", self.s2p.diameter_y):
            dy_changed, dy = imgui.input_float("Diameter Y", self.s2p.diameter_y)
        set_tooltip(
            "Expected cell diameter (Y axis) in pixels. Passed to Cellpose.\n"
            "Upstream default: 12."
        )
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("diameter_x", self.s2p.diameter_x):
            dx_changed, dx = imgui.input_float("Diameter X", self.s2p.diameter_x)
        set_tooltip(
            "Expected cell diameter (X axis) in pixels. Passed to Cellpose.\n"
            "Upstream default: 12."
        )
        _, self._s2p_diameter_lock = imgui.checkbox("Lock Y/X", self._s2p_diameter_lock)
        set_tooltip("Keep diameter_y and diameter_x in sync when editing either.")
        if dy_changed:
            self.s2p.diameter_y = dy
            if self._s2p_diameter_lock:
                self.s2p.diameter_x = dy
        if dx_changed:
            self.s2p.diameter_x = dx
            if self._s2p_diameter_lock:
                self.s2p.diameter_y = dx

        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("highpass_spatial", self.s2p.highpass_spatial):
            _, self.s2p.highpass_spatial = imgui.input_float(
                "Highpass spatial", self.s2p.highpass_spatial
            )
        set_tooltip(
            "Spatial high-pass filtering before Cellpose, as a multiple "
            "of diameter. 0.5 = LBM default."
        )
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("flow_threshold", self.s2p.flow_threshold):
            _, self.s2p.flow_threshold = imgui.input_float(
                "##flow_threshold", self.s2p.flow_threshold
            )
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        _emp_label(self, "flow_threshold", self.s2p.flow_threshold, "Flow threshold")
        set_tooltip(
            "Max allowed flow error per mask. LBM default: 0 (flow checking disabled).\n"
            "INCREASE (suite2p default 0.4) if missing masks.\n"
            "DECREASE if cellpose returns ill-shaped masks."
        )
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("cellprob_threshold", self.s2p.cellprob_threshold):
            _, self.s2p.cellprob_threshold = imgui.input_float(
                "##cellprob_threshold", self.s2p.cellprob_threshold
            )
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        _emp_label(
            self,
            "cellprob_threshold",
            self.s2p.cellprob_threshold,
            "Cellprob threshold",
        )
        set_tooltip(
            "Cell probability threshold for Cellpose. LBM default: -4.\n\n"
            "DECREASE if cellpose is missing masks (or masks too small).\n"
            "INCREASE (suite2p default 0.0) if too many masks / false positives."
        )
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _mbo():
            _, self.s2p.cellpose_niter = imgui.input_int(
                "Cellpose niter", self.s2p.cellpose_niter
            )
        set_tooltip(
            "Cellpose iteration count (model.eval(niter=...)).\n"
            "0 = let cellpose pick based on diameter (default).\n"
            "Bump (e.g. 200) for dense or small cells.\n"
            "MBO-only — composes into cellpose_settings.params at run time."
        )

    elif self.s2p.algorithm == "sparsery":
        imgui.text_colored(_SUBSECTION_COLOR, "Sparsery")
        imgui.spacing()

        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("functional_chan", self.s2p_db.functional_chan):
            _, self.s2p_db.functional_chan = imgui.input_int(
                "Functional chan", self.s2p_db.functional_chan
            )
        set_tooltip("Channel used for functional ROI extraction (1-based).")
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("threshold_scaling", self.s2p.threshold_scaling):
            _, self.s2p.threshold_scaling = imgui.input_float(
                "##threshold_scaling", self.s2p.threshold_scaling
            )
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        _emp_label(
            self,
            "threshold_scaling",
            self.s2p.threshold_scaling,
            "Threshold scaling",
        )
        set_tooltip("Scale ROI detection threshold; higher = fewer ROIs.")
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("highpass_neuropil", self.s2p.highpass_neuropil):
            _, self.s2p.highpass_neuropil = imgui.input_int(
                "Highpass neuropil", self.s2p.highpass_neuropil
            )
        set_tooltip(
            "Highpass filter in pixels on the binned movie to subtract neuropil."
        )
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("max_ROIs", self.s2p.max_ROIs):
            _, self.s2p.max_ROIs = imgui.input_int("Max ROIs", self.s2p.max_ROIs)
        set_tooltip("Hard cap on detected ROIs (sparsery only).")
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("spatial_scale", self.s2p.spatial_scale):
            _, self.s2p.spatial_scale = imgui.input_int(
                "Spatial scale", self.s2p.spatial_scale
            )
        set_tooltip(
            "ROI size scale: 0=auto, 1=6-pixel cells (LBM default), "
            "2=medium, 3=large, 4=very large."
        )
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("active_percentile", self.s2p.active_percentile):
            _, self.s2p.active_percentile = imgui.input_float(
                "Active percentile",
                self.s2p.active_percentile,
                0.0,
                0.0,
                "%.2f",
            )
        set_tooltip("Percentile of active pixels in the movie used for thresholding.")

    elif self.s2p.algorithm == "sourcery":
        imgui.text_colored(_SUBSECTION_COLOR, "Sourcery")
        imgui.spacing()

        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("functional_chan", self.s2p_db.functional_chan):
            _, self.s2p_db.functional_chan = imgui.input_int(
                "Functional chan", self.s2p_db.functional_chan
            )
        set_tooltip("Channel used for functional ROI extraction (1-based).")

        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("diameter_y", self.s2p.diameter_y):
            dy_changed_s, dy_s = imgui.input_float("Diameter Y", self.s2p.diameter_y)
        set_tooltip(
            "Expected cell diameter (Y axis) in pixels. Used by sourcery "
            "and cellpose. Upstream default: 12."
        )
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("diameter_x", self.s2p.diameter_x):
            dx_changed_s, dx_s = imgui.input_float("Diameter X", self.s2p.diameter_x)
        set_tooltip(
            "Expected cell diameter (X axis) in pixels. Used by sourcery "
            "and cellpose. Upstream default: 12."
        )
        _, self._s2p_diameter_lock = imgui.checkbox(
            "Lock Y/X##sourcery", self._s2p_diameter_lock
        )
        set_tooltip("Keep diameter_y and diameter_x in sync when editing either.")
        if dy_changed_s:
            self.s2p.diameter_y = dy_s
            if self._s2p_diameter_lock:
                self.s2p.diameter_x = dy_s
        if dx_changed_s:
            self.s2p.diameter_x = dx_s
            if self._s2p_diameter_lock:
                self.s2p.diameter_y = dx_s

        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("threshold_scaling", self.s2p.threshold_scaling):
            _, self.s2p.threshold_scaling = imgui.input_float(
                "##threshold_scaling", self.s2p.threshold_scaling
            )
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        _emp_label(
            self,
            "threshold_scaling",
            self.s2p.threshold_scaling,
            "Threshold scaling",
        )
        set_tooltip("Scale ROI detection threshold; higher = fewer ROIs.")
        with _hi("connected", self.s2p.connected):
            _, self.s2p.connected = imgui.checkbox("Connected", self.s2p.connected)
        set_tooltip("Keep ROIs fully connected (set False for dendrites).")
        imgui.set_next_item_width(_INPUT_WIDTH)
        with _hi("max_iterations", self.s2p.max_iterations):
            _, self.s2p.max_iterations = imgui.input_int(
                "Max iterations", self.s2p.max_iterations
            )
        set_tooltip("Maximum number of ROI-detection iterations.")
        with _hi("smooth_masks", self.s2p.smooth_masks):
            _, self.s2p.smooth_masks = imgui.checkbox(
                "Smooth masks", self.s2p.smooth_masks
            )
        set_tooltip("Smooth masks in the final ROI detection pass (sourcery).")


def _draw_signal_extraction_settings(self):
    with _hi("neuropil_extract", self.s2p.neuropil_extract):
        _, self.s2p.neuropil_extract = imgui.checkbox(
            "Extract Neuropil", self.s2p.neuropil_extract
        )
    set_tooltip(
        "Compute Fneu from neuropil masks. Default: True.\n\n"
        "WARNING: setting this to False crashes upstream suite2p — "
        "create_masks returns neuropil_masks=None but extract_traces "
        "doesn't handle None and raises 'NoneType is not iterable'. "
        "Bug is in suite2p/extraction/extract.py:60. Keep this on "
        "until upstream fixes it."
    )
    with _hi("allow_overlap", self.s2p.allow_overlap):
        _, self.s2p.allow_overlap = imgui.checkbox(
            "Allow Overlap", self.s2p.allow_overlap
        )
    set_tooltip("Allow overlapping ROI pixels during extraction.")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("min_neuropil_pixels", self.s2p.min_neuropil_pixels):
        _, self.s2p.min_neuropil_pixels = imgui.input_int(
            "Min Neuropil Pixels", self.s2p.min_neuropil_pixels
        )
    set_tooltip("Minimum neuropil pixels per ROI.")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("inner_neuropil_radius", self.s2p.inner_neuropil_radius):
        _, self.s2p.inner_neuropil_radius = imgui.input_int(
            "Inner Neuropil Radius", self.s2p.inner_neuropil_radius
        )
    set_tooltip("Pixels to exclude between ROI and neuropil region.")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("lam_percentile", self.s2p.lam_percentile):
        _, self.s2p.lam_percentile = imgui.input_float(
            "Lambda Percentile", self.s2p.lam_percentile, 0.0, 0.0, "%.2f"
        )
    set_tooltip("Percentile of Lambda used for neuropil exclusion.")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("snr_threshold", self.s2p.snr_threshold):
        _, self.s2p.snr_threshold = imgui.input_float(
            "SNR Threshold", self.s2p.snr_threshold, 0.0, 0.0, "%.2f"
        )
    set_tooltip("settings['extraction']['snr_threshold']. Default 0.0.")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("extract_batch_size", self.s2p.extract_batch_size):
        _, self.s2p.extract_batch_size = imgui.input_int(
            "Extract Batch Size", self.s2p.extract_batch_size
        )
    set_tooltip(
        "Frames per extraction batch (settings['extraction']['batch_size']). "
        "Default 500."
    )


def _draw_deconvolution_settings(self):
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("neuropil_coefficient", self.s2p.neuropil_coefficient):
        _, self.s2p.neuropil_coefficient = imgui.input_float(
            "Neuropil coefficient", self.s2p.neuropil_coefficient
        )
    set_tooltip("Neuropil coefficient for all ROIs (F_corrected = F - coeff * F_neu).")

    baseline_options = ["maximin", "constant", "prctile"]
    _legacy_map = {
        "constant_percentile": "prctile",
        "constant_prctile": "prctile",
    }
    if self.s2p.baseline in _legacy_map:
        self.s2p.baseline = _legacy_map[self.s2p.baseline]
    current_baseline_idx = (
        baseline_options.index(self.s2p.baseline)
        if self.s2p.baseline in baseline_options
        else 0
    )
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("baseline", self.s2p.baseline):
        baseline_changed, selected_baseline_idx = imgui.combo(
            "Baseline method", current_baseline_idx, baseline_options
        )
    if baseline_changed:
        self.s2p.baseline = baseline_options[selected_baseline_idx]
    set_tooltip(
        "maximin: moving baseline with min/max filters.\n"
        "constant: minimum of Gaussian-filtered trace.\n"
        "prctile: percentile of trace (controlled by 'Baseline percentile')."
    )

    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("win_baseline", self.s2p.win_baseline):
        _, self.s2p.win_baseline = imgui.input_float(
            "Baseline window (s)", self.s2p.win_baseline
        )
    set_tooltip("Window for maximin filter in seconds.")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("sig_baseline", self.s2p.sig_baseline):
        _, self.s2p.sig_baseline = imgui.input_float(
            "Baseline sigma (s)", self.s2p.sig_baseline
        )
    set_tooltip("Gaussian filter width in seconds for baseline computation.")
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("prctile_baseline", self.s2p.prctile_baseline):
        _, self.s2p.prctile_baseline = imgui.input_float(
            "Baseline percentile", self.s2p.prctile_baseline
        )
    set_tooltip("Percentile of trace for 'prctile' baseline method.")


def _draw_classification_settings(self):
    imgui.set_next_item_width(_INPUT_WIDTH)
    with _hi("preclassify", self.s2p.preclassify):
        _, self.s2p.preclassify = imgui.input_float(
            "Preclassify threshold", self.s2p.preclassify
        )
    set_tooltip(
        "Probability threshold to apply classifier before extraction. "
        "0 = skip pre-classification."
    )

    with _hi("use_builtin_classifier", self.s2p.use_builtin_classifier):
        _, self.s2p.use_builtin_classifier = imgui.checkbox(
            "Use built-in classifier", self.s2p.use_builtin_classifier
        )
    set_tooltip(
        "Forces the shipped suite2p classifier. Only takes effect when "
        "Classifier Path below is empty AND you have a saved "
        "~/.suite2p/classifiers/classifier_user.npy — otherwise suite2p "
        "would pick that user file; this forces the shipped one instead. "
        "Classification runs either way."
    )

    imgui.spacing()
    path_display = self.s2p.classifier_path if self.s2p.classifier_path else "(none)"
    with _hi("classifier_path", self.s2p.classifier_path):
        imgui.text("Classifier path:")
        imgui.push_text_wrap_pos(imgui.get_content_region_avail().x - 80)
        imgui.text(path_display)
        imgui.pop_text_wrap_pos()
    imgui.same_line()
    if imgui.button("Browse##classifier"):
        default_dir = (
            str(Path(self.s2p.classifier_path).parent)
            if self.s2p.classifier_path
            else str(Path.home())
        )
        self._classifier_dialog = pfd.open_file(
            "Select classifier file",
            default_dir,
            ["Classifier files", "*.npy *.pkl *.pickle", "All files", "*"],
        )
    set_tooltip("Select custom classifier file (e.g. .npy or .pkl)")

    if self._classifier_dialog is not None:
        if self._classifier_dialog.ready():
            result = self._classifier_dialog.result()
            if result:
                self.s2p.classifier_path = (
                    result[0] if isinstance(result, list) else result
                )
            self._classifier_dialog = None


def _column_natural_width(title: str, mode_kind) -> float:
    """Width of a column that fits its widest widget, label and (?) marker without clipping."""
    style = imgui.get_style()
    spacing_x = style.item_spacing.x
    frame_pad_x = style.frame_padding.x
    qm_w = imgui.calc_text_size("(?)").x
    cb_w = imgui.get_frame_height()
    inp_label_w = imgui.calc_text_size(_WORST_INPUT_LABEL[title]).x
    cb_label_w = imgui.calc_text_size(_WORST_CHECKBOX_LABEL[title]).x
    body_inp = _INPUT_WIDTH + spacing_x + inp_label_w + spacing_x + qm_w
    body_cb = cb_w + spacing_x + cb_label_w + spacing_x + qm_w
    body = max(body_inp, body_cb)
    title_w = imgui.calc_text_size(title).x
    radios = _TITLE_RADIO_LABELS[mode_kind]
    if radios:
        radio_labels = radios.split()
        n_radios = len(radio_labels)
        inner_sp = imgui.get_style().item_inner_spacing.x
        radio_widgets_w = sum(
            cb_w + inner_sp + imgui.calc_text_size(lbl).x for lbl in radio_labels
        )
        title_row_w = (
            title_w + spacing_x + radio_widgets_w + spacing_x * max(0, n_radios - 1)
        )
    else:
        title_row_w = title_w
    inner_pad = 2 * frame_pad_x
    _tooltip_pad = 34
    natural = max(body, title_row_w) + inner_pad + 8 + _tooltip_pad
    if title in _SELF_TITLED_COLUMNS:
        natural += inner_pad + 8
    return natural


def draw_section_suite2p(self):
    """Draw Suite2p configuration UI with button-based popups."""
    imgui.spacing()

    with imgui_ctx.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4)):
        with imgui_ctx.push_style_var(
            imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 2)
        ):
            _draw_section_suite2p_content(self)


def _draw_section_suite2p_content(self):
    """Inner content for suite2p section (called within style context)."""
    if self.s2p is None:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.7, 0.2, 1.0), "Suite2p is not installed."
        )
        imgui.text("Install with:")
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
            "uv pip install --no-deps lbm-suite2p-python && uv pip install suite2p rastermap torch torchvision",
        )
        return

    max_frames, num_planes, num_channels = _init_s2p_selection_state(self)

    s2p_path = self._s2p_outdir or self._saveas_outdir
    has_save_path = bool(s2p_path)

    _RUN_W = 220
    _BTN_W = 90

    _draw_current_dataset_section(self)
    from mbo_utilities.gui._gpu_status import draw_gpu_status

    draw_gpu_status(self.s2p.torch_device or "auto")

    imgui.spacing()
    imgui.separator()
    imgui.spacing()
    imgui.text_colored(_SUBSECTION_COLOR, "Select output data folder")
    set_tooltip(
        "Where Suite2p output is written. Each run creates a `suite2p` "
        "subfolder here with one `plane*` directory per z-plane. Type or "
        "paste a path, or click the folder icon to browse.",
        align="right",
    )
    imgui.spacing()
    imgui.spacing()
    if imgui.button(f"{fa.ICON_FA_FOLDER_OPEN}##s2p_browse"):
        default_dir = s2p_path or str(
            get_last_dir("suite2p_output") or pathlib.Path().home()
        )
        self._s2p_folder_dialog = pfd.select_folder(
            "Select Suite2p output folder", default_dir
        )
    imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
    imgui.set_next_item_width(-1)
    _path_changed, _new_path = imgui.input_text("##s2p_outdir", self._s2p_outdir or "")
    if _path_changed:
        self._s2p_outdir = _new_path
        s2p_path = _new_path
        has_save_path = bool(_new_path)

    if self._s2p_folder_dialog is not None and self._s2p_folder_dialog.ready():
        result = self._s2p_folder_dialog.result()
        if result:
            self._s2p_outdir = str(result)
            set_last_dir("suite2p_output", result)
        self._s2p_folder_dialog = None

    imgui.spacing()
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.text_colored(_SUBSECTION_COLOR, "Data slicing")
    set_tooltip(
        "Restrict which timepoints, z-planes, and channels feed the run. "
        "Click Set slice to define ranges using start:stop[:step] syntax — "
        "e.g. 1:1000:2 selects the first 1000 frames in steps of 2. The "
        "summary line below the button shows the resulting counts.",
        align="right",
    )
    imgui.spacing()
    imgui.spacing()
    if imgui.button("Set slice", imgui.ImVec2(_BTN_W, 0)):
        self._s2p_slicing_open = True
    _draw_s2p_selection_preview(self, max_frames, num_planes, num_channels)

    _draw_s2p_slicing_popup(self)

    imgui.spacing()
    imgui.spacing()

    imgui.spacing()
    imgui.separator()
    imgui.spacing()
    imgui.text_colored(_SUBSECTION_COLOR, "Data-specific options")
    set_tooltip(
        "Toggles for raw scan-phase correction (bidirectional alignment, "
        "FFT sub-pixel) and axial registration. Available only for raw "
        "MBO scans with multiple z-planes.",
        align="right",
    )
    imgui.spacing()
    imgui.spacing()
    if imgui.button("Open##data_options_btn", imgui.ImVec2(_BTN_W, 0)):
        _popup_states["data_options"] = True
        _do_sizer = self._data_options_sizer
        if _do_sizer is None:
            _do_sizer = PopupAutoSize("Data Options##data_options", auto_resize=False)
            self._data_options_sizer = _do_sizer
        _do_sizer.before_open()
        imgui.open_popup("Data Options##data_options")

    imgui.spacing()
    _has_phase_support = self.has_raster_scan_support
    _nz = self.nz
    _is_raw = self.is_mbo_scan
    _has_z_reg = _nz > 1 and _is_raw
    _frame_avg = self._s2p_frame_average
    if _has_phase_support or _has_z_reg or _frame_avg > 1:
        imgui.indent(12)
        if _frame_avg > 1:
            imgui.text(f"Frame Average: {_frame_avg}")
        if _has_phase_support:
            imgui.text(f"Fix Phase: {self._s2p_fix_phase}")
            imgui.text(f"Sub-Pixel (FFT): {self._s2p_use_fft}")
            imgui.text(f"Border (px): {self.border}")
            imgui.text(f"Max Offset: {self.max_offset}")
        if _has_z_reg:
            _reg_z = self._register_z
            imgui.text(f"Register Z-Planes: {_reg_z}")
            if _reg_z:
                imgui.text(f"Max frames: {self._axial_max_frames}")
                imgui.text(f"Max shift (px): {self._axial_max_reg_xy}")
        imgui.unindent(12)
    else:
        imgui.text_disabled("No options available for this data")

    imgui.spacing()
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.text_colored(_SUBSECTION_COLOR, "Parameters and settings")
    set_tooltip(
        "All Suite2p and LBM-Suite2p-Python knobs in one popup: Main, "
        "Registration, ROI Detection, Signal Extraction, Deconvolution, "
        "and Classification. Modified parameters appear below in this "
        "panel for a quick at-a-glance summary.",
        align="right",
    )
    imgui.spacing()
    imgui.spacing()
    _force_settings = self._force_pipe_settings
    if _force_settings:
        self._force_pipe_settings = False
    if (
        imgui.button("Open##pipe_settings_btn", imgui.ImVec2(_BTN_W, 0))
        or _force_settings
    ):
        _popup_states["pipeline_settings"] = True
        self._pipe_settings_just_opened = True
        # NOTE: before_open() runs inside the popup body: begin_child consumes the buffered window pos
        imgui.open_popup("Pipeline Settings##pipeline_settings_popup")

    imgui.spacing()

    _mods = collect_modified_params(self.s2p, self.s2p_db, self.s2p_extras)
    _n_mods = len(_mods)
    imgui.text(f"Modified parameters ({_n_mods})")
    if _mods:
        imgui.same_line()
        with _ghost_button():
            if imgui.small_button(f"{fa.ICON_FA_COPY}##mod_params_copy_all"):
                _lines = ["{"]
                for _field, _cur, _def, _src in _mods:
                    _lines.append(
                        f"    {_field!r}: {_cur!r},  # {_src}, default: {_def!r}"
                    )
                _lines.append("}")
                imgui.set_clipboard_text("\n".join(_lines))
        set_tooltip(
            "Copy all modified params as a Python dict. Each line tags its "
            "source pipeline (s2p = suite2p settings, lsp = lbm_suite2p_python "
            "kwarg) and the default value, as inline comments.",
            align="right",
        )

        _mod_h = min(180, imgui.get_frame_height_with_spacing() * (_n_mods + 2))
        if imgui.begin_child(
            "##mod_params",
            imgui.ImVec2(-1, _mod_h),
            imgui.ChildFlags_.borders,
        ):
            _table_flags = (
                imgui.TableFlags_.row_bg
                | imgui.TableFlags_.borders_inner_h
                | imgui.TableFlags_.sizing_stretch_prop
            )
            if imgui.begin_table("##mod_params_tbl", 3, _table_flags):
                imgui.table_setup_column(
                    "Parameter", imgui.TableColumnFlags_.width_stretch, 4.0
                )
                imgui.table_setup_column(
                    "Current", imgui.TableColumnFlags_.width_stretch, 2.5
                )
                imgui.table_setup_column(
                    "Default", imgui.TableColumnFlags_.width_stretch, 2.5
                )
                imgui.table_headers_row()
                for _field, _cur, _def, _src in _mods:
                    _cur_s = f"{_cur:.3g}" if isinstance(_cur, float) else str(_cur)
                    _def_s = f"{_def:.3g}" if isinstance(_def, float) else str(_def)
                    _name_color = _S2P_TITLE_COLOR if _src == "s2p" else _MBO_ONLY_COLOR
                    imgui.table_next_row()
                    imgui.table_set_column_index(0)
                    text_wrapped_cell(_field, _name_color)
                    imgui.table_set_column_index(1)
                    text_wrapped_cell(_cur_s)
                    imgui.table_set_column_index(2)
                    text_wrapped_cell(
                        _def_s,
                        imgui.get_style_color_vec4(imgui.Col_.text_disabled),
                    )
                imgui.end_table()
        imgui.end_child()
    else:
        imgui.text_disabled("Run pipeline to view parameters")

    imgui.spacing()
    imgui.spacing()

    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
    imgui.push_style_color(
        imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0)
    )
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.1, 0.45, 0.1, 1.0))

    if not has_save_path:
        imgui.begin_disabled()

    _run_avail = imgui.get_content_region_avail().x
    if _run_avail > _RUN_W:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (_run_avail - _RUN_W) * 0.5)
    button_clicked = imgui.button("Run Suite2p", imgui.ImVec2(_RUN_W, 0))

    if not has_save_path:
        imgui.end_disabled()

    imgui.pop_style_color(3)

    if not has_save_path and imgui.is_item_hovered(
        imgui.HoveredFlags_.allow_when_disabled
    ):
        imgui.set_tooltip("Set the output folder above (folder button or paste a path)")

    if button_clicked and has_save_path:
        run_process(self)

    if self._install_error:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.0, 0.0, 1.0),
            "lbm_suite2p_python is not installed: uv pip install lbm_suite2p_python",
        )

    rows = [
        [
            ("LBM-Suite2p-Python", _draw_lsp_settings, None, None),
            ("Registration", _draw_main_above_registration, "tri", "do_registration"),
            ("ROI Detection", _draw_roi_detection_settings, "tri", "do_detection"),
        ],
        [
            ("Classification", _draw_classification_settings, None, None),
            ("Extraction", _draw_signal_extraction_settings, None, None),
            ("Deconvolution", _draw_deconvolution_settings, "bool", "do_deconvolution"),
        ],
    ]
    n_rows = len(rows)

    viewport = imgui.get_main_viewport()
    style = imgui.get_style()
    spacing_x = style.item_spacing.x
    window_pad_x = style.window_padding.x

    col_naturals_per_row = [
        [_column_natural_width(t, mk) for (t, _, mk, _) in row] for row in rows
    ]

    row_natural_totals = [
        sum(nat) + spacing_x * (len(nat) - 1) for nat in col_naturals_per_row
    ]
    _content_w_static = max(row_natural_totals) + 2 * window_pad_x + 16
    _content_w_static = min(_content_w_static, viewport.size.x * 0.98)

    _just_opened = self._pipe_settings_just_opened
    _vp_size = viewport.size

    _pipe_sizer: PopupAutoSize = self._pipe_settings_sizer
    if _pipe_sizer is None:
        _pipe_sizer = PopupAutoSize("Pipeline Settings##pipeline_settings_popup")
        self._pipe_settings_sizer = _pipe_sizer
    _pipe_sizer.before_open()

    imgui.set_next_window_size_constraints(
        imgui.ImVec2(min(_content_w_static, _vp_size.x * 0.98), 200.0),
        imgui.ImVec2(_vp_size.x * 0.98, _vp_size.y * 0.98),
    )

    _popup_flags = _pipe_sizer.flags(imgui.WindowFlags_.no_saved_settings)

    opened, visible = imgui.begin_popup_modal(
        "Pipeline Settings##pipeline_settings_popup",
        p_open=True,
        flags=_popup_flags,
    )
    if opened:
        try:
            if not visible:
                _popup_states["pipeline_settings"] = False
                imgui.close_current_popup()
            else:
                avail = imgui.get_content_region_avail()

                col_idx = 0
                for row_i, row in enumerate(rows):
                    is_row2 = row_i == n_rows - 1
                    cols_in_row = len(row)
                    nat_widths = col_naturals_per_row[row_i]
                    nat_sum = sum(nat_widths)
                    avail_for_cols = avail.x - spacing_x * (cols_in_row - 1)
                    if avail_for_cols > nat_sum:
                        scale = avail_for_cols / nat_sum
                    else:
                        scale = 1.0
                    for ci, (title, draw_fn, mode_kind, mode_attr) in enumerate(row):
                        if ci > 0:
                            imgui.same_line()
                        col_w = max(1.0, nat_widths[ci] * scale)
                        child_size = imgui.ImVec2(col_w, 0)
                        child_flags = (
                            imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
                        )
                        if imgui.begin_child(
                            f"##pipe_col_{col_idx}",
                            child_size,
                            child_flags,
                        ):
                            self_titled = title in _SELF_TITLED_COLUMNS
                            if is_row2:
                                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                                imgui.set_next_item_allow_overlap()
                                imgui.push_style_color(
                                    imgui.Col_.text, imgui.ImVec4(1.0, 0.85, 0.4, 1.0)
                                )
                                expanded = imgui.collapsing_header(
                                    f"{title}##pipe_col_hdr_{col_idx}"
                                )
                                imgui.pop_style_color()
                            elif self_titled:
                                expanded = True
                            else:
                                imgui.text_colored(
                                    imgui.ImVec4(1.0, 0.85, 0.4, 1.0), title
                                )
                                expanded = True
                            skip_active = False
                            if mode_kind == "tri" and not self_titled:
                                cur = getattr(self.s2p, mode_attr)
                                imgui.same_line()
                                if imgui.radio_button(
                                    f"Skip##{mode_attr}_title", cur == 0
                                ):
                                    setattr(self.s2p, mode_attr, 0)
                                imgui.same_line()
                                if imgui.radio_button(
                                    f"Run##{mode_attr}_title", cur == 1
                                ):
                                    setattr(self.s2p, mode_attr, 1)
                                imgui.same_line()
                                if imgui.radio_button(
                                    f"Force##{mode_attr}_title", cur == 2
                                ):
                                    setattr(self.s2p, mode_attr, 2)
                                skip_active = getattr(self.s2p, mode_attr) == 0
                            elif mode_kind == "bool" and not self_titled:
                                cur = bool(getattr(self.s2p, mode_attr))
                                imgui.same_line()
                                if imgui.radio_button(
                                    f"Skip##{mode_attr}_title", not cur
                                ):
                                    setattr(self.s2p, mode_attr, False)
                                imgui.same_line()
                                if imgui.radio_button(f"Run##{mode_attr}_title", cur):
                                    setattr(self.s2p, mode_attr, True)
                                skip_active = not bool(getattr(self.s2p, mode_attr))

                            if expanded:
                                if not is_row2 and not self_titled:
                                    imgui.separator()
                                imgui.spacing()
                                imgui.begin_disabled(skip_active)
                                with tooltip_marks_right():
                                    try:
                                        draw_fn(self)
                                    except Exception as e:
                                        imgui.text_colored(
                                            imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                                            f"Error: {e}",
                                        )
                                imgui.end_disabled()
                        imgui.end_child()
                        col_idx += 1

                imgui.spacing()
                imgui.separator()
                _reset_label = "Defaults"
                _frame_pad_x = imgui.get_style().frame_padding.x
                _btn_w = imgui.calc_text_size("Run Suite2p").x + 2 * _frame_pad_x
                _close_w = 60
                _pad_x = imgui.get_style().window_padding.x

                imgui.push_style_color(
                    imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_active, imgui.ImVec4(0.1, 0.45, 0.1, 1.0)
                )
                if not has_save_path:
                    imgui.begin_disabled()
                _run_clicked = imgui.button(
                    "Run Suite2p##pipeline_settings_run", imgui.ImVec2(_btn_w, 0)
                )
                if not has_save_path:
                    imgui.end_disabled()
                imgui.pop_style_color(3)
                if not has_save_path and imgui.is_item_hovered(
                    imgui.HoveredFlags_.allow_when_disabled
                ):
                    imgui.set_tooltip(
                        "Set output path via Browse or by typing in the path field"
                    )
                if _run_clicked and has_save_path:
                    run_process(self)
                    _popup_states["pipeline_settings"] = False
                    imgui.close_current_popup()

                imgui.same_line()

                imgui.push_style_color(
                    imgui.Col_.button, imgui.ImVec4(0.55, 0.45, 0.15, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_hovered, imgui.ImVec4(0.70, 0.60, 0.22, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_active, imgui.ImVec4(0.45, 0.38, 0.12, 1.0)
                )
                if imgui.button(_reset_label, imgui.ImVec2(_btn_w, 0)):
                    from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
                        all_mapped_fields as _all_fields,
                    )
                    from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
                        get_default as _get_default,
                    )

                    for _f in _all_fields():
                        if hasattr(self.s2p, _f):
                            _target = self.s2p
                        elif hasattr(self.s2p_db, _f):
                            _target = self.s2p_db
                        else:
                            continue
                        _new = _get_default(_f)
                        _cur = getattr(_target, _f)
                        if _new is not None and _cur is not None:
                            _ct = type(_cur)
                            try:
                                if _ct is bool and not isinstance(_new, bool):
                                    _new = bool(_new)
                                elif _ct is int and isinstance(_new, bool):
                                    _new = int(_new)
                                elif _ct is int and isinstance(_new, float):
                                    _new = int(_new)
                                elif (
                                    _ct is float
                                    and isinstance(_new, int)
                                    and not isinstance(_new, bool)
                                ):
                                    _new = float(_new)
                            except (TypeError, ValueError, OverflowError):
                                continue
                        setattr(_target, _f, _new)
                    import dataclasses as _dc

                    from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
                        MBO_ONLY_FIELDS as _mbo_only,
                    )

                    for _obj in (self.s2p, self.s2p_db):
                        for _df in _dc.fields(_obj):
                            if _df.name in _mbo_only and _df.default is not _dc.MISSING:
                                setattr(_obj, _df.name, _df.default)
                    for _ef in _dc.fields(self.s2p_extras):
                        if _ef.default is not _dc.MISSING:
                            setattr(self.s2p_extras, _ef.name, _ef.default)
                        elif _ef.default_factory is not _dc.MISSING:
                            setattr(self.s2p_extras, _ef.name, _ef.default_factory())
                imgui.pop_style_color(3)
                set_tooltip(
                    "Reset every parameter in this popup to its dataclass "
                    "default. Both groups are tinted orange when modified.",
                    show_mark=False,
                )

                imgui.same_line()
                _force_legend = self._force_pipe_legend
                if _force_legend:
                    self._force_pipe_legend = False
                if (
                    imgui.button("Legend##pipe_legend", imgui.ImVec2(_btn_w, 0))
                    or _force_legend
                ):
                    imgui.open_popup("Legend##pipe_legend_popup")
                if imgui.begin_popup("Legend##pipe_legend_popup"):
                    imgui.text_colored(_S2P_TITLE_COLOR, "Suite2p parameter")
                    imgui.text_colored(_MBO_ONLY_COLOR, "LBM-Suite2p-Python parameter")
                    imgui.text_colored(_MODIFIED_COLOR, "Modified from default")
                    draw_boxed_label("Important parameter")
                    imgui.end_popup()

                imgui.same_line()
                imgui.set_cursor_pos_x(imgui.get_window_width() - _close_w - _pad_x)
                imgui.push_style_color(
                    imgui.Col_.button, imgui.ImVec4(0.55, 0.13, 0.13, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_hovered, imgui.ImVec4(0.65, 0.18, 0.18, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_active, imgui.ImVec4(0.45, 0.10, 0.10, 1.0)
                )
                if imgui.button("Close##pipeline_settings", imgui.ImVec2(_close_w, 0)):
                    _popup_states["pipeline_settings"] = False
                    imgui.close_current_popup()
                imgui.pop_style_color(3)

                if _just_opened:
                    self._pipe_settings_just_opened = False
        finally:
            imgui.end_popup()

    imgui.set_next_window_size(imgui.ImVec2(350, 0), imgui.Cond_.first_use_ever)
    opened, visible = imgui.begin_popup_modal(
        "Data Options##data_options",
        p_open=True,
        flags=imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.always_auto_resize,
    )
    if opened:
        try:
            if not visible:
                _popup_states["data_options"] = False
                imgui.close_current_popup()
            else:
                _draw_data_options_content(self)
                imgui.spacing()
                imgui.separator()
                if imgui.button("Close##data_opts", imgui.ImVec2(80, 0)):
                    _popup_states["data_options"] = False
                    imgui.close_current_popup()
        finally:
            imgui.end_popup()


def _build_channel_dirname(self, channel: int) -> str:
    """Build dimension-tagged dirname for a single channel run."""
    from mbo_utilities.arrays.features import TAG_REGISTRY, DimensionTag

    tags = []
    tags.append(DimensionTag(TAG_REGISTRY["C"], start=channel, stop=None, step=1))
    z_start = self._s2p_z_start
    z_stop = self._s2p_z_stop
    z_step = self._s2p_z_step
    tags.append(
        DimensionTag(TAG_REGISTRY["Z"], start=z_start, stop=z_stop, step=z_step)
    )
    tp = self._s2p_tp_parsed
    if tp and tp.final_indices:
        tp_start = tp.final_indices[0] + 1
        tp_stop = tp.final_indices[-1] + 1
        tags.append(
            DimensionTag(TAG_REGISTRY["T"], start=tp_start, stop=tp_stop, step=1)
        )
    return "_".join(tag.to_string() for tag in tags)


def run_process(self):
    """Runs the selected processing pipeline."""
    if self._current_pipeline != "suite2p":
        self.logger.error(f"Unknown pipeline selected: {self._current_pipeline}")
        return

    self.logger.debug(f"suite2p settings: {self.s2p}")
    if not _check_lsp_available():
        if not self._lsp_missing_warned:
            self.logger.warning(
                "lbm_suite2p_python is not installed. Please install it to run the Suite2p pipeline."
                "`uv pip install lbm_suite2p_python`",
            )
            self._lsp_missing_warned = True
        self._install_error = True
        return
    self._lsp_missing_warned = False

    if not self._install_error:
        selected_planes = self._selected_planes or {self.slice.z + 1}

        self.logger.info(
            f"Running Suite2p pipeline on {len(selected_planes)} plane(s)..."
        )

        use_background = self._s2p_background

        if use_background:
            from mbo_utilities.gui.widgets.process_manager import get_process_manager

            pm = get_process_manager()

            input_path = str(self.fpath) if self.fpath else ""

            s2p_path = self._s2p_outdir or self._saveas_outdir

            num_rois = (
                len(self.image_widget.graphics)
                if hasattr(self.image_widget, "graphics")
                else 1
            )
            roi = 1 if num_rois > 1 else None

            c_start = self._s2p_c_start
            c_stop = self._s2p_c_stop
            c_step = self._s2p_c_step
            selected_channels = list(range(c_start, c_stop + 1, c_step))
            multi_channel = len(selected_channels) > 1
            has_channels = self._s2p_last_num_channels > 1

            plane_list = sorted(selected_planes)

            for channel in selected_channels:
                if multi_channel:
                    output_dir = str(
                        Path(s2p_path) / _build_channel_dirname(self, channel)
                    )
                else:
                    output_dir = s2p_path

                worker_args = {
                    "input_path": input_path,
                    "reader_kwargs": widget_reader_kwargs(self.image_widget),
                    "output_dir": output_dir,
                    "planes": plane_list,
                    "roi": roi,
                    "num_timepoints": self.s2p_extras.target_timepoints,
                    "settings": self.s2p.to_dict(),
                    "db": self.s2p_db.to_dict(),
                    "fix_phase": self._s2p_fix_phase,
                    "use_fft": self._s2p_use_fft,
                    "frame_average": self._s2p_frame_average,
                    "channel": channel if (multi_channel or has_channels) else None,
                    "custom_metadata": dict(self._custom_metadata),
                    "tp_indices": (
                        list(self._s2p_tp_parsed.final_indices)
                        if self._s2p_tp_parsed is not None
                        else None
                    ),
                    "selected_planes_0based": [p - 1 for p in plane_list],
                    "register_z": self._register_z,
                    "max_frames": self._axial_max_frames,
                    "max_reg_xy": self._axial_max_reg_xy,
                    "s2p_settings": {
                        "keep_raw": self.s2p_db.keep_movie_raw,
                        "keep_reg": (not self.s2p.delete_bin),
                        "force_reg": (self.s2p.do_registration == 2),
                        "force_detect": (self.s2p.do_detection == 2),
                        "accept_all_cells": self.s2p_extras.accept_all_cells,
                        "norm_method": self.s2p_extras.norm_method,
                        "dff_window_size": self.s2p_extras.dff_window_size,
                        "dff_percentile": self.s2p_extras.dff_percentile,
                        "dff_smooth_window": self.s2p_extras.dff_smooth_window,
                        "save_json": self.s2p_extras.save_json,
                        "correct_neuropil": self.s2p_extras.correct_neuropil,
                        "cell_filters": build_cell_filters(
                            self.s2p_extras.min_diameter_um_enabled,
                            self.s2p_extras.min_diameter_um,
                            self.s2p_extras.max_diameter_um_enabled,
                            self.s2p_extras.max_diameter_um,
                            baseline_filter_enabled=self.s2p_extras.baseline_filter_enabled,
                            baseline_reject_negative_F0=self.s2p_extras.baseline_reject_negative_F0,
                            baseline_min_F0_abs_enabled=self.s2p_extras.baseline_min_F0_abs_enabled,
                            baseline_min_F0_abs=self.s2p_extras.baseline_min_F0_abs,
                            baseline_min_F0_rel_enabled=self.s2p_extras.baseline_min_F0_rel_enabled,
                            baseline_min_F0_rel=self.s2p_extras.baseline_min_F0_rel,
                            correct_neuropil=self.s2p_extras.correct_neuropil,
                        ),
                        "rastermap_kwargs": build_rastermap_kwargs(self.s2p_extras),
                        "force_rastermap": (self.s2p_extras.rastermap_mode == 2),
                        "workers": self.s2p_extras.workers,
                        "threads_per_worker": self.s2p_extras.threads_per_worker,
                        "skip_volumetric": self.s2p_extras.skip_volumetric,
                    },
                }

                if len(plane_list) == 1:
                    description = f"Suite2p plane{plane_list[0]:02d}"
                else:
                    description = f"Suite2p: {len(plane_list)} plane(s)"
                if multi_channel or has_channels:
                    description += f" ch{channel}"
                if roi:
                    description += f" ROI {roi}"

                pid = pm.spawn(
                    task_type="suite2p",
                    args=worker_args,
                    description=description,
                    output_path=output_dir,
                )

                if not pid:
                    self.logger.error(
                        f"Failed to start background process for {description}"
                    )
        else:
            c_start = self._s2p_c_start
            c_stop = self._s2p_c_stop
            c_step = self._s2p_c_step
            selected_channels = list(range(c_start, c_stop + 1, c_step))
            multi_channel = len(selected_channels) > 1
            has_channels = self._s2p_last_num_channels > 1

            s2p_path = self._s2p_outdir or self._saveas_outdir

            s2p_settings_dict = self.s2p.to_dict()
            s2p_db_dict = self.s2p_db.to_dict()
            target_timepoints = self.s2p_extras.target_timepoints
            keep_raw = self.s2p_db.keep_movie_raw
            keep_reg = not self.s2p.delete_bin
            force_reg = self.s2p.do_registration == 2
            force_detect = self.s2p.do_detection == 2
            accept_all_cells = self.s2p_extras.accept_all_cells
            norm_method = self.s2p_extras.norm_method
            dff_window_size = self.s2p_extras.dff_window_size
            dff_percentile = self.s2p_extras.dff_percentile
            dff_smooth_window = self.s2p_extras.dff_smooth_window
            save_json = self.s2p_extras.save_json
            correct_neuropil = self.s2p_extras.correct_neuropil
            cell_filters = build_cell_filters(
                self.s2p_extras.min_diameter_um_enabled,
                self.s2p_extras.min_diameter_um,
                self.s2p_extras.max_diameter_um_enabled,
                self.s2p_extras.max_diameter_um,
                baseline_filter_enabled=self.s2p_extras.baseline_filter_enabled,
                baseline_reject_negative_F0=self.s2p_extras.baseline_reject_negative_F0,
                baseline_min_F0_abs_enabled=self.s2p_extras.baseline_min_F0_abs_enabled,
                baseline_min_F0_abs=self.s2p_extras.baseline_min_F0_abs,
                baseline_min_F0_rel_enabled=self.s2p_extras.baseline_min_F0_rel_enabled,
                baseline_min_F0_rel=self.s2p_extras.baseline_min_F0_rel,
                correct_neuropil=correct_neuropil,
            )
            planar_rastermap_kwargs = build_planar_rastermap_kwargs(self.s2p_extras)
            force_rastermap = self.s2p_extras.rastermap_mode == 2
            fix_phase = self._s2p_fix_phase
            use_fft = self._s2p_use_fft
            frame_average = self._s2p_frame_average

            if not s2p_path:
                from mbo_utilities.file_io import get_last_savedir_path
                from mbo_utilities.preferences import get_mbo_dirs

                last_savedir = get_last_savedir_path()
                s2p_path = str(
                    Path(last_savedir) if last_savedir else get_mbo_dirs()["data"]
                )

            _fpath = self.fpath
            if isinstance(_fpath, (list, tuple)):
                _fpath = _fpath[0] if _fpath else None
            fpath_str = str(_fpath) if _fpath else ""

            tp_indices = (
                list(self._s2p_tp_parsed.final_indices)
                if self._s2p_tp_parsed is not None
                else None
            )
            selected_planes_0based = [p - 1 for p in sorted(selected_planes)]

            jobs = []
            for i, _arr in enumerate(self.image_widget.data):
                for channel in selected_channels:
                    if multi_channel:
                        base_out = Path(s2p_path) / _build_channel_dirname(
                            self, channel
                        )
                    else:
                        base_out = Path(s2p_path)

                    for z_plane in sorted(selected_planes):
                        current_z = z_plane - 1

                        if len(self.image_widget.graphics) > 1:
                            plane_dir = base_out / f"plane{z_plane:02d}_roi{i + 1:02d}"
                            roi = i + 1
                        else:
                            plane_dir = base_out / f"plane{z_plane:02d}_stitched"
                            roi = None

                        import copy as _copy

                        config = {
                            "arr": _arr,
                            "arr_idx": i,
                            "z_plane": current_z,
                            "plane": z_plane,
                            "channel": channel
                            if (multi_channel or has_channels)
                            else None,
                            "base_out": base_out,
                            "plane_dir": plane_dir,
                            "roi": roi,
                            "s2p_settings_dict": _copy.deepcopy(s2p_settings_dict),
                            "s2p_db_dict": _copy.deepcopy(s2p_db_dict),
                            "target_timepoints": target_timepoints,
                            "fpath": fpath_str,
                            "keep_raw": keep_raw,
                            "keep_reg": keep_reg,
                            "force_reg": force_reg,
                            "force_detect": force_detect,
                            "accept_all_cells": accept_all_cells,
                            "norm_method": norm_method,
                            "dff_window_size": dff_window_size,
                            "dff_percentile": dff_percentile,
                            "dff_smooth_window": dff_smooth_window,
                            "save_json": save_json,
                            "correct_neuropil": correct_neuropil,
                            "cell_filters": cell_filters,
                            "planar_rastermap_kwargs": planar_rastermap_kwargs,
                            "force_rastermap": force_rastermap,
                            "fix_phase": fix_phase,
                            "use_fft": use_fft,
                            "frame_average": frame_average,
                            "custom_metadata": dict(self._custom_metadata),
                            "tp_indices": tp_indices,
                            "selected_planes_0based": selected_planes_0based,
                            "logger": self.logger,
                        }

                        jobs.append(config)

            threading.Thread(
                target=_run_planes_sequential, args=(self, jobs), daemon=True
            ).start()


def _run_planes_sequential(self, jobs: list[dict]) -> None:
    for job_idx, config in enumerate(jobs):
        desc = f"plane {config['plane']}"
        if config["channel"] is not None:
            desc += f" ch{config['channel']}"
        self.logger.info(f"Processing {desc} ({job_idx + 1}/{len(jobs)})...")
        try:
            _run_plane_worker_thread(config)
        except Exception as e:
            self.logger.exception(f"Error processing {desc}: {e}")
    self.logger.info("Suite2p processing complete.")


def _rebase_worker_provenance(md: dict, fpath, logger) -> dict:
    """Repair stale recorded paths in worker metadata against the loaded tree.

    ``fpath`` is the viewer's loaded path: a list/tuple unwraps to its first
    element (legacy multi-file loads — str() of the list is path garbage).
    An empty/None fpath skips the rebase AND the data_path fallback
    entirely: Path('') is '.', so anchoring there would embed cwd-relative
    provenance (md['data_path'] = ['.']) into ops.npy.
    """
    if isinstance(fpath, (list, tuple)):
        fpath = fpath[0] if fpath else None
    if not fpath:
        return md

    from mbo_utilities.metadata.base import rebase_provenance_paths

    anchor = Path(str(fpath))
    anchor = anchor if anchor.is_dir() else anchor.parent
    md = rebase_provenance_paths(md, anchor, logger)
    if "data_path" not in md:
        md["data_path"] = [str(anchor)]
    return md


def _run_plane_worker_thread(config):
    """
    Decoupled pure worker function for processing planes on a thread.
    Takes a pre-configured dict of snapshot variables to prevent GUI data racing.
    """
    if not _check_lsp_available():
        if config["logger"]:
            config["logger"].error("lbm_suite2p_python is not installed.")
        return

    arr = config["arr"]
    arr_idx = config["arr_idx"]
    current_z = config["z_plane"]
    plane = config["plane"]
    channel = config["channel"]
    base_out = config["base_out"]
    plane_dir = config["plane_dir"]
    roi = config["roi"]

    if not base_out.exists():
        base_out.mkdir(parents=True, exist_ok=True)

    ops_path = plane_dir / "ops.npy"
    lazy_mdata = dict(arr.metadata)

    custom_metadata = config.get("custom_metadata") or {}
    if custom_metadata:
        lazy_mdata.update(custom_metadata)

    Lx = arr.shape[-1]
    Ly = arr.shape[-2]

    from mbo_utilities.metadata import OutputMetadata, get_param

    tp_indices = config.get("tp_indices")
    selected_planes_0based = config.get("selected_planes_0based")

    selections = {}
    if tp_indices is not None:
        selections["T"] = list(tp_indices)
    if selected_planes_0based is not None:
        selections["Z"] = list(selected_planes_0based)

    source_shape = tuple(arr._shape5d())
    source_dims = ("T", "C", "Z", "Y", "X")

    raw_src_fs = get_param(lazy_mdata, "fs")
    raw_src_dz = get_param(lazy_mdata, "dz")
    config["logger"].info(
        f"_run_plane_worker_thread: source fs={raw_src_fs}, dz={raw_src_dz}, "
        f"plane={plane}"
    )

    if raw_src_fs is None:
        config["logger"].warning(
            f"_run_plane_worker_thread: source metadata for plane {plane} has "
            f"NO fs field — reactive scaling cannot compute the output rate. "
            f"ops.npy fs will fall through to lbm_suite2p_python's default "
            f"(10 Hz). To fix: set fs via the metadata editor before running, "
            f"or fix the source TIFF metadata."
        )

    out_meta = OutputMetadata(
        source=lazy_mdata,
        source_shape=source_shape,
        source_dims=source_dims,
        selections=selections,
    )

    md = out_meta.to_dict()

    md = _rebase_worker_provenance(md, config["fpath"], config["logger"])

    for key in ("fs", "dz"):
        if key in md and md[key] is None:
            md.pop(key)

    md["process_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    md["original_file"] = config["fpath"]
    md["roi_index"] = arr_idx
    md["z_index"] = current_z
    md["plane"] = plane
    md["Ly"] = Ly
    md["Lx"] = Lx
    md["ops_path"] = str(ops_path)
    md["save_path"] = str(plane_dir)
    md["raw_file"] = str((plane_dir / "data_raw.bin").resolve())

    config["logger"].info(
        f"_run_plane_worker_thread: applied reactive metadata -> "
        f"fs={md.get('fs')}, dz={md.get('dz')} "
        f"(t-stride from {len(tp_indices) if tp_indices else 0} indices, "
        f"z-stride from {len(selected_planes_0based) if selected_planes_0based else 0} planes)"
    )

    settings_dict = config["s2p_settings_dict"]
    db_dict = config["s2p_db_dict"]

    extra_ops: dict = {}
    _settings_top = {"fs", "tau", "diameter"}
    _db_keys = {
        "nplanes",
        "nchannels",
        "data_path",
        "save_path0",
        "fast_disk",
        "save_folder",
        "h5py_key",
        "functional_chan",
        "nrois",
    }
    for key, value in md.items():
        if key == "fs":
            settings_dict["fs"] = value
        elif key in _settings_top:
            settings_dict[key] = value
        elif key in _db_keys:
            db_dict[key] = value
        else:
            extra_ops[key] = value

    if channel is not None:
        db_dict["functional_chan"] = 1
        settings_dict.setdefault("registration", {})["align_by_chan2"] = False

    from mbo_utilities.writer import imwrite

    plane_dir.mkdir(parents=True, exist_ok=True)

    if tp_indices:
        frames_arg = [i + 1 for i in tp_indices]
    else:
        frames_arg = None

    from lbm_suite2p_python.db_settings import db_settings_to_ops

    metadata_flat = {**db_settings_to_ops(db_dict, settings_dict), **extra_ops}

    imwrite(
        arr,
        plane_dir,
        ext=".bin",
        overwrite=True,
        register_z=False,
        planes=plane,
        channels=[channel] if channel is not None else None,
        output_name="data_raw.bin",
        roi=roi,
        metadata=metadata_flat,
        frames=frames_arg,
        num_frames=config["target_timepoints"] if frames_arg is None else None,
        fix_phase=config["fix_phase"],
        use_fft=config["use_fft"],
        frame_average=config.get("frame_average", 1),
    )

    from lbm_suite2p_python import run_plane

    raw_file = plane_dir / "data_raw.bin"

    np.save(plane_dir / "db.npy", db_dict, allow_pickle=True)
    np.save(plane_dir / "settings.npy", settings_dict, allow_pickle=True)

    config["logger"].info(
        f"Suite2p run — do_detection: {settings_dict.get('run', {}).get('do_detection')}, "
        f"force_detect: {config['force_detect']}"
    )
    config["logger"].info(
        f"Detection params — algorithm: {settings_dict.get('detection', {}).get('algorithm')}, "
        f"diameter: {settings_dict.get('diameter')}"
    )

    if (
        config.get("force_rastermap")
        and config.get("planar_rastermap_kwargs") is not None
    ):
        cached_model = plane_dir / "model.npy"
        if cached_model.is_file():
            cached_model.unlink()
            config["logger"].info(
                f"Force rastermap: removed cached {cached_model.name}"
            )

    try:
        run_plane(
            input_data=raw_file,
            save_path=plane_dir,
            db=db_dict,
            settings=settings_dict,
            ops=extra_ops if extra_ops else None,
            keep_raw=config["keep_raw"],
            keep_reg=config["keep_reg"],
            force_reg=config["force_reg"],
            force_detect=config["force_detect"],
            accept_all_cells=config.get("accept_all_cells", False),
            norm_method=config.get("norm_method", "dff"),
            dff_window_size=config["dff_window_size"],
            dff_percentile=config["dff_percentile"],
            dff_smooth_window=config["dff_smooth_window"]
            if config["dff_smooth_window"] > 0
            else None,
            save_json=config.get("save_json", False),
            correct_neuropil=config.get("correct_neuropil", True),
            cell_filters=config.get("cell_filters") or None,
            rastermap_kwargs=config.get("planar_rastermap_kwargs"),
        )
        config["logger"].info(
            f"Suite2p processing complete for plane {current_z}, roi {arr_idx}. Results in {plane_dir}"
        )

        stat_file = plane_dir / "stat.npy"
        if stat_file.exists():
            config["logger"].info("Detection succeeded - stat.npy created")
        else:
            config["logger"].warning(
                "Detection did not run - stat.npy not found. Check Suite2p output logs."
            )
    except Exception as e:
        config["logger"].exception(
            f"Suite2p processing failed for plane {current_z}, roi {arr_idx}: {e}"
        )
