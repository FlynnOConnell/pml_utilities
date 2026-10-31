"""The metadata registry: ``METADATA_PARAMS`` (one ``MetadataParameter`` per canonical key), ``VoxelSize``, the export denylist and provenance-path repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple


def _logger():
    # mbo_utilities.log runs the package init, which imports this module
    from mbo_utilities import log

    return log.get("metadata")


@dataclass
class MetadataParameter:
    """One physical quantity: its canonical key, the spellings that hold the same value verbatim (``aliases``) and the keys that hold a converted form (``transforms``: ``{key: (to_canonical, from_canonical)}``, e.g. ImageJ ``finterval`` = 1/``fs``)."""

    canonical: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    dtype: type = float
    unit: str | None = None
    default: Any = None
    description: str = ""
    label: str = ""
    transforms: dict[str, tuple] = field(default_factory=dict)


class VoxelSize(NamedTuple):
    """(dx, dy, dz) in µm; ``dz`` is None for an LBM stack until the user supplies it."""

    dx: float
    dy: float
    dz: float | None

    @property
    def pixel_resolution(self) -> tuple[float, float]:
        return (self.dx, self.dy)

    @property
    def voxel_size(self) -> tuple[float, float, float | None]:
        return (self.dx, self.dy, self.dz)

    def to_dict(self, include_aliases: bool = True) -> dict:
        """The canonical keys plus ``pixel_resolution`` and ``voxel_size``; with ``include_aliases`` the OME ``PhysicalSize*`` keys and ``z_step`` too."""
        result = {
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "pixel_resolution": self.pixel_resolution,
            "voxel_size": self.voxel_size,
        }

        if include_aliases:
            result["PhysicalSizeX"] = self.dx
            result["PhysicalSizeY"] = self.dy
            result["PhysicalSizeZ"] = self.dz
            result["PhysicalSizeXUnit"] = "micrometer"
            result["PhysicalSizeYUnit"] = "micrometer"
            result["PhysicalSizeZUnit"] = "micrometer"
            result["z_step"] = self.dz
        return result


def _reciprocal(value: Any) -> float:
    """``1 / value``: a rate to its interval, pixels per µm to µm per pixel, and back."""
    v = float(value)
    if v == 0:
        raise ZeroDivisionError("cannot invert a zero-valued metadata field")
    return 1.0 / v


METADATA_PARAMS: dict[str, MetadataParameter] = {
    "dx": MetadataParameter(
        canonical="dx",
        aliases=(
            "Dx",
            "PhysicalSizeX",
            "pixelResolutionX",
            "pixel_size_x",
            "pixel_resolution_um",
            "pixel_size_um",  # the MINI2P converter's one scalar for x and y
        ),
        dtype=float,
        unit="µm",
        default=1.0,
        description="Pixel size in X dimension (µm/pixel)",
        label="Pixel Size X",
        transforms={"XResolution": (_reciprocal, _reciprocal)},  # ImageJ: pixels per µm
    ),
    "dy": MetadataParameter(
        canonical="dy",
        aliases=(
            "Dy",
            "PhysicalSizeY",
            "pixelResolutionY",
            "pixel_size_y",
            "pixel_size_um",  # ALIAS_MAP keeps dy; get_param walks dx's own tuple so dx resolves it too
        ),
        dtype=float,
        unit="µm",
        default=1.0,
        description="Pixel size in Y dimension (µm/pixel)",
        label="Pixel Size Y",
        transforms={"YResolution": (_reciprocal, _reciprocal)},
    ),
    "dz": MetadataParameter(
        canonical="dz",
        aliases=(
            "Dz",
            "PhysicalSizeZ",
            "z_step",
            "z_step_um",
            "spacing",
            "pixelResolutionZ",
            "ZResolution",
            "axial_step",
            "axial_step_um",
        ),
        dtype=float,
        unit="µm",
        default=None,
        description="Voxel size in Z dimension (µm/z-step). Must be user-supplied for LBM.",
        label="Z Step",
    ),
    "fs": MetadataParameter(
        canonical="fs",
        aliases=(
            "frame_rate",
            "framerate",
            "fr",
            "sampling_frequency",
            "frameRate",
            "scanFrameRate",
            "fps",
            "frame_rate_hz",
        ),
        dtype=float,
        unit="Hz",
        default=None,
        description="Frame rate / sampling frequency (Hz)",
        label="Frame Rate",
        transforms={"finterval": (_reciprocal, _reciprocal)},
    ),
    "vps": MetadataParameter(
        canonical="vps",
        aliases=("volumes_per_second",),
        dtype=float,
        unit="Hz",
        default=None,
        description="Volumetric acquisition rate (volumes per second)",
        label="Volume Rate",
    ),
    "finterval": MetadataParameter(
        canonical="finterval",
        aliases=(
            "frame_interval",
            "FrameInterval",
            "dt",
            "time_interval",
            "frame_period",
            "TimeIncrement",
        ),
        dtype=float,
        unit="s",
        default=None,
        description="Frame interval in seconds (1/fs). Used by ImageJ/Fiji and OME.",
        label="Frame Interval",
        transforms={"fs": (_reciprocal, _reciprocal)},
    ),
    "Lx": MetadataParameter(
        canonical="Lx",
        aliases=(
            "lx",
            "LX",
            "width",
            "nx",
            "size_x",
            "SizeX",
            "image_width",
            "fov_x",
            "num_px_x",
            "page_width",
            "ImageWidth",
        ),
        dtype=int,
        unit="px",
        default=None,
        description="Image width in pixels",
    ),
    "Ly": MetadataParameter(
        canonical="Ly",
        aliases=(
            "ly",
            "LY",
            "height",
            "ny",
            "size_y",
            "SizeY",
            "image_height",
            "fov_y",
            "num_px_y",
            "page_height",
            "ImageLength",
        ),
        dtype=int,
        unit="px",
        default=None,
        description="Image height in pixels",
    ),
    "frames_per_file": MetadataParameter(
        canonical="frames_per_file",
        aliases=(
            "frames_per_folder",
            "nframes_per_file",
            "frames_per_tiff",
        ),
        dtype=list,
        default=None,
        description=(
            "Timepoints contributed by each source file, in the order of "
            "file_paths. Marks trial/acquisition boundaries along T for "
            "pipelines that treat each file as a separate trial."
        ),
        label="Frames per file",
    ),
    "file_paths": MetadataParameter(
        canonical="file_paths",
        aliases=(
            "filenames",
            "filelist",
            "tiff_list",
        ),
        dtype=list,
        default=None,
        description="Source files backing the array, ordered along T",
        label="Source files",
    ),
    "num_timepoints": MetadataParameter(
        canonical="num_timepoints",
        aliases=(
            "nframes",
            "num_frames",
            "n_frames",
            "frames",
            "T",
            "nt",
            "timepoints",
            "n_timepoints",
            "SizeT",
        ),
        dtype=int,
        default=None,
        description="Number of timepoints (T dimension) in the dataset",
        label="Timepoints",
    ),
    "num_zplanes": MetadataParameter(
        canonical="num_zplanes",
        aliases=(
            "num_planes",
            "nplanes",
            "n_planes",
            "planes",
            "Z",
            "nz",
            "num_z",
            "numPlanes",
            "zplanes",
            "slices",
            "SizeZ",
        ),
        dtype=int,
        default=1,
        description="Number of z-planes",
        label="Num Z-Planes",
    ),
    "nchannels": MetadataParameter(
        canonical="nchannels",
        aliases=(
            "num_channels",
            "n_channels",
            "channels",
            "C",
            "nc",
            "numChannels",
        ),
        dtype=int,
        default=1,
        description="Number of channels (typically 1 for calcium imaging)",
    ),
    "dtype": MetadataParameter(
        canonical="dtype",
        aliases=("data_type", "pixel_type", "datatype"),
        dtype=str,
        default="int16",
        description="Data type of pixel values",
    ),
    "size": MetadataParameter(
        canonical="size",
        aliases=("num_elements", "total_elements"),
        dtype=int,
        default=None,
        description="Total number of elements in the array (product of dimensions)",
    ),
    "shape": MetadataParameter(
        canonical="shape",
        aliases=("array_shape", "data_shape"),
        dtype=tuple,
        default=None,
        description="Array shape as tuple (T, Z, Y, X) or (T, Y, X) or (Y, X)",
    ),
    "stack_type": MetadataParameter(
        canonical="stack_type",
        aliases=("stackType",),
        dtype=str,
        default="single_plane",
        description="Stack type: lbm, piezo, or single_plane",
    ),
    "lbm_stack": MetadataParameter(
        canonical="lbm_stack",
        aliases=("is_lbm", "lbmStack"),
        dtype=bool,
        default=False,
        description="True if Light Beads Microscopy stack",
    ),
    "piezo_stack": MetadataParameter(
        canonical="piezo_stack",
        aliases=("is_piezo", "piezoStack"),
        dtype=bool,
        default=False,
        description="True if piezo-driven z-stack",
    ),
    "num_color_channels": MetadataParameter(
        canonical="num_color_channels",
        aliases=("color_channels", "ncolors", "num_colors", "n_channel", "SizeC"),
        dtype=int,
        default=1,
        description="Number of color channels (1 or 2)",
    ),
    "num_mrois": MetadataParameter(
        canonical="num_mrois",
        aliases=("num_rois", "scanimage_multirois", "numROIs", "nrois", "n_rois"),
        dtype=int,
        default=1,
        description="Number of mROIs (ScanImage multi-ROI scan regions)",
    ),
    "roi": MetadataParameter(
        canonical="roi",
        aliases=("roi_size", "roi_px"),
        dtype=tuple,
        unit="px",
        default=None,
        description="ROI dimensions as (width, height) in pixels",
    ),
    "fov": MetadataParameter(
        canonical="fov",
        aliases=("fov_px",),
        dtype=tuple,
        unit="px",
        default=None,
        description="Field of view as (x, y) in pixels (tiled)",
    ),
    "fov_um": MetadataParameter(
        canonical="fov_um",
        aliases=(),
        dtype=tuple,
        unit="µm",
        default=None,
        description="Field of view as (x, y) in µm (tiled)",
    ),
}


def _build_alias_map() -> dict[str, str]:
    alias_map = {}
    for param in METADATA_PARAMS.values():
        alias_map[param.canonical.lower()] = param.canonical
        for alias in param.aliases:
            alias_map[alias.lower()] = param.canonical
    return alias_map


ALIAS_MAP: dict[str, str] = _build_alias_map()


def get_canonical_name(name: str) -> str | None:
    """The canonical key for any registered spelling, None for an unregistered one."""
    return ALIAS_MAP.get(name.lower())


IMAGING_METADATA_KEYS: tuple[str, ...] = (  # the metadata viewer always shows these
    "fs",
    "finterval",  # with fs, so dt and frame_interval do not contradict it under "Other"
    "dx",
    "dy",
    "dz",
    "Lx",
    "Ly",
    "num_zplanes",
    "num_color_channels",
    "num_mrois",
    "num_timepoints",
    "dtype",
)


_SUITE2P_REGISTRATION_INTERNALS = (  # regPC alone is 500+ MB as JSON
    "regPC",
    "tPC",
    "regDX",
    "yblock",
    "xblock",
    "NRsm",
)

_SUITE2P_SUMMARY_IMAGES = (
    "meanImg",
    "meanImgE",
    "meanImg_chan2",
    "meanImg_crop",
    "Vmap",
    "Vcorr",
    "Vsplit",
    "Vmax",
    "max_proj",
    "refImg",
    "refImg1",
    "refAndMasks",
)

_PER_FRAME_VECTORS = (
    "xoff",
    "yoff",
    "corrXY",
    "xoff1",
    "yoff1",
    "corrXY1",
    "badframes",
    "badframes0",
    "ihop",
    "plane_times",
)

_MBO_ADDITIONS = (  # plane_shifts stays: viewers align planes from it at read time
    "processing_history",
    "_metadata_provenance",
    "roi_mode",
)

_SUITE2P_GEOMETRY = (
    "Ly",
    "Lx",
    "nframes",
    "nplanes",
    "nchannels",
    "num_rois",
    "aspect",
    "tau",
    "functional_chan",
    "align_by_chan",
)

_SUITE2P_PIPELINE_SETTINGS = (
    "do_registration",
    "keep_movie_raw",
    "two_step_registration",
    "nimg_init",
    "multiplane_parallel",
    "nbinned",
    "batch_size",
    "diameter",
    "cell_diameter",
    "spatial_scale",
    "spatscale_pix",
    "roidetect",
    "spikedetect",
    "neuropil_extract",
    "denoise",
    "anatomical_only",
    "sparse_mode",
    "connected",
    "threshold_scaling",
    "max_overlap",
    "max_iterations",
    "high_pass",
    "smooth_sigma",
    "smooth_sigma_time",
    "nonrigid",
    "block_size",
    "snr_thresh",
    "maxregshift",
    "use_builtin_classifier",
    "classifier_path",
    "preclassify",
    "chan2_thres",
    "lam_percentile",
    "allow_overlap",
    "inner_neuropil_radius",
    "min_neuropil_pixels",
    "neucoeff",
    "soma_crop",
    "win_baseline",
    "sig_baseline",
    "prctile_baseline",
    "data_path",
    "save_path",
    "save_path0",
    "save_folder",
    "fast_disk",
    "ops_path",
    "input_format",
    "save_NWB",
    "save_mat",
    "first_tiffs",
    "frames_include",
    "h5py",
    "h5py_key",
    "delete_bin",
    "combined",
    "report_time",
    "do_bidiphase",
    "bidiphase",
    "1Preg",
    "spatial_hp",
    "pre_smooth",
    "spatial_taper",
)

EXPORT_DENYLIST: frozenset[str] = frozenset(
    _SUITE2P_REGISTRATION_INTERNALS
    + _SUITE2P_SUMMARY_IMAGES
    + _PER_FRAME_VECTORS
    + _MBO_ADDITIONS
    + _SUITE2P_GEOMETRY
    + _SUITE2P_PIPELINE_SETTINGS
)

_MAX_EXPORT_ELEMENTS = (
    8192  # above this a value is image or vector payload unless allow-listed
)
_SIZE_GUARD_ALLOW = frozenset(
    {
        "plane_shifts",
        "scanphase",
        "frames_per_file",
        "roi_groups",
        "timepoint_selection",
        "file_paths",
        "ome",
    }
)


def _element_count(value) -> int:
    """Leaf elements in a scalar, ndarray or nested list or tuple."""
    total = 0
    stack = [value]
    while stack:
        v = stack.pop()
        size = getattr(v, "size", None)  # ndarrays and numpy scalars
        if isinstance(size, int):
            total += size
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        else:
            total += 1
    return total


def strip_for_export(md: dict) -> dict:
    """``md`` without the ``EXPORT_DENYLIST`` keys (suite2p-only fields, mbo internals) and without any value over ``_MAX_EXPORT_ELEMENTS`` that is not allow-listed."""
    out = {}
    for k, v in md.items():
        if k in EXPORT_DENYLIST:
            continue
        if k not in _SIZE_GUARD_ALLOW:
            n = _element_count(v)
            if n > _MAX_EXPORT_ELEMENTS:
                _logger().warning(
                    "dropping oversized metadata %r (%d elements) from export",
                    k,
                    n,
                )
                continue
        out[k] = v
    return out


_OPS_ARRAY_KEYS: frozenset[str] = (
    frozenset(  # suite2p indexes these with .shape, so they must be ndarrays
        _SUITE2P_SUMMARY_IMAGES
        + _PER_FRAME_VECTORS
        + _SUITE2P_REGISTRATION_INTERNALS
        + ("sdmov", "refImg0")
    )
)


def normalize_ops_arrays(ops: dict) -> dict:
    """``ops`` with its image and per-frame fields back as ndarrays after a trip through JSON.

    An empty one is dropped, not saved as an empty array: the consumers test
    for the key's absence and lbm_suite2p_python recomputes ``meanImgE``
    from ``meanImg`` when it is missing.
    """
    import numpy as np

    out = {}
    for k, v in ops.items():
        if k not in _OPS_ARRAY_KEYS:
            out[k] = v
            continue
        if isinstance(v, (list, tuple)):
            if not len(v):
                continue
            v = np.asarray(v)
            if v.dtype == np.float64:
                v = v.astype(np.float32)  # suite2p writes these as float32
        if isinstance(v, np.ndarray) and v.size == 0:
            continue
        out[k] = v
    return out


def repair_ops_file(path) -> bool:
    """Rewrite one ``ops.npy`` whose array fields are JSON lists (an older export; a detection-only re-run copies it as is); True when rewritten."""
    import numpy as np

    path = Path(path)
    if not path.exists():
        return False
    ops = np.load(path, allow_pickle=True).item()
    if not isinstance(ops, dict):
        return False
    bad = [
        k
        for k, v in ops.items()
        if k in _OPS_ARRAY_KEYS and isinstance(v, (list, tuple))
    ]
    if not bad:
        return False
    np.save(path, normalize_ops_arrays(ops))
    _logger().info("repaired %s: %s were lists, not arrays", path, ", ".join(bad))
    return True


def repair_ops_tree(path) -> int:
    """``repair_ops_file`` on a run dir, its plane children and the suite2p/planeNN level below; the count rewritten. A file inside the tree names its dir."""
    path = Path(path)
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        return 0
    targets = [path / "ops.npy"]
    for pattern in ("*/ops.npy", "*/*/ops.npy"):
        targets += sorted(path.glob(pattern))
    seen, fixed = set(), 0
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        fixed += bool(repair_ops_file(target))
    return fixed


PROVENANCE_PATH_KEYS: tuple[
    str, ...
] = (  # paths a run recorded; stale once its folder is copied elsewhere
    "data_path",
    "save_path",
    "save_path0",
    "fast_disk",
    "ops_path",
    "raw_file",
    "reg_file",
    "raw_source",
    "file_paths",
    "file_list",
)


def _path_basename(p) -> str:
    """The last path component, whichever OS wrote the separators."""
    return str(p).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _rebase_dirs(anchor: Path) -> list[Path]:
    """``anchor``, its three ancestors and each one's ``raw/``: a plane dir sits at <share>/<user>/<group>/<run>/<plane> with ``raw/`` beside <group>."""
    dirs: list[Path] = []
    for d in (
        anchor,
        anchor.parent,
        anchor.parent.parent,
        anchor.parent.parent.parent,
    ):
        for sub in (d, d / "raw"):
            if sub not in dirs:
                dirs.append(sub)
    return dirs


def _rebase_one(dead, dirs: list[Path]) -> Path | None:
    name = _path_basename(dead)
    if not name:
        return None
    for d in dirs:
        if (d / name).exists():
            return d / name
    return None


def rebase_provenance_paths(md: dict, anchor: Path | str, logger=None) -> dict:
    """A copy of ``md`` whose ``PROVENANCE_PATH_KEYS`` name files on this machine: a dead path is rebased by basename onto ``_rebase_dirs(anchor)``, or dropped.

    None, empty and non-path values pass through untouched.
    """
    lg = logger or _logger()
    anchor = Path(anchor)
    dirs = _rebase_dirs(anchor)
    out = dict(md)
    for key in PROVENANCE_PATH_KEYS:
        if key not in out:
            continue
        val = out[key]
        if val is None:
            continue
        if isinstance(val, (str, Path)):
            if not str(val):
                continue
            entries, was_scalar = [val], True
        elif isinstance(val, (list, tuple)):
            if not val or not all(isinstance(v, (str, Path)) for v in val):
                continue
            entries, was_scalar = list(val), False
        else:
            continue
        kept: list[str] = []
        for entry in entries:
            if Path(entry).exists():
                kept.append(str(entry))
                continue
            found = _rebase_one(entry, dirs)
            if found is not None:
                lg.info(f"provenance rebase: {key} {str(entry)!r} -> {found}")
                kept.append(str(found))
            else:
                lg.info(
                    f"provenance: dropping stale {key}={str(entry)!r} "
                    f"(path does not exist on this machine)"
                )
        if not kept:
            del out[key]
        elif was_scalar:
            out[key] = kept[0]
        else:
            out[key] = kept
    return out
