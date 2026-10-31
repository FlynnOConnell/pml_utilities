"""Suite2p output (``data.bin`` / ``data_raw.bin`` + ``ops.npy``) as a lazy array; plane dirs stack on Z."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import (
    ReductionMixin,
    Shape5DMixin,
    _imwrite_base,
    _normalize_key,
)
from mbo_utilities.arrays.features import MotionCorrection
from mbo_utilities.file_io import has_plane_ops, load_npy, load_plane_ops
from mbo_utilities.lazy_array import register_array_class
from mbo_utilities.metadata import get_param, normalize_ops_arrays
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

_SUITE2P_INFO = PipelineInfo(
    name="suite2p",
    description="Suite2p binary output files (registered/raw)",
    input_patterns=[
        "**/ops.npy",
        "**/data.bin",
        "**/data_raw.bin",
    ],
    output_patterns=[
        "**/ops.npy",
        "**/data.bin",
        "**/data_raw.bin",
        "**/stat.npy",
        "**/iscell.npy",
        "**/spks.npy",
        "**/F.npy",
        "**/Fneu.npy",
    ],
    input_extensions=["npy", "bin"],
    output_extensions=["npy", "bin"],
    marker_files=["ops.npy"],
    category="segmentation",
)
register_pipeline(_SUITE2P_INFO)

logger = log.get("arrays.suite2p")


def _extract_plane_number(name: str) -> int | None:
    """The N of ``planeN`` anywhere in ``name`` (``plane01_stitched``, ``plane14``)."""
    match = re.search(r"plane(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


_REG_TIF_RE = re.compile(r"file(\d+)_chan(\d+)\.tiff?$", re.IGNORECASE)


def find_suite2p_reg_tif_files(plane_dir: Path, channel: int = 0) -> list[Path]:
    """Return reg_tif files for ``channel`` in ``plane_dir/reg_tif/``, sorted by
    the numeric frame_start parsed from the filename.
    """
    reg_dir = plane_dir / "reg_tif"
    if not reg_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in reg_dir.iterdir():
        m = _REG_TIF_RE.match(p.name)
        if m and int(m.group(2)) == channel:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda kv: kv[0])
    return [p for _, p in out]


def find_suite2p_plane_dirs(directory: Path) -> list[Path]:
    """The subdirs of ``directory`` holding plane ops (``ops.npy``, or suite2p 1.0's ``settings.npy``) and a movie, sorted by plane number; unnumbered dirs last."""
    plane_dirs = []
    for subdir in directory.iterdir():
        if not subdir.is_dir():
            continue
        # an ops.npy alone is not a plane: roi_workflow's rois_<tag>/ carries one without a movie
        has_movie = (
            (subdir / "data.bin").exists()
            or _raw_bin(subdir).exists()
            or (subdir / "reg_tif").is_dir()
        )
        if has_plane_ops(subdir) and has_movie:
            plane_dirs.append(subdir)
    return sorted(
        plane_dirs,
        key=lambda p: (
            _extract_plane_number(p.name) is None,
            _extract_plane_number(p.name) or 0,
        ),
    )


def _raw_bin(plane_dir: Path) -> Path:
    """The unregistered binary: ``data_raw.bin``, or ``raw_data.bin`` as suite2p 1.0 names it."""
    v1 = plane_dir / "raw_data.bin"
    return (
        v1
        if v1.exists() and not (plane_dir / "data_raw.bin").exists()
        else plane_dir / "data_raw.bin"
    )


def registration_shifts(planes, fs, volume: bool) -> MotionCorrection | None:
    """Suite2p's rigid ``xoff`` / ``yoff`` as a ``MotionCorrection`` in px.

    ``planes`` is ``(plane dir name, ops)`` in z order; a volume's traces are
    named after their plane dir and listed in ``planes``. A plane with no
    shifts, or all-zero ones, gives no trace; no trace at all is None.
    """
    fs = fs or 1.0
    traces, by_plane = {}, {}
    for z, (name, ops) in enumerate(planes):
        xoff, yoff = ops.get("xoff"), ops.get("yoff")
        if xoff is None or yoff is None or not (np.any(xoff) or np.any(yoff)):
            continue
        for axis, shift in (("X", xoff), ("Y", yoff)):
            label = axis
            if volume:
                label = f"{axis} {name}"
                by_plane[label] = z
            traces[label] = (
                np.arange(len(shift)) / fs,
                np.asarray(shift, dtype=np.float64),
            )
    return MotionCorrection("suite2p", "px", traces, by_plane) if traces else None


class _SinglePlaneReader:
    """One plane's binary as (T, Y, X) int16, memory mapped."""

    def __init__(self, ops_path: Path, use_raw: bool = False):
        self.ops_path = ops_path
        # an older ops.npy holds the summary images as json lists; everything downstream indexes arrays
        self.metadata = normalize_ops_arrays(load_plane_ops(ops_path.parent))

        ops_dir = ops_path.parent
        self.raw_file = _raw_bin(ops_dir)
        self.reg_file = ops_dir / "data.bin"
        if use_raw and self.raw_file.exists():
            self.active_file = self.raw_file
        elif self.reg_file.exists():
            self.active_file = self.reg_file
        elif self.raw_file.exists():
            self.active_file = self.raw_file
        else:
            raise FileNotFoundError(
                f"No binary files found in {ops_dir}\n"
                f"Expected either:\n"
                f"  - {self.reg_file} (registered)\n"
                f"  - {self.raw_file} (raw)"
            )

        self.Ly = get_param(self.metadata, "Ly")
        self.Lx = get_param(self.metadata, "Lx")
        self.dtype = np.int16

        actual_bytes = self.active_file.stat().st_size
        bytes_per_frame = self.Ly * self.Lx * np.dtype(self.dtype).itemsize
        actual_nframes = actual_bytes // bytes_per_frame

        ops_nframes = get_param(self.metadata, "nframes")
        if actual_nframes != ops_nframes:
            logger.debug(
                f"ops.npy nframes={ops_nframes} differs from file size ({actual_nframes} frames). "
                f"Using actual frame count."
            )

        self.nframes = actual_nframes
        self.shape = (self.nframes, self.Ly, self.Lx)
        expected_bytes = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
        if actual_bytes > expected_bytes:
            leftover = actual_bytes - expected_bytes
            warnings.warn(
                f"Binary file {self.active_file.name} has {leftover:,} extra bytes. "
                f"Ignoring partial frame data.",
                UserWarning,
                stacklevel=2,
            )

        self._file = np.memmap(
            self.active_file, mode="r", dtype=self.dtype, shape=self.shape
        )

    def switch_channel(self, use_raw: bool = False):
        new_file = self.raw_file if use_raw else self.reg_file
        if not new_file.exists():
            raise FileNotFoundError(new_file)
        self._file = np.memmap(new_file, mode="r", dtype=self.dtype, shape=self.shape)
        self.active_file = new_file

    def __getitem__(self, key):
        return self._file[key]

    def close(self):
        self._file._mmap.close()


class _Suite2pRegTifPlaneReader:
    """One plane read from its ``reg_tif/`` tiffs, with ``_SinglePlaneReader``'s surface."""

    def __init__(self, ops_path: Path, channel: int = 0):
        from mbo_utilities.arrays.tiff import _SingleTiffPlaneReader

        self.ops_path = ops_path
        self.metadata = normalize_ops_arrays(load_plane_ops(ops_path.parent))
        plane_dir = ops_path.parent
        self.reg_tif_dir = plane_dir / "reg_tif"
        self._channel = channel

        files = find_suite2p_reg_tif_files(plane_dir, channel=channel)
        if not files:
            raise FileNotFoundError(
                f"No reg_tif files for chan{channel} in {self.reg_tif_dir}"
            )
        self.filenames = files
        self.active_file = self.reg_tif_dir

        self._inner = _SingleTiffPlaneReader(files)
        self.Ly = self._inner.Ly
        self.Lx = self._inner.Lx
        self.dtype = np.int16
        self.nframes = self._inner.nframes
        self.shape = (self.nframes, self.Ly, self.Lx)

        ops_ly = get_param(self.metadata, "Ly")
        ops_lx = get_param(self.metadata, "Lx")
        if (
            ops_ly is not None
            and ops_lx is not None
            and (ops_ly != self.Ly or ops_lx != self.Lx)
        ):
            logger.warning(
                f"reg_tif shape ({self.Ly}x{self.Lx}) differs from "
                f"ops.npy ({ops_ly}x{ops_lx}) at {self.reg_tif_dir}"
            )

    def __getitem__(self, key):
        return self._inner[key]

    def close(self):
        self._inner.close()


def _load_scanphase_sidecar(plane_dir: Path) -> dict | None:
    """The scan-phase replay record ``scanphase.npy`` wrote, None when the plane has none."""
    sp_file = plane_dir / "scanphase.npy"
    if not sp_file.exists():
        return None
    return load_npy(sp_file).item()


def _resolve_reconstruct_raw_source(
    sp: dict | None, ops: dict, plane_dir: Path | None = None
):
    """Open the raw movie the run registered from, as ``(array, reader_kwargs)``; ``(None, reader_kwargs)`` when no candidate opens.

    Candidates in order: the scan-phase sidecar's ``raw_source`` then its
    ``raw_files``, then ``ops["raw_source"]`` and ``ops["data_path"]``. With
    ``plane_dir``, a dead recorded path is also rebased by basename against
    the plane dir, its ancestors and their ``raw/`` subdirs
    (``metadata.base._rebase_dirs``), since a run dir copied from another
    machine embeds that machine's paths while the raw movie travels with it.
    """
    from mbo_utilities import imread

    reader_kwargs = {}
    cands: list = []
    if sp:
        reader_kwargs = dict(sp.get("reader_kwargs") or {})
        if sp.get("raw_source"):
            cands.append(sp["raw_source"])
        if sp.get("raw_files"):
            cands.append([Path(f) for f in sp["raw_files"]])
    if not reader_kwargs:
        reader_kwargs = dict(ops.get("reader_kwargs") or {})
    if ops.get("raw_source"):
        cands.append(ops["raw_source"])
    dp = ops.get("data_path")
    if dp:
        # vanilla suite2p stores data_path as a list of dirs or files, a streamed run as one path
        for entry in dp if isinstance(dp, (list, tuple)) else [dp]:
            p = Path(entry)
            if p.suffix and p.parent != p:
                cands.append(p.parent)
            cands.append(p)

    rebased_keys: set[str] = set()
    if plane_dir is not None:
        from mbo_utilities.metadata.base import _rebase_dirs, _rebase_one

        search_dirs = _rebase_dirs(Path(plane_dir))
        for c in list(cands):
            entries = c if isinstance(c, list) else [c]
            fixed = []
            for entry in entries:
                if Path(entry).exists():
                    fixed.append(Path(entry))
                    continue
                found = _rebase_one(entry, search_dirs)
                if found is None:
                    break
                fixed.append(found)
            if len(fixed) != len(entries):
                continue
            rb = fixed if isinstance(c, list) else fixed[0]
            if str(rb) != str(c):
                cands.append(rb)
                rebased_keys.add(str(rb))

    seen = set()
    for c in cands:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        exists = (
            all(Path(x).exists() for x in c)
            if isinstance(c, list)
            else Path(c).exists()
        )
        if not exists:
            continue
        try:
            arr = imread(c, **reader_kwargs)
        except Exception as e:
            logger.debug(f"reconstruct raw source {key} failed: {e}")
            continue
        if key in rebased_keys:
            logger.info(
                f"reconstruct raw source: recorded path was dead, rebased to {key}"
            )
        return arr, reader_kwargs
    return None, reader_kwargs


def _apply_registration(frames, yoff, xoff, yoff1, xoff1, blocks, device):
    """Apply suite2p's rigid (and nonrigid) shifts to int16 frames.

    No bidiphase: mbo's scan-phase correction is the bidirectional step.
    Without suite2p the frames get a rigid numpy roll (suite2p's ``-dy, -dx``
    convention) and the nonrigid shifts are skipped with a warning.
    """
    frames = np.ascontiguousarray(frames)
    try:
        import torch
        from suite2p.registration import register
    except ImportError as e:
        if yoff1 is not None:
            logger.warning(
                f"suite2p unavailable ({e}); applying rigid shifts only "
                "(nonrigid skipped, frames approximate)."
            )
        out = np.empty_like(frames)
        for k in range(len(frames)):
            out[k] = np.roll(frames[k], (-int(yoff[k]), -int(xoff[k])), axis=(0, 1))
    else:
        fr = torch.from_numpy(frames)
        if device is not None:
            fr = fr.to(device)
        dev = device if device is not None else torch.device("cpu")
        out = np.asarray(
            register.shift_frames(
                fr,
                yoff.astype(int),
                xoff.astype(int),
                yoff1,
                xoff1,
                blocks,
                device=dev,
            )
        )
    if out.ndim == 2:
        out = out[np.newaxis]
    return out.astype(np.int16, copy=False)


def _resolve_replay_device(recorded):
    """The run's recorded torch device, else cuda when available, else cpu; None without torch."""
    try:
        import torch
    except ImportError:
        return None
    if recorded:
        dev = torch.device(recorded)
        if dev.type != "cuda" or torch.cuda.is_available():
            return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Suite2pReconstructReader:
    """A registered plane rebuilt on read from the raw movie and ``reg_outputs.npy`` when no binary is on disk.

    Each frame is the raw frame read with the scan-phase correction recorded
    at write time, then shifted by the saved registration offsets; the
    surface is ``_SinglePlaneReader``'s.
    """

    def __init__(self, ops_path: Path):
        self.ops_path = Path(ops_path)
        plane_dir = self.ops_path.parent
        self.metadata = normalize_ops_arrays(load_plane_ops(plane_dir))

        reg_file = plane_dir / "reg_outputs.npy"
        reg = load_npy(reg_file).item()
        self._reg = {
            "yoff": np.asarray(reg["yoff"]),
            "xoff": np.asarray(reg["xoff"]),
            "yoff1": np.asarray(reg["yoff1"]) if reg.get("yoff1") is not None else None,
            "xoff1": np.asarray(reg["xoff1"]) if reg.get("xoff1") is not None else None,
        }
        self._nonrigid = bool(self.metadata.get("nonrigid", True))

        sp = _load_scanphase_sidecar(plane_dir)
        self._raw, _ = _resolve_reconstruct_raw_source(
            sp, self.metadata, plane_dir=plane_dir
        )
        if self._raw is None:
            from mbo_utilities.metadata.base import _rebase_dirs

            searched = ", ".join(str(d) for d in _rebase_dirs(plane_dir))
            raise FileNotFoundError(
                f"No data.bin, data_raw.bin, or reg_tif/ in {plane_dir}, and the "
                "raw source to rebuild it from could not be located "
                f"(raw_source={self.metadata.get('raw_source')!r}; also searched "
                f"rebase dirs: {searched}). Re-point the raw data or keep the binary."
            )

        self.Ly = int(get_param(self.metadata, "Ly"))
        self.Lx = int(get_param(self.metadata, "Lx"))
        self.dtype = np.int16
        self.nframes = int(len(self._reg["yoff"]))
        self.shape = (self.nframes, self.Ly, self.Lx)
        self.active_file = self.ops_path
        self.filenames = [self.ops_path]

        if sp is not None:
            self._offsets = np.asarray(sp.get("offsets"), dtype=np.float32)
            fm = sp.get("frame_map")
            self._frame_map = np.asarray(fm, dtype=np.int64) if fm is not None else None
            self._use_fft = bool(sp.get("use_fft", True))
            self._plane_index = int(
                sp.get("plane_index", (int(self.metadata.get("plane", 1)) - 1))
            )
            self._channel_index = int(sp.get("channel_index", 0))
        else:
            # no sidecar (a streamed run, or an already-corrected zarr/h5): the raw as imread
            # returns it plus the registration shifts; the run's per-frame scan-phase is lost
            self._offsets = None
            self._frame_map = None
            self._plane_index = (
                int(self.metadata.get("plane", 1)) - 1
                if self._raw._shape5d()[2] > 1
                else 0
            )
            self._channel_index = 0

        if self._offsets is not None and hasattr(self._raw, "fix_phase"):
            self._raw.fix_phase = True
            if hasattr(self._raw, "use_fft"):
                self._raw.use_fft = self._use_fft

        self._device = _resolve_replay_device(self.metadata.get("torch_device"))
        self._blocks = None
        if self._nonrigid and self._reg["yoff1"] is not None:
            try:
                from suite2p.registration import nonrigid as s2p_nonrigid
            except ImportError as e:
                logger.warning(f"nonrigid block layout unavailable ({e})")
            else:
                block_size = self.metadata.get("block_size") or (128, 128)
                self._blocks = s2p_nonrigid.make_blocks(
                    self.Ly, self.Lx, (int(block_size[0]), int(block_size[1]))
                )

    def _source_frame(self, i: int) -> int:
        return int(self._frame_map[i]) if self._frame_map is not None else int(i)

    def _read_registered(self, idx):
        """Registered int16 frames for registered-frame indices `idx`."""
        idx = np.asarray(idx, dtype=np.int64)
        raw = np.empty((len(idx), self.Ly, self.Lx), dtype=np.int16)
        c, z = self._channel_index, self._plane_index
        # consecutive frames with contiguous sources and one pinned offset are one slice read
        k = 0
        n = len(idx)
        while k < n:
            j = k + 1
            off = float(self._offsets[idx[k]]) if self._offsets is not None else 0.0
            s0 = self._source_frame(int(idx[k]))
            while j < n:
                off_j = (
                    float(self._offsets[idx[j]]) if self._offsets is not None else 0.0
                )
                if off_j != off or self._source_frame(int(idx[j])) != s0 + (j - k):
                    break
                j += 1
            if self._offsets is not None and hasattr(self._raw, "offset"):
                self._raw.offset = off
            block = np.asarray(self._raw[s0 : s0 + (j - k), c, z, :, :])
            raw[k:j] = block.reshape(j - k, self.Ly, self.Lx)
            k = j

        yoff1 = self._reg["yoff1"][idx] if self._reg["yoff1"] is not None else None
        xoff1 = self._reg["xoff1"][idx] if self._reg["xoff1"] is not None else None
        return _apply_registration(
            raw,
            self._reg["yoff"][idx],
            self._reg["xoff"][idx],
            yoff1,
            xoff1,
            self._blocks,
            self._device,
        )

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        t_key, y_key, x_key = key[0], key[1], key[2]

        t_is_int = isinstance(t_key, (int, np.integer))
        if t_is_int:
            t = int(t_key)
            if t < 0:
                t += self.nframes
            idx = [t]
        elif isinstance(t_key, slice):
            idx = list(range(*t_key.indices(self.nframes)))
        else:
            idx = [int(i) for i in t_key]

        out = self._read_registered(idx)[:, y_key, x_key]  # (N, Ly, Lx)
        return out[0] if t_is_int else out

    def close(self):
        self._raw.close()


class Suite2pArray(ReductionMixin, Shape5DMixin):
    """A suite2p run as (T, 1, Z, Y, X) int16: one plane dir, or a folder of them stacked on Z.

    ``filename`` is an ``ops.npy``, a ``.bin`` (that plane only) or a
    directory. ``use_raw`` prefers ``data_raw.bin``; ``use_reg_tif`` reads
    ``reg_tif/`` (chan0) even when a binary exists, otherwise ``reg_tif/``
    and then a rebuild from the raw movie are the fallbacks for a plane
    with no binary. ``metadata`` is the first plane's ops; T is the
    binary's size, never ``ops["nframes"]``.
    """

    _target_dtype = None

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        p = Path(file)
        if p.name == "ops.npy":
            return True
        if p.is_dir():
            if has_plane_ops(p):
                return True
            return any(has_plane_ops(d) for d in p.iterdir() if d.is_dir())
        return False

    def __init__(
        self,
        filename: str | Path,
        use_raw: bool = False,
        use_reg_tif: bool = False,
        dims: str | Sequence[str] | None = None,
    ):
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)

        self._use_raw = use_raw
        self._use_reg_tif = use_reg_tif
        self._planes: list[_SinglePlaneReader | _Suite2pRegTifPlaneReader] = []
        self._is_volumetric = False
        self._results = None
        self._results_read = False

        if path.is_dir():
            plane_dirs = find_suite2p_plane_dirs(path)
            if plane_dirs:
                self._init_volume(plane_dirs, use_raw)
            elif has_plane_ops(path):
                self._init_single_plane(path / "ops.npy", use_raw)
            else:
                raise ValueError(
                    f"Directory {path} has no plane subdirectories or ops.npy"
                )
        elif path.suffix == ".npy" and path.stem == "ops":
            plane_dirs = find_suite2p_plane_dirs(path.parent.parent)
            if len(plane_dirs) > 1:
                self._init_volume(plane_dirs, use_raw)
            else:
                self._init_single_plane(path, use_raw)
        elif path.suffix == ".bin":
            if not has_plane_ops(path.parent):
                raise FileNotFoundError(f"Missing ops.npy near {path}")
            self._init_single_plane(path.with_name("ops.npy"), use_raw)
        else:
            raise ValueError(f"Unsupported input: {path}")

    @property
    def source_path(self) -> Path:
        """The run root for a volume, the plane dir for one plane; a directory, since ``imread`` opens a bare ``.bin`` as ``BinArray``."""
        ops_parent = self._planes[0].ops_path.parent
        return ops_parent.parent if self._is_volumetric else ops_parent

    def _make_plane_reader(self, ops_path: Path):
        """The binary, ``reg_tif/`` or rebuilt reader for one plane, per what is on disk and ``use_reg_tif``."""
        plane_dir = ops_path.parent
        has_bin = (plane_dir / "data.bin").exists() or _raw_bin(plane_dir).exists()
        has_reg_tif = bool(find_suite2p_reg_tif_files(plane_dir, channel=0))

        if self._use_reg_tif:
            if not has_reg_tif:
                raise FileNotFoundError(
                    f"use_reg_tif=True but no reg_tif files in {plane_dir}/reg_tif/"
                )
            return _Suite2pRegTifPlaneReader(ops_path, channel=0)
        if has_bin:
            return _SinglePlaneReader(ops_path, use_raw=self._use_raw)
        if has_reg_tif:
            return _Suite2pRegTifPlaneReader(ops_path, channel=0)
        if (plane_dir / "reg_outputs.npy").exists():
            return _Suite2pReconstructReader(ops_path)
        raise FileNotFoundError(
            f"No data.bin, data_raw.bin, or reg_tif/ in {plane_dir}"
        )

    def _init_single_plane(self, ops_path: Path, use_raw: bool):
        self._is_volumetric = False
        reader = self._make_plane_reader(ops_path)
        self._planes = [reader]

        self._nframes = reader.nframes
        self._nz = 1
        self._ly = reader.Ly
        self._lx = reader.Lx
        self._dtype = reader.dtype
        self._metadata = dict(reader.metadata)
        self.num_rois = self._metadata.get("num_rois", 1)
        self.filenames = [reader.active_file]

    def _init_volume(self, plane_dirs: list[Path], use_raw: bool):
        self._is_volumetric = True
        self._planes = [
            self._make_plane_reader(pdir / "ops.npy") for pdir in plane_dirs
        ]

        shapes = [(p.Ly, p.Lx) for p in self._planes]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent spatial shapes across planes: {shapes}")

        nframes = [p.nframes for p in self._planes]
        if len(set(nframes)) != 1:
            logger.warning(
                f"Inconsistent frame counts across planes: {nframes}. "
                f"Using minimum: {min(nframes)}"
            )

        self._nframes = min(nframes)
        self._nz = len(self._planes)
        self._ly, self._lx = shapes[0]
        self._dtype = self._planes[0].dtype

        self._metadata = dict(self._planes[0].metadata)
        self._metadata["nplanes"] = self._nz
        self._metadata["num_planes"] = self._nz
        self._metadata["plane_dirs"] = [str(p.ops_path.parent) for p in self._planes]
        self.num_rois = get_param(self._metadata, "num_rois", default=1)
        self.filenames = [p.active_file for p in self._planes]

        # ops.npy's dz is authoritative; a stride-named dir without provenance only warns
        if self._nz > 1 and "_metadata_provenance" not in self._metadata:
            plane_nums = [_extract_plane_number(pdir.name) for pdir in plane_dirs]
            plane_nums = [n for n in plane_nums if n is not None]
            if len(plane_nums) == self._nz:
                steps = [
                    plane_nums[i + 1] - plane_nums[i]
                    for i in range(len(plane_nums) - 1)
                ]
                if steps and len(set(steps)) == 1 and steps[0] > 1:
                    logger.info(
                        f"Plane dir names suggest stride={steps[0]} but ops.npy "
                        "carries no provenance marker. Treating saved dz as the "
                        "base value; if this volume was written with stride "
                        "already baked into dz, override `dz` explicitly in ops."
                    )

        logger.info(
            f"Loaded Suite2p volume: {self._nframes} frames, {self._nz} planes, "
            f"{self._ly}x{self._lx} px"
        )

    @property
    def is_volumetric(self) -> bool:
        return self._is_volumetric

    @property
    def num_planes(self) -> int:
        return len(self._planes)

    @property
    def results(self):
        """The detection this run wrote, as ``results.Results``: one unit per
        plane in the array's z order, each stamped with its ``plane_dir`` and
        ``z``; None when no plane carries ``stat.npy`` and ``F.npy`` (a
        registration-only run). Read once, on first use.
        """
        if not self._results_read:
            from mbo_utilities.results import Results

            self._results_read = True
            try:
                self._results = Results.from_suite2p(
                    self.source_path,
                    plane_dirs=[p.ops_path.parent for p in self._planes],
                )
            except FileNotFoundError:
                self._results = None
        return self._results

    @results.setter
    def results(self, value) -> None:
        self._results, self._results_read = value, True

    @property
    def motion_correction(self) -> MotionCorrection | None:
        """Suite2p's rigid registration offsets (each plane's ``xoff`` and
        ``yoff``, in px) on the recording's time axis; None when no plane was
        registered or none moved. A volume is registered plane by plane, so
        its traces are named after their plane dir and listed in ``planes``.
        The nonrigid block offsets (``xoff1`` / ``yoff1``) are left out.
        """
        return registration_shifts(
            [(p.ops_path.parent.name, p.metadata) for p in self._planes],
            self.fs,
            self._is_volumetric,
        )

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        return (self._nframes, 1, self._nz, self._ly, self._lx)

    @property
    def dtype(self):
        from mbo_utilities.arrays._base import get_dtype

        return self._target_dtype or get_dtype(self._dtype)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def astype(self, dtype, copy=True):
        """Set the dtype reads are converted to."""
        self._target_dtype = np.dtype(dtype)
        return self

    def __len__(self) -> int:
        return self._nframes

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        key = key + (slice(None),) * (5 - len(key))

        t_key, c_key, z_key, y_key, x_key = key

        if isinstance(t_key, slice):
            start, stop, step = t_key.indices(self._nframes)
            t_key = slice(start, stop, step)
        elif isinstance(t_key, (int, np.integer)):
            t_key = int(t_key)
            if t_key < 0:
                t_key = self._nframes + t_key
            if t_key >= self._nframes:
                raise IndexError(
                    f"Time index {t_key} out of bounds for {self._nframes} frames"
                )

        if isinstance(z_key, (int, np.integer)):
            z_key = int(z_key)
            if z_key < 0:
                z_key = self._nz + z_key
            if z_key < 0 or z_key >= self._nz:
                raise IndexError(f"Z index {z_key} out of bounds for {self._nz} planes")
            out = self._planes[z_key][t_key, y_key, x_key]
        else:
            if isinstance(z_key, slice):
                z_indices = range(self._nz)[z_key]
            elif isinstance(z_key, (list, np.ndarray)):
                z_indices = z_key
            else:
                z_indices = range(self._nz)

            arrs = [self._planes[i][t_key, y_key, x_key] for i in z_indices]
            if arrs and arrs[0].ndim == 2:
                out = np.stack(arrs, axis=0)  # (Z, Y, X), t was an int
            else:
                out = np.stack(arrs, axis=1)  # (T, Z, Y, X)

        if out.ndim == 2:
            out = out[np.newaxis, np.newaxis, np.newaxis]  # (1, 1, 1, Y, X)
        elif out.ndim == 3 and isinstance(z_key, (int, np.integer)):
            out = out[:, np.newaxis, np.newaxis]  # (T, 1, 1, Y, X)
        elif out.ndim == 3:
            out = out[np.newaxis, np.newaxis]  # (1, 1, Z, Y, X)
        elif out.ndim == 4:
            out = out[:, np.newaxis]  # (T, 1, Z, Y, X)

        squeeze_axes = [i for i in range(3) if isinstance(key[i], (int, np.integer))]
        for ax in sorted(squeeze_axes, reverse=True):
            out = np.squeeze(out, axis=ax)

        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """The first frame of every plane, never the whole movie."""
        if self._is_volumetric:
            arrs = [p[0] for p in self._planes]
            data = np.stack(arrs, axis=0)
        else:
            data = self._planes[0][0]
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def switch_channel(self, use_raw: bool = False):
        """Point every plane at its raw or its registered binary."""
        if any(not isinstance(p, _SinglePlaneReader) for p in self._planes):
            raise NotImplementedError(
                "switch_channel is only supported for binary-backed planes "
                "(data.bin / data_raw.bin); reg_tif/ has no raw/registered split."
            )
        for plane in self._planes:
            plane.switch_channel(use_raw=use_raw)
        self.filenames = [p.active_file for p in self._planes]

    def close(self):
        for plane in self._planes:
            plane.close()

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite=False,
        target_chunk_mb=50,
        ext=".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
        **kwargs,
    ):
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            **kwargs,
        )

    def imshow(self, **kwargs):
        """An ``MboNDViewer`` of the array; a binary-backed single plane shows raw and registered side by side."""
        arrays = []
        names = []
        plane = self._planes[0]
        if self._is_volumetric:
            arrays.append(self)
            names.append("volume")
        elif isinstance(plane, _SinglePlaneReader):
            for use_raw, name in ((True, "raw"), (False, "registered")):
                if (plane.raw_file if use_raw else plane.reg_file).exists():
                    arrays.append(_plane_array(plane.ops_path, use_raw))
                    names.append(name)
        if not arrays:
            arrays.append(self)
            names.append(plane.active_file.name)

        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        histogram_widget = kwargs.get("histogram_widget", True)
        window_funcs = kwargs.get("window_funcs")

        from mbo_utilities.gui._ndviewer import MboNDViewer

        return MboNDViewer(
            data=arrays,
            names=names,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            figure_shape=(1, len(arrays)),
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs=window_funcs,
        )


def _plane_array(ops_path: Path, use_raw: bool) -> Suite2pArray:
    """A one-plane ``Suite2pArray`` over the raw or the registered binary, with no metadata or results."""
    reader = _SinglePlaneReader(ops_path, use_raw=use_raw)
    arr = Suite2pArray.__new__(Suite2pArray)
    arr._planes = [reader]
    arr._is_volumetric = False
    arr._nframes = reader.nframes
    arr._nz = 1
    arr._ly = reader.Ly
    arr._lx = reader.Lx
    arr._dtype = reader.dtype
    arr._metadata = {}
    arr._results, arr._results_read = None, True
    return arr


def _add_suite2p_labels(
    root_group,
    suite2p_dirs: list[Path],
    T: int,
    Z: int,
    Y: int,
    X: int,
    dtype,
    compression_level: int,
):
    """Write the accepted suite2p ROIs of each plane dir as the OME-NGFF 0.5
    label image ``labels/suite2p`` of ``root_group`` (``annotation.ngff
    .write_label_image``), ROI values counting up across planes.
    """
    from mbo_utilities.annotation.ngff import write_label_image

    logger.info("Creating labels array from Suite2p masks...")
    volume = np.zeros((Z, Y, X), np.uint32)
    roi_id = 1
    for zi, s2p_dir in enumerate(suite2p_dirs):
        stat_path = s2p_dir / "stat.npy"
        iscell_path = s2p_dir / "iscell.npy"
        if not stat_path.exists():
            logger.warning(f"stat.npy not found in {s2p_dir}, skipping z={zi}")
            continue
        stat = load_npy(stat_path)
        if iscell_path.exists():
            iscell = load_npy(iscell_path)[:, 0].astype(bool)
        else:
            iscell = np.ones(len(stat), dtype=bool)
        for roi_stat, is_cell in zip(stat, iscell, strict=False):
            ypix = np.asarray(roi_stat.get("ypix", []))
            xpix = np.asarray(roi_stat.get("xpix", []))
            if not is_cell or ypix.size == 0 or xpix.size == 0:
                continue
            volume[zi, ypix.clip(0, Y - 1), xpix.clip(0, X - 1)] = roi_id
            roi_id += 1
    labels_group = root_group.create_group("labels", overwrite=True)
    write_label_image(
        labels_group,
        "suite2p",
        volume,
        [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        source_image="../../0",
    )
    logger.info(f"Added {roi_id - 1} total ROIs across {Z} z-planes")


def load_ops(ops_input: str | Path | list[str | Path]):
    """The ops dict of a suite2p ``.npy`` file; a dict passes through, anything else gives ``{}``."""
    if isinstance(ops_input, (str, Path)):
        return load_npy(ops_input).item()
    if isinstance(ops_input, dict):
        return ops_input
    logger.warning("No valid ops file provided, returning empty dict.")
    return {}


# above IsoviewArray (90): a suite2p folder inside a .corrected/.fused tree is suite2p's
register_array_class(Suite2pArray, priority=100)
