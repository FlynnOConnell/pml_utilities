import pathlib
import pickle
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import log

try:
    import zarr  # noqa: F401  (availability probe for HAS_ZARR)

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

logger = log.get("file_io")


def expand_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    r"""
    Expand a path, list of paths, or wildcard pattern into a sorted list of actual files.

    Parameters
    ----------
    paths : str, Path, or list of (str or Path)
        Can be a single path, a wildcard pattern like "\\*.tif", a folder, or a list of those.

    Returns
    -------
    list of Path
        Sorted list of full paths to matching files.

    Examples
    --------
    >>> expand_paths("data/\\*.tif")
    [Path("data/img_000.tif"), Path("data/img_001.tif"), ...]

    >>> expand_paths(Path("data"))
    [Path("data/img_000.tif"), Path("data/img_001.tif"), ...]

    >>> expand_paths(["data/\\*.tif", Path("more_data")])
    [Path("data/img_000.tif"), Path("more_data/img_050.tif"), ...]
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    elif not isinstance(paths, (list, tuple)):
        raise TypeError(f"Expected str, Path, or sequence of them, got {type(paths)}")

    result = []
    for p in paths:
        p = Path(p)
        if "*" in str(p):
            result.extend(p.parent.glob(p.name))
        elif p.is_dir():
            result.extend(p.glob("*"))
        elif p.is_file():
            result.append(p)

    return sorted(p.resolve() for p in result if p.is_file())


def sort_by_si_filename(filename):
    """Sort ScanImage files by the last number in the filename (e.g., _00001, _00002, etc.)."""
    numbers = re.findall(r"\d+", str(filename))
    return int(numbers[-1]) if numbers else 0


def _is_leaf_zarr(path: Path) -> bool:
    """True for a ``.zarr`` directory with no ``.zarr`` directory directly inside it."""
    if not path.is_dir() or path.suffix != ".zarr":
        return False
    try:
        for child in path.iterdir():
            if child.is_dir() and child.suffix == ".zarr":
                return False
    except OSError:
        pass
    return True


def _walk_with_zarr_filter(path: Path, max_depth: int, exclude_dirs, depth: int = 0):
    """Yields the files under ``path`` down to ``max_depth`` levels, each leaf ``.zarr`` store as one entry.

    Names in ``exclude_dirs`` and unreadable directories are skipped.
    """
    if depth > max_depth:
        return
    try:
        entries = list(path.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name in exclude_dirs:
            continue
        if entry.is_file() or _is_leaf_zarr(entry):
            yield entry
        elif entry.is_dir():
            yield from _walk_with_zarr_filter(entry, max_depth, exclude_dirs, depth + 1)


def get_files(
    base_dir, str_contains="", max_depth=1, sort_ascending=True, exclude_dirs=None
) -> list[Path]:
    """
    Files under ``base_dir`` whose names contain ``str_contains``, down to ``max_depth`` subdirectory levels.

    A leaf ``.zarr`` store (one with no ``.zarr`` inside it) is returned as one entry and never
    walked into, so its chunk directories are not traversed.

    Parameters
    ----------
    base_dir : str or Path
        Directory to search; ``~`` is expanded and the path made absolute.
    str_contains : str, optional
        Substring a name must contain; empty matches every file.
    max_depth : int, optional
        Subdirectory levels to search; 0 is treated as 1.
    sort_ascending : bool, optional
        Sort by the last number in each path (:func:`sort_by_si_filename`), so "file2" comes
        before "file10".
    exclude_dirs : iterable of str, optional
        Directory names to skip; default ``.venv``, ``.git`` and ``__pycache__``.

    Returns
    -------
    list of Path
        Absolute paths of the matching files and leaf ``.zarr`` stores.

    Raises
    ------
    FileNotFoundError
        If the base directory does not exist.
    NotADirectoryError
        If the specified base_dir is not a directory.

    Examples
    --------
    >>> import mbo_utilities as mbo
    >>> # Get all files that contain "ops.npy" in their names by searching up to 3 levels deep:
    >>> ops_files = mbo.get_files("path/to/files", "ops.npy", max_depth=3)
    >>> # Get only files containing "tif" in the current directory (max_depth=1):
    >>> tif_files = mbo.get_files("path/to/files", "tif")
    """
    base_path = Path(base_dir).expanduser()
    # resolve() can break a UNC path (\\server\share) on Windows
    if not str(base_path).startswith("\\\\"):
        base_path = base_path.resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Directory '{base_path}' does not exist.")
    if not base_path.is_dir():
        raise NotADirectoryError(f"'{base_path}' is not a directory.")
    if max_depth == 0:
        max_depth = 1

    if exclude_dirs is None:
        exclude_dirs = [".venv", ".git", "__pycache__"]

    files = [
        entry
        for entry in _walk_with_zarr_filter(base_path, max_depth, exclude_dirs)
        if not str_contains or str_contains in entry.name
    ]

    if sort_ascending:
        files.sort(key=sort_by_si_filename)
    return files


class _CrossPlatformUnpickler(pickle.Unpickler):
    """Unpickles every pathlib path class as this platform's ``Path``."""

    def find_class(self, module, name):
        if module == "pathlib" and name in (
            "PosixPath",
            "WindowsPath",
            "PurePosixPath",
            "PureWindowsPath",
        ):
            return pathlib.Path
        return super().find_class(module, name)


def load_npy(path):
    """Load .npy file with cross-platform Path object handling.

    Handles pickled PosixPath/WindowsPath objects that fail to load
    across operating systems.
    """
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            _shape, _fortran_order, dtype = np.lib.format._read_array_header(f, version)

            if dtype.hasobject:
                f.seek(0)
                np.lib.format.read_magic(f)
                np.lib.format._read_array_header(f, version)
                return _CrossPlatformUnpickler(f).load()
            f.seek(0)
            return np.load(f, allow_pickle=True)
    except Exception:
        original_posix, original_windows = pathlib.PosixPath, pathlib.WindowsPath
        try:
            if sys.platform == "win32":
                pathlib.PosixPath = pathlib.WindowsPath
            else:
                pathlib.WindowsPath = pathlib.PosixPath
            return np.load(path, allow_pickle=True)
        finally:
            pathlib.PosixPath, pathlib.WindowsPath = original_posix, original_windows


def has_plane_ops(plane_dir) -> bool:
    """True when a suite2p plane dir carries its ops, as ``ops.npy`` or suite2p 1.0's ``settings.npy``."""
    plane_dir = Path(plane_dir)
    return (plane_dir / "ops.npy").is_file() or (plane_dir / "settings.npy").is_file()


def load_plane_ops(plane_dir) -> dict:
    """One suite2p plane's ops as a flat dict.

    suite2p 1.0 writes no ``ops.npy``: it splits it into ``settings.npy``
    (nested by stage, flattened here one level, top-level keys winning),
    ``reg_outputs.npy`` and ``detect_outputs.npy``, and stores no ``Ly`` /
    ``Lx`` / ``nframes``, which come from the reference image and the shifts.
    """
    plane_dir = Path(plane_dir)
    if (plane_dir / "ops.npy").is_file():
        return load_npy(plane_dir / "ops.npy").item()
    if not (plane_dir / "settings.npy").is_file():
        raise FileNotFoundError(f"no ops.npy or settings.npy in {plane_dir}")
    settings = load_npy(plane_dir / "settings.npy").item()
    ops = {}
    for value in settings.values():
        if isinstance(value, dict):
            ops.update(value)
    ops.update({k: v for k, v in settings.items() if not isinstance(v, dict)})
    for name in ("reg_outputs.npy", "detect_outputs.npy"):
        if (plane_dir / name).is_file():
            ops.update(load_npy(plane_dir / name).item())
    if "Ly" not in ops:
        for key in ("refImg", "meanImg"):
            if isinstance(ops.get(key), np.ndarray) and ops[key].ndim == 2:
                ops["Ly"], ops["Lx"] = ops[key].shape
                break
    if "nframes" not in ops and "yoff" in ops:
        ops["nframes"] = len(ops["yoff"])
    return ops


def get_last_savedir_path() -> Path:
    """Return path to settings file tracking last saved folder.

    .. deprecated::
        Use :func:`mbo_utilities.preferences.get_last_save_dir` instead.
    """
    import warnings

    warnings.warn(
        "get_last_savedir_path() is deprecated. Use mbo_utilities.preferences.get_last_save_dir() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from mbo_utilities.preferences import get_mbo_dirs

    return get_mbo_dirs()["user_settings"] / "last_savedir.json"
