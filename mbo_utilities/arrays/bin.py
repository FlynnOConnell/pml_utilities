"""
Binary array reader/writer.

This module provides BinArray for reading and writing Suite2p-format binary files
(.bin) without requiring ops.npy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities._writers import _convert_paths_to_strings
from mbo_utilities.arrays._base import ReductionMixin, _imwrite_base
from mbo_utilities.file_io import load_npy
from mbo_utilities.lazy_array import LazyArray
from mbo_utilities.metadata.base import normalize_ops_arrays
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.bin")

# register binary pipeline info
_BIN_INFO = PipelineInfo(
    name="binary",
    description="Raw binary files (.bin)",
    input_patterns=[
        "**/*.bin",
    ],
    output_patterns=[
        "**/*.bin",
    ],
    input_extensions=["bin"],
    output_extensions=["bin"],
    marker_files=[],
    category="reader",
)
register_pipeline(_BIN_INFO)


class BinArray(ReductionMixin, LazyArray):
    """A suite2p binary file (``data_raw.bin``, ``data.bin``) as
    ``(T, 1, 1, Y, X)``.

    The file is a flat ``(nframes, Ly, Lx)`` int16 memmap; its shape comes
    from ``shape`` or the ``ops.npy`` beside it. Indexing and assignment take
    5D keys.

    Examples
    --------
    >>> arr = BinArray("data_raw.bin", shape=(1000, 512, 512))
    >>> arr.shape
    (1000, 1, 1, 512, 512)
    >>> arr[0, 0, 0].shape
    (512, 512)
    """

    def __init__(
        self,
        filename: str | Path,
        shape: tuple[int, int, int] | None = None,
        dtype=np.int16,
        metadata: dict | None = None,
    ):
        self.filename = Path(filename)
        self.dtype = np.dtype(dtype)
        self._metadata = dict(metadata or {})
        if shape is None and self.filename.exists():
            ops_file = self.filename.parent / "ops.npy"
            if ops_file.exists():
                ops = load_npy(ops_file).item()
                ly, lx = ops.get("Ly"), ops.get("Lx")
                nframes = ops.get("nframes", ops.get("n_frames"))
                if None not in (ly, lx, nframes):
                    shape = (int(nframes), int(ly), int(lx))
                    self._metadata.update(ops)
        if shape is None:
            raise ValueError(
                f"Cannot infer shape for {self.filename}. "
                "Provide shape=(nframes, Ly, Lx) or ensure ops.npy exists."
            )
        mode = "r+" if self.filename.exists() else "w+"
        self._file = np.memmap(
            self.filename, mode=mode, dtype=self.dtype, shape=tuple(shape)
        )
        self.filenames = [self.filename]

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        t, y, x = self._file.shape
        return (t, 1, 1, y, x)

    @property
    def file(self) -> np.memmap:
        """The ``(nframes, Ly, Lx)`` memmap, as suite2p's ``BinaryFile`` has it."""
        return self._file

    def __getitem__(self, key):
        return self._file.reshape(self.shape)[key]

    def __setitem__(self, key, value):
        """Write into the memmap, clipping to int16 when ``value`` is wider."""
        value = np.asarray(value)
        if value.dtype != self.dtype and np.issubdtype(self.dtype, np.integer):
            value = np.clip(value, None, np.iinfo(self.dtype).max - 1)
        self._file.reshape(self.shape)[key] = value.astype(self.dtype)

    def __len__(self):
        return self.nt

    def __array__(self, dtype=None, copy=None):
        # one frame, so a histogram or preview never loads the whole file
        data = self._file[0]
        if dtype is not None:
            data = data.astype(dtype)
        return data

    @property
    def nframes(self):
        return self.nt

    @property
    def Ly(self):
        return self.ny

    @property
    def Lx(self):
        return self.nx

    def flush(self):
        """Flush the memmap to disk."""
        self._file.flush()

    def close(self):
        """Close the memmap file."""
        if hasattr(self._file, "_mmap"):
            self._file._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _imwrite(
        self,
        outpath: Path | str,
        planes=None,
        frames=None,
        channels=None,
        target_chunk_mb: int = 50,
        ext: str = ".bin",
        progress_callback=None,
        debug: bool = False,
        overwrite: bool = False,
        output_name: str | None = None,
        **kwargs,
    ):
        """Write BinArray to disk in various formats."""
        outpath = Path(outpath)
        outpath.mkdir(parents=True, exist_ok=True)

        ext_clean = ext.lower().lstrip(".")

        # Fast memmap copy only when no selection is requested. Any
        # frames/planes/channels filter must go through _imwrite_base
        # so the selection is honored.
        if (
            ext_clean == "bin"
            and frames is None
            and planes is None
            and channels is None
        ):
            md = dict(self.metadata) if self.metadata else {}
            md["Ly"] = self.Ly
            md["Lx"] = self.Lx
            md["num_timepoints"] = self.nframes
            md["nframes"] = self.nframes  # suite2p alias

            if output_name is None:
                output_name = "data_raw.bin"
            outfile = outpath / output_name

            if not outfile.exists() or overwrite:
                logger.info(f"Writing binary to {outfile}")
                new_file = np.memmap(
                    outfile, mode="w+", dtype=self.dtype, shape=self._file.shape
                )
                new_file[:] = self._file[:]
                new_file.flush()
                del new_file
            else:
                logger.info(f"Binary file already exists: {outfile}")

            # Write ops.npy (convert Path objects to strings for cross-platform compatibility)
            ops_file = outpath / "ops.npy"
            np.save(ops_file, normalize_ops_arrays(_convert_paths_to_strings(md)))
            logger.info(f"Wrote ops.npy to {ops_file}")
            return outpath

        # For other formats (or .bin with a selection), use common implementation
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            frames=frames,
            channels=channels,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            output_name=output_name,
            **kwargs,
        )
