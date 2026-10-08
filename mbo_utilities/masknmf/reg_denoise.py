"""Pre-registration denoising: compress, register to the compression, compress again.

One channel of one plane goes through three masknmf stages and lands in one
``results.hdf5``: the PMD compression of the raw movie (``raw/CompressionArray``),
the registration estimated on it and applied to the raw movie (masknmf's
registration groups), and the PMD compression of the registered movie
(``CompressionArray``). The root attrs ``mbo_pipeline = "reg_denoise"`` and
``mbo_provenance`` (source path, reader kwargs, channel, first frame, ``fs``,
settings, timings) are what ``is_reg_denoise`` and ``RegDenoiseRun`` read, so
the raw movie and its RTMC curves can be reopened from the file alone.

Shifts are estimated on the positive-polarity compression. An inverted movie
(``OphysArray(negative_indicator=True, include_mean=True)`` is ``2 * mean - x``)
moves the other way to first order, so shifts estimated on it come out flipped.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays.features import MotionCorrection
from mbo_utilities.masknmf.params import (
    REG_DENOISE_FILE,
    REG_DENOISE_PIPELINE,
    RegDenoiseSettings,
)

logger = log.get("masknmf.reg_denoise")

RAW_COMPRESSION_GROUP = "raw/CompressionArray"


def is_reg_denoise(path: Path | str) -> bool:
    """Whether ``path`` is an hdf5 file a ``run_reg_denoise`` wrote."""
    p = Path(path)
    if not (p.is_file() and p.suffix.lower() in (".h5", ".hdf5")):
        return False
    try:
        with h5py.File(p, "r") as f:
            return f.attrs.get("mbo_pipeline") == REG_DENOISE_PIPELINE
    except OSError:
        return False


def alignment(movie: np.ndarray, bin_frames: int = 20) -> float:
    """Mean correlation of each ``bin_frames``-frame average with the movie's mean image."""
    n = len(movie) // bin_frames * bin_frames
    binned = movie[:n].reshape(-1, bin_frames, *movie.shape[1:]).mean(1)
    mean = binned.mean(0).ravel()
    mean = (mean - mean.mean()) / mean.std()
    frames = binned.reshape(len(binned), -1)
    frames = (frames - frames.mean(1, keepdims=True)) / frames.std(1, keepdims=True)
    return float((frames @ mean / mean.size).mean())


def dense(movie, batch: int = 1000) -> np.ndarray:
    """A masknmf movie (registration or compression array) read into float32 numpy."""
    import torch

    out = np.empty(tuple(movie.shape), dtype=np.float32)
    for start in range(0, out.shape[0], batch):
        stop = min(start + batch, out.shape[0])
        chunk = movie[start:stop]
        out[start:stop] = (
            chunk.cpu().numpy() if torch.is_tensor(chunk) else np.asarray(chunk)
        )
    return out


def _source_movie(arr, settings: RegDenoiseSettings) -> np.ndarray:
    if arr.nz != 1:
        raise ValueError(
            f"pre-registration denoising runs on one plane; {arr.source_path} has {arr.nz} z-planes"
        )
    c = settings.channel - 1
    if not 0 <= c < arr.nc:
        raise ValueError(f"channel {settings.channel} is not in 1..{arr.nc}")
    return np.asarray(arr[settings.first_frame :, c, 0], dtype=np.float32)


def run_reg_denoise(
    source,
    save_path: Path | str | None = None,
    settings: RegDenoiseSettings | None = None,
    unit: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Run the three stages on one channel of ``source`` and write ``results.hdf5``.

    ``source`` is a path (``unit`` selects a ``.mesc`` unit) or a ``LazyArray``.
    ``save_path`` defaults to ``<stem>[_<unit>].reg_denoise/`` beside the source.
    The movie is held in memory as float32. Returns the results file.
    """
    import masknmf
    from mbo_utilities.reader import imread, source_reader_kwargs

    settings = settings or RegDenoiseSettings()
    arr = imread(source, unit=unit) if unit else imread(source)
    src = Path(arr.source_path)
    reader_kwargs = source_reader_kwargs(arr)
    if save_path is None:
        tag = reader_kwargs.get("unit", "").replace("/", "_")
        save_path = (
            src.parent / f"{src.stem}{'_' + tag if tag else ''}.{REG_DENOISE_PIPELINE}"
        )
    out = Path(save_path) / REG_DENOISE_FILE
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    tmp.unlink(missing_ok=True)

    device = settings.runtime.device
    batch = settings.runtime.frame_batch_size
    timings = {}

    t0 = time.time()
    raw = _source_movie(arr, settings)
    timings["read"] = time.time() - t0
    logger.info(
        f"reg_denoise: read {src.name} {reader_kwargs} channel {settings.channel} "
        f"frames {settings.first_frame}: -> {raw.shape} in {timings['read']:.1f}s"
    )

    t0 = time.time()
    reference = masknmf.CompressStrategy(
        device=device,
        frame_batch_size=batch,
        **settings.reference.strategy_kwargs(),
    ).compress(raw)
    reference.to(device)
    timings["reference compression"] = time.time() - t0
    logger.info(
        f"reg_denoise: reference compression in {timings['reference compression']:.1f}s"
    )

    if settings.registration.strategy != "rigid":
        raise ValueError(
            f"pre-registration denoising registers rigidly; got {settings.registration.strategy!r}"
        )
    t0 = time.time()
    corrector = masknmf.RigidMotionCorrector(
        device=device, batch_size=batch, **settings.registration.strategy_kwargs()
    )
    corrector.compute_template(reference)
    registration = corrector.motion_correct(reference_movie=reference, target_movie=raw)
    registration.export(str(tmp))
    registered = dense(registration, batch)
    shifts = registration.shifts.cpu().numpy()
    timings["registration"] = time.time() - t0
    before, after = alignment(raw), alignment(registered)
    logger.info(
        f"reg_denoise: registration in {timings['registration']:.1f}s, "
        f"frame-to-mean correlation {before:.4f} -> {after:.4f}"
    )
    if after <= before:
        logger.warning(
            "reg_denoise: registration lowered frame-to-mean correlation; check the shifts"
        )

    t0 = time.time()
    strategy = (
        masknmf.CompressDenoiseStrategy
        if settings.compression.denoise
        else masknmf.CompressStrategy
    )
    denoised = strategy(
        device=device, frame_batch_size=batch, **settings.compression.strategy_kwargs()
    ).compress(registered)
    timings["compression"] = time.time() - t0
    logger.info(
        f"reg_denoise: compression of the registered movie in {timings['compression']:.1f}s"
    )

    from masknmf.utils._serialization import save_dict

    save_dict(
        reference._to_dict(), str(tmp), group=RAW_COMPRESSION_GROUP, exists_ok=True
    )
    denoised.export(str(tmp))
    provenance = {
        "pipeline": REG_DENOISE_PIPELINE,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": {"path": str(src), "reader_kwargs": reader_kwargs},
        "channel": settings.channel,
        "first_frame": settings.first_frame,
        "n_frames": int(raw.shape[0]),
        "fs": arr.fs,
        "dx": arr.dx,
        "dy": arr.dy,
        "channel_names": arr.metadata.get("channel_names"),
        "alignment": {"raw": before, "registered": after},
        "shift_range_px": {
            "y": [float(shifts[:, 0].min()), float(shifts[:, 0].max())],
            "x": [float(shifts[:, 1].min()), float(shifts[:, 1].max())],
        },
        "settings": settings.to_dict(),
        "timing": timings,
        "masknmf_version": getattr(masknmf, "__version__", None),
    }
    with h5py.File(tmp, "a") as f:
        f.attrs["mbo_pipeline"] = REG_DENOISE_PIPELINE
        f.attrs["mbo_provenance"] = json.dumps(provenance, default=str)
    tmp.replace(out)
    logger.info(f"reg_denoise: wrote {out}")
    return out


def on_frames(motion: MotionCorrection | None, t: np.ndarray) -> dict[str, np.ndarray]:
    """Each trace of ``motion`` at the times ``t``, holding its last sample (RTMC is run-length encoded)."""
    if not motion:
        return {}
    out = {}
    for label, (ts, values) in motion.traces.items():
        idx = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
        out[label] = np.asarray(values, dtype=np.float32)[idx]
    return out


@dataclass
class RegDenoiseRun:
    """One ``results.hdf5`` from ``run_reg_denoise`` with its four movies.

    ``raw`` is the source channel reopened from the provenance (or
    ``raw_path``); ``registered`` is masknmf's registration of it; ``pmd_raw``
    and ``pmd_registered`` are the two compressions in raw units. All four are
    ``(T, Y, X)`` float32 in memory. ``rtmc`` is the source's own motion
    correction (MESc RTMC, µm), ``shifts`` masknmf's (px).
    """

    path: Path
    provenance: dict
    raw: np.ndarray
    registered: np.ndarray
    pmd_raw: np.ndarray
    pmd_registered: np.ndarray
    shifts: MotionCorrection
    rtmc: MotionCorrection | None

    @property
    def fs(self) -> float | None:
        return self.provenance.get("fs")

    @property
    def times(self) -> np.ndarray:
        """Each frame's time in seconds on the recording's clock, else frame numbers."""
        frames = self.provenance["first_frame"] + np.arange(len(self.raw))
        return frames / self.fs if self.fs else frames.astype(np.float64)

    @property
    def movies(self) -> dict[str, np.ndarray]:
        return {
            "raw": self.raw,
            "registered": self.registered,
            "pmd(raw)": self.pmd_raw,
            "pmd(registered)": self.pmd_registered,
        }

    @classmethod
    def open(
        cls,
        path: Path | str,
        raw_path: Path | str | None = None,
        device: str = "cpu",
    ) -> RegDenoiseRun:
        from masknmf.utils._serialization import load_dict

        import masknmf
        from mbo_utilities.reader import imread

        path = Path(path)
        with h5py.File(path, "r") as f:
            if f.attrs.get("mbo_pipeline") != REG_DENOISE_PIPELINE:
                raise ValueError(f"{path} is not a pre-registration denoising result")
            provenance = json.loads(f.attrs["mbo_provenance"])
        source = Path(raw_path or provenance["source"]["path"])
        if not source.exists():
            raise FileNotFoundError(
                f"the raw movie {source} is not on this machine; pass raw_path"
            )
        arr = imread(source, **provenance["source"]["reader_kwargs"])
        first, n = provenance["first_frame"], provenance["n_frames"]
        raw = np.asarray(
            arr[first : first + n, provenance["channel"] - 1, 0], dtype=np.float32
        )
        if raw.shape[0] != n:
            raise ValueError(
                f"{source} has {raw.shape[0]} frames from {first}, the run has {n}"
            )

        registration = masknmf.RigidRegistrationArray.from_hdf5(
            str(path), input_movie=raw, device=device
        )
        pmd_raw = masknmf.CompressionArray.from_tensors(
            **load_dict(str(path), RAW_COMPRESSION_GROUP), device=device
        )
        pmd_registered = masknmf.CompressionArray.from_hdf5(str(path), device=device)
        for pmd in (pmd_raw, pmd_registered):
            pmd.rescale = True
            pmd.include_trend = True

        shifts = registration.shifts.cpu().numpy()
        frames = first + np.arange(n)
        t = (
            frames / provenance["fs"]
            if provenance.get("fs")
            else frames.astype(np.float64)
        )
        return cls(
            path=path,
            provenance=provenance,
            raw=raw,
            registered=dense(registration),
            pmd_raw=dense(pmd_raw),
            pmd_registered=dense(pmd_registered),
            shifts=MotionCorrection(
                "masknmf", "px", {"Y": (t, shifts[:, 0]), "X": (t, shifts[:, 1])}
            ),
            rtmc=arr.motion_correction,
        )
